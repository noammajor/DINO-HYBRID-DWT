"""Timer-XL backbone wrapped for our DINO+MAE training — a thin adapter over the
*vendored, unmodified* Timer-XL Model (models/timer_xl/model.py).

We use ONLY the Timer-XL architecture; training is our own (DINO + real-space
MAE). The adapter reuses the original Timer-XL `embedding` (Linear patch_len ->
d_model) and `blocks` (TimerBlock: TimeAttention = RoPE relative positions +
same/different variate bias) exactly as shipped, and adds on top:

  • a learnable CLS token prepended to the patch tokens (PatchTST-style); its
    output is the DINO representation,
  • a learnable mask token for the iBOT/MAE student pass (embedding space;
    reconstruction is real-space, via PatchReconDecoder -> patch_len),
  • bidirectional attention (TimeAttention.mask_flag=False) — the original
    Timer-XL is a causal forecaster; for representation learning every token
    (incl. CLS) attends over the whole window,
  • channel independence: the channel axis is folded into the batch,
  • downstream task heads (prediction / classification) for the DINO linear-probe
    fine-tuning, so the whole pipeline is Timer-XL (no PatchTST).

head_type:
  'Dino'           forward(z) -> [bs, n_vars, d_model]            (CLS rep)
  'prediction'     forward(z) -> [bs, pred_len, n_vars]
  'classification' forward(z) -> [bs, n_classes]
plus (any head_type):
  forward_recon(z)  -> [bs, num_patch, n_vars, d_model]
  forward_ibot(z,m) -> [bs, num_patch, n_vars, d_model]
"""

import torch
import torch.nn as nn
from types import SimpleNamespace
from torch.nn.init import trunc_normal_

from .timer_xl import Model as TimerXLModel
from .timer_xl.SelfAttention_Family import TimeAttention
from .layers.revin import RevIN


def _proj(in_dim, out_dim, dropout, mlp_head=False, hidden_dim=512):
    if not mlp_head:
        return nn.Linear(in_dim, out_dim)
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim), nn.GELU(),
        nn.Dropout(dropout), nn.Linear(hidden_dim, out_dim),
    )


class PatchReconDecoder(nn.Module):
    """Maps patch token embeddings back to raw patch values (real-space MAE).

    Input:  [N, num_patch, d_model]   (N = bs * n_vars)
    Output: [N, num_patch, patch_len]
    """
    def __init__(self, d_model: int, patch_len: int):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, patch_len),
        )

    def forward(self, x):
        return self.decoder(x)


class PredictionHead(nn.Module):
    """Flatten all patch tokens (incl. CLS) → forecast_len, per variate."""
    def __init__(self, n_vars, d_model, num_patch, forecast_len, head_dropout=0, mlp_head=False):
        super().__init__()
        head_dim = d_model * num_patch
        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Identity() if mlp_head else nn.Dropout(head_dropout)
        self.linear  = _proj(head_dim, forecast_len, head_dropout, mlp_head=mlp_head)

    def forward(self, x):                       # x: [bs, nvars, d_model, num_patch]
        x = self.flatten(x)                     # [bs, nvars, d_model*num_patch]
        x = self.dropout(x)
        x = self.linear(x)                      # [bs, nvars, forecast_len]
        return x.transpose(2, 1)                # [bs, forecast_len, nvars]


class ClassificationHead(nn.Module):
    """Use the CLS token (DINO-trained) → n_classes."""
    def __init__(self, n_vars, d_model, n_classes, head_dropout, mlp_head=False):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Identity() if mlp_head else nn.Dropout(head_dropout)
        self.linear  = _proj(n_vars * d_model, n_classes, head_dropout, mlp_head=mlp_head)

    def forward(self, x):                       # x: [bs, nvars, d_model, num_patch+1]
        x = x[:, :, :, 0]                       # CLS token (position 0)
        x = self.flatten(x)                     # [bs, nvars*d_model]
        x = self.dropout(x)
        return self.linear(x)                   # [bs, n_classes]


class TimerXL(nn.Module):
    def __init__(self, c_in: int, patch_len: int, num_patch: int,
                 n_layers: int = 3, d_model: int = 128, n_heads: int = 8,
                 d_ff: int = 256, dropout: float = 0.1, step_size: int = None,
                 head_type: str = "Dino", act: str = "gelu", target_dim: int = 96,
                 head_dropout: float = 0.0, mlp_head: bool = False, **kwargs):
        super().__init__()
        self.n_vars    = c_in
        self.patch_len = patch_len
        self.num_patch = num_patch
        self.step_size = step_size if step_size is not None else patch_len
        self.head_type = head_type
        self.d_model   = d_model

        # ── vendored Timer-XL Model (reuse its .embedding + .blocks) ────────────
        _cfg = SimpleNamespace(
            input_token_len=patch_len, output_token_len=patch_len,
            d_model=d_model, n_heads=n_heads, e_layers=n_layers, d_ff=d_ff,
            dropout=dropout, activation=act, output_attention=False,
            covariate=False, flash_attention=False, use_norm=False,
        )
        self.timer = TimerXLModel(_cfg)
        for m in self.timer.modules():          # representation learning → bidirectional
            if isinstance(m, TimeAttention):
                m.mask_flag = False

        # ── DINO additions ─────────────────────────────────────────────────────
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, d_model))
        trunc_normal_(self.cls_token, std=0.02)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        trunc_normal_(self.mask_token, std=0.02)

        # ── downstream task heads (Timer-XL, no PatchTST) ───────────────────────
        self.head = None
        if head_type == "prediction":
            self.normalization = RevIN(c_in, affine=True)
            self.head = PredictionHead(c_in, d_model, num_patch + 1, target_dim,
                                       head_dropout=head_dropout, mlp_head=mlp_head)
        elif head_type == "classification":
            self.normalization = RevIN(c_in, affine=True)
            self.head = ClassificationHead(c_in, d_model, target_dim,
                                           head_dropout=head_dropout, mlp_head=mlp_head)

    @property
    def backbone(self):
        """Encoder params for the full-finetune optimizer path (linear-probe uses
        only .head). Property, so it does NOT add a duplicate state_dict prefix."""
        return self.timer

    # ── core encode: patches → token embeddings (CLS at index 0) ───────────────
    def _run(self, patches: torch.Tensor, mask=None):
        """patches: [bs, N, n_vars, patch_len] → [bs*n_vars, N+1, d_model], bs, C, N."""
        bs, N, C, P = patches.shape
        x = patches.permute(0, 2, 1, 3).reshape(bs * C, N, P)   # [bs*C, N, patch_len]
        x = self.timer.embedding(x)                            # [bs*C, N, d_model] (vendored)
        if mask is not None:                                   # iBOT/MAE mask (embedding space)
            m = mask.to(x.device).bool().unsqueeze(1).expand(-1, C, -1).reshape(bs * C, N)
            x = torch.where(m.unsqueeze(-1), self.mask_token.view(1, 1, self.d_model), x)
        cls = self.cls_token.expand(bs * C, 1, self.d_model)   # prepend CLS (PatchTST-style)
        x = torch.cat([cls, x], dim=1)                         # [bs*C, N+1, d_model]
        out, _ = self.timer.blocks(x, n_vars=1, n_tokens=N + 1)  # vendored Timer-XL attention
        return out, bs, C, N

    def _tokens_bcdn(self, z, step):
        """Encode (no mask) → [bs, n_vars, d_model, N+1] for the task heads."""
        patches = z.unfold(dimension=1, size=self.patch_len, step=step)
        out, bs, C, N = self._run(patches, mask=None)
        return out.reshape(bs, C, N + 1, self.d_model).permute(0, 1, 3, 2)

    def forward(self, z, padding_mask=None):
        if self.head_type == "Dino":
            patches = z.unfold(dimension=1, size=self.patch_len, step=self.step_size)
            out, bs, C, N = self._run(patches)
            return out[:, 0, :].reshape(bs, C, self.d_model)        # CLS rep [bs, C, d_model]

        if self.head_type == "prediction":
            z = self.normalization(z, mode='norm')
            x = self._tokens_bcdn(z, step=self.patch_len)           # [bs, C, d_model, N+1]
            y = self.head(x)                                        # [bs, pred_len, C]
            return self.normalization(y, mode='denorm')

        if self.head_type == "classification":
            z = self.normalization(z, mode='norm')
            x = self._tokens_bcdn(z, step=self.patch_len)
            return self.head(x)                                     # [bs, n_classes]

        raise ValueError(f"unknown head_type {self.head_type}")

    # ── SSL token outputs (DINO recon / iBOT-MAE) ──────────────────────────────
    def forward_recon(self, z):
        patches = z.unfold(dimension=1, size=self.patch_len, step=self.patch_len)
        out, bs, C, N = self._run(patches, mask=None)
        return out[:, 1:, :].reshape(bs, C, N, self.d_model).permute(0, 2, 1, 3)

    def forward_ibot(self, z, mask=None):
        patches = z.unfold(dimension=1, size=self.patch_len, step=self.patch_len)
        out, bs, C, N = self._run(patches, mask=mask)
        return out[:, 1:, :].reshape(bs, C, N, self.d_model).permute(0, 2, 1, 3)

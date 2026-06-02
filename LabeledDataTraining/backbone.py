"""
LMC label prediction backbone.

Pipeline
--------
1. Frozen TimeMixer.Model encoder  → M-scale enc_out_list
   enc_out_list[k]: [B*C, T_k, d_model]
2. Adaptive-avg-pool each scale over the time axis  → [B*C, d_model] per scale
   Stack  → [B*C, M, d_model]
3. Mean-pool over channels  → [B, M, d_model]
4. Shared linear K, V projections on the M scale tokens
5. 7 dedicated learned query vectors, one per label head
   Cross-attention (per-head query × shared KV)  → [B, d_model] per head
6. Label-specific MLP heads with dependency-path teacher forcing
   d_min  ──►  d_max   (d_max = d_min + softplus(δ))
   d_min, d_max  ──►  dirichlet   (clamped to (d_min, d_max))
   During training pass teacher_d_min / teacher_d_max (ground truth).
   At inference the predicted values are detached and chained.
"""

import sys
import os

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

# Make TimeMixer importable.
_TIMEMIXER_ROOT   = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'TimeMixer-main'))
_TIMEMIXER_MODELS = os.path.join(_TIMEMIXER_ROOT, 'models')
for _p in [_TIMEMIXER_ROOT, _TIMEMIXER_MODELS]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from TimeMixer import Model as TimeMixerModel  # noqa: E402
from heads import (                             # noqa: E402
    LatentNumHead, PositiveHead, DirichletMaxHead, DirichletHead,
)

# Head-index constants — kept in sync with heads.py Y-layout docstring.
_LATENT_NUM    = 0   # classification over discrete latent_num range
_DIRICHLET     = 1   # bounded regression conditioned on (d_min, d_max)
_WEIBULL_SHAPE = 2   # positive regression
_WEIBULL_SCALE = 3   # positive regression
_DIRICHLET_MIN = 4   # positive regression; anchors the dependency chain
_DIRICHLET_MAX = 5   # positive regression conditioned on d_min
_ESS_LS        = 6   # positive regression
_N_HEADS       = 7


class LMCBackbone(nn.Module):
    """Multi-scale cross-attention backbone for LMC label prediction.

    Args:
        backbone        : pretrained TimeMixer.Model (frozen by default)
        hidden_dim      : MLP hidden size for all prediction sub-heads
        min_latent      : smallest possible latent_num value
        max_latent      : largest possible latent_num value
        freeze_backbone : gradient does not flow into the backbone when True
    """

    def __init__(
        self,
        backbone: TimeMixerModel,
        hidden_dim: int = 64,
        min_latent: int = 2,
        max_latent: int = 10,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.backbone = backbone

        # Stop gradient flow into the pretrained encoder.
        if freeze_backbone:
            for p in backbone.parameters():
                p.requires_grad_(False)

        d = backbone.configs.d_model               # embedding dimension
        M = backbone.configs.down_sampling_layers + 1  # finest + coarser scales

        self.d_model = d
        self.M       = M
        self.scale   = d ** -0.5  # 1/sqrt(d) for scaled dot-product attention

        # ── Shared K, V projections ────────────────────────────────────────────
        # All 7 label heads attend to the same key/value space built from the
        # M-scale token sequence [B, M, d_model].  Only the query differs per head.
        self.shared_k = nn.Linear(d, d, bias=False)
        self.shared_v = nn.Linear(d, d, bias=False)

        # ── Per-head learned queries ───────────────────────────────────────────
        # Each query [1, 1, d_model] specialises the attention pattern for its
        # label, letting every head focus on different temporal scales.
        self.queries = nn.ParameterList(
            [nn.Parameter(torch.empty(1, 1, d)) for _ in range(_N_HEADS)]
        )
        for q in self.queries:
            trunc_normal_(q, std=0.02)

        # ── Prediction sub-heads ──────────────────────────────────────────────
        # Each head receives its own attention-aggregated [B, d_model] vector.
        self.latent_num_head    = LatentNumHead(d, hidden_dim, min_latent, max_latent)
        self.weibull_shape_head = PositiveHead(d, hidden_dim)
        self.weibull_scale_head = PositiveHead(d, hidden_dim)
        self.ess_ls_head        = PositiveHead(d, hidden_dim)
        self.d_min_head         = PositiveHead(d, hidden_dim)
        self.d_max_head         = DirichletMaxHead(d, hidden_dim)  # conditioned on d_min
        self.dirichlet_head     = DirichletHead(d, hidden_dim)     # conditioned on d_min, d_max

    # ── internal helpers ───────────────────────────────────────────────────────

    def _encode_multiscale(self, x: torch.Tensor) -> torch.Tensor:
        """Drive the frozen TimeMixer encoder and collapse into per-sample scale tokens.

        x: [B, T, C]  →  [B, M, d_model]

        Steps:
          • multi-scale avg-pool  →  M × [B, T_k, C]
          • CI=1 reshape  →  M × [B*C, T_k, 1]
          • pre_enc + enc_embedding + PDM blocks  →  M × [B*C, T_k, d_model]
          • mean over T_k  →  M × [B*C, d_model]
          • stack + mean over C  →  [B, M, d_model]
        """
        B, T, C = x.shape
        cfg     = self.backbone.configs
        dsw     = self.backbone.down_sampling_window  # downsampling stride

        # ── multi-scale downsampling ───────────────────────────────────────────
        # Replicates TimeMixer.Model.__multi_scale_process_inputs without the
        # x_mark (timestamp features) which we don't use.
        raw = [x]
        xd  = x.permute(0, 2, 1)   # [B, C, T]
        for _ in range(cfg.down_sampling_layers):
            xd = F.avg_pool1d(xd, dsw)
            raw.append(xd.permute(0, 2, 1))   # [B, T_k, C]

        # ── CI=1: reshape each scale to [B*C, T_k, 1] ─────────────────────────
        x_list = []
        for xs in raw:
            Bs, Tk, N = xs.shape
            xs = xs.permute(0, 2, 1).contiguous().reshape(Bs * N, Tk, 1)
            x_list.append(xs)

        # ── pre_enc + embed ────────────────────────────────────────────────────
        # pre_enc returns (x_list, None) for CI=1 (passthrough).
        x_list_enc, _ = self.backbone.pre_enc(x_list)
        enc_list = [self.backbone.enc_embedding(xs, None) for xs in x_list_enc]

        # ── PDM blocks ────────────────────────────────────────────────────────
        for pdm in self.backbone.pdm_blocks:
            enc_list = pdm(enc_list)               # M × [B*C, T_k, d_model]

        # ── pool over time, mean over channels ────────────────────────────────
        pooled = [enc_k.mean(dim=1) for enc_k in enc_list]    # M × [B*C, d_model]
        z = torch.stack(pooled, dim=1)                         # [B*C, M, d_model]
        z = z.reshape(B, C, self.M, self.d_model).mean(dim=1)  # [B, M, d_model]
        return z

    def _head_attn(
        self, K: torch.Tensor, V: torch.Tensor, head_idx: int
    ) -> torch.Tensor:
        """Scaled dot-product cross-attention for a single label head.

        The head's dedicated query attends over M scale tokens,
        weighting each scale by its relevance to this particular label.

        K, V : [B, M, d_model]  — shared across all heads
        returns: [B, d_model]   — head-specific aggregated representation
        """
        B = K.shape[0]
        q    = self.queries[head_idx].expand(B, 1, -1)                       # [B, 1, d_model]
        attn = torch.softmax(q @ K.transpose(-2, -1) * self.scale, dim=-1)   # [B, 1, M]
        return (attn @ V).squeeze(1)                                          # [B, d_model]

    # ── public forward ─────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        teacher_d_min: Optional[torch.Tensor] = None,
        teacher_d_max: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        x             : [B, T, C]
        teacher_d_min : [B, 1]  ground-truth d_min — inject during training
        teacher_d_max : [B, 1]  ground-truth d_max — inject during training

        Returns dict matching the LMCLabelHead output contract:
            latent_num_logits  [B, num_classes]  → CrossEntropyLoss
            weibull_shape      [B, 1]
            weibull_scale      [B, 1]
            ess_length_scale   [B, 1]
            dirichlet_min      [B, 1]
            dirichlet_max      [B, 1]
            dirichlet          [B, 1]
        """
        # Step 1: multi-scale encoding → [B, M, d_model]
        z = self._encode_multiscale(x)

        # Step 2: shared K, V computed once; all 7 heads reuse them.
        K = self.shared_k(z)   # [B, M, d_model]
        V = self.shared_v(z)   # [B, M, d_model]

        # Step 3: per-head cross-attention → list of 7 × [B, d_model]
        emb = [self._head_attn(K, V, h) for h in range(_N_HEADS)]

        # ── Independent heads ──────────────────────────────────────────────────
        latent_logits = self.latent_num_head(emb[_LATENT_NUM])
        weibull_shape = self.weibull_shape_head(emb[_WEIBULL_SHAPE])
        weibull_scale = self.weibull_scale_head(emb[_WEIBULL_SCALE])
        ess_ls        = self.ess_ls_head(emb[_ESS_LS])

        # ── Dependency path: d_min → d_max → dirichlet ────────────────────────
        # Each stage conditions on the previous output.
        # During training ground-truth values are injected (teacher forcing) so
        # each head trains independently of upstream prediction errors.
        # At inference the predictions are detached before being passed forward,
        # stopping gradients from flowing back through the chain.
        d_min      = self.d_min_head(emb[_DIRICHLET_MIN])
        d_min_cond = teacher_d_min if teacher_d_min is not None else d_min.detach()

        d_max      = self.d_max_head(emb[_DIRICHLET_MAX], d_min_cond)
        d_max_cond = teacher_d_max if teacher_d_max is not None else d_max.detach()

        dirichlet  = self.dirichlet_head(emb[_DIRICHLET], d_min_cond, d_max_cond)

        return {
            "latent_num_logits": latent_logits,
            "weibull_shape":     weibull_shape,
            "weibull_scale":     weibull_scale,
            "ess_length_scale":  ess_ls,
            "dirichlet_min":     d_min,
            "dirichlet_max":     d_max,
            "dirichlet":         dirichlet,
        }

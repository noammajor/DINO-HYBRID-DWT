"""
iTransformer encoder wrapped for DINO pretraining — VARIATE-TOKEN / INVERTED,
CROSS-VARIATE ATTENTION.

Unlike the channel-independent TimesNet/TSMixer backbones, iTransformer inverts the
data layout: the entire length-``seq_len`` series of each variate is embedded into a
SINGLE token (``DataEmbedding_inverted``: Linear(seq_len -> d_model)), and the
transformer encoder runs self-attention ACROSS variate tokens — i.e. it explicitly
models cross-variate dependencies. This is iTransformer's whole point.

Consequences of the inverted design (important):

1. FIXED seq_len.  The embedding is ``Linear(seq_len -> d_model)``, so the input
   length is baked into the weights. Pretraining and forecasting MUST use the same
   ``seq_len``. (This project pretrains in-domain, so that constraint is fine.)

2. There is NO timestep-token axis. Tokens are variates, not time positions. The DINO
   global representation is therefore ``[B, C, d_model]`` (one token per variate),
   which ``TSMultiCropWrapper`` reshapes to ``[B*C, d_model]`` before the DINOHead —
   exactly the convention the other backbones use.

3. Channel count is fixed at CONSTRUCTION (``c_in``), not at run time: the number of
   variate tokens equals ``c_in``. Different downstream channel counts need a backbone
   built for that count. (seq_len is likewise fixed; see point 1.)

Scope: this backbone is CLASSIC DINO ONLY (mlm_phi=0). iBOT/MAE are NOT supported
because there is no timestep-token axis to mask; ``forward_ibot``/``forward_recon``
exist only for interface parity and return the variate-token encoding with a dummy
singleton "time" axis so shapes never crash if they were ever called.
"""

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

# Self-contained: the iTransformer body (inverted embedding + cross-variate encoder)
# is vendored from Time-Series-Library into _tslib_layers, so tsdino_itransformer has
# NO dependency on the Time-Series-Library-main-2 repo.
from ._tslib_layers import (
    DataEmbedding_inverted,
    Encoder,
    EncoderLayer,
    FullAttention,
    AttentionLayer,
)


class PatchReconDecoder(nn.Module):
    """Maps token embeddings back to raw values for MAE/recon (interface parity).

    Input:  [N, num_tokens, d_model]  →  Output: [N, num_tokens, out_dim]

    Kept for import parity with the other tsdino_* packages; unused in DINO-only mode.
    """
    def __init__(self, d_model: int, out_dim: int):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, out_dim),
        )

    def forward(self, x):
        return self.decoder(x)


class iTransformerForDINO(nn.Module):
    """Variate-token (inverted) iTransformer encoder for DINO.

    Pipeline:
        [B,T,C] -- (optional) per-instance norm --------------------> [B,T,C]
        DataEmbedding_inverted(seq_len -> d_model)  (per variate)  -> [B, C, d_model]
        Encoder (cross-variate self-attention) x e_layers + LN     -> [B, C, d_model]

    Each variate becomes one token; ``forward`` returns the per-variate reps
    ``[B, C, d_model]``. ``TSMultiCropWrapper`` reshapes ``[B, C, d_model] ->
    [B*C, d_model]`` before the DINOHead.
    """

    def __init__(
        self,
        c_in: int,
        d_model: int,
        e_layers: int,
        d_ff: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        factor: int = 1,
        activation: str = "gelu",
        embed: str = "timeF",
        freq: str = "h",
        seq_len: int = 336,
        use_norm: int = 1,
    ):
        super().__init__()
        self.c_in = c_in              # number of variate tokens (FIXED at construction)
        self.d_model = d_model
        self.e_layers = e_layers
        self.seq_len = seq_len        # FIXED: inverted embedding bakes seq_len into weights
        self.num_tokens = c_in        # variate tokens (no timestep-token axis)
        self.use_norm = use_norm

        # Inverted embedding: Linear(seq_len -> d_model), applied per variate.
        self.enc_embedding = DataEmbedding_inverted(seq_len, d_model, embed, freq, dropout)
        # Cross-variate transformer encoder (standard, non-causal).
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=False),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                ) for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )

        # iBOT/MAE parity (unused in DINO-only): there is no timestep-token axis, so
        # the mask token is never applied. Kept so state_dicts / interfaces line up.
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        trunc_normal_(self.mask_token, std=0.02)
        # DINO-only backbones don't pool with a CLS query (variate tokens ARE the reps),
        # but keep a cls_query parameter for interface parity with the other backbones.
        self.cls_query = nn.Parameter(torch.zeros(1, 1, d_model))
        trunc_normal_(self.cls_query, std=0.02)

    # ── internal ────────────────────────────────────────────────────────────
    def _encode(self, x: torch.Tensor):
        """x: [B, T, C] → (enc [B, C, d_model], means, stdev).

        Instance-normalizes exactly like iTransformer.Model.forecast (Non-stationary
        Transformer style) and returns the norm stats so the forecast head can
        de-normalize. seq_len (T) is fixed by the inverted embedding.
        """
        if self.use_norm:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x = x / stdev
        else:
            # No normalization; return neutral stats so de-norm is a no-op.
            means = torch.zeros(x.shape[0], 1, x.shape[2], device=x.device, dtype=x.dtype)
            stdev = torch.ones(x.shape[0], 1, x.shape[2], device=x.device, dtype=x.dtype)

        enc = self.enc_embedding(x, None)             # [B, C, d_model]
        enc, _ = self.encoder(enc, attn_mask=None)    # [B, C, d_model]
        return enc, means, stdev

    # ── public ──────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Global DINO rep. x: [B, T, C] → [B, C, d_model] (one token per variate)."""
        enc, _, _ = self._encode(x)
        return enc                                    # [B, C, d_model]

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Per-variate tokens for classification. → [B, C, 1, d_model].

        iTransformer has no timestep-token axis, so a singleton "time" dimension is
        inserted to keep a 4-D [B, C, T, d_model] shape consistent with the other
        backbones (here T = 1).
        """
        enc, _, _ = self._encode(x)                   # [B, C, d_model]
        return enc.unsqueeze(2)                        # [B, C, 1, d_model]

    def forward_ibot(self, z: torch.Tensor, mask=None) -> torch.Tensor:
        """iBOT/recon token encoding — NOT supported (DINO-only backbone).

        iTransformer has no timestep-token axis to mask, so iBOT/MAE are unsupported.
        Provided only for interface parity: returns the variate-token encoding with a
        single dummy "time" token → [B, 1, C, d_model] (never exercised at mlm_phi=0).
        """
        enc, _, _ = self._encode(z)                   # [B, C, d_model]
        return enc.unsqueeze(1)                        # [B, 1, C, d_model]

    def forward_recon(self, z: torch.Tensor) -> torch.Tensor:
        """MAE recon token encoding — NOT supported (see forward_ibot). → [B, 1, C, d_model]."""
        return self.forward_ibot(z, mask=None)


class iTransformerForecastModel(nn.Module):
    """Pretrained iTransformerForDINO backbone + linear forecasting head.

    Mirrors ``TSMixerForecastModel``'s external API so main.py's forecast section and
    the linear-probe path (``model.head.parameters()``, ``model.backbone``) work
    unchanged.

    Forward (x: [B, T, C] → [B, pred_len, C]), following iTransformer.Model.forecast:
      1. backbone._encode  — instance-norm + inverted embedding + cross-variate encoder
                             → enc [B, C, d_model], plus norm stats (means, stdev)
      2. head: Linear(d_model -> pred_len) applied per variate token → [B, C, pred_len]
      3. permute → [B, pred_len, C]
      4. de-normalize (Non-stationary Transformer style) with means/stdev
    """

    def __init__(self, backbone: iTransformerForDINO, pred_len: int, use_revin: bool = True):
        super().__init__()
        self.backbone  = backbone
        self.pred_len  = pred_len
        self.use_revin = use_revin
        # iTransformer's forecast projection: per-variate-token Linear(d_model -> pred_len).
        self.head = nn.Linear(backbone.d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, C] → [B, pred_len, C]."""
        enc, means, stdev = self.backbone._encode(x)      # [B, C, d_model]
        dec = self.head(enc)                              # [B, C, pred_len]
        dec = dec.permute(0, 2, 1)                        # [B, pred_len, C]
        # De-Normalization (Non-stationary Transformer), matching iTransformer exactly.
        # means/stdev are [B, 1, C]; broadcast over the pred_len axis.
        dec = dec * stdev[:, :1, :] + means[:, :1, :]
        return dec

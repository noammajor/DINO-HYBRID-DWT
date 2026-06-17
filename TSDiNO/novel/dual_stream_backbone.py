"""
Disentangled Dual-Stream Backbone
=================================

Pipeline:

    x [B, T, C]
        │
        ▼   LearnedDecomp  (architectural, learned trend/season split)
    ┌───────────────┬────────────────┐
    │ trend [B,T,C] │ season [B,T,C] │
    ▼               ▼
  STREAM A (macro)            STREAM B (micro)
  coarse patches (len Pm)     fine patches (len Ps),  Ps < Pm
  → MLP-Mixer                 → Transformer (self-attention) + CLS
  → z_macro [B, C, d]         → z_micro [B, C, d]

Why these encoder choices:
  * **Trend → MLP-Mixer.** Trend is smooth and global; coarse patches + simple
    token/channel mixing capture low-frequency structure cheaply, without the
    quadratic cost (or the tendency to chase high-frequency detail) of attention.
  * **Season → attention.** Cycles/anomalies are localized and relational; fine
    patches + self-attention let each short segment attend to the others to model
    periodicity and recurring motifs. A CLS token summarises the season view.

Both streams are **channel-independent**: every variable is encoded separately
(the [B, C, ...] tensors are folded to [B*C, ...]), matching PatchTST/TSMixer in
this repo. Each stream returns a per-variable embedding [B, C, d_model]; the
DINO/multi-crop wrapper flattens that to [B*C, d_model] before its projection head.

This module ONLY produces the two embeddings — it does not apply DINO heads or
the loss. That keeps it a clean, testable building block. The phase-2 wiring into
main.py (`backbone_type="dualstream"`) will attach the heads and the
DualStreamDINOLoss.
"""

import torch
import torch.nn as nn

from .decomposition import LearnedDecomp


# ── helpers ─────────────────────────────────────────────────────────────────────

def _patch(x_bct: torch.Tensor, patch_len: int) -> torch.Tensor:
    """[B, C, T] -> [B, C, n_patches, patch_len] via non-overlapping windows."""
    return x_bct.unfold(dimension=2, size=patch_len, step=patch_len)


class _MixerBlock(nn.Module):
    """One MLP-Mixer block over a [N, n_patches, d] tensor.

    Token-mixing MLP (mixes across patches) then channel-mixing MLP (mixes across
    the feature dim), each with a LayerNorm + residual — the standard Mixer recipe.
    """

    def __init__(self, n_patches: int, d_model: int, expansion: int = 2):
        super().__init__()
        self.norm_tokens = nn.LayerNorm(d_model)
        self.token_mlp = nn.Sequential(
            nn.Linear(n_patches, n_patches * expansion), nn.GELU(),
            nn.Linear(n_patches * expansion, n_patches),
        )
        self.norm_channels = nn.LayerNorm(d_model)
        self.channel_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * expansion), nn.GELU(),
            nn.Linear(d_model * expansion, d_model),
        )

    def forward(self, x):                       # x: [N, n_patches, d]
        # token-mixing: operate across the patch axis (transpose to put it last)
        y = self.norm_tokens(x).transpose(1, 2)         # [N, d, n_patches]
        y = self.token_mlp(y).transpose(1, 2)           # [N, n_patches, d]
        x = x + y
        # channel-mixing: operate across the feature axis
        x = x + self.channel_mlp(self.norm_channels(x))
        return x


# ── streams ─────────────────────────────────────────────────────────────────────

class TrendStream(nn.Module):
    """Coarse-patch MLP-Mixer over the trend component -> z_macro [B, C, d]."""

    def __init__(self, c_in, seq_len, patch_len, d_model, n_layers=2):
        super().__init__()
        self.c_in = c_in
        self.patch_len = patch_len
        self.n_patches = seq_len // patch_len
        self.embed = nn.Linear(patch_len, d_model)
        self.blocks = nn.ModuleList(
            _MixerBlock(self.n_patches, d_model) for _ in range(n_layers))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, trend):                   # trend: [B, T, C]
        B, T, C = trend.shape
        p = _patch(trend.transpose(1, 2), self.patch_len)   # [B, C, n, patch_len]
        p = p.reshape(B * C, self.n_patches, self.patch_len)
        h = self.embed(p)                                   # [B*C, n, d]
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h).mean(dim=1)                        # pool patches -> [B*C, d]
        return h.reshape(B, C, -1)                          # [B, C, d]


class SeasonStream(nn.Module):
    """Fine-patch Transformer (CLS) over the season component -> z_micro [B, C, d]."""

    def __init__(self, c_in, seq_len, patch_len, d_model, n_heads=8, n_layers=2):
        super().__init__()
        self.c_in = c_in
        self.patch_len = patch_len
        self.n_patches = seq_len // patch_len
        self.embed = nn.Linear(patch_len, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches + 1, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, season):                  # season: [B, T, C]
        B, T, C = season.shape
        p = _patch(season.transpose(1, 2), self.patch_len)   # [B, C, n, patch_len]
        p = p.reshape(B * C, self.n_patches, self.patch_len)
        h = self.embed(p)                                    # [B*C, n, d]
        cls = self.cls.expand(h.shape[0], -1, -1)
        h = torch.cat([cls, h], dim=1) + self.pos            # prepend CLS + pos
        h = self.encoder(h)
        z = self.norm(h[:, 0])                               # CLS token -> [B*C, d]
        return z.reshape(B, C, -1)                           # [B, C, d]


# ── full backbone ───────────────────────────────────────────────────────────────

class DualStreamBackbone(nn.Module):
    """Decompose -> (trend stream, season stream) -> (z_macro, z_micro).

    forward(x) returns a dict {'macro': [B,C,d], 'micro': [B,C,d]}. The caller
    (multi-crop wrapper) decides how to head/aggregate them depending on the loss
    mode (concat / dual / cross).
    """

    def __init__(self, c_in, seq_len, d_model=128,
                 patch_macro=32, patch_micro=8,
                 trend_layers=2, season_layers=2, season_heads=8,
                 decomp_kernel=25, season_as_residual=False):
        super().__init__()
        self.c_in = c_in
        self.d_model = d_model
        self.decomp = LearnedDecomp(kernel_size=decomp_kernel,
                                    season_as_residual=season_as_residual)
        self.trend = TrendStream(c_in, seq_len, patch_macro, d_model, trend_layers)
        self.season = SeasonStream(c_in, seq_len, patch_micro, d_model,
                                   season_heads, season_layers)

    def forward(self, x):                       # x: [B, T, C]
        trend, season = self.decomp(x)
        return {"macro": self.trend(trend), "micro": self.season(season)}

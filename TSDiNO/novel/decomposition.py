"""
Learned Trend/Season Decomposition  (architectural pre-encoder block)
=====================================================================

WHAT THIS IS
------------
The first block of the Disentangled Dual-Stream backbone. It takes the raw
(already-augmented) series and splits it into two parallel components:

    x [B, T, C]  ->  trend [B, T, C]  +  season [B, T, C]

`trend`  = the slow, low-frequency macro component (drift / regime).
`season` = the fast, high-frequency micro component (cycles / spikes / anomalies).

Each component is then sent to its own specialised encoder (Stream A / Stream B),
so the network never has to cram trend and season into one shared embedding.

WHY IT'S AN ARCHITECTURE BLOCK, NOT AN AUGMENTATION
---------------------------------------------------
The repo already has a *DWT augmentation*: a fixed pywt transform applied to the
input to create teacher/student views. That shapes WHAT invariances DINO learns.

This block is different in kind:
  * its filters are `nn.Parameter`s — **trained end-to-end with the SSL loss**,
  * it lives **inside** the backbone and runs on every forward (GPU conv),
  * its job is to give trend and season their own **learned subspaces**, which
    shapes HOW the encoder represents the signal.

Per the design decision, the split is a **fully-learned convolution** (random
init, no wavelet prior) — the model discovers the trend/season boundary that
best serves the representation, rather than one we impose.

DESIGN NOTES
------------
* **Stride-1, length-preserving.** No downsampling: trend and season keep length
  T so the two streams stay time-aligned. This is required by the cross-stream
  objective, where a *local* season token of one crop is matched against the
  *global* trend of another crop.
* **Depthwise, channel-shared kernel.** One learned filter is applied to every
  variable independently (channel-independent), matching how PatchTST/TSMixer
  treat channels in this repo.
* **Two independent learned filters by default** (`trend_conv`, `season_conv`),
  so each component is free to specialise. Perfect reconstruction
  (x == trend + season) is therefore NOT enforced — these are *features*, not a
  lossless basis. Set `season_as_residual=True` to instead define
  season := x - trend (one learned filter, exact split).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedDecomp(nn.Module):
    """Split [B, T, C] -> (trend, season) with fully-learned depthwise filters.

    Parameters
    ----------
    kernel_size : int
        Temporal receptive field of each learned filter (odd is convenient for
        symmetric padding; even also works). Larger = smoother trend capacity.
    season_as_residual : bool
        If True,  season := x - trend  (single learned filter, exact split).
        If False (default), season is produced by a second, independent learned
        filter (more expressive; reconstruction not guaranteed).
    """

    def __init__(self, kernel_size: int = 25, season_as_residual: bool = False):
        super().__init__()
        self.kernel_size = kernel_size
        self.season_as_residual = season_as_residual

        # Learned depthwise time filters, stored as 1-D kernels [K] shared across
        # channels. Random init (no wavelet/moving-average prior).
        self.trend_kernel = nn.Parameter(torch.randn(kernel_size) * 0.02)
        if not season_as_residual:
            self.season_kernel = nn.Parameter(torch.randn(kernel_size) * 0.02)

    def _apply(self, x_ct: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """Apply a shared 1-D `kernel` to [B, C, T] along time, output length T."""
        B, C, T = x_ct.shape
        K = kernel.numel()
        weight = kernel.view(1, 1, K).expand(C, 1, K)   # same kernel on every channel
        total = K - 1                                    # pad to keep length T
        left, right = total // 2, total - total // 2
        x_pad = F.pad(x_ct, (left, right), mode="reflect")
        return F.conv1d(x_pad, weight, groups=C)

    def forward(self, x: torch.Tensor):
        """x: [B, T, C]  ->  trend [B, T, C], season [B, T, C]."""
        x_ct = x.transpose(1, 2)                          # [B, C, T] for conv1d
        trend = self._apply(x_ct, self.trend_kernel).transpose(1, 2)
        if self.season_as_residual:
            season = x - trend
        else:
            season = self._apply(x_ct, self.season_kernel).transpose(1, 2)
        return trend, season

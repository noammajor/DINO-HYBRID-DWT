"""
TSDiNO/novel — Disentangled Trend-Seasonality Dual-Stream backbone for DINO.

A self-contained, experimental package (not yet wired into the training loop):

    LearnedDecomp          x -> (trend, season)   [learned conv pre-encoder]
    DualStreamBackbone     -> {"macro": z, "micro": z}   [Mixer + attention]
    DualStreamDINOLoss     concat | dual | cross   [reuses main.py's DINOLoss]

See README.md for the design and the phase-2 plan to add
`backbone_type="dualstream"` in main.py.
"""

from .decomposition import LearnedDecomp
from .dual_stream_backbone import DualStreamBackbone, TrendStream, SeasonStream
from .dual_stream_loss import DualStreamDINOLoss

__all__ = [
    "LearnedDecomp",
    "DualStreamBackbone", "TrendStream", "SeasonStream",
    "DualStreamDINOLoss",
]

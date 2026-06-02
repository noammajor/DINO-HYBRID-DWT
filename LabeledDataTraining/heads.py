"""
Prediction heads for the 7 LMC label columns.

Y layout:
  [0] latent_num        — classification  (discrete int in [min_latent, max_latent])
  [1] dirichlet         — conditional regression on [d_min, d_max]
  [2] weibull_shape     — positive regression
  [3] weibull_scale     — positive regression
  [4] dirichlet_min     — positive regression
  [5] dirichlet_max     — positive regression conditioned on d_min  (d_max = d_min + δ)
  [6] ess_length_scale  — positive regression

Hierarchy:
  d_min  ──►  d_max  (d_max = d_min + softplus(net))
  d_min, d_max  ──►  dirichlet  (d_min + sigmoid(net) * (d_max - d_min))

During training pass teacher_d_min / teacher_d_max (ground truth) to
LMCLabelHead.forward() so the dependent heads are conditioned correctly.
At inference leave them as None and predictions are chained.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, out_dim),
    )


class LatentNumHead(nn.Module):
    """Softmax classification over discrete integer range [min_val, max_val]."""

    def __init__(self, d_model: int, hidden_dim: int = 64,
                 min_val: int = 2, max_val: int = 10):
        super().__init__()
        self.min_val = min_val
        self.num_classes = max_val - min_val + 1
        self.net = _mlp(d_model, hidden_dim, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, d_model) encoder output. Returns logits (B, num_classes). Use CrossEntropyLoss."""
        return self.net(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Returns integer predictions (B,)."""
        return self.net(x).argmax(dim=-1) + self.min_val


class PositiveHead(nn.Module):
    """Softplus regression — guarantees output > 0."""

    def __init__(self, d_model: int, hidden_dim: int = 64):
        super().__init__()
        self.net = _mlp(d_model, hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, d_model) encoder output. Returns (B, 1), strictly positive."""
        return F.softplus(self.net(x))


class DirichletMaxHead(nn.Module):
    """
    Predicts d_max = d_min + δ  where δ = softplus(net([x, d_min])) > 0.
    Hardcodes the constraint d_max > d_min.
    """

    def __init__(self, d_model: int, hidden_dim: int = 64):
        super().__init__()
        self.net = _mlp(d_model + 1, hidden_dim, 1)

    def forward(self, x: torch.Tensor, d_min: torch.Tensor) -> torch.Tensor:
        delta = F.softplus(self.net(torch.cat([x, d_min], dim=-1)))
        return d_min + delta                    # (B, 1), always > d_min


class DirichletHead(nn.Module):
    """
    Predicts dirichlet ∈ (d_min, d_max) via
        d_min + sigmoid(net([x, d_min, d_max])) * (d_max - d_min).
    Hardcodes the constraint d_min < dirichlet < d_max.
    """

    def __init__(self, d_model: int, hidden_dim: int = 64):
        super().__init__()
        self.net = _mlp(d_model + 2, hidden_dim, 1)

    def forward(self, x: torch.Tensor,
                d_min: torch.Tensor, d_max: torch.Tensor) -> torch.Tensor:
        frac = torch.sigmoid(self.net(torch.cat([x, d_min, d_max], dim=-1)))
        return d_min + frac * (d_max - d_min)   # (B, 1)


class LMCLabelHead(nn.Module):
    """
    Full 7-label head. Takes the raw embedding produced by the encoder, shape (B, d_model).

    Args:
        d_model:      encoder output dimension
        hidden_dim:   hidden size for all MLP sub-heads
        min_latent:   smallest possible latent_num value
        max_latent:   largest possible latent_num value
    """

    def __init__(self, d_model: int, hidden_dim: int = 64,
                 min_latent: int = 2, max_latent: int = 10):
        super().__init__()
        self.latent_num_head    = LatentNumHead(d_model, hidden_dim, min_latent, max_latent)
        self.weibull_shape_head = PositiveHead(d_model, hidden_dim)
        self.weibull_scale_head = PositiveHead(d_model, hidden_dim)
        self.ess_ls_head        = PositiveHead(d_model, hidden_dim)
        self.d_min_head         = PositiveHead(d_model, hidden_dim)
        self.d_max_head         = DirichletMaxHead(d_model, hidden_dim)
        self.dirichlet_head     = DirichletHead(d_model, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        teacher_d_min: Optional[torch.Tensor] = None,
        teacher_d_max: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        x             : (B, d_model)  encoder output embedding
        teacher_d_min : (B, 1) ground-truth d_min — use during training
        teacher_d_max : (B, 1) ground-truth d_max — use during training
        """
        latent_logits = self.latent_num_head(x)
        weibull_shape = self.weibull_shape_head(x)
        weibull_scale = self.weibull_scale_head(x)
        ess_ls        = self.ess_ls_head(x)

        d_min      = self.d_min_head(x)
        d_min_cond = teacher_d_min if teacher_d_min is not None else d_min
        d_max      = self.d_max_head(x, d_min_cond)
        d_max_cond = teacher_d_max if teacher_d_max is not None else d_max
        dirichlet  = self.dirichlet_head(x, d_min_cond, d_max_cond)

        return {
            "latent_num_logits": latent_logits,   # (B, num_classes)  → CrossEntropyLoss
            "weibull_shape":     weibull_shape,    # (B, 1)            → MSELoss / HuberLoss
            "weibull_scale":     weibull_scale,    # (B, 1)
            "ess_length_scale":  ess_ls,           # (B, 1)
            "dirichlet_min":     d_min,            # (B, 1)
            "dirichlet_max":     d_max,            # (B, 1)
            "dirichlet":         dirichlet,        # (B, 1)
        }

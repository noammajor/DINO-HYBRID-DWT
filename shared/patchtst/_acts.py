"""Self-contained activation / stochastic-depth helpers for the relocated
PatchTST encoder bundle.

Copied verbatim from TSDiNO/utils/util.py so this package has no dependency on
TSDiNO (which is now TimeMixer-only). Consumed by patchTST.py.
"""
import torch
import torch.nn as nn


def get_activation_fn(activation):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "glu":
        return nn.GLU()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "leakyrelu":
        return nn.LeakyReLU()
    else:
        raise RuntimeError("activation should be relu/gelu/glu/tanh/sigmoid/leakyrelu, not {}".format(activation))


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

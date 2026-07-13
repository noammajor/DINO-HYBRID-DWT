#!/usr/bin/env python
"""
visualize_attn_paper.py — the paper "attention over the input" figure.

One stacked image sharing the x-axis:
  row 0  : the real input time series (one channel)
  row 1..: the SAME series, re-drawn with its line COLOURED by each model's
           attention over time (WINO-TS, Jitter+Crop, ...).

"Attention" here is input-gradient saliency ‖∂‖z‖/∂x_t‖ — weight-dependent (unlike
TimeMixer's near-uniform pooling weights), so the coloured lines actually differ
between pre-training recipes. Each model's map is min-max normalised so the colour
shows *where* it attends (the pattern), which is what an attention map conveys.

Each backbone is built + loaded separately; a per-model param-checksum is printed so
you can confirm the models are genuinely different.

Usage:
  python vis/vis/visualize_attn_paper.py \
      --dataset etth1 --var -1 --window 0 \
      --checkpoints ./ckpt_wino/checkpoint_best.pth ./ckpt_vision/checkpoint_best.pth \
      --labels WINO-TS "Jitter+Crop"
"""
import os
import sys
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in (os.path.join(_ROOT, "TimeMixer-main", "models"),
           os.path.join(_ROOT, "TimeMixer-main"),
           _ROOT, _HERE,
           os.path.join(_ROOT, "tsdino_timemixer")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from visualize_augmentations import load_windows, load_backbone   # noqa: E402


def grad_saliency(bb, window, device):
    """window [T,C] → saliency [C,T] = |∂‖pooled z‖ / ∂x_t|."""
    x = torch.as_tensor(window, dtype=torch.float32, device=device).unsqueeze(0)
    x.requires_grad_(True)
    z = bb(x)
    bb.zero_grad(set_to_none=True)
    z.norm().backward()
    return x.grad[0].abs().transpose(0, 1).detach().cpu().numpy()   # [C, T]


def _colored_line(ax, t, y, w, cmap, lw=2.6):
    """Draw y(t) as a line whose colour encodes w (min-max normalised)."""
    pts = np.array([t, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    wn = (w - w.min()) / (np.ptp(w) + 1e-12)
    lc = LineCollection(segs, cmap=cmap, norm=Normalize(0, 1), linewidth=lw)
    lc.set_array(wn[:-1])
    ax.add_collection(lc)
    pad = 0.08 * (np.ptp(y) + 1e-9)
    ax.set_xlim(t.min(), t.max()); ax.set_ylim(y.min() - pad, y.max() + pad)
    return lc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="etth1")
    p.add_argument("--data_dir", default="/home/shared/datasets/data - forecasting timeseries")
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--var", type=int, default=-1, help="channel index to plot (-1 = last, e.g. OT)")
    p.add_argument("--window", type=int, default=0)
    p.add_argument("--seq_len", type=int, default=336)
    p.add_argument("--cmap", default="magma")
    p.add_argument("--outdir", default=os.path.join(_ROOT, "vis", "maps"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    labels = args.labels or [os.path.basename(os.path.dirname(c)) for c in args.checkpoints]
    while len(labels) < len(args.checkpoints):
        labels.append(f"ckpt{len(labels)}")
    os.makedirs(args.outdir, exist_ok=True)

    n_win = max(args.window + 1, 4)
    samples, _, n_vars, col_names = load_windows(
        args.dataset, args.data_dir, n_win, args.seq_len, 0, args.seed)

    def _checksum(bb):
        return sum(float(v.float().mean()) for v in bb.state_dict().values() if v.numel())

    backbones, bsl, _ref = [], args.seq_len, next((c for c in args.checkpoints if os.path.isfile(c)), None)
    for c in args.checkpoints:
        if c.lower() in ("random", "none", "init") or not os.path.isfile(c):
            print(f"  backbone: RANDOM-INIT baseline")
            bb, _ok, s = load_backbone(_ref or c, n_vars, args.seq_len, device)
            for prm in bb.parameters():
                (torch.nn.init.xavier_uniform_ if prm.dim() > 1 else torch.nn.init.zeros_)(prm)
            bb.eval()
        else:
            print(f"  backbone: {c}")
            bb, _ok, s = load_backbone(c, n_vars, args.seq_len, device)
        print(f"    param-checksum = {_checksum(bb):+.6f}")
        backbones.append(bb); bsl = s
    if bsl != args.seq_len:
        samples, _, n_vars, col_names = load_windows(args.dataset, args.data_dir, n_win, bsl, 0, args.seed)

    win = samples[args.window].numpy()                       # [T, C]
    v = args.var if args.var >= 0 else n_vars - 1
    ser = win[:, v]
    T = len(ser)
    t = np.arange(T)
    sal = [grad_saliency(bb, samples[args.window], device)[v] for bb in backbones]   # each [T]

    nrow = 1 + len(backbones)
    fig, axes = plt.subplots(nrow, 1, figsize=(9.5, 1.9 * nrow), sharex=True)
    axes = np.atleast_1d(axes)
    axes[0].plot(t, ser, color="0.12", lw=1.7)
    axes[0].set_title("Input series", loc="left", fontsize=11, fontweight="bold")
    lc = None
    for mi, lab in enumerate(labels):
        ax = axes[mi + 1]
        lc = _colored_line(ax, t, ser, sal[mi], args.cmap)
        ax.set_title(f"{lab}  —  attention over time", loc="left", fontsize=11, fontweight="bold")
    axes[-1].set_xlabel("timestep", fontsize=11)
    for ax in axes:
        ax.set_yticks([])
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
    cbar = fig.colorbar(lc, ax=list(axes), fraction=0.018, pad=0.02)
    cbar.set_label("attention (normalised)", fontsize=10)
    cbar.set_ticks([0, 1]); cbar.set_ticklabels(["low", "high"])
    fig.suptitle(f"Where the model attends — {args.dataset} · {col_names[v]}", fontsize=13)
    out = os.path.join(args.outdir, f"attn_paper_{args.dataset}_{col_names[v]}.png")
    fig.savefig(out, dpi=160, bbox_inches="tight"); plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()

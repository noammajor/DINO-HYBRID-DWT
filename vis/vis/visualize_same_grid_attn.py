#!/usr/bin/env python
"""
visualize_same_grid_attn.py — overlay the input series and each model's attention
map, all standard-scaled (z-scored) onto ONE grid, so the "graphs created by the
attention" can be compared against each other (and the signal) directly.

"Attention" = input-gradient saliency ‖∂‖z‖/∂x_t‖ (weight-dependent, unlike
TimeMixer's near-uniform pooling weights). Each trace is z-scored (mean 0, unit
std) so shapes are comparable on a shared y-axis.

Output: vis/maps/same_grid_attn_map_<dataset>_<channel>.png

Usage:
  python vis/vis/visualize_same_grid_attn.py \
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
    x = torch.as_tensor(window, dtype=torch.float32, device=device).unsqueeze(0)
    x.requires_grad_(True)
    z = bb(x)
    bb.zero_grad(set_to_none=True)
    z.norm().backward()
    return x.grad[0].abs().transpose(0, 1).detach().cpu().numpy()   # [C, T]


def _z(a):
    a = np.asarray(a, dtype=np.float64)
    return (a - a.mean()) / (a.std() + 1e-8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="etth1")
    p.add_argument("--data_dir", default="/home/shared/datasets/data - forecasting timeseries")
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--var", type=int, default=-1, help="channel index (-1 = last / OT)")
    p.add_argument("--window", type=int, default=0)
    p.add_argument("--seq_len", type=int, default=336)
    p.add_argument("--no_series", action="store_true", help="omit the input-series trace")
    p.add_argument("--smooth", type=int, default=11, help="odd moving-avg window to de-noise saliency (1=off)")
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
            print("  backbone: RANDOM-INIT baseline")
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

    win = samples[args.window].numpy()
    v = args.var if args.var >= 0 else n_vars - 1
    ser = win[:, v]
    T = len(ser); t = np.arange(T)

    def _smooth(a):
        k = args.smooth
        if k and k > 1:
            k = k + 1 if k % 2 == 0 else k              # force odd
            pad = k // 2
            ap = np.pad(a, pad, mode="reflect")         # reflect edges → no zero-pad artefacts
            return np.convolve(ap, np.ones(k) / k, mode="valid")[:len(a)]
        return a

    sal = [_smooth(grad_saliency(bb, samples[args.window], device)[v]) for bb in backbones]

    # Pearson r between the two attention shapes (on the smoothed, plotted signals)
    corr = float(np.corrcoef(_z(sal[0]), _z(sal[1]))[0, 1]) if len(sal) >= 2 else None

    colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]
    styles = ["-", "--", "-.", ":"]
    strip_cmaps = ["Reds", "Blues", "Greens", "Purples"]   # WINO=red map, Jitter=blue map
    nstr = len(backbones)

    # top: overlaid z-scored curves.  below: one matching attention-map strip per model.
    fig, axes = plt.subplots(
        1 + nstr, 1, figsize=(11, 4.6 + 0.85 * nstr),
        gridspec_kw={"height_ratios": [4] + [0.55] * nstr}, sharex=True)
    axes = np.atleast_1d(axes)
    ax = axes[0]
    if not args.no_series:
        ax.plot(t, _z(ser), color="0.55", lw=1.4, alpha=0.8, label="Input series", zorder=1)
    for mi, lab in enumerate(labels):
        ax.plot(t, _z(sal[mi]), color=colors[mi % len(colors)], ls=styles[mi % len(styles)],
                lw=2.2, label=f"{lab} attn", zorder=3)
    ax.set_ylabel("standardized (z-score)", fontsize=12)
    ax.set_title(f"Attention maps on a shared grid — {args.dataset} · {col_names[v]}", fontsize=13)
    if corr is not None:
        ax.text(0.985, 0.03, f"Pearson $r$ = {corr:+.2f}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", alpha=0.9))
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=11, ncol=nstr + (0 if args.no_series else 1), loc="upper center")
    ax.tick_params(labelsize=10)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # matching attention-map strips (each min-max normalised so its pattern shows)
    for mi in range(nstr):
        axs = axes[1 + mi]
        a = sal[mi]
        axs.imshow(a[None, :], aspect="auto", cmap=strip_cmaps[mi % len(strip_cmaps)],
                   extent=[0, T, 0, 1], vmin=float(a.min()), vmax=float(a.max()),
                   interpolation="bilinear")
        axs.set_yticks([])
        axs.set_ylabel(labels[mi], rotation=0, ha="right", va="center",
                       fontsize=10, color=colors[mi % len(colors)])
    axes[-1].set_xlabel("timestep", fontsize=12)
    fig.tight_layout()
    out = os.path.join(args.outdir, f"same_grid_attn_map_{args.dataset}_{col_names[v]}.png")
    fig.savefig(out, dpi=160, bbox_inches="tight"); plt.close(fig)
    print(f"saved: {out}")

    # quick numeric: how similar are the two attention shapes?
    if len(sal) >= 2:
        corr = float(np.corrcoef(_z(sal[0]), _z(sal[1]))[0, 1])
        print(f"  corr({labels[0]} attn, {labels[1]} attn) = {corr:+.3f}   "
              f"(≈1 ⇒ identical shape ⇒ models not distinguishable by this map)")


if __name__ == "__main__":
    main()

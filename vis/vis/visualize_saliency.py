#!/usr/bin/env python
"""
visualize_saliency.py — gradient-based "attention" maps + concentration metrics,
comparing pre-training recipes (WINO-TS wavelet vs a vision/crop baseline).

TimeMixer's pooled attention weights are near-uniform, so they can't tell the
recipes apart. Input-gradient saliency — how sensitive the pooled representation
is to each input timestep, ‖∂‖z‖ / ∂x_t‖ — *is* weight-dependent and non-uniform,
so it makes a proper "what the model looks at" map that differs across recipes.

Outputs (under --outdir, default vis/maps/):
  • saliency_contrast_<ds>.png   3-panel heatmap: model A | model B | (A − B), channels×time
  • saliency_overlay_<ds>.png    side-by-side per-channel overlay (series + saliency fill)
  • saliency_metrics_<ds>.png    grouped bars: normalized entropy / Gini / top-10% mass per model
  • saliency_metrics_<ds>.csv    the scalar metrics (mean ± over windows)

Concentration metrics (per channel, averaged over channels & windows):
  entropy H  ∈[0,1]  lower  = more focused
  Gini       ∈[0,1]  higher = more focused
  top10 mass ∈[0,1]  higher = more focused

Usage:
  python vis/vis/visualize_saliency.py \
      --dataset etth1 --window 0 \
      --checkpoints ./ckpt_wino/checkpoint_best.pth ./ckpt_vision/checkpoint_best.pth \
      --labels WINO-TS "Jitter+Crop"
"""
import os
import sys
import csv
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from visualize_augmentations import load_windows, load_backbone   # noqa: E402


def grad_saliency(backbone, window, device):
    """window: [T, C] tensor/array → saliency [C, T] = |∂‖pooled z‖ / ∂x_t|."""
    x = torch.as_tensor(window, dtype=torch.float32, device=device).unsqueeze(0)  # [1,T,C]
    x.requires_grad_(True)
    z = backbone(x)                       # [1, C, d]  (pooled representation)
    backbone.zero_grad(set_to_none=True)
    z.norm().backward()
    g = x.grad[0].abs()                   # [T, C]
    return g.transpose(0, 1).detach().cpu().numpy()   # [C, T]


def concentration(a):
    """a: 1-D nonneg saliency over time → (norm-entropy, gini, top-10% mass)."""
    a = np.clip(np.asarray(a, dtype=np.float64), 0, None)
    a = a / (a.sum() + 1e-12)
    T = len(a)
    H = float(-(a * np.log(a + 1e-12)).sum() / np.log(T))
    srt = np.sort(a); idx = np.arange(1, T + 1)
    gini = float((2 * (idx * srt).sum()) / (T * srt.sum() + 1e-12) - (T + 1) / T)
    k = max(1, int(0.1 * T))
    top10 = float(np.sort(a)[::-1][:k].sum())
    return H, gini, top10


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="etth1")
    p.add_argument("--data_dir", default="/home/shared/datasets/data - forecasting timeseries")
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--seq_len", type=int, default=336)
    p.add_argument("--window", type=int, default=0, help="window index used for the map/overlay figures")
    p.add_argument("--n_samples", type=int, default=16, help="windows averaged over for the metrics")
    p.add_argument("--outdir", default=os.path.join(_ROOT, "vis", "maps"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    labels = args.labels or [os.path.basename(os.path.dirname(c)) for c in args.checkpoints]
    while len(labels) < len(args.checkpoints):
        labels.append(f"ckpt{len(labels)}")
    os.makedirs(args.outdir, exist_ok=True)

    n_win = max(args.n_samples, args.window + 1)
    samples, _, n_vars, col_names = load_windows(
        args.dataset, args.data_dir, n_win, args.seq_len, 0, args.seed)
    backbones, bsl = [], args.seq_len
    for c in args.checkpoints:
        print(f"  backbone: {c}")
        bb, _ok, s = load_backbone(c, n_vars, args.seq_len, device)
        backbones.append(bb); bsl = s
    if bsl != args.seq_len:
        samples, _, n_vars, col_names = load_windows(args.dataset, args.data_dir, n_win, bsl, 0, args.seed)

    win = samples[args.window].numpy()                 # [T, C]
    T = win.shape[0]
    sal = [grad_saliency(bb, samples[args.window], device) for bb in backbones]   # each [C, T]
    C = sal[0].shape[0]
    tt = np.arange(T)

    # ── 1) 3-panel (or M-panel) heatmap contrast ──────────────────────────────
    colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]
    ncols = len(backbones) + (1 if len(backbones) == 2 else 0)
    fig, axs = plt.subplots(1, ncols, figsize=(5.2 * ncols, 0.45 * C + 1.6), squeeze=False)
    axs = axs[0]
    vmax = max(float(s.max()) for s in sal)
    im0 = None
    for mi, (s, lab) in enumerate(zip(sal, labels)):
        im0 = axs[mi].imshow(s, aspect="auto", cmap="magma", vmin=0, vmax=vmax,
                             extent=[0, T, C - 0.5, -0.5], interpolation="nearest")
        axs[mi].set_title(lab, fontsize=11); axs[mi].set_xlabel("timestep")
    axs[0].set_yticks(range(C)); axs[0].set_yticklabels(col_names, fontsize=8)
    fig.colorbar(im0, ax=list(axs[:len(backbones)]), label="saliency", fraction=0.02, pad=0.02)
    if len(backbones) == 2:
        diff = sal[0] - sal[1]; dmax = float(np.abs(diff).max()) or 1e-9
        imd = axs[2].imshow(diff, aspect="auto", cmap="RdBu_r", vmin=-dmax, vmax=dmax,
                            extent=[0, T, C - 0.5, -0.5], interpolation="nearest")
        axs[2].set_title(f"{labels[0]} $-$ {labels[1]}", fontsize=11); axs[2].set_xlabel("timestep")
        fig.colorbar(imd, ax=axs[2], fraction=0.04, pad=0.02)
    fig.suptitle(f"Input-gradient saliency — {args.dataset} (window {args.window})", fontsize=12)
    out1 = os.path.join(args.outdir, f"saliency_contrast_{args.dataset}.png")
    fig.savefig(out1, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"saved: {out1}")

    # ── 2) side-by-side per-channel overlay (series + saliency fill) ──────────
    fig, axes = plt.subplots(C, len(backbones), figsize=(5.5 * len(backbones), 1.6 * C),
                             squeeze=False, sharex=True)
    for c in range(C):
        ser = win[:, c]
        if len(ser) != T:
            ser = ser[np.linspace(0, len(ser) - 1, T).astype(int)]
        for mi in range(len(backbones)):
            ax = axes[c][mi]
            ax.plot(tt, ser, color="0.55", lw=0.8)
            ax2 = ax.twinx()
            a = sal[mi][c]
            ax2.fill_between(tt, 0, a, color=colors[mi % len(colors)], alpha=0.30)
            ax2.plot(tt, a, color=colors[mi % len(colors)], lw=1.0)
            ax2.set_ylim(0, max(a.max() * 1.15, 1e-9)); ax2.set_yticks([])
            ax.tick_params(labelsize=7)
        axes[c][0].set_ylabel(col_names[c], fontsize=8)
    for mi, lab in enumerate(labels):
        axes[0][mi].set_title(lab, fontsize=12, color=colors[mi % len(colors)])
    fig.suptitle(f"Input-gradient saliency over time — {args.dataset}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out2 = os.path.join(args.outdir, f"saliency_overlay_{args.dataset}.png")
    fig.savefig(out2, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"saved: {out2}")

    # ── 3) concentration metrics averaged over n_samples windows ──────────────
    metric_names = ["entropy (↓)", "Gini (↑)", "top10 mass (↑)"]
    acc = np.zeros((len(backbones), 3)); nwin = 0
    for wi in range(min(args.n_samples, len(samples))):
        for mi, bb in enumerate(backbones):
            s = grad_saliency(bb, samples[wi], device)     # [C, T]
            per = np.array([concentration(s[c]) for c in range(s.shape[0])]).mean(0)
            acc[mi] += per
        nwin += 1
    acc /= max(nwin, 1)

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    x = np.arange(3); bw = 0.8 / len(backbones)
    for mi, lab in enumerate(labels):
        ax.bar(x + mi * bw, acc[mi], width=bw, color=colors[mi % len(colors)], label=lab)
    ax.set_xticks(x + bw * (len(backbones) - 1) / 2); ax.set_xticklabels(metric_names)
    ax.set_ylabel("mean over channels & windows")
    ax.set_title(f"Saliency concentration — {args.dataset}", fontsize=11)
    ax.legend(frameon=False)
    for mi in range(len(backbones)):
        for j in range(3):
            ax.text(x[j] + mi * bw, acc[mi, j], f"{acc[mi, j]:.3f}",
                    ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    out3 = os.path.join(args.outdir, f"saliency_metrics_{args.dataset}.png")
    fig.savefig(out3, dpi=150); plt.close(fig)
    print(f"saved: {out3}")

    outc = os.path.join(args.outdir, f"saliency_metrics_{args.dataset}.csv")
    with open(outc, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model"] + metric_names)
        for mi, lab in enumerate(labels):
            w.writerow([lab] + [f"{acc[mi, j]:.6f}" for j in range(3)])
    print(f"saved: {outc}")
    print("\n  saliency concentration (mean over channels & windows):")
    print("  {:20s} {}".format("", "  ".join(f"{n:>13s}" for n in metric_names)))
    for mi, lab in enumerate(labels):
        print("  {:20s} {}".format(lab, "  ".join(f"{acc[mi, j]:>13.4f}" for j in range(3))))


if __name__ == "__main__":
    main()

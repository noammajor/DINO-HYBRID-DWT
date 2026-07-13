#!/usr/bin/env python
"""
visualize_aug_effect.py — what the WINO-TS augmentations actually do to a signal.

Produces (under --outdir, default vis/aaai/):
  1. aug_rho_J{J}_<ds>_<ch>.png   soft-threshold EASY view vs shrinkage ρ∈{.2,.4,.6,.8,1.}
                                  — one image per decomposition level J (2,3,4).
  2. aug_level_<ds>_<ch>.png      soft-threshold EASY view vs level J∈{2,3,4} (ρ fixed).
  3. aug_families_<ds>_<ch>.png   classic DWT vs jitter vs gaussian, EASY and HARD views.

"Easy" (teacher) = gentle/structure-preserving; "Hard" (student) = aggressive.
  DWT      easy = soft_threshold(ρ)             hard = high_perturb (noise on detail bands)
  Jitter   easy = mild contrast/brightness      hard = strong jitter+contrast+brightness
  Gaussian easy = small additive noise          hard = large additive noise

Usage:
  python vis/vis/visualize_aug_effect.py --dataset etth1 --var -1 --window 0
"""
import os
import sys
import random
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

from visualize_augmentations import load_windows                     # noqa: E402
from data_agumentation import DWTAugmentation, gaussian_noise, jitter_contrast   # noqa: E402


def _seed(s=0):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def _apply(aug, x, v):
    _seed(0)                                   # deterministic view for the figure
    return aug(x)[:, v].detach().cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="etth1")
    p.add_argument("--data_dir", default="/home/shared/datasets/data - forecasting timeseries")
    p.add_argument("--var", type=int, default=-1, help="channel index (-1 = last / OT)")
    p.add_argument("--window", type=int, default=0)
    p.add_argument("--seq_len", type=int, default=336)
    p.add_argument("--wavelet", default="sym4")
    p.add_argument("--rhos", nargs="+", type=float, default=[0.2, 0.4, 0.6, 0.8, 1.0])
    p.add_argument("--levels", nargs="+", type=int, default=[2, 3, 4])
    p.add_argument("--rho_fixed", type=float, default=0.6)
    p.add_argument("--outdir", default=os.path.join(_ROOT, "vis", "aaai"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    samples, _, n_vars, col_names = load_windows(
        args.dataset, args.data_dir, max(args.window + 1, 4), args.seq_len, 0, args.seed)
    x = samples[args.window].float()            # [T, C]
    v = args.var if args.var >= 0 else n_vars - 1
    orig = x[:, v].numpy()
    T = len(orig); t = np.arange(T)
    ch = col_names[v]

    # ── 1) ρ sweep, one image per level J ─────────────────────────────────────
    cmap = plt.cm.viridis
    for J in args.levels:
        fig, ax = plt.subplots(figsize=(10, 4.2))
        ax.plot(t, orig, color="0.25", lw=2.0, label="original", zorder=5)
        for i, rho in enumerate(args.rhos):
            aug = DWTAugmentation(wavelet=args.wavelet, level=J, mode="soft_threshold",
                                  soft_threshold_sigma=float(rho))
            ax.plot(t, _apply(aug, x, v), color=cmap(i / max(len(args.rhos) - 1, 1)),
                    lw=1.6, alpha=0.9, label=f"ρ={rho:g}")
        ax.set_title(f"Soft-threshold easy view — effect of shrinkage ρ  (J={J}) — {args.dataset}·{ch}",
                     fontsize=12)
        ax.set_xlabel("timestep"); ax.set_ylabel("value")
        ax.legend(frameon=False, ncol=len(args.rhos) + 1, fontsize=9, loc="upper center")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        fig.tight_layout()
        out = os.path.join(args.outdir, f"aug_rho_J{J}_{args.dataset}_{ch}.png")
        fig.savefig(out, dpi=160); plt.close(fig); print(f"saved: {out}")

    # ── 2) level J sweep (ρ fixed) ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(t, orig, color="0.25", lw=2.0, label="original", zorder=5)
    lcolors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    for i, J in enumerate(args.levels):
        aug = DWTAugmentation(wavelet=args.wavelet, level=J, mode="soft_threshold",
                              soft_threshold_sigma=args.rho_fixed)
        ax.plot(t, _apply(aug, x, v), color=lcolors[i % len(lcolors)], lw=1.7, label=f"J={J}")
    ax.set_title(f"Soft-threshold easy view — effect of level J  (ρ={args.rho_fixed:g}) — {args.dataset}·{ch}",
                 fontsize=12)
    ax.set_xlabel("timestep"); ax.set_ylabel("value")
    ax.legend(frameon=False, ncol=len(args.levels) + 1, fontsize=10, loc="upper center")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    out = os.path.join(args.outdir, f"aug_level_{args.dataset}_{ch}.png")
    fig.savefig(out, dpi=160); plt.close(fig); print(f"saved: {out}")

    # ── 3) families: DWT vs jitter vs gaussian, easy & hard ───────────────────
    fam = [
        ("DWT (ours)",
         DWTAugmentation(wavelet=args.wavelet, level=3, mode="soft_threshold", soft_threshold_sigma=0.6),
         DWTAugmentation(wavelet=args.wavelet, level=3, mode="high_perturb",
                         high_perturb_noise_range=(0.2, 0.5))),
        ("Jitter",
         jitter_contrast(jitter_range=(0.0, 0.03), contrast_range=(0.95, 1.05), brightness_range=(-0.05, 0.05)),
         jitter_contrast(jitter_range=(0.10, 0.30), contrast_range=(0.70, 1.30), brightness_range=(-0.20, 0.20))),
        ("Gaussian",
         gaussian_noise(std_range=(0.02, 0.05)),
         gaussian_noise(std_range=(0.10, 0.30))),
    ]
    fig, axes = plt.subplots(len(fam), 1, figsize=(10, 2.4 * len(fam)), sharex=True)
    axes = np.atleast_1d(axes)
    for r, (name, easy, hard) in enumerate(fam):
        ax = axes[r]
        ax.plot(t, orig, color="0.6", lw=1.1, label="original")
        ax.plot(t, _apply(easy, x, v), color="#2ca02c", lw=1.5, label="easy (teacher)")
        ax.plot(t, _apply(hard, x, v), color="#d62728", lw=1.3, alpha=0.85, label="hard (student)")
        ax.set_ylabel(name, fontsize=11)
        ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper right")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[-1].set_xlabel("timestep")
    fig.suptitle(f"Augmentation families — easy vs hard views — {args.dataset}·{ch}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(args.outdir, f"aug_families_{args.dataset}_{ch}.png")
    fig.savefig(out, dpi=160); plt.close(fig); print(f"saved: {out}")


if __name__ == "__main__":
    main()

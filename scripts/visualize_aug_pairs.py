#!/usr/bin/env python
"""
visualize_aug_pairs.py — visualize every DINO augmentation family on one signal.

Shows, for each augmentation, the original series (grey) overlaid with the
augmented view (colour), grouped by family:

    DWT       : soft_threshold (teacher view), low_pass
    Zero-Out  : dwt_zero_out_detail
    Gaussian  : dwt_high_perturb (student view), gaussian_blur
    Physical  : lorentz, galilean, boost, rotation, polar, hyperbolic_warp

It also renders the actual DINO teacher/student PAIR (dwt_soft_threshold vs
dwt_hard) side-by-side, since that's the pair used in pretraining.

Usage
-----
  # synthetic demo signal (no data needed):
  python scripts/visualize_aug_pairs.py

  # a real window from a registered forecast dataset:
  python scripts/visualize_aug_pairs.py --dataset etth1 --var 0 --seq_len 336

  # reproducibility / wavelet pool:
  python scripts/visualize_aug_pairs.py --seed 0 --wavelets sym4 sym6 db4

Output: a PNG under ./vis/ (or --out).
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tsdino_timemixer"))

from data_agumentation import (          # noqa: E402  (from tsdino_timemixer/)
    DWTAugmentation, SWTAugmentation,
    lorentz_transformation, galilien_transformation, boost_transformation,
    rotation_transformation, polar_transformation, hyperbolic_amplitude_warp,
    gaussian_blur,
)


# ── signal source ──────────────────────────────────────────────────────────
def synthetic_signal(n, seed):
    """etth1-ish: strong low-freq periodicity + a faster component + noise."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n)
    x = (np.sin(2 * np.pi * 3 * t) + 0.5 * np.sin(2 * np.pi * 1 * t)
         + 0.25 * np.sin(2 * np.pi * 11 * t) + 0.1 * rng.standard_normal(n))
    return x.astype(np.float32)


def _load_cols(dataset):
    """Return (dataframe, list-of-value-columns) for a registered dataset."""
    import pandas as pd
    from dataset_registry import get_dataset_info
    info = get_dataset_info(dataset)
    df = pd.read_csv(info["csv_path"])
    cols = [c for c in df.columns if c.lower() not in ("date", "timestamp")]
    return df, cols


def real_signal(df, cols, var, n):
    """One standardized middle window of channel `var` (name returned too)."""
    col = cols[var % len(cols)]
    s = df[col].values.astype(np.float32)
    s = (s - s.mean()) / (s.std() + 1e-8)
    start = max(0, (len(s) - n) // 2)          # a middle window
    return s[start:start + n], col


# ── augmentation registry (family, label, instance) ───────────────────────
def build_augs(wavelets, level, sigma):
    pool = wavelets
    dwt = lambda mode, **kw: DWTAugmentation(
        wavelet_pool=pool, level=level, mode=mode, **kw)
    return [
        # family,       label,                         instance
        ("DWT",      "dwt_soft_threshold (teacher)",  dwt("soft_threshold", soft_threshold_sigma=sigma)),
        ("DWT",      "dwt_low_pass",                  dwt("low_pass")),
        ("Zero-Out", "dwt_zero_out_detail",           dwt("zero_out_detail", zero_out_ratio=0.5, finest_levels=2)),
        ("Zero-Out", "swt_zero_out_detail",           SWTAugmentation(wavelet=pool[0], level=level, mode="zero_out_detail")),
        ("Gaussian", "dwt_high_perturb (student)",    dwt("high_perturb", high_perturb_noise_range=(0.10, 0.30))),
        ("Gaussian", "gaussian_blur",                 gaussian_blur(sigma_range=(0.8, 2.0))),
        ("Physical", "lorentz",                       lorentz_transformation()),
        ("Physical", "galilean",                      galilien_transformation()),
        ("Physical", "boost",                         boost_transformation()),
        ("Physical", "rotation",                      rotation_transformation()),
        ("Physical", "polar",                         polar_transformation()),
        ("Physical", "hyperbolic_warp",               hyperbolic_amplitude_warp()),
    ]


_FAMILY_COLOR = {"DWT": "#1f77b4", "Zero-Out": "#2ca02c",
                 "Gaussian": "#d62728", "Physical": "#9467bd"}


def apply_aug(aug, x_np):
    """x_np: [seq_len] -> augmented [seq_len] (numpy)."""
    x = torch.from_numpy(x_np).float().unsqueeze(1)   # [seq_len, 1]
    with torch.no_grad():
        y = aug(x)
    y = y.detach().cpu().numpy()
    return y[:, 0] if y.ndim == 2 else np.asarray(y).reshape(-1)[:len(x_np)]


def render(x, title_src, out, augs, args):
    """Save the augmentation grid + teacher/student pair for one signal `x`."""
    t = np.arange(len(x))
    ncol = 3
    nrow = int(np.ceil(len(augs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 2.6 * nrow), squeeze=False)
    for i, (fam, label, aug) in enumerate(augs):
        ax = axes[i // ncol][i % ncol]
        y = apply_aug(aug, x)
        ax.plot(t, x, color="0.6", lw=1.0, label="original", zorder=1)
        ax.plot(t[:len(y)], y, color=_FAMILY_COLOR[fam], lw=1.4, label="augmented", zorder=2)
        ax.set_title(f"[{fam}]  {label}", fontsize=9)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7, loc="upper right")
    for j in range(len(augs), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(f"DINO augmentations on {title_src}  "
                 f"(wavelets={','.join(args.wavelets)}, level={args.level}, sigma={args.sigma})",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"saved: {out}")

    teacher = DWTAugmentation(wavelet_pool=args.wavelets, level=args.level,
                              mode=args.teacher_mode, soft_threshold_sigma=args.sigma)
    student = DWTAugmentation(wavelet_pool=args.wavelets, level=args.level,
                              mode="high_perturb", high_perturb_noise_range=(0.10, 0.30))
    fig2, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(t, x, color="0.6", lw=1.0, label="original")
    ax.plot(t, apply_aug(teacher, x), color="#1f77b4", lw=1.5, label=f"teacher (dwt_{args.teacher_mode})")
    ax.plot(t, apply_aug(student, x), color="#d62728", lw=1.5, alpha=0.8, label="student (dwt_hard / high_perturb)")
    ax.set_title(f"DINO teacher/student pair — {title_src}", fontsize=10)
    ax.legend(fontsize=8); ax.tick_params(labelsize=8)
    fig2.tight_layout()
    out2 = out.replace(".png", "_pair.png")
    fig2.savefig(out2, dpi=130); plt.close(fig2)
    print(f"saved: {out2}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=None, help="registered dataset for a real window (else synthetic)")
    p.add_argument("--var", type=int, default=0, help="which channel of the dataset")
    p.add_argument("--all_vars", action="store_true", help="render one figure per channel of the dataset")
    p.add_argument("--seq_len", type=int, default=336)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wavelets", nargs="+", default=["sym4", "sym6", "sym8", "db4", "db6"])
    p.add_argument("--level", type=int, default=3)
    p.add_argument("--sigma", type=float, default=0.6, help="soft_threshold sigma")
    p.add_argument("--teacher_mode", default="low_pass",
                   choices=["low_pass", "soft_threshold"],
                   help="DWT mode for the teacher view in the pair figure (student is always high_perturb/dwt_hard)")
    p.add_argument("--outdir", default=str(ROOT / "vis"))
    args = p.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    import random as _r; _r.seed(args.seed)
    augs = build_augs(args.wavelets, args.level, args.sigma)

    if not args.dataset:
        render(synthetic_signal(args.seq_len, args.seed), "synthetic",
               os.path.join(args.outdir, "aug_pairs_synthetic.png"), augs, args)
        return

    df, cols = _load_cols(args.dataset)
    vars_to_do = range(len(cols)) if args.all_vars else [args.var]
    for v in vars_to_do:
        # reseed per var so the stochastic augs are comparable across channels
        np.random.seed(args.seed); torch.manual_seed(args.seed); _r.seed(args.seed)
        x, colname = real_signal(df, cols, v, args.seq_len)
        out = os.path.join(args.outdir, f"aug_pairs_{args.dataset}_var{v}_{colname}.png")
        render(x, f"{args.dataset} · ch {v} ({colname})", out, augs, args)


if __name__ == "__main__":
    main()

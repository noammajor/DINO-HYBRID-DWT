#!/usr/bin/env python
"""
visualize_invariance.py — representation invariance to high-frequency (wavelet
detail) perturbation: WINO-TS vs a vision/crop baseline.

Idea (the figure the attention map couldn't make): take clean windows, add
increasing Gaussian noise to the *detail* (high-frequency) wavelet coefficients,
reconstruct back to the time domain, and measure how far each backbone's
per-timestep representation drifts from the clean one. WINO-TS was trained (via
soft-thresholding) to be invariant to exactly those detail perturbations, so its
curve stays low/flat; a model trained on vision-style jitter/crop was not, so its
representation drifts up as the detail band is corrupted.

We use the PRE-POOLING token representation (forward_ibot). The cross-attention
pooling in forward() averages the high-frequency effect away — that is why the
pooled attention map looked uniform for both models.

Usage
-----
  python vis/vis/visualize_invariance.py \
      --dataset etth1 \
      --checkpoints ./checkpoints_etth1_..._full_dino_seed42/checkpoint_best.pth \
                    ./checkpoints_etth1_..._vision_jittercrop/checkpoint_best.pth \
      --labels WINO-TS "Jitter+Crop"
"""
import os
import sys
import argparse

import numpy as np
import torch
import pywt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
# Seed correct repo paths before importing visualize_augmentations (it mis-computes
# the repo root when two dirs deep, breaking `import data_agumentation`).
for _p in (os.path.join(_ROOT, "TimeMixer-main", "models"),
           os.path.join(_ROOT, "TimeMixer-main"),
           _ROOT, _HERE,
           os.path.join(_ROOT, "tsdino_timemixer")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Reuse the working data / backbone / representation machinery.
from visualize_augmentations import (              # noqa: E402
    load_windows, load_backbone, get_tokens, token_rel_l2,
)


def _perturb_detail(x, sigma, wavelet, level, rng):
    """x: [T, C] float array → detail-band-noised reconstruction, same shape.
    Per detail level, add Gaussian noise with std = sigma * std(level coeffs).
    The approximation (low-frequency) band is left untouched."""
    T, C = x.shape
    out = np.empty_like(x)
    for c in range(C):
        coeffs = pywt.wavedec(x[:, c], wavelet, level=level, mode="symmetric")
        new = [coeffs[0]]                                   # keep approximation
        for cd in coeffs[1:]:                               # perturb all detail bands
            new.append(cd + sigma * (cd.std() + 1e-8) * rng.standard_normal(cd.shape))
        rec = pywt.waverec(new, wavelet, mode="symmetric")
        if len(rec) < T:
            rec = np.pad(rec, (0, T - len(rec)))
        out[:, c] = rec[:T]
    return out.astype(np.float32)


@torch.no_grad()
def _drift(backbone, x_clean, x_pert, device):
    """Relative-L2 drift of the per-timestep token representation (sensitive to
    high-frequency change, unlike the pooled representation)."""
    z0 = get_tokens(backbone, x_clean, device)              # [T, C, d]
    z1 = get_tokens(backbone, torch.as_tensor(x_pert), device)
    return token_rel_l2(z0, z1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="etth1")
    p.add_argument("--data_dir", default="/home/shared/datasets/data - forecasting timeseries")
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--seq_len", type=int, default=336)
    p.add_argument("--n_samples", type=int, default=16, help="windows averaged over")
    p.add_argument("--repeats", type=int, default=4, help="noise draws per (sigma, window)")
    p.add_argument("--sigmas", nargs="+", type=float,
                   default=[0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0])
    p.add_argument("--wavelet", default="sym4")
    p.add_argument("--level", type=int, default=3)
    p.add_argument("--outdir", default=os.path.join(os.path.dirname(_HERE), "vis"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    labels = args.labels or [os.path.basename(os.path.dirname(c)) for c in args.checkpoints]
    while len(labels) < len(args.checkpoints):
        labels.append(f"ckpt{len(labels)}")

    samples, _, n_vars, _ = load_windows(
        args.dataset, args.data_dir, args.n_samples, args.seq_len, 0, args.seed)

    def _checksum(bb):
        return sum(float(v.float().mean()) for v in bb.state_dict().values() if v.numel())

    # Each checkpoint gets its OWN freshly-built backbone (load_backbone builds a new
    # TSMixerForDINO every call). A distinct param-checksum per model proves they are
    # genuinely different weights — if two match, the checkpoints are the same model.
    _ref = next((c for c in args.checkpoints if os.path.isfile(c)), None)
    backbones, bsl = [], args.seq_len
    for c in args.checkpoints:
        if c.lower() in ("random", "none", "init") or not os.path.isfile(c):
            print(f"  backbone: RANDOM-INIT baseline (arch matched to {os.path.basename(os.path.dirname(_ref)) if _ref else 'config defaults'})")
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
        print(f"  reloading windows at backbone seq_len={bsl}")
        samples, _, _, _ = load_windows(args.dataset, args.data_dir, args.n_samples, bsl, 0, args.seed)

    rng = np.random.default_rng(args.seed)
    curves = np.zeros((len(backbones), len(args.sigmas)))
    for si, sigma in enumerate(args.sigmas):
        for xt in samples:
            x = xt.numpy()
            for _ in range(args.repeats):
                xp = _perturb_detail(x, sigma, args.wavelet, args.level, rng)
                for mi, bb in enumerate(backbones):
                    curves[mi, si] += _drift(bb, xt, xp, device)
    curves /= (len(samples) * args.repeats)

    # ── the figure: representation drift vs perturbation strength ──────────────
    os.makedirs(args.outdir, exist_ok=True)
    # Fixed style per role when the usual 3 labels are present; else cycle.
    role_style = {
        "regular":    dict(color="#7f7f7f", ls="--", marker="s"),
        "wino-ts":    dict(color="#d62728", ls="-",  marker="o"),
        "wino":       dict(color="#d62728", ls="-",  marker="o"),
        "jitter+crop":dict(color="#1f77b4", ls="-.", marker="^"),
        "jitter":     dict(color="#1f77b4", ls="-.", marker="^"),
    }
    fallback = [dict(color=c, ls="-", marker="o") for c in ("#d62728", "#1f77b4", "#7f7f7f", "#2ca02c")]
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for mi, lab in enumerate(labels):
        st = role_style.get(lab.strip().lower(), fallback[mi % len(fallback)])
        ax.plot(args.sigmas, curves[mi], lw=2.5, ms=6.5, label=lab, **st)
    ax.set_xlabel(r"detail-band perturbation strength  $\sigma$", fontsize=12)
    ax.set_ylabel(r"representation drift  (relative $L_2$)", fontsize=12)
    ax.set_title(f"Robustness to high-frequency perturbation — {args.dataset}", fontsize=13)
    ax.grid(alpha=0.25); ax.legend(frameon=False, fontsize=11, loc="upper left")
    ax.set_ylim(bottom=0); ax.tick_params(labelsize=10)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    slug = "_vs_".join(l.replace(" ", "").replace("+", "") for l in labels)
    out = os.path.join(args.outdir, f"invariance_{args.dataset}_{slug}.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"saved: {out}")
    for mi, lab in enumerate(labels):
        auc = float(np.trapz(curves[mi], args.sigmas))
        print(f"  {lab:18s}  drift@σ=1.0={curves[mi, -1]:.4f}   AUC={auc:.4f}")


if __name__ == "__main__":
    main()

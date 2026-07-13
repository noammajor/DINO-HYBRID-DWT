#!/usr/bin/env python
"""
visualize_maps.py — diagnostic suite comparing pre-training recipes (e.g.
WINO-TS wavelet vs a vision/crop baseline) by REPRESENTATION INVARIANCE.

Motivation: TimeMixer's pooled attention (and token magnitude) are input-dominated,
so attention "maps" look identical across recipes. What *does* separate them is how
much the representation moves when the input is perturbed in different frequency /
augmentation families — each model is most invariant to the family it was trained on.

Outputs (under --outdir, default vis/maps/):
  • invariance_curve_<perturb>_<ds>.png   one per perturbation family: drift vs strength σ, a line per model
  • invariance_matrix_<ds>.png            models × perturbation-families heatmap at σ=--sigma_fixed
  • invariance_metrics_<ds>.csv           every scalar (drift per model / family / σ, plus AUC)

Perturbation families:
  detail(HF)  — noise on wavelet DETAIL bands   (WINO's training domain → expect WINO flat)
  approx(LF)  — noise on the APPROXIMATION band  (structure; expect BOTH sensitive — a control)
  gaussian    — broadband additive noise
  jitter      — per-timestep amplitude scaling   (vision-style)

Usage:
  python vis/vis/visualize_maps.py \
      --dataset etth1 \
      --checkpoints ./ckpt_wino/checkpoint_best.pth ./ckpt_vision/checkpoint_best.pth \
      --labels WINO-TS "Jitter+Crop"
"""
import os
import sys
import csv
import argparse

import numpy as np
import torch
import pywt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
# visualize_augmentations (two dirs deep) mis-computes the repo root, so seed the
# real paths first — TimeMixer models, repo root, then tsdino_timemixer LAST so its
# models/ + data_agumentation win.
for _p in (os.path.join(_ROOT, "TimeMixer-main", "models"),
           os.path.join(_ROOT, "TimeMixer-main"),
           _ROOT, _HERE,
           os.path.join(_ROOT, "tsdino_timemixer")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from visualize_augmentations import (              # noqa: E402
    load_windows, load_backbone, get_tokens, token_rel_l2,
)


# ── perturbation families (all operate on [T, C] float arrays) ─────────────────
def _dwt_perturb(x, sigma, rng, wavelet, level, band):
    T, C = x.shape
    out = np.empty_like(x)
    for c in range(C):
        co = pywt.wavedec(x[:, c], wavelet, level=level, mode="symmetric")
        new = list(co)
        if band == "approx":
            new[0] = co[0] + sigma * (co[0].std() + 1e-8) * rng.standard_normal(co[0].shape)
        else:  # detail: perturb every detail band
            for j in range(1, len(co)):
                new[j] = co[j] + sigma * (co[j].std() + 1e-8) * rng.standard_normal(co[j].shape)
        rec = pywt.waverec(new, wavelet, mode="symmetric")
        if len(rec) < T:
            rec = np.pad(rec, (0, T - len(rec)))
        out[:, c] = rec[:T]
    return out.astype(np.float32)


def _gaussian(x, sigma, rng):
    return (x + sigma * (x.std(0, keepdims=True) + 1e-8) * rng.standard_normal(x.shape)).astype(np.float32)


def _jitter(x, sigma, rng):
    scale = 1.0 + sigma * rng.standard_normal((x.shape[0], 1))
    return (x * scale).astype(np.float32)


def make_perturbs(wavelet, level):
    return {
        "detail(HF)": lambda x, s, rng: _dwt_perturb(x, s, rng, wavelet, level, "detail"),
        "approx(LF)": lambda x, s, rng: _dwt_perturb(x, s, rng, wavelet, level, "approx"),
        "gaussian":   lambda x, s, rng: _gaussian(x, s, rng),
        "jitter":     lambda x, s, rng: _jitter(x, s, rng),
    }


@torch.no_grad()
def _drift(bb, xt, xp, device):
    return token_rel_l2(get_tokens(bb, xt, device),
                        get_tokens(bb, torch.as_tensor(xp), device))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="etth1")
    p.add_argument("--data_dir", default="/home/shared/datasets/data - forecasting timeseries")
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--seq_len", type=int, default=336)
    p.add_argument("--n_samples", type=int, default=16)
    p.add_argument("--repeats", type=int, default=4)
    p.add_argument("--sigmas", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0])
    p.add_argument("--sigma_fixed", type=float, default=0.5, help="σ used for the matrix heatmap")
    p.add_argument("--wavelet", default="sym4")
    p.add_argument("--level", type=int, default=3)
    p.add_argument("--outdir", default=os.path.join(_ROOT, "vis", "maps"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    labels = args.labels or [os.path.basename(os.path.dirname(c)) for c in args.checkpoints]
    while len(labels) < len(args.checkpoints):
        labels.append(f"ckpt{len(labels)}")
    os.makedirs(args.outdir, exist_ok=True)
    perturbs = make_perturbs(args.wavelet, args.level)
    pnames = list(perturbs)

    samples, _, n_vars, _ = load_windows(
        args.dataset, args.data_dir, args.n_samples, args.seq_len, 0, args.seed)
    backbones, bsl = [], args.seq_len
    for c in args.checkpoints:
        print(f"  backbone: {c}")
        bb, _ok, s = load_backbone(c, n_vars, args.seq_len, device)
        backbones.append(bb); bsl = s
    if bsl != args.seq_len:
        samples, _, _, _ = load_windows(args.dataset, args.data_dir, args.n_samples, bsl, 0, args.seed)

    M, P, S = len(backbones), len(pnames), len(args.sigmas)
    # curves[model, perturb, sigma] = mean token drift
    curves = np.zeros((M, P, S))
    rng = np.random.default_rng(args.seed)
    for pi, pname in enumerate(pnames):
        for si, sigma in enumerate(args.sigmas):
            for xt in samples:
                x = xt.numpy()
                for _ in range(args.repeats):
                    xp = perturbs[pname](x, sigma, rng)
                    for mi, bb in enumerate(backbones):
                        curves[mi, pi, si] += _drift(bb, xt, xp, device)
    curves /= (len(samples) * args.repeats)

    colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]

    # ── 1) one drift-vs-σ curve figure per perturbation family ────────────────
    for pi, pname in enumerate(pnames):
        fig, ax = plt.subplots(figsize=(6.0, 4.1))
        for mi, lab in enumerate(labels):
            ax.plot(args.sigmas, curves[mi, pi], marker="o", lw=2.2,
                    color=colors[mi % len(colors)], label=lab)
        ax.set_xlabel(r"perturbation strength $\sigma$")
        ax.set_ylabel(r"representation drift (rel. $L_2$)")
        ax.set_title(f"Invariance to {pname} — {args.dataset}\n(flatter = more invariant)", fontsize=11)
        ax.grid(alpha=0.3); ax.legend(frameon=False); ax.set_ylim(bottom=0)
        fig.tight_layout()
        safe = pname.replace("(", "").replace(")", "")
        out = os.path.join(args.outdir, f"invariance_curve_{safe}_{args.dataset}.png")
        fig.savefig(out, dpi=150); plt.close(fig)
        print(f"saved: {out}")

    # ── 2) models × perturbation-family matrix at σ_fixed ─────────────────────
    si_fixed = int(np.argmin(np.abs(np.array(args.sigmas) - args.sigma_fixed)))
    mat = curves[:, :, si_fixed]                                  # [M, P]
    fig, ax = plt.subplots(figsize=(1.6 * P + 2, 1.0 * M + 1.6))
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_xticks(range(P)); ax.set_xticklabels(pnames, rotation=20, ha="right", fontsize=9)
    ax.set_yticks(range(M)); ax.set_yticklabels(labels, fontsize=10)
    for mi in range(M):
        for pi in range(P):
            ax.text(pi, mi, f"{mat[mi, pi]:.3f}", ha="center", va="center",
                    color="w" if mat[mi, pi] > mat.max() * 0.55 else "k", fontsize=9)
    ax.set_title(f"Representation drift by perturbation family — {args.dataset} "
                 f"(σ={args.sigmas[si_fixed]:g}; lower = more invariant)", fontsize=10)
    fig.colorbar(im, ax=ax, label="drift", fraction=0.046, pad=0.04)
    fig.tight_layout()
    outm = os.path.join(args.outdir, f"invariance_matrix_{args.dataset}.png")
    fig.savefig(outm, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"saved: {outm}")

    # ── 3) CSV dump of every scalar ───────────────────────────────────────────
    outc = os.path.join(args.outdir, f"invariance_metrics_{args.dataset}.csv")
    with open(outc, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "perturb", "sigma", "drift"])
        for mi, lab in enumerate(labels):
            for pi, pname in enumerate(pnames):
                for si, sigma in enumerate(args.sigmas):
                    w.writerow([lab, pname, sigma, f"{curves[mi, pi, si]:.6f}"])
        w.writerow([])
        w.writerow(["model", "perturb", "AUC(drift vs sigma)"])
        for mi, lab in enumerate(labels):
            for pi, pname in enumerate(pnames):
                w.writerow([lab, pname, f"{np.trapz(curves[mi, pi], args.sigmas):.6f}"])
    print(f"saved: {outc}")

    # ── console summary: the key contrast ─────────────────────────────────────
    print("\n  drift @ σ={:g}  (lower = more invariant):".format(args.sigmas[si_fixed]))
    print("  {:20s} {}".format("", "  ".join(f"{n:>11s}" for n in pnames)))
    for mi, lab in enumerate(labels):
        print("  {:20s} {}".format(lab, "  ".join(f"{mat[mi, pi]:>11.4f}" for pi in range(P))))


if __name__ == "__main__":
    main()

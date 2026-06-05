#!/usr/bin/env python3
"""
visualize_views.py — Plot teacher and student augmentation views for TSDiNO.

Shows:
  Row 1 : Original vs Teacher  (modwt_soft,  sym4)
  Row 2 : Original vs Student  (modwt_hard,  sym4)
  Row 3 : Teacher sym4 vs db4  (phase-distortion comparison)

Usage
-----
  # with real data
  python scripts/visualize_views.py --data_path /path/to/ETTh1.csv

  # generate a synthetic signal instead
  python scripts/visualize_views.py --synthetic

  # choose channel and window position
  python scripts/visualize_views.py --data_path /path/to/ETTh1.csv --channel 2 --window_start 200

  # optional: pass a checkpoint to compare encoder embeddings
  python scripts/visualize_views.py --data_path /path/to/ETTh1.csv --ckpt /path/to/checkpoint.pth
"""

import argparse
import os
import sys
import random
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from TSDiNO.data_agumentation import MODWTAugmentation


# ── helpers ───────────────────────────────────────────────────────────────────

def load_window(data_path: str, window_start: int, seq_len: int, channel: int):
    import pandas as pd
    df = pd.read_csv(data_path)
    numeric = df.select_dtypes(include="number")
    arr = numeric.values.astype(np.float32)          # [T, C]
    # instance-normalize the window (same as ReVIN)
    window = arr[window_start : window_start + seq_len]
    mean = window.mean(axis=0, keepdims=True)
    std  = window.std(axis=0, keepdims=True) + 1e-8
    window = (window - mean) / std
    return window, channel


def make_synthetic(seq_len: int):
    t = np.linspace(0, 4 * np.pi, seq_len).astype(np.float32)
    sig = (
        np.sin(t)
        + 0.5 * np.sin(3 * t + 0.5)
        + 0.1 * np.random.randn(seq_len).astype(np.float32)
    )
    window = sig[:, None]          # [T, 1]
    return window, 0


def apply_aug(window_np: np.ndarray, wavelet: str, mode: str,
              level: int = 3, sigma: float = 0.3,
              noise_range: tuple = (0.05, 0.12), finest_levels: int = 2) -> np.ndarray:
    aug = MODWTAugmentation(
        wavelet=wavelet, level=level, mode=mode,
        soft_threshold_sigma=sigma,
        high_perturb_noise_range=noise_range,
        finest_levels=finest_levels,
    )
    x = torch.tensor(window_np)
    out = aug(x).numpy()
    return out


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.flatten(), b.flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# ── optional: encoder embedding similarity ────────────────────────────────────

def embedding_similarities(ckpt_path: str, original: np.ndarray,
                            teacher: np.ndarray, student: np.ndarray):
    """
    Load a TSDiNO backbone checkpoint and return cosine similarities between
    the original embedding and each view's embedding.
    Returns None if loading fails.
    """
    try:
        import torch.nn as nn
        sys.path.insert(0, str(ROOT / "TSDiNO"))
        from TSDiNO.config import config as cfg

        ckpt = torch.load(ckpt_path, map_location="cpu")

        # try to find the student encoder weights
        state = ckpt.get("student", ckpt.get("model", ckpt))
        if isinstance(state, dict) and any(k.startswith("module.") for k in state):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}

        # build model
        from TSDiNO.main import build_model
        model = build_model(cfg)
        model.load_state_dict(state, strict=False)
        model.eval()

        def encode(x_np):
            # x_np: [T, C] → add batch dim → [1, T, C]
            x = torch.tensor(x_np).unsqueeze(0).float()
            with torch.no_grad():
                emb = model(x)
            return emb.squeeze(0).numpy()

        emb_orig    = encode(original)
        emb_teacher = encode(teacher)
        emb_student = encode(student)
        return {
            "orig↔teacher": cosine_sim(emb_orig, emb_teacher),
            "orig↔student": cosine_sim(emb_orig, emb_student),
            "teacher↔student": cosine_sim(emb_teacher, emb_student),
        }
    except Exception as e:
        print(f"[visualize] could not compute embeddings: {e}")
        return None


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path",    type=str,  default=None)
    p.add_argument("--synthetic",    action="store_true")
    p.add_argument("--channel",      type=int,  default=0)
    p.add_argument("--window_start", type=int,  default=0)
    p.add_argument("--seq_len",      type=int,  default=336)
    p.add_argument("--level",        type=int,  default=3)
    p.add_argument("--sigma",        type=float,default=0.3)
    p.add_argument("--noise_lo",     type=float,default=0.05)
    p.add_argument("--noise_hi",     type=float,default=0.12)
    p.add_argument("--n_student",    type=int,  default=3,
                   help="how many student-view samples to overlay (shows stochasticity)")
    p.add_argument("--ckpt",         type=str,  default=None,
                   help="optional checkpoint for embedding similarity")
    p.add_argument("--out",          type=str,  default="plots/aug_views.png")
    args = p.parse_args()

    random.seed(0)
    np.random.seed(0)

    # ── load data ──────────────────────────────────────────────────────────────
    if args.synthetic or args.data_path is None:
        print("[visualize] using synthetic signal")
        window, ch = make_synthetic(args.seq_len)
    else:
        window, ch = load_window(args.data_path, args.window_start,
                                 args.seq_len, args.channel)

    original = window                          # [T, C]
    noise_range = (args.noise_lo, args.noise_hi)

    # ── apply augmentations ───────────────────────────────────────────────────
    teacher_sym4 = apply_aug(original, "sym4", "soft_threshold",
                             args.level, args.sigma, finest_levels=2)
    teacher_db4  = apply_aug(original, "db4",  "soft_threshold",
                             args.level, args.sigma, finest_levels=2)
    students_sym4 = [
        apply_aug(original, "sym4", "high_perturb",
                  args.level, args.sigma, noise_range, finest_levels=2)
        for _ in range(args.n_student)
    ]

    t = np.arange(args.seq_len)
    orig_ch     = original[:, ch]
    teach_s4_ch = teacher_sym4[:, ch]
    teach_d4_ch = teacher_db4[:, ch]
    stud_chs    = [s[:, ch] for s in students_sym4]

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(
        f"Augmentation views  |  channel {ch}  |  seq_len={args.seq_len}  |  "
        f"MODWT level={args.level}  sigma={args.sigma}  noise={noise_range}",
        fontsize=10
    )

    # Row 1 — original vs teacher (sym4)
    ax = axes[0]
    ax.plot(t, orig_ch,     color="steelblue",  lw=1.5, label="original")
    ax.plot(t, teach_s4_ch, color="darkorange", lw=1.5, label="teacher  (sym4, soft_threshold)")
    ax.fill_between(t, orig_ch, teach_s4_ch, alpha=0.12, color="darkorange")
    ax.set_ylabel("amplitude")
    ax.set_title("Teacher view")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Row 2 — original vs student samples (sym4)
    ax = axes[1]
    ax.plot(t, orig_ch, color="steelblue", lw=1.5, label="original", zorder=3)
    colors = ["firebrick", "tomato", "salmon"]
    for i, sc in enumerate(stud_chs):
        ax.plot(t, sc, color=colors[i % len(colors)], lw=0.9, alpha=0.75,
                label=f"student sample {i+1}  (sym4, high_perturb)")
    ax.set_ylabel("amplitude")
    ax.set_title(f"Student view  ({args.n_student} noise realisations)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Row 3 — teacher sym4 vs db4 (phase comparison)
    ax = axes[2]
    ax.plot(t, teach_s4_ch, color="darkorange", lw=1.5, label="teacher  sym4  (near-linear phase)")
    ax.plot(t, teach_d4_ch, color="purple",     lw=1.5, label="teacher  db4   (asymmetric phase)", alpha=0.8)
    diff = teach_s4_ch - teach_d4_ch
    ax.fill_between(t, 0, diff, alpha=0.2, color="purple", label=f"sym4 − db4  (max |Δ|={np.abs(diff).max():.4f})")
    ax.set_xlabel("timestep")
    ax.set_ylabel("amplitude")
    ax.set_title("Phase comparison: sym4 vs db4 teacher views")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[visualize] saved → {out_path}")

    # ── signal-level cosine similarities (no model needed) ───────────────────
    print("\nSignal-level cosine similarities:")
    print(f"  orig ↔ teacher (sym4) : {cosine_sim(orig_ch, teach_s4_ch):.4f}")
    print(f"  orig ↔ teacher (db4)  : {cosine_sim(orig_ch, teach_d4_ch):.4f}")
    for i, sc in enumerate(stud_chs):
        print(f"  orig ↔ student {i+1}      : {cosine_sim(orig_ch, sc):.4f}")
    print(f"  teacher(sym4) ↔ db4   : {cosine_sim(teach_s4_ch, teach_d4_ch):.4f}")

    # ── optional embedding similarities ──────────────────────────────────────
    if args.ckpt:
        sims = embedding_similarities(args.ckpt, original, teacher_sym4, students_sym4[0])
        if sims:
            print("\nEncoder embedding cosine similarities:")
            for k, v in sims.items():
                print(f"  {k} : {v:.4f}")


if __name__ == "__main__":
    main()

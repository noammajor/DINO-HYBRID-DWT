#!/usr/bin/env python
"""
visualize_attn_signal.py — paper figure: attention-as-signal over the real series.

For each variable it produces a 3-panel, time-aligned figure:
  (A) the real series, with the LINE COLORED by the CLS attention weight
      (high-attention timesteps light up on the actual curve);
  (B) the normalized attention curve overlaid on the normalized series, with the
      Pearson correlation r annotated (shape similarity);
  (C) the attention heat strip, spread under the data graph.

It also prints a robust "similarity" summary: the correlation of attention with
  - the signal          corr(a, x)
  - the saliency         corr(a, |x - mean|)
  - the change/transitions corr(a, |dx/dt|)
averaged over `--n_windows` windows (mean +/- std), so the reported similarity is
not cherry-picked from a single window. The rendered figure uses `--window`.

Usage
-----
  python scripts/visualize_attn_signal.py \
      --dataset etth1 \
      --ckpt checkpoints_etth1_layers4_outdim1024_timemixer_grid_full_dino/checkpoint_best.pth \
      --window 0 --n_windows 50
"""
import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

ROOT = Path(__file__).parent.parent.resolve()
DINO = ROOT / "tsdino_timemixer"
for pth in (ROOT, ROOT / "TimeMixer-main", ROOT / "TimeMixer-main" / "models", DINO):
    p = str(pth)
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

from models.ts_mixer_backbone import TSMixerForDINO   # noqa: E402


# ── model / data helpers (shared with visualize_attn_map.py) ────────────────────
def load_config():
    spec = importlib.util.spec_from_file_location("dcfg", DINO / "config.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return dict(m.config)


def build_backbone(cfg, c_in, seq_len):
    return TSMixerForDINO(
        c_in=c_in, seq_len=seq_len,
        d_model=cfg.get("tsmixer_d_model", 128),
        e_layers=cfg.get("tsmixer_e_layers", 3),
        d_ff=cfg.get("tsmixer_d_ff", 256),
        dropout=cfg.get("dropout", 0.1),
        down_sampling_layers=cfg.get("tsmixer_down_sampling_layers", 3),
        down_sampling_window=cfg.get("tsmixer_down_sampling_window", 2),
        down_sampling_method=cfg.get("tsmixer_down_sampling_method", "avg"),
        decomp_method=cfg.get("tsmixer_decomp_method", "moving_avg"),
        moving_avg=cfg.get("tsmixer_moving_avg", 25),
        top_k=cfg.get("tsmixer_top_k", 5),
        use_norm=cfg.get("tsmixer_use_norm", 1),
        channel_independence=cfg.get("tsmixer_channel_independence", 1),
    ).eval()


def load_ckpt(backbone, ckpt_path, which):
    if not os.path.exists(ckpt_path):
        print(f"WARNING: checkpoint not found ({ckpt_path}) — using random init")
        return
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw = ck[which] if which in ck else ck.get("teacher", ck.get("model", ck))
    sd = {}
    for k, v in raw.items():
        k = k.replace("module.", "")
        if k.startswith("backbone."):
            k = k[len("backbone."):]
        sd[k] = v
    bsd = backbone.state_dict()
    filt = {k: v for k, v in sd.items() if k in bsd and bsd[k].shape == v.shape}
    miss, _ = backbone.load_state_dict(filt, strict=False)
    print(f"loaded {len(filt)}/{len(bsd)} backbone params  (missing {len(miss)})")


def get_attn(backbone, window):
    """window: [seq_len, C] float -> attn weights [C, T] (avg over heads)."""
    cap = {}
    def hook(_m, _i, out):
        cap["w"] = out[1].detach()
    h = backbone.global_attn.register_forward_hook(hook)
    x = torch.from_numpy(window).float().unsqueeze(0)   # [1, T, C]
    with torch.no_grad():
        backbone(x)
    h.remove()
    return cap["w"][:, 0, :].cpu().numpy()               # [C, S]


# ── similarity helpers ──────────────────────────────────────────────────────────
def _corr(u, v):
    u = np.asarray(u, float); v = np.asarray(v, float)
    if u.std() < 1e-9 or v.std() < 1e-9:
        return np.nan
    return float(np.corrcoef(u, v)[0, 1])


def _resample(ser, T):
    if len(ser) == T:
        return ser
    return ser[np.linspace(0, len(ser) - 1, T).astype(int)]


def _norm_range(v, lo=-2.0, hi=2.0):
    """Min-max scale v so its smallest value -> lo and largest -> hi."""
    v = np.asarray(v, float)
    vmin, vmax = v.min(), v.max()
    if vmax - vmin < 1e-12:
        return np.full_like(v, (lo + hi) / 2.0)
    return lo + (hi - lo) * (v - vmin) / (vmax - vmin)


def similarity(attn_c, ser):
    """attn_c: [T]; ser: series resampled to T. Returns dict of correlations."""
    sal = np.abs(ser - ser.mean())
    dx = np.abs(np.gradient(ser))
    return {
        "signal":   _corr(attn_c, ser),
        "saliency": _corr(attn_c, sal),
        "change":   _corr(attn_c, dx),
    }


# ── plotting ────────────────────────────────────────────────────────────────────
def plot_variable(ser, a, name, dataset, which, outpath, sims):
    """3-panel figure: colored series / overlay / attention strip."""
    T = len(a)
    tt = np.arange(T)
    ser_n = _norm_range(ser, -2.0, 2.0)   # min-max scaled to [-2, +2]
    a_n = _norm_range(a, -2.0, 2.0)        # attention on the same fixed band

    fig, (axA, axB, axC) = plt.subplots(
        3, 1, figsize=(11, 5.4), sharex=True,
        gridspec_kw={"height_ratios": [3, 2, 0.7]})

    # (A) series line colored by attention
    pts = np.array([tt, ser]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap="magma",
                        norm=plt.Normalize(a.min(), a.max()))
    lc.set_array(a[:-1]); lc.set_linewidth(2.4)
    axA.add_collection(lc)
    axA.set_xlim(0, T - 1); axA.set_ylim(ser.min(), ser.max())
    axA.set_ylabel("value")
    axA.set_title(f"{dataset} · {name} — series colored by attention ({which})", fontsize=11)
    cb = fig.colorbar(lc, ax=axA, pad=0.01); cb.set_label("attention", fontsize=8)

    # (B) normalized overlay + correlation
    axB.plot(tt, ser_n, color="0.45", lw=1.2, label="series (norm)")
    axB.plot(tt, a_n, color="#d62728", lw=1.4, label="attention (norm)")
    axB.fill_between(tt, -2.0, a_n, color="#d62728", alpha=0.15)
    axB.set_ylim(-2.2, 2.2)
    axB.set_yticks([-2, -1, 0, 1, 2])
    axB.set_ylabel("normalized [-2, 2]")
    txt = (f"corr(attn, signal) = {sims['signal']:+.2f}\n"
           f"corr(attn, |x-mean|) = {sims['saliency']:+.2f}\n"
           f"corr(attn, |dx|) = {sims['change']:+.2f}")
    axB.text(0.995, 0.95, txt, transform=axB.transAxes, ha="right", va="top",
             fontsize=8, family="monospace",
             bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))
    axB.legend(loc="upper left", fontsize=8, framealpha=0.85)

    # (C) attention heat strip
    im = axC.imshow(a[None, :], aspect="auto", cmap="magma",
                    extent=[0, T - 1, 0, 1], vmin=a.min(), vmax=a.max(),
                    interpolation="nearest")
    axC.set_yticks([]); axC.set_xlabel("timestep")
    fig.colorbar(im, ax=axC, pad=0.01).ax.tick_params(labelsize=7)

    fig.tight_layout()
    fig.savefig(outpath, dpi=140); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="etth1")
    p.add_argument("--ckpt", default="checkpoints_etth1_layers4_outdim1024_timemixer_grid_full_dino/checkpoint_best.pth")
    p.add_argument("--which", default="teacher", choices=["teacher", "student"])
    p.add_argument("--seq_len", type=int, default=None)
    p.add_argument("--window", type=int, default=0, help="window index to render")
    p.add_argument("--n_windows", type=int, default=50,
                   help="number of windows to average the similarity stats over")
    p.add_argument("--outdir", default=str(ROOT / "vis" / "attn_signal"))
    args = p.parse_args()

    cfg = load_config()
    seq_len = args.seq_len or cfg.get("num_patches", 21) * cfg.get("patch_len", 16)

    import pandas as pd
    from dataset_registry import get_dataset_info
    info = get_dataset_info(args.dataset)
    df = pd.read_csv(info["csv_path"])
    cols = [c for c in df.columns if c.lower() not in ("date", "timestamp")]
    X = df[cols].values.astype(np.float32)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    c_in = X.shape[1]

    backbone = build_backbone(cfg, c_in, seq_len)
    ckpt = args.ckpt if os.path.isabs(args.ckpt) else str(ROOT / args.ckpt)
    load_ckpt(backbone, ckpt, args.which)

    n_win = max(1, min(args.n_windows, X.shape[0] // seq_len))
    # ── accumulate similarity stats over many windows ──────────────────────────
    acc = {k: [[] for _ in range(c_in)] for k in ("signal", "saliency", "change")}
    for w in range(n_win):
        win = X[w * seq_len:(w + 1) * seq_len]
        if win.shape[0] < seq_len:
            break
        attn = get_attn(backbone, win)                 # [C, T]
        T = attn.shape[1]
        for c in range(c_in):
            sims = similarity(attn[c], _resample(win[:, c], T))
            for k in acc:
                if not np.isnan(sims[k]):
                    acc[k][c].append(sims[k])

    print(f"\n=== attention–signal similarity ({args.dataset}, {args.which}, "
          f"{n_win} windows) ===")
    print(f"{'variable':<16}{'corr(x)':>10}{'corr|x-mu|':>12}{'corr|dx|':>10}")
    for c in range(c_in):
        m = {k: (np.mean(acc[k][c]) if acc[k][c] else np.nan) for k in acc}
        print(f"{cols[c]:<16}{m['signal']:>+10.3f}{m['saliency']:>+12.3f}{m['change']:>+10.3f}")
    overall = {k: np.nanmean([np.mean(acc[k][c]) if acc[k][c] else np.nan
                              for c in range(c_in)]) for k in acc}
    print(f"{'MEAN':<16}{overall['signal']:>+10.3f}{overall['saliency']:>+12.3f}{overall['change']:>+10.3f}")

    # ── render the chosen window per variable ──────────────────────────────────
    os.makedirs(args.outdir, exist_ok=True)
    win = X[args.window * seq_len:(args.window + 1) * seq_len]
    attn = get_attn(backbone, win)
    T = attn.shape[1]
    for c in range(c_in):
        ser = _resample(win[:, c], T)
        sims = similarity(attn[c], ser)
        out = os.path.join(args.outdir, f"attn_signal_{args.dataset}_{args.which}_{cols[c]}_w{args.window}.png")
        plot_variable(ser, attn[c], cols[c], args.dataset, args.which, out, sims)
    print(f"\nsaved {c_in} figures -> {args.outdir}/")


if __name__ == "__main__":
    main()

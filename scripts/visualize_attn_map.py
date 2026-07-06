#!/usr/bin/env python
"""
visualize_attn_map.py — visualize TimeMixer's cross-attention global-pooling map.

TSMixerForDINO pools the T timestep tokens into a per-channel embedding via a
learned CLS query attending over all tokens (self.global_attn). This script loads
a pretrained checkpoint, runs an ETTh1 window, captures the attention weights
[C, T] with a forward hook, and plots:

  1) a heatmap  (channels × time)  of the attention weights, and
  2) per-channel  series + attention-over-time  overlays.

Usage
-----
  python scripts/visualize_attn_map.py \
      --dataset etth1 \
      --ckpt checkpoints_etth1_layers4_outdim1024_timemixer_grid_full_dino/checkpoint_best.pth
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

ROOT = Path(__file__).parent.parent.resolve()
DINO = ROOT / "tsdino_timemixer"
# TimeMixer layers first, then DINO LAST so tsdino_timemixer/models beats
# TimeMixer-main/models for `models.ts_mixer_backbone` (same trick as TSMixerClassification.py).
for pth in (ROOT, ROOT / "TimeMixer-main", ROOT / "TimeMixer-main" / "models", DINO):
    p = str(pth)
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

from models.ts_mixer_backbone import TSMixerForDINO   # noqa: E402


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
        # MultiheadAttention returns (attn_output, attn_weights[B*C, 1, S])
        cap["w"] = out[1].detach()
    h = backbone.global_attn.register_forward_hook(hook)
    x = torch.from_numpy(window).float().unsqueeze(0)   # [1, T, C]
    with torch.no_grad():
        backbone(x)
    h.remove()
    w = cap["w"]                       # [C, 1, S]  (B=1)
    return w[:, 0, :].cpu().numpy()    # [C, S]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="etth1")
    p.add_argument("--ckpt", default="checkpoints_etth1_layers4_outdim1024_timemixer_grid_full_dino/checkpoint_best.pth")
    p.add_argument("--which", default="teacher", choices=["teacher", "student"])
    p.add_argument("--seq_len", type=int, default=None)
    p.add_argument("--window", type=int, default=0, help="which non-overlapping window index")
    p.add_argument("--outdir", default=str(ROOT / "vis"))
    args = p.parse_args()

    cfg = load_config()
    seq_len = args.seq_len or cfg.get("num_patches", 21) * cfg.get("patch_len", 16)

    # ── load an ETTh1 window ───────────────────────────────────────────────
    import pandas as pd
    from dataset_registry import get_dataset_info
    info = get_dataset_info(args.dataset)
    df = pd.read_csv(info["csv_path"])
    cols = [c for c in df.columns if c.lower() not in ("date", "timestamp")]
    X = df[cols].values.astype(np.float32)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    s = args.window * seq_len
    win = X[s:s + seq_len]                     # [seq_len, C]
    c_in = win.shape[1]

    backbone = build_backbone(cfg, c_in, seq_len)
    ckpt = args.ckpt if os.path.isabs(args.ckpt) else str(ROOT / args.ckpt)
    load_ckpt(backbone, ckpt, args.which)

    attn = get_attn(backbone, win)            # [C, S]
    T = attn.shape[1]
    os.makedirs(args.outdir, exist_ok=True)
    tag = f"{args.dataset}_{args.which}_w{args.window}"

    # ── 1) heatmap: channels × time ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 3.2))
    im = ax.imshow(attn, aspect="auto", cmap="magma",
                   extent=[0, T, c_in - 0.5, -0.5], interpolation="nearest")
    ax.set_yticks(range(c_in)); ax.set_yticklabels(cols, fontsize=8)
    ax.set_xlabel("timestep"); ax.set_title(
        f"TimeMixer global_attn map — {args.dataset} ({args.which}, window {args.window})", fontsize=10)
    fig.colorbar(im, ax=ax, label="attention weight")
    fig.tight_layout()
    out1 = os.path.join(args.outdir, f"attn_map_{tag}_heatmap.png")
    fig.savefig(out1, dpi=130); plt.close(fig)
    print(f"saved: {out1}")

    # ── 2) per-channel: series + attention overlay ─────────────────────────
    ncol = 2
    nrow = int(np.ceil(c_in / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.5 * ncol, 2.0 * nrow), squeeze=False)
    tt = np.arange(T)
    for c in range(c_in):
        ax = axes[c // ncol][c % ncol]
        # resample series to attn length if downsampling changed T
        ser = win[:, c]
        if len(ser) != T:
            idx = np.linspace(0, len(ser) - 1, T).astype(int)
            ser = ser[idx]
        ax.plot(tt, ser, color="0.5", lw=0.9, label="series")
        ax2 = ax.twinx()
        a = attn[c]
        ax2.fill_between(tt, 0, a, color="#d62728", alpha=0.35, label="attention")
        ax2.plot(tt, a, color="#d62728", lw=1.0)
        ax2.set_ylim(0, max(a.max() * 1.1, 1e-6))
        ax.set_title(cols[c], fontsize=9); ax.tick_params(labelsize=7); ax2.tick_params(labelsize=7)
    for j in range(c_in, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(f"TimeMixer attention over time — {args.dataset} ({args.which})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out2 = os.path.join(args.outdir, f"attn_map_{tag}_overlay.png")
    fig.savefig(out2, dpi=130); plt.close(fig)
    print(f"saved: {out2}")


if __name__ == "__main__":
    main()

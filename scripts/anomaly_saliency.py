#!/usr/bin/env python
"""
anomaly_saliency.py — the "attention-like" anomaly figure that actually works.

The CLS attention in this model is near-uniform (DINO pooling collapses to mean),
so it does NOT localize anomalies. The model's real per-point saliency is the
RECONSTRUCTION ERROR: err[t, c] = (decoder(encoder(x))[t,c] - x[t,c])**2.

This script loads the fine-tuned detector (SMD_finetuned.pth from
anomaly_analysis.py — no retraining) and renders, for a contiguous test stretch:

  (1) anomaly_saliency_paint_<DS>.png
      the real series (one channel) drawn as a line COLORED by its own
      reconstruction error (viridis) — bright = the model finds this point hard
      to reconstruct = anomalous. Ground-truth spans shaded, threshold crossings
      marked. This is "paint the series in the saliency color + anomaly line".

  (2) anomaly_saliency_heatmap_<DS>.png
      per-(channel x time) reconstruction-error heatmap — the honest analog of an
      attention heatmap. It lights up in columns exactly at the anomalies.

Usage (server):
  python scripts/anomaly_saliency.py --dataset SMD \
      --model checkpoints/anomaly_finetuned/SMD_finetuned.pth --gpu 0
  # optional: --start 8000 --length 2000   (else auto-picks around biggest anomaly)
  #           --var 11                      (channel to paint; else most-variable)
"""
import argparse, os, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch

ROOT = Path(__file__).parent.parent.resolve()
DINO = ROOT / "tsdino_timemixer"
for p in (str(ROOT), str(ROOT / "shared"),
          str(ROOT / "TimeMixer-main" / "models"), str(ROOT / "TimeMixer-main")):
    if p not in sys.path:
        sys.path.insert(0, p)
sys.path.insert(0, str(DINO))
from models.ts_mixer_backbone import TSMixerForDINO  # noqa: E402


class _ReconDecoder(nn.Module):
    def __init__(self, d_model):
        super().__init__(); self.proj = nn.Linear(d_model, 1)
    def forward(self, z):
        return self.proj(z).squeeze(-1)


def load_config():
    import importlib.util
    spec = importlib.util.spec_from_file_location("dcfg", DINO / "config.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    cfg = dict(m.config)
    try:
        from data_paths import DATA_PATHS
        cfg = {**DATA_PATHS, **cfg}
    except Exception:
        pass
    return cfg


def longest_anomaly_stretch(labels, length):
    runs, s = [], None
    for i, v in enumerate(labels):
        if v == 1 and s is None: s = i
        if v == 0 and s is not None: runs.append((s, i)); s = None
    if s is not None: runs.append((s, len(labels)))
    if not runs:
        return 0, min(length, len(labels))
    a, b = max(runs, key=lambda r: r[1] - r[0])
    mid = (a + b) // 2
    start = max(0, mid - length // 2)
    return start, min(len(labels), start + length)


def shade(ax, lab, x0):
    s = None
    for t in range(len(lab) + 1):
        on = t < len(lab) and lab[t] == 1
        if on and s is None: s = t
        if (not on) and s is not None:
            ax.axvspan(x0 + s, x0 + t, color="red", alpha=0.13, lw=0); s = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True, help="SMD_finetuned.pth from anomaly_analysis.py")
    ap.add_argument("--patch_len", type=int, default=10)
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--length", type=int, default=2000)
    ap.add_argument("--var", type=int, default=None)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--outdir", default=str(ROOT / "vis" / "anomaly"))
    a = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_config()
    os.makedirs(a.outdir, exist_ok=True)

    ck = torch.load(a.model, map_location="cpu", weights_only=False)
    n_vars, win, thr = ck["n_vars"], ck["win"], ck["threshold"]
    d_model = cfg.get("tsmixer_d_model", 128)
    for k, v in (ck.get("cfg") or {}).items():
        if v is not None: cfg[k] = v

    bb = TSMixerForDINO(
        c_in=n_vars, seq_len=win, d_model=d_model, e_layers=cfg.get("tsmixer_e_layers", 3),
        d_ff=cfg.get("tsmixer_d_ff", 256), dropout=cfg.get("dropout", 0.1),
        down_sampling_layers=cfg.get("tsmixer_down_sampling_layers", 3),
        down_sampling_window=cfg.get("tsmixer_down_sampling_window", 2),
        down_sampling_method=cfg.get("tsmixer_down_sampling_method", "avg"),
        decomp_method=cfg.get("tsmixer_decomp_method", "moving_avg"),
        moving_avg=cfg.get("tsmixer_moving_avg", 25), top_k=cfg.get("tsmixer_top_k", 5),
        use_norm=cfg.get("tsmixer_use_norm", 1),
        channel_independence=cfg.get("tsmixer_channel_independence", 1),
    ).to(device).eval()
    bb.load_state_dict(ck["encoder"], strict=False)
    dec = _ReconDecoder(d_model).to(device).eval()
    dec.load_state_dict(ck["decoder"])
    print(f"loaded model: n_vars={n_vars} win={win} thr={thr:.5f} | metrics={ck.get('metrics')}")

    # ── attention hook (global_attn is used by the DINO forward, not forward_recon) ──
    _cap = {}
    def _hook(_m, _in, out):
        w = out[1] if isinstance(out, (tuple, list)) and len(out) > 1 else None
        if w is not None:
            _cap["a"] = w.detach()                       # [B*C, 1, T]
    hattn = bb.global_attn.register_forward_hook(_hook) if hasattr(bb, "global_attn") else None

    # ── contiguous test stream + per-(timestep, channel) reconstruction error ──
    from data_loaders.data_puller import AnomalyDataPuller
    ds = AnomalyDataPuller(cfg["anomaly_data_dir"], a.dataset, a.patch_len, win_size=win, which="test")
    X = np.asarray(ds.data, dtype=np.float32)          # [Ttotal, C]
    L = np.asarray(ds.labels).astype(int)              # [Ttotal]
    Tt = (len(X) // win) * win
    err = np.zeros((Tt, n_vars), dtype=np.float32)     # per (t, c)
    att = np.zeros((Tt, n_vars), dtype=np.float32)     # per (t, c) CLS attention
    with torch.no_grad():
        for i in range(0, Tt, win):
            w = torch.from_numpy(X[i:i+win]).float().unsqueeze(0).to(device)  # [1,win,C]
            z = bb.forward_recon(w)                     # [1,win,C,d_model]
            e = ((dec(z) - w) ** 2)[0].cpu().numpy()    # [win, C]
            err[i:i+win] = e
            if hattn is not None:
                _cap.clear()
                bb(w)                                   # trigger global_attn
                if "a" in _cap:
                    A = _cap["a"]                       # [C, 1, win] (B=1)
                    att[i:i+win] = A[:, 0, :].cpu().numpy().T   # [win, C]
    if hattn is not None: hattn.remove()
    X, L = X[:Tt], L[:Tt]
    score = err.mean(1)                                 # per-timestep anomaly score

    # ── stretch selection ─────────────────────────────────────────────────────
    if a.start is not None:
        x0, x1 = a.start, min(Tt, a.start + a.length)
    else:
        x0, x1 = longest_anomaly_stretch(L, a.length)
    sig, ec, lab = X[x0:x1], err[x0:x1], L[x0:x1]
    sc = score[x0:x1]
    tt = np.arange(x0, x1)
    anom = lab.astype(bool)
    c = a.var if a.var is not None else int(np.argmax(ec.mean(0)))  # channel with most error

    # ── (1) painted series ─────────────────────────────────────────────────────
    y = sig[:, c]
    col = ec[:, c]                                       # this channel's recon error
    vmax = np.percentile(col, 99.5) or 1.0
    pts = np.array([tt, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    fig, ax = plt.subplots(figsize=(12, 3.2))
    shade(ax, lab, x0)
    lc = LineCollection(segs, cmap="viridis", norm=plt.Normalize(0, vmax))
    lc.set_array(col[:-1]); lc.set_linewidth(1.6)
    ax.add_collection(lc)
    ax.set_xlim(tt[0], tt[-1]); ax.set_ylim(y.min() - .3, y.max() + .3)
    cb = fig.colorbar(lc, ax=ax, pad=0.01); cb.set_label("reconstruction error (saliency)")
    ax.set_ylabel(f"signal (var {c})"); ax.set_xlabel("timestep")
    ax.set_title(f"{a.dataset} — series painted by reconstruction-error saliency "
                 f"(F1={ck.get('metrics',{}).get('F1',float('nan')):.3f})", fontsize=11)
    ax.legend(handles=[Patch(facecolor="red", alpha=.13, label="ground-truth anomaly")],
              loc="upper right", fontsize=8)
    out1 = os.path.join(a.outdir, f"anomaly_saliency_paint_{a.dataset}.png")
    fig.savefig(out1, dpi=150, bbox_inches="tight"); plt.close(fig)

    # ── (2) channels x time error heatmap ─────────────────────────────────────
    fig, (axh, axr) = plt.subplots(2, 1, figsize=(12, 4.4), sharex=True,
                                   gridspec_kw={"height_ratios": [8, 1], "hspace": 0.05})
    H = ec.T                                             # [C, T]
    hv = np.percentile(H, 99.0) or 1.0
    im = axh.imshow(H, aspect="auto", cmap="magma", vmin=0, vmax=hv,
                    extent=[x0, x1, n_vars, 0])
    axh.set_ylabel("channel")
    cb = fig.colorbar(im, ax=axh, pad=0.01); cb.set_label("recon error")
    axh.set_title(f"{a.dataset} — per-channel reconstruction-error map "
                  "(bright columns = anomalies)", fontsize=11)
    axr.imshow(anom[None, :].astype(float), aspect="auto", cmap="Reds",
               vmin=0, vmax=1, extent=[x0, x1, 0, 1]); axr.set_yticks([])
    axr.set_ylabel("truth", rotation=0, ha="right", va="center", fontsize=9)
    axr.set_xlabel("timestep")
    out2 = os.path.join(a.outdir, f"anomaly_saliency_heatmap_{a.dataset}.png")
    fig.savefig(out2, dpi=150, bbox_inches="tight"); plt.close(fig)

    # ── (3) combined: top painted-saliency series, bottom attention heatmap ────
    ac = att[x0:x1]                                      # [T, C]
    fig, (axP, axA, axR) = plt.subplots(3, 1, figsize=(12, 5.6), sharex=True,
                                        gridspec_kw={"height_ratios": [4, 4, 0.5], "hspace": 0.08})
    # top: painted series (saliency)
    shade(axP, lab, x0)
    lc2 = LineCollection(segs, cmap="viridis", norm=plt.Normalize(0, vmax))
    lc2.set_array(col[:-1]); lc2.set_linewidth(1.6)
    axP.add_collection(lc2)
    axP.set_xlim(tt[0], tt[-1]); axP.set_ylim(y.min() - .3, y.max() + .3)
    fig.colorbar(lc2, ax=axP, pad=0.01).set_label("recon error")
    axP.set_ylabel(f"signal (var {c})")
    axP.set_title(f"{a.dataset} — top: series painted by reconstruction-error saliency   "
                  "bottom: CLS attention map (near-uniform)", fontsize=10)
    # bottom: attention heatmap channels x time
    av = np.percentile(ac.T, 99.0) if ac.any() else 1.0
    av0 = np.percentile(ac.T, 1.0) if ac.any() else 0.0
    imA = axA.imshow(ac.T, aspect="auto", cmap="magma", vmin=av0, vmax=max(av, av0 + 1e-9),
                     extent=[x0, x1, n_vars, 0])
    axA.set_ylabel("channel")
    fig.colorbar(imA, ax=axA, pad=0.01).set_label("attention")
    axR.imshow(anom[None, :].astype(float), aspect="auto", cmap="Reds",
               vmin=0, vmax=1, extent=[x0, x1, 0, 1]); axR.set_yticks([])
    axR.set_ylabel("truth", rotation=0, ha="right", va="center", fontsize=9)
    axR.set_xlabel("timestep")
    out3 = os.path.join(a.outdir, f"anomaly_saliency_combined_{a.dataset}.png")
    fig.savefig(out3, dpi=150, bbox_inches="tight"); plt.close(fig)

    print(f"saved: {out1}")
    print(f"saved: {out2}")
    print(f"saved: {out3}")
    print(f"stretch[{x0}:{x1}]  {int(anom.sum())} anomalous ts  painted var={c}  thr={thr:.4f}  "
          f"attn range [{att[x0:x1].min():.4f},{att[x0:x1].max():.4f}]")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
visualize_anomaly.py — anomaly-detection figure: attention + reconstruction score
over the real signal, with ground-truth anomaly spans.

Mirrors tsdino_timemixer/TSMixerAnomaly.py: loads a DINO teacher backbone from an
in-domain anomaly pretrain checkpoint, trains a quick linear reconstruction decoder
on the normal (train) stream (the decoder is NOT saved in the checkpoint, so we
retrain it — fast), then on the test stream picks the window with the most anomaly
labels and plots:

  (A) the signal, line colored by CLS attention, ground-truth anomaly spans shaded;
  (B) per-timestep reconstruction error (anomaly score) + threshold, same spans;
  (C) attention heat-strip, time-aligned.

Usage (on the server, where the data + checkpoints live):
  python scripts/visualize_anomaly.py --dataset SMD \
      --ckpt checkpoints/anomaly/<...>_anompre_SMD_cw100/checkpoint_best.pth
"""
import argparse, os, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

ROOT = Path(__file__).parent.parent.resolve()
DINO = ROOT / "tsdino_timemixer"
for p in (str(ROOT), str(ROOT / "shared"),
          str(ROOT / "TimeMixer-main" / "models"), str(ROOT / "TimeMixer-main")):
    if p not in sys.path:
        sys.path.insert(0, p)
# tsdino_timemixer inserted LAST -> first on sys.path, so `models.ts_mixer_backbone`
# resolves to tsdino_timemixer/models (not TimeMixer-main/models). Same trick as TSMixerAnomaly.py.
sys.path.insert(0, str(DINO))
from models.ts_mixer_backbone import TSMixerForDINO  # noqa: E402


class _ReconDecoder(nn.Module):          # matches TSMixerAnomaly._TSMixerReconDecoder
    def __init__(self, d_model):
        super().__init__(); self.proj = nn.Linear(d_model, 1)
    def forward(self, z):                # z: [B, T, C, d_model] -> [B, T, C]
        return self.proj(z).squeeze(-1)


def load_config():
    import importlib.util
    spec = importlib.util.spec_from_file_location("dcfg", DINO / "config.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    cfg = dict(m.config)
    try:  # anomaly_data_dir etc. live in data_paths.DATA_PATHS, not config.py
        from data_paths import DATA_PATHS
        cfg = {**DATA_PATHS, **cfg}
    except Exception:
        pass
    return cfg


def build_backbone(cfg, c_in, seq_len, device):
    bb = TSMixerForDINO(
        c_in=c_in, seq_len=seq_len,
        d_model=cfg.get("tsmixer_d_model", 128), e_layers=cfg.get("tsmixer_e_layers", 3),
        d_ff=cfg.get("tsmixer_d_ff", 256), dropout=cfg.get("dropout", 0.1),
        down_sampling_layers=cfg.get("tsmixer_down_sampling_layers", 3),
        down_sampling_window=cfg.get("tsmixer_down_sampling_window", 2),
        down_sampling_method=cfg.get("tsmixer_down_sampling_method", "avg"),
        decomp_method=cfg.get("tsmixer_decomp_method", "moving_avg"),
        moving_avg=cfg.get("tsmixer_moving_avg", 25), top_k=cfg.get("tsmixer_top_k", 5),
        use_norm=cfg.get("tsmixer_use_norm", 1),
        channel_independence=cfg.get("tsmixer_channel_independence", 1),
    ).to(device)
    return bb


def load_teacher(bb, ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw = ck["teacher"] if "teacher" in ck else ck.get("model", ck)
    sd = {}
    for k, v in raw.items():
        k = k.replace("module.", "")
        if k.startswith("backbone."):
            k = k[len("backbone."):]
        sd[k] = v
    miss, unexp = bb.load_state_dict(sd, strict=False)
    print(f"loaded {len(sd)} weights | missing {len(miss)} | unexpected {len(unexp)}")


def get_attn(bb, x):
    """x: [1, T, C] -> attention [C, T] (head-averaged CLS attention)."""
    cap = {}
    h = bb.global_attn.register_forward_hook(lambda m, i, o: cap.__setitem__("w", o[1].detach()))
    with torch.no_grad():
        bb(x)                              # DINO forward path (uses global_attn)
    h.remove()
    return cap["w"][:, 0, :].cpu().numpy()  # [C, T]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="SMD | MSL | SMAP | SWaT | PSM")
    ap.add_argument("--ckpt", required=True, help="anomaly pretrain checkpoint_best.pth")
    ap.add_argument("--patch_len", type=int, default=10)
    ap.add_argument("--num_patches", type=int, default=10)
    ap.add_argument("--decoder_epochs", type=int, default=5)
    ap.add_argument("--var", type=int, default=None, help="channel to plot (default: highest-error one)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--outdir", default=str(ROOT / "vis" / "anomaly"))
    a = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_config()
    win = a.num_patches * a.patch_len          # 100

    from data_loaders.data_puller import AnomalyDataPuller
    anom_dir = cfg["anomaly_data_dir"]
    ds_tr = AnomalyDataPuller(anom_dir, a.dataset, a.patch_len, win_size=win, which="train")
    ds_te = AnomalyDataPuller(anom_dir, a.dataset, a.patch_len, win_size=win, which="test")
    n_vars = ds_tr.n_vars
    tr = torch.utils.data.DataLoader(ds_tr, batch_size=64, shuffle=False)
    te = torch.utils.data.DataLoader(ds_te, batch_size=64, shuffle=False)
    print(f"{a.dataset}: n_vars={n_vars}  win={win}")

    bb = build_backbone(cfg, n_vars, win, device)
    load_teacher(bb, a.ckpt)
    d_model = cfg.get("tsmixer_d_model", 128)
    dec = _ReconDecoder(d_model).to(device)

    def encode(patches):                       # [B,P,PL,C] or [B,T,C] -> [B,T,C,d_model]
        if patches.dim() == 3: patches = patches.unsqueeze(-1)
        B, P, PL, C = patches.shape
        return bb.forward_recon(patches.reshape(B, P * PL, C).to(device))

    # ── train a quick linear decoder on the normal (train) stream ─────────────
    opt = torch.optim.Adam(dec.parameters(), lr=1e-3)
    for ep in range(a.decoder_epochs):
        dec.train(); losses = []
        for batch in tr:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            if x.dim() == 3: x = x.unsqueeze(-1)
            raw = x.to(device); B, P, PL, C = raw.shape
            recon = dec(encode(raw)); target = raw.reshape(B, P * PL, C)
            loss = F.mse_loss(recon, target)
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
        print(f"  decoder epoch {ep+1}/{a.decoder_epochs}  loss={np.mean(losses):.5f}")

    # ── test: per-window score, labels; keep the window with the most anomalies ─
    dec.eval()
    train_energy = []
    with torch.no_grad():
        for batch in tr:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            if x.dim() == 3: x = x.unsqueeze(-1)
            raw = x.to(device); B, P, PL, C = raw.shape
            err = ((dec(encode(raw)) - raw.reshape(B, P*PL, C))**2).mean(-1)   # [B,T]
            train_energy.append(err.cpu().numpy())
    train_energy = np.concatenate(train_energy).reshape(-1)

    best = None  # (n_anom, window_np, err_np, attn, labels)
    with torch.no_grad():
        for patches, labels in te:
            if patches.dim() == 3: patches = patches.unsqueeze(-1)
            raw = patches.to(device); B, P, PL, C = raw.shape
            xin = raw.reshape(B, P*PL, C)
            err = ((dec(encode(raw)) - xin)**2).mean(-1).cpu().numpy()          # [B,T]
            lab = labels.numpy()
            for b in range(B):
                n = int(lab[b].sum())
                if n > 0 and (best is None or n > best[0]):
                    attn = get_attn(bb, xin[b:b+1])                             # [C,T]
                    best = (n, xin[b].cpu().numpy(), err[b], attn, lab[b].astype(int))
    if best is None:
        print("No test window contained an anomaly — nothing to plot."); return
    n_anom, sig, err, attn, lab = best
    thr = np.percentile(np.concatenate([train_energy, err]), 95)               # display threshold
    T = sig.shape[0]
    # default channel: the one whose reconstruction error rises most inside the anomaly span
    if a.var is not None:
        c = a.var
    else:
        anom = lab.astype(bool)
        contrib = (sig[anom].std(0) if anom.any() else np.abs(sig).max(0))
        c = int(np.argmax(contrib))
    ser = sig[:, c]; a_c = attn[c]
    tt = np.arange(T)

    # ── plot ──────────────────────────────────────────────────────────────────
    os.makedirs(a.outdir, exist_ok=True)
    fig = plt.figure(figsize=(11, 5.6))
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 0.018], height_ratios=[3, 2, 0.7],
                          hspace=0.15, wspace=0.02)
    axA = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[1, 0], sharex=axA)
    axC = fig.add_subplot(gs[2, 0], sharex=axA)
    caxA = fig.add_subplot(gs[0, 1]); caxC = fig.add_subplot(gs[2, 1])

    def shade(ax):  # ground-truth anomaly spans
        s = None
        for t in range(T + 1):
            on = t < T and lab[t] == 1
            if on and s is None: s = t
            if (not on) and s is not None:
                ax.axvspan(s - 0.5, t - 0.5, color="red", alpha=0.15, lw=0); s = None

    # (A) signal colored by attention + anomaly spans
    pts = np.array([tt, ser]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap="magma", norm=plt.Normalize(a_c.min(), a_c.max()))
    lc.set_array(a_c[:-1]); lc.set_linewidth(2.2); axA.add_collection(lc)
    axA.set_xlim(0, T-1); axA.set_ylim(ser.min(), ser.max()); shade(axA)
    axA.set_ylabel("value"); axA.set_title(
        f"{a.dataset} · var {c} — signal colored by attention; red = ground-truth anomaly", fontsize=10)
    fig.colorbar(lc, cax=caxA).set_label("attention", fontsize=8)
    plt.setp(axA.get_xticklabels(), visible=False)

    # (B) reconstruction error (anomaly score) + threshold + spans
    axB.plot(tt, err, color="#1f77b4", lw=1.4, label="recon error (score)")
    axB.axhline(thr, color="k", ls="--", lw=0.8, label="threshold (95%)")
    axB.fill_between(tt, 0, err, color="#1f77b4", alpha=0.12); shade(axB)
    axB.set_ylabel("anomaly score"); axB.legend(loc="upper left", fontsize=8, framealpha=.85)
    plt.setp(axB.get_xticklabels(), visible=False)

    # (C) attention strip
    im = axC.imshow(a_c[None, :], aspect="auto", cmap="magma",
                    extent=[0, T-1, 0, 1], vmin=a_c.min(), vmax=a_c.max())
    axC.set_yticks([]); axC.set_xlabel("timestep"); fig.colorbar(im, cax=caxC)

    out = os.path.join(a.outdir, f"anomaly_{a.dataset}_var{c}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"saved: {out}  ({n_anom} anomalous timesteps in the plotted window)")


if __name__ == "__main__":
    main()

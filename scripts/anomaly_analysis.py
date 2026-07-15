#!/usr/bin/env python
"""
anomaly_analysis.py — fine-tune the WINO-TS anomaly detector on one dataset, SAVE
the model (encoder + reconstruction head), dump attention/score/label metrics for
custom plotting, and render analysis figures.

Pipeline (mirrors tsdino_timemixer/TSMixerAnomaly.py):
  1. load DINO teacher backbone from an in-domain anomaly pretrain checkpoint
  2. fine-tune encoder + linear reconstruction decoder on the normal (train) stream
  3. SAVE {encoder, decoder, cfg, threshold, metrics} to a checkpoint
  4. run the test stream: per-timestep reconstruction score, CLS attention, labels
  5. threshold (combined train+test) + point-adjustment -> P/R/F1
  6. DUMP metrics (.npz): score, labels, per-window attention (anomaly windows)
  7. FIGURES: paint (signal colored by attention + anomaly marks), full 3-panel
     (signal / score / attention), attention heatmap, anomaly-vs-normal histograms

Usage (server):
  python scripts/anomaly_analysis.py --dataset SMD \
      --ckpt anomaly/checkpoints_layers4_outdim1024_tsmixer_seed42_anompre_SMD_cw100/checkpoint_best.pth \
      --epochs 10 --gpu 0
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
sys.path.insert(0, str(DINO))          # tsdino_timemixer/models must win
from models.ts_mixer_backbone import TSMixerForDINO  # noqa: E402


class _ReconDecoder(nn.Module):
    def __init__(self, d_model):
        super().__init__(); self.proj = nn.Linear(d_model, 1)
    def forward(self, z):               # [B,T,C,d_model] -> [B,T,C]
        return self.proj(z).squeeze(-1)


def _adjustment(gt, pred):
    st = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not st:
            st = True
            for j in range(i, -1, -1):
                if gt[j] == 0: break
                if pred[j] == 0: pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0: break
                if pred[j] == 0: pred[j] = 1
        elif gt[i] == 0:
            st = False
        if st: pred[i] = 1
    return gt, pred


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


def build_backbone(cfg, c_in, seq_len, device):
    return TSMixerForDINO(
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
    cap = {}
    h = bb.global_attn.register_forward_hook(lambda m, i, o: cap.__setitem__("w", o[1].detach()))
    with torch.no_grad():
        bb(x)
    h.remove()
    return cap["w"][:, 0, :].cpu().numpy()     # [C, T]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ckpt", required=True, help="anomaly pretrain teacher checkpoint")
    ap.add_argument("--patch_len", type=int, default=10)
    ap.add_argument("--num_patches", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=10, help="anomaly fine-tune epochs")
    ap.add_argument("--anomaly_ratio", type=float, default=0.5)
    ap.add_argument("--var", type=int, default=None)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--outdir", default=str(ROOT / "vis" / "anomaly"))
    ap.add_argument("--save_dir", default=str(ROOT / "checkpoints" / "anomaly_finetuned"))
    a = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_config()
    win = a.num_patches * a.patch_len
    os.makedirs(a.outdir, exist_ok=True); os.makedirs(a.save_dir, exist_ok=True)

    from data_loaders.data_puller import AnomalyDataPuller
    anom_dir = cfg["anomaly_data_dir"]
    ds_tr = AnomalyDataPuller(anom_dir, a.dataset, a.patch_len, win_size=win, which="train")
    ds_te = AnomalyDataPuller(anom_dir, a.dataset, a.patch_len, win_size=win, which="test")
    n_vars = ds_tr.n_vars
    tr = torch.utils.data.DataLoader(ds_tr, batch_size=64, shuffle=False)
    te = torch.utils.data.DataLoader(ds_te, batch_size=64, shuffle=False)
    print(f"{a.dataset}: n_vars={n_vars}  win={win}  train_win={len(ds_tr)}  test_win={len(ds_te)}")

    bb = build_backbone(cfg, n_vars, win, device)
    load_teacher(bb, a.ckpt)
    d_model = cfg.get("tsmixer_d_model", 128)
    dec = _ReconDecoder(d_model).to(device)

    def encode(raw):
        if raw.dim() == 3: raw = raw.unsqueeze(-1)
        B, P, PL, C = raw.shape
        return bb.forward_recon(raw.reshape(B, P * PL, C).to(device))

    # ── 2. fine-tune encoder + decoder on the normal stream ───────────────────
    for p in bb.parameters(): p.requires_grad = True
    opt = torch.optim.Adam([{"params": dec.parameters(), "lr": 1e-3},
                            {"params": bb.parameters(), "lr": 1e-4}])
    for ep in range(a.epochs):
        bb.train(); dec.train(); losses = []
        for batch in tr:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            if x.dim() == 3: x = x.unsqueeze(-1)
            raw = x.to(device); B, P, PL, C = raw.shape
            loss = F.mse_loss(dec(encode(raw)), raw.reshape(B, P*PL, C))
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
        print(f"  finetune epoch {ep+1}/{a.epochs}  loss={np.mean(losses):.5f}")

    bb.eval(); dec.eval()

    # ── 3. save model (encoder + head) ────────────────────────────────────────
    model_path = os.path.join(a.save_dir, f"{a.dataset}_finetuned.pth")

    # ── 4. test/train energy + attention + labels ─────────────────────────────
    def energy(loader, want_attn=False):
        E, L, A = [], [], []
        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    x, lab = batch[0], batch[1].numpy()
                else:
                    x, lab = batch, None
                if x.dim() == 3: x = x.unsqueeze(-1)
                raw = x.to(device); B, P, PL, C = raw.shape
                xin = raw.reshape(B, P*PL, C)
                err = ((dec(encode(raw)) - xin) ** 2).mean(-1).cpu().numpy()   # [B,T]
                E.append(err)
                if lab is not None: L.append(lab)
                if want_attn:
                    for b in range(B):
                        A.append(get_attn(bb, xin[b:b+1]))                     # [C,T]
        return (np.concatenate(E), (np.concatenate(L) if L else None),
                (np.stack(A) if A else None))

    tr_E, _, _ = energy(tr)
    te_E, te_L, te_A = energy(te, want_attn=True)         # te_A: [Nwin, C, T]
    thr = np.percentile(np.concatenate([tr_E.reshape(-1), te_E.reshape(-1)]),
                        100 - a.anomaly_ratio)

    # ── 5. metrics (point-adjusted) ───────────────────────────────────────────
    score_flat = te_E.reshape(-1); gt = te_L.reshape(-1).astype(int)
    pred = (score_flat > thr).astype(int)
    gt, pred = _adjustment(gt.copy(), pred.copy())
    from sklearn.metrics import precision_recall_fscore_support
    P_, R_, F1_, _ = precision_recall_fscore_support(gt, pred, average="binary", zero_division=0)
    print(f"\n[{a.dataset}]  P={P_:.4f}  R={R_:.4f}  F1={F1_:.4f}  thr={thr:.5f}")

    torch.save({"encoder": bb.state_dict(), "decoder": dec.state_dict(),
                "cfg": {k: cfg.get(k) for k in ("tsmixer_d_model","tsmixer_e_layers","tsmixer_d_ff",
                        "tsmixer_down_sampling_layers","tsmixer_down_sampling_window")},
                "n_vars": n_vars, "win": win, "threshold": float(thr),
                "metrics": {"P": float(P_), "R": float(R_), "F1": float(F1_)}}, model_path)
    print(f"saved model (+head): {model_path}")

    # ── 6. dump metrics for custom plotting ───────────────────────────────────
    attn_mean = te_A.mean(1)                              # [Nwin, T] channel-avg attention
    anom_win = np.where(te_L.sum(1) > 0)[0]
    dump = os.path.join(a.outdir, f"anomaly_metrics_{a.dataset}.npz")
    np.savez_compressed(dump, score=te_E, labels=te_L, attn_mean=attn_mean,
                        attn_anom_windows=te_A[anom_win], anom_window_idx=anom_win,
                        threshold=thr, per_ts_score=score_flat, per_ts_label=te_L.reshape(-1))
    print(f"saved metrics dump: {dump}  (attn per window [{te_A.shape}])")

    # ── 7. figures on the most-anomalous window ───────────────────────────────
    w = int(np.argmax(te_L.sum(1)))
    sig_full = None  # need the raw signal for that window; re-fetch it
    # re-run just that window to get its raw signal
    idx = 0
    with torch.no_grad():
        for batch in te:
            x, lab = batch[0], batch[1]
            B = x.shape[0]
            if idx <= w < idx + B:
                b = w - idx
                xx = x[b].unsqueeze(0)
                if xx.dim() == 3: xx = xx.unsqueeze(-1)
                raw = xx.to(device); _, P, PL, C = raw.shape
                sig_full = raw.reshape(1, P*PL, C)[0].cpu().numpy()   # [T,C]
                break
            idx += B
    attn = te_A[w]; lab = te_L[w].astype(int); score = te_E[w]; T = win
    anom = lab.astype(bool)
    c = a.var if a.var is not None else int(np.argmax(sig_full[anom].std(0) if anom.any() else np.abs(sig_full).max(0)))
    ser = sig_full[:, c]; a_c = attn[c]; tt = np.arange(T)

    # (a) paint: signal colored by attention, anomalies circled
    fig = plt.figure(figsize=(12, 3.4)); gs = fig.add_gridspec(2, 2, width_ratios=[1,0.015],
        height_ratios=[6,0.5], hspace=0.05, wspace=0.02)
    ax = fig.add_subplot(gs[0,0]); cax = fig.add_subplot(gs[0,1]); axr = fig.add_subplot(gs[1,0], sharex=ax)
    pts = np.array([tt, ser]).T.reshape(-1,1,2); segs = np.concatenate([pts[:-1],pts[1:]],axis=1)
    lc = LineCollection(segs, cmap="magma", norm=plt.Normalize(a_c.min(), a_c.max()))
    lc.set_array(a_c[:-1]); lc.set_linewidth(2.4); ax.add_collection(lc)
    ax.set_xlim(0,T-1); ax.set_ylim(ser.min(),ser.max())
    ax.scatter(tt[anom], ser[anom], s=26, facecolors="none", edgecolors="red", linewidths=1.1, zorder=5, label="anomaly")
    m_an = a_c[anom].mean() if anom.any() else np.nan; m_no = a_c[~anom].mean()
    ax.set_ylabel("value"); ax.legend(loc="upper left", fontsize=8)
    ax.set_title(f"{a.dataset} · var {c} — attention on signal (anomaly {m_an:.2e} vs normal {m_no:.2e})", fontsize=10)
    fig.colorbar(lc, cax=cax).set_label("attention", fontsize=8); plt.setp(ax.get_xticklabels(), visible=False)
    axr.imshow(anom[None,:].astype(float), aspect="auto", cmap="Reds", extent=[0,T-1,0,1], vmin=0, vmax=1)
    axr.set_yticks([]); axr.set_xlabel("timestep")
    fig.savefig(os.path.join(a.outdir, f"anomaly_paint_{a.dataset}_var{c}.png"), dpi=140, bbox_inches="tight"); plt.close(fig)

    # (b) attention heatmap channels x time + labels
    fig, (axh, axl) = plt.subplots(2, 1, figsize=(12, 4.5), sharex=True,
        gridspec_kw={"height_ratios":[8,0.5]})
    im = axh.imshow(attn, aspect="auto", cmap="magma", extent=[0,T-1,n_vars-0.5,-0.5])
    axh.set_ylabel("channel"); axh.set_title(f"{a.dataset} — CLS attention heatmap (channels × time)", fontsize=10)
    fig.colorbar(im, ax=axh, label="attention")
    axl.imshow(anom[None,:].astype(float), aspect="auto", cmap="Reds", extent=[0,T-1,0,1], vmin=0, vmax=1)
    axl.set_yticks([]); axl.set_xlabel("timestep")
    fig.savefig(os.path.join(a.outdir, f"anomaly_heatmap_{a.dataset}.png"), dpi=140, bbox_inches="tight"); plt.close(fig)

    # (c) histogram: attention & score at anomaly vs normal (whole test set)
    aa = attn_mean.reshape(-1); ll = te_L.reshape(-1).astype(bool)
    fig, (h1, h2) = plt.subplots(1, 2, figsize=(11, 3.4))
    h1.hist(aa[~ll], bins=60, density=True, alpha=.6, label="normal", color="0.5")
    h1.hist(aa[ll],  bins=60, density=True, alpha=.6, label="anomaly", color="red")
    h1.set_title("attention (channel-avg) distribution"); h1.set_xlabel("attention"); h1.legend(fontsize=8)
    h2.hist(score_flat[~ll], bins=60, density=True, alpha=.6, label="normal", color="0.5")
    h2.hist(score_flat[ll],  bins=60, density=True, alpha=.6, label="anomaly", color="red")
    h2.axvline(thr, color="k", ls="--", lw=.8, label="threshold"); h2.set_yscale("log")
    h2.set_title("recon-error (score) distribution"); h2.set_xlabel("score"); h2.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(a.outdir, f"anomaly_hist_{a.dataset}.png"), dpi=140); plt.close(fig)

    print(f"\nfigures -> {a.outdir}/  (paint, heatmap, hist)")


if __name__ == "__main__":
    main()

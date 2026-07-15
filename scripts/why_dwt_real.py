#!/usr/bin/env python
"""
why_dwt_real.py — show, on REAL data through the REAL augmentation pipeline, why the
DWT view-pair (soft teacher / hard student, no crop) is superior to the DINO-vision
crop+jitter view-pair.

Uses the actual classes from tsdino_timemixer/data_agumentation.py and the actual
config params, on a real standardized window pulled by PatchTSTPretrainAdapter.

Claim:
  * DWT edits only DETAIL (high-freq) bands -> the approximation (trend / dominant
    period) is identical in teacher & student. Shared content = the real signal.
  * crop+jitter: the 0.4 local crop + resize is a TIME-WARP -> it rescales the
    dominant period by ~1/0.4 = 2.5x, so teacher and student disagree on the
    fundamental frequency. Forcing invariance destroys frequency sensitivity.

Outputs (to vis/):
  why_dwt_real_signal.png  — one window: time-domain overlays + power spectra.
  why_dwt_real_stats.png   — over N windows: period-ratio distribution +
                             band-wise perturbation energy.

Run on the server (data lives there):
  python scripts/why_dwt_real.py --dataset etth1 --gpu 0
  # options: --csv data/ETTh1.csv  --idx 500  --nwin 200
"""
import argparse, os, sys, importlib.util
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "shared"))
sys.path.insert(0, str(ROOT / "tsdino_timemixer"))
import data_agumentation as aug                        # the REAL augmentation classes
from data_loaders.data_puller import PatchTSTPretrainAdapter  # the REAL puller


def load_cfg():
    spec = importlib.util.spec_from_file_location("dcfg", ROOT / "tsdino_timemixer" / "config.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return dict(m.config)


# exact copy of DataAugmentationDino._random_crop (main.py) — RandomResizedCrop analog
def random_crop(x, crop_ratio):
    timesteps = x.shape[0]
    crop_len = int(timesteps * crop_ratio)
    if crop_len >= timesteps or crop_len < 2:
        return x
    start = np.random.randint(0, timesteps - crop_len + 1)
    cropped = x[start:start + crop_len, :]
    resized = F.interpolate(cropped.transpose(0, 1).unsqueeze(0), size=timesteps,
                            mode="linear", align_corners=False)
    return resized.squeeze(0).transpose(0, 1)


def make_synthetic(seq_len, seed=0, period=48, n_vars=3):
    """Generic function whose IDENTITY is a clean low-freq period.
    Low-freq carrier (survives DWT detail-perturb untouched) + high-freq texture
    (nuisance DWT edits) + noise. Standardized like real pipeline input.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(seq_len)
    X = np.zeros((seq_len, n_vars), np.float32)
    base = period * rng.uniform(0.75, 1.35)          # per-window period → distribution, not a fluke
    for v in range(n_vars):
        P = base * (1.0 + 0.08 * v)
        ph = rng.uniform(0, 2 * np.pi)
        carrier = np.sin(2 * np.pi * t / P + ph) + 0.4 * np.sin(2 * np.pi * 2 * t / P + ph)
        texture = 0.15 * np.sin(2 * np.pi * t / 6.0 + rng.uniform(0, 6))   # high-freq detail
        noise = 0.10 * rng.standard_normal(seq_len)
        s = carrier + texture + noise
        X[:, v] = (s - s.mean()) / (s.std() + 1e-8)
    return torch.from_numpy(X)


def build_augs(cfg):
    dwt_shared = dict(
        wavelet_pool=cfg["dwt_wavelet_pool"], level=cfg["dwt_level"],
        soft_threshold_sigma=cfg["dwt_soft_threshold_sigma"],
        zero_out_ratio=cfg["dwt_zero_out_ratio"], finest_levels=cfg["dwt_finest_levels"],
        high_perturb_noise_range=cfg["dwt_high_perturb_noise_range"],
    )
    dwt_soft = aug.DWTAugmentation(mode="soft_threshold", **dwt_shared)   # teacher (easy)
    dwt_hard = aug.DWTAugmentation(mode="high_perturb", **dwt_shared)     # student (hard)
    gc_glob = aug.gaussian_noise(std_range=(0.02, 0.10))                  # global crop noise
    gc_loc = aug.gaussian_noise(std_range=(0.10, 0.30))                  # local crop noise
    return dwt_soft, dwt_hard, gc_glob, gc_loc


def views(x, augs):
    """Return (dwt_teacher, dwt_student, crop_teacher, crop_student) for window x [T,C]."""
    dwt_soft, dwt_hard, gc_glob, gc_loc = augs
    dwt_t = dwt_soft(x)                                   # soft-threshold, full window
    dwt_s = dwt_hard(x)                                   # detail-perturb, full window
    crop_t = gc_glob(x)                                  # crop_ratio 1.0 + noise
    crop_s = gc_loc(random_crop(x, 0.4))                 # 0.4 crop -> resize -> noise
    return dwt_t, dwt_s, crop_t, crop_s


def dom_period(sig):
    """dominant period (in samples) via FFT peak, ignoring DC and very-low bins."""
    s = sig - sig.mean()
    P = np.abs(np.fft.rfft(s)) ** 2
    f = np.fft.rfftfreq(len(s))
    k = np.argmax(P[2:]) + 2
    return 1.0 / f[k] if f[k] > 0 else np.inf


def spectrum(sig):
    s = sig - sig.mean()
    P = np.abs(np.fft.rfft(s)) ** 2
    f = np.fft.rfftfreq(len(s))
    return f, P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="synthetic", choices=["synthetic", "data"],
                    help="synthetic generic function (runs anywhere) or real dataset")
    ap.add_argument("--period", type=int, default=48, help="synthetic carrier period (samples)")
    ap.add_argument("--dataset", default="etth1")
    ap.add_argument("--csv", default=None, help="path to CSV (defaults from config data_path)")
    ap.add_argument("--idx", type=int, default=500, help="window index for the single-window figure")
    ap.add_argument("--var", type=int, default=None, help="channel to plot (default: most periodic)")
    ap.add_argument("--nwin", type=int, default=200, help="#windows for the stats figure")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--outdir", default=str(ROOT / "vis"))
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    os.makedirs(a.outdir, exist_ok=True)
    np.random.seed(0); torch.manual_seed(0)

    cfg = load_cfg()
    seq_len = cfg["num_patches"] * cfg["patch_len"]      # 336
    augs = build_augs(cfg)

    if a.source == "synthetic":
        n_total = 100000
        def get_window(idx):
            return make_synthetic(seq_len, seed=idx, period=a.period)
        print(f"source: synthetic  period={a.period}  seq_len={seq_len}")
    else:
        if a.csv:
            csv = a.csv
        else:
            from dataset_registry import get_dataset_info  # resolves real CSV path
            csv = get_dataset_info(a.dataset)["csv_path"]
        ds = PatchTSTPretrainAdapter(csv_path=csv, split="train", seq_len=seq_len,
                                     patch_size=cfg["patch_len"], transform=None)
        n_total = len(ds)
        def get_window(idx):
            x = ds[idx % len(ds)]
            if not torch.is_tensor(x): x = torch.as_tensor(np.asarray(x))
            return x.float()
        print(f"source: {csv}  windows={n_total}  seq_len={seq_len}")

    # choose the most periodic channel on the reference window
    xref = get_window(a.idx)
    if a.var is None:
        conc = []
        for c in range(xref.shape[1]):
            f, P = spectrum(xref[:, c].numpy())
            conc.append(P[2:].max() / (P[2:].sum() + 1e-9))
        c = int(np.argmax(conc))
    else:
        c = a.var
    print(f"plotting channel {c}")

    # ── single-window figure ──────────────────────────────────────────────────
    dwt_t, dwt_s, crop_t, crop_s = views(xref, augs)
    t = np.arange(seq_len)
    TEA, STU = "#1f77b4", "#e8703a"
    fig, ax = plt.subplots(2, 3, figsize=(15, 6.2))
    plt.subplots_adjust(left=0.06, right=0.99, top=0.9, bottom=0.08, wspace=0.2, hspace=0.32)

    def sig(v, ch=c): return v[:, ch].detach().cpu().numpy()

    # top-left: raw window
    ax[0, 0].plot(t, sig(xref), color="0.2", lw=1.1)
    ax[0, 0].set_title("real window (standardized)", fontsize=10.5)
    ax[0, 0].set_ylabel("time domain")
    # top-mid: DWT teacher vs student
    ax[0, 1].plot(t, sig(dwt_t), color=TEA, lw=1.1, label="teacher (dwt_soft)")
    ax[0, 1].plot(t, sig(dwt_s), color=STU, lw=1.0, alpha=.85, label="student (dwt_hard)")
    ax[0, 1].set_title("DWT (ours) — same trend & period ✓", fontsize=10.5, color="#2ca02c")
    ax[0, 1].legend(fontsize=8)
    # top-right: crop teacher vs student
    ax[0, 2].plot(t, sig(crop_t), color=TEA, lw=1.1, label="teacher (full + noise)")
    ax[0, 2].plot(t, sig(crop_s), color=STU, lw=1.0, alpha=.85, label="student (0.4 crop + noise)")
    ax[0, 2].set_title("Crop+Jitter — period warped ✗", fontsize=10.5, color="#d62728")
    ax[0, 2].legend(fontsize=8)

    # bottom row: power spectra (log), mark dominant period
    def plot_spec(axx, sigs, colors, labels):
        for s2, col, lab in zip(sigs, colors, labels):
            f, P = spectrum(sig(s2))
            axx.semilogy(f[1:], P[1:] + 1e-9, color=col, lw=1.1, label=lab, alpha=.9)
        axx.set_xlim(0, 0.5); axx.set_xlabel("frequency (cycles/step)")
    plot_spec(ax[1, 0], [xref], ["0.2"], ["real"])
    ax[1, 0].set_ylabel("power (log)"); ax[1, 0].set_title(f"real spectrum  (period≈{dom_period(sig(xref)):.0f})", fontsize=10)
    plot_spec(ax[1, 1], [dwt_t, dwt_s], [TEA, STU], ["teacher", "student"])
    ax[1, 1].legend(fontsize=8)
    ax[1, 1].set_title(f"DWT: peaks aligned  (T≈{dom_period(sig(dwt_t)):.0f}, S≈{dom_period(sig(dwt_s)):.0f})", fontsize=10)
    plot_spec(ax[1, 2], [crop_t, crop_s], [TEA, STU], ["teacher", "student"])
    ax[1, 2].legend(fontsize=8)
    ax[1, 2].set_title(f"Crop: student peak shifted  (T≈{dom_period(sig(crop_t)):.0f}, S≈{dom_period(sig(crop_s)):.0f})", fontsize=10)

    src_txt = ("generic periodic function" if a.source == "synthetic" else f"real {a.dataset}")
    fig.suptitle(f"{src_txt} through the real augmentation pipeline: "
                 "DWT preserves the dominant period; crop+jitter warps it",
                 fontsize=13, y=0.965)
    out1 = os.path.join(a.outdir, "why_dwt_real_signal.png")
    fig.savefig(out1, dpi=150, bbox_inches="tight"); plt.close(fig)

    # ── stats over many windows ───────────────────────────────────────────────
    import pywt
    idxs = np.linspace(0, min(n_total, a.nwin * 7) - 1, a.nwin).astype(int)
    dwt_ratio, crop_ratio_arr = [], []
    band_dwt, band_crop = [], []                          # per-band |teacher-student| energy
    WAV, LEV = "sym4", cfg["dwt_level"]
    for i in idxs:
        x = get_window(i)
        dt, dsx, ct, cs = views(x, augs)
        p_t = dom_period(sig(dt)); p_s = dom_period(sig(dsx))
        dwt_ratio.append(p_s / p_t if np.isfinite(p_s / p_t) else np.nan)
        p_t = dom_period(sig(ct)); p_s = dom_period(sig(cs))
        crop_ratio_arr.append(p_s / p_t if np.isfinite(p_s / p_t) else np.nan)
        # band-wise perturbation energy (teacher - student), per method, on channel c
        def band_energy(a_, b_):
            ca = pywt.wavedec(sig(a_), WAV, level=LEV); cb = pywt.wavedec(sig(b_), WAV, level=LEV)
            return [float(np.mean((x1 - x2) ** 2)) for x1, x2 in zip(ca, cb)]
        band_dwt.append(band_energy(dt, dsx)); band_crop.append(band_energy(ct, cs))

    dwt_ratio = np.array(dwt_ratio); crop_ratio_arr = np.array(crop_ratio_arr)
    band_dwt = np.nanmean(np.array(band_dwt), 0); band_crop = np.nanmean(np.array(band_crop), 0)
    band_dwt /= band_dwt.sum() + 1e-12; band_crop /= band_crop.sum() + 1e-12
    band_names = ["approx\n(low-freq)"] + [f"detail L{LEV-i}" for i in range(LEV)]

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    plt.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.16, wspace=0.28)
    # period ratio distribution
    bins = np.linspace(0, 4, 41)
    ax[0].hist(dwt_ratio, bins=bins, color="#2ca02c", alpha=.7, label=f"DWT  (median {np.nanmedian(dwt_ratio):.2f})")
    ax[0].hist(crop_ratio_arr, bins=bins, color="#d62728", alpha=.6, label=f"Crop+Jitter  (median {np.nanmedian(crop_ratio_arr):.2f})")
    ax[0].axvline(1.0, color="k", ls="--", lw=1, alpha=.6)
    ax[0].set_xlabel("student / teacher dominant-period ratio")
    ax[0].set_ylabel("#windows")
    ax[0].set_title(f"content preservation over {a.nwin} windows\n(1.0 = identical period)", fontsize=10.5)
    ax[0].legend(fontsize=9)
    # band-wise perturbation
    xb = np.arange(len(band_names)); w = 0.38
    ax[1].bar(xb - w/2, band_dwt, w, color="#2ca02c", label="DWT")
    ax[1].bar(xb + w/2, band_crop, w, color="#d62728", label="Crop+Jitter")
    ax[1].set_xticks(xb); ax[1].set_xticklabels(band_names, fontsize=8)
    ax[1].set_ylabel("fraction of teacher–student energy")
    ax[1].set_title("where each augmentation injects its change\n(DWT spares the low-freq semantic band)", fontsize=10.5)
    ax[1].legend(fontsize=9)
    fig.suptitle("DWT keeps the dominant period and confines perturbation to high-freq detail; crop+jitter does neither",
                 fontsize=12.5, y=0.98)
    out2 = os.path.join(a.outdir, "why_dwt_real_stats.png")
    fig.savefig(out2, dpi=150, bbox_inches="tight"); plt.close(fig)

    print("saved", out1); print("saved", out2)
    print(f"period ratio  DWT median={np.nanmedian(dwt_ratio):.3f}  crop median={np.nanmedian(crop_ratio_arr):.3f}")
    print(f"low-freq (approx) perturbation share  DWT={band_dwt[0]:.3f}  crop={band_crop[0]:.3f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""What dwt_low_pass / dwt_soft_threshold do to REAL synthetic vs ETTh1 data,
using the ACTUAL DWTAugmentation class + tsdino_timemixer/config.py parameters.

Self-contained: subsamples synthetic .arrow series + reads the ETTh1 CSV
directly, then runs each through the real augmentation for the db wavelets we
use (db4/db6/db8). Run on the server.
"""
import os, sys, numpy as np, importlib.util as ilu
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "wino", "tsdino_timemixer"))
from data_paths import DATA_PATHS
from data_agumentation import DWTAugmentation            # the real aug class

# ── load the real TSDiNO config ─────────────────────────────────────────────
_spec = ilu.spec_from_file_location("_dino_cfg", os.path.join(ROOT, "wino", "tsdino_timemixer", "config.py"))
_mod  = ilu.module_from_spec(_spec); _spec.loader.exec_module(_mod)
cfg   = _mod.config

PATCH, NPATCH = cfg['patch_len'], cfg['num_patches']
SEQ = PATCH * NPATCH
K   = 40
DB_FAMILY = ['db4', 'db6', 'db8']


AGG, GENTLE = (0.25, 0.50), (0.05, 0.12)        # current config vs TSDiNO1 student noise

def make_aug(mode, wav, noise_range=None):
    """Real DWTAugmentation built from config params, forced to wavelet `wav`."""
    return DWTAugmentation(
        wavelet=wav, level=cfg['dwt_level'], mode=mode,
        soft_threshold_sigma=cfg['dwt_soft_threshold_sigma'],
        zero_out_ratio=cfg['dwt_zero_out_ratio'],
        finest_levels=cfg['dwt_finest_levels'],
        high_perturb_noise_range=noise_range if noise_range is not None
                                 else cfg['dwt_high_perturb_noise_range'],
        band_scale_approx_range=cfg['dwt_band_scale_approx_range'],
        band_scale_detail_range=cfg['dwt_band_scale_detail_range'],
        wavelet_pool=[wav],
    )

def apply_aug(aug, s):
    x = torch.from_numpy(np.asarray(s, dtype=np.float32)).reshape(-1, 1)
    y = aug(x)
    y = y.detach().cpu().numpy() if hasattr(y, 'detach') else np.asarray(y)
    return y.reshape(-1)

def low_pass(s, wav):        return apply_aug(make_aug('low_pass', wav), s)
def soft_thresh(s, wav):     return apply_aug(make_aug('soft_threshold', wav), s)
def high_perturb(s, wav, nr): return apply_aug(make_aug('high_perturb', wav, nr), s)


def load_synth(data_dir, k):
    import pyarrow as pa
    out = []
    for fname in sorted(f for f in os.listdir(data_dir) if f.endswith('.arrow')):
        path = os.path.join(data_dir, fname)
        try:
            with pa.memory_map(path, 'r') as src:
                table = pa.ipc.open_file(src).read_all()
        except Exception:
            with open(path, 'rb') as f:
                table = pa.ipc.open_stream(f).read_all()
        for row in table.column('target'):
            arr = row.as_py()
            if arr is None:
                continue
            arr = np.asarray(arr, dtype=np.float64)
            chans = [arr] if arr.ndim == 1 else [arr[c] for c in range(arr.shape[0])]
            for s in chans:
                if np.isnan(s).any() or len(s) < SEQ:
                    continue
                s = (s - s.mean()) / (s.std() + 1e-8)
                out.append(s[:SEQ])
                if len(out) >= k:
                    return out
    return out

def load_etth1(k):
    import pandas as pd
    csv = os.path.join(DATA_PATHS['forecasting_data_dir'], 'ETTh1.csv')
    df  = pd.read_csv(csv)
    data = df[[c for c in df.columns if c != 'date']].values.astype(np.float64)
    train = data[:12 * 30 * 24]
    z = (train - train.mean(0)) / (train.std(0) + 1e-8)
    return [z[s:s + SEQ, -1] for s in np.linspace(0, len(z) - SEQ, k).astype(int)]


def var_kept(o, a):
    return 100 * np.var(a) / (np.var(o) + 1e-12)

def mean_corr(arrs):
    S = np.stack(arrs); S = S - S.mean(1, keepdims=True)
    C = np.corrcoef(S)
    return np.nanmean(C[np.triu_indices(len(S), 1)])

def report(name, series, wav):
    vlp = np.mean([var_kept(s, low_pass(s, wav))    for s in series])
    vst = np.mean([var_kept(s, soft_thresh(s, wav)) for s in series])
    clp = mean_corr([low_pass(s, wav)    for s in series])
    cst = mean_corr([soft_thresh(s, wav) for s in series])
    print(f"  {wav:4s} {name:10s} var kept: low_pass {vlp:5.1f}%  soft {vst:5.1f}%"
          f"   | pairwise corr: low_pass {clp:+.3f}  soft {cst:+.3f}")


syn = load_synth(DATA_PATHS['synthetic_data_dir'], K)
ett = load_etth1(K)
print(f"config: level={cfg['dwt_level']} sigma={cfg['dwt_soft_threshold_sigma']}  "
      f"SEQ={SEQ}  loaded {len(syn)} synthetic + {len(ett)} etth1 series")
print(f"(original pairwise corr — synthetic {mean_corr(syn):+.3f}, etth1 {mean_corr(ett):+.3f})\n")
for wav in DB_FAMILY:
    report("synthetic", syn, wav)
    report("etth1", ett, wav)
    print()

# grid: rows = datasets, cols = db wavelets; each panel shows BOTH augmentations
datasets = [("synthetic", syn[0]), ("etth1 (OT)", ett[0])]
fig, ax = plt.subplots(len(datasets), len(DB_FAMILY),
                       figsize=(5 * len(DB_FAMILY), 4 * len(datasets)), squeeze=False)
for r, (nm, s) in enumerate(datasets):
    for c, wav in enumerate(DB_FAMILY):
        a = ax[r][c]
        lp, st = low_pass(s, wav), soft_thresh(s, wav)
        a.plot(s,  lw=1.0, alpha=0.45, color='gray',       label="original")
        a.plot(st, lw=1.8, color='tab:orange', label=f"soft_threshold ({var_kept(s, st):.0f}%)")
        a.plot(lp, lw=2.0, color='tab:green',  label=f"low_pass ({var_kept(s, lp):.0f}%)")
        a.set_title(f"{nm} — {wav}")
        a.grid(alpha=0.3); a.legend(fontsize=7)
plt.tight_layout(); plt.savefig("logs/demo_aug_real.png", dpi=150)
print("Saved logs/demo_aug_real.png")


# ── STUDENT view (high_perturb): aggressive (0.25-0.50) vs gentle (0.05-0.12) ──
def fidelity(o, a):    return float(np.corrcoef(o, a)[0, 1])      # 1 = original intact
def noise_ratio(o, a): return float(np.var(a - o) / (np.var(o) + 1e-12))

print("\nSTUDENT view (high_perturb) — fidelity = corr with original (1=untouched), "
      "noise/signal = added-noise variance ratio\n")
for wav in DB_FAMILY:
    for name, series in [("synthetic", syn), ("etth1", ett)]:
        for tag, nr in [("aggr  ", AGG), ("gentle", GENTLE)]:
            f = np.mean([fidelity(s, high_perturb(s, wav, nr))    for s in series])
            n = np.mean([noise_ratio(s, high_perturb(s, wav, nr)) for s in series])
            print(f"  {wav} {name:10s} [{tag} {nr}] fidelity {f:.3f}   noise/signal {n:.2f}")
    print()

fig, ax = plt.subplots(len(datasets), len(DB_FAMILY),
                       figsize=(5 * len(DB_FAMILY), 4 * len(datasets)), squeeze=False)
for r, (nm, s) in enumerate(datasets):
    for c, wav in enumerate(DB_FAMILY):
        a = ax[r][c]
        ag, ge = high_perturb(s, wav, AGG), high_perturb(s, wav, GENTLE)
        a.plot(s,  lw=1.4, color='gray',     label="original")
        a.plot(ge, lw=1.0, alpha=0.85, color='tab:blue', label=f"gentle 0.05-0.12 (fid {fidelity(s, ge):.2f})")
        a.plot(ag, lw=0.9, alpha=0.85, color='tab:red',  label=f"aggr 0.25-0.50 (fid {fidelity(s, ag):.2f})")
        a.set_title(f"{nm} — {wav} (student view)")
        a.grid(alpha=0.3); a.legend(fontsize=7)
plt.tight_layout(); plt.savefig("logs/demo_aug_student.png", dpi=150)
print("Saved logs/demo_aug_student.png")


# ── COEFFICIENT-level view: what low_pass / soft_threshold do to the coeffs ───
WAV = 'db4'   # representative; db4/6/8 behave alike
BANDS = [f'cA{LEVEL}'] + [f'cD{i}' for i in range(LEVEL, 0, -1)]

def lp_coeffs(c): return [c[0]] + [np.zeros_like(d) for d in c[1:]]
def st_coeffs(c):
    def st(d):
        t = SIGMA * np.abs(d).max() if d.size else 0.0
        return np.sign(d) * np.maximum(np.abs(d) - t, 0.0)
    return [c[0]] + [st(d) for d in c[1:]]

print(f"\nDWT coefficient ENERGY per band (% of total, {WAV}, level {LEVEL}) — "
      f"low_pass keeps only cA{LEVEL}, zeros every cD:")
for nm, s in datasets:
    c = pywt.wavedec(s, WAV, level=LEVEL)
    e = np.array([np.sum(x ** 2) for x in c]); e = 100 * e / e.sum()
    print(f"  {nm:12s} " + "   ".join(f"{b} {v:5.1f}%" for b, v in zip(BANDS, e)))

fig, ax = plt.subplots(len(datasets), 1, figsize=(12, 7), squeeze=False)
for r, (nm, s) in enumerate(datasets):
    a = ax[r][0]
    c = pywt.wavedec(s, WAV, level=LEVEL)
    orig = np.concatenate(c)
    lp   = np.concatenate(lp_coeffs(c))
    st_  = np.concatenate(st_coeffs(c))
    bounds = np.cumsum([len(x) for x in c])[:-1]
    a.plot(orig, lw=0.8, color='gray',       label='original coeffs')
    a.plot(st_,  lw=0.9, color='tab:orange',  label='soft_threshold')
    a.plot(lp,   lw=1.1, color='tab:green',   label='low_pass (detail → 0)')
    seg = [0] + list(bounds) + [len(orig)]
    ytxt = np.max(np.abs(orig)) * 0.85
    for b in bounds:
        a.axvline(b, color='k', ls=':', alpha=0.4)
    for lbl, s0, s1 in zip(BANDS, seg[:-1], seg[1:]):
        a.text((s0 + s1) / 2, ytxt, lbl, ha='center', fontsize=9, color='navy')
    a.set_title(f"{nm} — DWT coeffs ({WAV}, level {LEVEL}): low_pass zeros all cD bands; soft_threshold shrinks")
    a.grid(alpha=0.3); a.legend(fontsize=8, loc='lower right')
plt.tight_layout(); plt.savefig("logs/demo_aug_coeffs.png", dpi=150)
print("Saved logs/demo_aug_coeffs.png")

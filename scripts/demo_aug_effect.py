#!/usr/bin/env python
"""Show what dwt_low_pass vs dwt_soft_threshold do to the data.

Mirrors tsdino_timemixer/data_agumentation.py: pywt.wavedec(level=3), then
 - low_pass:       zero ALL detail coeffs, keep approximation
 - soft_threshold: sign(c)*max(|c|-sigma*max|c|,0) on each detail level

Usage:
    python scripts/demo_aug_effect.py            # writes a demo figure to vis/
"""
import numpy as np, pywt
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

WAVELET, LEVEL, SIGMA, N = "sym4", 3, 0.6, 336
rng = np.random.default_rng(0)

def low_pass(x):
    c = pywt.wavedec(x, WAVELET, level=LEVEL)
    return pywt.waverec([c[0]] + [np.zeros_like(d) for d in c[1:]], WAVELET)[:len(x)]

def soft_thresh(x):
    c = pywt.wavedec(x, WAVELET, level=LEVEL)
    def st(d):
        t = SIGMA * np.abs(d).max() if d.size else 0.0
        return np.sign(d) * np.maximum(np.abs(d) - t, 0.0)
    return pywt.waverec([c[0]] + [st(d) for d in c[1:]], WAVELET)[:len(x)]

t = np.linspace(0, 1, N)
# "etth1-like": strong low-freq periodicity (daily/weekly) + small noise
def etth1_like():
    return (np.sin(2*np.pi*3*t) + 0.5*np.sin(2*np.pi*1*t)
            + 0.1*rng.standard_normal(N))
# "synthetic-like" (kernel-synth style): high-freq, diverse, ~flat low band
def synth_like():
    f = rng.uniform(12, 28)
    return (np.sin(2*np.pi*f*t + rng.uniform(0, 6))
            + 0.6*np.sin(2*np.pi*rng.uniform(20, 40)*t)
            + 0.3*rng.standard_normal(N))

def var_kept(orig, aug):
    return 100 * np.var(aug) / np.var(orig)

# ── variance retained (single example each) ─────────────────────────────────
print("variance retained after augmentation (% of original):")
for name, gen in [("etth1-like", etth1_like), ("synthetic-like", synth_like)]:
    x = gen()
    print(f"  {name:15s}  low_pass {var_kept(x, low_pass(x)):5.1f}%   "
          f"soft_threshold {var_kept(x, soft_thresh(x)):5.1f}%")

# ── cross-sample similarity: can the teacher still tell samples apart? ───────
def mean_pairwise_corr(samples):
    S = np.stack(samples); S = S - S.mean(1, keepdims=True)
    C = np.corrcoef(S); return C[np.triu_indices(len(S), 1)].mean()

print("\nmean pairwise correlation across 20 DIFFERENT synthetic samples")
print("(higher = teacher sees them as more identical → collapse):")
synth = [synth_like() for _ in range(20)]
print(f"  original        {mean_pairwise_corr(synth):+.3f}")
print(f"  low_pass        {mean_pairwise_corr([low_pass(s) for s in synth]):+.3f}")
print(f"  soft_threshold  {mean_pairwise_corr([soft_thresh(s) for s in synth]):+.3f}")

# ── figure ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
for a, (name, gen) in zip(ax, [("etth1-like (low-freq)", etth1_like),
                               ("synthetic-like (high-freq)", synth_like)]):
    x = gen()
    a.plot(x, lw=1.0, alpha=0.6, label="original")
    a.plot(soft_thresh(x), lw=1.8, label="soft_threshold (teacher keeps structure)")
    a.plot(low_pass(x), lw=2.2, label="low_pass (teacher view)")
    a.set_title(name); a.grid(alpha=0.3); a.legend(fontsize=8)
plt.tight_layout(); plt.savefig("logs/demo_aug_effect.png", dpi=150)
print("\nSaved logs/demo_aug_effect.png")

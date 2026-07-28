#!/usr/bin/env python
"""
why_dwt.py — the function whose identity crop+jitter destroy but DWT preserves.

Signal:  x(t) = smooth carrier  +  a LOCALIZED oscillatory burst (Gabor atom)  +  noise
The burst (a transient at time t0, frequency f0) is the semantic "identity".

  * crop  : a local crop can excise the burst (it is localized in time); resizing the
            window back to full length TIME-WARPS whatever survives (frequency changes).
  * jitter: broadband additive noise with no denoising -> buries the burst.
  * DWT   : the burst is a compact wavelet atom (localized in time+scale).
            easy = soft-threshold detail  -> removes noise, keeps the atom.
            hard = perturb detail bands    -> keeps approximation, atom survives at the
                                              SAME time & frequency.
  => DWT's two views share the atom (useful invariant); crop+jitter's do not.

Usage:
    python scripts/why_dwt.py                     # writes the didactic figure to vis/
"""
import os, numpy as np, pywt

_VIS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vis")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(1)
N = 512
t = np.linspace(0, 1, N)

# ── signal: carrier + localized burst (identity) + noise ─────────────────────
T0, F0, W = 0.40, 20.0, 0.035
carrier = 0.45 * np.sin(2 * np.pi * 2.0 * t)
burst   = np.exp(-((t - T0) / W) ** 2) * np.sin(2 * np.pi * F0 * (t - T0))   # the identity
noise   = 0.12 * rng.standard_normal(N)
x = carrier + burst + noise
burst_tmpl = burst / (np.linalg.norm(burst) + 1e-9)

WAV, LEVEL = "db4", 5

# ── DWT augmentation ─────────────────────────────────────────────────────────
def dwt_easy(sig):                       # teacher: soft-threshold detail -> denoise
    c = pywt.wavedec(sig, WAV, level=LEVEL); out = [c[0]]
    for d in c[1:]:
        sigma = np.median(np.abs(d)) / 0.6745 + 1e-9
        out.append(pywt.threshold(d, 1.0 * sigma * np.sqrt(2 * np.log(len(sig))), mode="soft"))
    return pywt.waverec(out, WAV)[:N]

def dwt_hard(sig):                        # student: keep approx, perturb detail bands
    c = pywt.wavedec(sig, WAV, level=LEVEL); out = [c[0]]
    for d in c[1:]:
        out.append(d + 1.2 * np.std(d) * rng.standard_normal(len(d)))
    return pywt.waverec(out, WAV)[:N]

easy_d, hard_d = dwt_easy(x), dwt_hard(x)

# ── crop+jitter augmentation ─────────────────────────────────────────────────
def jitter(sig, std=0.12): return sig + std * rng.standard_normal(len(sig))
def crop_resize(sig, ratio, start):      # DINO local crop: window -> resize to N (warps time)
    L = int(len(sig) * ratio); s0 = int(start * len(sig)); seg = sig[s0:s0 + L]
    return np.interp(np.linspace(0, 1, N), np.linspace(0, 1, len(seg)), seg)

easy_c = jitter(x, 0.10)                              # global crop 1.0 + jitter (keeps burst)
hard_c = jitter(crop_resize(x, 0.40, 0.52), 0.10)    # local 40% window [0.52..0.92]: burst EXCISED

# ── identity-preservation metric: correlation with the clean burst template ──
def burst_score(sig): return float(abs(np.dot(sig / (np.linalg.norm(sig) + 1e-9), burst_tmpl)))
bd_e, bd_h = burst_score(easy_d), burst_score(hard_d)
bc_e, bc_h = burst_score(easy_c), burst_score(hard_c)

# ─────────────────────────── figure 1: time domain ──────────────────────────
TEACH, STUD, TRUE = "#1f77b4", "#e8703a", "0.5"
fig, ax = plt.subplots(2, 4, figsize=(15.5, 6.0))
plt.subplots_adjust(left=0.075, right=0.995, top=0.86, bottom=0.06, wspace=0.15, hspace=0.4)

def clean(a):
    a.set_xticks([]); a.set_yticks([]); a.set_ylim(-1.8, 1.8)
    for sp in a.spines.values(): sp.set_visible(False)

def mark_burst(a, present=True):
    col = "#2ca02c" if present else "#d62728"
    a.axvspan(T0 - 2.2 * W, T0 + 2.2 * W, color=col, alpha=0.10, lw=0)

rows = [("DWT  (ours)",   easy_d, hard_d, bd_e, bd_h, True),
        ("Crop + Jitter", easy_c, hard_c, bc_e, bc_h, False)]
for r, (name, easy, hard, se, sh, good) in enumerate(rows):
    for a in ax[r]: clean(a)
    ax[r, 0].plot(t, x, color="0.25", lw=1.1); mark_burst(ax[r, 0]); ax[r, 0].set_title("input $x$", fontsize=10.5)
    ax[r, 1].plot(t, easy, color=TEACH, lw=1.3); mark_burst(ax[r, 1]); ax[r, 1].set_title(f"easy (teacher)   burst={se:.2f}", fontsize=10.5)
    ax[r, 2].plot(t, hard, color=STUD, lw=1.3); mark_burst(ax[r, 2], se and sh > 0.25)
    ax[r, 2].set_title(f"hard (student)   burst={sh:.2f}", fontsize=10.5)
    ax[r, 3].plot(t, easy, color=TEACH, lw=1.2, alpha=.85, label="easy")
    ax[r, 3].plot(t, hard, color=STUD, lw=1.2, alpha=.85, label="hard")
    mark = "✓ shared" if good else "✗ lost"
    mc = "#2ca02c" if good else "#d62728"
    ax[r, 3].set_title(f"overlay — invariant  {mark}", fontsize=10.5, color=mc)
    ax[r, 0].text(-0.12, 0.5, name, rotation=90, va="center", ha="center",
                  transform=ax[r, 0].transAxes, fontsize=12, fontweight="bold", color=mc)
ax[0, 3].legend(loc="upper right", fontsize=8, framealpha=.9)
fig.suptitle("A localized burst (the identity): DWT keeps it in both views, crop+jitter destroys it",
             fontsize=13, y=0.955)
fig.text(0.5, 0.905, "shaded = where the burst lives.  crop excises it from the student view; "
         "jitter buries it; DWT (soft-threshold + detail-perturb) preserves the atom in place.",
         ha="center", fontsize=9.5, color="0.3")
out1 = os.path.join(_VIS, "why_dwt_time.png")
fig.savefig(out1, dpi=150, bbox_inches="tight"); plt.close(fig)

# ─────────────────────── figure 2: time-frequency scalograms ────────────────
scales = np.arange(2, 64)
def scalo(sig):
    cwt, freqs = pywt.cwt(sig, scales, "morl", sampling_period=1.0 / N)
    return np.abs(cwt), freqs
fig, ax = plt.subplots(2, 3, figsize=(13.5, 6.0))
plt.subplots_adjust(left=0.07, right=0.985, top=0.86, bottom=0.09, wspace=0.14, hspace=0.34)
panels = [("DWT  (ours)",   x, easy_d, hard_d, True),
          ("Crop + Jitter", x, easy_c, hard_c, False)]
for r, (name, xin, easy, hard, good) in enumerate(panels):
    for cidx, (sig, ttl) in enumerate([(xin, "input"), (easy, "easy (teacher)"), (hard, "hard (student)")]):
        S, fr = scalo(sig)
        ax[r, cidx].imshow(S, aspect="auto", cmap="magma", extent=[0, 1, scales[-1], scales[0]])
        ax[r, cidx].set_title(ttl, fontsize=10)
        ax[r, cidx].set_yticks([])
        if cidx == 0:
            mc = "#2ca02c" if good else "#d62728"
            ax[r, 0].text(-0.10, 0.5, name, rotation=90, va="center", ha="center",
                          transform=ax[r, 0].transAxes, fontsize=12, fontweight="bold", color=mc)
            ax[r, 0].set_ylabel("scale")
        ax[r, cidx].set_xlabel("time")
        # ring the burst atom location
        ax[r, cidx].scatter([T0], [scales[len(scales)//2]], s=1, alpha=0)  # keep limits
fig.suptitle("Time–frequency view: the burst is a compact atom — preserved by DWT, smeared/erased by crop",
             fontsize=12.5, y=0.955)
fig.text(0.5, 0.905, "DWT keeps the bright atom fixed in time & scale across easy/hard; "
         "crop shifts/erases it (the student panel is blank where the atom should be).",
         ha="center", fontsize=9.5, color="0.3")
out2 = os.path.join(_VIS, "why_dwt_tf.png")
fig.savefig(out2, dpi=150, bbox_inches="tight"); plt.close(fig)

print("saved", out1); print("saved", out2)
print(f"burst score  DWT: easy={bd_e:.2f} hard={bd_h:.2f}   |   crop+jitter: easy={bc_e:.2f} hard={bc_h:.2f}")

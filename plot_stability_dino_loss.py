#!/usr/bin/env python
"""Overlay DINO-loss-per-epoch (epoch average) curves for the
Testing_for_stability runs.

The epoch average is read from each "Averaged stats:" line — `dino_loss:
<last> (<avg>)`, where the parenthesized value is the running average over the
epoch. The epoch index comes from the preceding "Starting epoch N".

Usage:
    python plot_stability_dino_loss.py [glob] [out.png] [title]
"""
import re
import sys
import glob
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

pattern  = sys.argv[1] if len(sys.argv) > 1 else "logs/Testing_for_stability/*.log"
out_path = sys.argv[2] if len(sys.argv) > 2 else "logs/Testing_for_stability/dino_loss.png"
title    = sys.argv[3] if len(sys.argv) > 3 else "Testing for stability — DINO loss per epoch (avg)"

start_pat = re.compile(r"Starting epoch (\d+)")
dino_pat  = re.compile(r"Averaged stats:.*?dino_loss:\s*[0-9.]+\s*\(([0-9.]+)\)")
# fallback: when MLM is off (mlm_phi 0.0) the log has no dino_loss field —
# the total loss IS the DINO loss.
total_pat = re.compile(r"Averaged stats:.*?(?<!\w)loss:\s*[0-9.]+\s*\(([0-9.]+)\)")

runs = {}
for f in sorted(glob.glob(pattern)):
    label = os.path.splitext(os.path.basename(f))[0]
    cur_ep = None
    ep, lo = [], []
    with open(f) as fh:
        for line in fh:
            ms = start_pat.search(line)
            if ms:
                cur_ep = int(ms.group(1))
                continue
            if "Averaged stats:" not in line or cur_ep is None:
                continue
            ma = dino_pat.search(line) or total_pat.search(line)
            if ma:
                ep.append(cur_ep)
                lo.append(float(ma.group(1)))
    if ep:
        runs[label] = (ep, lo)

if not runs:
    sys.exit(f"No 'Averaged stats: ... dino_loss: (avg)' lines found under {pattern}")

plt.figure(figsize=(10, 6))
for label, (ep, lo) in runs.items():
    bi = min(range(len(lo)), key=lambda i: lo[i])
    line, = plt.plot(ep, lo, marker="o", ms=3, lw=1.3,
                     label=f"{label}  (best {lo[bi]:.4f}@{ep[bi]})")
    plt.scatter([ep[bi]], [lo[bi]], color=line.get_color(), s=55,
                edgecolor="black", zorder=5)

plt.xlabel("Epoch")
plt.ylabel("DINO loss (epoch avg)")
plt.title(title)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(out_path, dpi=150)
print(f"Saved {out_path}")
for label, (ep, lo) in runs.items():
    bi = min(range(len(lo)), key=lambda i: lo[i])
    print(f"  {label}: {len(ep)} epochs, best {lo[bi]:.4f} @ epoch {ep[bi]}")

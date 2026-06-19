#!/usr/bin/env python
"""Overlay validation-loss-per-epoch curves for the Testing_for_stability runs.

Usage:
    python plot_stability_val_loss.py [glob] [out.png] [title]
Defaults overlay every *.log under logs/Testing_for_stability/.
"""
import re
import sys
import glob
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

pattern  = sys.argv[1] if len(sys.argv) > 1 else "logs/Testing_for_stability/*.log"
out_path = sys.argv[2] if len(sys.argv) > 2 else "logs/Testing_for_stability/val_loss.png"
title    = sys.argv[3] if len(sys.argv) > 3 else "Testing for stability — validation loss per epoch"

val_pat = re.compile(r"Epoch (\d+) — val loss:\s*([0-9.]+)")

runs = {}
for f in sorted(glob.glob(pattern)):
    label = os.path.splitext(os.path.basename(f))[0]
    ep, lo = [], []
    with open(f) as fh:
        for line in fh:
            m = val_pat.search(line)
            if m:
                ep.append(int(m.group(1)))
                lo.append(float(m.group(2)))
    if ep:
        runs[label] = (ep, lo)

if not runs:
    sys.exit(f"No 'Epoch N — val loss:' lines found under {pattern}")

plt.figure(figsize=(10, 6))
for label, (ep, lo) in runs.items():
    bi = min(range(len(lo)), key=lambda i: lo[i])
    line, = plt.plot(ep, lo, marker="o", ms=3, lw=1.3,
                     label=f"{label}  (best {lo[bi]:.4f}@{ep[bi]})")
    plt.scatter([ep[bi]], [lo[bi]], color=line.get_color(), s=55,
                edgecolor="black", zorder=5)

plt.xlabel("Epoch")
plt.ylabel("Validation loss")
plt.title(title)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(out_path, dpi=150)
print(f"Saved {out_path}")
for label, (ep, lo) in runs.items():
    bi = min(range(len(lo)), key=lambda i: lo[i])
    print(f"  {label}: {len(ep)} epochs, best {lo[bi]:.4f} @ epoch {ep[bi]}")

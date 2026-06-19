#!/usr/bin/env python
"""Overlay validation-loss-per-epoch curves from an explicit list of log files.

Usage: plot_val_loss_files.py OUT.png TITLE log1[:label1] log2[:label2] ...
"""
import re
import sys
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out_path = sys.argv[1]
title = sys.argv[2]
specs = sys.argv[3:]

pat = re.compile(r"Epoch (\d+) — val loss:\s*([0-9.]+)")
plt.figure(figsize=(10, 6))

summary = []
for spec in specs:
    if ":" in spec and not spec.split(":", 1)[0].endswith(".log") is False:
        pass
    # split on last ':' only if label provided
    if "::" in spec:
        path, label = spec.split("::", 1)
    else:
        path, label = spec, None
    if label is None:
        label = os.path.basename(path).replace("pretrain_dino_patchtst_synthetic_", "").replace(".log", "")
    ep, lo = [], []
    with open(path) as fh:
        for line in fh:
            m = pat.search(line)
            if m:
                ep.append(int(m.group(1)))
                lo.append(float(m.group(2)))
    if not ep:
        print(f"WARN no val-loss lines in {path}")
        continue
    bi = min(range(len(lo)), key=lambda i: lo[i])
    line, = plt.plot(ep, lo, marker="o", ms=4, lw=1.4,
                     label=f"{label}  (best {lo[bi]:.3f}@{ep[bi]})")
    plt.scatter([ep[bi]], [lo[bi]], color=line.get_color(), s=60,
                edgecolor="black", zorder=5)
    summary.append((label, len(ep), lo[bi], ep[bi]))

plt.xlabel("Epoch")
plt.ylabel("Validation loss")
plt.title(title)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(out_path, dpi=150)
print(f"Saved {out_path}")
for label, n, best, be in summary:
    print(f"  {label}: {n} epochs, best {best:.4f} @ epoch {be}")

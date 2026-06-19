#!/usr/bin/env python
"""Parse a training log and plot validation loss per epoch."""
import re
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

log_path = sys.argv[1] if len(sys.argv) > 1 else "logs/pretrain_dino_tsmixer_synthetic_mae.log"
out_path = sys.argv[2] if len(sys.argv) > 2 else "logs/val_loss_per_epoch.png"
title = sys.argv[3] if len(sys.argv) > 3 else "DINO+MAE (tsmixer, synthetic) — validation loss per epoch"

pat = re.compile(r"Epoch (\d+) — val loss:\s*([0-9.]+)")
epochs, losses = [], []
with open(log_path) as f:
    for line in f:
        m = pat.search(line)
        if m:
            epochs.append(int(m.group(1)))
            losses.append(float(m.group(2)))

if not epochs:
    sys.exit("No 'Epoch N — val loss:' lines found.")

best_i = min(range(len(losses)), key=lambda i: losses[i])

plt.figure(figsize=(9, 5))
plt.plot(epochs, losses, marker="o", color="#1f77b4", label="val loss")
plt.scatter([epochs[best_i]], [losses[best_i]], color="red", zorder=5,
            label=f"best (ep {epochs[best_i]}, {losses[best_i]:.4f})")
plt.xlabel("Epoch")
plt.ylabel("Validation loss")
plt.title(title)
plt.xticks(epochs)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(out_path, dpi=150)
print(f"Saved {out_path}  ({len(epochs)} epochs, best={losses[best_i]:.4f} @ epoch {epochs[best_i]})")

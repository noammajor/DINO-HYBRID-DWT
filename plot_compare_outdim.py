#!/usr/bin/env python
"""Compare a per-epoch (epoch-average) loss between the out_dim=2048 and
out_dim=8192 Testing_for_stability runs.

For a run base name `<base>`:
    2048 → logs/Testing_for_stability/<base>.log
    8192 → logs/Testing_for_stability/<base>_8192.log

The metric is read from the "Averaged stats:" line (parenthesized running
average), paired with the epoch from the preceding "Starting epoch N".

Usage:
    # generate every requested comparison (default)
    python plot_compare_outdim.py
    python plot_compare_outdim.py --all

    # single comparison
    python plot_compare_outdim.py <base> <metric> [out.png] [title]
        metric ∈ {dino_loss, mae_loss, ibot_loss, loss}
        (dino_loss falls back to total loss when MLM is off)
"""
import re
import sys
import os
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIR = "logs/Testing_for_stability"
START_PAT = re.compile(r"Starting epoch (\d+)")


def parse(path, metric):
    """Return (epochs, values) of the epoch-average `metric` from one log."""
    field_pat = re.compile(rf"(?<!\w){metric}:\s*[0-9.]+\s*\(([0-9.]+)\)")
    total_pat = re.compile(r"(?<!\w)loss:\s*[0-9.]+\s*\(([0-9.]+)\)")
    cur, ep, lo = None, [], []
    if not os.path.exists(path):
        return ep, lo
    with open(path) as fh:
        for line in fh:
            m = START_PAT.search(line)
            if m:
                cur = int(m.group(1))
                continue
            if "Averaged stats:" not in line or cur is None:
                continue
            fm = field_pat.search(line)
            if not fm and metric == "dino_loss":
                fm = total_pat.search(line)   # MLM off → total loss IS dino loss
            if fm:
                ep.append(cur)
                lo.append(float(fm.group(1)))
    return ep, lo


def compare(base, metric, out_path, title):
    series = [
        ("2048", f"{DIR}/{base}.log",        "#1f77b4"),
        ("8192", f"{DIR}/{base}_8192.log",   "#d62728"),
    ]
    plotted = {}
    plt.figure(figsize=(10, 6))
    for tag, path, color in series:
        ep, lo = parse(path, metric)
        if not ep:
            print(f"  [skip] no '{metric}' data in {path}")
            continue
        bi = min(range(len(lo)), key=lambda i: lo[i])
        plt.plot(ep, lo, marker="o", ms=3, lw=1.4, color=color,
                 label=f"out_dim={tag}  (best {lo[bi]:.4f}@{ep[bi]})")
        plt.scatter([ep[bi]], [lo[bi]], color=color, s=55,
                    edgecolor="black", zorder=5)
        plotted[tag] = dict(zip(ep, lo))
    if not plotted:
        plt.close()
        print(f"  [skip] nothing to plot for {base} / {metric}")
        return

    # average distance between the two curves in log scale (mean |Δlog10|),
    # over the epochs both runs share
    if "2048" in plotted and "8192" in plotted:
        common = sorted(set(plotted["2048"]) & set(plotted["8192"]))
        diffs = [abs(math.log10(plotted["8192"][e]) - math.log10(plotted["2048"][e]))
                 for e in common
                 if plotted["2048"][e] > 0 and plotted["8192"][e] > 0]
        if diffs:
            avg = sum(diffs) / len(diffs)
            plt.gca().text(
                0.5, 1.015,
                f"avg log-distance (mean |Δlog₁₀|) = {avg:.3f}   "
                f"≈ {10**avg:.2f}× over {len(diffs)} epochs",
                transform=plt.gca().transAxes, ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="#333333")

    plt.xlabel("Epoch")
    plt.ylabel(f"{metric} (epoch avg)")
    plt.title(title, pad=24)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    extra = ""
    if "2048" in plotted and "8192" in plotted and diffs:
        extra = f"  | avg |Δlog10| = {avg:.3f} (~{10**avg:.2f}x, {len(diffs)} ep)"
    print(f"Saved {out_path}{extra}")


# (out_name, title, base, metric)
ALL_PLOTS = [
    # DINO loss per variant — 2048 vs 8192 (each its own)
    ("compare_dino_ibot.png",      "iBOT run — DINO loss: 2048 vs 8192",            "patchtst_ibot",        "dino_loss"),
    ("compare_dino_mae.png",       "MAE run (transformer) — DINO loss: 2048 vs 8192", "patchtst_transformer", "dino_loss"),
    ("compare_dino_dino_only.png", "DINO-only run — DINO loss: 2048 vs 8192",       "patchtst_dino_only",   "dino_loss"),
    ("compare_dino_db.png",        "DB run — DINO loss: 2048 vs 8192",              "patchtst_db",          "dino_loss"),
    # auxiliary MLM loss — 2048 vs 8192
    ("compare_ibot_loss.png",            "iBOT run — iBOT loss: 2048 vs 8192",       "patchtst_ibot",        "ibot_loss"),
    ("compare_mae_loss_transformer.png", "MAE run (transformer) — MAE loss: 2048 vs 8192", "patchtst_transformer", "mae_loss"),
    ("compare_mae_loss_db.png",          "DB run — MAE loss: 2048 vs 8192",          "patchtst_db",          "mae_loss"),
]


def main():
    args = sys.argv[1:]
    if not args or args[0] == "--all":
        for out_name, title, base, metric in ALL_PLOTS:
            compare(base, metric, f"{DIR}/{out_name}", title)
        return
    base   = args[0]
    metric = args[1] if len(args) > 1 else "dino_loss"
    out    = args[2] if len(args) > 2 else f"{DIR}/compare_{metric}_{base}.png"
    title  = args[3] if len(args) > 3 else f"{base} — {metric}: 2048 vs 8192"
    compare(base, metric, out, title)


if __name__ == "__main__":
    main()

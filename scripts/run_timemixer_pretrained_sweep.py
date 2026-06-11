#!/usr/bin/env python3
"""
run_timemixer_pretrained_sweep.py — Supervised TimeMixer forecasting with
pretrained TSMixer backbones.

Loads checkpoint_best.pth from each specified checkpoint directory,
injects the encoder weights into TimeMixer, and runs supervised forecasting
on all datasets.

Usage
-----
nohup python scripts/run_timemixer_pretrained_sweep.py --gpu 0 \
    > logs/timemixer_pretrained/launcher.log 2>&1 &

# specific checkpoints
nohup python scripts/run_timemixer_pretrained_sweep.py --gpu 0 \
    --ckpt_dirs checkpoints_synthetic_layers4_outdim8192_tsmixer \
                checkpoints_synthetic_layers4_outdim8192_tsmixer_ibot \
                checkpoints_synthetic_layers4_outdim8192_tsmixer_mae \
    > logs/timemixer_pretrained/launcher.log 2>&1 &
"""

import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from Train_and_downstream import run  # noqa: E402

DATASETS  = ["etth1", "etth2", "ettm1", "ettm2", "weather"]
PRED_LENS = [96, 192, 336, 720]


def _find_best_checkpoint(ckpt_dir: Path, checkpoint_name: str = None):
    """Return path to the best checkpoint in ckpt_dir.

    If checkpoint_name is given (e.g. 'checkpoint20.pth'), use that directly.
    Otherwise:
      1. checkpoint_best.pth  — saved whenever val loss improved
      2. highest-numbered checkpoint{N}.pth  — latest epoch as fallback
    """
    if checkpoint_name is not None:
        p = ckpt_dir / checkpoint_name
        return p if p.exists() else None
    best = ckpt_dir / "checkpoint_best.pth"
    if best.exists():
        return best
    epoch_ckpts = sorted(
        ckpt_dir.glob("checkpoint[0-9]*.pth"),
        key=lambda p: int("".join(filter(str.isdigit, p.stem)) or 0),
    )
    return epoch_ckpts[-1] if epoch_ckpts else None


DEFAULT_CKPT_DIRS = [
    "checkpoints_synthetic_layers4_outdim8192_tsmixer",
    "checkpoints_synthetic_layers4_outdim8192_tsmixer_ibot",
    "checkpoints_synthetic_layers4_outdim8192_tsmixer_mae",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu",        type=int,   default=0)
    p.add_argument("--ckpt_dirs",  nargs="+",  default=DEFAULT_CKPT_DIRS,
                   help="Checkpoint directory names (relative to project root)")
    p.add_argument("--datasets",   nargs="+",  default=DATASETS)
    p.add_argument("--pred_lens",  nargs="+",  type=int, default=PRED_LENS)
    p.add_argument("--encoder_layers", type=int, default=4)
    p.add_argument("--epochs_forecasting", type=int, default=None)
    p.add_argument("--checkpoint_name", type=str, default=None,
                   help="Specific checkpoint file to use (e.g. checkpoint20.pth). Default: checkpoint_best.pth")
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    out_csv = results_dir / "timemixer_pretrained_sweep.csv"

    fieldnames = ["ckpt", "dataset", "pred_len", "mse", "mae", "timestamp"]
    rows = []
    existing = set()
    if out_csv.exists():
        with open(out_csv) as f:
            for row in csv.DictReader(f):
                rows.append(row)
                existing.add((row["ckpt"], row["dataset"], int(row["pred_len"])))

    for ckpt_dir in args.ckpt_dirs:
        ckpt_path = _find_best_checkpoint(ROOT / ckpt_dir, args.checkpoint_name)
        if ckpt_path is None:
            print(f"\n[SKIP] {ckpt_dir} — no checkpoint found")
            continue
        print(f"  Using: {ckpt_path.name}")

        log_dir = ROOT / "logs" / "timemixer_pretrained" / ckpt_dir
        log_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  Checkpoint: {ckpt_dir}")
        print(f"{'='*60}")

        for dataset in args.datasets:
            log_path = log_dir / f"{dataset}.log"
            print(f"\n  [{dataset}]  log={log_path.relative_to(ROOT)}")

            with open(log_path, "w") as fh:
                old_out, old_err = sys.stdout, sys.stderr
                sys.stdout = sys.stderr = fh
                try:
                    result = run(
                        model              = "timemixer",
                        forecast_dataset   = dataset,
                        encoder_layers     = args.encoder_layers,
                        checkpoint         = str(ckpt_path),
                        pred_lens          = args.pred_lens,
                        epochs_forecasting = args.epochs_forecasting,
                    )
                except Exception as e:
                    print(f"ERROR: {e}", flush=True)
                    import traceback; traceback.print_exc()
                    result = None
                finally:
                    sys.stdout, sys.stderr = old_out, old_err

            if result is not None:
                mse = result[1] if isinstance(result, (list, tuple)) else result
                mae = result[2] if isinstance(result, (list, tuple)) and len(result) > 2 else "N/A"
                if mse is not None:
                    print(f"    → MSE={mse:.4f}")
                    ts = datetime.now().isoformat(timespec="seconds")
                    rows.append({
                        "ckpt":     ckpt_dir,
                        "dataset":  dataset,
                        "pred_len": "all",
                        "mse":      f"{mse:.6f}",
                        "mae":      str(mae),
                        "timestamp": ts,
                    })
                    with open(out_csv, "w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(rows)

    print(f"\nDone. Results → {out_csv.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

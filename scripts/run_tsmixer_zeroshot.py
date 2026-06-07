#!/usr/bin/env python3
"""
run_tsmixer_zeroshot.py — Zero-shot forecasting on 5 datasets using the
pretrained TSMixer backbone (checkpoints_synthetic_layers4_outdim8192_tsmixer).

Loads checkpoint_best.pth, freezes the backbone, and trains a linear probe
forecasting head for each pred_len in [96, 192, 336, 720].

Results saved to results/tsmixer_zeroshot.csv.

Usage
-----
nohup python scripts/run_tsmixer_zeroshot.py --gpu 7 \
    > logs/tsmixer_zeroshot/launcher.log 2>&1 &
"""

import argparse
import csv
import os
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from Train_and_downstream import run  # noqa: E402

DATASETS    = ["etth1", "etth2", "ettm1", "ettm2", "weather"]
PRED_LENS   = [96, 192, 336, 720]
FIELDNAMES  = ["dataset", "pred_len", "mse", "timestamp"]

# These are set dynamically in main() based on --ckpt_tag
LOG_FOLDER  = None
RESULTS_CSV = None


class _Tee:
    def __init__(self, original, fh):
        self._orig = original
        self._file = fh

    def write(self, data):
        self._orig.write(data); self._orig.flush()
        self._file.write(data); self._file.flush()

    def flush(self):
        self._orig.flush(); self._file.flush()

    def fileno(self):
        return self._orig.fileno()


@contextmanager
def log_to_file(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as fh:
        fh.write(f"# started {datetime.now().isoformat(timespec='seconds')}\n\n")
        orig = sys.stdout
        sys.stdout = _Tee(orig, fh)
        try:
            yield
        finally:
            sys.stdout = orig


def eval_one(dataset: str, pred_len: int, gpu: int,
             lr_forecasting: float = None, ckpt_tag: str = "tsmixer"):
    """Zero-shot forecasting for a single dataset/pred_len. Returns MSE or None."""
    log_path = LOG_FOLDER / dataset / f"pred{pred_len}.log"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    with log_to_file(log_path):
        try:
            result = run(
                model            = "dino",
                task             = "forecast",   # skip_train=True, linear probe only
                forecast_dataset = dataset,
                pred_lens        = [pred_len],
                backbone_type    = "tsmixer",
                encoder_layers   = 4,
                out_dim          = 8192,
                pretrain_source  = "synthetic",
                ckpt_tag         = ckpt_tag,
                checkpoints      = ["best"],
                linear_probe     = True,
                lr_forecasting   = lr_forecasting,
            )
            # run_dino returns (best_ckpt, best_mse, cls_acc, anom_result)
            if result is not None and result[1] is not None:
                return float(result[1])
        except Exception as e:
            print(f"ERROR: {e}", flush=True)
            import traceback; traceback.print_exc()
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu",            type=int,   default=0)
    p.add_argument("--datasets",       nargs="+",  default=DATASETS)
    p.add_argument("--pred_lens",      nargs="+",  type=int, default=PRED_LENS)
    p.add_argument("--lr_forecasting", type=float, default=None)
    p.add_argument("--ckpt_tag",       type=str,   default="tsmixer",
                   help="Checkpoint tag: tsmixer | tsmixer_ibot | tsmixer_mae (default: tsmixer)")
    args = p.parse_args()

    global LOG_FOLDER, RESULTS_CSV
    LOG_FOLDER  = ROOT / "logs"    / f"tsmixer_zeroshot_{args.ckpt_tag}"
    RESULTS_CSV = ROOT / "results" / f"tsmixer_zeroshot_{args.ckpt_tag}.csv"

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    LOG_FOLDER.mkdir(parents=True, exist_ok=True)
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)

    # Resume from partial results
    rows = []
    existing = set()
    if RESULTS_CSV.exists():
        with open(RESULTS_CSV) as f:
            for row in csv.DictReader(f):
                rows.append(row)
                existing.add((row["dataset"], int(row["pred_len"])))

    print(f"TSMixer zero-shot forecast")
    print(f"  Checkpoint : checkpoints_synthetic_layers4_outdim8192_tsmixer/checkpoint_best.pth")
    print(f"  Datasets   : {args.datasets}")
    print(f"  Pred lens  : {args.pred_lens}")
    print(f"  GPU        : {args.gpu}")

    for dataset in args.datasets:
        for pred_len in args.pred_lens:
            if (dataset, pred_len) in existing:
                print(f"\n[{dataset}/pred{pred_len}] already done, skipping.")
                continue

            print(f"\n[{dataset}/pred{pred_len}]  log={LOG_FOLDER.relative_to(ROOT)}/{dataset}/pred{pred_len}.log")
            mse = eval_one(dataset, pred_len, args.gpu,
                           lr_forecasting=args.lr_forecasting, ckpt_tag=args.ckpt_tag)

            if mse is not None:
                print(f"  → MSE={mse:.4f}")
                rows.append({
                    "dataset":  dataset,
                    "pred_len": pred_len,
                    "mse":      f"{mse:.6f}",
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                })
                existing.add((dataset, pred_len))
                with open(RESULTS_CSV, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                    writer.writeheader()
                    writer.writerows(rows)
                print(f"  → saved to {RESULTS_CSV.relative_to(ROOT)}")
            else:
                print(f"  → FAILED")

    print(f"\nDone. Results in {RESULTS_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

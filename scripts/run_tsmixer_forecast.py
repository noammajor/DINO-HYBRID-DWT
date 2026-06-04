#!/usr/bin/env python3
"""
run_tsmixer_forecast.py — Run forecasting on all 5 datasets using a pretrained
TSMixer checkpoint from ./checkpoints/checkpoint_best.pth.

Usage
-----
python scripts/run_tsmixer_forecast.py --gpu 4
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
LOGS    = ROOT / "logs"
SCRIPT  = ROOT / "Train_and_downstream.py"

DATASETS    = ["etth1", "etth2", "ettm1", "ettm2", "weather"]
LOG_FOLDER  = "Dino_TSMIXER_Regular_synthetic"
CKPT_DIR    = "checkpoints_synthetic_layers4_outdim8192_tsmixer"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu",               type=int, default=0)
    p.add_argument("--epochs_forecasting",type=int, default=None)
    p.add_argument("--dry_run",           action="store_true")
    args = p.parse_args()

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    log_dir = LOGS / LOG_FOLDER
    log_dir.mkdir(parents=True, exist_ok=True)

    for dataset in DATASETS:
        log_path = log_dir / f"{dataset}.log"

        cmd = [
            sys.executable, str(SCRIPT),
            "--model",            "dino",
            "--task",             "forecast",
            "--forecast_dataset", dataset,
            "--backbone_type",    "tsmixer",
            "--encoder_layers",   "4",
            "--out_dim",          "8192",
            "--output_dir",       str(ROOT / CKPT_DIR),
            "--checkpoints",      "best",
        ]
        if args.epochs_forecasting:
            cmd += ["--epochs_forecasting", str(args.epochs_forecasting)]

        print(f"\n[{dataset}]  log={log_path.relative_to(ROOT)}")
        print(f"  {' '.join(cmd)}")

        if args.dry_run:
            continue

        with open(log_path, "w") as fh:
            fh.write(f"# started {datetime.now().isoformat(timespec='seconds')}\n\n")
            fh.flush()
            result = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)

        status = "OK" if result.returncode == 0 else f"FAILED (rc={result.returncode})"
        print(f"  → {status}")


if __name__ == "__main__":
    main()

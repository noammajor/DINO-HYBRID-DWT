#!/usr/bin/env python3
"""
run_tsmixer_forecast.py — Forecast on all 5 datasets using a pretrained TSMixer checkpoint.

Calls run() directly (same as run_layer_forecast.py) so output_dir can be passed explicitly.

Usage
-----
nohup python scripts/run_tsmixer_forecast.py --gpu 7 \
    > logs/Dino_TSMIXER_Regular_synthetic/launcher.log 2>&1 &
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from Train_and_downstream import run  # noqa: E402

DATASETS   = ["etth1", "etth2", "ettm1", "ettm2", "weather"]
CKPT_DIR   = str(ROOT / "checkpoints_synthetic_layers4_outdim8192_tsmixer")
LOG_FOLDER = ROOT / "logs" / "Dino_TSMIXER_Regular_synthetic"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu",               type=int, default=0)
    p.add_argument("--epochs_forecasting",type=int, default=None)
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    LOG_FOLDER.mkdir(parents=True, exist_ok=True)

    for dataset in DATASETS:
        log_path = LOG_FOLDER / f"{dataset}.log"
        print(f"\n[{dataset}]  log={log_path.relative_to(ROOT)}")

        with open(log_path, "w") as fh:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout = sys.stderr = fh
            try:
                run(
                    model            = "dino",
                    task             = "forecast",
                    forecast_dataset = dataset,
                    backbone_type    = "tsmixer",
                    encoder_layers   = 4,
                    out_dim          = 8192,
                    output_dir       = CKPT_DIR,
                    checkpoints      = ["best"],
                    epochs_forecasting = args.epochs_forecasting,
                )
            except Exception as e:
                print(f"ERROR: {e}", flush=True)
                import traceback; traceback.print_exc()
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr

        print(f"  → done")


if __name__ == "__main__":
    main()

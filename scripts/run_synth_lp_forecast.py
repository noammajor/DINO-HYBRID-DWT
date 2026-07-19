#!/usr/bin/env python3
"""
run_synth_lp_forecast.py — LINEAR-PROBE forecasting on the synthetic-pretrained
DINO (WINO-TS) TSMixer checkpoint, for a clean LP-vs-LP comparison against the
previous paper's synthetic linear-probe results (Table 12).

Mirrors run_tsmixer_forecast.py but freezes the backbone (linear_probe=True) and
uses the DINO-synthetic checkpoint tag ("tsmixer", not "tsmixer_ibot"), i.e.
    checkpoints_synthetic_layers4_outdim8192_tsmixer/checkpoint_best.pth

Usage
-----
nohup python scripts/run_synth_lp_forecast.py --gpu 0 \
    > logs/dino_synth_lp/launcher.log 2>&1 &
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from Train_and_downstream import run  # noqa: E402

DATASETS   = ["etth1", "etth2", "ettm1", "ettm2", "weather", "electricity", "traffic"]
LOG_FOLDER = ROOT / "logs" / "dino_synth_lp"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu",                type=int, default=0)
    p.add_argument("--epochs_forecasting", type=int, default=None)
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    LOG_FOLDER.mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        log_path = LOG_FOLDER / f"{dataset}.log"
        print(f"\n[{dataset}]  log={log_path.relative_to(ROOT)}", flush=True)

        with open(log_path, "w") as fh:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout = sys.stderr = fh
            try:
                run(
                    model              = "dino_timemixer",
                    task               = "forecast",
                    forecast_dataset   = dataset,
                    backbone_type      = "tsmixer",
                    encoder_layers     = 4,
                    out_dim            = 8192,
                    ckpt_tag           = "tsmixer",
                    checkpoints        = ["best"],
                    linear_probe       = True,          # <-- frozen backbone
                    epochs_forecasting = args.epochs_forecasting,
                )
            except Exception as e:
                print(f"ERROR: {e}", flush=True)
                import traceback; traceback.print_exc()
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr

        print(f"  → done", flush=True)


if __name__ == "__main__":
    main()

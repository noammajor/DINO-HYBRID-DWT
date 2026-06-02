#!/usr/bin/env python3
"""
run_lmc_training.py — Launcher for LMC pretraining.

Follows the same pattern as scripts/run_single.py:
  • sets CUDA_VISIBLE_DEVICES
  • redirects all output to logs/pretrain_lmc.log
  • delegates actual training to LabelTraining.py __main__

Usage
-----
python LabeledDataTraining/run_lmc_training.py \
    --data_dir /home/shared/datasets/TS_synthetic_labeled \
    --gpu 4
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT      = Path(__file__).resolve().parent.parent
LOGS_DIR  = ROOT / "logs"
TRAIN_SCRIPT = Path(__file__).resolve().parent / "LabelTraining.py"


def parse_args():
    p = argparse.ArgumentParser(description="Launch LMC pretraining with log redirect.")

    p.add_argument("--data_dir",   required=True,
                   help="Directory containing X.npy and Y.npy.")
    p.add_argument("--checkpoint", default=None,
                   help="Optional DINO .pth to warm-start the encoder.")
    p.add_argument("--gpu",        type=int, default=0)

    # pass-through hyperparams forwarded to LabelTraining.py
    p.add_argument("--c_in",          type=int,   default=None)
    p.add_argument("--seq_len",       type=int,   default=512)
    p.add_argument("--epochs",        type=int,   default=30)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--min_lr",        type=float, default=1e-5)
    p.add_argument("--batch_size",    type=int,   default=256)
    p.add_argument("--hidden_dim",    type=int,   default=64)
    p.add_argument("--freeze_backbone", type=lambda x: x.lower() != "false", default=False)
    p.add_argument("--min_latent",    type=int,   default=2)
    p.add_argument("--max_latent",    type=int,   default=10)
    p.add_argument("--val_frac",      type=float, default=0.05)
    p.add_argument("--test_frac",     type=float, default=0.05)
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--saveckp_freq",  type=int,   default=1)
    p.add_argument("--seed",          type=int,   default=42)

    return p.parse_args()


def main():
    args = parse_args()

    log_path = LOGS_DIR / "pretrain_lmc.log"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Build the command that LabelTraining.py __main__ will parse.
    cmd = [sys.executable, str(TRAIN_SCRIPT),
           "--data_dir",  args.data_dir,
           "--seq_len",   str(args.seq_len),
           "--epochs",    str(args.epochs),
           "--lr",        str(args.lr),
           "--min_lr",    str(args.min_lr),
           "--batch_size",str(args.batch_size),
           "--hidden_dim",str(args.hidden_dim),
           "--freeze_backbone", str(args.freeze_backbone),
           "--min_latent",str(args.min_latent),
           "--max_latent",str(args.max_latent),
           "--val_frac",  str(args.val_frac),
           "--test_frac", str(args.test_frac),
           "--num_workers",str(args.num_workers),
           "--saveckp_freq", str(args.saveckp_freq),
           "--seed",      str(args.seed),
           "--gpu",       str(args.gpu),
    ]
    if args.c_in:
        cmd += ["--c_in", str(args.c_in)]
    if args.checkpoint:
        cmd += ["--checkpoint", args.checkpoint]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print(f"[LMC pretrain]  GPU={args.gpu}  log={log_path.relative_to(ROOT)}")
    print(f"  {' '.join(cmd)}")

    with open(log_path, "w") as fh:
        fh.write(f"# started {datetime.now().isoformat(timespec='seconds')}\n\n")
        fh.flush()
        result = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)

    status = "OK" if result.returncode == 0 else f"FAILED (rc={result.returncode})"
    print(f"  → {status}")


if __name__ == "__main__":
    main()

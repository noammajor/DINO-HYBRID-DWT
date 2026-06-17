#!/usr/bin/env python3
"""
run_ctx_sweep.py — Sweep over context lengths (seq_len) for pretrain + forecasting.

Each context length gets its own pretrain + forecast cycle with a unique checkpoint
tag so runs never overwrite each other.

Context length is expressed in timesteps; the script converts to num_patches
internally (num_patches = ctx_len // patch_len, default patch_len=16).

Examples:
    # DINO, 4 context lengths, single dataset
    python scripts/run_ctx_sweep.py \\
        --model dino_timemixer --dataset etth1 --name dino_ctx \\
        --ctx_lens 96 192 336 720 \\
        --layers 4 --epochs 75 --lr 0.001 --gpu 0

    # DINO + MODWT augmentation
    python scripts/run_ctx_sweep.py \\
        --model dino_timemixer --dataset etth1 --name dino_modwt_ctx \\
        --ctx_lens 96 192 336 720 \\
        --ckpt_tag modwt --aug_global modwt_soft --aug_local modwt_hard \\
        --layers 4 --epochs 75 --lr 0.001 --gpu 0

    # All 4 ETT datasets, sequential
    for ds in etth1 etth2 ettm1 ettm2; do
        python scripts/run_ctx_sweep.py \\
            --model dino_timemixer --dataset $ds --name dino_ctx_${ds} \\
            --ctx_lens 96 192 336 720 --layers 4 --epochs 75 --gpu 0
    done
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

_python    = sys.executable
PATCH_LEN  = 16   # timesteps per patch — must match TSDiNO config

IN_DOMAIN_DATASETS = ["etth1", "etth2", "ettm1", "ettm2", "weather"]

MODEL_DEFAULT_LR = {
    "dino_timemixer": 5e-4,
    "dino_patchtst":  5e-4,
    "dino_ts2vec":   5e-4,
    "patchtst":   5e-5,
    "timemixer":  1e-4,
    "autoformer": 1e-4,
    "fedformer":  1e-4,
    "dlinear":    5e-3,
}


def _run(cmd: list, gpu: int, log_path: Path, dry_run: bool, label: str) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"\n[{label}]  GPU={gpu}  log={log_path.relative_to(ROOT)}")
    print(f"  {' '.join(str(c) for c in cmd)}")
    if dry_run:
        return 0
    with open(log_path, "w") as fh:
        fh.write(f"# started {datetime.now().isoformat(timespec='seconds')}\n\n")
        fh.flush()
        result = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
    status = "OK" if result.returncode == 0 else f"FAILED (rc={result.returncode})"
    print(f"  → {status}")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Sweep over context lengths for pretrain + forecasting"
    )
    parser.add_argument("--model",    required=True,
                        help="Model to run (dino_timemixer, dino_patchtst, patchtst, …)")
    parser.add_argument("--dataset",  required=True, choices=IN_DOMAIN_DATASETS,
                        help="Dataset to pretrain and forecast on")
    parser.add_argument("--name",     required=True,
                        help="Base run name — each ctx_len appends _ctx{N}")
    parser.add_argument("--ctx_lens", nargs="+", type=int, required=True,
                        help="Context lengths in timesteps, e.g. --ctx_lens 96 192 336 720")
    parser.add_argument("--patch_len", type=int, default=PATCH_LEN,
                        help=f"Patch size in timesteps (default: {PATCH_LEN})")

    # Architecture
    parser.add_argument("--layers",       type=int,   default=4)
    parser.add_argument("--embed_dim",    type=int,   default=None)
    parser.add_argument("--backbone_type", type=str,  default=None,
                        help="patchtst | tsmixer")

    # Training
    parser.add_argument("--epochs",       type=int,   default=None)
    parser.add_argument("--lr",           type=float, default=None)

    # Checkpoint / augmentation
    parser.add_argument("--ckpt_tag",    type=str,   default=None,
                        help="Base ckpt tag — _ctx{N} is appended automatically")
    parser.add_argument("--aug_global",  type=str,   default=None)
    parser.add_argument("--aug_local",   type=str,   default=None)
    parser.add_argument("--mlm_phi",     type=float, default=None)
    parser.add_argument("--mlm_mode",    type=str,   default=None,
                        help="mae | ibot")

    parser.add_argument("--gpu",     type=int, default=0)
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without running them")
    args = parser.parse_args()

    lr = args.lr or MODEL_DEFAULT_LR.get(args.model, 5e-4)

    for ctx_len in args.ctx_lens:
        if ctx_len % args.patch_len != 0:
            print(f"WARNING: ctx_len={ctx_len} is not divisible by patch_len={args.patch_len}, skipping.")
            continue

        num_patches = ctx_len // args.patch_len
        base_tag    = args.ckpt_tag or ""
        ctx_tag     = f"{base_tag}_ctx{ctx_len}" if base_tag else f"ctx{ctx_len}"
        log_base    = ROOT / "logs" / "ctx_sweep" / f"{args.name}_ctx{ctx_len}"

        print(f"\n{'='*60}")
        print(f"  ctx_len={ctx_len}  num_patches={num_patches}  tag={ctx_tag}")
        print(f"{'='*60}")

        base_cmd = [
            _python, str(ROOT / "Train_and_downstream.py"),
            "--model",          args.model,
            "--lr",             str(lr),
            "--encoder_layers", str(args.layers),
            "--ckpt_tag",       ctx_tag,
            "--num_patches",    str(num_patches),
        ]
        if args.embed_dim:
            base_cmd += ["--embed_dim", str(args.embed_dim)]
        if args.backbone_type:
            base_cmd += ["--backbone_type", args.backbone_type]
        if args.epochs:
            base_cmd += ["--epochs", str(args.epochs)]
        if args.aug_global:
            base_cmd += ["--aug_global", args.aug_global]
        if args.aug_local:
            base_cmd += ["--aug_local", args.aug_local]
        if args.mlm_phi is not None:
            base_cmd += ["--mlm_phi", str(args.mlm_phi)]
        if args.mlm_mode:
            base_cmd += ["--mlm_mode", args.mlm_mode]

        # pretrain
        rc = _run(
            base_cmd + [
                "--pretrain_only",    "true",
                "--pretrain_dataset", args.dataset,
                "--forecast_dataset", args.dataset,
            ],
            args.gpu,
            log_base / "pretrain.log",
            args.dry_run,
            f"pretrain/{args.model}/{args.dataset}/ctx{ctx_len}",
        )
        if rc != 0:
            print(f"\nPretraining failed for ctx_len={ctx_len} — skipping forecast.")
            continue

        # forecast
        rc = _run(
            base_cmd + [
                "--task",             "forecast",
                "--pretrain_dataset", args.dataset,
                "--forecast_dataset", args.dataset,
            ],
            args.gpu,
            log_base / "forecast.log",
            args.dry_run,
            f"forecast/{args.model}/{args.dataset}/ctx{ctx_len}",
        )
        if rc != 0:
            print(f"\nForecasting failed for ctx_len={ctx_len}.")

    print(f"\n{'='*60}")
    print(f"  Sweep done.  Logs: logs/ctx_sweep/{args.name}_ctx*/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

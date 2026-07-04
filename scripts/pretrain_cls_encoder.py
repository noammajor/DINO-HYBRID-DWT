#!/usr/bin/env python3
"""
pretrain_cls_encoder.py — Pre-train 8-layer encoders with 1152-timestep context for classification.

The standard pre-training uses short context windows aligned with forecasting (336–512 ts).
This script pre-trains with num_patches=72 (72 × 16 = 1152 timesteps), covering the full
length of the longest UEA classification datasets (SelfRegulationSCP2 T=1152).

Checkpoint locations:
  dino_timemixer → checkpoints_layers8_cw1152/
  dino_patchtst  → checkpoints_patchtst_layers8_cw1152/
  patchtst       → PatchTST_self_supervised/saved_models/  (context_points=1152)

Usage:
    python pretrain_cls_encoder.py
    python pretrain_cls_encoder.py --models dino_timemixer --ckpt_tag tsmixer
    python pretrain_cls_encoder.py --models dino_timemixer dino_patchtst patchtst
    python pretrain_cls_encoder.py --pretrain_source monash+synthetic
    python pretrain_cls_encoder.py --gpu_override 2
    python pretrain_cls_encoder.py --dry_run
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

DEFAULT_ENCODER_LAYERS   = 8
DEFAULT_PREDICTOR_LAYERS = 4       # half of encoder layers
DEFAULT_NUM_PATCHES      = 72      # 72 × 16 = 1152 timesteps
DEFAULT_PATCH_SIZE       = 16

# Per-model LR (same as run_layer_sweep.py for 8-layer configs)
MODEL_LR = {
    "dino_timemixer": 5e-4,
    "dino_patchtst":  5e-4,
    "dino_ts2vec":   5e-4,
    "patchtst":       5e-5,
}

MODEL_GPU = {
    "dino_timemixer": 0,
    "dino_patchtst":  1,
    "dino_ts2vec":    2,
    "patchtst":       3,
}

ALL_MODELS = list(MODEL_GPU.keys())


def launch_model(model: str, gpu: int, pretrain_source: str,
                 log_dir: Path, dry_run: bool,
                 encoder_layers: int, predictor_layers: int,
                 num_patches: int, patch_size: int,
                 log_tag: str = "",
                 ckpt_tag: str = None,
                 mlm_phi: float = None,
                 mlm_mode: str = None,
                 subset_frac: float = None):
    lr = MODEL_LR[model]
    cw = num_patches * patch_size
    ckpt_suffix = f"_{ckpt_tag}" if ckpt_tag else ""
    tag_suffix = f"_{log_tag}" if log_tag else ""
    log_path = log_dir / f"{model}{ckpt_suffix}_layers{encoder_layers}_{pretrain_source}_cw{cw}{tag_suffix}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(ROOT / "Train_and_downstream.py"),
        "--model",            model,
        "--pretrain_only",    "true",
        "--encoder_layers",   str(encoder_layers),
        "--predictor_layers", str(predictor_layers),
        "--num_patches",      str(num_patches),
        "--lr",               str(lr),
        "--pretrain_source",  pretrain_source,
    ]
    if ckpt_tag is not None and model.startswith("dino"):
        cmd += ["--ckpt_tag", ckpt_tag]
    if mlm_phi is not None and model.startswith("dino"):
        cmd += ["--mlm_phi", str(mlm_phi)]
    if mlm_mode is not None and model.startswith("dino"):
        cmd += ["--mlm_mode", mlm_mode]
    if subset_frac is not None:
        cmd += ["--subset_frac", str(subset_frac)]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    ckpt_info = f"  ckpt_tag={ckpt_tag}" if ckpt_tag and model.startswith("dino") else ""
    mlm_info  = f"  mlm_phi={mlm_phi}"  if mlm_phi  is not None and model.startswith("dino") else ""
    mlm_info += f"  mlm_mode={mlm_mode}" if mlm_mode is not None and model.startswith("dino") else ""
    print(f"  [{model:12s}] GPU={gpu}  layers={encoder_layers}  "
          f"num_patches={num_patches}  cw={cw}  lr={lr}{ckpt_info}{mlm_info}"
          f"  log={log_path.relative_to(ROOT)}")

    if dry_run:
        print(f"    CMD: {' '.join(cmd)}")
        return None

    fh = open(log_path, "w")
    fh.write(f"# started {datetime.now().isoformat(timespec='seconds')}\n\n")
    proc = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
    proc._log_fh = fh
    proc._label  = model
    return proc


def main():
    parser = argparse.ArgumentParser(
        description="Pre-train encoders with a long context window for classification"
    )
    parser.add_argument("--models", nargs="+", default=ALL_MODELS,
                        choices=ALL_MODELS, metavar="MODEL",
                        help=f"Models to train (default: all). Choices: {ALL_MODELS}")
    parser.add_argument("--pretrain_source", type=str, default="synthetic",
                        choices=["monash", "synthetic", "monash+synthetic"],
                        help="Pre-training data source (default: synthetic)")
    parser.add_argument("--encoder_layers",   type=int, default=DEFAULT_ENCODER_LAYERS,
                        help=f"Encoder depth (default: {DEFAULT_ENCODER_LAYERS})")
    parser.add_argument("--predictor_layers", type=int, default=DEFAULT_PREDICTOR_LAYERS,
                        help=f"Predictor depth (default: {DEFAULT_PREDICTOR_LAYERS})")
    parser.add_argument("--num_patches",      type=int, default=DEFAULT_NUM_PATCHES,
                        help=f"Number of patches in the context window (default: {DEFAULT_NUM_PATCHES})")
    parser.add_argument("--patch_size",       type=int, default=DEFAULT_PATCH_SIZE,
                        help=f"Patch size (default: {DEFAULT_PATCH_SIZE}). Used for log naming + cw display; "
                             "actual patch size is set by each model's config file.")
    parser.add_argument("--gpu_override", type=int, default=None,
                        help="Run the task on this GPU (overrides per-model assignment).")
    parser.add_argument("--ckpt_tag", type=str, default=None,
                        help="Tag appended to the checkpoint dir name, e.g. 'tsmixer' → checkpoints_synthetic_layers8_tsmixer/ (dino only)")
    parser.add_argument("--log_tag", type=str, default="",
                        help="Extra suffix for the log filename only.")
    parser.add_argument("--mlm_phi", type=float, default=None,
                        help="MLM/iBOT loss weight for DINO (0.0 = pure DINO loss, default: from config)")
    parser.add_argument("--subset_frac", type=float, default=None,
                        help="Fraction of pretraining data to use, e.g. 0.5 for 50%% (default: all)")
    parser.add_argument("--mlm_mode", type=str, default=None,
                        help="MLM variant: ibot (teacher-guided CE) or mae (MSE vs ground truth) (dino only)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without running them")
    args = parser.parse_args()

    cw = args.num_patches * args.patch_size
    log_dir = ROOT / "logs" / "cls_encoder_pretrain"

    print("=" * 60)
    print(f"  Classification encoder pre-training")
    print(f"  encoder_layers  : {args.encoder_layers}")
    print(f"  predictor_layers: {args.predictor_layers}")
    print(f"  num_patches     : {args.num_patches}  (context = {cw} timesteps)")
    print(f"  patch_size      : {args.patch_size}")
    print(f"  models          : {args.models}")
    print(f"  pretrain_source : {args.pretrain_source}")
    if args.ckpt_tag:
        print(f"  ckpt_tag        : {args.ckpt_tag}  (dino only)")
    if args.mlm_phi is not None:
        print(f"  mlm_phi         : {args.mlm_phi}  (dino only)")
    if args.dry_run:
        print("  DRY RUN")
    print("=" * 60 + "\n")

    procs = []
    for model in args.models:
        gpu = args.gpu_override if args.gpu_override is not None else MODEL_GPU[model]
        proc = launch_model(model, gpu, args.pretrain_source, log_dir, args.dry_run,
                            encoder_layers=args.encoder_layers,
                            predictor_layers=args.predictor_layers,
                            num_patches=args.num_patches,
                            patch_size=args.patch_size,
                            log_tag=args.log_tag,
                            ckpt_tag=args.ckpt_tag,
                            mlm_phi=args.mlm_phi,
                            mlm_mode=args.mlm_mode,
                            subset_frac=args.subset_frac)
        if proc is not None:
            procs.append(proc)

    if not procs:
        return

    print(f"\nWaiting for {len(procs)} workers …")
    failed = []
    for proc in procs:
        proc.wait()
        proc._log_fh.close()
        status = "OK" if proc.returncode == 0 else f"FAILED (rc={proc.returncode})"
        print(f"  {proc._label}: {status}")
        if proc.returncode != 0:
            failed.append(proc._label)

    if failed:
        print(f"\nWARNING: {len(failed)} failed: {failed}")
    else:
        print(f"\nAll {len(procs)} workers completed.")


if __name__ == "__main__":
    main()

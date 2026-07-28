#!/usr/bin/env python3
"""
run_single.py — Pretrain a single model on one in-domain dataset, then forecast.

Saves the best checkpoint to:
    {project_root}/Models/{name}/best_chkp_{dataset}.pt

Then runs in-domain forecasting (pred_lens 96/192/336/720) on the same dataset.

Usage:
    python scripts/run_single.py --model dino_timemixer --dataset etth1 --name my_run
    python scripts/run_single.py --model patchtst --dataset weather \\
        --name ptst_w8 --layers 8 --embed_dim 256 --lr 1e-4 --gpu 0
    python scripts/run_single.py --model patchtst --dataset ettm2 --name ptst_test \\
        --skip_pretrain   # skip training, jump straight to forecasting
"""

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

IN_DOMAIN_DATASETS = ["etth1", "etth2", "ettm1", "ettm2", "weather",
                      "electricity", "traffic",
                      "exchange", "wind", "solar", "metr_la", "aqwan", "aqshunyi",
                      "czelan", "zafnoo", "pm2_5", "temp"]
ALL_MODELS         = ["dino_timemixer", "dino_patchtst", "dino_itransformer", "patchtst",
                      "timemixer", "autoformer", "fedformer", "dlinear"]

# Supervised models train directly on forecasting — no pretrain phase, no checkpoint copy.
SUPERVISED_MODELS  = {"timemixer", "autoformer", "fedformer", "dlinear"}

MODEL_DEFAULT_LR = {
    "dino_timemixer": 5e-4,
    "dino_patchtst":  5e-4,
    "dino_ts2vec":   5e-4,
    "patchtst":       5e-5,
    "timemixer":      1e-4,
    "autoformer":     1e-4,
    "fedformer":      1e-4,
    "dlinear":        5e-3,
}

_python = sys.executable


# ── checkpoint locator ────────────────────────────────────────────────────────

def _find_src_checkpoint(model: str, dataset: str, layers: int, out_dim: int = None, ckpt_tag: str = None) -> Path:
    """Return the path where Train_and_downstream.py saves the best checkpoint."""
    if model in ("dino_timemixer", "dino_patchtst", "dino_ts2vec"):
        _base       = {"dino_timemixer": "checkpoints", "dino_patchtst": "checkpoints_patchtst", "dino_ts2vec": "checkpoints_ts2vec"}[model]
        _outdim_tag = f"_outdim{out_dim}" if out_dim is not None else ''
        _ckpt_tag   = f"_{ckpt_tag}" if ckpt_tag else ''
        return ROOT / f"{_base}_{dataset}_layers{layers}{_outdim_tag}{_ckpt_tag}" / "checkpoint_best.pth"

    elif model == "patchtst":
        save_dir = (ROOT / "models" / "PatchTST_self_supervised" / "saved_models" /
                    dataset / "masked_patchtst" / "based_model" / f"layers{layers}")
        candidates = [p for p in save_dir.glob("*.pth") if "_epoch" not in p.name]
        return candidates[0] if candidates else save_dir / "checkpoint_best.pth"

    else:
        raise ValueError(f"Unknown model: {model}")


# ── subprocess launcher ───────────────────────────────────────────────────────

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


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Single-run in-domain pretrain + forecasting with a unified checkpoint path"
    )
    parser.add_argument("--model",    required=True, choices=ALL_MODELS,
                        help="Model to run")
    parser.add_argument("--dataset",  required=True, choices=IN_DOMAIN_DATASETS,
                        help="Dataset to pretrain and forecast on")
    parser.add_argument("--name",     required=True,
                        help="Run name — checkpoint saved as Models/{name}/best_chkp_{dataset}.pt")
    parser.add_argument("--layers",   type=int, default=8,
                        help="Number of encoder layers (default: 8)")
    parser.add_argument("--embed_dim", type=int, default=None,
                        help="Embedding dim / d_model override")
    parser.add_argument("--out_dim", type=int, default=None,
                        help="DINO output bins / prototype count (DINO only)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of pretraining epochs")
    parser.add_argument("--epochs_forecasting", type=int, default=None,
                        help="Number of forecasting fine-tune epochs (DINO)")
    parser.add_argument("--checkpoints", nargs="+", default=None,
                        help="Checkpoint epochs to evaluate during forecasting, e.g. --checkpoints 1 3 5 10")
    parser.add_argument("--lr",       type=float, default=None,
                        help="Pretraining LR (default: model-specific)")
    parser.add_argument("--lr_forecasting", type=float, default=None,
                        help="Forecast fine-tune LR (head_lr=encoder_lr; DINO)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Pretrain batch size override (per-run; e.g. small for high-channel datasets)")
    parser.add_argument("--warmup_epochs", type=int, default=None,
                        help="Number of LR warmup epochs (DINO only)")
    parser.add_argument("--ckpt_tag", type=str, default=None,
                        help="Extra tag appended to checkpoint directory name (e.g. 'wrLR')")
    parser.add_argument("--aug_global", type=str, default=None,
                        help="Global (teacher) augmentation type, overrides config")
    parser.add_argument("--aug_local",  type=str, default=None,
                        help="Local (student) augmentation type, overrides config")
    parser.add_argument("--n_global_crops", type=int, default=None,
                        help="Number of global (teacher) crops — DINO multi-crop")
    parser.add_argument("--n_local_crops",  type=int, default=None,
                        help="Number of local (student) crops — DINO multi-crop")
    parser.add_argument("--global_crop_ratio", type=float, default=None,
                        help="Crop ratio for global crops (1.0 = no crop)")
    parser.add_argument("--local_crop_ratio",  type=float, default=None,
                        help="Crop ratio for local crops (e.g. 0.4)")
    parser.add_argument("--dwt_level", type=int, default=None,
                        help="DWT decomposition depth J (e.g. 2/3/4)")
    parser.add_argument("--batch_size_forecast", type=int, default=None,
                        help="Forecast batch size override (e.g. lower for electricity)")
    parser.add_argument("--dwt_wavelet_pool", nargs="+", default=None,
                        help="DWT wavelet pool for augmentation, e.g. sym4 sym6 sym8 db4 db6 coif2 "
                             "(DINO only; 'full' family = the 6 listed)")
    parser.add_argument("--soft_threshold_sigma", type=float, default=None,
                        help="ρ (shrinkage ratio) for soft-threshold DWT/SWT/MODWT aug "
                             "(config default 0.6). Only affects *_soft_threshold aug types.")
    parser.add_argument("--mlm_phi",    type=float, default=None,
                        help="MLM weight: phi*DINO+(1-phi)*MLM (DINO only)")
    parser.add_argument("--mlm_mode",   type=str,   default=None,
                        help="MLM variant: ibot (teacher-guided CE) or mae (MSE vs ground truth)")
    parser.add_argument("--backbone_type", type=str, default=None,
                        help="patchtst | tsmixer — overrides config (default: patchtst)")
    parser.add_argument("--gpu",      type=int, default=0,
                        help="GPU index via CUDA_VISIBLE_DEVICES (default: 0)")
    parser.add_argument("--seed",     type=int, default=None,
                        help="Random seed")
    parser.add_argument("--skip_pretrain", action="store_true",
                        help="Skip pretraining, go straight to forecasting")
    parser.add_argument("--linear_probe", type=str, default=None, choices=["true", "false"],
                        help="DINO forecast mode: true=probe (backbone frozen), false=full fine-tune. "
                             "Default None keeps Train_and_downstream's default (true).")
    parser.add_argument("--dry_run",       action="store_true",
                        help="Print commands without executing them")
    args = parser.parse_args()

    lr          = args.lr or MODEL_DEFAULT_LR[args.model]
    target_dir  = ROOT / "Models" / args.name
    target_ckpt = target_dir / f"best_chkp_{args.dataset}.pt"
    log_base    = ROOT / "logs" / "single" / args.name

    print(f"\n{'='*60}")
    print(f"  model:     {args.model}")
    print(f"  dataset:   {args.dataset}")
    print(f"  layers:    {args.layers}" +
          (f"   embed_dim: {args.embed_dim}" if args.embed_dim else ""))
    print(f"  lr:        {lr}")
    print(f"  gpu:       {args.gpu}")
    print(f"  ckpt  →    {target_ckpt}")
    if args.skip_pretrain:
        print(f"  [skip_pretrain]")
    if args.dry_run:
        print(f"  [dry_run]")
    print(f"{'='*60}")

    # base flags shared by both pretrain and forecast calls
    # Supervised models (e.g. timemixer) manage their own architecture via their
    # own config — don't override encoder_layers unless the user set it explicitly.
    base_cmd = [
        _python, str(ROOT / "Train_and_downstream.py"),
        "--model", args.model,
        "--lr",    str(lr),
    ]
    if args.model not in SUPERVISED_MODELS or args.layers != 8:
        base_cmd += ["--encoder_layers", str(args.layers)]
    if args.embed_dim:
        base_cmd += ["--embed_dim", str(args.embed_dim)]
    if args.batch_size:
        base_cmd += ["--batch_size", str(args.batch_size)]
    if args.out_dim:
        base_cmd += ["--out_dim", str(args.out_dim)]
    if args.epochs:
        base_cmd += ["--epochs", str(args.epochs)]
    if args.epochs_forecasting:
        base_cmd += ["--epochs_forecasting", str(args.epochs_forecasting)]
    if args.lr_forecasting is not None:
        base_cmd += ["--lr_forecasting", str(args.lr_forecasting)]
    if args.seed is not None:
        base_cmd += ["--seed", str(args.seed)]
    if args.warmup_epochs is not None:
        base_cmd += ["--warmup_epochs", str(args.warmup_epochs)]
    if args.ckpt_tag is not None:
        base_cmd += ["--ckpt_tag", args.ckpt_tag]
    if args.aug_global is not None:
        base_cmd += ["--aug_global", args.aug_global]
    if args.aug_local is not None:
        base_cmd += ["--aug_local", args.aug_local]
    if args.n_global_crops is not None:
        base_cmd += ["--n_global_crops", str(args.n_global_crops)]
    if args.n_local_crops is not None:
        base_cmd += ["--n_local_crops", str(args.n_local_crops)]
    if args.global_crop_ratio is not None:
        base_cmd += ["--global_crop_ratio", str(args.global_crop_ratio)]
    if args.local_crop_ratio is not None:
        base_cmd += ["--local_crop_ratio", str(args.local_crop_ratio)]
    if args.dwt_wavelet_pool is not None:
        base_cmd += ["--dwt_wavelet_pool"] + list(args.dwt_wavelet_pool)
    if args.dwt_level is not None:
        base_cmd += ["--dwt_level", str(args.dwt_level)]
    if args.batch_size_forecast is not None:
        base_cmd += ["--batch_size_forecast", str(args.batch_size_forecast)]
    if args.soft_threshold_sigma is not None:
        base_cmd += ["--soft_threshold_sigma", str(args.soft_threshold_sigma)]
    if args.mlm_phi is not None:
        base_cmd += ["--mlm_phi", str(args.mlm_phi)]
    if args.mlm_mode is not None:
        base_cmd += ["--mlm_mode", args.mlm_mode]
    if args.backbone_type is not None:
        base_cmd += ["--backbone_type", args.backbone_type]

    # ── pretrain ──────────────────────────────────────────────────────────────
    _skip_pretrain = args.skip_pretrain or (args.model in SUPERVISED_MODELS)
    if not _skip_pretrain:
        rc = _run(
            base_cmd + [
                "--pretrain_only",    "true",
                "--pretrain_dataset", args.dataset,
                "--forecast_dataset", args.dataset,
            ],
            args.gpu,
            log_base / "pretrain.log",
            args.dry_run,
            f"pretrain/{args.model}/{args.dataset}",
        )
        if rc != 0:
            print("\nPretraining failed — aborting.")
            sys.exit(rc)

    # ── copy best checkpoint to unified path ──────────────────────────────────
    if args.model not in SUPERVISED_MODELS:
        src_ckpt = _find_src_checkpoint(args.model, args.dataset, args.layers, args.out_dim, args.ckpt_tag)
        if not args.dry_run:
            if src_ckpt.exists():
                target_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_ckpt, target_ckpt)
                print(f"\nCheckpoint → {target_ckpt}")
            else:
                print(f"\nWarning: expected checkpoint not found at {src_ckpt}")
        else:
            print(f"\n[dry_run] copy  {src_ckpt}\n         →     {target_ckpt}")

    # ── forecast ──────────────────────────────────────────────────────────────
    # Supervised models train directly on the forecasting task — don't pass
    # --task forecast (which sets skip_train=True and skips all training).
    if args.model in SUPERVISED_MODELS:
        forecast_cmd = base_cmd + [
            "--forecast_dataset", args.dataset,
            "--pretrain_dataset", args.dataset,
        ]
    else:
        forecast_cmd = base_cmd + [
            "--task",             "forecast",
            "--pretrain_dataset", args.dataset,
            "--forecast_dataset", args.dataset,
        ]
        if args.linear_probe is not None:
            forecast_cmd += ["--linear_probe", args.linear_probe]
    if args.checkpoints:
        forecast_cmd += ["--checkpoints"] + [str(c) for c in args.checkpoints]

    rc = _run(
        forecast_cmd,
        args.gpu,
        log_base / "forecast.log",
        args.dry_run,
        f"forecast/{args.model}/{args.dataset}",
    )
    if rc != 0:
        print("\nForecasting failed.")
        sys.exit(rc)

    print(f"\n{'='*60}")
    print(f"  Done.")
    print(f"  Checkpoint: {target_ckpt}")
    print(f"  Logs:       {log_base.relative_to(ROOT)}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

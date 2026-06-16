#!/usr/bin/env python3
"""
run_aug_family_grid.py — DWT-aug × objective × dataset grid (in-domain).

Grid:
    objectives  : dino | ibot (DINO+iBOT) | mae (DINO+MAE)
    families    : db | full | zero   (DWT augmentation config — wavelet pool / aug type)
    datasets    : etth1 etth2 ettm1 ettm2 weather electricity

Layout (matches "one GPU per family, 3 models on it"):
    * one GPU per FAMILY (3 GPUs total)
    * on each GPU the 3 OBJECTIVES run concurrently (3 threads)
    * within each objective the DATASETS run sequentially (pretrain → forecast)

So with 6 datasets that's 3 families × 3 objectives × 6 datasets = 54 pipelines.

Logs:
    logs/{root}/{family}/{dataset}/{objective}_pretrain.log
    logs/{root}/{family}/{dataset}/{objective}_forecast.log

Checkpoints (Train_and_downstream naming):
    checkpoints_{dataset}_layers{L}_{root}_{family}_{objective}/checkpoint_best.pth

Usage:
    python scripts/run_aug_family_grid.py --root aug_grid --gpus 3 4 5
    python scripts/run_aug_family_grid.py --root aug_grid --gpus 3 4 5 --dry_run
    python scripts/run_aug_family_grid.py --root aug_grid --gpus 3 4 5 \\
        --datasets etth1 etth2 --objectives dino ibot
"""

import argparse
import os
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

# ── grid definition ────────────────────────────────────────────────────────────
BACKBONE       = "tsmixer"
ENCODER_LAYERS = 4
EPOCHS         = 80
LR             = 5e-4
MLM_PHI        = 0.6          # weight for DINO+iBOT / DINO+MAE blends (phi*DINO + (1-phi)*MLM)
SEED           = 42          # fixed seed for all runs (overridable via --seed); also tags checkpoints _seedN

DATASETS = ["etth1", "etth2", "ettm1", "ettm2", "weather", "electricity"]

# Objective → extra flags. dino = pure DINO (MLM off).
OBJECTIVES = {
    "dino": [],
    "ibot": ["--mlm_phi", str(MLM_PHI), "--mlm_mode", "ibot"],
    "mae":  ["--mlm_phi", str(MLM_PHI), "--mlm_mode", "mae"],
}

# Family → DWT augmentation flags. Teacher view stays on the stable default
# (dwt_soft_threshold) for all; only the wavelet pool / student view changes.
FAMILIES = {
    "db":   ["--dwt_wavelet_pool", "db4", "db6", "db8"],
    "full": ["--dwt_wavelet_pool", "sym4", "sym6", "sym8", "db4", "db6", "coif2"],
    "zero": ["--dwt_wavelet_pool", "db6", "--aug_local", "dwt_zero_out_detail"],
}

FAMILY_ORDER    = ["db", "full", "zero"]   # GPU i ← FAMILY_ORDER[i]
OBJECTIVE_ORDER = ["dino", "ibot", "mae"]

_python = str(Path(sys.executable).parent / "python")


# ── command builders ───────────────────────────────────────────────────────────

def _common_flags(objective: str, family: str, dataset: str, ckpt_tag: str) -> list:
    return (
        ["--model", "dino",
         "--backbone_type", BACKBONE,
         "--encoder_layers", str(ENCODER_LAYERS),
         "--seed", str(SEED),
         "--ckpt_tag", ckpt_tag]
        + OBJECTIVES[objective]
        + FAMILIES[family]
    )


def pretrain_cmd(objective, family, dataset, ckpt_tag) -> list:
    return [
        _python, str(ROOT / "Train_and_downstream.py"),
        "--pretrain_only", "true",
        "--pretrain_dataset", dataset,
        "--forecast_dataset", dataset,
        "--epochs", str(EPOCHS),
        "--lr", str(LR),
    ] + _common_flags(objective, family, dataset, ckpt_tag)


def forecast_cmd(objective, family, dataset, ckpt_tag) -> list:
    # Forecast loads the pretrained checkpoint (matched by ckpt_tag/layers/backbone).
    # Aug/objective flags don't affect the forecast model but are harmless to pass.
    return [
        _python, str(ROOT / "Train_and_downstream.py"),
        "--task", "forecast",
        "--pretrain_dataset", dataset,
        "--forecast_dataset", dataset,
    ] + _common_flags(objective, family, dataset, ckpt_tag)


# ── subprocess launcher ────────────────────────────────────────────────────────

def _run(cmd, gpu, log_path: Path, dry_run, label) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"  [{label}] GPU={gpu}  log={log_path.relative_to(ROOT)}")
    if dry_run:
        print(f"    CMD: {' '.join(cmd)}")
        return 0
    with open(log_path, "w") as fh:
        fh.write(f"# started {datetime.now().isoformat(timespec='seconds')}\n")
        fh.write(f"# {' '.join(cmd)}\n\n")
        fh.flush()
        result = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
    status = "OK" if result.returncode == 0 else f"FAILED (rc={result.returncode})"
    print(f"    {label}: {status}")
    return result.returncode


# ── per-(family,objective) pipeline — runs in its own thread ────────────────────

def _pipeline(objective, family, gpu, datasets, root, skip_pretrain, dry_run):
    log_base = ROOT / "logs" / root / family
    for dataset in datasets:
        ckpt_tag = f"{root}_{family}_{objective}"
        if not skip_pretrain:
            rc = _run(pretrain_cmd(objective, family, dataset, ckpt_tag), gpu,
                      log_base / dataset / f"{objective}_pretrain.log", dry_run,
                      f"pretrain/{family}/{objective}/{dataset}")
            if rc != 0:
                print(f"    !! {family}/{objective}/{dataset} pretrain failed — skipping forecast")
                continue
        _run(forecast_cmd(objective, family, dataset, ckpt_tag), gpu,
             log_base / dataset / f"{objective}_forecast.log", dry_run,
             f"forecast/{family}/{objective}/{dataset}")
    print(f"  [{family}/{objective}] pipeline complete.")


# ── main ────────────────────────────────────────────────────────────────────────

def main():
    global SEED
    p = argparse.ArgumentParser(description="DWT-aug × objective × dataset in-domain grid")
    p.add_argument("--root", required=True, help="Root tag for logs/checkpoints folder")
    p.add_argument("--gpus", nargs="+", type=int, required=True,
                   help=f"One GPU per family, in order {FAMILY_ORDER} (e.g. --gpus 3 4 5)")
    p.add_argument("--families",   nargs="+", default=FAMILY_ORDER,    choices=FAMILY_ORDER)
    p.add_argument("--objectives", nargs="+", default=OBJECTIVE_ORDER, choices=OBJECTIVE_ORDER)
    p.add_argument("--datasets",   nargs="+", default=DATASETS)
    p.add_argument("--seed",       type=int, default=SEED)
    p.add_argument("--skip_pretrain", action="store_true")
    p.add_argument("--dry_run",       action="store_true")
    args = p.parse_args()
    SEED = args.seed

    if len(args.gpus) != len(args.families):
        p.error(f"--gpus ({len(args.gpus)}) must match number of families ({len(args.families)})")
    fam_gpu = dict(zip(args.families, args.gpus))

    print(f"\n{'='*60}")
    print(f"  AUG-FAMILY GRID   root={args.root}")
    print(f"  backbone={BACKBONE}  layers={ENCODER_LAYERS}  epochs={EPOCHS}  lr={LR}  mlm_phi={MLM_PHI}  seed={SEED}")
    print(f"  families→gpu: {fam_gpu}")
    print(f"  objectives:   {args.objectives}   (concurrent per GPU)")
    print(f"  datasets:     {args.datasets}     (sequential per objective)")
    print(f"  total pipelines: {len(args.families)*len(args.objectives)*len(args.datasets)}")
    if args.dry_run:
        print("  [DRY RUN]")
    print(f"{'='*60}\n")

    threads = []
    for family in args.families:
        for objective in args.objectives:
            t = threading.Thread(
                target=_pipeline,
                args=(objective, family, fam_gpu[family], args.datasets,
                      args.root, args.skip_pretrain, args.dry_run),
                name=f"{family}/{objective}", daemon=True,
            )
            threads.append(t)
            t.start()

    for t in threads:
        t.join()

    print(f"\nGrid complete. Logs in logs/{args.root}/")


if __name__ == "__main__":
    main()

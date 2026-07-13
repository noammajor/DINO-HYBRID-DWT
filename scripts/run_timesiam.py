#!/usr/bin/env python3
"""
run_timesiam.py — run TimeSiam SSL (pretrain -> fine-tune forecasting) on OUR
datasets, at OUR context length, so the numbers are comparable to WINO-TS.

TimeSiam already uses the standard TSLib data loaders (Dataset_ETT_hour /
Dataset_ETT_minute / Dataset_Custom) with the exact same split borders as our
forecast pipeline. So "using ours" == pointing TimeSiam's data_provider at our
CSV directory (DATA_PATHS['forecasting_data_dir']) at seq_len 336. This script
wires that up and drives the two-stage TimeSiam workflow per dataset.

Usage
-----
  python scripts/run_timesiam.py --datasets etth1 etth2 ettm1 ettm2 weather --gpu 0
  python scripts/run_timesiam.py --datasets all --backbone iTransformer --gpu 0
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
TS   = ROOT / "TimeSiam-main"
sys.path.insert(0, str(ROOT))
from data_paths import DATA_PATHS
from dataset_registry import get_dataset_info

DATA_DIR = DATA_PATHS["forecasting_data_dir"]
PRED_LENS = [96, 192, 336, 720]

# our dataset name -> TimeSiam --data key (ETT get their dedicated split classes;
# everything else uses the generic 'custom' -> Dataset_Custom we added).
ETT_KEY = {"etth1": "ETTh1", "etth2": "ETTh2", "ettm1": "ETTm1", "ettm2": "ETTm2"}
# ETTh = hourly, ETTm = 15-min, others hourly by default (time-features only).
FREQ = {"etth1": "h", "etth2": "h", "ettm1": "t", "ettm2": "t"}

ALL_DATASETS = ["etth1", "etth2", "ettm1", "ettm2", "weather", "electricity",
                "exchange", "solar", "traffic", "aqshunyi", "aqwan", "czelan", "pm2_5"]


def _ts_key(ds):
    return ETT_KEY.get(ds, "custom")


def _run(cmd, log_path, dry):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("  CMD:", " ".join(cmd))
    print("  LOG:", log_path)
    if dry:
        return 0
    with open(log_path, "w") as fh:
        p = subprocess.Popen(cmd, cwd=str(TS), stdout=fh, stderr=subprocess.STDOUT)
        return p.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["etth1", "etth2", "ettm1", "ettm2", "weather"],
                    help="dataset names (from dataset_registry), or 'all'")
    ap.add_argument("--backbone", default="iTransformer",
                    help="TimeSiam encoder backbone (iTransformer, PatchTST, DLinear, ...)")
    ap.add_argument("--seq_len", type=int, default=None,
                    help="context length; default = our num_patches*patch_len (336)")
    ap.add_argument("--pretrain_epochs", type=int, default=50)
    ap.add_argument("--e_layers", type=int, default=1)
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--d_ff", type=int, default=512)
    ap.add_argument("--mask_rate", type=float, default=0.25)
    ap.add_argument("--sampling_range", type=int, default=6)
    ap.add_argument("--lineage_tokens", type=int, default=2)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()

    datasets = ALL_DATASETS if a.datasets == ["all"] else a.datasets
    # default context length = our 336 (21 patches x 16)
    seq_len = a.seq_len or 21 * 16

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(a.gpu)  # applied via the child; also passed below

    for ds in datasets:
        info = get_dataset_info(ds)
        csv  = info["csv_filename"]
        c_in = info["c_in"]
        key  = _ts_key(ds)
        freq = FREQ.get(ds, "h")
        mid  = f"{ds}_{a.backbone}_sl{seq_len}"          # unique checkpoint id
        logdir = ROOT / "logs" / "timesiam" / ds
        ckpt = f"./outputs/pretrain_checkpoints/{mid}/ckpt_best.pth"

        common = [
            "--root_path", DATA_DIR + ("" if DATA_DIR.endswith("/") else "/"),
            "--data_path", csv, "--model_id", mid, "--model", a.backbone,
            "--data", key, "--features", "M", "--freq", freq,
            "--seq_len", str(seq_len),
            "--enc_in", str(c_in), "--dec_in", str(c_in), "--c_out", str(c_in),
            "--e_layers", str(a.e_layers), "--d_model", str(a.d_model), "--d_ff", str(a.d_ff),
            "--gpu", str(a.gpu),
        ]

        print(f"\n{'='*70}\n  TimeSiam | {ds} | {a.backbone} | c_in={c_in} | seq_len={seq_len} | data={key}\n{'='*70}")

        # ── (1) pretrain ──────────────────────────────────────────────────────
        pre = ["python", "-u", "run.py", "--task_name", "timesiam", "--is_training", "0",
               *common, "--d_layers", "1",
               "--mask_rate", str(a.mask_rate), "--sampling_range", str(a.sampling_range),
               "--lineage_tokens", str(a.lineage_tokens), "--train_epochs", str(a.pretrain_epochs)]
        rc = _run(pre, logdir / "pretrain.log", a.dry_run)
        if rc != 0 and not a.dry_run:
            print(f"  [pretrain FAILED rc={rc}] — skipping {ds}"); continue

        # ── (2) fine-tune forecast per horizon ────────────────────────────────
        for pl in PRED_LENS:
            ft = ["python", "-u", "run.py", "--task_name", "fine_tune", "--is_training", "1",
                  *common, "--label_len", "48", "--pred_len", str(pl),
                  "--lineage_tokens", str(a.lineage_tokens),
                  "--representation_using", "concat", "--load_checkpoints", ckpt]
            _run(ft, logdir / f"forecast_pl{pl}.log", a.dry_run)

    print("\nDONE.  logs -> logs/timesiam/<dataset>/")


if __name__ == "__main__":
    main()

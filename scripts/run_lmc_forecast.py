#!/usr/bin/env python3
"""
run_lmc_forecast.py — Forecasting using a pretrained LMC backbone.

Extracts the TSMixer encoder from an LMC checkpoint (saved by run_lmc_training.py)
and evaluates it on standard forecasting datasets using the DINO pipeline.

The LMC backbone's encoder weights (backbone.*) are re-keyed to the DINO teacher
format (module.backbone.*) so that TSDiNO's test_run loads them cleanly.

Usage
-----
python scripts/run_lmc_forecast.py \
    --lmc_checkpoint /path/to/checkpoint_best_labeldata.pth \
    --gpu 0

python scripts/run_lmc_forecast.py \
    --lmc_checkpoint /path/to/checkpoint_best_labeldata.pth \
    --datasets etth1 etth2 weather \
    --pred_lens 96 192 \
    --gpu 0
"""

import argparse
import csv
import os
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATASETS  = ["etth1", "etth2", "ettm1", "ettm2", "weather", "electricity", "traffic"]
PRED_LENS = [96, 192, 336, 720]

# Where the converted checkpoint is written (must stay stable across datasets
# because run() uses this as output_dir).
CKPT_DIR   = ROOT / "checkpoints_lmc_forecast"
LOG_FOLDER = ROOT / "logs" / "lmc_forecast"


def _convert_lmc_to_dino(lmc_ckpt_path: str) -> tuple[dict, dict]:
    """Extract encoder from an LMC checkpoint and reformat for the DINO pipeline.

    LMCBackbone state-dict layout:
        backbone.*          — TSMixerForDINO encoder weights
        shared_k/v.*        — cross-attention (LMC-only, discarded)
        queries.*           — per-head queries (LMC-only, discarded)
        *_head.*            — 7 label-prediction heads (LMC-only, discarded)

    DINO teacher checkpoint format expected by test_run:
        teacher['module.backbone.*']  — after stripping 'module.' the keys are
                                        'backbone.*', which matches
                                        TSMixerForecastModel.backbone.*
    """
    raw = torch.load(lmc_ckpt_path, map_location="cpu", weights_only=False)
    lmc_sd = raw["model"]

    teacher_sd = {
        f"module.{k}": v
        for k, v in lmc_sd.items()
        if k.startswith("backbone.")
    }
    return {"teacher": teacher_sd}, raw.get("cfg", {})


def _warn_arch_mismatch(lmc_cfg: dict) -> None:
    """Print a warning if the LMC architecture differs from the DINO config defaults."""
    lmc_layers = lmc_cfg.get("tsmixer_e_layers")
    if lmc_layers is not None and lmc_layers != 3:
        print(
            f"  [WARN] LMC was trained with tsmixer_e_layers={lmc_layers} but the "
            f"DINO forecasting pipeline defaults to 3.  Only the first 3 encoder "
            f"layers will be loaded.  To fix, update TSDiNO/config.py: "
            f"'tsmixer_e_layers': {lmc_layers}."
        )


# ── logging: tee stdout to file ───────────────────────────────────────────────

class _Tee:
    def __init__(self, original, file_handle):
        self._orig = original
        self._file = file_handle

    def write(self, data):
        self._orig.write(data)
        self._orig.flush()
        self._file.write(data)
        self._file.flush()

    def flush(self):
        self._orig.flush()
        self._file.flush()

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


def main():
    p = argparse.ArgumentParser(
        description="Evaluate an LMC-pretrained backbone on forecasting datasets."
    )
    p.add_argument("--lmc_checkpoint", required=True,
                   help="Path to LMC checkpoint (e.g. checkpoint_best_labeldata.pth)")
    p.add_argument("--gpu",            type=int,  default=0)
    p.add_argument("--datasets",       nargs="+", default=DATASETS)
    p.add_argument("--pred_lens",      nargs="+", type=int, default=PRED_LENS)
    p.add_argument("--linear_probe",   type=lambda x: x.lower() != "false", default=True,
                   help="True (freeze backbone, train head only) or False (fine-tune all)")
    p.add_argument("--epochs_forecasting", type=int,   default=None,
                   help="Override forecast training epochs (default: from DINO config)")
    p.add_argument("--lr_forecasting",    type=float, default=None,
                   help="Override forecasting head learning rate (default: from DINO config)")
    p.add_argument("--out_csv",        type=str, default=None,
                   help="Results CSV path (default: results/lmc_forecast.csv)")
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # ── validate checkpoint ───────────────────────────────────────────────────
    lmc_path = Path(args.lmc_checkpoint).resolve()
    if not lmc_path.exists():
        print(f"[ERROR] Checkpoint not found: {lmc_path}")
        sys.exit(1)

    # ── convert ───────────────────────────────────────────────────────────────
    print(f"Loading LMC checkpoint: {lmc_path}")
    dino_ckpt, lmc_cfg = _convert_lmc_to_dino(str(lmc_path))
    print(f"  Encoder keys extracted: {len(dino_ckpt['teacher'])}")
    _warn_arch_mismatch(lmc_cfg)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    converted_path = CKPT_DIR / "checkpoint_best.pth"
    torch.save(dino_ckpt, converted_path)
    print(f"  Saved DINO-format checkpoint → {converted_path}")

    # ── forecast loop ─────────────────────────────────────────────────────────
    from Train_and_downstream import run

    out_csv = Path(args.out_csv) if args.out_csv else ROOT / "results" / "lmc_forecast.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    LOG_FOLDER.mkdir(parents=True, exist_ok=True)

    fieldnames = ["dataset", "pred_len", "mse", "mae", "timestamp"]
    rows: list[dict] = []

    for dataset in args.datasets:
        print(f"\n{'='*60}")
        print(f"  LMC Forecast — {dataset}")
        print(f"{'='*60}")

        for pred_len in args.pred_lens:
            log_path = LOG_FOLDER / f"{dataset}_pred{pred_len}.log"
            print(f"  pred_len={pred_len}  log={log_path.relative_to(ROOT)}")

            mse = None
            with log_to_file(log_path):
                try:
                    result = run(
                        model="dino",
                        skip_train=True,
                        forecast_dataset=dataset,
                        backbone_type="tsmixer",
                        checkpoints=["best"],
                        output_dir=str(CKPT_DIR),
                        pred_lens=[pred_len],
                        linear_probe=args.linear_probe,
                        epochs_forecasting=args.epochs_forecasting,
                        lr_forecasting=args.lr_forecasting,
                    )
                    if result is not None and result[1] is not None:
                        mse = result[1]
                except Exception as exc:
                    print(f"[ERROR] {dataset}/pred{pred_len}: {exc}")
                    import traceback; traceback.print_exc()

            if mse is not None:
                print(f"  → pred_len={pred_len}  MSE={mse:.6f}")
                ts = datetime.now().isoformat(timespec="seconds")
                rows.append({
                    "dataset":  dataset,
                    "pred_len": pred_len,
                    "mse":      f"{mse:.6f}",
                    "mae":      "N/A",
                    "timestamp": ts,
                })
                with open(out_csv, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)

    if rows:
        print(f"\nResults saved → {out_csv}")
    else:
        print("\nNo results to save.")


if __name__ == "__main__":
    main()

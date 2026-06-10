#!/usr/bin/env python3
"""
run_lmc_classification.py — Classification using a pretrained LMC backbone.

Extracts the encoder from an LMC checkpoint and evaluates it on UEA
classification datasets using the DINO pipeline.

Usage
-----
python scripts/run_lmc_classification.py \
    --lmc_checkpoint ./checkpoints/checkpoint_best_labeldata.pth \
    --gpu 6

python scripts/run_lmc_classification.py \
    --lmc_checkpoint ./checkpoints/checkpoint_best_labeldata.pth \
    --datasets EthanolConcentration Heartbeat UWaveGestureLibrary \
    --gpu 6
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

DATASETS = [
    "JapaneseVowels", "SpokenArabicDigits", "FaceDetection",
    "Handwriting", "UWaveGestureLibrary",
    "Heartbeat", "SelfRegulationSCP1", "SelfRegulationSCP2",
    "EthanolConcentration",
]

CKPT_DIR   = ROOT / "checkpoints_lmc_classification"
LOG_FOLDER = ROOT / "logs" / "lmc_classification"


def _convert_lmc_to_dino(lmc_ckpt_path: str) -> tuple[dict, dict]:
    """Extract encoder from an LMC checkpoint and reformat for the DINO pipeline.

    Same conversion as run_lmc_forecast.py — backbone.* → module.backbone.*
    in teacher format so TSDiNO's train_classification loads them cleanly.
    """
    raw    = torch.load(lmc_ckpt_path, map_location="cpu", weights_only=False)
    lmc_sd = raw["model"]
    teacher_sd = {
        f"module.{k}": v
        for k, v in lmc_sd.items()
        if k.startswith("backbone.")
    }
    return {"teacher": teacher_sd}, raw.get("cfg", {})


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
        description="Evaluate an LMC-pretrained backbone on classification datasets."
    )
    p.add_argument("--lmc_checkpoint", required=True,
                   help="Path to LMC checkpoint (e.g. checkpoints/checkpoint_best_labeldata.pth)")
    p.add_argument("--gpu",          type=int,  default=0)
    p.add_argument("--datasets",     nargs="+", default=DATASETS)
    p.add_argument("--linear_probe", type=lambda x: x.lower() != "false", default=True,
                   help="True (freeze backbone) or False (fine-tune all)")
    p.add_argument("--out_csv",      type=str,  default=None,
                   help="Results CSV path (default: results/lmc_classification.csv)")
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    lmc_path = Path(args.lmc_checkpoint).resolve()
    if not lmc_path.exists():
        print(f"[ERROR] Checkpoint not found: {lmc_path}")
        sys.exit(1)

    print(f"Loading LMC checkpoint: {lmc_path}")
    dino_ckpt, lmc_cfg = _convert_lmc_to_dino(str(lmc_path))
    backbone_type = lmc_cfg.get("backbone_type", "tsmixer")
    print(f"  Backbone type   : {backbone_type}")
    print(f"  Encoder keys    : {len(dino_ckpt['teacher'])}")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    converted_path = CKPT_DIR / "checkpoint_best.pth"
    torch.save(dino_ckpt, converted_path)
    print(f"  Saved DINO-format checkpoint → {converted_path}")

    from Train_and_downstream import run

    out_csv = Path(args.out_csv) if args.out_csv else ROOT / "results" / "lmc_classification.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    LOG_FOLDER.mkdir(parents=True, exist_ok=True)

    fieldnames = ["dataset", "accuracy", "timestamp"]
    rows: list[dict] = []

    for dataset in args.datasets:
        print(f"\n{'='*60}")
        print(f"  LMC Classification — {dataset}")
        print(f"{'='*60}")

        log_path = LOG_FOLDER / f"{dataset}.log"
        print(f"  log={log_path.relative_to(ROOT)}")

        acc = None
        with log_to_file(log_path):
            try:
                result = run(
                    model="dino",
                    skip_train=True,
                    classification_dataset=dataset,
                    backbone_type=backbone_type,
                    checkpoints=["best"],
                    output_dir=str(CKPT_DIR),
                    linear_probe=args.linear_probe,
                )
                if result is not None and result[2] is not None:
                    acc = result[2]
            except Exception as exc:
                print(f"[ERROR] {dataset}: {exc}")
                import traceback; traceback.print_exc()

        if acc is not None:
            print(f"  → Accuracy: {acc:.4f}")
            rows.append({
                "dataset":   dataset,
                "accuracy":  f"{acc:.4f}",
                "timestamp": datetime.now().isoformat(timespec="seconds"),
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

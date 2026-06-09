#!/usr/bin/env python3
"""
run_lmc_training.py — LMC pretraining entry point.

Usage
-----
nohup python LabeledDataTraining/run_lmc_training.py \
    --data_dir /home/shared/datasets/TS_synthetic_labeled \
    --gpu 4 > logs/pretrain_lmc.log 2>&1 &
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "LabeledDataTraining"))
sys.path.insert(0, str(ROOT / "TSDiNO"))

# Cache models.ts_mixer_backbone from TSDiNO before LabelTraining.py
# inserts TimeMixer-main/ into sys.path (which has a conflicting models/__init__.py).
from models.ts_mixer_backbone import TSMixerForDINO as _TSMixerForDINO  # noqa

from config import config as _dino_cfg  # noqa: E402
from LabelTraining import train_lmc     # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",      required=True)
    p.add_argument("--checkpoint",    default=None)
    p.add_argument("--c_in",          type=int,   default=None)
    p.add_argument("--seq_len",       type=int,   default=512)
    p.add_argument("--epochs",        type=int,   default=30)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--min_lr",        type=float, default=1e-5)
    p.add_argument("--batch_size",    type=int,   default=256)
    p.add_argument("--hidden_dim",    type=int,   default=64)
    p.add_argument("--backbone_type",   type=str,   default="tsmixer",
                   choices=["tsmixer", "patchtst"],
                   help="Encoder backbone: 'tsmixer' (default) or 'patchtst'")
    p.add_argument("--freeze_backbone", type=lambda x: x.lower() != "false", default=False)
    p.add_argument("--min_latent",    type=int,   default=2)
    p.add_argument("--max_latent",    type=int,   default=10)
    p.add_argument("--val_frac",      type=float, default=0.05)
    p.add_argument("--test_frac",     type=float, default=0.05)
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--saveckp_freq",  type=int,   default=1)
    p.add_argument("--gpu",           type=int,   default=0)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--output_dir",    type=str,   default=None,
                   help="Override checkpoint output dir (default: from DINO config)")
    return p.parse_args()


def main():
    args = parse_args()

    # Set GPU before any CUDA calls.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    cfg = dict(_dino_cfg)
    cfg["c_in"]              = args.c_in or cfg.get("c_in", 7)
    cfg["seq_len"]           = args.seq_len
    cfg["data_dir_labeled"]  = args.data_dir
    cfg["checkpoint_path"]   = args.checkpoint
    cfg["epochs_labeled"]    = args.epochs
    cfg["lr_labeled"]        = args.lr
    cfg["min_lr_labeled"]    = args.min_lr
    cfg["batch_size_labeled"]= args.batch_size
    cfg["hidden_dim_labeled"]= args.hidden_dim
    cfg["backbone_type"]     = args.backbone_type
    cfg["freeze_backbone"]   = args.freeze_backbone
    cfg["min_latent"]        = args.min_latent
    cfg["max_latent"]        = args.max_latent
    cfg["val_frac"]          = args.val_frac
    cfg["test_frac"]         = args.test_frac
    cfg["num_workers"]       = args.num_workers
    cfg["saveckp_freq"]      = args.saveckp_freq
    cfg["gpu"]               = args.gpu
    cfg["seed"]              = args.seed
    if args.output_dir is not None:
        cfg["output_dir"]      = args.output_dir

    print("=" * 60)
    print("  LMC Pretraining")
    print("=" * 60)
    print(f"  data_dir   : {cfg['data_dir_labeled']}")
    print(f"  checkpoint : {cfg['checkpoint_path'] or 'random init'}")
    print(f"  output_dir : {cfg['output_dir']}  (suffix: _labeldata)")
    if cfg.get("backbone_type", "tsmixer") == "patchtst":
        seq_len   = cfg.get("seq_len", 512)
        patch_len = cfg.get("patch_len", 16)
        print(f"  encoder    : PatchTST  d_model={cfg.get('embed_dim', 128)}  "
              f"layers={cfg.get('n_layers', 4)}  "
              f"patches={seq_len // patch_len}  patch_len={patch_len}")
    else:
        print(f"  encoder    : TSMixer  d_model={cfg['tsmixer_d_model']}  "
              f"layers={cfg['tsmixer_e_layers']}  "
              f"scales={cfg['tsmixer_down_sampling_layers'] + 1}")
    print(f"  training   : epochs={cfg['epochs_labeled']}  "
          f"lr={cfg['lr_labeled']}  batch={cfg['batch_size_labeled']}")
    print(f"  backbone   : {'frozen' if cfg['freeze_backbone'] else 'unfrozen (pretraining)'}")
    print("=" * 60)

    train_lmc(cfg)


if __name__ == "__main__":
    main()

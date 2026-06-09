"""
TSDiNO TSMixer classification.

Pretrained DINO teacher TSMixerForDINO encoder (frozen by default; unfrozen if
linear_probe=False) + ClassificationHead trained on the classification split.

Head: global CLS pooling [B, C, d_model] → flatten(C * d_model) → dropout → linear → n_classes.
Identical interface to NTP / PatchTST / TimeDaRT classification scripts:
  classification(config, checkpoint_path, train, val, test, n_classes, linear_probe, mlp_head)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

_HERE             = Path(__file__).parent
_TIMEMIXER_ROOT   = (_HERE / ".." / "TimeMixer-main").resolve()
_TIMEMIXER_MODELS = _TIMEMIXER_ROOT / "models"
for _p in [str(_TIMEMIXER_MODELS), str(_TIMEMIXER_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.insert(0, str(_HERE))  # TSDiNO/models/ must beat TimeMixer-main/models/

from models.ts_mixer_backbone import TSMixerForDINO  # noqa: E402


class ClassificationHead(nn.Module):
    """Global CLS rep [B, C, d_model] → flatten(C * d_model) → dropout → linear → n_classes."""

    def __init__(self, c_in, d_model, n_classes, head_dropout, mlp_head: bool = False, hidden_dim: int = 512):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Identity() if mlp_head else nn.Dropout(head_dropout)
        if mlp_head:
            self.linear = nn.Sequential(
                nn.Linear(c_in * d_model, hidden_dim),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(hidden_dim, n_classes),
            )
        else:
            self.linear = nn.Linear(c_in * d_model, n_classes)

    def forward(self, x):
        """x: [B, C, d_model]  →  [B, n_classes]"""
        x = self.flatten(x)   # [B, C * d_model]
        x = self.dropout(x)
        return self.linear(x)


def classification(config, checkpoint_path,
                   classification_train, classification_val,
                   classification_test, n_classes,
                   linear_probe=True, mlp_head: bool = False):
    """
    Classification with pretrained TSMixerForDINO teacher encoder.

    Args:
        config                : dict — DINO config (same keys used for pretraining).
        checkpoint_path       : path to pretrained .pth checkpoint (teacher weights).
        classification_train/val/test : DataLoaders — each batch is
                                (patches [B, P, PL, C], labels [B], padding_mask [B, P]).
        n_classes             : total number of target classes.
        linear_probe          : True → freeze backbone, train head only.
                                False → fine-tune backbone + head jointly.
        mlp_head              : True → 2-layer MLP head; False → single linear.
    Returns:
        test accuracy (float)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n=== TSDiNO TSMixer Classification ===")
    print(f"Loading checkpoint: {checkpoint_path}")

    # Infer n_vars from first batch
    sample_patches, _, _ = next(iter(classification_train))
    n_v = sample_patches.shape[-1]   # [B, P, PL, n_vars]

    # ── Build backbone ─────────────────────────────────────────────────────────
    backbone = TSMixerForDINO(
        c_in                 = n_v,
        seq_len              = config.get("seq_len", 512),
        d_model              = config.get("tsmixer_d_model", 16),
        e_layers             = config.get("tsmixer_e_layers", 2),
        d_ff                 = config.get("tsmixer_d_ff", 32),
        dropout              = config.get("dropout", 0.1),
        down_sampling_layers = config.get("tsmixer_down_sampling_layers", 3),
        down_sampling_window = config.get("tsmixer_down_sampling_window", 2),
        down_sampling_method = config.get("tsmixer_down_sampling_method", "avg"),
        decomp_method        = config.get("tsmixer_decomp_method", "moving_avg"),
        moving_avg           = config.get("tsmixer_moving_avg", 25),
        top_k                = config.get("tsmixer_top_k", 5),
        use_norm             = config.get("tsmixer_use_norm", 1),
        channel_independence = config.get("tsmixer_channel_independence", 1),
    ).to(device)

    d_model = config.get("tsmixer_d_model", 16)

    # ── Load teacher checkpoint ────────────────────────────────────────────────
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        ckpt   = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        raw_sd = ckpt["teacher"]
        new_sd = {}
        for k, v in raw_sd.items():
            k = k.replace("module.", "")
            if k.startswith("backbone."):
                k = k[len("backbone."):]
            new_sd[k] = v
        bbone_dict = backbone.state_dict()
        filtered   = {k: v for k, v in new_sd.items()
                      if k in bbone_dict and bbone_dict[k].shape == v.shape}
        missing, unexpected = backbone.load_state_dict(filtered, strict=False)
        print(f"  Loaded {len(filtered)}/{len(bbone_dict)} backbone params "
              f"| missing: {len(missing)} | unexpected (DINO-only): {len(unexpected)}")
    else:
        print(f"  WARNING: checkpoint not found at {checkpoint_path}, using random init")

    # ── Freeze / unfreeze ──────────────────────────────────────────────────────
    if linear_probe:
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False
        print(f"  [TSMixer classify] MODE: linear probe — encoder FROZEN")
    else:
        print(f"  [TSMixer classify] MODE: full fine-tuning — encoder UNFROZEN")

    cls_head = ClassificationHead(
        c_in         = n_v,
        d_model      = d_model,
        n_classes    = n_classes,
        head_dropout = config.get("head_dropout", 0.1),
        mlp_head     = mlp_head,
    ).to(device)

    n_epochs    = config.get("epoch_classification", 20)
    _all_params = list(backbone.parameters()) + list(cls_head.parameters())
    _cfg_head_lr = config.get("lr_classification")
    _cfg_enc_lr  = config.get("lr_classification_encoder")
    head_lr = float(_cfg_head_lr) if _cfg_head_lr is not None else 1e-3
    enc_lr  = float(_cfg_enc_lr)  if _cfg_enc_lr  is not None else head_lr
    if linear_probe:
        optimizer  = torch.optim.Adam(cls_head.parameters(), lr=head_lr, weight_decay=1e-4)
        _max_lrs   = head_lr
        _trainable = sum(p.numel() for p in cls_head.parameters())
    else:
        optimizer = torch.optim.Adam([
            {"params": cls_head.parameters(), "lr": head_lr},
            {"params": backbone.parameters(), "lr": enc_lr},
        ], weight_decay=1e-4)
        _max_lrs   = [head_lr, enc_lr]
        _trainable = sum(p.numel() for p in _all_params)
        print(f"  [TSMixer classify] head_lr={head_lr}  encoder_lr={enc_lr}")
    _total = sum(p.numel() for p in _all_params)
    print(f"  Trainable: {_trainable:,} / {_total:,} params")

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=_max_lrs,
        total_steps=n_epochs * len(classification_train),
        pct_start=0.3, anneal_strategy='cos',
    )

    for epoch in range(n_epochs):
        cls_head.train()
        if not linear_probe:
            backbone.train()
        correct, total = 0, 0
        for patches, labels, padding_mask in classification_train:
            # [B, P, PL, C] → [B, T, C]
            B, P, PL, C = patches.shape
            x      = patches.reshape(B, P * PL, C).float().to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(not linear_probe):
                enc = backbone(x)         # [B, C, d_model]
            logits = cls_head(enc)        # [B, n_classes]
            loss   = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            correct += (logits.argmax(1) == labels).sum().item()
            total   += len(labels)
        if epoch % 5 == 0:
            print(f"  Epoch {epoch:3d} | train acc {correct/total:.4f}")

    cls_head.eval()
    tc, tt = 0, 0
    with torch.no_grad():
        for patches, labels, padding_mask in classification_test:
            B, P, PL, C = patches.shape
            x      = patches.reshape(B, P * PL, C).float().to(device)
            labels = labels.to(device)
            enc    = backbone(x)
            logits = cls_head(enc)
            tc += (logits.argmax(1) == labels).sum().item()
            tt += len(labels)
    test_acc = tc / tt
    print(f"[TSMixer DINO] Test Accuracy: {test_acc:.4f}")
    return test_acc

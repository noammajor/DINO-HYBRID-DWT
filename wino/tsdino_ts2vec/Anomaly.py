"""
TSDiNO anomaly detection (reconstruction-based).

Pretrained DINO teacher encoder (frozen by default; unfrozen if linear_probe=False)
+ linear reconstruction decoder trained on normal data only.
Anomaly score = per-timestep reconstruction MSE.
Threshold = percentile of combined train+test energy.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from models.ts2vec_backbone import TS2VecForDINO as PatchTST

# Shared anomaly metrics — single source of truth (RAW/PA/EVENT/RANGE/AFFIL).
import sys as _sys
from pathlib import Path as _Path
_SHARED = str(_Path(__file__).resolve().parents[2] / "shared")
if _SHARED not in _sys.path:
    _sys.path.insert(0, _SHARED)
import anomaly_metrics as am  # noqa: E402


class _LinearReconDecoder(nn.Module):
    """Channel-mixed: per-timestep token reps [B, T, d_model] → raw series [B, T, C]."""
    def __init__(self, d_model: int, c_in: int, mlp_head: bool = False, hidden_dim: int = 512):
        super().__init__()
        if mlp_head:
            self.proj = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, c_in),
            )
        else:
            self.proj = nn.Linear(d_model, c_in)

    def forward(self, z):
        # z: [B, T, d_model]  →  [B, T, C]
        return self.proj(z)


def anomaly_detection(args, path_num, anomaly_train, anomaly_test,
                      anomaly_ratio: float = 1.0,
                      checkpoint_path: str = None,
                      linear_probe: bool = True,
                      mlp_head: bool = False):
    """
    Reconstruction-based anomaly detection with DINO teacher encoder
    (frozen by default; unfrozen if linear_probe=False).

    Args:
        args             : argparse Namespace (same config used for training)
        path_num         : checkpoint number used to build path if checkpoint_path
                           is not given (e.g. 100 → checkpoint0100.pth)
        anomaly_train    : DataLoader — batches of patches [B, P, patch_len, n_vars]
        anomaly_test     : DataLoader — batches of (patches, labels [B, T])
        anomaly_ratio    : top-X% of combined energy flagged as anomaly
        checkpoint_path  : optional explicit path to a .pth checkpoint file.
                           If provided, path_num and args.output_dir are ignored
                           for checkpoint loading.
        linear_probe     : if True, freeze backbone and train only decoder. If False, fine-tune backbone + decoder.
    Returns:
        dict with f1, precision, recall, accuracy, threshold
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Build backbone (PatchTSTEncoder only, no DINOHead) ───────────────────
    backbone = PatchTST(
        c_in=args.c_in,
        target_dim=args.pred_len,
        patch_len=args.patch_len,
        num_patch=args.num_patches,
        n_layers=args.n_layers,
        hidden_dims=getattr(args, "ts2vec_hidden_dims", 64),
        n_heads=args.n_heads,
        d_model=args.embed_dim,
        shared_embedding=True,
        d_ff=args.d_ff,
        dropout=0.0,
        head_dropout=0.0,
        act='gelu',
        head_type='Dino',
        res_attention=False,
        drop_path_rate=0.0,
        step_size=args.step_size,
    )  # full TS2VecForDINO — use forward_recon for token reps

    # ── Resolve checkpoint path ───────────────────────────────────────────────
    if checkpoint_path is None:
        if isinstance(path_num, int):
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint{path_num:04d}.pth')
        else:
            checkpoint_path = os.path.join(args.output_dir, 'checkpoint_best.pth')

    print(f"\n=== TSDiNO Anomaly Detection ===")
    print(f"Loading checkpoint: {checkpoint_path}")

    if os.path.exists(checkpoint_path):
        ckpt   = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        raw_sd = ckpt['teacher']
        # TSMultiCropWrapper.backbone = TS2VecForDINO → strip one 'backbone.' so the
        # TS2VecForDINO encoder/mask_token/norm keys (backbone.*) match.
        new_sd = {k[len('backbone.'):]: v
                  for k, v in raw_sd.items() if k.startswith('backbone.')}
        missing, unexpected = backbone.load_state_dict(new_sd, strict=False)
        print(f"  Loaded {len(new_sd)} weights | missing: {len(missing)} | unexpected: {len(unexpected)}")
    else:
        print(f"  WARNING: checkpoint not found — using random init.")

    backbone = backbone.to(device)
    if linear_probe:
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False
        print(f"  [TSDiNO anomaly] MODE: linear probe — encoder FROZEN")
    else:
        for p in backbone.parameters():
            p.requires_grad = True
        print(f"  [TSDiNO anomaly] MODE: full fine-tune — encoder UNFROZEN")

    d_model = args.embed_dim
    decoder = _LinearReconDecoder(d_model, args.c_in, mlp_head=mlp_head).to(device)
    # LR: args value if set, else hardcoded default.
    _cfg_head_lr = getattr(args, "lr_anomaly", None)
    _cfg_enc_lr  = getattr(args, "lr_anomaly_encoder", None)
    head_lr = float(_cfg_head_lr) if _cfg_head_lr is not None else 1e-3
    enc_lr  = float(_cfg_enc_lr)  if _cfg_enc_lr  is not None else head_lr
    if linear_probe:
        optimizer = torch.optim.Adam(decoder.parameters(), lr=head_lr)
    else:
        optimizer = torch.optim.Adam([
            {"params": decoder.parameters(),  "lr": head_lr},
            {"params": backbone.parameters(), "lr": enc_lr},
        ])
        print(f"  [TSDiNO anomaly] head_lr={head_lr}  encoder_lr={enc_lr}")
    n_epochs  = getattr(args, 'epoch_anomaly', 10)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    def _encode(raw):
        """raw: [B, P, PL, C] or [B, T, C] → per-timestep reps [B, T, d_model]."""
        if raw.dim() == 4:
            B, P, PL, C = raw.shape
            raw = raw.reshape(B, P * PL, C)
        raw = raw.to(device)
        z = backbone.forward_recon(raw)   # [B, T, 1, d_model]
        return z.squeeze(2)               # [B, T, d_model]

    # ── (1) train decoder (with validation + early stopping) ─────────────────
    all_batches   = list(anomaly_train)
    n_val_b       = max(1, len(all_batches) // 5)
    train_batches = all_batches[:-n_val_b]
    val_batches   = all_batches[-n_val_b:]

    patience   = getattr(args, 'anomaly_patience', 3)
    best_val   = float('inf')
    best_state = None
    no_improve = 0

    print(f"  Training decoder ({n_epochs} epochs, patience={patience}) …")
    for epoch in range(n_epochs):
        # train
        decoder.train()
        if not linear_probe:
            backbone.train()
        total_loss = 0.0
        for batch in train_batches:
            patches = batch[0] if isinstance(batch, (list, tuple)) else batch
            if patches.dim() == 3: patches = patches.unsqueeze(-1)
            raw = patches.to(device)
            B, P, PL, C = raw.shape
            with torch.set_grad_enabled(not linear_probe):
                z = _encode(raw)
            recon  = decoder(z)
            target = raw.reshape(B, P * PL, C)
            loss   = F.mse_loss(recon, target)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
        # validate
        decoder.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_batches:
                patches = batch[0] if isinstance(batch, (list, tuple)) else batch
                if patches.dim() == 3: patches = patches.unsqueeze(-1)
                raw = patches.to(device)
                B, P, PL, C = raw.shape
                z      = _encode(raw)
                recon  = decoder(z)
                target = raw.reshape(B, P * PL, C)
                val_loss += F.mse_loss(recon, target).item()
        val_loss /= len(val_batches)
        # early stopping
        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in decoder.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break
        scheduler.step()
        if epoch % max(1, n_epochs // 5) == 0:
            print(f"    epoch {epoch+1:3d} | train {total_loss/len(train_batches):.6f} | val {val_loss:.6f}")
    if best_state is not None:
        decoder.load_state_dict(best_state)

    # ── (2) train energy ─────────────────────────────────────────────────────
    decoder.eval()
    train_energy = []
    with torch.no_grad():
        for batch in anomaly_train:
            patches = batch[0] if isinstance(batch, (list, tuple)) else batch
            if patches.dim() == 3: patches = patches.unsqueeze(-1)
            raw = patches.to(device)
            B, P, PL, C = raw.shape
            z      = _encode(raw)
            recon  = decoder(z)
            target = raw.reshape(B, P * PL, C)
            score  = F.mse_loss(recon, target, reduction="none").mean(dim=-1)  # [B, T]
            train_energy.append(score.cpu().numpy())
    train_energy = np.concatenate(train_energy).reshape(-1)

    # ── (3) test energy ──────────────────────────────────────────────────────
    test_energy, all_labels = [], []
    with torch.no_grad():
        for patches, labels in anomaly_test:
            if patches.dim() == 3: patches = patches.unsqueeze(-1)
            raw = patches.to(device)
            B, P, PL, C = raw.shape
            z      = _encode(raw)
            recon  = decoder(z)
            target = raw.reshape(B, P * PL, C)
            score  = F.mse_loss(recon, target, reduction="none").mean(dim=-1)  # [B, T]
            test_energy.append(score.cpu().numpy())
            all_labels.append(labels.numpy())
    test_energy = np.concatenate(test_energy).reshape(-1)
    gt          = np.concatenate(all_labels).reshape(-1).astype(int)

    # ── (4) threshold & predict ───────────────────────────────────────────────
    threshold = np.percentile(np.concatenate([train_energy, test_energy]),
                              100 - anomaly_ratio)
    pred = (test_energy > threshold).astype(int)

    # ── (5) metrics ───────────────────────────────────────────────────────────
    # All variants (RAW/PA/EVENT/RANGE/AFFIL) via the shared single-source module.
    # Primary f1/precision/recall follow point-adjust, OFF by default (it inflates
    # F1). Set TS_ANOMALY_ADJUST=1 to make the point-adjusted numbers primary.
    point_adjust = os.environ.get("TS_ANOMALY_ADJUST", "0") == "1"
    m = am.compute_all(gt, pred, point_adjust=point_adjust)
    print(f"  [TSDiNO] threshold={threshold:.6f}  (top {anomaly_ratio}%)")
    print(am.format_table(m, title="TSDiNO Anomaly Detection"))

    return dict(**m, threshold=threshold)

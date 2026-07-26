"""
TSDiNO anomaly detection with TSMixerForDINO backbone (reconstruction-based).

Pretrained DINO teacher TSMixerForDINO encoder (frozen by default)
+ linear reconstruction decoder trained on normal data only.
Anomaly score = per-timestep reconstruction MSE.
Threshold = percentile of combined train+test energy.

NOTE: TSMixer PDM blocks have linear layers fixed to seq_len (set at pretraining
time). The AnomalyDataPuller win_size must therefore match cfg["seq_len"]; the
caller in Train_and_downstream.py handles this automatically.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Ensure TSDiNO/models/ is on the path so models.ts_mixer_backbone resolves
# to TSDiNO/models/ts_mixer_backbone.py, not TimeMixer-main/models/.
_HERE             = Path(__file__).parent
_TIMEMIXER_ROOT   = (_HERE / ".." / "TimeMixer-main").resolve()
_TIMEMIXER_MODELS = _TIMEMIXER_ROOT / "models"
for _p in [str(_TIMEMIXER_MODELS), str(_TIMEMIXER_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.insert(0, str(_HERE))  # TSDiNO/models/ must beat TimeMixer-main/models/

from models.ts_mixer_backbone import TSMixerForDINO  # noqa: E402

# Shared anomaly metrics — single source of truth (RAW/PA/EVENT/RANGE/AFFIL).
_SHARED = str((_HERE / ".." / ".." / "shared").resolve())
if _SHARED not in sys.path:
    sys.path.insert(0, _SHARED)
import anomaly_metrics as am  # noqa: E402


class _TSMixerReconDecoder(nn.Module):
    """[B, T, C, d_model] → [B, T, C]  (per-timestep linear projection)."""

    def __init__(self, d_model: int, mlp_head: bool = False, hidden_dim: int = 512):
        super().__init__()
        if mlp_head:
            self.proj = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.proj = nn.Linear(d_model, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, T, C, d_model]
        return self.proj(z).squeeze(-1)  # [B, T, C]


def anomaly_detection(
    cfg,
    path_num,
    anomaly_train,
    anomaly_test,
    anomaly_ratio: float = 1.0,
    checkpoint_path: str = None,
    linear_probe: bool = True,
    mlp_head: bool = False,
):
    """
    Reconstruction-based anomaly detection using a pretrained TSMixerForDINO encoder.

    Args:
        cfg             : dict — DINO config (same keys used for pretraining).
                          Required keys: c_in, seq_len, tsmixer_d_model,
                          tsmixer_e_layers, tsmixer_d_ff,
                          tsmixer_down_sampling_layers/window/method,
                          tsmixer_decomp_method, tsmixer_moving_avg, tsmixer_top_k.
        path_num        : int checkpoint number or any non-int value → checkpoint_best.pth.
        anomaly_train   : DataLoader — [B, P, patch_len, C] batches (normal data only).
        anomaly_test    : DataLoader — ([B, P, patch_len, C], [B, T]) batches with labels.
        anomaly_ratio   : top-X% of combined energy flagged as anomaly.
        checkpoint_path : explicit .pth path; overrides path_num + cfg["output_dir"].
        linear_probe    : True → freeze backbone, train decoder only.
                          False → fine-tune backbone + decoder jointly.
        mlp_head        : True → 2-layer MLP decoder; False → single Linear.
    Returns:
        dict with keys: f1, precision, recall, accuracy, threshold.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Build backbone ─────────────────────────────────────────────────────────
    backbone = TSMixerForDINO(
        c_in                 = cfg["c_in"],
        seq_len              = cfg.get("seq_len", 512),
        d_model              = cfg.get("tsmixer_d_model", 16),
        e_layers             = cfg.get("tsmixer_e_layers", 2),
        d_ff                 = cfg.get("tsmixer_d_ff", 32),
        dropout              = cfg.get("dropout", 0.1),
        down_sampling_layers = cfg.get("tsmixer_down_sampling_layers", 3),
        down_sampling_window = cfg.get("tsmixer_down_sampling_window", 2),
        down_sampling_method = cfg.get("tsmixer_down_sampling_method", "avg"),
        decomp_method        = cfg.get("tsmixer_decomp_method", "moving_avg"),
        moving_avg           = cfg.get("tsmixer_moving_avg", 25),
        top_k                = cfg.get("tsmixer_top_k", 5),
        use_norm             = cfg.get("tsmixer_use_norm", 1),
        channel_independence = cfg.get("tsmixer_channel_independence", 1),
    )
    d_model = cfg.get("tsmixer_d_model", 16)

    # ── Resolve checkpoint ─────────────────────────────────────────────────────
    if checkpoint_path is None:
        out_dir = cfg.get("output_dir", "./checkpoints")
        if isinstance(path_num, int):
            checkpoint_path = os.path.join(out_dir, f"checkpoint{path_num:04d}.pth")
        else:
            checkpoint_path = os.path.join(out_dir, "checkpoint_best.pth")

    print(f"\n=== TSDiNO TSMixer Anomaly Detection ===")
    print(f"Loading checkpoint: {checkpoint_path}")

    if os.path.exists(checkpoint_path):
        ckpt   = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        raw_sd = ckpt["teacher"]
        # TSMultiCropWrapper(TSMixerForDINO): keys are backbone.*  (+ optional module. prefix)
        # DINO-specific parameters (cls_query, global_attn, mask_token) are silently
        # skipped via strict=False — they are not needed for reconstruction.
        new_sd = {}
        for k, v in raw_sd.items():
            k = k.replace("module.", "")
            if k.startswith("backbone."):
                k = k[len("backbone."):]
            new_sd[k] = v
        missing, unexpected = backbone.load_state_dict(new_sd, strict=False)
        print(f"  Loaded {len(new_sd)} weights | "
              f"missing: {len(missing)} | unexpected (DINO-only): {len(unexpected)}")
    else:
        print("  WARNING: checkpoint not found — using random init.")

    backbone = backbone.to(device)
    if linear_probe:
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False
        print("  MODE: linear probe — encoder FROZEN")
    else:
        for p in backbone.parameters():
            p.requires_grad = True
        print("  MODE: full fine-tune — encoder UNFROZEN")

    decoder   = _TSMixerReconDecoder(d_model, mlp_head=mlp_head).to(device)
    head_lr   = float(cfg.get("lr_anomaly") or 1e-3)
    enc_lr    = float(cfg.get("lr_anomaly_encoder") or head_lr)
    if linear_probe:
        optimizer = torch.optim.Adam(decoder.parameters(), lr=head_lr)
    else:
        optimizer = torch.optim.Adam([
            {"params": decoder.parameters(),  "lr": head_lr},
            {"params": backbone.parameters(), "lr": enc_lr},
        ])
        print(f"  head_lr={head_lr}  encoder_lr={enc_lr}")
    n_epochs  = int(cfg.get("epoch_anomaly", 10))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    def _encode(patches: torch.Tensor) -> torch.Tensor:
        """patches [B, P, patch_len, C] → encoder tokens [B, T, C, d_model]."""
        if patches.dim() == 3:
            patches = patches.unsqueeze(-1)
        patches = patches.to(device)
        B, P, PL, C = patches.shape
        x = patches.reshape(B, P * PL, C)     # [B, T, C]
        return backbone.forward_recon(x)       # [B, T, C, d_model]

    # ── (1) Train decoder with validation + early stopping ────────────────────
    all_batches   = list(anomaly_train)
    n_val_b       = max(1, len(all_batches) // 5)
    train_batches = all_batches[:-n_val_b]
    val_batches   = all_batches[-n_val_b:]

    patience   = int(cfg.get("anomaly_patience", 3))
    best_val   = float("inf")
    best_state = None
    no_improve = 0

    print(f"  Training decoder ({n_epochs} epochs, patience={patience}) …")
    for epoch in range(n_epochs):
        decoder.train()
        if not linear_probe:
            backbone.train()
        total_loss = 0.0
        for batch in train_batches:
            patches = batch[0] if isinstance(batch, (list, tuple)) else batch
            if patches.dim() == 3:
                patches = patches.unsqueeze(-1)
            raw = patches.to(device)
            B, P, PL, C = raw.shape
            with torch.set_grad_enabled(not linear_probe):
                z = _encode(raw)
            recon  = decoder(z)                    # [B, T, C]
            target = raw.reshape(B, P * PL, C)     # [B, T, C]
            loss   = F.mse_loss(recon, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        decoder.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_batches:
                patches = batch[0] if isinstance(batch, (list, tuple)) else batch
                if patches.dim() == 3:
                    patches = patches.unsqueeze(-1)
                raw = patches.to(device)
                B, P, PL, C = raw.shape
                z      = _encode(raw)
                recon  = decoder(z)
                target = raw.reshape(B, P * PL, C)
                val_loss += F.mse_loss(recon, target).item()
        val_loss /= len(val_batches)

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in decoder.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"    Early stopping at epoch {epoch + 1}")
                break
        scheduler.step()
        if epoch % max(1, n_epochs // 5) == 0:
            print(f"    epoch {epoch+1:3d} | "
                  f"train {total_loss / len(train_batches):.6f} | val {val_loss:.6f}")

    if best_state is not None:
        decoder.load_state_dict(best_state)

    # ── (2) Train energy ───────────────────────────────────────────────────────
    decoder.eval()
    train_energy = []
    with torch.no_grad():
        for batch in anomaly_train:
            patches = batch[0] if isinstance(batch, (list, tuple)) else batch
            if patches.dim() == 3:
                patches = patches.unsqueeze(-1)
            raw = patches.to(device)
            B, P, PL, C = raw.shape
            z      = _encode(raw)
            recon  = decoder(z)
            target = raw.reshape(B, P * PL, C)
            score  = F.mse_loss(recon, target, reduction="none").mean(dim=-1)  # [B, T]
            train_energy.append(score.cpu().numpy())
    train_energy = np.concatenate(train_energy).reshape(-1)

    # ── (3) Test energy ────────────────────────────────────────────────────────
    test_energy, all_labels = [], []
    with torch.no_grad():
        for patches, labels in anomaly_test:
            if patches.dim() == 3:
                patches = patches.unsqueeze(-1)
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

    # ── (4) Threshold & predict ────────────────────────────────────────────────
    # Leak-free by default: the threshold is set from the (all-normal) TRAIN
    # reconstruction energy only, so the test set never influences it. Set
    # TS_ANOMALY_THRESH=combined to restore the old TSLib train+test thresholding.
    if os.environ.get("TS_ANOMALY_THRESH", "train") == "combined":
        _thr_pool = np.concatenate([train_energy, test_energy])
    else:
        _thr_pool = train_energy
    threshold = np.percentile(_thr_pool, 100 - anomaly_ratio)
    pred = (test_energy > threshold).astype(int)

    # ── (5) Metrics ────────────────────────────────────────────────────────────
    # All variants (RAW/PA/EVENT/RANGE/AFFIL) via the shared single-source module.
    # Primary f1/precision/recall follow point-adjust, OFF by default (it inflates
    # F1). Set TS_ANOMALY_ADJUST=1 to make the point-adjusted numbers primary.
    point_adjust = os.environ.get("TS_ANOMALY_ADJUST", "0") == "1"
    m = am.compute_all(gt, pred, point_adjust=point_adjust)
    print(f"  [TSDiNO TSMixer] threshold={threshold:.6f}  (top {anomaly_ratio}%)")
    print(am.format_table(m, title="TSDiNO TSMixer Anomaly Detection"))

    return dict(**m, threshold=threshold)

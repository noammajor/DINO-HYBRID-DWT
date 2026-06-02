"""
Supervised training of LMCBackbone on labeled LMC synthetic data.

Loads a pretrained TimeMixer encoder from a DINO checkpoint, wraps it in
LMCBackbone, and trains the cross-attention aggregator + 7 MLP heads to
predict the LMC generation parameters for each time series.

Expected cfg keys:
  # data
  data_dir_labeled    : path to directory containing X.npy / Y.npy
  c_in                : number of channels in the labeled dataset
  seq_len             : time-series length (default 512)

  # encoder (TimeMixer)
  tsmixer_d_model, tsmixer_e_layers, tsmixer_d_ff
  tsmixer_down_sampling_layers, tsmixer_down_sampling_window
  tsmixer_down_sampling_method, tsmixer_decomp_method
  tsmixer_moving_avg, tsmixer_top_k
  dropout

  # checkpoint
  checkpoint_path     : DINO .pth file  (teacher weights); None → random init

  # training
  epochs_labeled      : (default 30)
  lr_labeled          : peak LR        (default 3e-4)
  min_lr_labeled      : min LR at end  (default 1e-5)
  batch_size_labeled  : (default 256)
  val_frac            : (default 0.05)
  test_frac           : (default 0.05)
  num_workers         : (default 4)
  freeze_backbone     : (default True)
  hidden_dim_labeled  : MLP hidden size (default 64)
  min_latent          : (default 2)
  max_latent          : (default 10)
  output_dir_labeled  : checkpoint output dir (default ./lmc_checkpoints)
  seed                : (default 42)
"""

import os
import sys
import math
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from tqdm import tqdm

# ── path setup ────────────────────────────────────────────────────────────────
_HERE           = Path(__file__).parent
_TIMEMIXER_ROOT   = (_HERE / ".." / "TimeMixer-main").resolve()
_TIMEMIXER_MODELS = _TIMEMIXER_ROOT / "models"
for _p in [str(_HERE), str(_TIMEMIXER_ROOT), str(_TIMEMIXER_MODELS)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from TimeMixer import Model as TimeMixerModel  # noqa: E402  # type: ignore
from backbone import LMCBackbone               # noqa: E402
from dataset  import make_loaders              # noqa: E402


# ── encoder construction ──────────────────────────────────────────────────────

def _build_timemixer(cfg: dict) -> TimeMixerModel:
    """Instantiate a full TimeMixer.Model and optionally load pretrained weights.

    task_name='anomaly_detection' is used so TimeMixer builds only the minimal
    encoder-side parameters; the task head (projection_layer) is never called.
    """
    configs = SimpleNamespace(
        task_name                = 'anomaly_detection',
        seq_len                  = cfg.get("seq_len", 512),
        label_len                = 0,    # decoder field; unused in encoder path
        pred_len                 = 0,    # decoder field; unused in encoder path
        enc_in                   = cfg["c_in"],
        c_out                    = cfg["c_in"],
        d_model                  = cfg["tsmixer_d_model"],
        d_ff                     = cfg["tsmixer_d_ff"],
        e_layers                 = cfg["tsmixer_e_layers"],
        dropout                  = cfg.get("dropout", 0.1),
        embed                    = 'timeF',
        freq                     = 'h',
        down_sampling_layers     = cfg["tsmixer_down_sampling_layers"],
        down_sampling_window     = cfg["tsmixer_down_sampling_window"],
        down_sampling_method     = cfg["tsmixer_down_sampling_method"],
        decomp_method            = cfg["tsmixer_decomp_method"],
        moving_avg               = cfg["tsmixer_moving_avg"],
        top_k                    = cfg["tsmixer_top_k"],
        channel_independence     = 1,
        use_norm                 = 1,
        use_future_temporal_feature = 0,
    )
    model = TimeMixerModel(configs)

    ckpt_path = cfg.get("checkpoint_path")
    if ckpt_path and os.path.exists(ckpt_path):
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd  = raw["teacher"]   # EMA teacher weights — best representations

        # Checkpoint keys come from TSMultiCropWrapper(TSMixerForDINO):
        #   module.backbone.pdm_blocks.*  →  strip 'module.' and 'backbone.'
        # TimeMixer.Model shares the same pdm_blocks / enc_embedding / normalize_layers,
        # so the remaining keys map 1-to-1. strict=False silently ignores the
        # DINO-only keys (cls_query, global_attn, mask_token).
        new_sd = {}
        for k, v in sd.items():
            k = k.replace("module.", "")
            if k.startswith("backbone."):
                k = k[len("backbone."):]
            new_sd[k] = v

        missing, unexpected = model.load_state_dict(new_sd, strict=False)
        print(f"✓ Loaded encoder weights from {ckpt_path}")
        print(f"  Matched: {len(new_sd) - len(missing)}  "
              f"Missing: {len(missing)}  Unexpected (DINO-only): {len(unexpected)}")
    else:
        print("  No checkpoint — TimeMixer initialised randomly.")

    return model


# ── loss ──────────────────────────────────────────────────────────────────────

def _compute_loss(
    preds:      dict,
    y:          dict,
    min_latent: int,
    ce_loss:    nn.CrossEntropyLoss,
    huber:      nn.HuberLoss,
) -> tuple[torch.Tensor, dict]:
    """Multi-task loss across all 7 LMC heads.

    Returns (total_loss, per_head_loss_dict).
    Per-head losses are detached scalars for logging only.
    """
    # latent_num stored as the actual integer (e.g. 3); shift to 0-based class
    # index so CrossEntropyLoss sees classes 0…(max_latent - min_latent).
    target_class = y["latent_num"] - min_latent           # [B]  long

    l_latent    = ce_loss(preds["latent_num_logits"], target_class)
    l_wshape    = huber(preds["weibull_shape"],    y["weibull_shape"])
    l_wscale    = huber(preds["weibull_scale"],    y["weibull_scale"])
    l_ess       = huber(preds["ess_length_scale"], y["ess_length_scale"])
    l_dmin      = huber(preds["dirichlet_min"],    y["dirichlet_min"])
    l_dmax      = huber(preds["dirichlet_max"],    y["dirichlet_max"])
    l_dirichlet = huber(preds["dirichlet"],        y["dirichlet"])

    total = l_latent + l_wshape + l_wscale + l_ess + l_dmin + l_dmax + l_dirichlet

    per_head = {
        "latent_num":       l_latent.item(),
        "weibull_shape":    l_wshape.item(),
        "weibull_scale":    l_wscale.item(),
        "ess_length_scale": l_ess.item(),
        "dirichlet_min":    l_dmin.item(),
        "dirichlet_max":    l_dmax.item(),
        "dirichlet":        l_dirichlet.item(),
    }
    return total, per_head


# ── train / eval loops ────────────────────────────────────────────────────────

def _run_epoch(
    model:      LMCBackbone,
    loader:     torch.utils.data.DataLoader,
    optimizer,
    scheduler,
    ce_loss:    nn.CrossEntropyLoss,
    huber:      nn.HuberLoss,
    device:     torch.device,
    min_latent: int,
    training:   bool,
) -> dict:
    """One full pass over loader. Returns mean per-head losses."""
    model.train(training)
    ctx = torch.enable_grad() if training else torch.no_grad()

    totals   = {}
    n_batches = 0

    with ctx:
        for x, y in tqdm(loader, desc="train" if training else "eval ", leave=False):
            x = x.to(device, non_blocking=True)                        # [B, T, C]
            y = {k: v.to(device, non_blocking=True) for k, v in y.items()}

            # Teacher forcing during training: pass ground-truth d_min / d_max
            # so d_max and dirichlet heads learn without upstream prediction error.
            # At eval, None triggers the detach-and-chain inference path.
            t_dmin = y["dirichlet_min"] if training else None
            t_dmax = y["dirichlet_max"] if training else None

            preds = model(x, teacher_d_min=t_dmin, teacher_d_max=t_dmax)
            loss, per_head = _compute_loss(preds, y, min_latent, ce_loss, huber)

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

            for k, v in per_head.items():
                totals[k] = totals.get(k, 0.0) + v
            n_batches += 1

    return {k: v / n_batches for k, v in totals.items()}


# ── main ──────────────────────────────────────────────────────────────────────

def train_lmc(cfg: dict):
    """Full training run."""
    gpu    = cfg.get("gpu", 0)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    seed       = cfg.get("seed", 42)
    min_latent = cfg.get("min_latent", 2)
    max_latent = cfg.get("max_latent", 10)
    epochs     = cfg.get("epochs_labeled", 30)
    lr         = cfg.get("lr_labeled", 3e-4)
    min_lr     = cfg.get("min_lr_labeled", 1e-5)
    # Save alongside DINO checkpoints in the same output_dir, suffixed with _labeldata.
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)

    # ── data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = make_loaders(
        data_dir    = cfg["data_dir_labeled"],
        batch_size  = cfg.get("batch_size_labeled", 256),
        val_frac    = cfg.get("val_frac",  0.05),
        test_frac   = cfg.get("test_frac", 0.05),
        num_workers = cfg.get("num_workers", 4),
        seed        = seed,
    )
    print(f"  Train: {len(train_loader.dataset):,}  "
          f"Val: {len(val_loader.dataset):,}  "
          f"Test: {len(test_loader.dataset):,}")

    # ── model ─────────────────────────────────────────────────────────────────
    encoder = _build_timemixer(cfg)
    model   = LMCBackbone(
        backbone        = encoder,
        hidden_dim      = cfg.get("hidden_dim_labeled", 64),
        min_latent      = min_latent,
        max_latent      = max_latent,
        freeze_backbone = cfg.get("freeze_backbone", True),
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Params — trainable: {trainable:,} / total: {total:,}")

    # ── losses ────────────────────────────────────────────────────────────────
    ce_loss = nn.CrossEntropyLoss()
    huber   = nn.HuberLoss(delta=1.0)

    # ── optimiser ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4,
    )

    # Cosine LR with short linear warmup.
    total_steps  = epochs * len(train_loader)
    warmup_steps = int(0.05 * total_steps)

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine   = 0.5 * (1 + math.cos(math.pi * progress))
        return (min_lr + cosine * (lr - min_lr)) / lr

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    saveckp_freq = cfg.get("saveckp_freq", 1)
    log_path     = output_dir / "log_labeldata.txt"

    # ── training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        train_losses = _run_epoch(
            model, train_loader, optimizer, scheduler,
            ce_loss, huber, device, min_latent, training=True,
        )
        val_losses = _run_epoch(
            model, val_loader, None, None,
            ce_loss, huber, device, min_latent, training=False,
        )

        train_total = sum(train_losses.values())
        val_total   = sum(val_losses.values())
        current_lr  = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch:3d}/{epochs}  "
              f"train={train_total:.4f}  val={val_total:.4f}  "
              f"lr={current_lr:.2e}")
        print("  train  " +
              "  ".join(f"{k[:8]}={v:.4f}" for k, v in train_losses.items()))
        print("  val    " +
              "  ".join(f"{k[:8]}={v:.4f}" for k, v in val_losses.items()))

        # Mirror DINO's log.txt — one JSON line per epoch.
        log_stats = {
            "epoch": epoch,
            "train_loss": train_total,
            "val_loss": val_total,
            "lr": current_lr,
            **{f"train_{k}": v for k, v in train_losses.items()},
            **{f"val_{k}":   v for k, v in val_losses.items()},
        }
        with log_path.open("a") as f:
            f.write(str(log_stats) + "\n")

        save_dict = {
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss":  val_total,
            "cfg":       cfg,
        }

        # Periodic checkpoint — mirrors DINO's saveckp_freq behaviour.
        if saveckp_freq and epoch % saveckp_freq == 0:
            torch.save(save_dict, output_dir / f"checkpoint{epoch}_labeldata.pth")

        if val_total < best_val_loss:
            best_val_loss = val_total
            torch.save(save_dict, output_dir / "checkpoint_best_labeldata.pth")
            print(f"  → saved checkpoint_best_labeldata.pth  (val={val_total:.4f})")

    # ── test ──────────────────────────────────────────────────────────────────
    ckpt = torch.load(output_dir / "checkpoint_best_labeldata.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    test_losses = _run_epoch(
        model, test_loader, None, None,
        ce_loss, huber, device, min_latent, training=False,
    )
    test_total = sum(test_losses.values())
    print(f"\nTest  total={test_total:.4f}")
    print("  " + "  ".join(f"{k[:8]}={v:.4f}" for k, v in test_losses.items()))

    return model


if __name__ == "__main__":
    import argparse

    _HERE = Path(__file__).parent
    sys.path.insert(0, str(_HERE))
    sys.path.insert(0, str(_HERE.parent / "TSDiNO"))

    from TSDiNO.config import config as _dino_cfg  # noqa: E402

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
    p.add_argument("--freeze_backbone", type=lambda x: x.lower() != "false", default=False)
    p.add_argument("--min_latent",    type=int,   default=2)
    p.add_argument("--max_latent",    type=int,   default=10)
    p.add_argument("--val_frac",      type=float, default=0.05)
    p.add_argument("--test_frac",     type=float, default=0.05)
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--saveckp_freq",  type=int,   default=1)
    p.add_argument("--gpu",           type=int,   default=0)
    p.add_argument("--seed",          type=int,   default=42)
    a = p.parse_args()

    cfg = dict(_dino_cfg)
    cfg["c_in"]              = a.c_in or cfg.get("c_in", 7)
    cfg["seq_len"]           = a.seq_len
    cfg["data_dir_labeled"]  = a.data_dir
    cfg["checkpoint_path"]   = a.checkpoint
    cfg["epochs_labeled"]    = a.epochs
    cfg["lr_labeled"]        = a.lr
    cfg["min_lr_labeled"]    = a.min_lr
    cfg["batch_size_labeled"]= a.batch_size
    cfg["hidden_dim_labeled"]= a.hidden_dim
    cfg["freeze_backbone"]   = a.freeze_backbone
    cfg["min_latent"]        = a.min_latent
    cfg["max_latent"]        = a.max_latent
    cfg["val_frac"]          = a.val_frac
    cfg["test_frac"]         = a.test_frac
    cfg["num_workers"]       = a.num_workers
    cfg["saveckp_freq"]      = a.saveckp_freq
    cfg["gpu"]               = a.gpu
    cfg["seed"]              = a.seed

    print("=" * 60)
    print("  LMC Pretraining")
    print("=" * 60)
    print(f"  data_dir   : {cfg['data_dir_labeled']}")
    print(f"  checkpoint : {cfg['checkpoint_path'] or 'random init'}")
    print(f"  output_dir : {cfg['output_dir']}  (suffix: _labeldata)")
    print(f"  encoder    : d_model={cfg['tsmixer_d_model']}  "
          f"layers={cfg['tsmixer_e_layers']}  "
          f"scales={cfg['tsmixer_down_sampling_layers'] + 1}")
    print(f"  training   : epochs={cfg['epochs_labeled']}  "
          f"lr={cfg['lr_labeled']}  batch={cfg['batch_size_labeled']}")
    print(f"  backbone   : {'frozen' if cfg['freeze_backbone'] else 'unfrozen (pretraining)'}")
    print("=" * 60)

    train_lmc(cfg)

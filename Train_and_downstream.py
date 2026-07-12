"""
Unified training + forecasting runner for:
  - dino_timemixer  (wavelet-based DINO, TimeMixer backbone)
  - dino_patchtst   (wavelet-based DINO, PatchTST backbone)
  - dino_ts2vec     (wavelet-based DINO, TS2Vec dilated-conv backbone)
  - patchtst        (MAE)

Usage
-----
  python Train_and_downstream.py --model dino_timemixer
  python Train_and_downstream.py --model dino_patchtst
  python Train_and_downstream.py --model patchtst

Colab
-----
  !python Train_and_downstream.py --model dino_timemixer
  or call run(model="dino_timemixer", skip_train=False) directly after importing.
"""

import os, sys, copy, argparse, random
import functools
import subprocess
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch

# Make sure the project root (where dataset_registry.py lives) is importable
_PROJECT_ROOT = str(Path(__file__).parent.resolve())
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dataset_registry import get_dataset_info
from data_paths import DATA_PATHS

GLOBAL_SEED = 42
_SEED_TAG   = ''   # set by run() when seed is provided; used by runners to suffix checkpoint paths

# Per-dataset anomaly detection hyperparameters matching TSLib reference configs
_ANOMALY_RATIO = {
    "SMD":  0.5,   # TSLib uses 0.5 for SMD
    "MSL":  1.0,
    "SMAP": 1.0,
    "PSM":  1.0,
    "SWaT": 1.0,
}

def _get_anomaly_ratio(dataset: str, cfg: dict) -> float:
    """Return TSLib-matched anomaly ratio, falling back to config or 1.0."""
    return _ANOMALY_RATIO.get(dataset, cfg.get("anomaly_ratio", 1.0))

def _set_seed(seed: int = GLOBAL_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── helpers ──────────────────────────────────────────────────────────────────

def _add_path(p):
    """Prepend p to sys.path if not already present."""
    p = str(Path(p).resolve())
    if p not in sys.path:
        sys.path.insert(0, p)

def _get_forecast_bs(config, default=256):
    """Return forecast batch size — env var TS_FORECAST_BS (set by run_layer_forecast.py) takes priority."""
    env = os.environ.get("TS_FORECAST_BS")
    if env is not None:
        return int(env)
    return config.get("batch_size_forecast", default)

def _get_cls_bs(config, key="batch_size", default=64):
    """Return classification batch size — env var TS_CLS_BS takes priority over config."""
    env = os.environ.get("TS_CLS_BS")
    if env is not None:
        return int(env)
    return config.get(key, default)

def _get_forecast_lr(config, key="lr_forecasting", default=2e-4):
    """Return forecast LR scaled by TS_FORECAST_LR_SCALE (set by run_layer_forecast.py)."""
    base = config.get(key, config.get("lr_forcasting", default))
    scale = float(os.environ.get("TS_FORECAST_LR_SCALE", 1.0))
    return base * scale

def _resolve_pretrain_source(config):
    """Return 'monash', 'synthetic', or 'monash+synthetic' (or None for CSV pretraining).

    Checks the explicit 'pretrain_source' key first (set this in your config).
    Falls back to the legacy pretrain_on_monash + synthetic_data_dir flags.
    """
    if 'pretrain_source' in config:
        return config['pretrain_source']   # None means CSV-only
    if config.get('pretrain_on_monash', False):
        return 'monash+synthetic' if config.get('synthetic_data_dir') else 'monash'
    return None

def _config_to_dino_args(cfg):
    """
    Convert the DINO config dict (TSDINOALT 4/config.py) into the
    SimpleNamespace that train_TS_DINO / test_run expect.
    """
    local_crops = cfg.get("local_crops", [])
    global_crops = cfg.get("global_crops", [])

    args = SimpleNamespace(
        # ── task ──────────────────────────────────────────────────────────
        task                        = cfg.get("task", "dino"),
        test_only                   = cfg.get("test_only", False),
        seed                        = cfg.get("seed", GLOBAL_SEED),
        output_dir                  = cfg.get("output_dir", "./checkpoints"),
        saveckp_freq                = cfg.get("saveckp_freq", 10),

        # ── data ──────────────────────────────────────────────────────────
        data_path                   = cfg.get("data_path", ""),
        data_path_forecast_training = cfg.get("data_path_forecast_training", ""),
        data_path_forecast_test     = cfg.get("data_path_forecast_test", ""),
        data_path_classification    = cfg.get("data_path_classification", "UCI HAR Dataset"),
        num_workers                 = cfg.get("num_workers", 4),
        batch_size_per_gpu          = cfg.get("batch_size_per_gpu", 64),
        batch_size_forecast         = _get_forecast_bs(cfg, 256),

        # ── model architecture ────────────────────────────────────────────
        c_in                        = cfg.get("c_in", 7),
        ts2vec_hidden_dims          = cfg.get("ts2vec_hidden_dims", 64),  # TS2Vec backbone only
        patch_len                   = cfg.get("patch_len", 12),
        step_size                   = cfg.get("step_size", 12),
        num_patches                 = cfg.get("num_patches", 32),
        n_layers                    = cfg.get("n_layers", 5),
        tsmixer_e_layers            = cfg.get("tsmixer_e_layers", 3),  # actual TimeMixer/tsmixer backbone depth
        n_heads                     = cfg.get("n_heads", 16),
        embed_dim                   = cfg.get("embed_dim", 128),
        d_ff                        = cfg.get("d_ff", 512),
        dropout                     = cfg.get("dropout", 0.1),
        head_dropout                = cfg.get("head_dropout", 0.1),
        head_dropout_forecasting    = cfg.get("head_dropout_forecasting", 0.2),
        drop_path_rate              = cfg.get("drop_path_rate", 0.1),

        # ── DINO head ─────────────────────────────────────────────────────
        out_dim                     = cfg.get("out_dim", 20000),
        use_bn_in_head              = cfg.get("use_bn_in_head", False),
        norm_last_layer             = cfg.get("norm_last_layer", True),

        # ── DINO loss / temperatures ──────────────────────────────────────
        warmup_teacher_temp         = cfg.get("warmup_teacher_temp", 0.04),
        teacher_temp                = cfg.get("teacher_temp", 0.04),
        warmup_teacher_temp_epochs  = cfg.get("warmup_teacher_temp_epochs", 0),

        # ── EMA teacher ───────────────────────────────────────────────────
        momentum_teacher            = cfg.get("momentum_teacher", 0.9995),

        # ── optimizer ─────────────────────────────────────────────────────
        optimizer                   = cfg.get("optimizer", "adamw"),
        lr                          = cfg.get("lr", 0.0005),
        min_lr                      = cfg.get("min_lr", 1e-6),
        warmup_epochs               = cfg.get("warmup_epochs", 10),
        weight_decay                = cfg.get("weight_decay", 0.04),
        weight_decay_end            = cfg.get("weight_decay_end", 0.4),
        clip_grad                   = cfg.get("clip_grad", 3.0),
        use_fp16                    = cfg.get("use_fp16", False),
        freeze_last_layer           = cfg.get("freeze_last_layer", 1),

        # ── training schedule ─────────────────────────────────────────────
        epochs                      = cfg.get("epochs", 20),

        # ── augmentation (derived from crop specs) ────────────────────────
        # local_crops_number  = crop ratio of the first local crop
        # transformation_group_size = total number of local crops
        local_crops_number          = local_crops[0]["crop_ratio"] if local_crops else 0.5,
        transformation_group_size   = len(local_crops) if local_crops else 2,

        # ── distributed (defaults for single-GPU / CPU) ───────────────────
        dist_url                    = cfg.get("dist_url", "env://"),
        gpu                         = None,
        rank                        = 0,
        world_size                  = 1,
        dist_backend                = "nccl",

        # ── downstream: forecasting ───────────────────────────────────────
        pred_len                            = cfg.get("pred_len", 96),
        epochs_forecasting                  = cfg.get("epochs_forecasting", 10),
        lr_forecasting                      = _get_forecast_lr(cfg, "lr_forecasting", 0.001),
        min_lr_forecasting                  = cfg.get("min_lr_forecasting", 1e-5),
        parms_for_training_forecasting      = cfg.get("parms_for_training_forecasting", []),
        parms_for_testing_forecasting       = cfg.get("parms_for_testing_forecasting", []),
        path_num                            = cfg.get("path_num", 0),

        # ── downstream: classification ────────────────────────────────────
        n_classes                   = cfg.get("n_classes", 10),
        epochs_classification       = cfg.get("epochs_classification", 50),
        lr_classification           = cfg.get("lr_classification", 0.001),
        lr_classification_encoder   = cfg.get("lr_classification_encoder", None),
        min_lr_classification       = cfg.get("min_lr_classification", 1e-6),
        batch_size_classification   = cfg.get("batch_size_classification", 64),
        seq_len_classification      = cfg.get("seq_len_classification", 128),
        c_in_classification         = cfg.get("c_in_classification", 9),
        mlm_phi                     = cfg.get("mlm_phi", 0.0),
        mlm_mask_ratio              = cfg.get("mlm_mask_ratio", 0.4),
        ibot_out_dim                = cfg.get("ibot_out_dim", cfg.get("out_dim", 65536)),
        backbone_type               = cfg.get("backbone_type", "tsmixer"),  # TSDiNO is TimeMixer-only
    )
    return args


# ── DINO ──────────────────────────────────────────────────────────────────────

def run_dino(skip_train: bool = False,
             pretrain_dataset: str = None,
             forecast_dataset: str = None,
             classification_dataset=None,
             anomaly_dataset: str = None,
             pred_lens=None,
             checkpoints=None,
             pretrain_only: bool = False,
             classification_only: bool = False,
             pretrain_on_classification: bool = False,
             pretrain_on_anomaly: bool = False,
             pretrain_val_fraction: float = 0.1,
             epochs_classification: int = None,
             cls_head_mode: str = "both",
             lr_classification: float = None,
             lr_classification_encoder: float = None,
             label_smoothing: float = None,
             cls_kfold: int = None,
             encoder_layers: int = None,
             predictor_layers: int = None,
             lr: float = None,
             pretrain_source: str = None,
             checkpoint: str = None,
             num_patches: int = None,
             seq_len: int = None,
             seed: int = None,
             linear_probe: bool = True,
             head_type: str = "linear",
             output_dir: str = None,
             embed_dim: int = None,
             out_dim: int = None,
             d_ff: int = None,
             n_heads: int = None,
             epochs: int = None,
             epochs_forecasting: int = None,
             warmup_epochs: int = None,
             ckpt_tag: str = None,
             aug_global: str = None,
             aug_local: str = None,
             n_global_crops: int = None,
             n_local_crops: int = None,
             global_crop_ratio: float = None,
             local_crop_ratio: float = None,
             mlm_phi: float = None,
             mlm_mode: str = None,
             mlm_block_size: int = None,
             backbone_type: str = None,
             dwt_wavelet_pool: list = None,
             soft_threshold_sigma: float = None,
             use_koleo: bool = None,
             koleo_weight: float = None,
             use_vicreg: bool = None,
             vicreg_std_coeff: float = None,
             vicreg_cov_coeff: float = None,
             synthetic_data_dir: str = None,
             subset_frac: float = None,
             window_stride: int = None,
             lr_forecasting: float = None,
             batch_size: int = None,
             tsmixer_e_layers: int = None,
             patch_len: int = None,
             backbone: str = "timemixer"):
    if backbone not in ("timemixer", "patchtst", "ts2vec", "timesnet", "timerxl", "itransformer"):
        raise ValueError(f"run_dino: unknown backbone '{backbone}' (expected 'timemixer', 'patchtst', 'ts2vec', 'timesnet', 'timerxl' or 'itransformer')")
    root_dir   = Path(__file__).parent
    _dino_dirs = {"timemixer": "tsdino_timemixer", "patchtst": "tsdino_patchtst", "ts2vec": "tsdino_ts2vec", "timesnet": "tsdino_timesnet", "timerxl": "tsdino_timerxl", "itransformer": "tsdino_itransformer"}
    dino_dir   = root_dir / _dino_dirs[backbone]
    shared_dir = root_dir / "shared"
    _add_path(root_dir)          # so `from tsdino_common import …` resolves inside main.py
    _add_path(dino_dir)
    _add_path(shared_dir)

    import sys as _sys, importlib.util as _ilu

    # Load TSDiNO config.py directly by file path — avoids any sys.path collision
    _cfg_spec = _ilu.spec_from_file_location("_dino_config", dino_dir / "config.py")
    _cfg_mod  = _ilu.module_from_spec(_cfg_spec)
    _cfg_spec.loader.exec_module(_cfg_mod)
    dino_cfg  = {**DATA_PATHS, **dict(_cfg_mod.config)}

    # Both backbone configs default output_dir to ./checkpoints — keep the patchtst
    # backbone's checkpoints in a separate tree so the two never collide.
    if backbone in ("patchtst", "ts2vec", "timesnet", "timerxl", "itransformer"):
        _od = dino_cfg.get('output_dir', './checkpoints').rstrip('/')
        if Path(_od).name == 'checkpoints':
            dino_cfg['output_dir'] = str(Path(_od).parent / f'checkpoints_{backbone}')

    # Inject under the bare name so that main.py's `from config import config as cfg` resolves
    # to our freshly loaded version, not whatever 'config' may already be cached in sys.modules.
    _sys.modules["config"] = _cfg_mod

    # Both backbones ship a top-level `models` package (ts_mixer_backbone vs patchTST).
    # Evict any stale `main`/`models*`/`data_agumentation`/`dataPuller` so switching
    # backbones within one process (e.g. a notebook) loads the right variant's modules.
    for _m in [m for m in _sys.modules
               if m == "main" or m == "models" or m.startswith("models.")
               or m in ("data_agumentation", "dataPuller")]:
        _sys.modules.pop(_m, None)
    _main_spec = _ilu.spec_from_file_location("_dino_main", dino_dir / "main.py")
    dino_main  = _ilu.module_from_spec(_main_spec)
    _sys.modules["main"] = dino_main                     # register before exec (handles internal refs)
    _main_spec.loader.exec_module(dino_main)

    if pred_lens is None:
        pred_lens = [96, 192, 336, 720]

    dino_cfg = dict(dino_cfg)
    if pretrain_source is not None:
        dino_cfg['pretrain_source'] = pretrain_source
    elif pretrain_dataset is not None:
        dino_cfg['pretrain_source'] = None  # None propagates into dino_main.cfg via update()
    if output_dir is not None:
        dino_cfg['output_dir'] = output_dir
    if embed_dim is not None:
        dino_cfg['embed_dim'] = embed_dim
    if out_dim is not None:
        dino_cfg['out_dim'] = out_dim
    if d_ff is not None:
        dino_cfg['d_ff'] = d_ff
    if n_heads is not None:
        dino_cfg['n_heads'] = n_heads
    if tsmixer_e_layers is not None:
        dino_cfg['tsmixer_e_layers'] = tsmixer_e_layers
    if patch_len is not None:
        dino_cfg['patch_len'] = patch_len
        dino_cfg['step_size'] = patch_len   # non-overlapping patches
    if epochs is not None:
        dino_cfg['epochs'] = epochs
    if epochs_forecasting is not None:
        dino_cfg['epochs_forecasting'] = epochs_forecasting
    # Classification-head tuning knobs — propagate for the STANDARD classify path
    # (the pretrain_on_classification branch sets its own copies below).
    if lr_classification is not None:
        dino_cfg['lr_classification'] = lr_classification
    if lr_classification_encoder is not None:
        dino_cfg['lr_classification_encoder'] = lr_classification_encoder
    if epochs_classification is not None:
        dino_cfg['epochs_classification'] = epochs_classification
    if encoder_layers is not None:
        dino_cfg['n_layers'] = encoder_layers
        _psrc = dino_cfg.get('pretrain_source')
        if _psrc and _psrc != 'monash':
            _src_tag = f"_{_psrc.replace('+', '_')}"
        elif not _psrc and pretrain_dataset:
            _src_tag = f"_{pretrain_dataset}"
        else:
            _src_tag = ''
        _outdim_tag = f"_outdim{dino_cfg['out_dim']}" if dino_cfg.get('out_dim') is not None else ''
        _bbone = backbone_type or dino_cfg.get('backbone_type') or ('tsmixer' if backbone == 'timemixer' else backbone)
        _ckpt_tag = f"_{ckpt_tag}" if ckpt_tag else (f"_{_bbone}" if _bbone == 'tsmixer' else '')
        dino_cfg['output_dir'] = dino_cfg.get('output_dir', './checkpoints').rstrip('/') + f'{_src_tag}_layers{encoder_layers}{_outdim_tag}{_ckpt_tag}' + _SEED_TAG
    if seq_len is not None:
        _patch_len = dino_cfg.get('patch_len', 16)
        dino_cfg['num_patches'] = seq_len // _patch_len
        # Also drive the anomaly window (AnomalyDataPuller win_size reads cfg['seq_len']).
        # Without this it falls back to 512 → 32 patches, mismatching the backbone's
        # num_patches (e.g. 72) positional embeddings and crashing the forward pass.
        dino_cfg['seq_len'] = seq_len
    if num_patches is not None:
        dino_cfg['num_patches'] = num_patches
        if classification_dataset is not None:
            _cw = num_patches * dino_cfg.get('patch_len', 16)
            _base = dino_cfg.get('output_dir', './checkpoints').rstrip('/')
            dino_cfg['output_dir'] = str(Path(_base).parent / 'classification' / (Path(_base).name + f'_cw{_cw}'))
    if lr is not None:
        dino_cfg['lr'] = lr
    if warmup_epochs is not None:
        dino_cfg['warmup_epochs'] = warmup_epochs
    if aug_global is not None:
        _ng  = n_global_crops if n_global_crops is not None else 1
        _gcr = global_crop_ratio if global_crop_ratio is not None else 1.0
        dino_cfg['global_crops'] = [{"type": aug_global, "crop_ratio": _gcr} for _ in range(_ng)]
    if aug_local is not None:
        _nl  = n_local_crops if n_local_crops is not None else 1
        _lcr = local_crop_ratio if local_crop_ratio is not None else 1.0
        dino_cfg['local_crops']  = [{"type": aug_local, "crop_ratio": _lcr} for _ in range(_nl)]
    if mlm_phi is not None:
        dino_cfg['mlm_phi'] = mlm_phi
    if mlm_mode is not None:
        dino_cfg['mlm_mode'] = mlm_mode
    if mlm_block_size is not None:
        dino_cfg['mlm_block_size'] = mlm_block_size
    if batch_size is not None:
        dino_cfg['batch_size_per_gpu'] = batch_size
    if backbone_type is not None:
        dino_cfg['backbone_type'] = backbone_type
    if synthetic_data_dir is not None:
        dino_cfg['synthetic_data_dir'] = synthetic_data_dir
    if subset_frac is not None:
        dino_cfg['pretrain_subset_frac'] = subset_frac
    if window_stride is not None:
        dino_cfg['window_stride'] = window_stride
        dino_cfg['window_step']   = window_stride   # tsdino mains read 'window_step' for the arrow puller stride
    if dwt_wavelet_pool is not None:
        dino_cfg['dwt_wavelet_pool'] = dwt_wavelet_pool
    if soft_threshold_sigma is not None:
        # ρ (shrinkage ratio) for soft-threshold DWT/SWT/MODWT augmentation:
        # threshold = ρ · max(|detail coeffs|) per level.
        dino_cfg['dwt_soft_threshold_sigma'] = soft_threshold_sigma
    if use_koleo is not None:
        dino_cfg['use_koleo'] = use_koleo
    if koleo_weight is not None:
        dino_cfg['koleo_weight'] = koleo_weight
    if use_vicreg is not None:
        dino_cfg['use_vicreg'] = use_vicreg
    if vicreg_std_coeff is not None:
        dino_cfg['vicreg_std_coeff'] = vicreg_std_coeff
    if vicreg_cov_coeff is not None:
        dino_cfg['vicreg_cov_coeff'] = vicreg_cov_coeff
    if seed is not None:
        dino_cfg['seed'] = seed

    # ── pretrain ON the classification data, then probe + fine-tune ────────────
    # Self-supervised DINO on the classification dataset's TRAIN series (no labels),
    # then TWO downstream runs from the same checkpoint: a linear probe (frozen
    # backbone) and a full fine-tune. seq_len is sized PER DATASET from the series
    # length so we don't up-sample short series to a fixed window.
    if pretrain_on_classification:
        if backbone != "timemixer":
            raise ValueError("pretrain_on_classification is only wired for backbone='timemixer'")
        if classification_dataset is None:
            raise ValueError("pretrain_on_classification=True requires classification_dataset")
        import torch.nn.functional as _F
        _shared_dir = str(root_dir / "shared")
        if _shared_dir not in sys.path:
            sys.path.insert(0, _shared_dir)
        from data_loaders.data_puller import ClassificationDataPuller, UEADataset

        cls_dir = dino_cfg["classification_data_dir"]
        cls_bs  = _get_cls_bs(dino_cfg, "batch_size_classification", 64)
        p_s     = dino_cfg.get("patch_len", 16)
        _cls_path = Path(cls_dir) / classification_dataset
        _is_uea   = _cls_path.exists() and bool(list(_cls_path.glob("*_TRAIN.ts")))

        # PULL mode: --checkpoint given → skip SSL, load an existing backbone and
        # fine-tune on it. The cls window is FIXED to the backbone's pretrain window
        # (num_patches × patch_len, default 21×16=336) because TSMixer PDM layers are
        # seq_len-fixed; cls series are padded/truncated to it.
        _pull = checkpoint is not None
        _pull_np  = num_patches if num_patches is not None else dino_cfg.get('num_patches', 21)
        _pull_seq = _pull_np * p_s

        if _is_uea:
            _ds_train = UEADataset(str(_cls_path), classification_dataset, split="train")
            _ds_test  = UEADataset(str(_cls_path), classification_dataset, split="test",
                                   _shared=_ds_train)
            n_classes = _ds_train.n_classes
            n_vars    = _ds_train._samples[0].shape[-1]
            if _pull:
                seq_len   = _pull_seq                        # match the pulled backbone
                n_patches = _pull_np
            else:
                _max_T    = max(s.shape[0] for s in _ds_train._samples + _ds_test._samples)
                seq_len   = int(np.ceil(_max_T / p_s)) * p_s     # per-dataset window
                n_patches = seq_len // p_s

            def _cls_collate(batch, _ps=p_s, _sl=seq_len, _nP=n_patches):
                xs, ys, orig_lens = zip(*batch)
                orig_lens = torch.stack(orig_lens)                        # (B,)
                xs = torch.stack([
                    x[:_sl] if x.shape[0] >= _sl
                    else _F.pad(x, (0, 0, 0, _sl - x.shape[0]))
                    for x in xs
                ])                                                        # [B, seq_len, C]
                patch_starts = torch.arange(_nP) * _ps
                padding_mask = patch_starts.unsqueeze(0) < orig_lens.unsqueeze(1)   # [B, P]
                xs = xs.reshape(len(batch), _nP, _ps, xs.shape[-1])       # [B, P, PL, C]
                return xs, torch.stack(ys), padding_mask

            cls_train = torch.utils.data.DataLoader(
                _ds_train, batch_size=cls_bs, shuffle=True,  collate_fn=_cls_collate)
            cls_test  = torch.utils.data.DataLoader(
                _ds_test,  batch_size=cls_bs, shuffle=False, collate_fn=_cls_collate)
        else:
            if _pull:
                raise NotImplementedError(
                    "Pull mode (--checkpoint) is wired for UEA .ts datasets only; "
                    f"'{classification_dataset}' is npy/pt format.")
            _tr = ClassificationDataPuller(cls_dir, classification_dataset, p_s, which="train")
            _te = ClassificationDataPuller(cls_dir, classification_dataset, p_s, which="test")
            n_classes = _tr.n_classes
            n_vars    = _tr.X.shape[2]
            seq_len   = _tr.X.shape[1]                        # already padded per dataset
            n_patches = _tr.n_patches
            cls_train = torch.utils.data.DataLoader(_tr, batch_size=cls_bs, shuffle=True)
            cls_test  = torch.utils.data.DataLoader(_te, batch_size=cls_bs, shuffle=False)

        # Configure DINO for this dataset (seq_len drives both pretrain + probe backbone)
        dino_cfg['pretrain_classification_dataset'] = classification_dataset
        dino_cfg['pretrain_source'] = None
        dino_cfg['pretrain_val_fraction'] = pretrain_val_fraction
        if epochs_classification is not None:
            # key actually read by TSMixerClassification.classification()
            dino_cfg['epoch_classification']  = epochs_classification
            dino_cfg['epochs_classification'] = epochs_classification
        # downstream head tuning: discriminative LR + label smoothing + best-epoch select
        dino_cfg['cls_best_epoch'] = True
        if lr_classification is not None:
            dino_cfg['lr_classification'] = lr_classification
        if lr_classification_encoder is not None:
            dino_cfg['lr_classification_encoder'] = lr_classification_encoder
        if label_smoothing is not None:
            dino_cfg['label_smoothing'] = label_smoothing
        dino_cfg['c_in']        = n_vars
        dino_cfg['seq_len']     = seq_len
        dino_cfg['num_patches'] = n_patches
        dino_cfg['patch_len']   = p_s
        dino_cfg['step_size']   = p_s
        _base = dino_cfg.get('output_dir', './checkpoints').rstrip('/')
        if f"clspre_{classification_dataset}" not in _base:
            dino_cfg['output_dir'] = str(
                Path(_base).parent / 'classification' /
                (Path(_base).name + f'_clspre_{classification_dataset}_cw{seq_len}'))

        dino_main.cfg.update(dino_cfg)
        args = _config_to_dino_args(dino_cfg)
        args.linear_probe = linear_probe
        args.mlp_head     = (head_type == "mlp")
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 60)
        _mode_hdr = "PULL EXISTING BACKBONE (no pretrain)" if _pull else "PRETRAIN ON CLASSIFICATION DATA"
        print(f"  MODEL: DINO (tsdino_timemixer) — {_mode_hdr}")
        print(f"  dataset: {classification_dataset}   n_vars={n_vars}  "
              f"n_classes={n_classes}  seq_len={seq_len} ({n_patches}×{p_s})")
        if _pull:
            print(f"  backbone: {os.path.abspath(checkpoint)}")
        else:
            _n_train = len(cls_train.dataset)
            _n_val_est = int(round(_n_train * pretrain_val_fraction))
            _val_note = (f"holdout {_n_val_est} for best-by-val"
                         if _n_val_est >= 32 else "too small → final-epoch checkpoint")
            print(f"  pretrain val: frac={pretrain_val_fraction} ({_val_note})")
            print(f"  output_dir: {args.output_dir}")
        print("=" * 60)

        if _pull:
            _ckpt = os.path.abspath(checkpoint)
            if not os.path.exists(_ckpt):
                raise FileNotFoundError(f"--checkpoint not found: {_ckpt}")
            print(f"\n[DINO] Pull mode — no pretraining; loading backbone {_ckpt}")
        else:
            if not skip_train:
                print("\n[DINO] Pretraining on classification train series …")
                dino_main.train_TS_DINO(args)
            else:
                print("[DINO] skip_train=True — reusing existing checkpoint.")
            _ckpt = os.path.join(args.output_dir, "checkpoint_best.pth")
            if not os.path.exists(_ckpt):
                _epochs_ckpts = sorted(Path(args.output_dir).glob("checkpoint*.pth"))
                if _epochs_ckpts:
                    _ckpt = str(_epochs_ckpts[-1])
                else:
                    raise FileNotFoundError(f"No pretrained checkpoint found in {args.output_dir}")

        _tm_cls_spec = _ilu.spec_from_file_location(
            "tsmixer_classification", dino_dir / "TSMixerClassification.py")
        _tm_cls_mod  = _ilu.module_from_spec(_tm_cls_spec)
        _tm_cls_spec.loader.exec_module(_tm_cls_mod)

        _all_modes = {"linear_probe": True, "fine_tune": False}
        if cls_head_mode == "both":
            _modes = [("linear_probe", True), ("fine_tune", False)]
        elif cls_head_mode in ("fine_tune", "finetune"):
            _modes = [("fine_tune", False)]
        elif cls_head_mode in ("linear_probe", "probe"):
            _modes = [("linear_probe", True)]
        else:
            raise ValueError(f"cls_head_mode must be both|fine_tune|linear_probe, got '{cls_head_mode}'")

        _kfolds = max(1, int(cls_kfold or 1))
        results = {}
        for _name, _lp in _modes:
            print(f"\n{'='*60}\n  [DINO] Classification ({_name}) on {classification_dataset}"
                  f"{f'  [{_kfolds}-fold val]' if _kfolds > 1 else ''}\n{'='*60}")
            if _kfolds > 1:
                import statistics as _stat
                _accs = []
                for _f in range(_kfolds):
                    acc_f = _tm_cls_mod.classification(
                        dino_cfg, _ckpt, cls_train, None, cls_test, n_classes,
                        linear_probe=_lp, mlp_head=(head_type == "mlp"),
                        cv_fold=_f, cv_folds=_kfolds)
                    _accs.append(acc_f)
                    print(f"  [{_name}] fold {_f+1}/{_kfolds}: test acc={acc_f:.4f}")
                _mean = sum(_accs) / len(_accs)
                _std  = _stat.pstdev(_accs) if len(_accs) > 1 else 0.0
                results[_name] = _mean
                results[_name + "_std"] = _std
                print(f"  [{_name}] {_kfolds}-fold test: mean={_mean:.4f} ± {_std:.4f}  "
                      f"(min={min(_accs):.4f}  max={max(_accs):.4f})")
            else:
                acc = _tm_cls_mod.classification(
                    dino_cfg, _ckpt, cls_train, None, cls_test, n_classes,
                    linear_probe=_lp, mlp_head=(head_type == "mlp"))
                results[_name] = acc
                print(f"  [{_name}] Test Accuracy: {acc:.4f}")

        print(f"\n{'='*60}")
        print(f"  SUMMARY — {classification_dataset} (pretrained on its own train series)")
        print("  " + "    ".join(f"{k.replace('_', ' ')}: {v:.4f}" for k, v in results.items()))
        print(f"{'='*60}")
        return results

    # ── pretrain ON the anomaly data, then fine-tune the detector ──────────────
    # Self-supervised DINO on the anomaly dataset's NORMAL (train) stream — no
    # labels — then a reconstruction fine-tune from that same checkpoint. The
    # window (seq_len) is shared by both stages because the TSMixer PDM layers are
    # seq_len-fixed; default 100 ts (num_patches 10 × patch_len 10).
    if pretrain_on_anomaly:
        if backbone != "timemixer":
            raise ValueError("pretrain_on_anomaly is only wired for backbone='timemixer'")
        if anomaly_dataset is None:
            raise ValueError("pretrain_on_anomaly=True requires anomaly_dataset")
        _shared_dir = str(root_dir / "shared")
        if _shared_dir not in sys.path:
            sys.path.insert(0, _shared_dir)
        from data_loaders.data_puller import AnomalyDataPuller

        anom_dir = dino_cfg["anomaly_data_dir"]
        anom_bs  = dino_cfg.get("batch_size_anomaly", 64)
        # window = num_patches × patch_len (must divide evenly). Defaults: 10 × 10 = 100.
        _a_ps = patch_len   if patch_len   is not None else 10
        _a_np = num_patches if num_patches is not None else (seq_len // _a_ps if seq_len else 10)
        _a_win = _a_np * _a_ps

        # downstream reconstruction loaders (StandardScaled sliding windows)
        _ds_tr = AnomalyDataPuller(anom_dir, anomaly_dataset, _a_ps, win_size=_a_win, which="train")
        _ds_te = AnomalyDataPuller(anom_dir, anomaly_dataset, _a_ps, win_size=_a_win, which="test")
        n_vars = _ds_tr.n_vars
        anom_train = torch.utils.data.DataLoader(_ds_tr, batch_size=anom_bs, shuffle=False)
        anom_test  = torch.utils.data.DataLoader(_ds_te, batch_size=anom_bs, shuffle=False)

        # Configure DINO for SSL-on-anomaly (seq_len drives pretrain + recon backbone)
        dino_cfg['pretrain_anomaly_dataset'] = anomaly_dataset
        dino_cfg['pretrain_source']          = None
        dino_cfg['pretrain_val_fraction']    = pretrain_val_fraction
        # Pure DINO by default (config default mlm_phi=0.75 → DINO+iBOT, which
        # ~doubles memory and changes the masking/forward shapes). Honour an
        # explicit --mlm_phi override if the caller set one.
        dino_cfg['mlm_phi'] = mlm_phi if mlm_phi is not None else 0.0
        dino_cfg['c_in']        = n_vars
        dino_cfg['seq_len']     = _a_win
        dino_cfg['num_patches'] = _a_np
        dino_cfg['patch_len']   = _a_ps
        dino_cfg['step_size']   = _a_ps
        _base = dino_cfg.get('output_dir', './checkpoints').rstrip('/')
        if f"anompre_{anomaly_dataset}" not in _base:
            dino_cfg['output_dir'] = str(
                Path(_base).parent / 'anomaly' /
                (Path(_base).name + f'_anompre_{anomaly_dataset}_cw{_a_win}'))

        dino_main.cfg.update(dino_cfg)
        args = _config_to_dino_args(dino_cfg)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

        _anom_ratio = _get_anomaly_ratio(anomaly_dataset, dino_cfg)
        print("\n" + "=" * 60)
        print(f"  MODEL: DINO (tsdino_timemixer) — PRETRAIN ON ANOMALY DATA + FINE-TUNE")
        print(f"  dataset: {anomaly_dataset}   n_vars={n_vars}  "
              f"seq_len={_a_win} ({_a_np}×{_a_ps})   ratio={_anom_ratio}")
        print(f"  output_dir: {args.output_dir}")
        print("=" * 60)

        if not skip_train:
            print("\n[DINO] Pretraining on anomaly train stream …")
            dino_main.train_TS_DINO(args)
        else:
            print("[DINO] skip_train=True — reusing existing checkpoint.")
        _ckpt = os.path.join(args.output_dir, "checkpoint_best.pth")
        if not os.path.exists(_ckpt):
            _epochs_ckpts = sorted(Path(args.output_dir).glob("checkpoint*.pth"))
            if _epochs_ckpts:
                _ckpt = str(_epochs_ckpts[-1])
            else:
                raise FileNotFoundError(f"No pretrained checkpoint found in {args.output_dir}")

        _tm_anom_spec = _ilu.spec_from_file_location(
            "tsmixer_anomaly", dino_dir / "TSMixerAnomaly.py")
        _tm_anom_mod  = _ilu.module_from_spec(_tm_anom_spec)
        _tm_anom_spec.loader.exec_module(_tm_anom_mod)
        dino_cfg["output_dir"] = args.output_dir

        print(f"\n{'='*60}\n  [DINO] Anomaly fine-tune (encoder+decoder) on {anomaly_dataset}\n{'='*60}")
        anom_result = _tm_anom_mod.anomaly_detection(
            dino_cfg, "best", anom_train, anom_test,
            anomaly_ratio=_anom_ratio,
            linear_probe=False,                      # fine-tune encoder + decoder
            mlp_head=(head_type == "mlp"),
            checkpoint_path=_ckpt)

        print(f"\n{'='*60}")
        print(f"  SUMMARY — {anomaly_dataset} (pretrained on its own train stream, fine-tuned)")
        if isinstance(anom_result, dict):
            print("  " + "    ".join(
                f"{k}: {v:.4f}" for k, v in anom_result.items()
                if isinstance(v, (int, float))))
        print(f"{'='*60}")
        return anom_result

    pretrain_source = _resolve_pretrain_source(dino_cfg)
    use_global_data = pretrain_source is not None

    # Resolve forecast dataset (always needed for downstream)
    if not (anomaly_dataset is not None and forecast_dataset is None):
        forecast_dataset = forecast_dataset or dino_cfg.get("forecast_dataset")
    dino_cfg["lr_forecasting"] = _get_forecast_lr(dino_cfg, "lr_forecasting")
    if lr_forecasting is not None:
        dino_cfg["lr_forecasting"] = lr_forecasting
        dino_cfg["min_lr_forecasting"] = lr_forecasting / 10
    if pretrain_only and use_global_data:
        dino_cfg['saveckp_freq'] = 1  # save every epoch

    if use_global_data:
        # No pretrain CSV needed; c_in = 1 (univariate global data)
        dino_cfg["c_in"] = 1
        if not pretrain_only and classification_dataset is None and anomaly_dataset is None:
            if forecast_dataset is None:
                raise ValueError("forecast_dataset must be set when pretraining on global data")
        if not pretrain_only and forecast_dataset is not None:
            ds_fore = get_dataset_info(forecast_dataset)
            dino_cfg["c_in"]                           = ds_fore["c_in"]
            dino_cfg["data_path_forecast_training"]    = ds_fore["csv_path"]
            dino_cfg["data_path_forecast_test"]        = ds_fore["csv_path"]
            dino_cfg["parms_for_training_forecasting"] = ds_fore["columns"]
            dino_cfg["parms_for_testing_forecasting"]  = ds_fore["columns"]
        if pretrain_source in ('monash', 'monash+synthetic'):
            monash_dir = dino_cfg['monash_data_dir']
            if not os.path.isabs(monash_dir):
                dino_cfg['monash_data_dir'] = str((dino_dir / monash_dir).resolve())
        if pretrain_source in ('synthetic', 'monash+synthetic'):
            _synth_key = 'synthetic_mix_data_dir' if pretrain_source == 'monash+synthetic' else 'synthetic_data_dir'
            synth_dir = dino_cfg[_synth_key]
            if not os.path.isabs(synth_dir):
                synth_dir = str((dino_dir / synth_dir).resolve())
            dino_cfg['synthetic_data_dir'] = synth_dir
        _src_label = {
            'monash':           f"Monash ({dino_cfg['monash_data_dir']})",
            'synthetic':        f"Synthetic ({dino_cfg['synthetic_data_dir']})",
            'monash+synthetic': "Monash + Synthetic",
        }.get(pretrain_source, pretrain_source)
        print("\n" + "="*60)
        print(f"  MODEL: DINO  (tsdino_{backbone})")
        if pretrain_only:
            print(f"  pretrain: {_src_label}  [pretrain only]")
        else:
            print(f"  pretrain: {_src_label}   forecast: {forecast_dataset}")
        print("="*60)
    else:
        pretrain_dataset = pretrain_dataset or dino_cfg.get("pretrain_dataset")
        forecast_dataset = forecast_dataset or pretrain_dataset
        if pretrain_dataset is None:
            raise ValueError("pretrain_dataset not set — specify via run() or config.py")
        ds_pre  = get_dataset_info(pretrain_dataset)
        ds_fore = get_dataset_info(forecast_dataset)
        dino_cfg["data_path"]                      = ds_pre["csv_path"]
        dino_cfg["data_path_forecast_training"]    = ds_fore["csv_path"]  # required by DINO dataset2 even during pretrain_only
        # c_in drives the model channel count. For a forecast-only run (skip_train)
        # the model must match the FORECAST data — this is what enables cross-domain
        # (load a checkpoint pretrained on dataset A, forecast dataset B with a
        # different channel count; the c_in-dependent RevIN layers are shape-skipped
        # on load, exactly like the synthetic-backbone path). Only when we actually
        # pretrain here does c_in need to be the pretrain dataset's.
        dino_cfg["c_in"]                           = ds_fore["c_in"] if skip_train else ds_pre["c_in"]
        _xdom = (pretrain_dataset != forecast_dataset)
        print("\n" + "="*60)
        print(f"  MODEL: DINO  (tsdino_{backbone})")
        print(f"  pretrain: {pretrain_dataset}   forecast: {forecast_dataset}"
              + ("   [CROSS-DOMAIN]" if _xdom else ""))
        print("="*60)
    if not pretrain_only and forecast_dataset is not None:
        dino_cfg["data_path_forecast_test"]        = ds_fore["csv_path"]
        dino_cfg["parms_for_training_forecasting"] = ds_fore["columns"]
        dino_cfg["parms_for_testing_forecasting"]  = ds_fore["columns"]

    # Propagate overrides into the module-level cfg dict that train_TS_DINO reads directly.
    # cfg in main.py is imported as `from config import config as cfg` — it's a reference
    # to the same dict object, so updating it in-place propagates everywhere.
    dino_main.cfg.update(dino_cfg)

    args = _config_to_dino_args(dino_cfg)

    # Resolve data paths relative to dino_dir so they work from any CWD
    # (skip if already absolute — e.g. injected from dataset_registry)
    for attr in ('data_path', 'data_path_forecast_training',
                 'data_path_forecast_test', 'data_path_classification'):
        val = getattr(args, attr, '')
        if val and not os.path.isabs(val):
            setattr(args, attr, str(dino_dir / val))

    args.linear_probe = linear_probe
    args.mlp_head = (head_type == "mlp")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # ── pretraining ──────────────────────────────────────────────────────────
    if not skip_train:
        print("\n[DINO] Starting pretraining …")
        dino_main.train_TS_DINO(args)
    else:
        print("[DINO] Skipping pretraining.")

    if pretrain_only:
        print("\n[DINO] Pretrain-only mode — skipping forecasting.")
        return

    # If a direct checkpoint path is given, use it (TEMP: for local testing, remove after)
    best_ckpt = os.path.abspath(checkpoint) if checkpoint else "best"
    best_mse  = None

    if not classification_only and forecast_dataset is not None:
        # ── forecasting downstream ────────────────────────────────────────────
        print("\n[DINO] Running forecasting downstream task …")
        ckpts = checkpoints if checkpoints is not None else ["best"]
        best_ckpt = None
        best_mse  = float('inf')

        for pred_len in pred_lens:
            args.pred_len = pred_len
            is_search = (pred_len == pred_lens[0])
            ckpts_to_run = ckpts if is_search else [best_ckpt if best_ckpt is not None else ckpts[-1]]

            print(f"\n[DINO] pred_len={pred_len}"
                  + ("" if is_search else f"  [best ckpt={ckpts_to_run[0]}]"))
            for ckpt in ckpts_to_run:
                args.path_num = ckpt
                _ckpt_label = 'random init' if ckpt == 0 else ('best' if ckpt == 'best' else f'epoch {ckpt}')
                print(f"  → checkpoint {ckpt} ({_ckpt_label})")
                mse = dino_main.test_run(args)
                if is_search and mse is not None and mse < best_mse:
                    best_mse  = mse
                    best_ckpt = ckpt

            if is_search:
                print(f"\n[DINO] Best checkpoint at pred_len={pred_lens[0]}: "
                      f"epoch {best_ckpt} (MSE={best_mse:.6f})")

    # ── classification downstream ─────────────────────────────────────────────
    cls_acc = None
    if classification_dataset is not None:
        from data_loaders.data_puller import ClassificationDataPuller, make_uea_dataloaders
        cls_dir = dino_cfg["classification_data_dir"]
        cls_bs  = dino_cfg.get("batch_size_classification", 64)
        p_s        = args.patch_len
        _n_patches = 72                         # classification encoder always uses 72 patches
        _target_T  = _n_patches * p_s
        import torch.nn.functional as _F
        def _dino_patch_collate(batch, _ps=p_s, _tT=_target_T, _nP=_n_patches):
            xs, ys, orig_lens = zip(*batch)
            orig_lens = torch.stack(orig_lens)                                   # (B,)
            max_t = max(x.shape[0] for x in xs)
            xs = torch.stack([_F.pad(x, (0, 0, 0, max_t - x.shape[0])) for x in xs])
            B_, T_, C_ = xs.shape
            if T_ != _tT:
                idx = torch.linspace(0, T_ - 1, _tT).long()
                xs  = xs[:, idx, :]
                patch_idx    = idx[torch.arange(_nP) * _ps]
                padding_mask = patch_idx.unsqueeze(0) < orig_lens.unsqueeze(1)  # (B, P)
            else:
                patch_starts = torch.arange(_nP) * _ps
                padding_mask = patch_starts.unsqueeze(0) < orig_lens.unsqueeze(1)
            xs = xs.reshape(B_, _nP, _ps, C_)
            return xs, torch.stack(ys), padding_mask
        if list(Path(os.path.join(cls_dir, classification_dataset)).glob("*_TRAIN.ts")):
            _raw_train, _, _raw_test, n_classes = make_uea_dataloaders(
                cls_dir, classification_dataset, batch_size=cls_bs)
            cls_train = torch.utils.data.DataLoader(
                _raw_train.dataset, batch_size=cls_bs, shuffle=True,
                collate_fn=_dino_patch_collate)
            cls_val   = None
            cls_test  = torch.utils.data.DataLoader(
                _raw_test.dataset, batch_size=cls_bs, shuffle=False,
                collate_fn=_dino_patch_collate)
        else:
            _mk = lambda split: torch.utils.data.DataLoader(
                ClassificationDataPuller(cls_dir, classification_dataset, p_s, which=split),
                batch_size=cls_bs, shuffle=(split == "train"))
            cls_train = _mk("train"); cls_val = _mk("val"); cls_test = _mk("test")
            n_classes = cls_train.dataset.n_classes
        _path_num_cls = best_ckpt if best_ckpt is not None else 0
        if backbone == "timemixer":
            # TimeMixer backbone: dedicated TSMixer classification head.
            _tm_cls_spec = _ilu.spec_from_file_location(
                "tsmixer_classification", dino_dir / "TSMixerClassification.py")
            _tm_cls_mod  = _ilu.module_from_spec(_tm_cls_spec)
            _tm_cls_spec.loader.exec_module(_tm_cls_mod)
            _out_dir = args.output_dir
            if _path_num_cls == "best" or _path_num_cls == 0:
                _cls_ckpt = os.path.join(_out_dir, "checkpoint_best.pth")
            else:
                _cls_ckpt = os.path.join(_out_dir, f"checkpoint{_path_num_cls:04d}.pth")
            cls_acc = _tm_cls_mod.classification(
                dino_cfg, _cls_ckpt, cls_train, cls_val, cls_test, n_classes,
                linear_probe=linear_probe,
                mlp_head=(head_type == "mlp"))
        else:
            # PatchTST backbone: classification lives in its main.py (train_classification),
            # which loads the checkpoint itself from args.path_num / args.output_dir.
            args.path_num     = _path_num_cls
            args.linear_probe = linear_probe
            args.mlp_head     = (head_type == "mlp")
            cls_acc = dino_main.train_classification(
                args, cls_train, cls_val, cls_test, n_classes)
        print(f"\n{'='*60}")
        print(f"  [DINO] Classification on {classification_dataset}")
        print(f"  Test Accuracy: {cls_acc:.4f}")
        print(f"{'='*60}")

    # ── anomaly detection downstream ──────────────────────────────────────────
    anom_result = None
    if anomaly_dataset is not None:
        from data_loaders.data_puller import AnomalyDataPuller
        anom_dir    = dino_cfg["anomaly_data_dir"]
        anom_bs     = dino_cfg.get("batch_size_anomaly", 64)
        _anom_ratio = _get_anomaly_ratio(anomaly_dataset, dino_cfg)
        _path_num   = best_ckpt if best_ckpt is not None else 0

        # TSDiNO is TimeMixer-only. TSMixer PDM blocks have linear layers fixed to
        # seq_len — the anomaly window must match the pretraining seq_len exactly.
        _seq_len = dino_cfg.get("seq_len", 512)
        p_s      = args.patch_len
        anom_train = torch.utils.data.DataLoader(
            AnomalyDataPuller(anom_dir, anomaly_dataset, p_s,
                              win_size=_seq_len, which="train"),
            batch_size=anom_bs, shuffle=False)
        anom_test  = torch.utils.data.DataLoader(
            AnomalyDataPuller(anom_dir, anomaly_dataset, p_s,
                              win_size=_seq_len, which="test"),
            batch_size=anom_bs, shuffle=False)
        # Same function name across backbones, but the FIRST arg type differs:
        #   timemixer → TSMixerAnomaly.py, reads a cfg DICT (cfg["c_in"], cfg.get(...))
        #   patchtst  → Anomaly.py,        reads an args NAMESPACE (args.c_in, args.num_patches)
        _anom_file = "TSMixerAnomaly.py" if backbone == "timemixer" else "Anomaly.py"
        _tm_spec = _ilu.spec_from_file_location("dino_anomaly", dino_dir / _anom_file)
        _tm_mod  = _ilu.module_from_spec(_tm_spec)
        _tm_spec.loader.exec_module(_tm_mod)
        dino_cfg["output_dir"] = args.output_dir
        _anom_cfg = dino_cfg if backbone == "timemixer" else args
        anom_result = _tm_mod.anomaly_detection(
            _anom_cfg, _path_num, anom_train, anom_test,
            anomaly_ratio=_anom_ratio,
            linear_probe=linear_probe,
            mlp_head=(head_type == "mlp"))

    return best_ckpt, best_mse, cls_acc, anom_result


# ── PatchTST ──────────────────────────────────────────────────────────────────

def run_patchtst(skip_train: bool = False, pretrain_dataset: str = None, forecast_dataset: str = None,
                 classification_dataset=None, anomaly_dataset: str = None,
                 pretrain_only: bool = False, classification_only: bool = False, pred_lens=None,
                 checkpoints=None, random_encoder: bool = False, encoder_layers: int = None,
                 predictor_layers: int = None, lr: float = None, pretrain_source: str = None,
                 num_patches: int = None, seed: int = None, linear_probe: bool = True,
                 head_type: str = "linear", embed_dim: int = None, epochs: int = None,
                 batch_size: int = None):
    if pred_lens is None:
        pred_lens = [96, 192, 336, 720]
    patchtst_dir = Path(__file__).parent / "PatchTST_self_supervised"
    shared_dir    = Path(__file__).parent / "shared"
    _add_path(shared_dir)

    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "config_patchtst", patchtst_dir / "config_patchtst.py")
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    cfg = {**DATA_PATHS, **dict(_mod.config)}
    if batch_size is not None:
        cfg['batch_size'] = batch_size          # per-run pretrain batch override
    if pretrain_source is not None:
        cfg['pretrain_source'] = pretrain_source
    elif pretrain_dataset is not None and pretrain_dataset not in ('monash', 'synthetic', 'monash+synthetic'):
        cfg['pretrain_source'] = None  # force in-domain CSV pretraining
    if encoder_layers is not None:
        cfg['n_layers'] = encoder_layers
        cfg['pretrained_model_id'] = encoder_layers  # unique checkpoint per layer config
    if embed_dim is not None:
        cfg['d_model'] = embed_dim
    if num_patches is not None:
        cfg['context_points'] = num_patches * cfg.get('patch_len', 16)

    pretrain_source = _resolve_pretrain_source(cfg)
    _pretrain_dset  = pretrain_dataset or cfg.get("pretrain_dataset", "ettm1")
    _forecast_dset  = None if (pretrain_only or classification_only or (anomaly_dataset is not None and forecast_dataset is None)) else (forecast_dataset or cfg.get("forecast_dataset") or _pretrain_dset)

    # pretrain_source overrides _pretrain_dset for global data
    if pretrain_source in ('monash', 'synthetic', 'monash+synthetic'):
        _pretrain_dset = pretrain_source

    monash_dir = synth_dir = None
    if _pretrain_dset in ('monash', 'monash+synthetic'):
        monash_dir = cfg["monash_data_dir"]
        if not os.path.isabs(monash_dir):
            monash_dir = str((patchtst_dir / monash_dir).resolve())
    if _pretrain_dset in ('synthetic', 'monash+synthetic'):
        _synth_key = 'synthetic_mix_data_dir' if _pretrain_dset == 'monash+synthetic' else 'synthetic_data_dir'
        synth_dir = cfg[_synth_key]
        if not os.path.isabs(synth_dir):
            synth_dir = str((patchtst_dir / synth_dir).resolve())

    _src_label = {
        'monash':           f"Monash ({monash_dir})",
        'synthetic':        f"Synthetic ({synth_dir})",
        'monash+synthetic': "Monash + Synthetic",
    }.get(_pretrain_dset, _pretrain_dset)
    print("\n" + "="*60)
    print("  MODEL: PatchTST (self-supervised)")
    if pretrain_only:
        print(f"  pretrain: {_src_label}  [pretrain only]")
    else:
        print(f"  pretrain: {_src_label}   forecast: {_forecast_dset}")
    print("="*60)

    patchtst_dir = patchtst_dir.resolve()
    _add_path(patchtst_dir)

    # Build common pretrain args from config
    pretrain_cmd = [
        sys.executable, "patchtst_pretrain.py",
        "--dset_pretrain",       _pretrain_dset,
        "--context_points",      str(cfg.get("context_points",      512)),
        "--patch_len",           str(cfg.get("patch_len",           12)),
        "--stride",              str(cfg.get("stride",              12)),
        "--n_layers",            str(cfg.get("n_layers",            3)),
        "--n_heads",             str(cfg.get("n_heads",             16)),
        "--d_model",             str(cfg.get("d_model",             128)),
        "--d_ff",                str(cfg.get("d_ff",                512)),
        "--dropout",             str(cfg.get("dropout",             0.2)),
        "--head_dropout",        str(cfg.get("head_dropout",        0.2)),
        "--mask_ratio",          str(cfg.get("mask_ratio",          0.4)),
        "--n_epochs_pretrain",   str(epochs if epochs is not None else cfg.get("n_epochs_pretrain", 10)),
        "--batch_size",          str(cfg.get("batch_size",          64)),
        "--revin",               str(int(cfg.get("revin",           True))),
        "--pretrained_model_id", str(cfg.get("pretrained_model_id", 1)),
        "--seed",                str(seed if seed is not None else GLOBAL_SEED),
    ]
    if monash_dir is not None:
        pretrain_cmd += ["--monash_data_dir",    monash_dir,
                         "--monash_min_len",      str(cfg["monash_min_len"])]
    if synth_dir is not None:
        pretrain_cmd += ["--synthetic_data_dir", synth_dir]
    if lr is not None:
        pretrain_cmd += ["--lr", str(lr)]
    # Save dir: single source of truth for both pretrain write and downstream read.
    # Three layouts: classification-cw (sweep), seed-tagged, or default.
    if num_patches is not None:
        _cw = num_patches * cfg.get('patch_len', 16)
        _ptst_save_dir = str(
            patchtst_dir / "saved_models" / "classification" /
            _pretrain_dset / "masked_patchtst" / cfg.get("model_type", "based_model") /
            f"layers{cfg.get('n_layers', 3)}_cw{_cw}{_SEED_TAG}"
        )
    elif _SEED_TAG:
        _ptst_save_dir = str(
            patchtst_dir / "saved_models" / _pretrain_dset /
            "masked_patchtst" / cfg.get("model_type", "based_model") /
            f"layers{cfg.get('n_layers', 3)}{_SEED_TAG}"
        )
    else:
        _ptst_save_dir = str(
            patchtst_dir / "saved_models" / _pretrain_dset /
            "masked_patchtst" / cfg.get("model_type", "based_model") /
            f"layers{cfg.get('n_layers', 3)}"
        )
    pretrain_cmd += ["--save_dir", _ptst_save_dir]

    # ── pretraining ───────────────────────────────────────────────────────────
    if not skip_train:
        print(f"\n[PatchTST] Starting pretraining on {_pretrain_dset} …")
        result = subprocess.run(pretrain_cmd, cwd=patchtst_dir, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print("[PatchTST] Pretraining exited with errors.")
            print(result.stderr)
            return
    else:
        print("[PatchTST] Skipping pretraining.")

    # ── resolve checkpoint path (needed for both forecast and classify) ────────
    n_ep    = epochs if epochs is not None else cfg.get("n_epochs_pretrain", 10)
    ctx     = cfg.get("context_points", 512)
    p_len   = cfg.get("patch_len", 12)
    stride  = cfg.get("stride", 12)
    m_ratio = cfg.get("mask_ratio", 0.4)
    m_id    = cfg.get("pretrained_model_id", 1)
    model_fname_base = (f"patchtst_pretrained_cw{ctx}_patch{p_len}_stride{stride}"
                        f"_epochs-pretrain{n_ep}_mask{m_ratio}_model{m_id}")
    _ckpt_epoch = checkpoints[0] if (checkpoints and checkpoints[0] is not None) else None
    model_fname = f"{model_fname_base}_{_ckpt_epoch}.pth" if _ckpt_epoch is not None else f"{model_fname_base}.pth"
    pretrained_model_path = None if random_encoder else os.path.join(_ptst_save_dir, model_fname)

    if pretrain_only:
        print("\n[PatchTST] Pretrain-only mode — skipping forecasting.")
        return

    # ── forecasting downstream ────────────────────────────────────────────────
    mse_val, mae_val = None, None
    if _forecast_dset is None:
        print("\n[PatchTST] No forecast_dataset — skipping forecasting.")
    else:
        import re as _re
        print(f"\n[PatchTST] Running forecasting fine-tuning on {_forecast_dset} …")
        for _pl in pred_lens:
            print(f"\n[PatchTST] pred_len={_pl}")
            result = subprocess.run(
                [sys.executable, "patchtst_finetune.py",
                 "--dset_finetune",      _forecast_dset,
                 "--is_finetune",        str(int(not linear_probe)),
                 "--is_linear_probe",    str(int(linear_probe)),
                 "--context_points",  str(cfg.get("context_points", 512)),
                 "--patch_len",       str(cfg.get("patch_len", 16)),
                 "--stride",          str(cfg.get("stride", 16)),
                 "--n_layers",        str(cfg.get("n_layers", 3)),
                 "--n_heads",         str(cfg.get("n_heads", 16)),
                 "--d_model",         str(cfg.get("d_model", 128)),
                 "--d_ff",            str(cfg.get("d_ff", 512)),
                 "--dropout",         str(cfg.get("dropout", 0.2)),
                 "--head_dropout",    str(cfg.get("head_dropout_forecasting", cfg.get("head_dropout", 0.2))),
                 "--target_points",   str(_pl),
                 "--pretrained_model", str(pretrained_model_path) if pretrained_model_path is not None else "",
                 "--random_encoder",   str(int(random_encoder)),
                 "--batch_size",       str(_get_forecast_bs(cfg, 256)),
                 "--num_workers",      str(cfg.get("num_workers", 4)),
                 "--lr",               str(cfg.get("finetune_lr", 1e-4)),
                 "--seed",             str(seed if seed is not None else GLOBAL_SEED),
                 "--mlp_head",         str(int(head_type == "mlp"))],
                cwd=patchtst_dir, capture_output=True, text=True,
            )
            print(result.stdout)
            if result.returncode != 0:
                print(f"[PatchTST] pred_len={_pl} exited with errors.")
                print(result.stderr)
                continue

            _score_match = _re.search(r"score:\s*\[array\(([\d.]+)[^)]*\)[^,]*,\s*array\(([\d.]+)", result.stdout)
            if _score_match:
                _mse = float(_score_match.group(1))
                _mae = float(_score_match.group(2))
                print(f"[PatchTST] pred_len={_pl}  MSE={_mse:.4f}  MAE={_mae:.4f}")
                if mse_val is None or _mse < mse_val:
                    mse_val, mae_val = _mse, _mae

    # ── classification downstream ─────────────────────────────────────────────
    cls_acc = None
    if classification_dataset is not None:
        from patchtst_classification import classification as ptst_classify
        from data_loaders.data_puller import ClassificationDataPuller, make_uea_dataloaders
        cls_dir    = cfg["classification_data_dir"]
        cls_bs     = _get_cls_bs(cfg, "batch_size", 64)
        p_s        = cfg.get("patch_len", 16)
        _n_patches = 72                               # classification encoder always uses 72 patches
        _target_T  = _n_patches * p_s
        import torch.nn.functional as _F
        def _ptst_patch_collate(batch, _ps=p_s, _tT=_target_T, _nP=_n_patches):
            xs, ys, orig_lens = zip(*batch)
            orig_lens = torch.stack(orig_lens)                                   # (B,)
            max_t = max(x.shape[0] for x in xs)
            xs = torch.stack([_F.pad(x, (0, 0, 0, max_t - x.shape[0])) for x in xs])
            B_, T_, C_ = xs.shape
            if T_ != _tT:
                idx = torch.linspace(0, T_ - 1, _tT).long()
                xs  = xs[:, idx, :]
                patch_idx    = idx[torch.arange(_nP) * _ps]
                padding_mask = patch_idx.unsqueeze(0) < orig_lens.unsqueeze(1)  # (B, P)
            else:
                patch_starts = torch.arange(_nP) * _ps
                padding_mask = patch_starts.unsqueeze(0) < orig_lens.unsqueeze(1)
            xs = xs.reshape(B_, _nP, _ps, C_)
            return xs, torch.stack(ys), padding_mask
        if list(Path(os.path.join(cls_dir, classification_dataset)).glob("*_TRAIN.ts")):
            _raw_tr, _, _raw_te, n_classes = make_uea_dataloaders(
                cls_dir, classification_dataset, batch_size=cls_bs)
            cls_train = torch.utils.data.DataLoader(
                _raw_tr.dataset, batch_size=cls_bs, shuffle=True,
                collate_fn=_ptst_patch_collate)
            cls_val   = None
            cls_test  = torch.utils.data.DataLoader(
                _raw_te.dataset, batch_size=cls_bs, shuffle=False,
                collate_fn=_ptst_patch_collate)
        else:
            _mk = lambda split: torch.utils.data.DataLoader(
                ClassificationDataPuller(cls_dir, classification_dataset, p_s, which=split),
                batch_size=cls_bs, shuffle=(split == "train"))
            cls_train = _mk("train"); cls_val = _mk("val"); cls_test = _mk("test")
            n_classes = cls_train.dataset.n_classes
        cls_acc = ptst_classify(cfg, pretrained_model_path, cls_train, cls_val, cls_test, n_classes,
                                linear_probe=linear_probe,
                                mlp_head=(head_type == "mlp"))
        print(f"\n{'='*60}")
        print(f"  [PatchTST] Classification on {classification_dataset}")
        print(f"  Test Accuracy: {cls_acc:.4f}")
        print(f"{'='*60}")

    # ── anomaly detection downstream ──────────────────────────────────────────
    anom_result = None
    if anomaly_dataset is not None:
        from patchtst_anomaly import anomaly_detection as ptst_anomaly
        from data_loaders.data_puller import AnomalyDataPuller
        anom_dir = cfg["anomaly_data_dir"]
        anom_bs  = cfg.get("batch_size", 64)
        p_s      = cfg.get("patch_len", 16)
        anom_train = torch.utils.data.DataLoader(
            AnomalyDataPuller(anom_dir, anomaly_dataset, p_s, which="train"),
            batch_size=anom_bs, shuffle=False)
        anom_test  = torch.utils.data.DataLoader(
            AnomalyDataPuller(anom_dir, anomaly_dataset, p_s, which="test"),
            batch_size=anom_bs, shuffle=False)
        anom_result = ptst_anomaly(cfg, pretrained_model_path, anom_train, anom_test,
                                   anomaly_ratio=_get_anomaly_ratio(anomaly_dataset, cfg),
                                   linear_probe=linear_probe,
                                   mlp_head=(head_type == "mlp"))

    return mse_val, mae_val, cls_acc, anom_result


# ── TimeMixer ─────────────────────────────────────────────────────────────────

class _FlatWindowAdapterTM(torch.utils.data.Dataset):
    """
    Wraps PatchTSTForcastingAdapter and produces zero time marks of the correct
    shape for TimeMixer / TSLib DataEmbedding_wo_pos (timeF encoding needs
    mark_dim features). Returns (seq_x, seq_y, xmark, ymark) where marks are
    zeros of shape [T, mark_dim].

    label_len is accepted for call sites that prepend an encoder-overlap prefix
    to seq_y (Autoformer/FEDformer); the marks are sized to the reshaped tensors
    regardless, so it needs no special handling here.
    """
    _FREQ_DIM = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}

    def __init__(self, patched_ds, freq: str = 'h', label_len: int = 0):
        self._ds = patched_ds
        self._mark_dim = self._FREQ_DIM.get(freq, 4)

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        ctx, tgt = self._ds[idx]
        seq_x = ctx.reshape(-1, ctx.shape[-1])   # [seq_len, C]
        seq_y = tgt.reshape(-1, tgt.shape[-1])   # [label_len + pred_len, C]
        xmark = torch.zeros(seq_x.shape[0], self._mark_dim)
        ymark = torch.zeros(seq_y.shape[0], self._mark_dim)
        return seq_x, seq_y, xmark, ymark


def run_timemixer(skip_train: bool = False,
                  pretrain_dataset: str = None,
                  forecast_dataset: str = None,
                  classification_dataset: str = None,
                  anomaly_dataset: str = None,
                  pretrain_only: bool = False,
                  pred_lens=None,
                  encoder_layers: int = None,
                  lr: float = None,
                  linear_probe: bool = True,
                  head_type: str = "linear",
                  embed_dim: int = None,
                  epochs: int = None,
                  epochs_forecasting: int = None,
                  pretrain_source: str = None,
                  checkpoint: str = None,
                  ckpt_tag: str = None):
    """
    TimeMixer: supervised multi-scale mixing model.
    No pretraining — trains directly on each downstream task.
    Uses PatchTSTForcastingAdapter for forecasting (same splits as all other models).
    """
    if pred_lens is None:
        pred_lens = [96, 192, 336, 720]

    timemixer_dir = Path(__file__).parent / "TimeMixer-main"
    shared_dir    = Path(__file__).parent / "shared"
    _add_path(shared_dir)

    # Put timemixer_dir first so its exp/models packages take priority.
    import sys as _sys, importlib.util as _ilu, torch
    from types import SimpleNamespace

    _tm_str = str(timemixer_dir)
    if _tm_str not in _sys.path:
        _sys.path.insert(0, _tm_str)
    # Evict any cached exp/models modules from other models so timemixer's win.
    for _key in list(_sys.modules.keys()):
        if _key in ('exp', 'models') or _key.startswith('exp.') or _key.startswith('models.'):
            _sys.modules.pop(_key, None)

    # ── load config ────────────────────────────────────────────────────────────
    _spec = _ilu.spec_from_file_location("config_timemixer", timemixer_dir / "config_timemixer.py")
    _mod  = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    cfg = {**DATA_PATHS, **dict(_mod.config)}

    if encoder_layers is not None:
        cfg['e_layers'] = encoder_layers
    if embed_dim is not None:
        cfg['d_model'] = embed_dim
        cfg['d_ff']    = embed_dim * 2
    if lr is not None:
        cfg['learning_rate'] = lr

    _gpu_idx = 0
    _device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    seq_len   = cfg['seq_len']
    patch_len = cfg.get('patch_len', 16)
    freq      = cfg.get('freq', 'h')

    print("\n" + "="*60)
    print(f"  MODEL: TimeMixer  (supervised, no pretraining)")
    print(f"  forecast: {forecast_dataset or '(none)'}  cls: {classification_dataset or '(none)'}")
    print(f"  e_layers={cfg['e_layers']}  d_model={cfg['d_model']}  seq_len={seq_len}")
    print("="*60)

    if pretrain_only:
        print("[TimeMixer] pretrain_only=True — TimeMixer has no pretraining phase, skipping.")
        return

    # base args for Exp_Long_Term_Forecast
    base_args = SimpleNamespace(
        model                       = 'TimeMixer',
        task_name                   = 'long_term_forecast',
        use_gpu                     = torch.cuda.is_available(),
        gpu                         = _gpu_idx,
        use_multi_gpu               = False,
        devices                     = str(_gpu_idx),
        device_ids                  = [_gpu_idx],
        seq_len                     = seq_len,
        label_len                   = cfg.get('label_len', 0),
        pred_len                    = 96,
        enc_in                      = 1,
        dec_in                      = 1,
        c_out                       = 1,
        d_model                     = cfg['d_model'],
        n_heads                     = cfg['n_heads'],
        e_layers                    = cfg['e_layers'],
        d_layers                    = cfg.get('d_layers', 1),
        d_ff                        = cfg['d_ff'],
        dropout                     = cfg['dropout'],
        embed                       = freq,      # DataEmbedding_wo_pos uses embed= for mode
        freq                        = freq,
        factor                      = cfg.get('factor', 1),
        moving_avg                  = cfg.get('moving_avg', 25),
        channel_independence        = cfg.get('channel_independence', 1),
        decomp_method               = cfg.get('decomp_method', 'moving_avg'),
        use_norm                    = cfg.get('use_norm', 1),
        down_sampling_layers        = cfg.get('down_sampling_layers', 3),
        down_sampling_window        = cfg.get('down_sampling_window', 2),
        down_sampling_method        = cfg.get('down_sampling_method', 'avg'),
        use_future_temporal_feature = cfg.get('use_future_temporal_feature', 0),
        top_k                       = cfg.get('top_k', 5),
        num_kernels                 = cfg.get('num_kernels', 6),
        features                    = cfg.get('features', 'M'),
        output_attention            = False,
        data                        = 'custom',
        root_path                   = '/tmp',
        data_path                   = 'data.csv',
        inverse                     = False,
        checkpoints                 = str(Path(__file__).parent / 'outputs' / 'timemixer_forecast'),
        num_workers                 = cfg.get('num_workers', 4),
        train_epochs                = epochs if epochs is not None else cfg.get('train_epochs', 10),
        batch_size                  = cfg.get('batch_size', 16),
        learning_rate               = cfg['learning_rate'],
        patience                    = cfg.get('patience', 5),
        lradj                       = cfg.get('lradj', 'TST'),
        pct_start                   = cfg.get('pct_start', 0.2),
        loss                        = cfg.get('loss', 'MSE'),
        drop_last                   = cfg.get('drop_last', True),
        use_amp                     = False,
    )

    # TimeMixer's embed arg is the string name like 'timeF'; the DataEmbedding_wo_pos
    # constructor takes embed_type as its own arg. Keep the alias consistent.
    base_args.embed = cfg.get('embed', 'timeF')

    # ── forecasting downstream ─────────────────────────────────────────────────
    best_mse, best_mae, best_pred = float('inf'), float('inf'), None

    if forecast_dataset is not None:
        from data_loaders.data_puller import PatchTSTForcastingAdapter
        from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
        import types as _types, numpy as _np

        ds_info  = get_dataset_info(forecast_dataset)
        _csv     = ds_info["csv_path"]
        _c_in    = ds_info["c_in"]
        _fc_bs   = _get_forecast_bs(cfg, 128)
        _fc_nw   = cfg.get('num_workers', 4)
        _n_epochs_fc = (epochs_forecasting if epochs_forecasting is not None
                        else epochs if epochs is not None
                        else cfg.get('epochs_forecasting', 10))

        # TM_REAL_MARKS=1 → use TimeMixer's native dataset (real calendar marks,
        # timeenc=1), faithful to the original repo but on OUR splits/HPs.
        # Default (0) keeps the zero-marks adapter we've been using for split-parity
        # with PatchTST and the DINO models.
        _use_real_marks = os.environ.get("TM_REAL_MARKS", "0") == "1"
        if _use_real_marks:
            from data_provider.data_loader import (
                Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom)
            _root      = os.path.dirname(os.path.abspath(_csv))
            _fname     = os.path.basename(_csv)
            _fname_low = _fname.lower()
            _TM_DS = (Dataset_ETT_hour  if 'etth' in _fname_low else
                      Dataset_ETT_minute if 'ettm' in _fname_low else
                      Dataset_Custom)
            print(f"  [TimeMixer] marks=REAL (timeenc=1, {_TM_DS.__name__})")
        else:
            print("  [TimeMixer] marks=ZERO (split-parity adapter)")

        for pred_len in pred_lens:
            def _fc_loader(split, _pl=pred_len):
                if _use_real_marks:
                    ds = _TM_DS(root_path=_root, flag=split,
                                size=[seq_len, base_args.label_len, _pl],
                                features='M', data_path=_fname,
                                timeenc=1, freq=freq)
                else:
                    ds = _FlatWindowAdapterTM(
                        PatchTSTForcastingAdapter(_csv, split, seq_len, _pl, patch_len),
                        freq=freq)
                return torch.utils.data.DataLoader(
                    ds, batch_size=_fc_bs, shuffle=(split == 'train'),
                    num_workers=_fc_nw, drop_last=True)

            ft_args = SimpleNamespace(**vars(base_args))
            ft_args.pred_len      = pred_len
            ft_args.enc_in        = _c_in
            ft_args.dec_in        = _c_in
            ft_args.c_out         = _c_in
            ft_args.train_epochs  = _n_epochs_fc
            ft_args.learning_rate = _get_forecast_lr(cfg, 'lr_forecasting', 5e-4)
            ft_args.batch_size    = _fc_bs

            # ── resolve checkpoint path (needed for setting string) ───────────
            _ckpt_path = checkpoint
            if _ckpt_path is None and ckpt_tag is not None:
                _ckpt_path = str(
                    Path(__file__).parent /
                    f"checkpoints_synthetic_layers{cfg['e_layers']}_{ckpt_tag}" /
                    "checkpoint_best.pth"
                )

            _ckpt_suffix = (f"_{Path(_ckpt_path).parent.name}" if _ckpt_path else "")
            setting = (f"timemixer_{forecast_dataset}_pl{pred_len}"
                       f"_dm{cfg['d_model']}_el{cfg['e_layers']}{_ckpt_suffix}")

            print(f"\n[TimeMixer] Forecasting pred_len={pred_len} on {forecast_dataset} …")

            _tm_train = _fc_loader('train')
            _tm_val   = _fc_loader('val')
            _tm_test  = _fc_loader('test')

            exp = Exp_Long_Term_Forecast(ft_args)

            def _get_data(self, flag):
                loader = {'train': _tm_train, 'val': _tm_val, 'test': _tm_test}[flag]
                return loader.dataset, loader
            exp._get_data = _types.MethodType(_get_data, exp)

            # ── load pretrained backbone into exp.model ────────────────────────
            if _ckpt_path and os.path.exists(_ckpt_path):
                _raw = torch.load(_ckpt_path, map_location="cpu", weights_only=False)
                _sd  = _raw.get("teacher", _raw.get("model", _raw))
                _new_sd = {}
                for _k, _v in _sd.items():
                    _k = _k.replace("module.", "")
                    if _k.startswith("backbone."):
                        _k = _k[len("backbone."):]
                    _new_sd[_k] = _v
                # Skip shape-mismatched keys (e.g. normalize_layers trained on c_in=1)
                _model_sd = exp.model.state_dict()
                _filtered = {k: v for k, v in _new_sd.items()
                             if k in _model_sd and _model_sd[k].shape == v.shape}
                _skipped  = [k for k in _new_sd if k in _model_sd and _model_sd[k].shape != _new_sd[k].shape]
                _miss, _unexp = exp.model.load_state_dict(_filtered, strict=False)
                print(f"  [TimeMixer] Loaded pretrained backbone: {_ckpt_path}")
                print(f"  Matched: {len(_filtered)}  Missing: {len(_miss)}  Skipped (shape mismatch): {len(_skipped)}")
                if _skipped:
                    print(f"  Skipped keys: {_skipped}")
            elif _ckpt_path:
                print(f"  [TimeMixer] WARNING: checkpoint not found: {_ckpt_path}")

            if not skip_train:
                exp.train(setting)

            # ── evaluate ──────────────────────────────────────────────────────
            exp.model.eval()
            preds_list, trues_list = [], []
            _fdev = exp.device
            with torch.no_grad():
                for batch_x, batch_y, batch_x_mark, batch_y_mark in _tm_test:
                    batch_x      = batch_x.float().to(_fdev)
                    batch_y      = batch_y.float().to(_fdev)
                    batch_x_mark = batch_x_mark.float().to(_fdev)
                    batch_y_mark = batch_y_mark.float().to(_fdev)
                    # down_sampling_layers > 0 → no decoder input needed
                    dec_inp  = None if ft_args.down_sampling_layers > 0 else (
                        torch.zeros_like(batch_y[:, -pred_len:, :]).float().to(_fdev))
                    outputs  = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    f_dim    = -1 if ft_args.features == 'MS' else 0
                    outputs  = outputs[:, -pred_len:, f_dim:]
                    batch_y  = batch_y[:, -pred_len:, f_dim:]
                    preds_list.append(outputs.detach().cpu().numpy())
                    trues_list.append(batch_y.detach().cpu().numpy())
            exp.model.train()

            preds_arr = _np.concatenate(preds_list, axis=0)
            trues_arr = _np.concatenate(trues_list, axis=0)
            mse = float(_np.mean((preds_arr - trues_arr) ** 2))
            mae = float(_np.mean(_np.abs(preds_arr - trues_arr)))
            print(f"  [TimeMixer] pred_len={pred_len} → test MSE={mse:.4f}  MAE={mae:.4f}")

            if mse < best_mse:
                best_mse  = mse
                best_mae  = mae
                best_pred = pred_len

    # ── classification downstream ──────────────────────────────────────────────
    cls_acc = None
    if classification_dataset is not None:
        from data_loaders.data_puller import ClassificationDataPuller, make_uea_dataloaders
        from models import TimeMixer as _TM
        import torch.nn as _nn
        import numpy as _np

        cls_dir = cfg["classification_data_dir"]
        cls_bs  = _get_cls_bs(cfg, "batch_size_classification", 64)
        p_s     = patch_len

        # Determine if UEA .ts format or our npy/pt format
        _cls_path = Path(cls_dir) / classification_dataset
        _is_uea   = bool(list(_cls_path.glob("*_TRAIN.ts"))) if _cls_path.exists() else False

        if _is_uea:
            _raw_train, _, _raw_test, n_classes = make_uea_dataloaders(
                cls_dir, classification_dataset, batch_size=cls_bs)
            # Extract underlying datasets and rebuild with patch collation
            _ds_train = _raw_train.dataset
            _ds_test  = _raw_test.dataset
            n_vars   = _ds_train._samples[0].shape[-1]
            _max_T   = max(s.shape[0] for s in _ds_train._samples + _ds_test._samples)
            _seq_len = int(_np.ceil(_max_T / p_s)) * p_s

            def _uea_collate(batch):
                import torch.nn.functional as F
                xs, ys, orig_lens = zip(*batch)
                orig_lens = torch.stack(orig_lens)
                max_t = max(x.shape[0] for x in xs)
                xs = torch.stack([F.pad(x, (0, 0, 0, max_t - x.shape[0])) for x in xs])
                T = xs.shape[1]
                if T < _seq_len:
                    xs = torch.cat([xs, torch.zeros(xs.shape[0], _seq_len - T, xs.shape[2])], dim=1)
                elif T > _seq_len:
                    xs = xs[:, :_seq_len, :]
                # padding mask at timestep level: True = real data
                ts_mask = torch.arange(_seq_len).unsqueeze(0) < orig_lens.unsqueeze(1)
                return xs, torch.stack(ys), ts_mask.float()

            cls_train = torch.utils.data.DataLoader(
                _ds_train, batch_size=cls_bs, shuffle=True,  collate_fn=_uea_collate)
            cls_test  = torch.utils.data.DataLoader(
                _ds_test,  batch_size=cls_bs, shuffle=False, collate_fn=_uea_collate)
        else:
            def _mk(split):
                ds = ClassificationDataPuller(cls_dir, classification_dataset, p_s, which=split)
                return torch.utils.data.DataLoader(ds, batch_size=cls_bs, shuffle=(split == "train"))
            cls_train = _mk("train")
            cls_test  = _mk("test")
            n_classes = cls_train.dataset.n_classes
            n_vars    = cls_train.dataset.X.shape[2]
            _seq_len  = cls_train.dataset.X.shape[1]

        # Build TimeMixer for classification (channel_independence=0, no downsampling)
        cls_args = SimpleNamespace(**vars(base_args))
        cls_args.task_name          = 'classification'
        cls_args.seq_len            = _seq_len
        cls_args.pred_len           = 0
        cls_args.enc_in             = n_vars
        cls_args.dec_in             = n_vars
        cls_args.c_out              = n_vars
        cls_args.num_class          = n_classes
        cls_args.channel_independence = 0   # multivariate mode required for classification
        cls_args.down_sampling_layers = 0   # no downsampling — seq_len varies per dataset

        cls_model = _TM.Model(cls_args).float().to(_device)
        cls_optim = torch.optim.Adam(cls_model.parameters(),
                                     lr=cfg.get('lr_classification', 1e-3))
        cls_crit  = _nn.CrossEntropyLoss()
        n_epochs_cls = cfg.get('epochs_classification', 30)
        patience_cls = cfg.get('patience_classification', 5)
        best_acc, no_improve = 0.0, 0

        print(f"\n[TimeMixer] Classification on {classification_dataset} "
              f"({n_classes} classes, {n_vars} vars, seq={_seq_len}) …")

        for epoch in range(n_epochs_cls):
            cls_model.train()
            for batch in cls_train:
                if _is_uea:
                    bx, by, bmask = batch
                    bx = bx.float().to(_device)
                else:
                    # ClassificationDataPuller: (patches, y, padding_mask)
                    patches, by, pmask = batch
                    bx    = patches.reshape(patches.shape[0], -1, patches.shape[-1]).float().to(_device)
                    # expand patch-level mask to timestep-level
                    bmask = pmask.float().repeat_interleave(p_s, dim=1).to(_device)
                by = by.long().to(_device)
                bmask = bmask.to(_device)

                logits = cls_model(bx, bmask, None, None)
                # cls_model returns [B, num_class]
                loss = cls_crit(logits, by)
                cls_optim.zero_grad()
                loss.backward()
                _nn.utils.clip_grad_norm_(cls_model.parameters(), max_norm=4.0)
                cls_optim.step()

            # ── validation on test set ────────────────────────────────────────
            cls_model.eval()
            all_preds, all_true = [], []
            with torch.no_grad():
                for batch in cls_test:
                    if _is_uea:
                        bx, by, bmask = batch
                        bx = bx.float().to(_device)
                    else:
                        patches, by, pmask = batch
                        bx    = patches.reshape(patches.shape[0], -1, patches.shape[-1]).float().to(_device)
                        bmask = pmask.float().repeat_interleave(p_s, dim=1).to(_device)
                    by = by.long().to(_device)
                    bmask = bmask.to(_device)
                    logits = cls_model(bx, bmask, None, None)
                    all_preds.append(logits.argmax(dim=-1).cpu())
                    all_true.append(by.cpu())
            acc = (torch.cat(all_preds) == torch.cat(all_true)).float().mean().item()
            print(f"  Epoch {epoch+1}/{n_epochs_cls}  test acc={acc:.4f}")
            if acc > best_acc:
                best_acc  = acc
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience_cls:
                    print("  Early stopping.")
                    break

        cls_acc = best_acc
        print(f"\n{'='*60}")
        print(f"  [TimeMixer] Classification on {classification_dataset}")
        print(f"  Test Accuracy: {cls_acc:.4f}")
        print(f"{'='*60}")

    # ── anomaly detection downstream ───────────────────────────────────────────
    anom_result = None
    if anomaly_dataset is not None:
        from data_loaders.data_puller import AnomalyDataPuller
        from models import TimeMixer as _TM
        import torch.nn as _nn
        import numpy as _np
        from sklearn.metrics import f1_score

        anom_dir = cfg["anomaly_data_dir"]
        anom_bs  = cfg.get('batch_size', 64)

        anom_train_ds = AnomalyDataPuller(anom_dir, anomaly_dataset, patch_len, which="train")
        anom_test_ds  = AnomalyDataPuller(anom_dir, anomaly_dataset, patch_len, which="test")
        _anom_seq     = anom_train_ds.padded_T
        _anom_vars    = anom_train_ds.n_vars

        anom_train_loader = torch.utils.data.DataLoader(
            anom_train_ds, batch_size=anom_bs, shuffle=False, drop_last=False)
        anom_test_loader  = torch.utils.data.DataLoader(
            anom_test_ds,  batch_size=anom_bs, shuffle=False, drop_last=False)

        # Build TimeMixer for anomaly detection (reconstruction objective)
        an_args = SimpleNamespace(**vars(base_args))
        an_args.task_name           = 'anomaly_detection'
        an_args.seq_len             = _anom_seq
        an_args.pred_len            = 0
        an_args.enc_in              = _anom_vars
        an_args.dec_in              = _anom_vars
        an_args.c_out               = _anom_vars
        an_args.channel_independence = 0
        an_args.down_sampling_layers = 0

        an_model = _TM.Model(an_args).float().to(_device)
        an_optim = torch.optim.Adam(an_model.parameters(), lr=cfg.get('learning_rate', 1e-3))
        an_crit  = _nn.MSELoss()
        n_epochs_an = cfg.get('train_epochs', 10)

        print(f"\n[TimeMixer] Anomaly detection on {anomaly_dataset} "
              f"({_anom_vars} vars, win={_anom_seq}) …")

        if not skip_train:
            for epoch in range(n_epochs_an):
                an_model.train()
                ep_loss = []
                for batch in anom_train_loader:
                    patches = batch[0].float().to(_device)   # [B, n_patches, patch_len, C]
                    bx = patches.reshape(patches.shape[0], -1, patches.shape[-1])  # [B, T, C]
                    recon = an_model(bx, None, None, None)
                    loss  = an_crit(recon, bx)
                    an_optim.zero_grad()
                    loss.backward()
                    an_optim.step()
                    ep_loss.append(loss.item())
                print(f"  Epoch {epoch+1}/{n_epochs_an} loss={_np.mean(ep_loss):.4f}")

        # ── compute anomaly scores (reconstruction error per timestep) ─────────
        an_model.eval()
        scores_list, labels_list = [], []
        with torch.no_grad():
            for batch in anom_test_loader:
                patches, lbls = batch[0].float().to(_device), batch[1]
                bx    = patches.reshape(patches.shape[0], -1, patches.shape[-1])
                recon = an_model(bx, None, None, None)
                err   = ((recon - bx) ** 2).mean(dim=-1)   # [B, T]
                scores_list.append(err.cpu().numpy())
                labels_list.append(lbls.numpy())

        scores = _np.concatenate(scores_list, axis=0).reshape(-1)  # per-timestep MSE
        labels = _np.concatenate(labels_list, axis=0).reshape(-1)

        _ratio = _get_anomaly_ratio(anomaly_dataset, cfg)
        thresh = _np.percentile(scores, 100 - _ratio)
        preds  = (scores > thresh).astype(int)
        f1    = f1_score(labels, preds, zero_division=0)
        print(f"  [TimeMixer] Anomaly {anomaly_dataset} → F1={f1:.4f} (thresh={thresh:.4f})")
        anom_result = f1

    return best_pred, best_mse, best_mae, cls_acc, anom_result


# ── Autoformer / FEDformer / DLinear (TSLib supervised baselines) ─────────────

def _run_tslib_forecast(
    model_name: str,
    cfg: dict,
    forecast_dataset: str = None,
    skip_train: bool = False,
    pred_lens=None,
    epochs: int = None,
    encoder_layers: int = None,
    embed_dim: int = None,
    lr: float = None,
    pretrain_only: bool = False,
    pretrain_dataset: str = None,
    classification_dataset=None,
    anomaly_dataset=None,
):
    """Generic runner for any Exp_Long_Term_Forecast-compatible TSLib model."""
    if pretrain_only:
        print(f"[{model_name}] pretrain_only=True — supervised model, skipping.")
        return

    if pred_lens is None:
        pred_lens = [96, 192, 336, 720]

    timemixer_dir = Path(__file__).parent / "TimeMixer-main"
    shared_dir    = Path(__file__).parent / "shared"
    _add_path(shared_dir)

    import sys as _sys, importlib.util as _ilu, torch, types as _types
    from types import SimpleNamespace
    import numpy as _np

    _tm_str = str(timemixer_dir)
    if _tm_str not in _sys.path:
        _sys.path.insert(0, _tm_str)
    for _key in list(_sys.modules.keys()):
        if _key in ('exp', 'models') or _key.startswith('exp.') or _key.startswith('models.'):
            _sys.modules.pop(_key, None)

    from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

    if encoder_layers is not None: cfg['e_layers'] = encoder_layers
    if embed_dim      is not None:
        cfg['d_model'] = embed_dim
        cfg['d_ff']    = embed_dim * 2
    if lr             is not None: cfg['learning_rate'] = lr

    _gpu_idx = 0
    seq_len   = cfg['seq_len']
    patch_len = cfg.get('patch_len', 16)
    freq      = cfg.get('freq', 'h')
    label_len = cfg.get('label_len', 0)

    print(f"\n{'='*60}")
    print(f"  MODEL: {model_name}  (supervised, no pretraining)")
    print(f"  forecast: {forecast_dataset or '(none)'}")
    print(f"  e_layers={cfg.get('e_layers','N/A')}  d_model={cfg.get('d_model','N/A')}  seq_len={seq_len}")
    print(f"{'='*60}")

    base_args = SimpleNamespace(
        model                       = model_name,
        task_name                   = 'long_term_forecast',
        use_gpu                     = torch.cuda.is_available(),
        gpu                         = _gpu_idx,
        use_multi_gpu               = False,
        devices                     = str(_gpu_idx),
        device_ids                  = [_gpu_idx],
        seq_len                     = seq_len,
        label_len                   = label_len,
        pred_len                    = 96,
        enc_in                      = 1,
        dec_in                      = 1,
        c_out                       = 1,
        d_model                     = cfg.get('d_model', 512),
        n_heads                     = cfg.get('n_heads', 8),
        e_layers                    = cfg.get('e_layers', 2),
        d_layers                    = cfg.get('d_layers', 1),
        d_ff                        = cfg.get('d_ff', 2048),
        dropout                     = cfg.get('dropout', 0.05),
        embed                       = freq,
        freq                        = freq,
        factor                      = cfg.get('factor', 1),
        moving_avg                  = cfg.get('moving_avg', 25),
        activation                  = cfg.get('activation', 'gelu'),
        output_attention            = False,
        data                        = 'custom',
        root_path                   = '/tmp',
        data_path                   = 'data.csv',
        inverse                     = False,
        checkpoints                 = str(Path(__file__).parent / 'outputs' / f'{model_name.lower()}_forecast'),
        num_workers                 = cfg.get('num_workers', 4),
        train_epochs                = epochs if epochs is not None else cfg.get('train_epochs', 10),
        batch_size                  = cfg.get('batch_size', 32),
        learning_rate               = cfg['learning_rate'],
        patience                    = cfg.get('patience', 5),
        lradj                       = cfg.get('lradj', 'TST'),
        pct_start                   = cfg.get('pct_start', 0.2),
        loss                        = cfg.get('loss', 'MSE'),
        drop_last                   = cfg.get('drop_last', True),
        use_amp                     = False,
        modes                       = cfg.get('modes', 32),
        mode_select                 = cfg.get('mode_select', 'random'),
        features                    = cfg.get('features', 'M'),
        # unused but avoids AttributeError in Exp_Long_Term_Forecast
        channel_independence        = 1,
        down_sampling_layers        = 0,
        down_sampling_window        = 1,
        down_sampling_method        = 'avg',
        decomp_method               = 'moving_avg',
        use_norm                    = 1,
        use_future_temporal_feature = 0,
        top_k                       = 5,
        num_kernels                 = 6,
    )

    if forecast_dataset is not None:
        from data_loaders.data_puller import PatchTSTForcastingAdapter

        ds_info  = get_dataset_info(forecast_dataset)
        _csv     = ds_info['csv_path']
        _c_in    = ds_info['c_in']
        _fc_bs   = cfg.get('batch_size_forecast', cfg.get('batch_size', 32))
        _fc_nw   = cfg.get('num_workers', 4)
        _n_epochs_fc = epochs if epochs is not None else cfg.get('train_epochs', 10)

        best_mse, best_mae, best_pred = float('inf'), float('inf'), None

        for pred_len in pred_lens:
            def _fc_loader(split, _pl=pred_len):
                ds = _FlatWindowAdapterTM(
                    PatchTSTForcastingAdapter(_csv, split, seq_len, _pl, patch_len,
                                             label_len=label_len),
                    freq=freq, label_len=label_len)
                return torch.utils.data.DataLoader(
                    ds, batch_size=_fc_bs, shuffle=(split == 'train'),
                    num_workers=_fc_nw, drop_last=(split == 'train'))

            _tm_train = _fc_loader('train')
            _tm_val   = _fc_loader('val')
            _tm_test  = _fc_loader('test')

            ft_args = SimpleNamespace(**vars(base_args))
            ft_args.pred_len      = pred_len
            ft_args.enc_in        = _c_in
            ft_args.dec_in        = _c_in
            ft_args.c_out         = _c_in
            ft_args.train_epochs  = _n_epochs_fc
            ft_args.batch_size    = _fc_bs

            setting = (f"{model_name.lower()}_{forecast_dataset}_pl{pred_len}"
                       f"_dm{cfg.get('d_model','na')}_el{cfg.get('e_layers','na')}")

            print(f"\n[{model_name}] Forecasting pred_len={pred_len} on {forecast_dataset} …")

            exp = Exp_Long_Term_Forecast(ft_args)

            def _get_data(self, flag):
                loader = {'train': _tm_train, 'val': _tm_val, 'test': _tm_test}[flag]
                return loader.dataset, loader
            exp._get_data = _types.MethodType(_get_data, exp)

            if not skip_train:
                exp.train(setting)

            # ── evaluate ──────────────────────────────────────────────────────
            exp.model.eval()
            preds_list, trues_list = [], []
            _fdev = exp.device

            with torch.no_grad():
                for batch_x, batch_y, batch_x_mark, batch_y_mark in _tm_test:
                    B = batch_x.shape[0]
                    batch_x      = batch_x.float().to(_fdev)
                    batch_y      = batch_y.float().to(_fdev)
                    batch_x_mark = batch_x_mark.float().to(_fdev)

                    if label_len > 0:
                        # Encoder-decoder models: last label_len timesteps as decoder context
                        dec_ctx   = batch_x[:, -label_len:, :]
                        dec_zeros = torch.zeros(B, pred_len, _c_in, device=_fdev)
                        dec_inp   = torch.cat([dec_ctx, dec_zeros], dim=1)
                        mark_dim  = batch_x_mark.shape[-1]
                        dec_mark  = torch.zeros(B, label_len + pred_len, mark_dim, device=_fdev)
                    else:
                        dec_inp  = None
                        dec_mark = None

                    outputs = exp.model(batch_x, batch_x_mark, dec_inp, dec_mark)
                    outputs = outputs[:, -pred_len:, :]
                    preds_list.append(outputs.detach().cpu().numpy())
                    trues_list.append(batch_y[:, -pred_len:, :].detach().cpu().numpy())

            exp.model.train()

            preds_arr = _np.concatenate(preds_list, axis=0)
            trues_arr = _np.concatenate(trues_list, axis=0)
            mse = float(_np.mean((preds_arr - trues_arr) ** 2))
            mae = float(_np.mean(_np.abs(preds_arr - trues_arr)))
            print(f"  [{model_name}] pred_len={pred_len} → test MSE={mse:.4f}  MAE={mae:.4f}")

            if mse < best_mse:
                best_mse  = mse
                best_mae  = mae
                best_pred = pred_len

        if best_pred is not None:
            print(f"\n[{model_name}] Best: pred_len={best_pred}  MSE={best_mse:.4f}  MAE={best_mae:.4f}")


def _load_tslib_cfg(config_filename: str) -> dict:
    import importlib.util as _ilu
    timemixer_dir = Path(__file__).parent / "TimeMixer-main"
    spec = _ilu.spec_from_file_location("_cfg", timemixer_dir / config_filename)
    mod  = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {**DATA_PATHS, **dict(mod.config)}


def run_autoformer(skip_train: bool = False, pretrain_dataset: str = None,
                   forecast_dataset: str = None, classification_dataset=None,
                   anomaly_dataset=None, pretrain_only: bool = False,
                   pred_lens=None, encoder_layers: int = None, lr: float = None,
                   embed_dim: int = None, epochs: int = None, **_):
    cfg = _load_tslib_cfg("config_autoformer.py")
    _run_tslib_forecast('Autoformer', cfg, forecast_dataset=forecast_dataset,
                        skip_train=skip_train, pred_lens=pred_lens, epochs=epochs,
                        encoder_layers=encoder_layers, embed_dim=embed_dim, lr=lr,
                        pretrain_only=pretrain_only)


def run_fedformer(skip_train: bool = False, pretrain_dataset: str = None,
                  forecast_dataset: str = None, classification_dataset=None,
                  anomaly_dataset=None, pretrain_only: bool = False,
                  pred_lens=None, encoder_layers: int = None, lr: float = None,
                  embed_dim: int = None, epochs: int = None, **_):
    cfg = _load_tslib_cfg("config_fedformer.py")
    _run_tslib_forecast('FEDformer', cfg, forecast_dataset=forecast_dataset,
                        skip_train=skip_train, pred_lens=pred_lens, epochs=epochs,
                        encoder_layers=encoder_layers, embed_dim=embed_dim, lr=lr,
                        pretrain_only=pretrain_only)


def run_dlinear(skip_train: bool = False, pretrain_dataset: str = None,
                forecast_dataset: str = None, classification_dataset=None,
                anomaly_dataset=None, pretrain_only: bool = False,
                pred_lens=None, encoder_layers: int = None, lr: float = None,
                embed_dim: int = None, epochs: int = None, **_):
    cfg = _load_tslib_cfg("config_dlinear.py")
    _run_tslib_forecast('DLinear', cfg, forecast_dataset=forecast_dataset,
                        skip_train=skip_train, pred_lens=pred_lens, epochs=epochs,
                        encoder_layers=encoder_layers, embed_dim=embed_dim, lr=lr,
                        pretrain_only=pretrain_only)


# ── entry point ───────────────────────────────────────────────────────────────

RUNNERS = {
    "dino_timemixer":  functools.partial(run_dino, backbone="timemixer"),
    "dino_patchtst":   functools.partial(run_dino, backbone="patchtst"),
    "dino_ts2vec":     functools.partial(run_dino, backbone="ts2vec"),
    "dino_timesnet":   functools.partial(run_dino, backbone="timesnet"),
    "dino_timerxl":    functools.partial(run_dino, backbone="timerxl"),
    "dino_itransformer": functools.partial(run_dino, backbone="itransformer"),
    "patchtst":        run_patchtst,
    "patchtst_random": lambda skip_train=False, pretrain_dataset=None, forecast_dataset=None, classification_dataset=None, anomaly_dataset=None, pretrain_only=False, classification_only=False, pred_lens=None, checkpoints=None, encoder_layers=None, pretrain_source=None, num_patches=None, linear_probe=True, head_type="linear": run_patchtst(skip_train=skip_train, pretrain_dataset=pretrain_dataset, forecast_dataset=forecast_dataset, classification_dataset=classification_dataset, anomaly_dataset=anomaly_dataset, pretrain_only=pretrain_only, classification_only=classification_only, pred_lens=pred_lens, checkpoints=checkpoints, random_encoder=True, encoder_layers=encoder_layers, pretrain_source=pretrain_source, num_patches=num_patches, linear_probe=linear_probe, head_type=head_type),
    "timemixer":       run_timemixer,
    "autoformer":      run_autoformer,
    "fedformer":       run_fedformer,
    "dlinear":         run_dlinear,
}

def run(model: str,
        task: str = None,
        skip_train: bool = False,
        dataset: str = None,
        pretrain_dataset: str = None,
        forecast_dataset: str = None,
        classification_dataset=None,
        anomaly_dataset: str = None,
        checkpoint: str = None,
        pred_lens=None,
        checkpoints=None,
        pretrain_only: bool = False,
        pretrain_on_classification: bool = False,
        pretrain_on_anomaly: bool = False,
        pretrain_val_fraction: float = 0.1,
        epochs_classification: int = None,
        cls_head_mode: str = "both",
        lr_classification: float = None,
        lr_classification_encoder: float = None,
        label_smoothing: float = None,
        cls_kfold: int = None,
        pred_len: int = None,
        encoder_layers: int = None,
        predictor_layers: int = None,
        lr: float = None,
        lr_pred: float = None,
        pretrain_source: str = None,
        gpu: int = None,
        num_patches: int = None,
        seq_len: int = None,
        seed: int = None,
        pretrain_cls_model: bool = False,
        linear_probe: bool = True,
        head_type: str = "linear",
        output_dir: str = None,
        embed_dim: int = None,
        predictor_embed_dim: int = None,
        out_dim: int = None,
        d_ff: int = None,
        n_heads: int = None,
        epochs: int = None,
        epochs_forecasting: int = None,
        warmup_epochs: int = None,
        ckpt_tag: str = None,
        aug_global: str = None,
        aug_local: str = None,
        n_global_crops: int = None,
        n_local_crops: int = None,
        global_crop_ratio: float = None,
        local_crop_ratio: float = None,
        mlm_phi: float = None,
        mlm_mode: str = None,
        mlm_block_size: int = None,
        batch_size: int = None,
        backbone_type: str = None,
        dwt_wavelet_pool: list = None,
        soft_threshold_sigma: float = None,
        use_koleo: bool = None,
        koleo_weight: float = None,
        use_vicreg: bool = None,
        vicreg_std_coeff: float = None,
        vicreg_cov_coeff: float = None,
        synthetic_data_dir: str = None,
        subset_frac: float = None,
        window_stride: int = None,
        phi: float = None,
        tsmixer_e_layers: int = None,
        patch_len: int = None,
        lr_forecasting: float = None):
    """
    Unified entry point. Each run handles ONE task.

    task="pretrain"   — pretrain only (same as pretrain_only=True)
    task="forecast"   — skip pretraining, run forecasting only (same as skip_train=True)
    task="classify"   — skip pretraining, run classification only

    Backwards compatible — old flags (skip_train, pretrain_only) still work when task=None.

    Examples:
        run(model="dino", task="pretrain")
        run(model="dino", task="forecast", forecast_dataset="etth1", skip_train=True)
        run(model="dino", task="classify",
            classification_dataset="EthanolConcentration", skip_train=True)

        # Old style still works:
        run(model="dino", skip_train=False)
        run(model="dino", pretrain_only=True)
    """
    # ── resolve task → old flags (backwards compat) ───────────────────────────
    classification_only = False
    if task is not None:
        task = task.lower()
        if task == "pretrain":
            pretrain_only = True
            skip_train    = False
        elif task == "forecast":
            skip_train    = True
            pretrain_only = False
            classification_dataset = None   # force no classification
        elif task == "classify":
            skip_train           = True
            pretrain_only        = False
            classification_only  = True
            forecast_dataset     = None     # force no forecasting
            anomaly_dataset      = None
        elif task == "anomaly":
            skip_train    = True
            pretrain_only = False
            forecast_dataset       = None
            classification_dataset = None
        else:
            raise ValueError(f"Unknown task '{task}'. Choose: pretrain | forecast | classify | anomaly")

    global _SEED_TAG
    if seed is not None:
        _set_seed(seed)
        _SEED_TAG = f'_seed{seed}'
    else:
        _set_seed()
        _SEED_TAG = ''
    model = model.lower()
    if model not in RUNNERS:
        raise ValueError(f"Unknown model '{model}'. Choose from: {list(RUNNERS)}")
    # 'dataset' is shorthand for pretrain_dataset == forecast_dataset
    if dataset is not None:
        pretrain_dataset = pretrain_dataset or dataset
        forecast_dataset = forecast_dataset or dataset
    runner = RUNNERS[model]
    import inspect
    sig = inspect.signature(runner)
    kwargs = dict(skip_train=skip_train,
                  pretrain_dataset=pretrain_dataset,
                  forecast_dataset=forecast_dataset)
    if 'pretrain_only'          in sig.parameters: kwargs['pretrain_only']          = pretrain_only
    if 'pretrain_on_classification' in sig.parameters: kwargs['pretrain_on_classification'] = pretrain_on_classification
    if 'pretrain_on_anomaly'   in sig.parameters: kwargs['pretrain_on_anomaly']   = pretrain_on_anomaly
    if 'pretrain_val_fraction' in sig.parameters: kwargs['pretrain_val_fraction'] = pretrain_val_fraction
    if 'epochs_classification' in sig.parameters: kwargs['epochs_classification'] = epochs_classification
    if 'cls_head_mode'         in sig.parameters: kwargs['cls_head_mode']         = cls_head_mode
    if 'lr_classification'     in sig.parameters: kwargs['lr_classification']     = lr_classification
    if 'lr_classification_encoder' in sig.parameters: kwargs['lr_classification_encoder'] = lr_classification_encoder
    if 'label_smoothing'       in sig.parameters: kwargs['label_smoothing']       = label_smoothing
    if 'cls_kfold'             in sig.parameters: kwargs['cls_kfold']             = cls_kfold
    if 'classification_only'   in sig.parameters: kwargs['classification_only']   = classification_only
    if 'pred_lens'              in sig.parameters: kwargs['pred_lens']              = pred_lens
    if 'checkpoints'            in sig.parameters: kwargs['checkpoints']            = checkpoints
    if 'pred_len'               in sig.parameters: kwargs['pred_len']               = pred_len
    if 'encoder_layers'         in sig.parameters: kwargs['encoder_layers']         = encoder_layers
    if 'predictor_layers'       in sig.parameters: kwargs['predictor_layers']       = predictor_layers
    if 'lr'                     in sig.parameters: kwargs['lr']                     = lr
    if 'lr_pred'                in sig.parameters: kwargs['lr_pred']                = lr_pred
    if 'classification_dataset' in sig.parameters: kwargs['classification_dataset'] = classification_dataset
    if 'anomaly_dataset'        in sig.parameters: kwargs['anomaly_dataset']        = anomaly_dataset
    if 'checkpoint'             in sig.parameters: kwargs['checkpoint']             = checkpoint
    if 'ckpt_tag'               in sig.parameters: kwargs['ckpt_tag']               = ckpt_tag
    if 'pretrain_source'        in sig.parameters: kwargs['pretrain_source']        = pretrain_source
    if 'gpu'                    in sig.parameters: kwargs['gpu']                    = gpu
    if 'num_patches'            in sig.parameters: kwargs['num_patches']            = num_patches
    if 'seq_len'               in sig.parameters: kwargs['seq_len']               = seq_len
    if 'seed'                   in sig.parameters: kwargs['seed']                   = seed
    if 'pretrain_cls_model'     in sig.parameters: kwargs['pretrain_cls_model']     = pretrain_cls_model
    if 'linear_probe'           in sig.parameters: kwargs['linear_probe']           = linear_probe
    if 'head_type'              in sig.parameters: kwargs['head_type']              = head_type
    if 'output_dir'             in sig.parameters: kwargs['output_dir']             = output_dir
    if 'embed_dim'              in sig.parameters: kwargs['embed_dim']              = embed_dim
    if 'predictor_embed_dim'   in sig.parameters: kwargs['predictor_embed_dim']   = predictor_embed_dim
    if 'out_dim'               in sig.parameters: kwargs['out_dim']               = out_dim
    if 'd_ff'                  in sig.parameters: kwargs['d_ff']                  = d_ff
    if 'n_heads'               in sig.parameters: kwargs['n_heads']               = n_heads
    if 'epochs'                in sig.parameters: kwargs['epochs']                = epochs
    if 'epochs'                in sig.parameters: kwargs['epochs']                = epochs
    if 'epochs_forecasting'    in sig.parameters: kwargs['epochs_forecasting']    = epochs_forecasting
    if 'warmup_epochs'         in sig.parameters: kwargs['warmup_epochs']         = warmup_epochs
    if 'ckpt_tag'              in sig.parameters: kwargs['ckpt_tag']              = ckpt_tag
    if 'aug_global'            in sig.parameters: kwargs['aug_global']            = aug_global
    if 'aug_local'             in sig.parameters: kwargs['aug_local']             = aug_local
    if 'n_global_crops'        in sig.parameters: kwargs['n_global_crops']        = n_global_crops
    if 'n_local_crops'         in sig.parameters: kwargs['n_local_crops']         = n_local_crops
    if 'global_crop_ratio'     in sig.parameters: kwargs['global_crop_ratio']     = global_crop_ratio
    if 'local_crop_ratio'      in sig.parameters: kwargs['local_crop_ratio']      = local_crop_ratio
    if 'mlm_phi'               in sig.parameters: kwargs['mlm_phi']               = mlm_phi
    if 'mlm_mode'              in sig.parameters: kwargs['mlm_mode']              = mlm_mode
    if 'mlm_block_size'        in sig.parameters: kwargs['mlm_block_size']        = mlm_block_size
    if 'batch_size'            in sig.parameters: kwargs['batch_size']            = batch_size
    if 'backbone_type'         in sig.parameters: kwargs['backbone_type']         = backbone_type
    if 'dwt_wavelet_pool'      in sig.parameters: kwargs['dwt_wavelet_pool']      = dwt_wavelet_pool
    if 'soft_threshold_sigma'  in sig.parameters: kwargs['soft_threshold_sigma']  = soft_threshold_sigma
    if 'use_koleo'             in sig.parameters: kwargs['use_koleo']             = use_koleo
    if 'koleo_weight'          in sig.parameters: kwargs['koleo_weight']          = koleo_weight
    if 'use_vicreg'            in sig.parameters: kwargs['use_vicreg']            = use_vicreg
    if 'vicreg_std_coeff'      in sig.parameters: kwargs['vicreg_std_coeff']      = vicreg_std_coeff
    if 'vicreg_cov_coeff'      in sig.parameters: kwargs['vicreg_cov_coeff']      = vicreg_cov_coeff
    if 'synthetic_data_dir'    in sig.parameters: kwargs['synthetic_data_dir']    = synthetic_data_dir
    if 'subset_frac'           in sig.parameters: kwargs['subset_frac']           = subset_frac
    if 'window_stride'         in sig.parameters: kwargs['window_stride']         = window_stride
    if 'phi'                   in sig.parameters: kwargs['phi']                   = phi
    if 'tsmixer_e_layers'      in sig.parameters: kwargs['tsmixer_e_layers']      = tsmixer_e_layers
    if 'patch_len'             in sig.parameters: kwargs['patch_len']             = patch_len
    if 'lr_forecasting'        in sig.parameters: kwargs['lr_forecasting']        = lr_forecasting
    return runner(**kwargs)


if __name__ == "__main__":
    from dataset_registry import DATASETS as _DATASETS
    parser = argparse.ArgumentParser(description="Unified training + forecasting runner")
    parser.add_argument(
        "--model", type=str, required=True,
        choices=list(RUNNERS),
        help="Which model to run: dino_timemixer | dino_patchtst | dino_ts2vec | dino_itransformer | patchtst | timemixer | autoformer | fedformer | dlinear",
    )
    parser.add_argument(
        "--pretrain_dataset", type=str, default=None,
        choices=list(_DATASETS),
        help=f"Dataset for pretraining. Available: {list(_DATASETS)}",
    )
    parser.add_argument(
        "--forecast_dataset", type=str, default=None,
        choices=list(_DATASETS),
        help="Dataset for forecasting downstream (defaults to pretrain_dataset).",
    )
    parser.add_argument(
        "--skip_train", type=str, default="false",
        choices=["true", "false"],
        help="Skip pretraining and go straight to forecasting (true | false)",
    )
    parser.add_argument(
        "--pretrain_only", type=str, default="false",
        choices=["true", "false"],
        help="Run pretraining only, skip downstream evaluation (true | false)",
    )
    parser.add_argument(
        "--pretrain_on_classification", type=str, default="false",
        choices=["true", "false"],
        help="DINO-pretrain on the classification dataset's TRAIN series (no labels), "
             "then run BOTH a linear probe and a full fine-tune from the same checkpoint. "
             "Requires --classification_dataset; backbone=timemixer only.",
    )
    parser.add_argument(
        "--pretrain_on_anomaly", type=str, default="false",
        choices=["true", "false"],
        help="DINO-pretrain on the anomaly dataset's NORMAL (train) stream (no labels), "
             "then fine-tune the reconstruction detector (encoder+decoder) from the same "
             "checkpoint. Window = num_patches × patch_len (default 10×10=100). "
             "Requires --anomaly_dataset; backbone=timemixer only.",
    )
    parser.add_argument(
        "--epochs_classification", type=int, default=None,
        help="Epochs for the downstream classification head(s) in the "
             "pretrain_on_classification flow. Default: config epoch_classification (20).",
    )
    parser.add_argument(
        "--cls_head_mode", type=str, default="both",
        choices=["both", "fine_tune", "linear_probe"],
        help="Which downstream head(s) to run after classification pretraining: "
             "'both' (probe + fine-tune), 'fine_tune' only, or 'linear_probe' only.",
    )
    parser.add_argument(
        "--lr_classification", type=float, default=None,
        help="Downstream classification HEAD learning rate (pretrain_on_classification flow).",
    )
    parser.add_argument(
        "--lr_classification_encoder", type=float, default=None,
        help="Downstream ENCODER learning rate for fine-tuning (discriminative LR). "
             "Set lower than --lr_classification (e.g. 2e-4) to preserve pretrained features.",
    )
    parser.add_argument(
        "--label_smoothing", type=float, default=None,
        help="Label smoothing for the classification cross-entropy (e.g. 0.1).",
    )
    parser.add_argument(
        "--cls_kfold", type=int, default=None,
        help="k-fold validation for the pretrain_on_classification head: each fold holds out "
             "a different 1/k of TRAIN as the best-epoch val, trains on the rest, evaluates the "
             "fixed _TEST, and reports mean±std across folds. Default 1 (single 10%% holdout).",
    )
    parser.add_argument(
        "--pretrain_val_fraction", type=float, default=0.1,
        help="Fraction of the classification TRAIN series held out as an SSL val set "
             "to pick the pretraining checkpoint (best-by-val). Auto-skipped when the "
             "holdout would be < pretrain_val_min (32) samples → final epoch is used. "
             "Set 0 to always use the final epoch. (pretrain_on_classification only)",
    )
    parser.add_argument(
        "--linear_probe", type=str, default="true",
        choices=["true", "false"],
        help="Forecast head mode: true = linear probe (backbone FROZEN), "
             "false = fine-tune the full backbone end-to-end on the target.",
    )
    parser.add_argument("--task", type=str, default=None,
                        choices=["pretrain", "forecast", "classify", "anomaly"],
                        help="Task to run: pretrain | forecast | classify. "
                             "Overrides --skip_train / --pretrain_only when set.")
    parser.add_argument("--classification_dataset", type=str, default=None,
                        help="Dataset name for classification (subfolder under Classification_TS dir)")
    parser.add_argument("--anomaly_dataset",  type=str, default=None,
                        help="Dataset name for anomaly detection (subfolder under Anomaly_TS dir)")
    parser.add_argument("--checkpoint",       type=str, default=None,
                        help="Path to pretrained checkpoint to load for classification")
    parser.add_argument("--encoder_layers",   type=int,   default=None,
                        help="Override number of encoder transformer layers")
    parser.add_argument("--predictor_layers", type=int,   default=None,
                        help="Override number of predictor layers (JEPA models only)")
    parser.add_argument("--lr",               type=float, default=None,
                        help="Override pretraining learning rate")
    parser.add_argument("--pretrain_source",  type=str,   default=None,
                        choices=["monash", "synthetic", "monash+synthetic"],
                        help="Override pretrain data source (dino only)")
    parser.add_argument("--num_patches",      type=int,   default=None,
                        help="Override number of patches (context window = num_patches × patch_size). "
                             "Note: with --classification_dataset this relocates the checkpoint lookup "
                             "to a classification/<name>_cw<cw> subdir. Use --seq_len to set the window "
                             "size WITHOUT relocating (keeps the flat pretrain checkpoint dir).")
    parser.add_argument("--seq_len",          type=int,   default=None,
                        help="Context window in timesteps; sets num_patches = seq_len // patch_len "
                             "without relocating the checkpoint dir (unlike --num_patches).")
    parser.add_argument("--embed_dim",           type=int, default=None,
                        help="Override embedding dim / d_model for the encoder")
    parser.add_argument("--predictor_embed_dim", type=int, default=None,
                        help="Override predictor embedding dim (JEPA only)")
    parser.add_argument("--out_dim",             type=int, default=None,
                        help="Override DINO output bins (out_dim / prototype count)")
    parser.add_argument("--d_ff",                type=int, default=None,
                        help="Override feed-forward dim d_ff for the encoder")
    parser.add_argument("--n_heads",             type=int, default=None,
                        help="Override number of attention heads")
    parser.add_argument("--lr_pred",             type=float, default=None,
                        help="Override predictor learning rate (JEPA only)")
    parser.add_argument("--epochs",              type=int, default=None,
                        help="Override number of pretraining epochs")
    parser.add_argument("--epochs_forecasting",  type=int, default=None,
                        help="Override number of forecasting fine-tune epochs (DINO)")
    parser.add_argument("--checkpoints", nargs="+", default=None,
                        help="Checkpoint epochs to evaluate during forecasting, e.g. --checkpoints 1 3 5 10 best")
    parser.add_argument("--pred_lens", nargs="+", type=int, default=None,
                        help="Forecast horizons to run, e.g. --pred_lens 96 (default: 96 192 336 720)")
    parser.add_argument("--seed",             type=int,   default=None,
                        help="Random seed (also suffixes checkpoint paths with _seedN)")
    parser.add_argument("--pretrain_cls_model", type=str, default="false",
                        help="Use cls embedding model for pretraining (saves to separate _cls checkpoint)")
    parser.add_argument("--head", type=str, default="linear",
                        choices=["linear", "mlp"],
                        help="Downstream head: 'linear' (single Linear) or 'mlp' (1-hidden-layer MLP)")
    parser.add_argument("--warmup_epochs", type=int, default=None,
                        help="Number of LR warmup epochs (DINO only)")
    parser.add_argument("--ckpt_tag", type=str, default=None,
                        help="Extra tag appended to checkpoint directory name (e.g. 'wrLR')")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Explicit checkpoint output directory. If set WITHOUT --encoder_layers, "
                             "it is used verbatim (no auto _src/_layers/_seed suffixing).")
    parser.add_argument("--aug_global", type=str, default=None,
                        help="Global (teacher) augmentation type, overrides config (e.g. 'galilien', 'dwt_soft_threshold')")
    parser.add_argument("--aug_local",  type=str, default=None,
                        help="Local (student) augmentation type, overrides config (e.g. 'lorentz', 'dwt_high_perturb')")
    parser.add_argument("--n_global_crops", type=int, default=None,
                        help="Number of global (teacher) crops — DINO multi-crop (e.g. 2)")
    parser.add_argument("--n_local_crops",  type=int, default=None,
                        help="Number of local (student) crops — DINO multi-crop (e.g. 6)")
    parser.add_argument("--global_crop_ratio", type=float, default=None,
                        help="Crop ratio for global crops (1.0 = no crop)")
    parser.add_argument("--local_crop_ratio",  type=float, default=None,
                        help="Crop ratio for local crops (e.g. 0.4 for DINO-style small crops)")
    parser.add_argument("--mlm_phi",    type=float, default=None,
                        help="MLM mixing weight: phi*DINO + (1-phi)*MLM (DINO only)")
    parser.add_argument("--batch_size", type=int,   default=None,
                        help="Override pretrain batch_size_per_gpu (e.g. lower for 21-ch weather)")
    parser.add_argument("--mlm_block_size", type=int, default=None,
                        help="MLM/MAE masking granularity: mask contiguous spans of N timesteps "
                             "(8 = block masking ON; 1 = per-step masking OFF). Overrides config.")
    parser.add_argument("--mlm_mode",   type=str,   default=None,
                        help="MLM variant: ibot (teacher-guided CE) or mae (MSE vs ground truth)")
    parser.add_argument("--backbone_type", type=str, default=None,
                        help="TSDiNO is TimeMixer-only (tsmixer); kept for forward-compat, no PatchTST path here")
    parser.add_argument("--tsmixer_e_layers", type=int, default=None,
                        help="TimeMixer/tsmixer backbone depth (PDM blocks). Overrides config tsmixer_e_layers (default 3). "
                             "Must match at pretrain AND forecast/load time.")
    parser.add_argument("--patch_len", type=int, default=None,
                        help="Patch size for the PatchTST backbone (also sets stride = non-overlapping). "
                             "Overrides config patch_len (default 16). Must match at pretrain AND forecast/load time.")
    parser.add_argument("--soft_threshold_sigma", type=float, default=None,
                        help="ρ (shrinkage ratio) for soft-threshold DWT/SWT/MODWT augmentation: "
                             "threshold = ρ·max(|detail coeffs|) per level (config default 0.6). "
                             "Only affects *_soft_threshold aug types.")
    parser.add_argument("--dwt_wavelet_pool", nargs="+", default=None,
                        help="Wavelet pool for random-per-sample DWT aug (e.g. sym4 sym6 sym8)")
    parser.add_argument("--use_koleo", type=str, default=None,
                        help="true|false — enable KoLeo regularizer on global feature (DINO only)")
    parser.add_argument("--koleo_weight", type=float, default=None,
                        help="KoLeo scalar weight (default 0.1 from config)")
    parser.add_argument("--use_vicreg", type=str, default=None,
                        help="true|false — enable VICReg var+cov regularizer on global feature (DINO only)")
    parser.add_argument("--vicreg_std_coeff", type=float, default=None,
                        help="VICReg variance term weight")
    parser.add_argument("--vicreg_cov_coeff", type=float, default=None,
                        help="VICReg covariance term weight")
    parser.add_argument("--synthetic_data_dir", type=str, default=None,
                        help="Override synthetic .arrow data dir (DINO synthetic pretraining only)")
    parser.add_argument("--subset_frac", type=float, default=None,
                        help="Train on a fresh random fraction of the pretrain windows each epoch (e.g. 0.25). DINO only.")
    parser.add_argument("--window_stride", type=int, default=None,
                        help="Stride between sliding windows for pretrain + forecast-train (default 1). Subsamples the DINO data pullers. Test eval stays stride-1.")
    args = parser.parse_args()
    run(model=args.model,
        task=args.task,
        pred_lens=args.pred_lens,
        skip_train=args.skip_train.lower() == "true",
        pretrain_dataset=args.pretrain_dataset,
        forecast_dataset=args.forecast_dataset,
        pretrain_only=args.pretrain_only.lower() == "true",
        pretrain_on_classification=args.pretrain_on_classification.lower() == "true",
        pretrain_on_anomaly=args.pretrain_on_anomaly.lower() == "true",
        pretrain_val_fraction=args.pretrain_val_fraction,
        epochs_classification=args.epochs_classification,
        cls_head_mode=args.cls_head_mode,
        lr_classification=args.lr_classification,
        lr_classification_encoder=args.lr_classification_encoder,
        label_smoothing=args.label_smoothing,
        cls_kfold=args.cls_kfold,
        classification_dataset=args.classification_dataset,
        anomaly_dataset=args.anomaly_dataset,
        checkpoint=args.checkpoint,
        encoder_layers=args.encoder_layers,
        predictor_layers=args.predictor_layers,
        lr=args.lr,
        lr_pred=args.lr_pred,
        pretrain_source=args.pretrain_source,
        num_patches=args.num_patches,
        seq_len=args.seq_len,
        embed_dim=args.embed_dim,
        predictor_embed_dim=args.predictor_embed_dim,
        out_dim=args.out_dim,
        d_ff=args.d_ff,
        n_heads=args.n_heads,
        epochs=args.epochs,
        epochs_forecasting=args.epochs_forecasting,
        warmup_epochs=args.warmup_epochs,
        ckpt_tag=args.ckpt_tag,
        output_dir=args.output_dir,
        tsmixer_e_layers=args.tsmixer_e_layers,
        patch_len=args.patch_len,
        aug_global=args.aug_global,
        aug_local=args.aug_local,
        n_global_crops=args.n_global_crops,
        n_local_crops=args.n_local_crops,
        global_crop_ratio=args.global_crop_ratio,
        local_crop_ratio=args.local_crop_ratio,
        mlm_phi=args.mlm_phi,
        mlm_mode=args.mlm_mode,
        mlm_block_size=args.mlm_block_size,
        batch_size=args.batch_size,
        backbone_type=args.backbone_type,
        dwt_wavelet_pool=args.dwt_wavelet_pool,
        soft_threshold_sigma=args.soft_threshold_sigma,
        use_koleo=(args.use_koleo.lower() == "true") if args.use_koleo is not None else None,
        koleo_weight=args.koleo_weight,
        use_vicreg=(args.use_vicreg.lower() == "true") if args.use_vicreg is not None else None,
        vicreg_std_coeff=args.vicreg_std_coeff,
        vicreg_cov_coeff=args.vicreg_cov_coeff,
        synthetic_data_dir=args.synthetic_data_dir,
        subset_frac=args.subset_frac,
        window_stride=args.window_stride,
        checkpoints=[int(c) if c.isdigit() else c for c in args.checkpoints] if args.checkpoints else None,
        seed=args.seed,
        pretrain_cls_model=args.pretrain_cls_model.lower() == "true",
        linear_probe=args.linear_probe.lower() == "true",
        head_type=args.head)

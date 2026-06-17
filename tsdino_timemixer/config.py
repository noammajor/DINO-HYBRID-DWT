config = {

    # ── Task ─────────────────────────────────────────────────────────────────
    # "dino"  |  "classification"  |  "forecasting"
    "task": "dino",
    "seed": 42,
    "output_dir": "./checkpoints",
    "saveckp_freq": 1,
    "test_only": False,

    # ── Datasets ──────────────────────────────────────────────────────────────
    # Names must match keys in dataset_registry.py.
    # forecast_dataset defaults to pretrain_dataset when left as None.
    "pretrain_dataset":  "etth1",
    "forecast_dataset":  "etth1",

    # ── Data ──────────────────────────────────────────────────────────────────
    # Paths are relative to the TSDINOALT 4/ directory
    "data_path": "data/ETTh1.csv",
    "data_path_forecast_training": "data/ETTh1.csv",
    "data_path_forecast_test": "data/ETTh1.csv",
    "data_path_classification": "UCI HAR Dataset",
    "num_workers": 6,
    "batch_size_per_gpu": 128,

    # ── Model architecture ────────────────────────────────────────────────────
    "c_in": 7,          # number of input variables  (9 for UCI HAR)
    "patch_len": 16,
    "step_size": 16,    # stride between patches within window; window=(21-1)*16+16=336
    "window_step": 336, # stride between windows; =window_size for non-overlapping
    "num_patches": 21,  # window length in patches → 21×16 = 336 timesteps
    "n_layers": 4,
    "n_heads": 16,
    "embed_dim": 128,
    "d_ff": 512,
    "dropout": 0.1,
    "head_dropout": 0.1,
    "head_dropout_forecasting": 0.2,
    "drop_path_rate": 0.1,

    # ── DINO head ─────────────────────────────────────────────────────────────
    "out_dim": 1024,
    "use_bn_in_head": False,
    "norm_last_layer": True,

    # ── DINO loss / teacher temperatures ─────────────────────────────────────
    # Warm up teacher temp from soft→sharp: softer early (0.06) avoids the
    # uniform collapse seen on the large synthetic set, then sharpens to 0.04 to
    # keep the learning signal (flat 0.07 was too soft and killed learning).
    "warmup_teacher_temp": 0.06,   # softer start (anti-collapse)
    "teacher_temp": 0.04,          # sharpen to 0.04 after warmup
    "warmup_teacher_temp_epochs": 5,

    # ── Anti-collapse regularizers on the global student feature (optional) ────
    # Both default OFF. Applied to the pre-head global DINO embedding, so they
    # work for dino-only, dino+mae and dino+ibot alike. Zero overhead when off.
    "use_koleo":        False,   # KoLeo entropic regularizer (DINOv2)
    "koleo_weight":     0.1,     # scalar weight when use_koleo=True
    "use_vicreg":       False,   # VICReg variance+covariance regularizer (invariance handled by DINO)
    "vicreg_std_coeff": 1.0,     # variance (hinge) term weight when use_vicreg=True
    "vicreg_cov_coeff": 0.04,    # covariance term weight when use_vicreg=True

    # ── EMA teacher ───────────────────────────────────────────────────────────
    "momentum_teacher": 0.9996,     # base EMA (cosine-scheduled up to 1.0)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    "optimizer": "adamw",           # "adamw" | "sgd"
    "lr": 1e-4,                     # halved (peak ~5e-5 after batch scaling) — collapse fired exactly when LR hit its peak
    "min_lr": 1e-6,
    "warmup_epochs": 3,
    "weight_decay": 0.04,
    "weight_decay_end": 0.1,
    "clip_grad": 3.0,
    "use_fp16": False,
    "freeze_last_layer": 1,

    # ── DINO pretraining ──────────────────────────────────────────────────────
    "epochs": 150,

    # ── DWT defaults (shared across all dwt_* aug types) ─────────────────────
    #
    # Things to experiment with:
    #   wavelet family  – "haar" (sharpest transitions), "db4" (smooth, 4 vanishing moments),
    #                     "db8" (smoother), "sym4"/"sym8" (near-symmetric), "coif2" (compact)
    #   level           – how many frequency bands to create.
    #                     higher = coarser decomposition, more bands to manipulate.
    #                     rule of thumb: level ≤ log2(seq_len) - 1
    #   soft_threshold_sigma  – fraction of max(|coeff|) used as threshold per level.
    #                           0.1 = light denoising, 0.5 = aggressive denoising.
    #   zero_out_ratio        – fraction of finest-level coeffs to drop (0.0–1.0).
    #   finest_levels         – how many of the finest detail levels to perturb (1 = only finest).
    #   high_perturb_noise_range – (min_σ, max_σ) of Gaussian noise added to all detail coeffs.
    #
    "dwt_wavelet":                  "sym4",         # fallback when dwt_wavelet_pool is None
    # Random wavelet pool: one wavelet is drawn per sample. Set to None to use dwt_wavelet deterministically.
    # sym* = near-linear phase (less distortion); db* = slight phase distortion (harder student view).
    # Pools to try:  sym-only ['sym4','sym6','sym8'] | mixed ['sym4','sym6','db4','db6'] | None (fixed)
    "dwt_wavelet_pool":             ['sym4', 'sym6', 'sym8', 'db4', 'db6'],  # random per sample: sym* (low phase distortion) + db* (higher phase distortion)
    "dwt_level":                    3,              # try: 2, 3, 4
    "dwt_soft_threshold_sigma":     0.6,            # bumped: 0.3 → 0.6 (more aggressive teacher smoothing)
    "dwt_zero_out_ratio":           0.4,
    "dwt_finest_levels":            3,              # bumped: 2 → 3 (student perturbs three finest bands)
    "dwt_high_perturb_noise_range": (0.2, 0.5),  # gentler student noise (TSDiNO1 working value) — testing if (0.25,0.50) was the collapse driver
    "dwt_band_scale_approx_range":  (0.80, 1.20),  # wider: (0.9,1.1) → (0.80,1.20)
    "dwt_band_scale_detail_range":  (0.40, 1.60),  # wider: (0.6,1.4) → (0.40,1.60)

    # ── Non-DWT transform defaults ────────────────────────────────────────────
    # These are global fallbacks; override per-crop with the same key in the spec.
    "lorentz_v_range":              (0.2,  0.6),   # relativistic velocity fraction
    "polar_warp_range":             (0.7,  1.3),
    "galilien_a_range":             (0.8,  1.2),
    "rotation_angle_range":         (0.0,  0.3927),  # 0 – π/8 radians
    "boost_b_range":                (0.01, 0.3),
    "hyperbolic_warp_range":        (0.5,  1.5),
    "hyperbolic_shift_magnitude":   0.3,

    # ── Augmentation views ────────────────────────────────────────────────────
    #
    # global_crops → seen by BOTH student and teacher (teacher only processes these).
    #                Keep crop_ratio = 1.0 for stable teacher targets.
    #
    # local_crops  → seen by the student only.
    #                Use crop_ratio < 1.0 to give the student shorter sub-windows
    #                (analogous to small crops in image DINO).
    #
    # Each entry is a dict:
    #   "type"       str | list[str]  – if list, one is drawn at random per sample
    #   "crop_ratio" float            – fraction of seq_len to keep (1.0 = no crop)
    #   Per-crop param overrides are also accepted (e.g. "v_range", "zero_out_ratio").
    #
    # ── Available aug types ───────────────────────────────────────────────────
    #
    #  Convenience aliases (works for all three prefixes):
    #   "<prefix>_soft" → soft_threshold  (teacher-stable global view)
    #   "<prefix>_hard" → high_perturb    (student-aggressive local view)
    #
    #  DWT types (prefix "dwt_"):
    #   "dwt_soft" / "dwt_soft_threshold"  — soft-threshold detail coeffs; clean global view.
    #   "dwt_hard" / "dwt_high_perturb"    — Gaussian noise on all detail coeffs.
    #   "dwt_zero_out_detail" — randomly zero finest-level detail coeffs.
    #   "dwt_low_pass"        — zero all detail coeffs (maximally smooth).
    #   "dwt_band_scale"      — randomly scale each frequency band.
    #
    #  SWT types (prefix "swt_") — shift-invariant, no downsampling:
    #   "swt_soft" / "swt_soft_threshold"  — soft-threshold on all-length detail arrays.
    #   "swt_hard" / "swt_high_perturb"    — Gaussian noise on all detail coeffs.
    #   "swt_zero_out_detail" — randomly zero finest-level detail coeffs.
    #   "swt_low_pass"        — zero all detail coeffs (maximally smooth).
    #   "swt_band_scale"      — randomly scale each frequency band.
    #
    #  MODWT types (prefix "modwt_") — energy-normalised SWT, shift-invariant:
    #   "modwt_soft" / "modwt_soft_threshold"  — soft-threshold on normalised detail coeffs.
    #   "modwt_hard" / "modwt_high_perturb"    — Gaussian noise on all detail coeffs.
    #   "modwt_zero_out_detail" — randomly zero finest-level detail coeffs.
    #   "modwt_low_pass"        — zero all detail coeffs (maximally smooth).
    #   "modwt_band_scale"      — randomly scale each frequency band.
    #
    #  Non-DWT types:
    #   "lorentz"          — relativistic Lorentz boost: γ(x − v·t).
    #                        Override: v_range
    #   "polar"            — polar-coordinate warp.       Override: warp_range
    #   "galilien"         — Galilean scaling (x · a).    Override: a_range
    #   "rotation"         — 2-D rotation with time axis. Override: angle_range
    #   "boost"            — additive linear time trend.  Override: b_range
    #   "hyperbolic_warp"  — tanh amplitude warp.         Override: warp_range
    #   "hyperbolic_geom"  — Poincaré-disk Möbius shift.  Override: shift_magnitude
    #
    # ─────────────────────────────────────────────────────────────────────────

    # ── Teacher view (global crop) ────────────────────────────────────────────
    "global_crops": [
        {"type": "dwt_soft_threshold", "crop_ratio": 1.0},
    ], #changed from low_pass to soft_threshold --- low_pass caused teacher collapse on synthetic; re-testing stability on etth1

    # ── Student view (local crop) ─────────────────────────────────────────────
    "local_crops": [
        {"type": "dwt_hard", "crop_ratio": 1.0},
    ],

    # ── Patch reconstruction (MAE-style auxiliary loss) ────────────────────────
    # Student encoder sees masked patches → reconstruction head.
    # Teacher encoder sees full input    → reconstruction head.
    # Loss: MSE between the two reconstructions at masked positions.
    "use_reconstruction": False,   # set True to enable
    "mlm_phi":        0.75,        # phi*DINO + (1-phi)*MLM  (0 = MLM disabled; 0<phi<1 blends both heads)
    "mlm_mode":       "ibot",      # "ibot" = teacher-guided CE | "mae" = MSE vs ground truth
    "ibot_out_dim":   1024,        # iBOT patch head output dim — kept at out_dim for TSMixer (65536 OOMs with timestep tokens)
    "mlm_mask_ratio": 0.4,         # fraction of patches to mask for MLM
    "recon_mask_ratio":   0.4,    # fraction of patches to mask for student
    "recon_loss_weight":  1.0,    # weight of reconstruction loss relative to DINO loss

    # ── Downstream: Forecasting ───────────────────────────────────────────────
    "pred_len": 96,
    "epochs_forecasting": 30,
    "lr_forecasting": 2e-5,
    "min_lr_forecasting": 2e-6,
    "batch_size_forecast": 128,
    "parms_for_training_forecasting": ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'],
    "parms_for_testing_forecasting":  ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'],

    # ── Downstream: Classification ────────────────────────────────────────────
    "n_classes": 6,
    "epochs_classification": 20,
    "lr_classification": 0.001,
    "min_lr_classification": 1e-6,
    "batch_size_classification": 16,
    "seq_len_classification": 128,  # UCI HAR fixed window
    "c_in_classification": 9,       # UCI HAR sensor count

    # checkpoint to load for downstream tasks  (0 = random init)
    "path_num": 0,

    # ── Local overrides (TEMP: remove after local testing) ────────────────────

    # ── Backbone: TimeMixer (TSMixer) ─────────────────────────────────────────
    # This TSDiNO is TimeMixer-only. Multi-scale season/trend mixing.
    # Values match TimeMixer's own config_timemixer.py defaults.
    # NOTE: if tsmixer_down_sampling_layers > 0, all global_crops and local_crops
    # must have crop_ratio=1.0 (the PDM Linear layers are sized for seq_len).
    "tsmixer_d_model":              128,  # embedding dim inside the TSMixer backbone
    "tsmixer_d_ff":                 256,  # feed-forward dim inside each PDM block
    "tsmixer_e_layers":             3,    # number of PastDecomposableMixing blocks
    "tsmixer_down_sampling_layers": 3,    # multi-scale levels (seq_len / 2^k per level)
    "tsmixer_down_sampling_window": 2,    # pooling stride per downsampling step
    "tsmixer_down_sampling_method": "avg",  # "avg" | "max"
    "tsmixer_decomp_method":        "moving_avg",
    "tsmixer_moving_avg":           25,   # moving-average kernel size (must be odd)
    "tsmixer_top_k":                5,    # DFT top-k components (only for decomp_method='dft_decomp')

    # ── Pretraining data source ───────────────────────────────────────────────
    # pretrain_source: "monash" | "synthetic" | "monash+synthetic" | None (in-domain CSV)
    "pretrain_source":    None,

    # ── Distributed ───────────────────────────────────────────────────────────
    "dist_url": "env://",
}

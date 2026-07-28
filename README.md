# WINO-TS: Wavelet-INvariance self-distillation for Time Series

WINO-TS is a self-supervised pre-training method for time-series foundation
models. It adapts the DINO self-distillation recipe to the time domain by
building its two views in the **wavelet domain**: a momentum *teacher* sees an
"easy" (denoised) wavelet view while the *student* sees a "hard" (perturbed)
wavelet view, and the student is trained to match the teacher's distribution
over the unseen view. The result is an encoder whose representation is invariant
to wavelet-basis and detail-coefficient perturbations, which transfers well to
long-horizon forecasting and anomaly detection.

The default WINO-TS model is a **TimeMixer** backbone (`--model dino_timemixer`);
PatchTST / iTransformer / TimesNet / TimeRxL backbones are also available as
`dino_patchtst`, `dino_itransformer`, `dino_timesnet`, `dino_timerxl`.

## Method

- **Wavelet views.** Each input window is transformed with a discrete wavelet
  transform (DWT), its detail coefficients are perturbed, and it is
  reconstructed back to the time domain. The *easy/teacher* view uses gentle
  soft-thresholding (`dwt_soft_threshold`); the *hard/student* view uses stronger
  Gaussian perturbation of the detail bands (`dwt_high_perturb`).
- **Transforms.** DWT (`dwt_*`), stationary WT (`swt_*`), and MODWT (`modwt_*`)
  are all supported.
- **Wavelet pool.** The basis for each view is drawn from a pool
  (`dwt_wavelet_pool`, e.g. `{sym4,sym6,sym8,db4,db6,coif2}`). Sampling is
  controlled by `--wavelet_sampling_mode`:
  `independent` (default — each view draws its own basis), `shared` (one basis
  per sample, both views), or `fixed` (single basis).
- **View pairing.** `--view_pairing` selects the DINO loss pairing:
  `default`, `hard_student`, `same_view`, or `symmetric`.
- **Optional generative head.** `--mlm_phi` blends the DINO loss with a
  masked-reconstruction (iBOT / MAE-style) objective (`0` = pure DINO).

## Downstream evaluation

Pre-trained encoders are evaluated on three tasks, under both **linear probing**
(frozen backbone + trained head) and **end-to-end fine-tuning**:

1. **Long-horizon forecasting** — horizons `{96, 192, 336, 720}`, 336-step input
   context.
2. **Anomaly detection** — reconstruction-based, 100-step windows, threshold at
   the expected anomaly ratio, TSLib point-adjustment + F1.
3. **Classification** — UEA archive.

### Baselines and the Time-Series-Library base

Supervised and self-supervised baselines — TimeSiam, TS2Vec, TimeMixer,
TimeBase, SparseTSF, PatchTST, DLinear, iTransformer, FEDformer, TimesNet,
Autoformer — are run through the [Time-Series-Library](Time-Series-Library-main-2/)
(TSLib), so every method shares the same data loading, splits, normalization,
context length, and metric computation. Differences therefore reflect the
pre-training objective, not the evaluation setup.

## Quickstart

```bash
# 1. install
pip install -r requirements.txt

# 2. point the code at your data (one-time per machine)
$EDITOR data_paths.py     # set forecasting_data_dir, anomaly_data_dir, etc.

# 3. pretrain WINO-TS, then run a downstream task
python Train_and_downstream.py --model dino_timemixer \
    --pretrain_dataset etth1 --forecast_dataset etth1 --pretrain_only true
python Train_and_downstream.py --model dino_timemixer \
    --pretrain_dataset etth1 --forecast_dataset etth1 --task forecast
```

[Train_and_downstream.py](Train_and_downstream.py) is the unified entry point.
`--model` is the only required flag; everything else falls back to the per-model
config. Common overrides: `--encoder_layers`, `--out_dim`, `--lr`,
`--batch_size`, `--epochs`, `--seed`, `--task {pretrain,forecast,classify,anomaly}`,
`--pretrain_only true`.

### Augmentation / ablation flags

| Flag | Meaning |
|------|---------|
| `--aug_global`, `--aug_local` | Crop augmentation type for teacher/student views (e.g. `dwt_soft_threshold` / `dwt_high_perturb`). |
| `--n_global_crops`, `--n_local_crops` | Number of easy / hard crops. |
| `--global_crop_ratio`, `--local_crop_ratio` | Sub-window ratio per view (1.0 = no crop). |
| `--dwt_wavelet_pool` | Wavelet pool (space-separated, e.g. `sym4 sym6 db4`). A single entry ⇒ fixed basis. |
| `--wavelet_sampling_mode` | `independent` \| `shared` \| `fixed`. |
| `--view_pairing` | `default` \| `hard_student` \| `same_view` \| `symmetric`. |
| `--mlm_phi` | DINO/reconstruction blend (`0` = pure DINO). |
| `--ckpt_tag` | Suffix for the checkpoint directory (keeps ablations separate). |

Some forecast-time knobs are environment variables:
`TS_FORECAST_ENC_LR_SCALE` (encoder LR = head LR × scale),
`TS_FORECAST_MULTISCALE=1`, `TS_FORECAST_LR_FIND=1` (LR range test).

## Config

**Change behavior through the CLI flags — that is the intended, primary
interface** (see [Quickstart](#quickstart) and the
[augmentation / ablation flags](#augmentation--ablation-flags) above). Each model
also ships a Python `config = {...}` dict that supplies the *defaults*, loaded by
[Train_and_downstream.py](Train_and_downstream.py) and merged with
[data_paths.py](data_paths.py); **any CLI flag overrides the matching config
value at runtime.** Edit the config dict only for the handful of defaults that
aren't exposed as flags. For reproducibility, prefer passing flags (plus
`--ckpt_tag` to keep runs separate) over hand-editing configs between runs.

| Model (backbone) | Config file |
|------------------|-------------|
| WINO-TS (TimeMixer) | [tsdino_timemixer/config.py](tsdino_timemixer/config.py) |
| WINO-TS (PatchTST)  | [tsdino_patchtst/config.py](tsdino_patchtst/config.py) |
| JEPA                | [JEPA/config_files/config_jepa.py](JEPA/config_files/config_jepa.py) |
| PatchTST (MAE)      | [PatchTST_self_supervised/config_patchtst.py](PatchTST_self_supervised/config_patchtst.py) |

### Shared data paths

[data_paths.py](data_paths.py) holds the absolute locations of the pre-training
corpus and the downstream sets. **Every entry ships as an `"ADD HERE"`
placeholder — set each to the absolute path on your machine before running**
(one-time). What each key points to:

| Key | Contents |
|-----|----------|
| `monash_data_dir` | Monash pre-training corpus |
| `synthetic_data_dir` | synthetic `.arrow` pre-training files |
| `synthetic_mix_data_dir` | smaller curated synthetic mix |
| `forecasting_data_dir` | forecasting CSVs (ETT, weather, electricity, …) |
| `classification_data_dir` | UEA classification datasets |
| `anomaly_data_dir` | anomaly datasets (SMD/MSL/SMAP/PSM/SWaT) |

### Defaults reference ([tsdino_timemixer/config.py](tsdino_timemixer/config.py))

The config keys below hold the defaults; **most have a CLI flag that overrides
them at runtime — prefer the flag** (e.g. `--encoder_layers`, `--out_dim`,
`--lr`, `--epochs`, `--aug_global/--aug_local`, `--dwt_wavelet_pool`,
`--wavelet_sampling_mode`, `--view_pairing`, `--mlm_phi`, `--lr_forecasting`).
Only edit the config for a default with no corresponding flag.

- **Architecture** — `encoder_layers`, `out_dim`, `tsmixer_d_model`, `d_ff`,
  `n_heads`, `patch_len`, `num_patches` (context = `num_patches × patch_len`;
  336 for forecasting, 100 for anomaly).
- **Views / augmentation** — `global_crops`, `local_crops`, `dwt_wavelet`,
  `dwt_wavelet_pool`, `wavelet_sampling_mode`, `view_pairing`, `dwt_level`,
  `dwt_soft_threshold_sigma`, `dwt_high_perturb_noise_range`.
- **Pre-training** — `epochs`, `batch_size_per_gpu`, `lr`, `weight_decay`,
  `momentum_teacher`, teacher temperatures, `mlm_phi`.
- **Forecasting** — `epochs_forecasting`, `lr_forecasting`,
  `batch_size_forecast`.

## Scripts

Helpers live in [scripts/](scripts/). Every runnable script supports `--dry_run`
(print the commands without launching) and pins the GPU with `--gpu` or
`--gpu_override`. Across the sweep scripts, the model registry is:
`dino_timemixer` (WINO-TS), `dino_patchtst`, `dino_ts2vec`, `patchtst`,
`patchtst_random`, `timemixer`.

### Single runs & in-domain

**[run_single.py](scripts/run_single.py)** — pretrain one model on one dataset,
then forecast. This is the closest wrapper around `Train_and_downstream.py` and
exposes every WINO-TS knob.

| Flag | Notes |
|------|-------|
| `--model`, `--dataset`, `--name` | model id, dataset, run name |
| `--layers 8`, `--embed_dim`, `--out_dim` | architecture |
| `--epochs`, `--epochs_forecasting`, `--lr`, `--batch_size`, `--batch_size_forecast`, `--warmup_epochs` | optimization |
| `--aug_global`, `--aug_local`, `--n_global_crops`, `--n_local_crops`, `--global_crop_ratio`, `--local_crop_ratio` | view augmentation |
| `--dwt_level`, `--dwt_wavelet_pool`, `--soft_threshold_sigma` | wavelet params |
| `--mlm_phi`, `--mlm_mode` | DINO / reconstruction blend |
| `--backbone_type`, `--ckpt_tag`, `--seed`, `--gpu 0` | misc |
| `--skip_pretrain`, `--linear_probe {true,false}` | reuse ckpt / probe vs fine-tune |

**[run_indomain.py](scripts/run_indomain.py)** — in-domain pretrain+forecast
sweep over `--models` × `--datasets`. Flags: `--models`, `--datasets`,
`--gpu_override`, `--seed`, `--skip_pretrain`, `--dry_run`.

**[run_ctx_sweep.py](scripts/run_ctx_sweep.py)** — sweep context lengths
(`--ctx_lens 96 192 336 720`) for one `--model`/`--dataset`. Also takes
`--patch_len`, `--layers 4`, `--backbone_type`, `--epochs`, `--lr`,
`--aug_global/--aug_local`, `--mlm_phi/--mlm_mode`, `--ckpt_tag`, `--gpu`.

**[run_aug_family_grid.py](scripts/run_aug_family_grid.py)** — DWT-family ×
objective × dataset grid. `--families` (dwt/swt/modwt pools), `--objectives`
(dino/mae/ibot via `mlm_phi`), `--datasets`, `--encoder_layers`, `--seed`,
`--gpus`, `--root`, `--sequential`, `--skip_pretrain`, `--dry_run`.

### Multi-axis sweeps

**[run_layer_sweep.py](scripts/run_layer_sweep.py)** — pretrain across encoder
depths. `--models`, `--layers 2 4 8 12 24`, `--pretrain_source`,
`--gpu_override`, `--dry_run`.

**[run_layer_forecast.py](scripts/run_layer_forecast.py)** — forecasting across
(model × depth × dataset) with a tournament checkpoint search
(96→top3→192→top2→336→best→720). `--best_only` skips the tournament;
`--linear_probe {true,false}`, `--head {linear,mlp}`, `--layers`, `--datasets`,
`--pred_lens` (with `--best_only`), `--pretrain_source`, `--out_csv`,
`--output_dir`, `--log_tag`, `--gpu_override`, `--dry_run`.

**[run_layer_classification.py](scripts/run_layer_classification.py)** — UEA
classification sweep (model × depth × dataset). `--layers`, `--datasets`,
`--linear_probe {true,false}`, `--head {linear,mlp}`, `--pretrain_source`,
`--out_csv`, `--log_tag`, `--gpu_override`, `--dry_run`. Resumes from the CSV.

**[run_anomaly_sweep.py](scripts/run_anomaly_sweep.py)** — anomaly detection
(model × depth × dataset). Frozen encoder + trained reconstruction decoder by
default. `--layers`, `--datasets SMD MSL SMAP SWaT PSM`, `--linear_probe`,
`--head`, `--anomaly_ratio`, `--out_csv`, `--log_tag`, `--pretrain_source`,
`--gpu_override`, `--dry_run`. Resumes from the CSV.

**[run_finetune_8layers.py](scripts/run_finetune_8layers.py)** — end-to-end
fine-tune of all three tasks (`linear_probe=False`). `--tasks forecast classify
anomaly`, `--encoder_layers`, `--pred_lens`, `--checkpoint`,
`--{forecast,classification,anomaly}_datasets`, `--pretrain_source`,
`--gpu_override`, `--dry_run`.

**[run_seed_analysis.py](scripts/run_seed_analysis.py)** — reproducibility
sweep: pretrain + downstream across `--seeds`. `--phases pretrain forecast
classify anomaly`, `--{forecast,classification}_datasets`, `--skip_pretrain`,
`--head`, `--pretrain_source`, `--gpu_override`, `--dry_run`.

**[pretrain_cls_encoder.py](scripts/pretrain_cls_encoder.py)** — pretrain with a
long (num_patches=72 → cw=1152) context for classification. `--models`,
`--encoder_layers 8`, `--predictor_layers`, `--num_patches 72`, `--patch_size`,
`--pretrain_source`, `--gpu_override`, `--dry_run`.

**[run_monash.py](scripts/run_monash.py)** — pretrain each model on Monash,
saving a checkpoint per epoch. `--models`, `--gpu`.

### Baselines

- **[run_timesiam.py](scripts/run_timesiam.py)** — TimeSiam SSL (pretrain →
  fine-tune forecast). `--datasets`, `--backbone iTransformer`, `--seq_len`,
  `--pretrain_epochs 50`, `--mask_rate 0.25`, `--sampling_range 6`,
  `--lineage_tokens 2`, `--force_pretrain`, `--gpu`.
- **[run_tsmixer_forecast.py](scripts/run_tsmixer_forecast.py)** /
  **[run_tsmixer_zeroshot.py](scripts/run_tsmixer_zeroshot.py)** — forecast /
  zero-shot forecast from a pretrained TSMixer checkpoint. `--datasets`,
  `--pred_lens`, `--lr_forecasting`, `--ckpt_tag`, `--gpu`.
- **[run_timemixer_pretrained_sweep.py](scripts/run_timemixer_pretrained_sweep.py)**
  — supervised TimeMixer forecasting from checkpoint dirs. `--ckpt_dirs`,
  `--datasets`, `--pred_lens`, `--encoder_layers 4`, `--epochs_forecasting`,
  `--checkpoint_name`, `--gpu`.
- **Cross-domain transfer** (all take `--source`, `--target`, `--pred_lens`,
  `--device`): [tslib_xdomain_zeroshot.py](scripts/tslib_xdomain_zeroshot.py) and
  [tslib_xdomain_probe.py](scripts/tslib_xdomain_probe.py) (`--model`,
  `--head_epochs`) for any TSLib model;
  [sparsetsf_xdomain_zeroshot.py](scripts/sparsetsf_xdomain_zeroshot.py),
  [timebase_xdomain_zeroshot.py](scripts/timebase_xdomain_zeroshot.py),
  [patchtst_xdomain_forecast.py](scripts/patchtst_xdomain_forecast.py).

### Synthetic pre-training data

The GP generators in
[scripts/synthetic_data_generation/](scripts/synthetic_data_generation/) produce
GluonTS `.arrow` files: `kernel-synth.py` (univariate) and `LMC_Synth.py`
(multivariate, linear coregionalization). Drop the output into
`synthetic_data_dir` and run with `--pretrain_source synthetic` (or
`monash+synthetic`). [count_dataset_sizes.py](scripts/count_dataset_sizes.py)
tallies exact timestep counts across the corpora.

### Visualization & analysis

- **[visualize_aug_pairs.py](scripts/visualize_aug_pairs.py)** — every DINO
  augmentation family on one signal. `--dataset`, `--var`/`--all_vars`,
  `--seq_len 336`, `--wavelets`, `--level`, `--sigma`,
  `--teacher_mode {low_pass,soft_threshold}`, `--outdir`.
- **[visualize_views.py](scripts/visualize_views.py)** — teacher vs student
  views. `--data_path`/`--synthetic`, `--channel`, `--seq_len`, `--level`,
  `--sigma`, `--noise_lo/--noise_hi`, `--n_student`, `--ckpt`, `--out`.
- **[why_dwt_real.py](scripts/why_dwt_real.py)** / **why_dwt.py** — motivate the
  DWT augmentation on real vs synthetic signals.
- **anomaly_analysis.py**, **anomaly_paper_figure.py**, **anomaly_saliency.py**,
  **visualize_anomaly.py**, **visualize_attn_map.py**, **visualize_attn_signal.py**
  — anomaly and attention figure helpers.

## Acknowledgements

This codebase builds on prior open-source releases: DINO
([arXiv:2104.14294](https://arxiv.org/abs/2104.14294)), TimeMixer, PatchTST
([arXiv:2211.14730](https://arxiv.org/abs/2211.14730)), TS2Vec, TimeSiam, and the
Time-Series-Library.

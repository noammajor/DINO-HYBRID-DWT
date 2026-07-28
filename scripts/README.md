# `scripts/` — experiment drivers & figures

Command-line drivers used to reproduce the paper's experiments. Everything here wraps
`Train_and_downstream.py` (repo root) or a vendored baseline under `models/`, and follows
the same conventions:

- **GPU** is selected with `--gpu N` (sets `CUDA_VISIBLE_DEVICES`; the child sees it as `cuda:0`).
- **Logs** go under `logs/…`; long runs are launched with `nohup … & disown` so they survive the terminal.
- **`--dry_run`** (where available) prints the exact commands without executing them.
- WINO-TS backbones live in `wino/`; baseline repos live in `models/`; auxiliary tasks (JEPA, MAE, NTP) in `aux_tasks/`.

## Quick reference

| Script | Category | What it does |
|---|---|---|
| `run_single.py` | driver | Pretrain + forecast one model on one dataset (fully configurable) |
| `run_indomain.py` | driver | In-domain pretrain+forecast sweep (models × datasets, threaded per GPU) |
| `run_seed_analysis.py` | driver | 5-seed reproducibility sweep across forecast/classify/anomaly |
| `run_timesiam.py` | baseline | TimeSiam SSL baseline (pretrain → fine-tune) at our 336 context |
| `run_tsmixer_forecast.py` | baseline | Fine-tuned forecasting for the iBOT-pretrained TSMixer backbone (5 datasets) |
| `run_tsmixer_zeroshot.py` | baseline | Frozen-backbone linear-probe forecasting sweep (resumable CSV) |
| `run_timemixer_pretrained_sweep.py` | baseline | Supervised TimeMixer initialized from pretrained TSMixer checkpoints |
| `pretrain_cls_encoder.py` | baseline | Pretrain deep (8-layer, cw1152) encoders for UEA classification |
| `anomaly_analysis.py` | anomaly | Fine-tune + evaluate + visualize the WINO-TS anomaly detector on one dataset |
| `patchtst_xdomain_forecast.py` | cross-domain | Supervised PatchTST cross-domain transfer (linear-probe head on target) |
| `tslib_xdomain_probe.py` | cross-domain | Frozen-encoder linear-probe transfer for any TSLib model |
| `tslib_xdomain_zeroshot.py` | cross-domain | Zero-shot transfer for any TSLib model (also a library, imported by others) |
| `sparsetsf_xdomain_zeroshot.py` | cross-domain | Zero-shot cross-domain transfer for SparseTSF |
| `timebase_xdomain_zeroshot.py` | cross-domain | Zero-shot cross-domain transfer for TimeBase |
| `why_dwt.py` | figure | Synthetic didactic figure: DWT preserves identity, crop+jitter destroys it |
| `why_dwt_real.py` | figure | Same argument through the REAL augmentation pipeline / real data |

> Dependency note: `tslib_xdomain_zeroshot.py` is also a **library** — it exports `run_pair(...)`
> and helpers used by `sparsetsf_xdomain_zeroshot.py` and `timebase_xdomain_zeroshot.py`. Don't remove it.
> `patchtst_xdomain_forecast.py`, `tslib_xdomain_probe.py`, and the two zero-shot wrappers all import
> `run_tslib_benchmark.py` (repo root) as `B` for shared data-loader/model-builder helpers.

---

## Drivers

### run_single.py

**Purpose**: Pretrains a single model on one in-domain dataset and then runs in-domain forecasting (pred_lens 96/192/336/720) on that same dataset. It wraps `Train_and_downstream.py`, copies the resulting best checkpoint to a unified path (`Models/{name}/best_chkp_{dataset}.pt`), and is the go-to for a single, fully-configurable experiment. Supervised models (timemixer/autoformer/fedformer/dlinear) skip pretraining and train directly on forecasting.

**Flags**:

| Flag | Type/choices | Default | Description |
|------|--------------|---------|-------------|
| `--model` | dino_timemixer, dino_patchtst, dino_itransformer, patchtst, timemixer, autoformer, fedformer, dlinear (required) | — | Model to run |
| `--dataset` | etth1, etth2, ettm1, ettm2, weather, electricity, traffic, exchange, wind, solar, metr_la, aqwan, aqshunyi, czelan, zafnoo, pm2_5, temp (required) | — | Dataset to pretrain and forecast on |
| `--name` | str (required) | — | Run name; checkpoint saved as `Models/{name}/best_chkp_{dataset}.pt` |
| `--layers` | int | 8 | Number of encoder layers |
| `--embed_dim` | int | None | Embedding dim / d_model override |
| `--out_dim` | int | None | DINO output bins / prototype count (K; DINO only) |
| `--epochs` | int | None | Number of pretraining epochs |
| `--epochs_forecasting` | int | None | Number of forecasting fine-tune epochs (DINO) |
| `--checkpoints` | list (nargs+) | None | Checkpoint epochs to evaluate during forecasting, e.g. `1 3 5 10` |
| `--lr` | float | None (model-specific default) | Pretraining LR |
| `--lr_forecasting` | float | None | Forecast fine-tune LR (head_lr=encoder_lr; DINO) |
| `--batch_size` | int | None | Pretrain batch size override |
| `--warmup_epochs` | int | None | LR warmup epochs (DINO only) |
| `--ckpt_tag` | str | None | Extra tag appended to checkpoint directory name |
| `--aug_global` | str | None | Global (teacher) augmentation type, overrides config |
| `--aug_local` | str | None | Local (student) augmentation type, overrides config |
| `--n_global_crops` | int | None | Number of global (teacher) crops — DINO multi-crop |
| `--n_local_crops` | int | None | Number of local (student) crops — DINO multi-crop |
| `--global_crop_ratio` | float | None | Crop ratio for global crops (1.0 = no crop) |
| `--local_crop_ratio` | float | None | Crop ratio for local crops (e.g. 0.4) |
| `--dwt_level` | int | None | DWT decomposition depth J (e.g. 2/3/4) |
| `--batch_size_forecast` | int | None | Forecast batch size override (e.g. lower for electricity) |
| `--dwt_wavelet_pool` | list (nargs+) | None | DWT wavelet pool, e.g. `sym4 sym6 sym8 db4 db6 coif2` ('full' family = the 6 listed) |
| `--soft_threshold_sigma` | float | None (config 0.6) | ρ (shrinkage ratio) for soft-threshold DWT/SWT/MODWT aug |
| `--mlm_phi` | float | None | MLM weight: phi*DINO+(1-phi)*MLM (DINO only) |
| `--mlm_mode` | str | None | MLM variant: `ibot` (teacher-guided CE) or `mae` (MSE vs ground truth) |
| `--backbone_type` | str | None | `patchtst` \| `tsmixer` — overrides config |
| `--gpu` | int | 0 | GPU index via CUDA_VISIBLE_DEVICES |
| `--seed` | int | None | Random seed (suffixes checkpoint dir with `_seedN`) |
| `--skip_pretrain` | flag | False | Skip pretraining, go straight to forecasting (reuse existing checkpoint) |
| `--linear_probe` | true, false | None | DINO forecast mode: true=probe (backbone frozen), false=full fine-tune |
| `--dry_run` | flag | False | Print commands without executing |

**Example**:
```bash
python scripts/run_single.py --model dino_timemixer --dataset etth1 --name my_run
# reuse a pretrained backbone, full fine-tune, custom forecast LR:
python scripts/run_single.py --model dino_timemixer --dataset weather --name w_ft \
    --skip_pretrain --linear_probe false --layers 4 --out_dim 1024 \
    --ckpt_tag aug_family_grid_full_dino --seed 42 --lr_forecasting 1e-4 --gpu 0
```

**Outputs / notes**:
- Unified checkpoint → `Models/{name}/best_chkp_{dataset}.pt` (non-supervised models only). Source located by `_find_src_checkpoint`: DINO reads `checkpoints[_patchtst|_ts2vec]_{dataset}_layers{L}[_outdim{K}][_{tag}]/checkpoint_best.pth`; plain `patchtst` reads `models/PatchTST_self_supervised/saved_models/{dataset}/masked_patchtst/based_model/layers{L}/*.pth`.
- Logs → `logs/single/{name}/pretrain.log` and `logs/single/{name}/forecast.log`.
- Runs synchronously; aborts if pretrain or forecast returns non-zero. Default LRs: dino_* = 5e-4, patchtst = 5e-5, timemixer/autoformer/fedformer = 1e-4, dlinear = 5e-3.
- `SUPERVISED_MODELS` = {timemixer, autoformer, fedformer, dlinear}: pretrain skipped, no checkpoint copy, trained directly.

### run_indomain.py

**Purpose**: In-domain pretrain + forecast sweep over each selected model × dataset. Each model runs in its own thread on its own GPU; datasets are sequential within a thread. Reproduces the full in-domain benchmark in one command.

**Flags**:

| Flag | Type/choices | Default | Description |
|------|--------------|---------|-------------|
| `--models` | dino_timemixer, dino_patchtst, dino_ts2vec, patchtst (nargs+) | all four | Models to run |
| `--datasets` | etth1, etth2, ettm1, ettm2, weather (nargs+) | all five | Datasets to run |
| `--gpu_override` | int | None | Run all models on this GPU (overrides per-model map) |
| `--seed` | int | None | Random seed (suffixes checkpoint paths with `_seedN`) |
| `--skip_pretrain` | flag | False | Use existing checkpoints |
| `--dry_run` | flag | False | Print commands only |

**Example**:
```bash
python scripts/run_indomain.py --models dino_timemixer patchtst --datasets etth1 ettm2 --seed 42
```

**Outputs / notes**: Logs → `logs/indomain/{dataset}/{pretrain,forecast}_{model}.log`. Fixed `ENCODER_LAYERS=8`, predictor_layers=4, forecast head `linear`. GPU map: dino_timemixer=0, dino_patchtst=1, dino_ts2vec=2, patchtst=3. LRs: dino_*=5e-4, patchtst=5e-5. No unified checkpoint copy.

### run_seed_analysis.py

**Purpose**: Reproducibility sweep across 5 seeds (default `[2003,123,456,789,1337]`). For each seed, every model is pretrained (8 layers) then evaluated on forecasting, classification, and anomaly; checkpoints saved with `_seedN`. Regenerates the seed-analysis CSVs and measures variance.

**Flags**:

| Flag | Type/choices | Default | Description |
|------|--------------|---------|-------------|
| `--seeds` | int (nargs+) | 2003 123 456 789 1337 | Seeds to sweep |
| `--models` | dino_timemixer, dino_patchtst, dino_ts2vec, patchtst (nargs+) | all four | Models |
| `--pretrain_source` | monash, synthetic, monash+synthetic | monash | Pretrain data source |
| `--skip_pretrain` | flag | False | Use existing checkpoints |
| `--phases` | pretrain, forecast, classify, anomaly (nargs+) | all four | Run only these phases |
| `--forecast_datasets` | etth1…traffic (nargs+) | all | Forecast datasets |
| `--classification_datasets` | 10 UEA sets (nargs+) | all | Classification datasets |
| `--anomaly_datasets` | SMD, MSL, SMAP, SWaT, PSM (nargs+) | all | Anomaly datasets |
| `--gpu_override` | int | None | Run all models on this GPU |
| `--dry_run` | flag | False | Print commands only |
| `--head` | linear, mlp | linear | Downstream head type |

**Example**:
```bash
python scripts/run_seed_analysis.py --phases forecast --forecast_datasets electricity traffic \
    --pretrain_source synthetic --gpu_override 3
```

**Outputs / notes**: CSVs → `results/seed_analysis_{forecast,classification,anomaly}.csv`. Logs → `logs/forecasting/seed_analysis/seed{S}/…`. Two pretrains per model/seed (standard + classification-specific `--num_patches 72` = cw1152). Anomaly datasets launch in parallel; forecast/classify run sequentially; seeds sequential.

---

## Baselines

### run_timesiam.py

**Purpose**: TimeSiam SSL baseline (pretrain → fine-tune forecasting) on **our** datasets at **our** 336 context, so its numbers are directly comparable to WINO-TS. Points TimeSiam's TSLib loaders at our CSV dir and drives the two-stage workflow (pretrain, then fine-tune per horizon) per dataset.

**Flags**:

| Flag | Type/choices | Default | Description |
|------|--------------|---------|-------------|
| `--datasets` | names or `all` (nargs+) | etth1 etth2 ettm1 ettm2 weather | Datasets (from dataset_registry) |
| `--backbone` | str | iTransformer | TimeSiam encoder (iTransformer, PatchTST, DLinear, …) |
| `--seq_len` | int | None (→336) | Context length |
| `--pretrain_epochs` | int | 50 | Pretrain epochs (paper default) |
| `--mask_rate` | float | 0.25 | Mask rate |
| `--sampling_range` | int | 6 | Sampling range |
| `--lineage_tokens` | int | 2 | Lineage tokens |
| `--gpu` | int | 0 | GPU index |
| `--force_pretrain` | flag | False | Re-run pretrain even if a checkpoint exists |
| `--dry_run` | flag | False | Print commands only |

**Example**:
```bash
python scripts/run_timesiam.py --datasets all --backbone iTransformer --gpu 0
```

**Outputs / notes**: Runs `models/TimeSiam-main/run.py` (cwd). Pretrain ckpts → `models/TimeSiam-main/outputs/pretrain_checkpoints/{KEY}/ckpt_best.pth` (skip if exists unless `--force_pretrain`). Logs → `logs/timesiam/{dataset}/{pretrain.log, forecast_pl{96,192,336,720}.log}`. GPU isolated as cuda:0 so `run.py` is always passed `--gpu 0`. Per-dataset PatchTST hyperparams from `PATCHTST_PAPER` (paper-verbatim for ETT + weather/electricity/exchange/traffic; others fall back to `_default`, flagged at runtime).

### run_tsmixer_forecast.py

**Purpose**: Fine-tuned forecasting across the 5 ETT/weather datasets using the iBOT-pretrained TSMixer backbone. Calls `Train_and_downstream.run()` in-process, looping datasets on one GPU, with a fixed config (`dino_timemixer`, tsmixer, 4 layers, out_dim 8192, `tsmixer_ibot` tag).

**Flags**:

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--gpu` | int | 0 | GPU index |
| `--epochs_forecasting` | int | None | Forecast fine-tune epochs (None = config default) |

**Example**: `python scripts/run_tsmixer_forecast.py --gpu 7 --epochs_forecasting 20`

**Outputs / notes**: Logs → `logs/Dino_TSMIXER_Regular_synthetic/{dataset}.log` (stdout redirected per dataset). Locates checkpoint via the `_synthetic_layers4_outdim8192_tsmixer` suffix. Per-dataset exceptions are caught and the loop continues.

### run_tsmixer_zeroshot.py

**Purpose**: Zero-shot forecasting — loads the pretrained TSMixer backbone, freezes it, trains only a linear-probe head per `pred_len`. Sweeps datasets × pred_lens, records MSE per cell, resumable (skips cells already in the CSV).

**Flags**:

| Flag | Type/choices | Default | Description |
|------|--------------|---------|-------------|
| `--gpu` | int | 0 | GPU index |
| `--datasets` | str (nargs+) | etth1 etth2 ettm1 ettm2 weather | Datasets |
| `--pred_lens` | int (nargs+) | 96 192 336 720 | Horizons (one `run()` per horizon) |
| `--lr_forecasting` | float | None | Linear-probe head LR |
| `--ckpt_tag` | tsmixer \| tsmixer_ibot \| tsmixer_mae | tsmixer | Pretrained variant; also sets log/result paths |

**Example**: `python scripts/run_tsmixer_zeroshot.py --gpu 3 --ckpt_tag tsmixer_ibot --datasets etth1 weather --pred_lens 96 336`

**Outputs / notes**: CSV → `results/tsmixer_zeroshot_{tag}.csv` (`dataset,pred_len,mse,timestamp`; incremental/resumable). Logs → `logs/tsmixer_zeroshot_{tag}/{dataset}/pred{H}.log` (tee'd to console). Loads `checkpoints_synthetic_layers4_outdim8192_{tag}/checkpoint_best.pth`. Only `result[1]` (MSE) recorded.

### run_timemixer_pretrained_sweep.py

**Purpose**: Supervised TimeMixer forecasting where the encoder is initialized from a pretrained TSMixer DINO checkpoint. For each checkpoint dir it finds the best checkpoint, injects encoder weights into a `timemixer` model, and runs supervised forecasting over all datasets (all horizons together). Sweeps multiple backbones (plain/iBOT/MAE) to compare fine-tuned performance.

**Flags**:

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--gpu` | int | 0 | GPU index |
| `--ckpt_dirs` | str (nargs+) | tsmixer, tsmixer_ibot, tsmixer_mae dirs | Checkpoint dirs (relative to root) |
| `--datasets` | str (nargs+) | etth1…weather | Datasets |
| `--pred_lens` | int (nargs+) | 96 192 336 720 | Horizons (all in one `run()` per dataset) |
| `--encoder_layers` | int | 4 | Encoder depth injected into TimeMixer |
| `--epochs_forecasting` | int | None | Supervised forecast epochs |
| `--checkpoint_name` | str | None | Specific checkpoint file (else best, else highest-numbered) |

**Example**:
```bash
python scripts/run_timemixer_pretrained_sweep.py --gpu 0 \
    --ckpt_dirs checkpoints_synthetic_layers4_outdim8192_tsmixer_ibot --checkpoint_name checkpoint20.pth
```

**Outputs / notes**: CSV → `results/timemixer_pretrained_sweep.csv` (`ckpt,dataset,pred_len,mse,mae,timestamp`; `pred_len` written as `"all"`; incremental). Logs → `logs/timemixer_pretrained/{ckpt_dir}/{dataset}.log`. `_find_best_checkpoint`: explicit name → `checkpoint_best.pth` → highest-numbered; dir skipped if none.

### pretrain_cls_encoder.py

**Purpose**: Pretrains deep (default 8-layer) encoders with a long 1152-timestep context (`num_patches=72 × patch_size=16`) for downstream UEA classification. Spawns one subprocess per model (`Train_and_downstream.py --pretrain_only true`), each pinned to its own GPU, and waits for all. Produces classification-oriented checkpoints.

**Flags**:

| Flag | Type/choices | Default | Description |
|------|--------------|---------|-------------|
| `--models` | dino_timemixer, dino_patchtst, dino_ts2vec, patchtst (nargs+) | all 4 | Models to pretrain |
| `--pretrain_source` | monash, synthetic, monash+synthetic | synthetic | Pretrain data source |
| `--encoder_layers` | int | 8 | Encoder depth |
| `--predictor_layers` | int | 4 | Predictor depth |
| `--num_patches` | int | 72 | Patches in context (× patch_size = 1152) |
| `--patch_size` | int | 16 | Log-name/display only (actual size from config) |
| `--gpu_override` | int | None | Run all on one GPU (overrides per-model map) |
| `--ckpt_tag` | str | None | Suffix on checkpoint dir (dino* only) |
| `--log_tag` | str | "" | Extra suffix on log filename |
| `--mlm_phi` | float | None | MLM/iBOT weight (0.0 = pure DINO; dino* only) |
| `--subset_frac` | float | None | Fraction of pretraining data |
| `--out_dim` | int | None | DINO head output dim (dino* only) |
| `--mlm_mode` | str | None | `ibot` or `mae` (dino* only) |
| `--dry_run` | flag | False | Print commands only |

**Example**: `python scripts/pretrain_cls_encoder.py --models dino_timemixer --mlm_mode ibot --mlm_phi 1.0 --out_dim 8192`

**Outputs / notes**: Logs → `logs/cls_encoder_pretrain/{model}{_tag}_layers{L}_{source}_cw{cw}{_log_tag}.log`. Per-model LRs: dino_*=5e-4, patchtst=5e-5. GPU map: dino_timemixer=0, dino_patchtst=1, dino_ts2vec=2, patchtst=3. Conditional flags (`--ckpt_tag/--mlm_phi/--mlm_mode/--out_dim`) only for dino* models. Waits on all subprocesses, prints per-model OK/FAILED.

---

## Anomaly

### anomaly_analysis.py

**Purpose**: Fine-tunes the WINO-TS DINO anomaly detector (TSMixer teacher backbone + linear reconstruction decoder) on one anomaly dataset, then saves the model, dumps per-timestep metrics to `.npz`, and renders analysis figures. Loads a DINO teacher from an anomaly-pretrain checkpoint, fine-tunes encoder+decoder on the normal stream, computes reconstruction-error scores with point-adjustment (P/R/F1), and paints signal/attention/score visualizations.

**Flags**:

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--dataset` | str (required) | — | Anomaly dataset (e.g. `SMD`) |
| `--ckpt` | str (required) | — | Anomaly-pretrain teacher checkpoint (backbone weights) |
| `--patch_len` | int | 10 | Patch length (`win = num_patches × patch_len`) |
| `--num_patches` | int | 10 | Patches per window |
| `--epochs` | int | 10 | Anomaly fine-tune epochs |
| `--anomaly_ratio` | float | 0.5 | Threshold = `(100 − ratio)`-percentile of combined train+test scores |
| `--var` | int | None | Channel to render (auto-picks highest-std over anomalies if unset) |
| `--gpu` | int | 0 | GPU index (CPU fallback) |
| `--outdir` | str | `<ROOT>/vis/anomaly` | Metrics `.npz` + PNG figures |
| `--save_dir` | str | `<ROOT>/checkpoints/anomaly_finetuned` | Fine-tuned model checkpoint |

**Example**:
```bash
python scripts/anomaly_analysis.py --dataset SMD \
    --ckpt anomaly/checkpoints_layers4_outdim1024_tsmixer_seed42_anompre_SMD_cw100/checkpoint_best.pth \
    --epochs 10 --gpu 0
```

**Outputs / notes**: Model → `{save_dir}/{dataset}_finetuned.pth`. Metrics → `{outdir}/anomaly_metrics_{dataset}.npz`. Figures → `{outdir}/anomaly_{paint,painted_score,heatmap,hist}_{dataset}*.png`. Imports `TSMixerForDINO` from `wino/tsdino_timemixer`; reads its `config.py`; uses `AnomalyDataPuller`; captures attention via a hook on `bb.global_attn`. Point-adjusted binary P/R/F1.

---

## Cross-domain transfer

All four import `run_tslib_benchmark.py` (root) as `B` for shared loaders/model builders. Results are **printed to stdout** (redirect to a log); no files written.

### tslib_xdomain_zeroshot.py  *(also a library)*

**Purpose**: Supervised **zero-shot** cross-domain transfer for any TSLib model. Trains the full model on SOURCE (early-stopped on source val), then infers on the TARGET test set with **no target adaptation**. Architecture-agnostic zero-shot baseline. **Also imported by the SparseTSF/TimeBase wrappers** — exports `run_pair(model, source, target, device, pred_lens)` plus `_base_args/_loader/_make_forward/_train/_evaluate` and `_CROSS_C_MODELS`.

**Flags**: `--model` (required, TSLib name), `--source` (required), `--target` (required), `--device` (`auto`), `--pred_lens` (`96 192 336 720`).

**Example**: `python scripts/xdomain/tslib_xdomain_zeroshot.py --model DLinear --source etth1 --target etth2 --device cuda:0`

**Notes**: Cross-C transfer only for `_CROSS_C_MODELS = {PatchTST, DLinear, SparseTSF, TimeMixer, iTransformer}` — differing `c_in` with an unsupported model is `[skip]`ped; for supported ones the model is rebuilt at target-C and shape-matched weights copied (RevIN affine re-init). TimeMixer gets `down_sampling_layers=3` injected to avoid an IndexError.

### tslib_xdomain_probe.py

**Purpose**: Supervised **cross-domain transfer via frozen source encoder + linear-probe head** for TSLib models with a clean encoder/head split (PatchTST, iTransformer, TimeMixer). Phase 1 trains full model on SOURCE; Phase 2 freezes encoder, re-inits+trains only the head on TARGET; reports MSE/MAE on TARGET test.

**Flags**: `--model` (PatchTST\|TimeMixer\|iTransformer, required), `--source` (required), `--target` (required), `--device` (`auto`), `--pred_lens` (`96 192 336 720`), `--head_epochs` (`20`).

**Example**: `python scripts/xdomain/tslib_xdomain_probe.py --model iTransformer --source etth1 --target etth2 --device cuda:0`

**Notes**: Head modules per model in `HEAD_PREFIXES` (PatchTST→`head`, iTransformer→`projection`, TimeMixer→`predict_layers/projection_layer/out_res_layers/regression_layers`). **No cross-C handling** — assumes compatible variable counts (ETT are all 7-var).

### patchtst_xdomain_forecast.py

**Purpose**: Supervised PatchTST cross-domain transfer: train full PatchTST on SOURCE, freeze patch-embed+encoder, re-init the linear head, train ONLY the head on TARGET (linear probe), report MSE/MAE on TARGET test. The supervised analog of the DINO frozen-backbone transfer.

**Flags**: `--source` (required), `--target` (required), `--device` (`auto`), `--pred_lens` (`96 192 336 720`), `--head_epochs` (`20`).

**Example**: `python scripts/xdomain/patchtst_xdomain_forecast.py --source etth1 --target etth2 --device cuda:0`

### sparsetsf_xdomain_zeroshot.py

**Purpose**: Zero-shot cross-domain transfer for **SparseTSF** (not a TSLib model — built/trained directly, reusing the benchmark loaders). Train on SOURCE, infer on TARGET test, no adaptation.

**Flags**: `--source` (etth1…weather, required), `--target` (etth1…weather, required), `--device` (`auto`), `--pred_lens` (`96 192 336 720`).

**Example**: `python scripts/xdomain/sparsetsf_xdomain_zeroshot.py --source etth1 --target etth2 --device cuda:0`

**Notes**: Loads `models/SparseTSF-main/models/SparseTSF.py` by path. Fixed knobs: SEQ_LEN 336, PATCH_LEN 16, D_MODEL 128, EPOCHS 30, BATCH 256; per-source `period_len`/`lr` in `SP_CFG` (`period_len` must divide 336 and every pred_len). Channel-agnostic → cross-C rebuilds + copies shape-matched weights.

### timebase_xdomain_zeroshot.py

**Purpose**: Zero-shot cross-domain transfer for **TimeBase** (built/trained directly, with an orthogonality-regularized loss). Train on SOURCE, infer on TARGET test, no adaptation.

**Flags**: `--source` (etth1…weather, required), `--target` (etth1…weather, required), `--device` (`auto`), `--pred_lens` (`96 192 336 720`).

**Example**: `python scripts/xdomain/timebase_xdomain_zeroshot.py --source etth1 --target etth2 --device cuda:0`

**Notes**: Loads `models/TimeBase-main/models/TimeBase.py` by path. Fixed: SEQ_LEN 336, ORTHO_WEIGHT 0.2, LR 1e-2, EPOCHS 10, BATCH 64; per-source `period_len`/`basis_num` in `TB_CFG`. Loss = `MSE + 0.2·ortho`. `individual=0` → basis shared across channels, so cross-C rebuilds + copies shape-matched weights.

---

## Figures

### why_dwt.py

**Purpose**: Standalone (no argparse) didactic figure: a fully synthetic argument that the DWT view-pair preserves a signal's "identity" (a localized Gabor burst) while a DINO-vision crop+jitter pair destroys it. Independent of real data or models.

**No flags** — hardcoded constants at the top: `N=512`, burst `T0=0.40, F0=20, W=0.035`, `WAV="db4", LEVEL=5`, `crop_resize(x, 0.40, 0.52)`, jitter std 0.10, RNG seed 1.

**Example**: `python scripts/why_dwt.py`

**Outputs / notes**: Writes two PNGs to **absolute hardcoded paths** `…/vis/why_dwt_time.png` and `…/vis/why_dwt_tf.png` (edit the paths for another machine; the `vis/` dir is recreated). Deps: `numpy`, `pywt`, `matplotlib` (Agg) — no repo-internal deps.

### why_dwt_real.py

**Purpose**: Same DWT-vs-crop+jitter argument but through the **REAL** augmentation pipeline (the actual `DWTAugmentation`/`gaussian_noise`/`jitter_contrast` classes + real config params) on synthetic or real dataset windows. Validates the augmentation design on real code paths.

**Flags**:

| Flag | Type/choices | Default | Description |
|------|--------------|---------|-------------|
| `--source` | synthetic, data | synthetic | Signal source |
| `--period` | int | 48 | Synthetic carrier period (synthetic only) |
| `--crop_ratio` | float | 0.3 | Vision crop ratio (smaller = more time-warp) |
| `--dwt_hard_scale` | float | 1.6 | DWT student detail-band perturbation multiplier |
| `--vision_mode` | contrast, gauss | contrast | Vision distortion type |
| `--dataset` | str | etth1 | Dataset key (when `--source data`) |
| `--csv` | str | None | Explicit CSV (overrides `--dataset`) |
| `--idx` | int | 500 | Window index for the single-window figure |
| `--var` | int | None | Channel to plot (auto-selects most periodic) |
| `--nwin` | int | 200 | Windows for the statistics figure |
| `--gpu` | int | 0 | CUDA_VISIBLE_DEVICES |
| `--outdir` | str | `<ROOT>/vis` | Output dir |

**Example**:
```bash
python scripts/why_dwt_real.py --source data --dataset etth1 --gpu 0
```

**Outputs / notes**: Writes `why_dwt_real_signal.png` and `why_dwt_real_stats.png` to `--outdir`. Loads `wino/tsdino_timemixer/config.py` dynamically; imports the real `data_agumentation` classes + `PatchTSTPretrainAdapter`. `--source data` is meant to run on the server where the data lives.

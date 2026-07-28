#!/bin/bash
# Timer-XL in-domain DINO+MAE grid: 3 DWT families × the given datasets.
# Small config: 4 layers, d_model 128, d_ff 512, 8 heads, out_dim 1024, 80 epochs.
#
# Usage:   ./scripts/run_timerxl_indomain_grid.sh <GPU> <dataset> [dataset ...]
# Example: ./scripts/run_timerxl_indomain_grid.sh 3 etth1 etth2 ettm1
#          (launch one per free GPU with a slice of the datasets)
#
# Datasets (registry keys): etth1 etth2 ettm1 ettm2 weather electricity traffic
set -u
GPU=${1:?usage: run_timerxl_indomain_grid.sh <GPU> <dataset>...}; shift
DATASETS="$@"
R="$(cd "$(dirname "$0")/.." && pwd)"   # repo root (this script lives in scripts/)
cd "$R"

COMMON="--model dino_timerxl --pretrain_only true \
  --encoder_layers 4 --seed 42 --epochs 80 \
  --mlm_phi 0.6 --mlm_mode mae \
  --embed_dim 128 --d_ff 512 --n_heads 8 --out_dim 1024"

# DWT families (in-domain timerxl): full uses sym/db pool (no coif2),
# db uses db-only pool, zero uses db6 + zero-out-detail local aug.
fam_flags () {
  case "$1" in
    full) echo "--dwt_wavelet_pool sym4 sym6 sym8 db4 db6" ;;
    db)   echo "--dwt_wavelet_pool db4 db6 db8" ;;
    zero) echo "--dwt_wavelet_pool db6 --aug_local dwt_zero_out_detail" ;;
  esac
}

# Channel-independent backbone folds C into the batch, so high-channel datasets
# must use a smaller batch (effective batch ≈ batch_size × c_in). ETT (7ch) use
# the config default. Tune if you hit OOM / want to fill the GPU more.
batch_flags () {
  case "$1" in
    traffic)     echo "--batch_size 128" ;;  # 862 ch
    electricity) echo "--batch_size 128" ;;  # 321 ch
    weather)     echo "--batch_size 128" ;;  # 21 ch
    *)           echo "" ;;                   # ETT* (7 ch) → config default (128)
  esac
}

for D in $DATASETS; do
  for F in full db zero; do
    OUT="logs/AAAI/timerxl_indomain/$D"; mkdir -p "$OUT"
    echo "===== $D / $F (GPU $GPU) ====="
    CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python Train_and_downstream.py $COMMON $(batch_flags "$D") \
      --pretrain_dataset "$D" --forecast_dataset "$D" \
      $(fam_flags "$F") --ckpt_tag timerxl_grid_${F}_mae \
      > "$OUT/${F}_mae.log" 2>&1
    echo "  done -> $OUT/${F}_mae.log"
  done
done

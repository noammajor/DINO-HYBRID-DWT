#!/usr/bin/env bash
# Downstream forecast at all horizons {96,192,336,720} reusing the already-
# pretrained MAE / NTP encoders (no re-pretraining). For each flow x dataset it
# loads checkpoint_best.pth and runs forecast.py once per pred_len.
#
# Usage:  MAE_GPU=0 NTP_GPU=1 ./run_forecast_lengths.sh

set -u
REPO="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${DATA_DIR:-$(cd "$REPO" && python -c "from data_paths import DATA_PATHS; print(DATA_PATHS['forecasting_data_dir'])")}"

SEQ_LEN=336
PRED_LENS="96 192 336 720"
MODE="${MODE:-linear_probe}"
FC_EPOCHS=30
# must match the pretrained encoder
D_MODEL=128; D_FF=256; E_LAYERS=4; DS_LAYERS=3; DS_WINDOW=2
MAE_LR=1e-3; NTP_LR=0.01

MAE_GPU="${MAE_GPU:-0}"
NTP_GPU="${NTP_GPU:-1}"

# name csv c_in
DATASETS=(
  "etth1   ETTh1.csv   7"
  "etth2   ETTh2.csv   7"
  "ettm1   ETTm1.csv   7"
  "ettm2   ETTm2.csv   7"
  "weather weather.csv 21"
)

run_flow () {   # $1=flow(mae|ntp)  $2=gpu  $3=lr
  local flow=$1 gpu=$2 lr=$3 e name csv cin pl ckpt log
  for e in "${DATASETS[@]}"; do
    read -r name csv cin <<< "$e"
    ckpt="$REPO/timemixer_$flow/checkpoints/$name/checkpoint_best.pth"
    if [ ! -f "$ckpt" ]; then
      echo "[$flow|$name] no checkpoint ($ckpt) — skip"; continue
    fi
    for pl in $PRED_LENS; do
      log="$REPO/logs/timemixer_$flow/$name"; mkdir -p "$log"
      echo "[$flow|gpu$gpu|$name|$pl] forecast ($MODE)"
      CUDA_VISIBLE_DEVICES=$gpu python "$REPO/timemixer_$flow/forecast.py" \
        --init_ckpt "$ckpt" --mode "$MODE" \
        --data_path "$DATA_DIR/$csv" --c_in "$cin" --seq_len $SEQ_LEN --pred_len "$pl" \
        --lr "$lr" --epochs $FC_EPOCHS \
        --d_model $D_MODEL --d_ff $D_FF --e_layers $E_LAYERS \
        --down_sampling_layers $DS_LAYERS --down_sampling_window $DS_WINDOW \
        > "$log/forecast_${pl}.log" 2>&1
    done
  done
}

[ -d "$DATA_DIR" ] || { echo "DATA_DIR not found: $DATA_DIR" >&2; exit 1; }
echo "data=$DATA_DIR  seq=$SEQ_LEN  preds=$PRED_LENS  mode=$MODE  (MAE->gpu$MAE_GPU  NTP->gpu$NTP_GPU)"
run_flow mae "$MAE_GPU" $MAE_LR &
run_flow ntp "$NTP_GPU" $NTP_LR &
wait
echo "ALL DONE — per-horizon logs at logs/timemixer_{mae,ntp}/<dataset>/forecast_<pred>.log"

#!/usr/bin/env bash
# Train both TimeMixer SSL flows on all datasets, then forecast in-domain.
#   MAE -> GPU 6   |   NTP -> GPU 7   (run in parallel)
# LR = 0.01 (matches TimeMixer). Splits/normalization come from PatchTST's
# loaders — the same splits used elsewhere in this repo.
#
# Usage:
#   DATA_DIR=/path/to/csvs ./run_timemixer_ssl.sh
# (DATA_DIR must contain ETTh1.csv, ETTh2.csv, ETTm1.csv, ETTm2.csv, ...)
set -u

REPO="$(cd "$(dirname "$0")" && pwd)"
# Resolve the CSV dir from the same source tsdino uses (data_paths.py); override
# with DATA_DIR=/path ./run_timemixer_ssl.sh if needed.
DATA_DIR="${DATA_DIR:-$(cd "$REPO" && python -c "from data_paths import DATA_PATHS; print(DATA_PATHS['forecasting_data_dir'])")}"

# ── knobs ────────────────────────────────────────────────────────────────────
LR=0.01                 # TimeMixer learning rate
SEQ_LEN=336             # input window (as used in this repo)
PRED_LEN=96             # NTP horizon + downstream forecast horizon
BLOCK_LEN=8             # MAE block size
MASK_RATIO=0.4          # MAE mask fraction
PRE_EPOCHS=80           # pretraining epochs
FC_EPOCHS=30            # downstream forecasting epochs
MODE=linear_probe       # downstream: linear_probe | finetune
# backbone (this repo's defaults; TimeMixer paper uses d_model=16 d_ff=32 e_layers=2)
D_MODEL=128; D_FF=256; E_LAYERS=3
DS_LAYERS=3; DS_WINDOW=2

# name  csv  c_in   ("all datasets" = the 4 ETT; add weather/ECL/traffic below)
DATASETS=(
  "etth1 ETTh1.csv 7"
  "etth2 ETTh2.csv 7"
  "ettm1 ETTm1.csv 7"
  "ettm2 ETTm2.csv 7"
  "weather weather.csv 21"
  # "electricity electricity.csv 321"
  # "traffic traffic.csv 862"
)

run_flow () {   # $1 = flow (mae|ntp)   $2 = gpu id
  local flow=$1 gpu=$2 entry name csv cin out log
  for entry in "${DATASETS[@]}"; do
    read -r name csv cin <<< "$entry"
    out="$REPO/timemixer_$flow/checkpoints/$name"
    log="$REPO/logs/timemixer_$flow/$name"; mkdir -p "$log"

    if [ "$flow" = mae ]; then
      task_args="--block_len $BLOCK_LEN --mask_ratio $MASK_RATIO"
    else
      task_args="--pred_len $PRED_LEN"
    fi

    echo "[$flow|gpu$gpu|$name] pretrain -> $log/pretrain.log"
    CUDA_VISIBLE_DEVICES=$gpu python "$REPO/timemixer_$flow/train.py" \
      --data_path "$DATA_DIR/$csv" --c_in "$cin" --seq_len $SEQ_LEN $task_args \
      --lr $LR --epochs $PRE_EPOCHS --output_dir "$out" \
      --d_model $D_MODEL --d_ff $D_FF --e_layers $E_LAYERS \
      --down_sampling_layers $DS_LAYERS --down_sampling_window $DS_WINDOW \
      > "$log/pretrain.log" 2>&1

    echo "[$flow|gpu$gpu|$name] forecast ($MODE) -> $log/forecast.log"
    CUDA_VISIBLE_DEVICES=$gpu python "$REPO/timemixer_$flow/forecast.py" \
      --init_ckpt "$out/checkpoint_best.pth" --mode $MODE \
      --data_path "$DATA_DIR/$csv" --c_in "$cin" --seq_len $SEQ_LEN --pred_len $PRED_LEN \
      --lr $LR --epochs $FC_EPOCHS \
      --d_model $D_MODEL --d_ff $D_FF --e_layers $E_LAYERS \
      --down_sampling_layers $DS_LAYERS --down_sampling_window $DS_WINDOW \
      > "$log/forecast.log" 2>&1
    echo "[$flow|gpu$gpu|$name] DONE  (test metrics at tail of $log/forecast.log)"
  done
}

[ -d "$DATA_DIR" ] || { echo "DATA_DIR not found: $DATA_DIR" >&2; exit 1; }
echo "data=$DATA_DIR  lr=$LR  seq=$SEQ_LEN  pred=$PRED_LEN  mode=$MODE"
run_flow mae 6 &
run_flow ntp 7 &
wait
echo "ALL DONE — test MSE/MAE in logs/timemixer_{mae,ntp}/<dataset>/forecast.log"

#!/usr/bin/env bash
# Regular (supervised) TimeMixer baseline — runs TimeMixer-main/run.py directly.
# This is the vanilla TimeMixer long-term-forecasting model, NOT the SSL
# (MAE/NTP) flows in run_timemixer_ssl.sh and NOT the DINO variant.
#
# Defaults to GPU 7 and the 6 new forecasting datasets.
#   GPU=7 ./run_timemixer_baseline.sh
#   GPU=7 PRED_LEN=192 ./run_timemixer_baseline.sh
#   TM_DATASETS=$'solar Solar.csv\nwind Wind.csv' ./run_timemixer_baseline.sh
set -u

REPO="$(cd "$(dirname "$0")" && pwd)"
TM="$REPO/models/TimeMixer-main"
# CSV root from the same source the rest of the repo uses (data_paths.py).
DATA_DIR="${DATA_DIR:-$(cd "$REPO" && python -c "from data_paths import DATA_PATHS; print(DATA_PATHS['forecasting_data_dir'])")}"

GPU="${GPU:-7}"

# ── knobs (match config_timemixer.py / this repo's TimeMixer setup) ───────────
SEQ_LEN="${SEQ_LEN:-336}"
PRED_LENS="${PRED_LENS:-96 192 336 720}"   # sweep all 4 horizons (TimeMixer default)
E_LAYERS="${E_LAYERS:-2}"
D_MODEL="${D_MODEL:-128}"
D_FF="${D_FF:-256}"
DS_LAYERS="${DS_LAYERS:-3}"
DS_WINDOW="${DS_WINDOW:-2}"
LR="${LR:-0.01}"
EPOCHS="${EPOCHS:-20}"
BATCH="${BATCH:-128}"

# name  csv   (channel count + target column are auto-detected from the header)
DATASETS=(
  "exchange Exchange.csv"
  "wind Wind.csv"
  "solar Solar.csv"
  "metr_la METR-LA.csv"
  "aqwan AQWan.csv"
  "aqshunyi AQShunyi.csv"
)
# Override the list from the env (newline-separated "name csv" rows).
if [ -n "${TM_DATASETS:-}" ]; then
  DATASETS=(); while IFS= read -r _row; do
    [ -n "$_row" ] && DATASETS+=("$_row")
  done <<< "$TM_DATASETS"
fi

[ -d "$DATA_DIR" ] || { echo "DATA_DIR not found: $DATA_DIR" >&2; exit 1; }
echo "data=$DATA_DIR  gpu=$GPU  seq=$SEQ_LEN  pred=[$PRED_LENS]  lr=$LR  epochs=$EPOCHS"

for entry in "${DATASETS[@]}"; do
  read -r name csv <<< "$entry"
  csv_path="$DATA_DIR/$csv"
  [ -f "$csv_path" ] || { echo "SKIP $name: missing $csv_path" >&2; continue; }

  # Dataset_Custom needs enc_in = #channels and a --target that actually exists
  # (it does cols.remove(target)). Use channel count + the last data column.
  read -r CIN TARGET <<< "$(python - "$csv_path" <<'PY'
import sys, pandas as pd
cols = list(pd.read_csv(sys.argv[1], nrows=0).columns)
data = [c for c in cols if c != 'date']
print(len(data), data[-1])
PY
)"

  log="$REPO/logs/timemixer_baseline/$name"; mkdir -p "$log"
  for PRED_LEN in $PRED_LENS; do
    echo "[gpu$GPU|$name|pl$PRED_LEN] c_in=$CIN target=$TARGET -> $log/pred${PRED_LEN}.log"
    CUDA_VISIBLE_DEVICES=$GPU python -u "$TM/run.py" \
      --task_name long_term_forecast --is_training 1 \
      --model TimeMixer --data custom \
      --root_path "$DATA_DIR/" --data_path "$csv" \
      --model_id "${name}_${SEQ_LEN}_${PRED_LEN}" \
      --features M --target "$TARGET" \
      --seq_len $SEQ_LEN --label_len 0 --pred_len $PRED_LEN \
      --enc_in $CIN --dec_in $CIN --c_out $CIN \
      --e_layers $E_LAYERS --d_model $D_MODEL --d_ff $D_FF \
      --down_sampling_layers $DS_LAYERS --down_sampling_window $DS_WINDOW \
      --down_sampling_method avg \
      --learning_rate $LR --train_epochs $EPOCHS --patience 10 --batch_size $BATCH \
      --des Exp --itr 1 --gpu 0 \
      > "$log/pred${PRED_LEN}.log" 2>&1
    echo "[gpu$GPU|$name|pl$PRED_LEN] DONE  (test MSE/MAE at tail of $log/pred${PRED_LEN}.log)"
  done
done
echo "ALL DONE — metrics in logs/timemixer_baseline/<dataset>/run.log"

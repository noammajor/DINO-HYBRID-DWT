#!/usr/bin/env bash
# Run the SparseTSF + TimeBase baselines, two configurations in parallel:
#   ours_336  : seq_len 336 (matches the TimeMixer flows)   SparseTSF->GPU0  TimeBase->GPU1
#   orig_720  : seq_len 720 (the repos' original config)     SparseTSF->GPU5  TimeBase->GPU6
# Both use each repo's standard Autoformer splits (= ours) and full per-dataset/
# per-horizon tuning. Data dir resolved from data_paths.py.
#
# Usage:  ./run_baselines.sh        (or DATA_DIR=/path ./run_baselines.sh)

set -u
REPO="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${DATA_DIR:-$(cd "$REPO" && python -c "from data_paths import DATA_PATHS; print(DATA_PATHS['forecasting_data_dir'])")}"
PRED_LENS="96 192 336 720"

# GPU assignment (override via env). ours_336 -> 0,1   orig_720 -> 4,5
SPARSE_GPU_OURS="${SPARSE_GPU_OURS:-0}"
TB_GPU_OURS="${TB_GPU_OURS:-1}"
SPARSE_GPU_ORIG="${SPARSE_GPU_ORIG:-4}"
TB_GPU_ORIG="${TB_GPU_ORIG:-5}"

# SparseTSF (model_type=mlp default, d_model 128, batch 256, 30 epochs):
#   name csv data_key enc_in period lr
SPARSE=(
  "ETTh1   ETTh1.csv   ETTh1  7  24  0.02"
  "ETTh2   ETTh2.csv   ETTh2  7  24  0.03"
  "ETTm1   ETTm1.csv   ETTm1  7  4   0.02"
  "ETTm2   ETTm2.csv   ETTm2  7  4   0.02"
  "weather weather.csv custom 21 4   0.02"
)

# TimeBase full per-horizon tuning (extracted from their 720 scripts):
#   name csv data_key enc_in pred period basis ortho lr batch
TIMEBASE=(
  "ETTh1   ETTh1.csv   ETTh1  7  96  24  6  0.16 1e-1 64"
  "ETTh1   ETTh1.csv   ETTh1  7  192 24  6  0.16 4e-1 256"
  "ETTh1   ETTh1.csv   ETTh1  7  336 24  6  0.08 4e-1 256"
  "ETTh1   ETTh1.csv   ETTh1  7  720 24  6  0.12 5e-2 64"
  "ETTh2   ETTh2.csv   ETTh2  7  96  24  6  0.2  2e-1 512"
  "ETTh2   ETTh2.csv   ETTh2  7  192 24  4  0.08 6e-2 512"
  "ETTh2   ETTh2.csv   ETTh2  7  336 24  6  0.12 4e-1 64"
  "ETTh2   ETTh2.csv   ETTh2  7  720 24  6  0.12 4e-1 64"
  "ETTm1   ETTm1.csv   ETTm1  7  96  4   18 0.04 2e-2 512"
  "ETTm1   ETTm1.csv   ETTm1  7  192 4   20 0.04 2e-2 256"
  "ETTm1   ETTm1.csv   ETTm1  7  336 4   20 0.08 2e-2 256"
  "ETTm1   ETTm1.csv   ETTm1  7  720 6   20 0.12 1e-2 128"
  "ETTm2   ETTm2.csv   ETTm2  7  96  4   20 0.04 1e-2 64"
  "ETTm2   ETTm2.csv   ETTm2  7  192 4   20 0.04 1e-2 64"
  "ETTm2   ETTm2.csv   ETTm2  7  336 4   20 0.04 1e-2 64"
  "ETTm2   ETTm2.csv   ETTm2  7  720 6   20 0.04 1e-2 64"
  "weather weather.csv custom 21 96  4   6  0.04 2e-2 512"
  "weather weather.csv custom 21 192 4   6  0.04 2e-2 512"
  "weather weather.csv custom 21 336 4   6  0.08 5e-2 512"
  "weather weather.csv custom 21 720 4   6  0.04 5e-2 512"
)

run_sparsetsf () {   # $1=seq_len  $2=gpu  $3=tag
  local seq=$1 gpu=$2 tag=$3 e name csv dk enc per lr pl log
  cd "$REPO/models/SparseTSF-main" || exit 1
  for e in "${SPARSE[@]}"; do
    read -r name csv dk enc per lr <<< "$e"
    for pl in $PRED_LENS; do
      log="$REPO/logs/baselines/$tag/SparseTSF/$name"; mkdir -p "$log"
      echo "[SparseTSF|$tag|gpu$gpu|$name|$pl]"
      python -u run_longExp.py --is_training 1 --model SparseTSF \
        --root_path "$DATA_DIR/" --data_path "$csv" --data "$dk" --features M \
        --model_id "${name}_${seq}_${pl}" \
        --seq_len "$seq" --pred_len "$pl" --period_len "$per" --enc_in "$enc" \
        --train_epochs 30 --patience 5 --itr 1 --batch_size 256 \
        --learning_rate "$lr" --gpu "$gpu" \
        > "$log/${seq}_${pl}.log" 2>&1
    done
  done
}

run_timebase () {    # $1=seq_len  $2=gpu  $3=tag
  local seq=$1 gpu=$2 tag=$3 e name csv dk enc pl per basis ow lr bs log
  cd "$REPO/models/TimeBase-main" || exit 1
  for e in "${TIMEBASE[@]}"; do
    read -r name csv dk enc pl per basis ow lr bs <<< "$e"
    log="$REPO/logs/baselines/$tag/TimeBase/$name"; mkdir -p "$log"
    echo "[TimeBase|$tag|gpu$gpu|$name|$pl]"
    python -u run_longExp.py --is_training 1 --model LightTimeBaseTST \
      --root_path "$DATA_DIR/" --data_path "$csv" --data "$dk" --features M \
      --model_id "${name}_${seq}_${pl}" \
      --seq_len "$seq" --pred_len "$pl" --period_len "$per" --enc_in "$enc" \
      --basis_num "$basis" --orthogonal_weight "$ow" \
      --train_epochs 30 --patience 5 --itr 1 --batch_size "$bs" \
      --learning_rate "$lr" --gpu "$gpu" \
      > "$log/${seq}_${pl}.log" 2>&1
  done
}

[ -d "$DATA_DIR" ] || { echo "DATA_DIR not found: $DATA_DIR" >&2; exit 1; }
echo "data=$DATA_DIR"
echo "  ours_336: SparseTSF->gpu$SPARSE_GPU_OURS  TimeBase->gpu$TB_GPU_OURS   |   orig_720: SparseTSF->gpu$SPARSE_GPU_ORIG  TimeBase->gpu$TB_GPU_ORIG"
run_sparsetsf 336 "$SPARSE_GPU_OURS" ours_336 &
run_timebase  336 "$TB_GPU_OURS"     ours_336 &
run_sparsetsf 720 "$SPARSE_GPU_ORIG" orig_720 &
run_timebase  720 "$TB_GPU_ORIG"     orig_720 &
wait
echo "ALL DONE — results in logs/baselines/{ours_336,orig_720}/{SparseTSF,TimeBase}/<dataset>/<seq>_<pred>.log"

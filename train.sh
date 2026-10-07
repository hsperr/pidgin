#!/usr/bin/env bash
# Public Pidgin V1 training recipe. See README.md for checkpoint selection and splits.
# ./train.sh [OUT] | ./train.sh --smoke [OUT]
set -euo pipefail
cd "$(dirname "$0")"

SMOKE=0
if [[ ${1:-} == --smoke ]]; then SMOKE=1; shift; fi
OUT=${1:-runs/training}
PY=${PYTHON:-python}
SEED=${SEED:-1}
THREADS=${THREADS:-8}
DATA=${DATA:-data/dds_results_100M.npy}
CODE_WORD_PENALTY=${CODE_WORD_PENALTY:-0.2}

COMMON=(--data "$DATA" --seed "$SEED" --threads "$THREADS"
        --val-start 3028000 --val-count 5000 --eval-start -10000 --eval-count 10000
        --train-pool-start 3033000 --train-pool-end 99990000 --train-block-size 1000000
        --eval-every 1000)
GROUND=(--patience 5000)
FOURSEAT=(--train-block-every 2000 --episodes 512 --silent-frac 0.25 --snapshot-every 1000)
OWN_SIMPLE=(--steps 30000)
TABLE=(--eval-every 3000 --patience 12000)
if (( SMOKE )); then
  DATA=${DATA_SMOKE:-data/smoke_128.npz}
  COMMON=(--data "$DATA" --seed "$SEED" --threads 1
          --train-pool-start 0 --train-pool-end 96 --train-block-size 32
          --val-start 96 --val-count 16 --eval-start -16 --eval-count 16 --eval-every 1)
  GROUND=(--width 16 --suit-width 8 --depth 1 --batch 16 --max-steps 2
          --train-block-every 2 --patience 0)
  FOURSEAT=(--train-block-every 2 --episodes 8 --steps 2 --silent-frac 0.25 --snapshot-every 1)
  OWN_SIMPLE=(--steps 2)
  TABLE=(--eval-every 1 --patience 0)
fi

# Completed stages are skipped. Interrupted stages need a new output directory,
# or explicit four-seat --resume using the options saved in that stage's run.json.
completed() {
  [[ -f "$1/$2" && -f "$1/best.pt" ]]
}

echo "== stage 1: grounding from random weights"
if ! completed "$OUT/1_ground" result.json; then
  "$PY" -u -m training.ground "${COMMON[@]}" "${GROUND[@]}" --out "$OUT/1_ground"
fi

own_stage() {
  local dest=$1 init=$2
  shift 2
  if ! completed "$dest" eval.json; then
    "$PY" -u -m training.fourseat.train "${COMMON[@]}" "${FOURSEAT[@]}" \
      --out "$dest" --init "$init" --select own --table-weight 0 --pg-lr 1e-4 --patience 0 \
      --max-level5-rise 0.15 --max-own-drop 60 --max-double-rate 0.6 "$@"
  fi
  [[ -f "$dest/last.pt" ]] || { echo "Missing $dest/last.pt" >&2; exit 1; }
}

echo "== stage 2: own-contract self-play"
own_stage "$OUT/2_own" "$OUT/1_ground/best.pt"

echo "== stage 3: own-contract simplicity fine-tune"
own_stage "$OUT/3_simple" "$OUT/2_own/last.pt" \
  "${OWN_SIMPLE[@]}" --code-word-penalty "$CODE_WORD_PENALTY"

echo "== stage 4: Pidgin V1 table-score training with simplicity cost"
if ! completed "$OUT/4_D" eval.json; then
  "$PY" -u -m training.fourseat.train "${COMMON[@]}" "${FOURSEAT[@]}" \
    --out "$OUT/4_D" --init "$OUT/3_simple/last.pt" \
    --select imp --imp-opponent "$OUT/3_simple/last.pt" --table-weight 1 \
    --code-word-penalty "$CODE_WORD_PENALTY" \
    --pg-lr 1e-5 --critic-lr 1e-3 \
    --double-tau 0.1 --xx-tau 0.1 --sac-tau 0.5 \
    --double-value-lr 1e-5 --xx-value-lr 1e-5 --sac-value-lr 1e-5 \
    --double-cf-weight 0 --xx-cf-weight 0 --sac-cf-weight 0 --gate-pg \
    --any-seat-double --league-frac 0.5 --league-every 1000 \
    "${TABLE[@]}" --max-level5-rise 1 --max-sac-rate 0.2 --max-double-rate 0.4
fi

echo "== done: $OUT/4_D/best.pt"

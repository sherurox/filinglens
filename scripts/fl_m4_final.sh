#!/usr/bin/env bash
# FL-M4 after the Optuna study (T11) -> final retrain. Runs unattended (about 4 h 15 min).
#   1. Dev check, only if Optuna's best trial isn't the baseline trial (both trained on 400 rows):
#      dev eval of both adapters + paired bootstrap. Rule fixed before the search: the best
#      trial replaces the ablation winner (teacher, r16, lr 2e-4) only if the lower end of the
#      95% CI of (best - baseline) dev macro-F1 is above 0; otherwise it's a tie and the
#      ablation winner stays.
#   2. Final retrain on all 6,259 filtered teacher rows, 1 epoch, with the chosen setting.
#   3. Deterministic dev eval of the final adapter + paired bootstrap against base.
# Test is NOT touched here (T12, once).
# Usage (repo root, MLflow running): nohup bash scripts/fl_m4_final.sh "AC power" >> data/interim/logs/final.log 2>&1 &
set -euo pipefail

NOTE="${1:-AC power, Best performance, VRAM capped to free}"
SUMMARY="${OPTUNA_SUMMARY:-bench/optuna.json}"
TRAIN=~/venvs/fl-train/bin/python
GPU=~/venvs/fl-gpu/bin/python
EVAL=(--gpu-memory-utilization 0.84 --max-model-len 4608 --max-num-seqs 1)
LOGS=data/interim/logs
mkdir -p "$LOGS"
set -a; source .env; set +a

mlflow_up() {  # MLflow answers, or is started once; otherwise stop
  local url="${MLFLOW_TRACKING_URI:-http://127.0.0.1:55000}/health"
  curl -fsS --max-time 5 "$url" > /dev/null 2>&1 && return 0
  echo "    $(date -u +%FT%TZ) MLflow not answering at $url; trying docker compose up" >&2
  if docker compose up -d --wait mlflow > /dev/null 2>&1 \
      && curl -fsS --max-time 5 "$url" > /dev/null 2>&1; then
    return 0
  fi
  echo "    MLflow still down: stopped (start Docker Desktop, then rerun this script)" >&2
  return 1
}

dev_eval() {  # dev_eval <run name>: deterministic dev predictions + scores into the run
  $GPU -m filinglens.extract.fields generate --variant zero --split dev "${EVAL[@]}" \
    --max-lora-rank 16 --lora "data/interim/adapters/$1/adapter" >> "$LOGS/$1.dev.log" 2>&1
  mlflow_up
  uv run python -m filinglens.train.evaluate --run-name "$1" --split dev >> "$LOGS/$1.dev.log" 2>&1
}

read -r BEST BASE SAME BEST_RANK BEST_LR < <(python3 -c "
import json; s = json.load(open('$SUMMARY'))
print(s['best']['run_name'], s['baseline']['run_name'], s['best_is_baseline'],
      s['best']['rank'], s['best']['lr'])")
echo "=== $(date -u +%FT%TZ) optuna best $BEST | baseline $BASE | best is baseline: $SAME"

RANK=16
LR=2e-4
if [[ "$SAME" == "False" ]]; then
  dev_eval "$BASE"
  dev_eval "$BEST"
  uv run python -m filinglens.train.compare --a "$BASE" --b "$BEST" >> "$LOGS/optuna_confirm.log" 2>&1
  LOW=$(python3 -c "import json
print(json.load(open('bench/compare_dev_${BASE}_vs_${BEST}.json'))['category_macro_f1']['diff_ci95'][0])")
  if python3 -c "import sys; sys.exit(0 if $LOW > 0 else 1)"; then
    RANK=$BEST_RANK
    LR=$BEST_LR
    echo "=== dev confirms the Optuna best (CI low $LOW > 0): rank $RANK, lr $LR"
  else
    echo "=== dev tie (CI low $LOW <= 0): keep the ablation winner, rank 16, lr 2e-4"
  fi
fi

NAME=$($TRAIN -c "from filinglens.train.sft import TrainConfig as C
print(C(arm='teacher', rank=$RANK, learning_rate=$LR, epochs=1.0).run_name())")
echo "=== $(date -u +%FT%TZ) final retrain $NAME (all teacher rows)"
if [[ ! -f "bench/sft_$NAME.json" ]]; then
  mlflow_up
  $TRAIN -m filinglens.train.sft --arm teacher --teacher-limit -1 --epochs 1 \
    --rank "$RANK" --lr "$LR" --note "$NOTE" >> "$LOGS/$NAME.log" 2>&1
fi
dev_eval "$NAME"
uv run python -m filinglens.train.compare --a base-s1a --b "$NAME" >> "$LOGS/$NAME.dev.log" 2>&1
python3 - "$NAME" <<'EOF'
import json, sys
name = sys.argv[1]
t = json.load(open(f"bench/sft_{name}.json"))
s = json.load(open(f"bench/fields_zero_dev_{name}.json"))["all_rows"]
c = json.load(open(f"bench/compare_dev_base-s1a_vs_{name}.json"))["category_macro_f1"]
print(f"    steps {t['optimizer_steps']}  {t['train_runtime_s']} s  peak {t['peak_reserved_mib']} MiB  "
      f"mlflow_synced {t.get('mlflow_synced')}")
print(f"    dev macro-F1 {s['category_macro_f1']}  acc {s['category_accuracy']}  vs base "
      f"{c['diff_b_minus_a']:+} (95% CI {c['diff_ci95']}, P(not better) {c['share_b_not_better']})")
EOF
echo "=== $(date -u +%FT%TZ) done: $NAME"

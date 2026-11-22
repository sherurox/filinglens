#!/usr/bin/env bash
# FL-M4 T10 ablations on the 1,500-row teacher subset: train -> deterministic dev eval -> MLflow,
# then a paired comparison against the base model. Runs one after another (about 1 h each).
#   1. both arm (silver 191 + teacher 1,500), r16, lr 2e-4, 1 epoch
#   2. the better arm on dev (teacher or both, by category macro-F1), LoRA rank 8
#   3. the better arm, learning rate 1e-4
# A run whose bench/sft_<name>.json already exists is not retrained (safe to restart).
# Full output of each step goes to data/interim/logs/<name>.log; progress lines go to the terminal.
# Usage (repo root, MLflow running):  bash scripts/fl_m4_ablate.sh "AC power, Best performance"
set -euo pipefail

NOTE="${1:-AC power, Best performance, VRAM capped to free}"
TRAIN=~/venvs/fl-train/bin/python
GPU=~/venvs/fl-gpu/bin/python
EVAL=(--gpu-memory-utilization 0.84 --max-model-len 4608 --max-num-seqs 1)
LOGS=data/interim/logs
mkdir -p "$LOGS"
set -a; source .env; set +a

name_of() {  # name_of <arm> <rank> <lr>
  $TRAIN -c "from filinglens.train.sft import TrainConfig as C
print(C(arm='$1', rank=$2, learning_rate=$3, epochs=1.0, teacher_limit=1500).run_name())"
}

run() {  # run <arm> <rank> <lr>
  local name log
  name=$(name_of "$1" "$2" "$3")
  log="$LOGS/$name.log"
  echo "=== $(date -u +%FT%TZ) $name (log: $log)"
  if [[ ! -f "bench/sft_$name.json" ]]; then
    $TRAIN -m filinglens.train.sft --arm "$1" --teacher-limit 1500 --epochs 1 \
      --rank "$2" --lr "$3" --note "$NOTE" >> "$log" 2>&1
  fi
  $GPU -m filinglens.extract.fields generate --variant zero --split dev "${EVAL[@]}" \
    --max-lora-rank 16 --lora "data/interim/adapters/$name/adapter" >> "$log" 2>&1
  uv run python -m filinglens.train.evaluate --run-name "$name" --split dev >> "$log" 2>&1
  uv run python -m filinglens.train.compare --a base-s1a --b "$name" >> "$log" 2>&1
  python3 - "$name" <<'EOF'
import json, sys
name = sys.argv[1]
s = json.load(open(f"bench/fields_zero_dev_{name}.json"))["all_rows"]
c = json.load(open(f"bench/compare_dev_base-s1a_vs_{name}.json"))["category_macro_f1"]
print(f"    dev macro-F1 {s['category_macro_f1']}  acc {s['category_accuracy']}  "
      f"vs base {c['diff_b_minus_a']:+} (95% CI {c['diff_ci95']}, P(not better) {c['share_b_not_better']})")
EOF
}

f1() { python3 -c "import json; print(json.load(open('bench/fields_zero_dev_$1.json'))['all_rows']['category_macro_f1'])"; }

run both 16 2e-4
TEACHER=teacher-n1500-r16-lr0.0002-e1
BOTH=$(name_of both 16 2e-4)
BEST=$(python3 -c "print('both' if $(f1 "$BOTH") > $(f1 "$TEACHER") else 'teacher')")
echo "=== better arm on dev: $BEST (teacher $(f1 "$TEACHER"), both $(f1 "$BOTH"))"
run "$BEST" 8 2e-4
run "$BEST" 16 1e-4
echo "=== $(date -u +%FT%TZ) done"

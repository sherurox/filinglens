#!/usr/bin/env bash
# FL-M5 T7: the serving benchmark (D1), in three phases, under the rule committed in
# bench/m5_serving_rule.json.
#   bash scripts/fl_m5_serving.sh load  "AC power, Best performance"   # dev prompts only (~25 min)
#   bash scripts/fl_m5_serving.sh test  "AC power, Best performance"   # the one test run (~10 min)
#   bash scripts/fl_m5_serving.sh table                                 # bench/serving.csv + chart
# load: each GPU build is served by vLLM (serve/vllm_server.py), then the GGUF build by llama-server
# with 16 slots; one server at a time, VRAM back to idle before the next. test: refuses to run
# unless the rule file is committed and unchanged, and if any test prediction file already exists.
set -euo pipefail

PHASE="${1:?usage: fl_m5_serving.sh load|test|table [\"<power mode>\"]}"
NOTE="${2:-}"
GPU_PY="$HOME/venvs/fl-gpu/bin/python"
TRAIN_PY="$HOME/venvs/fl-train/bin/python"
LLAMA="$HOME/src/llama.cpp/build/bin/llama-server"
GGUF=data/interim/serve/gguf/student-q4_k_m.gguf
RULE=bench/m5_serving_rule.json
VLLM_BUILDS=(merged-bf16 fp8-dynamic gptq-w4a16 awq-w4a16)
COMPARED=(fp8-dynamic gptq-w4a16 awq-w4a16 gguf-q4_k_m)
EVAL=(--gpu-memory-utilization 0.83 --max-model-len 4608 --max-num-seqs 1)
LOAD=(--levels 1 4 16 --requests 64 --warmup 4)
LLAMA_LOAD=(-np 16 -c 49152 -t 12 -tb 24 --jinja --seed 0)
LLAMA_EVAL=(-np 1 -c 4608 -t 12 -tb 24 --jinja --seed 0)
SERVER_PID=""

idle_vram() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1; }
stop_server() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID"; wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap stop_server EXIT INT TERM
wait_healthy() {  # wait_healthy <url> <log>: up to 6 min; stop early if the server died
  for _ in $(seq 1 180); do
    curl -sf "$1/health" > /dev/null && return 0
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "server exited; last lines of $2:"; tail -n 30 "$2"; exit 1; }
    sleep 2
  done
  echo "server not healthy after 6 min; last lines of $2:"; tail -n 30 "$2"; exit 1
}
wait_vram_back() {  # wait_vram_back <idle MiB>: up to 60 s for the driver to release memory
  for _ in $(seq 1 30); do
    (( $(idle_vram) <= $1 + 300 )) && return 0
    sleep 2
  done
  echo "warning: VRAM still at $(idle_vram) MiB (idle was $1)"
}
need_note() { [[ -n "$NOTE" ]] || { echo "give the power mode as the 2nd argument"; exit 1; }; }

phase_load() {
  need_note
  for b in "${VLLM_BUILDS[@]}"; do
    echo "=== load $b  $(date +%H:%M:%S)"
    idle=$(idle_vram); log=logs/vllm_server_$b.log
    "$GPU_PY" -m filinglens.serve.vllm_server --build "$b" > "$log" 2>&1 & SERVER_PID=$!
    wait_healthy http://127.0.0.1:58000 "$log"
    uv run python -m filinglens.serve.loadgen --url http://127.0.0.1:58000 --tag "$b" "${LOAD[@]}" \
      --note "$NOTE" --idle-vram-mib "$idle" --server-log "$log" \
      --server-cmd "python -m filinglens.serve.vllm_server --build $b"
    stop_server; wait_vram_back "$idle"
  done
  echo "=== load gguf-q4_k_m (CPU)  $(date +%H:%M:%S)"
  idle=$(idle_vram); log=logs/llama_server_load.log
  "$LLAMA" -m "$GGUF" --host 127.0.0.1 --port 58081 "${LLAMA_LOAD[@]}" > "$log" 2>&1 & SERVER_PID=$!
  wait_healthy http://127.0.0.1:58081 "$log"
  uv run python -m filinglens.serve.loadgen --url http://127.0.0.1:58081 --tag gguf-q4_k_m \
    "${LOAD[@]}" --note "$NOTE" --idle-vram-mib "$idle" --server-log "$log" \
    --server-cmd "llama-server -m $GGUF ${LLAMA_LOAD[*]}"
  stop_server
}

phase_test() {
  need_note
  git ls-files --error-unmatch "$RULE" > /dev/null 2>&1 || { echo "$RULE is not committed"; exit 1; }
  git diff --quiet HEAD -- "$RULE" || { echo "$RULE differs from its commit"; exit 1; }
  for b in "${VLLM_BUILDS[@]}" gguf-q4_k_m; do
    f=data/interim/fields/fields_zero_test_$b.jsonl
    [[ ! -e "$f" ]] || { echo "$f exists: the test split was already run for $b"; exit 1; }
  done
  echo "rule $(git log -1 --format=%h -- "$RULE") ok; power: $NOTE"
  for b in "${VLLM_BUILDS[@]}"; do
    echo "=== test $b  $(date +%H:%M:%S)"
    "$GPU_PY" -m filinglens.extract.fields generate --variant zero --split test --allow-test \
      "${EVAL[@]}" --model "data/interim/serve/$b" --tag "$b" > "logs/fields_test_$b.log" 2>&1 \
      || { echo "FAILED: $b"; tail -n 20 "logs/fields_test_$b.log"; exit 1; }
  done
  echo "=== test gguf-q4_k_m  $(date +%H:%M:%S)"
  log=logs/llama_server_test.log
  "$LLAMA" -m "$GGUF" --host 127.0.0.1 --port 58081 "${LLAMA_EVAL[@]}" > "$log" 2>&1 & SERVER_PID=$!
  wait_healthy http://127.0.0.1:58081 "$log"
  "$TRAIN_PY" -m filinglens.serve.llama_client generate --split test --allow-test --tag gguf-q4_k_m \
    > logs/fields_test_gguf-q4_k_m.log
  stop_server
  for b in "${VLLM_BUILDS[@]}" gguf-q4_k_m; do
    uv run python -m filinglens.extract.fields score --variant zero --split test --allow-test \
      --tag "$b" > /dev/null
  done
  for b in "${COMPARED[@]}"; do
    uv run python -m filinglens.train.compare --a merged-bf16 --b "$b" --split test --allow-test \
      > /dev/null
  done
  echo "test done  $(date +%H:%M:%S)"
}

case "$PHASE" in
  load) phase_load ;;
  test) phase_test ;;
  table) uv run python -m filinglens.serve.serving_table ;;
  *) echo "unknown phase $PHASE"; exit 1 ;;
esac

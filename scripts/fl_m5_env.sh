#!/usr/bin/env bash
# FL-M5 serving environment -> bench/serving_env.txt.
# Rerun after each new tool (T3 quantizers, T4 llama.cpp, T8 ONNX Runtime, T12 FastAPI): a tool
# that isn't there yet is printed as "not installed", so the file always says what was present.
# Usage (repo root, services up): bash scripts/fl_m5_env.sh "AC power, Best performance"
set -uo pipefail  # no -e: a missing tool must not stop the record

NOTE="${1:?usage: bash scripts/fl_m5_env.sh \"<power mode note>\"}"
OUT=bench/serving_env.txt
TRAIN_PY="$HOME/venvs/fl-train/bin/python"
GPU_PY="$HOME/venvs/fl-gpu/bin/python"
QUANT_PY="$HOME/venvs/fl-quant/bin/python"        # T3 quantizers
ONNX_PY="$HOME/venvs/fl-onnx/bin/python"          # T8 ONNX Runtime
LLAMA_DIR="${LLAMA_CPP_DIR:-$HOME/src/llama.cpp}"  # T4

packages() {  # $1 = python, rest = distribution names
  local py="$1"; shift
  if [[ -x "$py" ]]; then "$py" scripts/env_info.py packages "$@"; else echo "($py not found)"; fi
}

{
  echo "# FL-M5 serving environment"
  echo "recorded_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "command: bash scripts/fl_m5_env.sh \"$NOTE\""
  dirty=$(git status --porcelain | awk '{print $2}' | grep -v '^bench/' | tr '\n' ' ')
  echo "git: $(git rev-parse --short HEAD); uncommitted outside bench/: ${dirty:-none}"
  echo "power: $NOTE"

  echo; echo "## gpu (idle, before any model is loaded)"
  nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.used,memory.total,temperature.gpu,power.draw,power.limit,pstate \
    --format=csv 2>&1
  echo "note: WSL can't list Windows processes; memory.used includes the Windows desktop and apps"
  if command -v nvcc >/dev/null; then nvcc --version | tail -n 2; else echo "nvcc: not installed"; fi

  echo; echo "## host"
  lscpu | grep -E '^(Model name|CPU\(s\)|Thread\(s\) per core):' | sed 's/  */ /g'
  free -m | awk '/^Mem:/ {print "ram_mib total " $2 ", available " $7}'
  echo "kernel: $(uname -r)"
  df -h / /mnt/d 2>/dev/null | awk 'NR == 1 || /\// {print}'

  echo; echo "## containers (name, image, image ID)"
  for c in filinglens-postgres filinglens-qdrant filinglens-mlflow; do
    docker inspect --format '{{.Name}} {{.Config.Image}} {{.Image}}' "$c" 2>&1 | sed 's#^/##'
  done

  echo; echo "## python packages: uv env"
  uv run --quiet python scripts/env_info.py packages \
    mlflow numpy scipy pandas psycopg httpx fastapi uvicorn opentelemetry-sdk google-genai

  echo; echo "## python packages: fl-train venv ($TRAIN_PY)"
  packages "$TRAIN_PY" torch transformers peft accelerate bitsandbytes safetensors huggingface-hub \
    sentencepiece onnx onnxruntime onnxruntime-gpu mlflow
  [[ -x "$TRAIN_PY" ]] && "$TRAIN_PY" -c \
    "import torch; print('torch cuda', torch.version.cuda, '| capability', torch.cuda.get_device_capability(0))" 2>&1

  echo; echo "## python packages: fl-gpu venv ($GPU_PY)"
  packages "$GPU_PY" vllm torch transformers compressed-tensors triton mlflow

  echo; echo "## python packages: fl-quant venv ($QUANT_PY)"
  packages "$QUANT_PY" llmcompressor compressed-tensors torch transformers

  echo; echo "## python packages: fl-onnx venv ($ONNX_PY)"
  packages "$ONNX_PY" onnxruntime-gpu onnx numpy nvidia-cublas nvidia-cudnn-cu13

  echo; echo "## llama.cpp ($LLAMA_DIR)"
  if [[ -x "$LLAMA_DIR/build/bin/llama-server" ]]; then
    echo "commit: $(git -C "$LLAMA_DIR" rev-parse HEAD 2>&1)"
    "$LLAMA_DIR/build/bin/llama-server" --version 2>&1 | head -n 3
    echo "LICENSE: $(head -n 1 "$LLAMA_DIR/LICENSE" 2>&1)"
  else
    echo "not built yet (T4)"
  fi

  echo; echo "## models (local Hugging Face cache)"
  if [[ -x "$TRAIN_PY" ]]; then
    "$TRAIN_PY" scripts/env_info.py models \
      Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
      BAAI/bge-reranker-base@2cfc18c9415c912f9d8155881c133215df768a70 \
      BAAI/bge-base-en-v1.5@a5beb1e3e68b9ab74eb54cfd186867f64f240e1a
  fi
} > "$OUT" 2>&1

cat "$OUT"

#!/usr/bin/env bash
# FL-M5 T4: build llama.cpp (CPU only, the "edge" target) and the GGUF builds of the merged student.
#   merged-bf16 (T2) --convert_hf_to_gguf.py--> student-bf16.gguf --llama-quantize--> student-q4_k_m.gguf
# Records commit, build flags, times, sizes and the Q4_K_M file's SHA-256 in bench/gguf_build.json.
# Usage (repo root): bash scripts/fl_m5_gguf.sh            (LLAMA_CPP_REF=<tag or sha> to pin a commit)
# llama.cpp and its gguf-py are MIT. The converter runs in the fl-train venv (torch, transformers,
# safetensors already there) and imports gguf-py from the llama.cpp checkout. It also needs
# `sentencepiece` (Apache-2.0): the Qwen converter tries the SentencePiece path first and only falls
# back to Qwen3's BPE tokenizer when that attempt fails as expected (no tokenizer.model); without
# the package the ImportError escapes instead (26 Nov). Install: uv pip install --python <fl-train> sentencepiece
# A failed step prints the tail of its log and stops. An existing build at the same commit is reused.
set -euo pipefail

LLAMA_DIR="${LLAMA_CPP_DIR:-$HOME/src/llama.cpp}"
LLAMA_REF="${LLAMA_CPP_REF:-}"
TRAIN_PY="$HOME/venvs/fl-train/bin/python"
MERGED=data/interim/serve/merged-bf16
OUT=data/interim/serve/gguf
JOBS=16
CMAKE_FLAGS=(-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_NATIVE=ON -DLLAMA_CURL=OFF)

[[ -f "$MERGED/config.json" ]] || { echo "missing $MERGED (run T2 first)"; exit 1; }
if [[ ! -d "$LLAMA_DIR/.git" ]]; then
  mkdir -p "$(dirname "$LLAMA_DIR")"
  git clone https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
fi
if [[ -n "$LLAMA_REF" ]]; then
  git -C "$LLAMA_DIR" fetch --tags --quiet && git -C "$LLAMA_DIR" checkout --quiet "$LLAMA_REF"
fi
COMMIT=$(git -C "$LLAMA_DIR" rev-parse HEAD)
echo "llama.cpp $COMMIT"

now() { date +%s.%N; }
step() {  # step <name> <log> <command...>: run quietly; on failure show why and stop
  local name="$1" log="$2"; shift 2
  echo "--- $name"
  if ! "$@" > "$log" 2>&1; then
    echo "FAILED: $name (last 40 lines of $log)"; tail -n 40 "$log"; exit 1
  fi
}
mkdir -p logs "$OUT"
"$TRAIN_PY" -c "import sentencepiece" 2>/dev/null \
  || { echo "missing sentencepiece in $TRAIN_PY (see header)"; exit 1; }

BIN="$LLAMA_DIR/build/bin"
STAMP="$LLAMA_DIR/build/.fl_m5_commit"
t0=$(now)
if [[ -x "$BIN/llama-server" && -x "$BIN/llama-quantize" && "$(cat "$STAMP" 2>/dev/null)" == "$COMMIT" ]]; then
  BUILD="reused"
else
  step "cmake" logs/llama_cmake.log cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" "${CMAKE_FLAGS[@]}"
  step "build" logs/llama_build.log \
    cmake --build "$LLAMA_DIR/build" -j "$JOBS" --target llama-server llama-quantize
  echo "$COMMIT" > "$STAMP"
  BUILD="built"
fi
t1=$(now)
step "convert to bf16 GGUF" logs/gguf_convert.log "$TRAIN_PY" "$LLAMA_DIR/convert_hf_to_gguf.py" \
  "$MERGED" --outtype bf16 --outfile "$OUT/student-bf16.gguf"
t2=$(now)
step "quantize Q4_K_M" logs/gguf_quantize.log \
  "$BIN/llama-quantize" "$OUT/student-bf16.gguf" "$OUT/student-q4_k_m.gguf" Q4_K_M
t3=$(now)

BUILD="$BUILD" COMMIT="$COMMIT" FLAGS="${CMAKE_FLAGS[*]}" T0="$t0" T1="$t1" T2="$t2" T3="$t3" OUT="$OUT" \
LICENSE_LINE="$(head -n 1 "$LLAMA_DIR/LICENSE")" uv run --quiet python - <<'EOF'
import hashlib, json, os
from datetime import UTC, datetime
from pathlib import Path

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

e = os.environ
out = Path(e["OUT"])
t = [float(e[k]) for k in ("T0", "T1", "T2", "T3")]
record = {
    "command": "bash scripts/fl_m5_gguf.sh",
    "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    "llama_cpp_commit": e["COMMIT"],
    "llama_cpp_license": e["LICENSE_LINE"],
    "cmake_flags": e["FLAGS"],
    "source": "data/interim/serve/merged-bf16",
    "build": e["BUILD"],
    "build_s": round(t[1] - t[0], 1) if e["BUILD"] == "built" else None,
    "convert_bf16_s": round(t[2] - t[1], 1),
    "quantize_q4_k_m_s": round(t[3] - t[2], 1),
    "bf16_gguf_bytes": (out / "student-bf16.gguf").stat().st_size,
    "q4_k_m_gguf_bytes": (out / "student-q4_k_m.gguf").stat().st_size,
    "q4_k_m_sha256": sha(out / "student-q4_k_m.gguf"),
}
Path("bench/gguf_build.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
EOF
grep -iE "llama_model_quantize_impl: (model size|quant size)" logs/gguf_quantize.log || true

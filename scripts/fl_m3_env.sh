#!/usr/bin/env bash
# Record the FL-M3 retrieval environment in bench/retrieval_env.txt (FL-M3 T1).
#
# Versions and licenses come from the running containers, the installed package metadata and the
# Hugging Face model cards, not from memory. It only reads: the vector extension is created once by
# hand (T1) and by filinglens.retrieval.stores later.
#
#   bash scripts/fl_m3_env.sh "AC power, Best performance"
set -euo pipefail

POWER_NOTE=${1:?usage: bash scripts/fl_m3_env.sh "<power source and Windows power mode>"}
OUT=bench/retrieval_env.txt
TRAIN_PY=${TRAIN_PY:-$HOME/venvs/fl-train/bin/python}
QDRANT_URL=${QDRANT_URL:-http://127.0.0.1:56333}
MODELS=(BAAI/bge-base-en-v1.5 BAAI/bge-reranker-base)

# Version and license of each installed distribution, from its own metadata.
read -r -d '' PKG_META <<'PY' || true
import importlib.metadata as md
import sys

for name in sys.argv[1:]:
    try:
        meta = md.metadata(name)
    except md.PackageNotFoundError:
        print(f"{name}: not installed")
        continue
    lic = (meta.get("License-Expression") or meta.get("License") or "-").splitlines()[0][:60]
    classifiers = [c.split("::")[-1].strip() for c in meta.get_all("Classifier") or []
                   if c.startswith("License ::")]
    shown = ", ".join(classifiers) or "-"
    print(f"{name} {meta['Version']} | license: {lic} | classifiers: {shown}")
PY

# The license the qdrant package itself declares in the image's SPDX bill of materials.
read -r -d '' SPDX <<'PY' || true
import json
import sys

doc = json.load(open(sys.argv[1]))
for pkg in doc.get("packages", []):
    if pkg.get("name") == "qdrant":
        print(f"qdrant {pkg.get('versionInfo')} | licenseDeclared: {pkg.get('licenseDeclared')}")
PY

# Model revision (commit SHA) and card license; T4/T8 load exactly these revisions.
read -r -d '' MODEL_META <<'PY' || true
import sys

from huggingface_hub import model_info

for repo in sys.argv[1:]:
    info = model_info(repo)
    card = getattr(info.card_data, "license", None) if info.card_data else None
    tags = [t for t in info.tags or [] if t.startswith("license:")]
    print(f"{repo} revision {info.sha} | card license: {card} | tags: {', '.join(tags) or '-'}")
PY

psql_q() { docker exec filinglens-postgres psql -U filinglens -d filinglens -tA -c "$1"; }

spdx=$(mktemp)
trap 'rm -f "$spdx"' EXIT

{
  echo "# FL-M3 retrieval environment"
  echo "recorded_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "command: bash scripts/fl_m3_env.sh \"$POWER_NOTE\""
  echo "git: $(git rev-parse --short HEAD); uncommitted outside bench/:" \
    "$(git status --short -- . ':!bench' | awk '{print $2}' | tr '\n' ' ')"
  echo "power: $POWER_NOTE"
  echo
  echo "## containers (image, image ID)"
  for c in filinglens-postgres filinglens-qdrant filinglens-mlflow; do
    docker inspect --format "$c {{.Config.Image}} {{.Image}}" "$c"
  done
  echo
  echo "## postgres + pgvector"
  psql_q "select version()"
  echo "vector extension: $(psql_q "select extversion from pg_extension where extname = 'vector'")"
  echo "kg.nodes: $(psql_q "select count(*) from kg.nodes")"
  echo "kg.edges: $(psql_q "select count(*) from kg.edges")"
  lic=$(docker exec filinglens-postgres head -3 /usr/share/doc/pgvector/LICENSE | tr -s '\n' ' ')
  echo "pgvector LICENSE: $lic"
  echo
  echo "## qdrant"
  echo "GET /: $(curl -fsS "$QDRANT_URL/")"
  echo "GET /readyz: $(curl -fsS "$QDRANT_URL/readyz")"
  docker cp -q filinglens-qdrant:/qdrant/qdrant.spdx.json "$spdx"
  uv run python - "$spdx" <<<"$SPDX"
  echo
  echo "## python packages: uv env"
  uv run python - faiss-cpu qdrant-client pgvector psycopg numpy pandas scipy mlflow <<<"$PKG_META"
  echo
  echo "## python packages: fl-train venv ($TRAIN_PY)"
  "$TRAIN_PY" - torch transformers huggingface-hub tokenizers numpy mlflow <<<"$PKG_META"
  echo
  echo "## models"
  "$TRAIN_PY" - "${MODELS[@]}" <<<"$MODEL_META"
  echo
  echo "## gpu"
  nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total,temperature.gpu \
    --format=csv,noheader
} | tee "$OUT"

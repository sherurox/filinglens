#!/usr/bin/env bash
# FL-M6 agents environment -> bench/agents_env.txt.
# Rerun after each new tool (T13 promptfoo): a tool that isn't there yet is printed as
# "not installed", so the file always says what was present.
# Every package the `agents` group resolves to (direct and transitive) is listed with the license
# its own metadata declares, so nothing in the agents stack goes unrecorded.
# Usage (repo root, services up): bash scripts/fl_m6_env.sh
set -uo pipefail  # no -e: a missing tool must not stop the record

OUT=bench/agents_env.txt
REDTEAM_DIR=redteam  # T13: promptfoo is installed here with a lock file (decided in T13)

{
  echo "# FL-M6 agents environment"
  echo "recorded_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "command: bash scripts/fl_m6_env.sh"
  dirty=$(git status --porcelain | awk '{print $2}' | grep -v '^bench/' | tr '\n' ' ')
  echo "git: $(git rev-parse --short HEAD); uncommitted outside bench/: ${dirty:-none}"
  echo "uv: $(uv --version 2>&1)"
  echo "python (uv env): $(uv run --quiet python -c 'import sys; print(sys.version.split()[0])' 2>&1)"

  echo; echo "## node (promptfoo needs ^20.20.0 or >=22.22.0)"
  if command -v node >/dev/null; then
    echo "node: $(node --version) at $(command -v node)"
    echo "npm: $(npm --version 2>&1)"
  else
    echo "node: not installed"
  fi
  if [[ -x "$REDTEAM_DIR/node_modules/.bin/promptfoo" ]]; then
    echo "promptfoo: $("$REDTEAM_DIR/node_modules/.bin/promptfoo" --version 2>&1 | tail -n 1)"
    echo "promptfoo license: $(node -p "require('./$REDTEAM_DIR/node_modules/promptfoo/package.json').license" 2>&1)"
  else
    echo "promptfoo: not installed (T13)"
  fi

  echo; echo "## containers (name, image, image ID)"
  for c in filinglens-postgres filinglens-qdrant filinglens-mlflow; do
    docker inspect --format '{{.Name}} {{.Config.Image}} {{.Image}}' "$c" 2>&1 | sed 's#^/##'
  done

  echo; echo "## agents group: direct dependencies (pyproject.toml)"
  sed -n '/^agents = \[/,/^\]/p' pyproject.toml

  echo; echo "## agents group: every resolved package, version and declared license"
  mapfile -t names < <(
    uv export --only-group agents --no-emit-project --no-hashes --no-header \
      --format requirements-txt 2>/dev/null |
      grep -E '^[A-Za-z0-9]' | sed -E 's/[^A-Za-z0-9._-].*//' | sort -u
  )
  if ((${#names[@]})); then
    uv run --quiet --group agents python scripts/env_info.py packages "${names[@]}"
  else
    echo "(uv export found no agents group)"
  fi

  echo; echo "## shared packages the crew also uses (api group)"
  uv run --quiet --group api --group agents python scripts/env_info.py packages \
    psycopg pgvector onnxruntime tokenizers numpy opentelemetry-sdk
} > "$OUT" 2>&1

cat "$OUT"

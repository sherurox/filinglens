#!/usr/bin/env bash
# FL-M5 T12: build the API image and record its measured sizes and startup in bench/api_image.json.
#   set -a; source .env; set +a; docker compose up -d --wait postgres
#   bash scripts/fl_m5_image.sh
# Sizes, all measured: `docker image inspect .Size` (with Docker Desktop's containerd image store
# this is the compressed content size, what a registry stores), the gzipped `docker save` stream,
# and the unpacked root filesystem (`du -sxb /` inside a container of the image). Startup: seconds
# from `docker run` to the first 200 from /health, on the compose network next to Postgres.
set -euo pipefail

IMAGE=filinglens-api:local
NAME=fl-api-measure
: "${POSTGRES_PASSWORD:?source .env first}"

uv export --only-group api --no-emit-project --format requirements-txt -o requirements-api.txt >/dev/null
git diff --quiet -- requirements-api.txt || echo "note: requirements-api.txt changed (commit it)"

start=$(date +%s.%N)
docker build -t "$IMAGE" . >logs/docker_build.log 2>&1
BUILD_S=$(echo "$(date +%s.%N) - $start" | bc)

CONTENT=$(docker image inspect "$IMAGE" -f '{{.Size}}')
GZIP=$(docker save "$IMAGE" | gzip -1 | wc -c)
UNPACKED=$(docker run --rm --entrypoint sh "$IMAGE" -c 'du -sxb / 2>/dev/null | cut -f1')
IMAGE_ID=$(docker image inspect "$IMAGE" -f '{{.Id}}')
BASE=$(docker image inspect python:3.12-slim -f '{{index .RepoDigests 0}}' 2>/dev/null || echo unknown)
LAYERS=$(docker history "$IMAGE" --no-trunc --format '{{.Size}}|{{.CreatedBy}}' | head -n 12)

docker rm -f "$NAME" >/dev/null 2>&1 || true
start=$(date +%s.%N)
docker run -d --name "$NAME" --network filinglens_default -p 127.0.0.1:8082:8080 \
  -e GEMINI_API_KEY -e POSTGRES_PASSWORD -e POSTGRES_HOST=filinglens-postgres \
  -e POSTGRES_PORT=5432 "$IMAGE" >/dev/null
for _ in $(seq 1 300); do curl -sf localhost:8082/health >/dev/null && break; sleep 0.1; done
HEALTHY_S=$(echo "$(date +%s.%N) - $start" | bc)
HEALTH=$(curl -s localhost:8082/health)
USER_ID=$(docker exec "$NAME" id -u)
docker rm -f "$NAME" >/dev/null

BUILD_S=$BUILD_S CONTENT=$CONTENT GZIP=$GZIP UNPACKED=$UNPACKED IMAGE_ID=$IMAGE_ID BASE=$BASE \
LAYERS=$LAYERS HEALTHY_S=$HEALTHY_S HEALTH=$HEALTH USER_ID=$USER_ID \
uv run python - <<'PY'
import json, os
from datetime import UTC, datetime
e = os.environ
budget = 500_000_000  # Artifact Registry free storage: 0.5 GB (02_FREE_TIERS.md)
out = {
    "command": "bash scripts/fl_m5_image.sh",
    "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    "image": "filinglens-api:local",
    "image_id": e["IMAGE_ID"],
    "base": e["BASE"],
    "build_s": round(float(e["BUILD_S"]), 1),
    "bytes": {
        "docker_image_inspect_size": int(e["CONTENT"]),
        "docker_save_gzip": int(e["GZIP"]),
        "unpacked_rootfs_du": int(e["UNPACKED"]),
    },
    "registry_budget_bytes": budget,
    "under_budget": int(e["CONTENT"]) < budget and int(e["GZIP"]) < budget,
    "layers": [dict(zip(("size", "created_by"), l.split("|", 1))) for l in e["LAYERS"].splitlines()],
    "startup": {"seconds_to_healthy": round(float(e["HEALTHY_S"]), 2), "health": json.loads(e["HEALTH"])},
    "runs_as_uid": int(e["USER_ID"]),
}
open("bench/api_image.json", "w").write(json.dumps(out, indent=2) + "\n")
print(json.dumps({k: v for k, v in out.items() if k != "layers"}, indent=2))
PY

# FilingLens API (FL-M5 T12): FastAPI + the int8 ONNX bge-base query encoder; no torch.
# Build context = the repo root; .dockerignore lets in only the files copied below.
#   uv export --only-group api --no-emit-project --format requirements-txt -o requirements-api.txt
#   docker build -t filinglens-api:local .
# The base is pinned by digest (python:3.12-slim's multi-arch index), so every build uses the same bytes.
# Runs as a non-root user and listens on $PORT (Cloud Run sets it; 8080 by default).
FROM python:3.12-slim@sha256:191083cdf6c7bda8c76158983163ead5d5cf7b77cf3033e805bb0f5296f51b4b

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Pinned, hash-checked dependencies exported from uv.lock (the `api` group only).
COPY requirements-api.txt .
RUN pip install --require-hashes --no-deps -r requirements-api.txt

# Code, the list prices / power the cost meter uses, and the query encoder.
COPY src/filinglens /app/src/filinglens
COPY bench/router_inputs.json /app/bench/router_inputs.json
COPY data/interim/serve/onnx/bge-base.eager-opt-int8.onnx /app/model/bge-base-int8.onnx
COPY data/interim/serve/onnx/bge-base.tokenizer.json /app/model/bge-base.tokenizer.json

ENV PYTHONPATH=/app/src \
    EMBED_MODEL=/app/model/bge-base-int8.onnx \
    EMBED_TOKENIZER=/app/model/bge-base.tokenizer.json \
    ROUTER_INPUTS=/app/bench/router_inputs.json \
    GEMINI_QUOTA_FILE=/tmp/gemini_quota.json \
    PORT=8080

RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin app
USER app
EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn filinglens.api.app:app --host 0.0.0.0 --port ${PORT}"]

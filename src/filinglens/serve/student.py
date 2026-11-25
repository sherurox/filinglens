"""Fetch the FL-M4 student from the MLflow registry and prove it is the adapter we evaluated (T1).

    uv run python -m filinglens.serve.student fetch

downloads `filinglens-risk-student` version 1 into data/interim/serve/student-v1/ and writes
bench/student_fetch.json. The check: every file of the registry copy has the same SHA-256 as the
local FL-M4 adapter (same file names, nothing missing or extra), and the weights' SHA-256 is
reported against the registry tags. Every later FL-M5 step (merge, quantize, serve) reads the
adapter from this verified copy only, so "we serve the registered model" can be checked.

Exit code 1 when the two trees differ.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

from filinglens.train.tracking import file_sha256, tracking_uri

MODEL_NAME = "filinglens-risk-student"
VERSION = "1"
LOCAL_ADAPTER = Path("data/interim/adapters/teacher-n1500-r16-lr0.0002-e1/adapter")
OUT_ROOT = Path("data/interim/serve")
WEIGHTS = "adapter_model.safetensors"
SHA256 = re.compile(r"[0-9a-f]{64}")


def tree_sha256(root: Path) -> dict[str, str]:
    """Relative POSIX path -> SHA-256 for every regular file under root, in sorted order."""
    if not root.is_dir():
        raise FileNotFoundError(f"not a directory: {root}")
    files = sorted(p for p in root.rglob("*") if p.is_file())
    return {p.relative_to(root).as_posix(): file_sha256(p) for p in files}


def compare_trees(registry: dict[str, str], local: dict[str, str]) -> dict:
    """Which files match, differ, or exist on one side only. Identical = all match, none missing."""
    common = registry.keys() & local.keys()
    differ = sorted(k for k in common if registry[k] != local[k])
    only_registry = sorted(registry.keys() - local.keys())
    only_local = sorted(local.keys() - registry.keys())
    return {
        "identical": not (differ or only_registry or only_local) and bool(common),
        "files_same": len(common) - len(differ),
        "differ": differ,
        "only_registry": only_registry,
        "only_local": only_local,
    }


def sha_tags(tags: dict[str, str], hashes: dict[str, str]) -> dict[str, bool]:
    """For every tag whose value is a SHA-256: does it equal one of the registry files' hashes?

    A False is not an error by itself: a tag can hash something else (the data manifest)."""
    known = set(hashes.values())
    out = {}
    for key, value in sorted(tags.items()):
        v = str(value).strip().lower()
        if SHA256.fullmatch(v):
            out[key] = v in known
    return out


def find_weights(hashes: dict[str, str]) -> str | None:
    """The adapter weights' path in the tree (top level or nested), or None."""
    matches = [k for k in hashes if k == WEIGHTS or k.endswith(f"/{WEIGHTS}")]
    return matches[0] if len(matches) == 1 else None


def fetch(name: str, version: str, local: Path, out_root: Path) -> dict:
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(tracking_uri())
    client = MlflowClient()
    mv = client.get_model_version(name, version)
    model = client.get_registered_model(name)

    dest = out_root / f"student-v{version}"
    if dest.exists():
        shutil.rmtree(dest)  # a rerun starts empty, so a stale file can't pass the check
    dest.mkdir(parents=True)
    uri = mlflow.artifacts.download_artifacts(artifact_uri=mv.source, dst_path=str(dest))
    downloaded = Path(uri)

    registry_hashes = tree_sha256(downloaded)
    local_hashes = tree_sha256(local)
    tags = {f"version.{k}": v for k, v in mv.tags.items()}
    tags |= {f"model.{k}": v for k, v in model.tags.items()}
    weights = find_weights(registry_hashes)
    weights_sha = registry_hashes.get(weights) if weights else None
    comparison = compare_trees(registry_hashes, local_hashes)
    return {
        "model": name,
        "version": version,
        "source": mv.source,
        "run_id": mv.run_id,
        "status": str(mv.status),
        "tags": dict(sorted(tags.items())),
        "registry_copy": downloaded.as_posix(),
        "local_adapter": local.as_posix(),
        "registry_files": registry_hashes,
        "comparison": comparison,
        "weights_file": weights,
        "weights_sha256": weights_sha,
        "weights_sha256_in_tags": weights_sha is not None
        and weights_sha in {str(v).strip().lower() for v in tags.values()},
        "sha256_tags_match_a_file": sha_tags(tags, registry_hashes),
        "ok": comparison["identical"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["fetch"])
    parser.add_argument("--name", default=MODEL_NAME)
    parser.add_argument("--version", default=VERSION)
    parser.add_argument("--local", type=Path, default=LOCAL_ADAPTER)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--bench", type=Path, default=Path("bench"))
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)

    body = fetch(args.name, args.version, args.local, args.out_root)
    result = {
        "command": " ".join(["python -m filinglens.serve.student", *argv]),
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        **body,
    }
    args.bench.mkdir(parents=True, exist_ok=True)
    out = args.bench / "student_fetch.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

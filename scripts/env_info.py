"""Versions and licenses for the environment records (scripts/fl_m5_env.sh).

Runs under any of the three Pythons (uv env, fl-train, fl-gpu) with the standard library only,
so the same file reports every venv:
    <python> scripts/env_info.py packages vllm torch ...
    <python> scripts/env_info.py models Qwen/Qwen3-1.7B@<sha> BAAI/bge-reranker-base@<sha>
`models` reads the local Hugging Face cache only (no network): the snapshot at the pinned
revision (`repo@sha`; FL-M3 loads by revision, which stores no `refs/main`), or the one `main`
points to when no revision is given, and the `license:` line of that model card's front matter.
"""

from __future__ import annotations

import importlib.metadata as md
import os
import sys
from pathlib import Path


def package_line(name: str) -> str:
    try:
        dist = md.distribution(name)
    except md.PackageNotFoundError:
        return f"{name} | not installed"
    meta = dist.metadata
    license_ = meta.get("License-Expression") or meta.get("License") or "-"
    license_ = " ".join(license_.split())[:60]
    all_classifiers = meta.get_all("Classifier") or []
    classifiers = [c.split(" :: ")[-1] for c in all_classifiers if c.startswith("License ::")]
    shown = ", ".join(classifiers) or "-"
    return f"{name} {dist.version} | license: {license_} | classifiers: {shown}"


def hub_cache() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    return Path(home) / "hub"


def card_license(readme: Path) -> str:
    """The `license:` value from a model card's YAML front matter, or '-'."""
    lines = readme.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines or lines[0].strip() != "---":
        return "-"
    for line in lines[1:]:
        if line.strip() == "---":
            break
        if line.startswith("license:"):
            return line.split(":", 1)[1].strip().strip("'\"") or "-"
    return "-"


def model_line(spec: str, cache: Path | None = None) -> str:
    repo_id, _, revision = spec.partition("@")
    folder = (cache or hub_cache()) / f"models--{repo_id.replace('/', '--')}"
    if not revision:
        ref = folder / "refs" / "main"
        if not ref.is_file():
            return f"{repo_id} | not in the local cache (no refs/main)"
        revision = ref.read_text().strip()
    if not (folder / "snapshots" / revision).is_dir():
        return f"{repo_id} revision {revision} | not in the local cache"
    readme = folder / "snapshots" / revision / "README.md"
    license_ = card_license(readme) if readme.is_file() else "- (no README in snapshot)"
    return f"{repo_id} revision {revision} | card license: {license_}"


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[0] not in {"packages", "models"}:
        print(__doc__, file=sys.stderr)
        return 2
    line = package_line if argv[0] == "packages" else model_line
    for name in argv[1:]:
        print(line(name))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

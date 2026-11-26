"""Start vLLM's OpenAI-compatible server on one FL-M5 build, from a .py with a __main__ guard (T5).

    ~/venvs/fl-gpu/bin/python -m filinglens.serve.vllm_server --build merged-bf16

The spec asks for vLLM to run from a .py file with an `if __name__ == "__main__":` guard (WSL uses
the spawn start method, which re-imports the main module in every worker). This wrapper gives every
GPU build the same fixed, recorded flags and then replaces itself with `vllm serve` (exec), so the
server is the only process left and Ctrl-C reaches it directly.

Flags: served model name = the build folder name; 127.0.0.1 only; max model length 4,608 (as the
evals: prompts up to 2,335 tokens + 400 generated); --max-num-seqs 16 (the highest benchmark level);
CUDA graphs on (vLLM's serving default; the accuracy runs used eager mode, which only changes speed,
and --eager reproduces that). gpu_memory_utilization is the same 0.83 as the evals.
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path

BUILD_DIR = Path("data/interim/serve")
PORT = 58000


def serve_argv(
    vllm: str,
    build: str,
    port: int = PORT,
    util: float = 0.83,
    max_len: int = 4608,
    max_seqs: int = 16,
    eager: bool = False,
) -> list[str]:
    path = BUILD_DIR / build
    argv = [
        vllm,
        "serve",
        path.as_posix(),
        "--served-model-name",
        build,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--gpu-memory-utilization",
        str(util),
        "--max-model-len",
        str(max_len),
        "--max-num-seqs",
        str(max_seqs),
        "--seed",
        "0",
    ]
    if eager:
        argv.append("--enforce-eager")
    return argv


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--build", required=True, help="a folder under data/interim/serve")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.83)
    parser.add_argument("--max-model-len", type=int, default=4608)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument(
        "--eager", action="store_true", help="no CUDA graphs (as the accuracy runs)"
    )
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    if not (BUILD_DIR / args.build / "config.json").is_file():
        raise SystemExit(f"no build at {BUILD_DIR / args.build}")
    vllm = str(Path(sys.executable).parent / "vllm")  # the vllm CLI of the venv running this
    command = serve_argv(
        vllm,
        args.build,
        args.port,
        args.gpu_memory_utilization,
        args.max_model_len,
        args.max_num_seqs,
        args.eager,
    )
    print("exec:", shlex.join(command), flush=True)
    os.execv(command[0], command)


if __name__ == "__main__":
    main()

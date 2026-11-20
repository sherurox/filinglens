"""Kaggle driver for the FL-M4 teacher run (T4): paste into one code cell of a Kaggle notebook.

Notebook settings (right-hand panel): Accelerator = GPU T4 x2, Internet = On, and add the private
dataset shreyask007/filinglens-teacher-inputs as input. COMMIT is filled in by the sed command
that copies this file to the clipboard, so the notebook installs exactly that repo commit.

MODE = "smoke": 200 pool rows on GPU 0 (run interactively, a few minutes of quota).
MODE = "full":  dev, test and the first N pool rows on GPU 0; the other half of the pool on
                GPU 1 (run with Save Version -> Save & Run All, so it keeps going in the background
                and its /kaggle/working output can be downloaded with the Kaggle CLI).
"""

import json
import subprocess
import sys
import time
from pathlib import Path

COMMIT = "main"
MODE = "smoke"
N = 200  # pool rows (a balanced prefix of pool.jsonl); for "full", set after the smoke test
MODEL = "Qwen/Qwen3-8B-AWQ"
OUT = Path("/kaggle/working/teacher")
PY = sys.executable


def sh(cmd: str) -> str:
    print("+", cmd, flush=True)
    done = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
    print(done.stdout[-4000:], done.stderr[-2000:], sep="\n", flush=True)
    return done.stdout


started = time.time()
OUT.mkdir(parents=True, exist_ok=True)
gpus = sh("nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader")
sh(f"{PY} -m pip install -q uv")
sh(f"{PY} -m uv pip install --python {PY} -q vllm --torch-backend=auto")
sh(f'{PY} -m pip install -q --no-deps "git+https://github.com/sherurox/filinglens@{COMMIT}"')
sh(f"{PY} -c \"from huggingface_hub import snapshot_download; snapshot_download('{MODEL}')\"")
versions = sh(f"{PY} -m pip list 2>/dev/null | grep -i -E '^(vllm|torch|transformers) '")

teacher = f"{PY} -m filinglens.train.kaggle_teacher --inputs /kaggle/input --out-dir {OUT}"
if MODE == "smoke":
    sh(f"CUDA_VISIBLE_DEVICES=0 {teacher} --split pool --limit {N}")
elif MODE == "full":
    gpu0 = " && ".join(
        [
            f"CUDA_VISIBLE_DEVICES=0 {teacher} --split dev",
            f"CUDA_VISIBLE_DEVICES=0 {teacher} --split test --allow-test",
            f"CUDA_VISIBLE_DEVICES=0 {teacher} --split pool --limit {N} --shard 0 --num-shards 2",
        ]
    )
    gpu1 = f"CUDA_VISIBLE_DEVICES=1 {teacher} --split pool --limit {N} --shard 1 --num-shards 2"
    procs = [
        subprocess.Popen(
            gpu0, shell=True, stdout=open(OUT / "gpu0.log", "w"), stderr=subprocess.STDOUT
        ),
        subprocess.Popen(
            gpu1, shell=True, stdout=open(OUT / "gpu1.log", "w"), stderr=subprocess.STDOUT
        ),
    ]
    codes = [p.wait() for p in procs]
    print("exit codes:", codes)
    if any(codes):
        print((OUT / "gpu0.log").read_text()[-3000:], (OUT / "gpu1.log").read_text()[-3000:])
        raise SystemExit(f"a worker failed: {codes}")
else:
    raise SystemExit(f"unknown MODE {MODE!r}")

driver = {
    "commit": COMMIT,
    "mode": MODE,
    "n": N,
    "model": MODEL,
    "gpus": gpus.strip().splitlines(),
    "versions": versions.strip().splitlines(),
    "wall_s_incl_install": round(time.time() - started, 1),
}
(OUT / "driver.json").write_text(json.dumps(driver, indent=2) + "\n")
print(json.dumps(driver, indent=2))
for run in sorted(OUT.glob("*_run.json")):
    print(run.name, run.read_text())

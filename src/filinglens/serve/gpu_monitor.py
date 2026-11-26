"""Sample GPU memory, power, temperature and utilization while a benchmark level runs (FL-M5 T5).

One `nvidia-smi --query-gpu=... -lms 500` process streams a CSV line every 0.5 s; a reader thread
parses them. In WSL, memory.used includes the Windows desktop (about 2 GB idle), so the record also
keeps the first sample: peak minus that baseline is what the server added. No nvidia-smi (a CPU-only
host) -> an empty summary, never an error.
"""

from __future__ import annotations

import shutil
import statistics
import subprocess
import threading
import time

FIELDS = ("memory.used", "power.draw", "temperature.gpu", "utilization.gpu")


def parse_gpu_line(line: str) -> dict[str, float] | None:
    """'2199, 18.36, 45.30, 3' -> floats; None for blank or unparsable lines ([N/A], headers)."""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != len(FIELDS):
        return None
    try:
        return {field: float(value) for field, value in zip(FIELDS, parts, strict=True)}
    except ValueError:
        return None


def summarize(samples: list[dict[str, float]]) -> dict:
    if not samples:
        return {"samples": 0}
    mem = [s["memory.used"] for s in samples]
    return {
        "samples": len(samples),
        "memory_used_mib_first": mem[0],
        "memory_used_mib_peak": max(mem),
        "memory_added_mib_peak": max(mem) - mem[0],
        "power_w_mean": round(statistics.fmean(s["power.draw"] for s in samples), 2),
        "power_w_max": max(s["power.draw"] for s in samples),
        "temperature_c_max": max(s["temperature.gpu"] for s in samples),
        "utilization_pct_mean": round(statistics.fmean(s["utilization.gpu"] for s in samples), 1),
    }


class GpuMonitor:
    """with GpuMonitor() as gpu: ...; gpu.summary() afterwards."""

    def __init__(self, interval_ms: int = 500) -> None:
        self.interval_ms = interval_ms
        self.samples: list[dict[str, float]] = []
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self) -> GpuMonitor:
        exe = shutil.which("nvidia-smi")
        if exe is None:
            return self
        query = f"--query-gpu={','.join(FIELDS)}"
        self._proc = subprocess.Popen(
            [exe, query, "--format=csv,noheader,nounits", f"-lms={self.interval_ms}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()
        time.sleep(self.interval_ms / 1000)  # one sample before the load starts: the baseline
        return self

    def _read(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            sample = parse_gpu_line(line)
            if sample is not None:
                self.samples.append(sample)

    def __exit__(self, *exc) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def summary(self) -> dict:
        return summarize(self.samples)

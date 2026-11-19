"""Table extraction with a small vision-language model (FL-M2 T10).

Qwen3-VL-2B-Instruct (Apache-2.0) reads each page image and returns the table as JSON rows
({"label", "values"}), constrained by a JSON schema so every answer parses. Two variants:
- image: the image alone;
- image+ocr: the image plus the Tesseract rows (T9, psm 4 chosen on dev) as a hint, so the model
  can fix OCR's mistakes instead of reading every digit itself.

This script only writes predictions; scoring runs in the main environment with the same metric as
OCR (`python -m filinglens.extract.tables score`). It imports only the standard library, Pillow and
vLLM, so it runs in the GPU venv:

    ~/venvs/fl-gpu/bin/python -m filinglens.extract.tables_vlm --variant image --split dev

Pages are cropped to their content and scaled to at most MAX_PIXELS before they're sent, which
bounds the number of image tokens (about one per 32 x 32 pixels for Qwen3-VL).
"""

import argparse
import base64
import io
import json
import math
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

TABLES = Path("gold/tables/tables.jsonl")
OCR_DIR = Path("data/interim/tables")
OCR_PSM = 4
MODEL = "Qwen/Qwen3-VL-2B-Instruct"
MAX_PIXELS = 1_600_000
MARGIN = 16

SCHEMA = {
    "type": "object",
    "properties": {
        "rows": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "values": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["label", "values"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["rows"],
    "additionalProperties": False,
}

# Prompt history (each version is run on dev only; the final one runs on test once):
# 1: rules only. Dev showed two structural failures: every cell emitted as its own row (a
#    16-row table became 48 rows), and the title and year headers turned into rows.
# 2: adds what a row is, what to leave out, and a worked example.
PROMPT_VERSION = 2
INSTRUCTIONS = (
    "Extract the table in this image as JSON rows, from top to bottom. One JSON row = one "
    "horizontal line of the table: its label and ALL the numbers on that line together. Never "
    "put a number in a row of its own, and never repeat the label as a value. Leave out the "
    "title or sentence above the table.\n"
    "label = the words at the start of the line (empty for a header line that has only numbers, "
    "such as years). values = every number on the line from left to right, copied exactly as "
    'printed, keeping parentheses, commas, decimal points, "%" and dashes ("—"), without "$". '
    "A line with words and no numbers gets an empty values list.\n"
    "Example. The lines\n"
    "  (In millions)        2025        2024\n"
    "  Net sales      $ 391,035   $ 383,285\n"
    "  Cost of sales   (210,352)   (214,137)\n"
    "  Operating expenses:\n"
    'become {"rows": [{"label": "(In millions)", "values": ["2025", "2024"]}, '
    '{"label": "Net sales", "values": ["391,035", "383,285"]}, '
    '{"label": "Cost of sales", "values": ["(210,352)", "(214,137)"]}, '
    '{"label": "Operating expenses:", "values": []}]}'
)


def load_pages(split: str, allow_test: bool, path: Path = TABLES) -> list[dict]:
    if split == "test" and not allow_test:
        raise PermissionError("the test pages are for final runs only: pass --allow-test")
    pages = [json.loads(line) for line in path.read_text().splitlines() if line]
    return [p for p in pages if p["split"] == split]


def ocr_hint(rows: list[dict]) -> str:
    lines = [" | ".join([r["label"], *r["values"]]) for r in rows]
    return (
        "\n\nOCR text of the same table (one line per row: label | values; it may contain "
        "errors, so trust the image when they disagree):\n" + "\n".join(lines)
    )


def prepare_image(path: Path, max_pixels: int = MAX_PIXELS) -> bytes:
    """PNG bytes of the page cropped to its non-white content and scaled to <= max_pixels."""
    from PIL import Image, ImageOps

    image = Image.open(path).convert("RGB")
    box = ImageOps.invert(image.convert("L")).point(lambda v: 255 if v > 16 else 0).getbbox()
    if box:
        left, top, right, bottom = box
        image = image.crop(
            (
                max(0, left - MARGIN),
                max(0, top - MARGIN),
                min(image.width, right + MARGIN),
                min(image.height, bottom + MARGIN),
            )
        )
    scale = math.sqrt(max_pixels / (image.width * image.height))
    if scale < 1:
        image = image.resize((int(image.width * scale), int(image.height * scale)), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def message(png: bytes, hint: str = "") -> list[dict]:
    url = "data:image/png;base64," + base64.b64encode(png).decode()
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": url}},
                {"type": "text", "text": INSTRUCTIONS + hint},
            ],
        }
    ]


def parse(raw: str, page_id: str) -> dict:
    try:
        rows = json.loads(raw)["rows"]
        return {"page_id": page_id, "valid_json": True, "rows": rows}
    except (json.JSONDecodeError, KeyError, TypeError):
        return {"page_id": page_id, "valid_json": False, "rows": [], "raw": raw[-300:]}


class PeakGpuMemory:
    """Peak used GPU memory sampled with NVML (vLLM's engine runs in another process)."""

    def __init__(self, interval: float = 0.5):
        self.peak_mib = None
        self._stop = threading.Event()
        self._interval = interval

    def __enter__(self):
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:  # no NVML: report null
            return self
        self.peak_mib = 0

        def sample():
            while not self._stop.is_set():
                used = pynvml.nvmlDeviceGetMemoryInfo(handle).used // 2**20
                self.peak_mib = max(self.peak_mib, used)
                self._stop.wait(self._interval)

        threading.Thread(target=sample, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        return False


def generate(conversations: list[list[dict]], args: argparse.Namespace) -> list[str]:
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    llm = LLM(
        model=args.model,
        quantization=args.quantization,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 1},
        seed=0,
    )
    params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        structured_outputs=StructuredOutputsParams(json=SCHEMA),
    )
    outputs = llm.chat(conversations, params)
    return [o.outputs[0].text for o in outputs]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", choices=["image", "image+ocr"], required=True)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--max-model-len", type=int, default=6144)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--out-dir", type=Path, default=OCR_DIR)
    parser.add_argument("--bench", type=Path, default=Path("bench"))
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)

    pages = load_pages(args.split, args.allow_test)
    hints = {}
    if args.variant == "image+ocr":
        ocr_file = args.out_dir / f"tables_ocr_{args.split}_psm{OCR_PSM}.jsonl"
        ocr = [json.loads(line) for line in ocr_file.read_text().splitlines() if line]
        hints = {o["page_id"]: ocr_hint(o["rows"]) for o in ocr}
    conversations = [
        message(prepare_image(Path(p["image"])), hints.get(p["page_id"], "")) for p in pages
    ]

    started = time.perf_counter()
    with PeakGpuMemory() as peak:
        raws = generate(conversations, args)
    elapsed = time.perf_counter() - started
    records = [parse(raw, p["page_id"]) for raw, p in zip(raws, pages, strict=True)]

    name = f"tables_vlm_{args.variant.replace('+', '_')}_{args.split}_p{PROMPT_VERSION}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{name}.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    run = {
        "command": " ".join(["python -m filinglens.extract.tables_vlm", *argv]),
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": args.model,
        "quantization": args.quantization,
        "variant": args.variant,
        "prompt_version": PROMPT_VERSION,
        "split": args.split,
        "pages": len(pages),
        "valid_json": sum(r["valid_json"] for r in records),
        "elapsed_s_incl_model_load": round(elapsed, 1),
        "peak_gpu_memory_mib": peak.peak_mib,
        "max_pixels": MAX_PIXELS,
        "predictions": str(out),
    }
    (args.bench / f"{name}_run.json").write_text(json.dumps(run, indent=2) + "\n")
    print(json.dumps(run, indent=2))


if __name__ == "__main__":
    main()

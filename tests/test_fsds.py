"""FSDS downloader tests: in-memory zips on a fake server, fake clock, files in tmp_path."""

import asyncio
import io
import random
import zipfile
from datetime import date

import httpx
import pytest
from fakes import FakeClock

from filinglens.ingest.edgar_client import EdgarClient
from filinglens.ingest.fsds import FSDS_URL, last_complete_quarter, quarters, run

UA = "FilingLens Test test@example.org"


def make_zip(members=("sub.txt", "num.txt", "tag.txt", "pre.txt")) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in members:
            zf.writestr(name, "adsh\tcik\n0000000001-24-000001\t1\n")
    return buffer.getvalue()


def make_client(bodies: dict[str, bytes | int]):
    """Fake SEC: `bodies` maps quarter -> zip bytes, or an HTTP status code to return."""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        calls.append(url)
        quarter = url.rsplit("/", 1)[-1].removesuffix(".zip")
        body = bodies.get(quarter, 404)
        if isinstance(body, int):
            return httpx.Response(body)
        return httpx.Response(200, content=body)

    client = EdgarClient(
        UA, clock=FakeClock(), transport=httpx.MockTransport(handler), rng=random.Random(0)
    )
    return client, calls


def run_once(client, wanted, root):
    async def go():
        async with client:
            return await run(client, wanted, root)

    return asyncio.run(go())


def test_quarters_cross_year_boundaries():
    assert quarters("2025q3", "2026q2") == ["2025q3", "2025q4", "2026q1", "2026q2"]
    assert len(quarters("2018q1", "2025q4")) == 32


@pytest.mark.parametrize(
    ("today", "expected"),
    [(date(2026, 11, 16), "2026q3"), (date(2026, 1, 5), "2025q4"), (date(2026, 4, 1), "2026q1")],
)
def test_last_complete_quarter(today, expected):
    assert last_complete_quarter(today) == expected


def test_download_verifies_and_rerun_fetches_nothing(tmp_path):
    client, calls = make_client({"2024q1": make_zip(), "2024q2": make_zip()})
    results = run_once(client, ["2024q1", "2024q2"], tmp_path)
    assert [r.status for r in results] == ["downloaded", "downloaded"]
    assert all(len(r.sha256) == 64 for r in results)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["2024q1.zip", "2024q2.zip"]

    client, calls = make_client({})
    results = run_once(client, ["2024q1", "2024q2"], tmp_path)
    assert [r.status for r in results] == ["skipped_existing", "skipped_existing"]
    assert calls == []


def test_unpublished_quarter_is_reported_not_failed(tmp_path):
    client, _ = make_client({"2026q2": make_zip()})  # 2026q3 -> 404
    results = run_once(client, ["2026q2", "2026q3"], tmp_path)
    assert [r.status for r in results] == ["downloaded", "not_published"]
    assert not (tmp_path / "2026q3.zip").exists()


@pytest.mark.parametrize(
    ("body", "problem"),
    [(b"this is not a zip", "BadZipFile"), (make_zip(("sub.txt", "num.txt")), "missing members")],
)
def test_bad_zip_is_rejected_and_leaves_nothing(tmp_path, body, problem):
    client, _ = make_client({"2024q1": body})
    (result,) = run_once(client, ["2024q1"], tmp_path)
    assert result.status == "failed"
    assert problem in result.error
    assert list(tmp_path.iterdir()) == []  # no zip, no .part


def test_url_format():
    assert FSDS_URL.format(quarter="2024q1") == (
        "https://www.sec.gov/files/dera/data/financial-statement-data-sets/2024q1.zip"
    )

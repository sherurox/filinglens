"""VLM table extraction: image preparation, prompt, parsing (no GPU needed)."""

import base64
import io
import json

import pytest
from PIL import Image

from filinglens.extract.tables_vlm import (
    SCHEMA,
    load_pages,
    message,
    ocr_hint,
    parse,
    prepare_image,
)


def test_prepare_image_crops_white_space_and_caps_pixels(tmp_path):
    page = Image.new("RGB", (1632, 2112), "white")
    page.paste(Image.new("RGB", (1500, 300), "black"), (60, 60))
    path = tmp_path / "page.png"
    page.save(path)
    small = Image.open(io.BytesIO(prepare_image(path, max_pixels=1_600_000)))
    assert small.width <= 1500 + 32 and small.height <= 300 + 32  # cropped to content + margin
    tiny = Image.open(io.BytesIO(prepare_image(path, max_pixels=100_000)))
    assert tiny.width * tiny.height <= 100_000


def test_message_has_image_and_instructions():
    content = message(b"png-bytes", ocr_hint([{"label": "Net sales", "values": ["391035"]}]))
    image, text = content[0]["content"]
    assert base64.b64decode(image["image_url"]["url"].split(",")[1]) == b"png-bytes"
    assert "Net sales | 391035" in text["text"] and "trust the image" in text["text"]


def test_parse_valid_and_cut_off_answers():
    ok = parse(json.dumps({"rows": [{"label": "A", "values": ["1"]}]}), "p1")
    assert ok == {"page_id": "p1", "valid_json": True, "rows": [{"label": "A", "values": ["1"]}]}
    bad = parse('{"rows": [{"label": "A", "val', "p2")
    assert not bad["valid_json"] and bad["rows"] == []


def test_schema_and_test_guard(tmp_path):
    assert SCHEMA["properties"]["rows"]["items"]["required"] == ["label", "values"]
    path = tmp_path / "t.jsonl"
    path.write_text(json.dumps({"page_id": "a", "split": "test"}) + "\n")
    with pytest.raises(PermissionError):
        load_pages("test", allow_test=False, path=path)

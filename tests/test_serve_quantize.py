import json

import pytest

from filinglens.serve.quantize import (
    BUILDS,
    NEEDS_DATA,
    answer,
    calibration_rows,
    ids_digest,
    quantization_summary,
    vram_cap_fraction,
)


def row(task_no, pid=None):
    return {
        "task_no": task_no,
        "para_id": pid or f"p{task_no}",
        "category": "liquidity",
        "severity": 2,
        "first_seen_year": 2021,
    }


def test_calibration_rows_first_n_by_task_number():
    train = [row(5), row(1), row(3), row(2)]
    assert [r["task_no"] for r in calibration_rows(train, 3)] == [1, 2, 3]


def test_calibration_rows_too_few_or_zero_raises():
    with pytest.raises(ValueError):
        calibration_rows([row(1)], 2)
    with pytest.raises(ValueError):
        calibration_rows([row(1)], 0)


def test_answer_has_the_eval_json_shape_in_order():
    text = answer(row(1, "p1"), {"p1": "because"})
    assert list(json.loads(text)) == [
        "rationale",
        "category",
        "severity",
        "first_seen_year",
        "unsure",
    ]
    assert json.loads(text)["rationale"] == "because"
    assert json.loads(answer(row(2), {}))["rationale"] == ""


def test_ids_digest_depends_on_order():
    assert ids_digest(["a", "b"]) != ids_digest(["b", "a"])
    assert len(ids_digest(["a"])) == 64


def test_quantization_summary_w4a16_and_fp8():
    w4 = {
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "format": "pack-quantized",
            "ignore": ["lm_head"],
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 4,
                        "type": "int",
                        "strategy": "group",
                        "group_size": 128,
                    },
                    "input_activations": None,
                }
            },
        }
    }
    s = quantization_summary(w4)
    assert s["format"] == "pack-quantized" and s["ignore"] == ["lm_head"]
    assert s["config_groups"]["group_0"]["weights"]["group_size"] == 128
    assert s["config_groups"]["group_0"]["input_activations"] is None

    fp8 = {
        "quantization_config": {
            "config_groups": {
                "g": {
                    "weights": {"num_bits": 8, "type": "float", "strategy": "channel"},
                    "input_activations": {"num_bits": 8, "type": "float", "dynamic": True},
                }
            }
        }
    }
    assert quantization_summary(fp8)["config_groups"]["g"]["input_activations"]["dynamic"] is True
    assert quantization_summary({})["config_groups"] == {}


def test_builds_and_calibration_needs():
    assert BUILDS == {
        "awq": "awq-w4a16",
        "gptq": "gptq-w4a16",
        "fp8": "fp8-dynamic",
        "rtn": "rtn-w4a16",
    }
    assert NEEDS_DATA == {"awq", "gptq"}


def test_vram_cap_is_free_minus_margin_over_total():
    assert vram_cap_fraction(6999, 8124, 256) == round(6743 / 8124, 4)


def test_vram_cap_refuses_when_nothing_is_free():
    with pytest.raises(ValueError):
        vram_cap_fraction(200, 8124, 256)
    with pytest.raises(ValueError):
        vram_cap_fraction(5000, 0, 256)

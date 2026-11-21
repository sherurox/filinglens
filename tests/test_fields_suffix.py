"""fields.output_suffix: FL-M2 baseline names unchanged; adapters and tags get a suffix."""

from argparse import Namespace
from pathlib import Path

from filinglens.extract.fields import output_suffix


def test_baseline_has_no_suffix() -> None:
    assert output_suffix(Namespace()) == ""  # an old-style args object without the new fields
    assert output_suffix(Namespace(lora=None, tag=None)) == ""


def test_lora_uses_the_run_name() -> None:
    lora = Path("data/interim/adapters/silver-r16-lr0.0002-e2/adapter")
    assert output_suffix(Namespace(lora=lora, tag=None)) == "_silver-r16-lr0.0002-e2"


def test_tag_wins() -> None:
    assert output_suffix(Namespace(lora=None, tag="regress")) == "_regress"
    assert output_suffix(Namespace(lora=Path("x/run/adapter"), tag="t")) == "_t"

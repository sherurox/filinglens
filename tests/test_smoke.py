"""Smoke tests: the package installs from src/ and the env template is complete."""

import sys
from pathlib import Path

import filinglens

ROOT = Path(__file__).resolve().parents[1]


def test_package_imports_from_src():
    assert Path(filinglens.__file__).parts[-3:] == ("src", "filinglens", "__init__.py")


def test_python_is_312():
    assert sys.version_info[:2] == (3, 12)


def test_env_example_lists_sec_user_agent():
    assert "SEC_USER_AGENT=" in (ROOT / ".env.example").read_text()

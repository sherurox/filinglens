import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "env_info.py"
spec = importlib.util.spec_from_file_location("env_info", SCRIPT)
env_info = importlib.util.module_from_spec(spec)
spec.loader.exec_module(env_info)


def fake_model(cache: Path, repo_id: str, revision: str, readme: str | None) -> None:
    folder = cache / f"models--{repo_id.replace('/', '--')}"
    (folder / "refs").mkdir(parents=True)
    (folder / "refs" / "main").write_text(revision + "\n")
    snap = folder / "snapshots" / revision
    snap.mkdir(parents=True)
    if readme is not None:
        (snap / "README.md").write_text(readme)


def test_model_line_reads_revision_and_card_license(tmp_path):
    fake_model(tmp_path, "Org/M", "abc123", "---\nlibrary_name: x\nlicense: apache-2.0\n---\n# M\n")
    expected = "Org/M revision abc123 | card license: apache-2.0"
    assert env_info.model_line("Org/M", tmp_path) == expected


def test_license_outside_front_matter_is_ignored(tmp_path):
    fake_model(tmp_path, "Org/M", "r1", "# M\nlicense: mit\n")
    assert env_info.model_line("Org/M", tmp_path).endswith("card license: -")


def test_model_not_cached(tmp_path):
    assert env_info.model_line("Org/None", tmp_path) == (
        "Org/None | not in the local cache (no refs/main)"
    )


def test_pinned_revision_without_refs_main(tmp_path):
    fake_model(tmp_path, "Org/M", "abc123", "---\nlicense: mit\n---\n")
    (tmp_path / "models--Org--M" / "refs" / "main").unlink()
    assert env_info.model_line("Org/M@abc123", tmp_path) == (
        "Org/M revision abc123 | card license: mit"
    )
    assert env_info.model_line("Org/M@other", tmp_path) == (
        "Org/M revision other | not in the local cache"
    )


def test_missing_package_is_reported_not_raised():
    assert env_info.package_line("surely-not-a-real-dist-xyz") == (
        "surely-not-a-real-dist-xyz | not installed"
    )


def test_installed_package_line_has_version():
    line = env_info.package_line("pytest")
    assert line.startswith("pytest ") and "| license:" in line

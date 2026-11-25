import hashlib

import pytest

from filinglens.serve.student import compare_trees, find_weights, sha_tags, tree_sha256


def write(path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def test_tree_sha256_relative_posix_paths_and_real_hashes(tmp_path):
    write(tmp_path / "adapter_config.json", b"{}")
    write(tmp_path / "sub" / "w.bin", b"abc")
    hashes = tree_sha256(tmp_path)
    assert list(hashes) == ["adapter_config.json", "sub/w.bin"]
    assert hashes["sub/w.bin"] == hashlib.sha256(b"abc").hexdigest()


def test_tree_sha256_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        tree_sha256(tmp_path / "nope")


def test_compare_identical():
    a = {"x": "1", "y": "2"}
    result = compare_trees(a, dict(a))
    assert result["identical"] and result["files_same"] == 2


def test_compare_reports_differ_and_one_sided_files():
    result = compare_trees({"x": "1", "y": "2", "r": "3"}, {"x": "1", "y": "9", "l": "4"})
    assert not result["identical"]
    assert result["differ"] == ["y"]
    assert result["only_registry"] == ["r"] and result["only_local"] == ["l"]
    assert result["files_same"] == 1


def test_compare_two_empty_trees_is_not_identical():
    assert not compare_trees({}, {})["identical"]


def test_sha_tags_only_looks_at_sha256_values():
    h = "a" * 64
    tags = {"version.adapter_sha256": h.upper(), "version.data_version": "b" * 64, "x": "r16"}
    assert sha_tags(tags, {"adapter_model.safetensors": h}) == {
        "version.adapter_sha256": True,
        "version.data_version": False,
    }


def test_find_weights_top_level_nested_or_ambiguous():
    assert find_weights({"adapter_model.safetensors": "h"}) == "adapter_model.safetensors"
    nested = "adapter/adapter_model.safetensors"
    assert find_weights({nested: "h"}) == nested
    two = {"a/adapter_model.safetensors": "h", "b/adapter_model.safetensors": "h"}
    assert find_weights(two) is None
    assert find_weights({"other.bin": "h"}) is None

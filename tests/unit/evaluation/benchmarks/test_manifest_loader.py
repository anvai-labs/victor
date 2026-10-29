"""Shared benchmark manifest loader (JSONL, list, and object manifests)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from victor.evaluation.benchmarks.manifest_loader import load_benchmark_manifest


def _load(path: Path):
    return load_benchmark_manifest(path, dataset_label="DS label", manifest_label="MF label")


def test_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    missing = tmp_path / "absent.json"
    with pytest.raises(FileNotFoundError, match=r"^Benchmark dataset not found: .*absent\.json$"):
        _load(missing)


def test_jsonl_skips_blank_lines_and_has_no_defaults(tmp_path: Path) -> None:
    path = tmp_path / "tasks.jsonl"
    path.write_text('{"id": "a"}\n\n   \n{"id": "b"}\n')
    assert _load(path) == ([{"id": "a"}, {"id": "b"}], None, {})


def test_json_list_has_no_defaults(tmp_path: Path) -> None:
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps([{"id": "a"}]))
    assert _load(path) == ([{"id": "a"}], None, {})


def test_object_manifest_returns_tasks_defaults_and_metadata(tmp_path: Path) -> None:
    path = tmp_path / "m.json"
    path.write_text(
        json.dumps({"tasks": [{"id": "t"}], "defaults": {"x": 1}, "metadata": {"name": "n"}})
    )
    assert _load(path) == ([{"id": "t"}], {"x": 1}, {"name": "n"})


def test_object_manifest_null_or_absent_keys_default_to_empty(tmp_path: Path) -> None:
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"defaults": None, "metadata": None}))
    # Object manifests always carry defaults (empty), distinguishing them from lists.
    assert _load(path) == ([], {}, {})


@pytest.mark.parametrize(
    "payload,message",
    [
        ("42", "DS label dataset payload must be a list or object manifest"),
        ('{"tasks": {}}', "MF label manifest 'tasks' must be a list"),
        ('{"defaults": [1]}', "MF label manifest 'defaults' must be an object"),
        ('{"metadata": "x"}', "MF label manifest 'metadata' must be an object"),
    ],
)
def test_malformed_manifest_messages_use_the_given_labels(
    tmp_path: Path, payload: str, message: str
) -> None:
    path = tmp_path / "m.json"
    path.write_text(payload)
    with pytest.raises(ValueError) as excinfo:
        _load(path)
    assert str(excinfo.value) == message


def test_mixin_load_manifest_uses_runner_labels_and_sets_defaults(tmp_path: Path) -> None:
    from victor.evaluation.benchmarks.manifest_loader import ManifestBenchmarkMixin

    class Runner(ManifestBenchmarkMixin):
        _manifest_dataset_label = "R data"
        _manifest_label = "R"

        def __init__(self) -> None:
            self._defaults = {"untouched": True}

        def _build_manifest_metadata(self, metadata, records):  # type: ignore[no-untyped-def]
            return ("meta", metadata, len(records))

    obj = tmp_path / "o.json"
    obj.write_text(json.dumps({"tasks": [{"id": 1}], "defaults": {"d": 1}, "metadata": {"m": 1}}))
    runner = Runner()
    assert runner._load_manifest(obj) == ([{"id": 1}], ("meta", {"m": 1}, 1))
    assert runner._defaults == {"d": 1}

    lst = tmp_path / "l.json"
    lst.write_text("[]")
    list_runner = Runner()
    assert list_runner._load_manifest(lst) == ([], ("meta", {}, 0))
    assert list_runner._defaults == {"untouched": True}

    bad = tmp_path / "b.json"
    bad.write_text('{"tasks": 1}')
    with pytest.raises(ValueError, match=r"^R manifest 'tasks' must be a list$"):
        Runner()._load_manifest(bad)
    bad.write_text("1")
    with pytest.raises(ValueError, match=r"^R data dataset payload"):
        Runner()._load_manifest(bad)

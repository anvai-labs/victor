"""Browser-task runner manifest loading: byte-identical error text and defaults."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from victor.evaluation.benchmarks.browser_tasks import BrowserTaskBenchmarkRunner
from victor.evaluation.benchmarks.deep_research import DeepResearchBenchmarkRunner
from victor.evaluation.protocol import BenchmarkType


def _browser(path: Path) -> BrowserTaskBenchmarkRunner:
    return BrowserTaskBenchmarkRunner(BenchmarkType.CLAW_BENCH, path)


@pytest.mark.parametrize(
    "payload,message",
    [
        ("42", "Browser benchmark dataset payload must be a list or object manifest"),
        ('{"tasks": {}}', "Browser benchmark manifest 'tasks' must be a list"),
        ('{"defaults": [1]}', "Browser benchmark manifest 'defaults' must be an object"),
        ('{"metadata": "x"}', "Browser benchmark manifest 'metadata' must be an object"),
    ],
)
def test_browser_manifest_errors_keep_their_wording(
    tmp_path: Path, payload: str, message: str
) -> None:
    path = tmp_path / "m.json"
    path.write_text(payload)
    with pytest.raises(ValueError) as excinfo:
        _browser(path)
    assert str(excinfo.value) == message


@pytest.mark.parametrize(
    "payload,message",
    [
        ("42", "Deep research benchmark dataset payload must be a list or object manifest"),
        ('{"tasks": {}}', "Deep research manifest 'tasks' must be a list"),
        ('{"defaults": [1]}', "Deep research manifest 'defaults' must be an object"),
        ('{"metadata": "x"}', "Deep research manifest 'metadata' must be an object"),
    ],
)
def test_deep_research_manifest_errors_keep_their_wording(
    tmp_path: Path, payload: str, message: str
) -> None:
    path = tmp_path / "m.json"
    path.write_text(payload)
    with pytest.raises(ValueError) as excinfo:
        DeepResearchBenchmarkRunner(path)
    assert str(excinfo.value) == message


def test_object_manifest_sets_defaults_and_list_manifest_keeps_them_empty(
    tmp_path: Path,
) -> None:
    obj = tmp_path / "obj.json"
    obj.write_text(json.dumps({"tasks": [{"task_id": "t1"}], "defaults": {"hints": ["h"]}}))
    runner = _browser(obj)
    assert runner._defaults == {"hints": ["h"]}
    assert runner._records == [{"task_id": "t1"}]

    lst = tmp_path / "list.json"
    lst.write_text(json.dumps([{"task_id": "t2"}]))
    list_runner = _browser(lst)
    assert list_runner._defaults == {}
    assert list_runner._records == [{"task_id": "t2"}]


def test_missing_dataset_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _browser(tmp_path / "nope.jsonl")

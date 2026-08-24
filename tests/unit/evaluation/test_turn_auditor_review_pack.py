# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

from victor.evaluation.turn_auditor_review_pack import (
    ManifestSource,
    build_review_pack,
    finalize_review_pack,
)


def _record(task_id: str = "task-1", *, benchmark: str = "") -> dict:
    return {
        "task_id": task_id,
        "session_id": f"session-{task_id}",
        "status": "failed",
        "reward": "partial",
        "tests_passed": 1,
        "tests_total": 2,
        "trace": {
            "benchmark": benchmark,
            "turns": 2,
            "messages": [
                {"role": "user", "content": "Fix the failing test"},
                {"role": "assistant", "content": "I changed the implementation."},
            ],
            "tool_calls": [
                {
                    "name": "read_file",
                    "arguments": {"path": "module.py"},
                    "result": "source",
                    "success": True,
                },
                {
                    "name": "shell",
                    "arguments": {"command": "pytest"},
                    "result": "1 failed",
                    "success": False,
                },
            ],
        },
    }


def _manifest(path: Path, *records: dict) -> Path:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


def test_build_review_pack_normalizes_and_records_provenance(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "eval_manifest_run.jsonl", _record())

    pack = build_review_pack([ManifestSource(manifest, "code-fix")])

    assert pack["kind"] == "evr6_review_pack"
    assert pack["selection"]["emitted_cases"] == 1
    case = pack["cases"][0]
    assert case["task_id"] == "code-fix:task-1"
    assert case["family"] == "code-fix"
    assert case["review"] == {
        "status": "pending",
        "oracle_alarm_step": None,
        "oracle_source": "",
        "notes": "",
    }
    assert [step["index"] for step in case["trace"]["steps"]] == [0, 1, 2]
    assert case["trace"]["metadata"]["source"]["line"] == 1
    assert len(case["trace"]["metadata"]["source"]["manifest_sha256"]) == 64
    assert case["trace"]["metadata"]["outcome"]["reward"] == "partial"
    assert case["review_context"]["tool_steps"][1]["result"] == "1 failed"
    assert "auditor_id" not in pack
    assert "observations" not in case


def test_build_review_pack_derives_family_and_can_omit_context(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "manifest.jsonl", _record(benchmark="repair"))

    pack = build_review_pack([ManifestSource(manifest)], include_context=False)

    assert pack["cases"][0]["family"] == "repair"
    assert "review_context" not in pack["cases"][0]


def test_build_review_pack_requires_real_family(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "manifest.jsonl", _record())

    with pytest.raises(ValueError, match="use --source FAMILY=PATH"):
        build_review_pack([ManifestSource(manifest)])


def test_build_review_pack_skips_empty_and_duplicate_tasks(tmp_path: Path) -> None:
    empty = _record("empty")
    empty["trace"] = {"messages": [], "tool_calls": []}
    manifest = _manifest(tmp_path / "manifest.jsonl", _record("same"), _record("same"), empty)

    pack = build_review_pack([ManifestSource(manifest, "family")])

    assert len(pack["cases"]) == 1
    assert pack["selection"]["skipped"] == {
        "missing_task_id": 0,
        "empty_trace": 1,
        "duplicate_task_id": 1,
    }


def test_build_review_pack_rejects_malformed_json(tmp_path: Path) -> None:
    manifest = tmp_path / "bad.jsonl"
    manifest.write_text("{bad}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="bad.jsonl:1: invalid JSON"):
        build_review_pack([ManifestSource(manifest, "family")])


def test_finalize_requires_explicit_disposition(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "manifest.jsonl", _record())
    pack = build_review_pack([ManifestSource(manifest, "code-fix")])

    with pytest.raises(ValueError, match="status must be included or excluded"):
        finalize_review_pack(pack)


def test_finalize_emits_only_included_label_cases(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "manifest.jsonl", _record("positive"), _record("healthy"))
    pack = build_review_pack([ManifestSource(manifest, "code-fix")])
    pack["cases"][0]["review"].update(
        {
            "status": "included",
            "oracle_alarm_step": 1,
            "oracle_source": "human-review:alice@rev1",
        }
    )
    pack["cases"][1]["review"]["status"] = "excluded"

    labels = finalize_review_pack(pack)

    assert labels["label_provenance"]["included_cases"] == 1
    assert labels["label_provenance"]["excluded_cases"] == 1
    assert len(labels["label_provenance"]["review_pack_sha256"]) == 64
    assert labels["cases"] == [
        {
            "task_id": "code-fix:positive",
            "family": "code-fix",
            "oracle_source": "human-review:alice@rev1",
            "oracle_alarm_step": 1,
            "trace": pack["cases"][0]["trace"],
        }
    ]
    assert "review" not in labels["cases"][0]
    assert "review_context" not in labels["cases"][0]
    assert "observations" not in labels["cases"][0]


@pytest.mark.parametrize("alarm_step", [True, "1", 99])
def test_finalize_rejects_invalid_alarm_step(tmp_path: Path, alarm_step: object) -> None:
    manifest = _manifest(tmp_path / "manifest.jsonl", _record())
    pack = build_review_pack([ManifestSource(manifest, "code-fix")])
    pack["cases"][0]["review"].update(
        {
            "status": "included",
            "oracle_alarm_step": alarm_step,
            "oracle_source": "review@rev1",
        }
    )

    with pytest.raises(ValueError, match="oracle_alarm_step"):
        finalize_review_pack(pack)


def test_finalize_rejects_auditor_output_and_missing_oracle_source(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "manifest.jsonl", _record())
    pack = build_review_pack([ManifestSource(manifest, "code-fix")])
    pack["cases"][0]["review"]["status"] = "included"

    with pytest.raises(ValueError, match="missing oracle_source"):
        finalize_review_pack(pack)

    pack["cases"][0]["review"]["oracle_source"] = "review@rev1"
    pack["cases"][0]["observations"] = []
    with pytest.raises(ValueError, match="must not contain model observations"):
        finalize_review_pack(pack)

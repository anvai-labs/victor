"""Tests for the SWE-bench trajectory → judge-stratum converter."""

import json
from pathlib import Path

from victor.evaluation.swe_bench_stratum import (
    manifest_to_stratum,
    render_swe_bench_view,
    write_stratum_jsonl,
)


def _instance(task_id, status, *, issue="Fix the bug", final="Done.", edits=None, tools=None):
    return {
        "task_id": task_id,
        "status": status,
        "tests_passed": 15 if status == "passed" else 10,
        "tests_total": 15,
        "trace": {
            "messages": [
                {"role": "user", "content": issue},
                {"role": "assistant", "content": final},
            ],
            "tool_calls": tools or [{"name": "edit", "arguments": "{'path': 'a.py'}"}],
            "file_edits": edits or [{"path": "a.py", "diff": "- old\n+ new"}],
            "turns": 1,
        },
    }


class TestRender:
    def test_render_has_sections_and_patch(self):
        text = render_swe_bench_view(_instance("astropy__1", "failed"))
        assert "TASK:\nFix the bug" in text
        assert "PATCH:" in text
        assert "--- a.py" in text and "+ new" in text
        assert "TOOL ACTIVITY:" in text

    def test_render_is_blinded_no_status(self):
        text = render_swe_bench_view(_instance("t", "passed")).lower()
        assert "status" not in text
        assert "passed" not in text
        assert "tests_passed" not in text

    def test_render_handles_missing_trace(self):
        text = render_swe_bench_view({"task_id": "t", "status": "failed"})
        assert "(no file edits)" in text
        assert "(none)" in text


class TestManifestToStratum:
    def test_labels_from_verifier_status(self, tmp_path: Path):
        p = tmp_path / "eval_manifest_x.jsonl"
        p.write_text(
            "\n".join(
                json.dumps(i)
                for i in [
                    _instance("resolved-1", "passed"),
                    _instance("failed-1", "failed"),
                    _instance("failed-2", "failed"),
                ]
            )
        )
        stratum = manifest_to_stratum(p)
        labels = {e.task_id: e.label for e in stratum}
        assert labels == {"resolved-1": 1, "failed-1": 0, "failed-2": 0}
        assert all(e.family == "swe-bench" for e in stratum)

    def test_skips_error_and_running_instances(self, tmp_path: Path):
        p = tmp_path / "m.jsonl"
        p.write_text(
            "\n".join(
                json.dumps(i)
                for i in [
                    _instance("ok", "failed"),
                    _instance("errored", "error"),
                    _instance("mid", "running"),
                ]
            )
        )
        stratum = manifest_to_stratum(p)
        assert [e.task_id for e in stratum] == ["ok"]

    def test_jsonl_round_trip(self, tmp_path: Path):
        p = tmp_path / "m.jsonl"
        p.write_text(json.dumps(_instance("t", "passed")))
        stratum = manifest_to_stratum(p)
        out = tmp_path / "stratum.jsonl"
        assert write_stratum_jsonl(stratum, out) == 1
        row = json.loads(out.read_text().splitlines()[0])
        assert row["label"] == 1 and row["source"] == "swe-bench"
        assert "PATCH:" in row["text"]

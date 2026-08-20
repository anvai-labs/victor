"""Tests for the SWE-bench trajectory → judge-stratum converter."""

import json
from pathlib import Path

from victor.evaluation.swe_bench_stratum import (
    StratumExample,
    gate_stratum,
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

    def test_render_prefers_generated_patch_over_file_edits(self):
        # The ground-truth combined diff (git diff) is the completing effect; it
        # must win over the per-edit capture, which is often empty even when the
        # run modified files. Here file_edits is empty but a patch was applied.
        inst = _instance("astropy__2", "passed")
        inst["trace"]["file_edits"] = []
        inst["trace"][
            "generated_patch"
        ] = "diff --git a/mod.py b/mod.py\n@@ -1 +1 @@\n-return None\n+return handle(x)"
        text = render_swe_bench_view(inst)
        assert "PATCH:" in text
        assert "diff --git a/mod.py" in text
        assert "+return handle(x)" in text
        assert "(no file edits)" not in text


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


class TestGateStratum:
    @staticmethod
    def _examples(labels):
        # Encode each example's gold label into its text so a stub judge can be
        # deterministic without an LLM.
        return [
            StratumExample(task_id=f"t{i}", family="swe-bench", text=f"gold={g}", label=g)
            for i, g in enumerate(labels)
        ]

    def test_perfect_judge_scores_alpha_one(self):
        ex = self._examples([1, 0, 1, 0, 0])
        res = gate_stratum(ex, lambda t: 1 if "gold=1" in t else 0, judge_name="oracle")
        assert res.n == 5 and res.n_pos == 2 and res.n_neg == 3
        assert res.agree == 5 and res.false_pos == 0 and res.false_neg == 0
        assert res.krippendorff_alpha == 1.0

    def test_constant_yes_judge_is_all_false_positive(self):
        # The gemma-on-real-SWE-bench failure mode: credit everything complete.
        ex = self._examples([1, 0, 0, 0, 0])
        res = gate_stratum(ex, lambda t: 1, judge_name="constant-yes")
        assert res.true_pos == 1 and res.false_pos == 4
        assert res.true_neg == 0 and res.false_neg == 0
        # No discrimination → α at or below zero.
        assert res.krippendorff_alpha is not None and res.krippendorff_alpha <= 0.0

    def test_result_dict_round_trips(self):
        res = gate_stratum(self._examples([1, 0]), lambda t: 0, judge_name="j")
        d = res.to_dict()
        assert d["judge"] == "j" and d["n"] == 2
        assert {"true_pos", "false_pos", "true_neg", "false_neg", "krippendorff_alpha"} <= set(d)

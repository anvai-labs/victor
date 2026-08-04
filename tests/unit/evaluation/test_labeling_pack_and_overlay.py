"""Tests for the EVR-2 labeling-pack export and human-label overlay (ADR-011).

The two contracts that matter:
1. The exported pack is BLINDED — no gold, no verifier output, no judge scores.
2. The overlay math is correct and the pre-registered thresholds behave:
   verifier-κ failure is stop-the-line; per-family α below n=16 is directional
   only; partial label files are rejected, not silently shrunk.
"""

import json
from pathlib import Path

import pytest

from victor.evaluation.calibration_corpus import default_corpus
from victor.evaluation.human_label_overlay import (
    HUMAN_JUDGE_ALPHA_THRESHOLD,
    HUMAN_VERIFIER_KAPPA_THRESHOLD,
    LabelsError,
    compute_overlay,
    load_labels,
    load_report_samples,
)
from victor.evaluation.judge_calibration_harness import (
    JudgeCalibrationHarness,
    alternating_scripted_executor,
)
from victor.evaluation.labeling_pack import make_labeling_pack_sink


def _run_with_pack(tmp_path: Path, variants: int = 2):
    pack_dir = tmp_path / "pack"
    harness = JudgeCalibrationHarness(default_corpus(variants=variants))
    reports = harness.run_multi_judge(
        alternating_scripted_executor(period=2),
        {"activity": lambda _p, tr, _w: 1.0 if tr.tool_steps() else 0.0},
        record_sink=make_labeling_pack_sink(pack_dir),
    )
    return pack_dir, reports["activity"]


class TestLabelingPackExport:
    def test_pack_has_one_record_per_task_plus_manifest(self, tmp_path: Path):
        pack_dir, report = _run_with_pack(tmp_path)
        records = sorted(pack_dir.glob("[0-9]*.json"))
        assert len(records) == len(report.samples)
        assert (pack_dir / "README.md").exists()
        template = (pack_dir / "labels.template.jsonl").read_text().splitlines()
        assert len(template) == len(report.samples)
        assert {json.loads(line)["task_id"] for line in template} == {
            s.task_id for s in report.samples
        }
        assert all(json.loads(line)["label"] is None for line in template)

    def test_records_are_blinded_and_contain_the_judge_view(self, tmp_path: Path):
        pack_dir, _ = _run_with_pack(tmp_path)
        for path in pack_dir.glob("[0-9]*.json"):
            record = json.loads(path.read_text())
            # The judge's view is present…
            assert record["prompt"]
            assert "final_message" in record["transcript"]
            assert isinstance(record["workspace"]["files"], dict)
            # …and the blinding contract holds: no verifier or judge output.
            assert "gold" not in record
            assert "judged" not in record
            assert "verify" not in record

    def test_manifest_survives_partial_runs(self, tmp_path: Path):
        # The sink rewrites the manifest per record, so a run that dies partway
        # still leaves a labelable pack.
        pack_dir = tmp_path / "pack"
        sink = make_labeling_pack_sink(pack_dir)
        corpus = default_corpus(variants=1)
        harness = JudgeCalibrationHarness(corpus[:2])
        harness.run_multi_judge(
            alternating_scripted_executor(period=2),
            {"j": lambda _p, _t, _w: 1.0},
            record_sink=sink,
        )
        template = (pack_dir / "labels.template.jsonl").read_text().splitlines()
        assert len(template) == 2


class TestLoadLabels:
    def _write(self, tmp_path: Path, lines) -> Path:
        p = tmp_path / "labels.jsonl"
        p.write_text("\n".join(json.dumps(row) for row in lines) + "\n")
        return p

    def test_round_trip(self, tmp_path: Path):
        p = self._write(
            tmp_path,
            [
                {"task_id": "a", "label": 1, "annotator": "vj"},
                {"task_id": "b", "label": 0, "annotator": "vj", "rationale": "no effect"},
            ],
        )
        assert load_labels(p) == {"a": 1.0, "b": 0.0}

    def test_unlabeled_rows_are_an_error_not_a_skip(self, tmp_path: Path):
        p = self._write(tmp_path, [{"task_id": "a", "label": 1}, {"task_id": "b", "label": None}])
        with pytest.raises(LabelsError, match="unlabeled"):
            load_labels(p)

    def test_duplicate_and_invalid_labels_rejected(self, tmp_path: Path):
        dup = self._write(tmp_path, [{"task_id": "a", "label": 1}, {"task_id": "a", "label": 0}])
        with pytest.raises(LabelsError, match="duplicate"):
            load_labels(dup)
        bad = self._write(tmp_path, [{"task_id": "a", "label": 0.5}])
        with pytest.raises(LabelsError, match="label must be 0 or 1"):
            load_labels(bad)


def _samples(rows):
    return [
        {"task_id": tid, "family": family, "gold": gold, "judged": judged}
        for tid, family, gold, judged in rows
    ]


class TestComputeOverlay:
    def test_perfect_agreement_passes_everything(self):
        # 20 tasks in one family (>= MIN_FAMILY_N), human == verifier == judge.
        rows = [(f"t{i}", "qa", float(i % 2), float(i % 2)) for i in range(20)]
        overlay = compute_overlay(
            {"rubric-llm": _samples(rows)}, {f"t{i}": float(i % 2) for i in range(20)}
        )
        assert not overlay.stop_the_line
        assert all(v.passed for v in overlay.verdicts)
        assert overlay.human_vs_verifier.cohens_kappa == 1.0
        assert overlay.human_vs_judge["rubric-llm"].krippendorff_alpha == 1.0

    def test_verifier_disagreement_is_stop_the_line(self):
        # Human disagrees with verifier gold on 8/20 → κ well below 0.8.
        rows = [(f"t{i}", "qa", float(i % 2), float(i % 2)) for i in range(20)]
        human = {f"t{i}": (1.0 - (i % 2) if i < 8 else float(i % 2)) for i in range(20)}
        overlay = compute_overlay({"j": _samples(rows)}, human)
        assert overlay.stop_the_line
        verifier_verdict = next(v for v in overlay.verdicts if v.name == "human_vs_verifier_kappa")
        assert not verifier_verdict.passed
        assert "STOP THE LINE" in verifier_verdict.detail

    def test_judge_below_alpha_fails_without_stopping_the_line(self):
        # Human == verifier (κ=1) but the judge answers a constant 1.0 → α ≤ 0.
        rows = [(f"t{i}", "qa", float(i % 2), 1.0) for i in range(20)]
        human = {f"t{i}": float(i % 2) for i in range(20)}
        overlay = compute_overlay({"credulous": _samples(rows)}, human)
        assert not overlay.stop_the_line
        judge_verdict = next(v for v in overlay.verdicts if "credulous" in v.name)
        assert not judge_verdict.passed

    def test_thin_families_are_directional_not_gating(self):
        # 4 items in a family (< 16): even α=0 there must not fail the verdict
        # when the big family passes — it is reported as directional.
        rows = [(f"big{i}", "qa", float(i % 2), float(i % 2)) for i in range(20)]
        rows += [(f"thin{i}", "refactor", float(i % 2), 1.0) for i in range(4)]
        human = {tid: gold for tid, _f, gold, _j in rows}
        overlay = compute_overlay({"j": _samples(rows)}, human)
        judge_verdict = next(v for v in overlay.verdicts if v.name == "human_vs_judge_alpha[j]")
        assert judge_verdict.passed
        assert "directional" in judge_verdict.detail

    def test_secondary_labels_reported_never_gating(self):
        rows = [(f"t{i}", "qa", float(i % 2), float(i % 2)) for i in range(20)]
        human = {f"t{i}": float(i % 2) for i in range(20)}
        secondary = dict(human)
        secondary["t0"] = 1.0 - secondary["t0"]  # one disagreement
        overlay = compute_overlay({"j": _samples(rows)}, human, secondary_labels=secondary)
        qc = next(v for v in overlay.verdicts if v.name == "human_vs_secondary_kappa")
        assert qc.passed  # QC is informational even with disagreement
        assert overlay.human_vs_secondary is not None
        assert overlay.human_vs_secondary.cohens_kappa < 1.0

    def test_no_matching_task_ids_is_an_error(self):
        rows = [("t1", "qa", 1.0, 1.0)]
        with pytest.raises(LabelsError, match="no task_ids"):
            compute_overlay({"j": _samples(rows)}, {"other": 1.0})

    def test_report_roundtrip_and_thresholds_recorded(self, tmp_path: Path):
        rows = [(f"t{i}", "qa", float(i % 2), float(i % 2)) for i in range(20)]
        overlay = compute_overlay({"j": _samples(rows)}, {f"t{i}": float(i % 2) for i in range(20)})
        out = tmp_path / "human_overlay.json"
        overlay.save(out)
        data = json.loads(out.read_text())
        assert data["thresholds"]["human_judge_alpha"] == HUMAN_JUDGE_ALPHA_THRESHOLD
        assert data["thresholds"]["human_verifier_kappa"] == HUMAN_VERIFIER_KAPPA_THRESHOLD
        assert data["stop_the_line"] is False


class TestLoadReportSamples:
    def test_reads_saved_calibration_report(self, tmp_path: Path):
        _, report = _run_with_pack(tmp_path)
        path = tmp_path / "activity.json"
        report.save(path)
        samples = load_report_samples(path)
        assert len(samples) == len(report.samples)
        assert {"task_id", "family", "gold", "judged"} <= set(samples[0])

    def test_empty_report_rejected(self, tmp_path: Path):
        p = tmp_path / "r.json"
        p.write_text(json.dumps({"samples": []}))
        with pytest.raises(LabelsError):
            load_report_samples(p)


class TestPackReplayExecutor:
    def test_replay_reproduces_gold_and_transcripts(self, tmp_path: Path):
        from victor.evaluation.labeling_pack import make_pack_replay_executor

        # Original run: scripted executor + pack export.
        pack_dir = tmp_path / "pack"
        corpus = default_corpus(variants=2)
        harness = JudgeCalibrationHarness(corpus)
        judge = {"j": lambda _p, tr, _w: 1.0 if tr.tool_steps() else 0.0}
        original = harness.run_multi_judge(
            alternating_scripted_executor(period=2),
            judge,
            record_sink=make_labeling_pack_sink(pack_dir),
        )["j"]

        # Replay: same corpus, executor restores pack snapshots.
        replayed = JudgeCalibrationHarness(default_corpus(variants=2)).run_multi_judge(
            make_pack_replay_executor(pack_dir), judge
        )["j"]

        assert [(s.task_id, s.gold) for s in replayed.samples] == [
            (s.task_id, s.gold) for s in original.samples
        ]
        assert [(s.judged) for s in replayed.samples] == [(s.judged) for s in original.samples]

    def test_replay_rejects_corpus_mismatch(self, tmp_path: Path):
        from victor.evaluation.labeling_pack import make_pack_replay_executor

        pack_dir = tmp_path / "pack"
        harness = JudgeCalibrationHarness(default_corpus(variants=1))
        harness.run_multi_judge(
            alternating_scripted_executor(period=2),
            {"j": lambda _p, _t, _w: 1.0},
            record_sink=make_labeling_pack_sink(pack_dir),
        )
        bigger = JudgeCalibrationHarness(default_corpus(variants=2))
        with pytest.raises(KeyError, match="corpus/pack mismatch"):
            bigger.run_multi_judge(make_pack_replay_executor(pack_dir), {"j": lambda *_: 1.0})

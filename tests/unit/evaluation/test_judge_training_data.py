"""Tests for the judge-training dataset generator (E2, PR-J1).

Contracts: labels equal the programmatic verifier's outputs; the rendered
text is blinded (no gold/verifier content); generation is deterministic;
the variant split keeps (family, variant) groups on one side.
"""

from pathlib import Path

from victor.evaluation.calibration_corpus import default_corpus
from victor.evaluation.judge_calibration_harness import (
    JudgeCalibrationHarness,
    alternating_scripted_executor,
    hard_scripted_executor,
)
from victor.evaluation.judge_training_data import (
    generate_training_examples,
    load_jsonl,
    split_by_variant,
    write_jsonl,
)


class TestGeneration:
    def test_labels_equal_verifier_gold(self):
        # Same corpus + executor sequence through the harness produces the
        # same gold the generator emits as labels.
        corpus = default_corpus(variants=2)
        examples = generate_training_examples(
            corpus, alternating_scripted_executor(period=2), source="easy-p2"
        )
        report = JudgeCalibrationHarness(default_corpus(variants=2)).run(
            alternating_scripted_executor(period=2), lambda *_: 1.0
        )
        assert [(e.task_id, float(e.label)) for e in examples] == [
            (s.task_id, s.gold) for s in report.samples
        ]

    def test_hard_executor_contributes_flawed_negatives(self):
        examples = generate_training_examples(
            default_corpus(variants=5), hard_scripted_executor(), source="hard"
        )
        labels = {e.label for e in examples}
        assert labels == {0, 1}
        # Flawed cases: negative label WITH tool activity rendered in the text.
        flawed = [e for e in examples if e.label == 0 and "TOOL ACTIVITY:\n- edit" in e.text]
        assert flawed, "hard executor must produce looks-solved-but-wrong negatives"

    def test_text_is_blinded(self):
        examples = generate_training_examples(
            default_corpus(variants=1), alternating_scripted_executor(period=2), source="s"
        )
        for e in examples:
            lowered = e.text.lower()
            assert "gold" not in lowered
            assert "verifier" not in lowered
            assert "label" not in lowered

    def test_deterministic(self):
        args = (default_corpus(variants=2), alternating_scripted_executor(period=2))
        a = generate_training_examples(*args, source="s")
        b = generate_training_examples(
            default_corpus(variants=2), alternating_scripted_executor(period=2), source="s"
        )
        assert [(x.task_id, x.label, x.text) for x in a] == [
            (x.task_id, x.label, x.text) for x in b
        ]


class TestSplitAndSerialization:
    def test_split_by_variant_keeps_groups_whole(self):
        examples = generate_training_examples(
            default_corpus(variants=4), alternating_scripted_executor(period=2), source="s"
        )
        train, dev = split_by_variant(examples, dev_variant_start=3)
        assert train and dev
        train_variants = {e.task_id.rsplit("-", 1)[1] for e in train}
        dev_variants = {e.task_id.rsplit("-", 1)[1] for e in dev}
        assert train_variants.isdisjoint(dev_variants)
        assert dev_variants == {"03"}

    def test_jsonl_round_trip(self, tmp_path: Path):
        examples = generate_training_examples(
            default_corpus(variants=1), alternating_scripted_executor(period=2), source="s"
        )
        path = tmp_path / "train.jsonl"
        assert write_jsonl(examples, path) == len(examples)
        assert load_jsonl(path) == examples


class TestVariantStartSlicing:
    """The real-data run drops low variants to stay disjoint from the eval pack."""

    def test_slicing_drops_low_variants(self):
        from victor.evaluation.calibration_corpus import default_corpus

        full = default_corpus(variants=18)
        sliced = full[16 * 6 :]  # the generator's slice for --variant-start 16
        variants = {t.task_id.rsplit("-", 1)[1] for t in sliced}
        assert variants == {"16", "17"}
        # None of the eval pack's variants (00-15) survive.
        assert not any(int(v) < 16 for v in variants)


class TestRealNegativeSynthesis:
    """Parse a rendered view and synthesize an effect-removed real-styled negative."""

    def test_parse_round_trips_prompt_tools_and_final(self):
        from victor.evaluation.calibration_rubric_judge import render_judged_content
        from victor.evaluation.judge_calibration_harness import Transcript, TranscriptStep
        from victor.evaluation.judge_training_data import parse_rendered_view

        tr = Transcript(
            steps=(
                TranscriptStep(kind="tool", content="edit settings.toml"),
                TranscriptStep(kind="message", content="ignored in render"),
                TranscriptStep(kind="tool", content="write port=8000"),
            ),
            final_message="Done — created the file.",
        )
        import tempfile

        ws = Path(tempfile.mkdtemp())
        (ws / "settings.toml").write_text("port = 8000\n")
        text = render_judged_content("Create settings.toml with port=8000", tr, ws)
        prompt, parsed = parse_rendered_view(text)
        assert prompt == "Create settings.toml with port=8000"
        assert parsed.final_message == "Done — created the file."
        # Only tool steps survive the render/parse round trip (render drops messages).
        assert [s.content for s in parsed.tool_steps()] == ["edit settings.toml", "write port=8000"]

    def test_synthesized_negative_shows_unsolved_workspace_and_label_0(self):
        from victor.evaluation.calibration_corpus import default_corpus
        from victor.evaluation.judge_calibration_harness import Transcript, TranscriptStep
        from victor.evaluation.judge_training_data import synthesize_effect_removed_negative

        # A file-create task: the fixture has no created file → verifier scores 0
        # even though the transcript claims success.
        task = next(t for t in default_corpus(variants=1) if t.family == "file-create")
        tr = Transcript(
            steps=(TranscriptStep(kind="tool", content="wrote the file"),),
            final_message="Done — created it.",
        )
        neg = synthesize_effect_removed_negative(task, tr)
        assert neg.label == 0
        assert neg.source == "real-neg"
        assert "Done — created it." in neg.text  # real claim retained

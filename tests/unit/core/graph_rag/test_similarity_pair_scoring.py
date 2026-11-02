"""Score pinned function pairs, preserving gaps and the actual cosine edges."""

import hashlib
from pathlib import Path

import pytest

from victor.core.graph_rag.similarity_evaluation import Corpus
from victor.core.graph_rag.similarity_pair_scoring import score_corpus


def corpus(root, *, extension="py", language="python"):
    projects, units, pairs, roots = [], [], [], {}
    for split in ("calibration", "test"):
        directory = root / split
        directory.mkdir()
        roots[split] = directory
        projects.append(
            {
                "id": split,
                "repository": f"fixture:{split}",
                "revision": "v1",
                "split": split,
                "origin": "controlled_fixture",
            }
        )
        for side in ("a", "b"):
            name = f"{split}_{side}"
            text = (
                f"def {name}(x, y):\n total=x-y\n scaled=total*3\n return scaled+4\n"
                if language == "python"
                else f"fn {name}(x:i32,y:i32)->i32 {{ let total=x-y; let scaled=total*3; scaled+4 }}\n"
            )
            file = directory / f"{name}.{extension}"
            file.write_text(text)
            units.append(
                {
                    "id": name,
                    "project": split,
                    "file": file.name,
                    "language": language,
                    "symbol": name,
                    "line": 1,
                    "end_line": 4 if language == "python" else 1,
                    "source_hash": hashlib.sha256(file.read_bytes()).hexdigest(),
                }
            )
        pairs.append(
            {
                "id": split,
                "left": f"{split}_a",
                "right": f"{split}_b",
                "relation": "structure",
                "positive": True,
                "family": split,
                "rationale": "Consistent function renaming",
                "domain": "syntax-only",
                "reviewed_by": "fixture-author",
            }
        )
    return (
        Corpus.from_dict(
            {
                "schema": "victor-similarity-corpus-v1",
                "projects": projects,
                "units": units,
                "pairs": pairs,
            }
        ),
        roots,
    )


def test_production_syntax_scores_are_actual_pairs(tmp_path):
    data, roots = corpus(tmp_path)
    result = score_corpus(data, roots)
    assert [item.score for item in result.observations] == [1, 1]
    assert result.descriptor["compiler_evidence"] == "not_evaluated"
    assert len(result.sources) == 4
    assert all(item["input_hash"] for item in result.sources)


def test_changed_source_and_stale_span_are_explicit_abstentions(tmp_path):
    data, roots = corpus(tmp_path)
    (roots["test"] / "test_a.py").write_text("def secret():\n return 42\n")
    manifest = data.to_dict()
    next(unit for unit in manifest["units"] if unit["id"] == "calibration_a")["line"] = 2
    data = Corpus.from_dict(manifest)
    result = score_corpus(data, roots)
    assert all(item.score is None for item in result.observations)
    assert "span/symbol" in result.observations[0].error
    assert "hash mismatch" in result.observations[1].error


def test_missing_frontend_is_not_a_zero_similarity(tmp_path, monkeypatch):
    from victor.core.graph_rag import language_parsers

    data, roots = corpus(tmp_path, extension="rs", language="rust")
    monkeypatch.setattr(language_parsers, "create_parser", lambda language: None)
    result = score_corpus(data, roots, channel="portable")
    assert all(
        item.score is None and "Frontend unavailable" in item.error for item in result.observations
    )


def test_source_symlinks_cannot_be_used_to_satisfy_labels(tmp_path):
    data, roots = corpus(tmp_path)
    source = roots["test"] / "test_a.py"
    target = tmp_path / "external.py"
    target.write_bytes(source.read_bytes())
    source.unlink()
    source.symlink_to(target)
    result = score_corpus(data, roots)
    assert result.observations[0].score == 1
    assert result.observations[1].score is None


def test_source_budget_and_missing_root_are_rejected(tmp_path):
    data, roots = corpus(tmp_path)
    assert all(
        item.score is None for item in score_corpus(data, roots, max_file_bytes=2).observations
    )
    with pytest.raises(ValueError, match="roots"):
        score_corpus(data, {})
    with pytest.raises(ValueError, match="Body configuration"):
        score_corpus(data, roots, cache_path=Path("unused.sqlite"))


def test_engine_mutation_invalidates_all_scores(tmp_path, monkeypatch):
    from victor.core.graph_rag import similarity_pair_scoring as scoring

    data, roots = corpus(tmp_path)
    original = scoring.detector_profile
    call = 0

    def changing(*args, **kwargs):
        nonlocal call
        call += 1
        result = original(*args, **kwargs)
        result["test_generation"] = call
        return result

    monkeypatch.setattr(scoring, "detector_profile", changing)
    result = score_corpus(data, roots)
    assert all(
        item.score is None and "changed during" in item.error for item in result.observations
    )


def test_body_pair_runner_uses_production_windowing(tmp_path):
    from victor.core.graph_rag.body_clones import LocalBodyEncoder
    from .test_body_clones import Model, options

    data, roots = corpus(tmp_path)
    encoder = LocalBodyEncoder(options(), model_factory=lambda *a, **k: Model())
    result = score_corpus(data, roots, channel="body", body_options=options(), encoder=encoder)
    assert all(item.score == pytest.approx(1) for item in result.observations)
    assert result.details["body_stats"]["retrieval"] == "requested_pair_cosine"
    assert result.details["body_profile"]["local_files_only"]


def test_controlled_corpus_exposes_numeric_domain_false_equivalence():
    import json
    from victor.core.graph_rag.similarity_evaluation import calibrate, evaluate

    for package in (
        "tree_sitter_rust",
        "tree_sitter_typescript",
        "tree_sitter_java",
        "tree_sitter_cpp",
    ):
        pytest.importorskip(package)
    root = Path(__file__).resolve().parents[3] / "fixtures" / "graph_similarity"
    data = Corpus.from_dict(json.loads((root / "controls.json").read_text()))
    run = score_corpus(data, {key: root / key for key in data.projects}, channel="portable")
    assert all(item.score is not None for item in run.observations)
    policy = calibrate(data, run.observations, channel=run.channel, profile=run.profile)
    assert policy.rejected["behavior|python|rust"] == "no_threshold_meets_precision"
    assert policy.rejected["behavior|rust|typescript"] == "no_threshold_meets_precision"
    report = evaluate(data, run.observations, policy, profile=run.profile)
    assert not report["decision_complete"]
    assert not report["production_quality_qualified"]

"""Labels, coverage and calibration must not manufacture evidence."""

from copy import deepcopy
import hashlib

import pytest

from victor.core.graph_rag.similarity_evaluation import (
    Corpus,
    Observation,
    calibrate,
    evaluate,
)


def manifest():
    projects = [
        {
            "id": split,
            "repository": f"fixture:{split}",
            "revision": "fixture-v1",
            "split": split,
            "origin": "controlled_fixture",
        }
        for split in ("calibration", "test")
    ]
    units, pairs = [], []
    for split in ("calibration", "test"):
        for index, positive in enumerate((True, True, False, False)):
            ids = []
            for side in ("left", "right"):
                name = f"{split}-{index}-{side}"
                ids.append(name)
                units.append(
                    {
                        "id": name,
                        "project": split,
                        "file": f"{name}.py",
                        "language": "python",
                        "symbol": "f",
                        "line": 1,
                        "end_line": 3,
                        "source_hash": hashlib.sha256(name.encode()).hexdigest(),
                    }
                )
            pairs.append(
                {
                    "id": f"{split}-{index}",
                    "left": ids[0],
                    "right": ids[1],
                    "relation": "behavior",
                    "positive": positive,
                    "family": f"{split}-{index}",
                    "rationale": "Checked control",
                    "domain": "bounded pure integer inputs",
                    "reviewed_by": "fixture-author",
                }
            )
    return {
        "schema": "victor-similarity-corpus-v1",
        "projects": projects,
        "units": units,
        "pairs": pairs,
    }


def observations():
    return [
        Observation(f"{split}-{i}", score)
        for split in ("calibration", "test")
        for i, score in enumerate((0.97, 0.91, 0.8, 0.4))
    ]


def test_threshold_uses_calibration_labels_only():
    data = manifest()
    corpus = Corpus.from_dict(data)
    policy = calibrate(corpus, observations(), channel="syntax", profile="a" * 64)
    assert policy.thresholds == {"behavior|python|python": 0.91}
    changed = deepcopy(data)
    for pair in changed["pairs"]:
        if pair["id"].startswith("test-"):
            pair["positive"] = not pair["positive"]
    assert (
        calibrate(
            Corpus.from_dict(changed), observations(), channel="syntax", profile="a" * 64
        ).thresholds
        == policy.thresholds
    )
    report = evaluate(corpus, observations(), policy, profile="a" * 64)
    assert report["test"]["overall"]["precision"] == 1.0
    assert report["test"]["overall"]["recall"] == 1.0
    assert not report["production_quality_qualified"]


@pytest.mark.parametrize("leak", ["repository", "family", "source"])
def test_split_leakage_is_rejected(leak):
    data = manifest()
    if leak == "repository":
        data["projects"][1]["repository"] = data["projects"][0]["repository"]
    elif leak == "family":
        data["pairs"][4]["family"] = data["pairs"][0]["family"]
    else:
        data["units"][8]["source_hash"] = data["units"][0]["source_hash"]
    with pytest.raises(ValueError, match="leakage"):
        Corpus.from_dict(data)


def test_abstentions_are_not_true_negatives_or_silent_false_negatives():
    corpus = Corpus.from_dict(manifest())
    scores = observations()
    scores[-4] = Observation("test-0", None, "unavailable frontend")
    policy = calibrate(corpus, scores, channel="syntax", profile="a" * 64)
    report = evaluate(corpus, scores, policy, profile="a" * 64)
    metrics = report["test"]["overall"]
    assert metrics["scored_pairs"] == 3 and metrics["abstained_pairs"] == 1
    assert metrics["recall"] == 1.0
    assert metrics["end_to_end_recall"] == 0.5
    assert metrics["coverage"] == 0.75
    assert not report["coverage_complete"]


def test_no_feasible_threshold_is_not_an_empty_perfect_prediction():
    corpus = Corpus.from_dict(manifest())
    scores = [Observation(item.pair_id, 1.0) for item in observations()]
    policy = calibrate(corpus, scores, channel="syntax", profile="a" * 64)
    assert not policy.thresholds
    assert policy.rejected == {"behavior|python|python": "no_threshold_meets_precision"}
    report = evaluate(corpus, scores, policy, profile="a" * 64)
    assert report["test"]["overall"]["precision"] is None
    assert not report["decision_complete"]


@pytest.mark.parametrize("score", [float("nan"), float("inf"), 1.01, True])
def test_invalid_scores_fail(score):
    with pytest.raises(ValueError):
        Observation("test-0", score)


@pytest.mark.parametrize("bad", ["duplicate", "missing", "extra"])
def test_observations_must_cover_exactly_labelled_pairs(bad):
    scores = observations()
    if bad == "duplicate":
        scores.append(scores[0])
    elif bad == "missing":
        scores.pop()
    else:
        scores.append(Observation("unknown", 0.5))
    with pytest.raises(ValueError):
        calibrate(Corpus.from_dict(manifest()), scores, channel="syntax", profile="a" * 64)


def test_profile_change_cannot_reuse_threshold_policy():
    corpus = Corpus.from_dict(manifest())
    policy = calibrate(corpus, observations(), channel="syntax", profile="a" * 64)
    with pytest.raises(ValueError, match="profile"):
        evaluate(corpus, observations(), policy, profile="b" * 64)


def test_empty_or_unreviewed_labels_cannot_claim_quality():
    data = manifest()
    data["pairs"][0]["reviewed_by"] = ""
    with pytest.raises(ValueError):
        Corpus.from_dict(data)


def test_score_ties_do_not_turn_positive_pair_names_into_perfect_rank():
    corpus = Corpus.from_dict(manifest())
    scores = [Observation(item.pair_id, 0.9) for item in observations()]
    policy = calibrate(corpus, scores, channel="syntax", profile="a" * 64)
    result = evaluate(corpus, scores, policy, profile="a" * 64, k=2)
    metrics = result["test"]["facets"]["behavior|python|python"]
    assert metrics["labelled_pair_precision_at_k"] == 0.5
    assert result["test"]["overall"]["labelled_pair_precision_at_k"] is None


def test_empty_test_split_is_rejected_even_with_an_unused_test_project():
    data = manifest()
    data["pairs"] = [pair for pair in data["pairs"] if pair["id"].startswith("calibration")]
    data["units"] = [unit for unit in data["units"] if unit["project"] == "calibration"]
    with pytest.raises(ValueError, match="test pairs"):
        Corpus.from_dict(data)


def test_changed_calibration_scores_cannot_reuse_policy():
    corpus = Corpus.from_dict(manifest())
    scores = observations()
    policy = calibrate(corpus, scores, channel="body", profile="a" * 64)
    scores[0] = Observation(scores[0].pair_id, 0.98)
    with pytest.raises(ValueError, match="observations changed"):
        evaluate(corpus, scores, policy, profile="a" * 64)


def test_corpus_and_policy_do_not_allow_accidental_mutation():
    corpus = Corpus.from_dict(manifest())
    policy = calibrate(corpus, observations(), channel="syntax", profile="a" * 64)
    with pytest.raises(TypeError):
        corpus.units["injected"] = corpus.units["test-0-left"]
    with pytest.raises(TypeError):
        policy.thresholds["behavior|python|python"] = 0


def test_fabricated_policy_cannot_claim_calibration_evidence():
    from dataclasses import replace

    corpus = Corpus.from_dict(manifest())
    scores = observations()
    policy = calibrate(corpus, scores, channel="syntax", profile="a" * 64)
    forged = replace(policy, thresholds={"behavior|python|python": 0.1})
    with pytest.raises(ValueError, match="thresholds"):
        evaluate(corpus, scores, forged, profile="a" * 64)


@pytest.mark.parametrize("project_alias", [False, True])
def test_canonical_function_aliases_cannot_inflate_support(project_alias):
    data = manifest()
    source = dict(data["units"][0])
    source["id"] = "aliased-unit"
    if project_alias:
        project = {**data["projects"][0], "id": "same-repository-alias"}
        data["projects"].append(project)
        source["project"] = project["id"]
    data["units"].append(source)
    data["pairs"].append({**data["pairs"][0], "id": "fake-extra-pair", "left": source["id"]})
    with pytest.raises(ValueError, match="canonical source"):
        Corpus.from_dict(data)

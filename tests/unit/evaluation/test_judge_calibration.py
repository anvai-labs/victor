# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for LLM-judge reliability gating (EVR-2, ADR-011)."""

import math

import pytest

from victor.evaluation.judge_calibration import (
    JudgeReliability,
    JudgeReliabilityGate,
    OrderSwapEnsembleJudge,
    cohens_kappa,
    evaluate_judge_agreement,
    krippendorff_alpha,
)

# --- Cohen's kappa ---------------------------------------------------------------------------------


def test_kappa_perfect_agreement():
    assert cohens_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0


def test_kappa_no_chance_corrected_agreement():
    # Constant-but-opposite raters: observed agreement 0, expected 0 -> kappa 0.
    assert cohens_kappa([1, 1, 1, 1], [0, 0, 0, 0]) == 0.0


def test_kappa_partial():
    k = cohens_kappa(["a", "a", "b", "b"], ["a", "b", "b", "b"])
    assert -1.0 <= k <= 1.0 and k < 1.0


def test_kappa_length_mismatch_raises():
    with pytest.raises(ValueError):
        cohens_kappa([1, 0], [1])


# --- Krippendorff's alpha --------------------------------------------------------------------------


def test_alpha_perfect_interval():
    assert krippendorff_alpha([[0.0, 0.0], [1.0, 1.0]]) == 1.0


def test_alpha_max_disagreement_two_items():
    # Hand-derived: D_o=0.5, D_e=1/3 -> alpha = 1 - 1.5 = -0.5.
    assert math.isclose(krippendorff_alpha([[0.0, 1.0], [1.0, 0.0]]), -0.5, rel_tol=1e-9)


def test_alpha_nominal_perfect():
    assert krippendorff_alpha([["x", "x"], ["y", "y"]], level="nominal") == 1.0


def test_alpha_tolerates_missing_and_no_pairs():
    # One ratable unit (the other has a single value -> not pairable).
    assert krippendorff_alpha([[0.5, 0.5], [None, 0.2]]) == 1.0
    # No pairable values at all -> nan.
    assert math.isnan(krippendorff_alpha([[0.5], [None]]))


# --- evaluate_judge_agreement + gate ---------------------------------------------------------------


def test_evaluate_agreement_perfect_then_gate_trusts():
    gold = [0.2, 0.8, 0.5, 1.0]
    rel = evaluate_judge_agreement(gold, list(gold))
    assert rel.krippendorff_alpha == 1.0
    assert JudgeReliabilityGate(alpha_threshold=0.7).is_trusted(rel) is True


def test_gate_rejects_low_alpha_and_nan():
    low = JudgeReliability(n=4, krippendorff_alpha=0.3)
    decision = JudgeReliabilityGate(0.7).decide(low)
    assert decision.trusted is False and "falling back" in decision.reason

    nan_rel = JudgeReliability(n=0, krippendorff_alpha=float("nan"))
    assert JudgeReliabilityGate(0.7).is_trusted(nan_rel) is False


def test_evaluate_agreement_with_categorizer_computes_kappa():
    gold = [0.1, 0.9, 0.5]
    judged = [0.15, 0.85, 0.55]
    rel = evaluate_judge_agreement(gold, judged, categorize=lambda s: round(s * 2) / 2)
    assert rel.cohens_kappa is not None


# --- Order-swap ensemble ---------------------------------------------------------------------------


def test_ensemble_cancels_position_bias():
    # A judge that scores purely by position (first = best) is fully position-biased.
    def positional(cands):
        n = len(cands)
        return [(n - 1 - pos) / (n - 1) for pos in range(n)]

    ensemble = OrderSwapEnsembleJudge(positional, swaps=2)  # identity + reverse
    scores = ensemble.score(["a", "b", "c"])
    # With identity+reverse over 3 items every candidate visits symmetric positions -> all equal.
    assert all(math.isclose(s, 0.5, abs_tol=1e-9) for s in scores)


def test_ensemble_preserves_content_judge():
    # A content-based judge (scores the candidate's own value) is unaffected by reordering.
    ensemble = OrderSwapEnsembleJudge(lambda cands: list(cands), swaps=3)
    scores = ensemble.score([0.2, 0.8, 0.5])
    assert [round(s, 6) for s in scores] == [0.2, 0.8, 0.5]


def test_ensemble_is_deterministic():
    judge = OrderSwapEnsembleJudge(lambda cands: [hash(c) % 7 / 6 for c in cands], swaps=2)
    assert judge.score(["x", "y", "z"]) == judge.score(["x", "y", "z"])


def test_ensemble_empty_and_singleton():
    ens = OrderSwapEnsembleJudge(lambda cands: [1.0] * len(cands))
    assert ens.score([]) == []
    assert ens.score(["only"]) == [1.0]

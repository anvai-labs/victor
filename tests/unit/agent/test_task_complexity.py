"""TaskComplexityService after its move to the runtime (layering round 3)."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

import victor.agent.task_complexity as tc
from victor.agent.task_complexity import TaskComplexityService
from victor.core.task.protocols import TaskComplexity


def test_record_outcome_is_a_compatible_no_op():
    service = TaskComplexityService(use_semantic=False, use_rl=True)
    assert service.use_rl is True
    assert (
        service.record_outcome(
            message="fix the authentication bug",
            predicted=TaskComplexity.MEDIUM,
            success=True,
            actual_tools_used=8,
            actual_turns=5,
        )
        is None
    )


def test_record_outcome_never_consults_the_rl_coordinator():
    """The removed lookup called a non-existent ``RLCoordinator.get_instance``;
    nothing may reach the coordinator from the classifier any more."""
    from victor.framework.rl.coordinator import RLCoordinator

    # Record rather than raise: the old lookup swallowed every exception.
    lookup = mock.Mock()
    with (
        mock.patch.object(RLCoordinator, "get_instance", lookup, create=True),
        mock.patch("victor.framework.rl.coordinator.get_rl_coordinator", lookup),
    ):
        service = TaskComplexityService(use_semantic=False)
        service.record_outcome("list files", TaskComplexity.SIMPLE, False, 1, 1)
        service.classify("list all files in the repo")
    lookup.assert_not_called()


def test_module_has_no_framework_rl_dependency():
    tree = ast.parse(Path(tc.__file__).read_text(encoding="utf-8"))
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert not {m for m in imported if m.startswith("victor.framework")}, imported


def test_classification_still_routes_through_core_task_vocabulary():
    service = TaskComplexityService(use_semantic=False)
    result = service.classify("list all files in the repo")
    assert isinstance(result.complexity, TaskComplexity)
    assert result.tool_budget == service.budgets[result.complexity]


# --- inference chain -------------------------------------------------------


def _bare(**kwargs) -> TaskComplexityService:
    """A service with no built-in regex patterns, so each stage is isolated."""
    with mock.patch.object(tc, "PATTERNS", {}):
        return TaskComplexityService(use_semantic=False, **kwargs)


def test_empty_and_direct_response_short_circuit():
    service = _bare()
    assert service.classify("   ").complexity is TaskComplexity.SIMPLE
    exact = service.classify("Reply with exactly: OK")
    assert exact.tool_budget == 1
    assert exact.matched_patterns == ["direct_response:exact_response"]


def test_custom_classifier_wins_and_none_falls_through():
    sentinel = tc.TaskClassification(
        complexity=TaskComplexity.COMPLEX, tool_budget=7, confidence=1.0, matched_patterns=["c"]
    )
    service = _bare(custom_classifiers=[lambda m: None, lambda m: sentinel])
    assert service.classify("anything at all") is sentinel


def test_high_confidence_custom_pattern_and_invalid_pattern():
    service = _bare(
        custom_patterns={
            TaskComplexity.ACTION: [(r"\bdeploy\b", 0.95, "deploy"), (r"(", 0.5, "broken")]
        }
    )
    result = service.classify("please deploy it")
    assert (result.complexity, result.confidence, result.matched_patterns) == (
        TaskComplexity.ACTION,
        0.95,
        ["deploy"],
    )


def test_score_based_fallback_and_default():
    service = _bare(
        custom_patterns={
            TaskComplexity.ANALYSIS: [(r"audit", 0.5, "audit"), (r"deep", 0.5, "deep")],
            TaskComplexity.SIMPLE: [(r"audit", 0.2, "weak")],
        }
    )
    scored = service.classify("deep audit")
    assert scored.complexity is TaskComplexity.ANALYSIS
    assert scored.confidence == pytest.approx(0.8)
    default = service.classify("zzqx blorp")
    assert (default.complexity, default.confidence) == (TaskComplexity.MEDIUM, 0.3)


def test_semantic_stage_maps_task_type_above_threshold():
    from victor.classification import TaskType

    service = _bare()
    service.use_semantic = True
    hit = SimpleNamespace(
        task_type=TaskType.SEARCH, confidence=0.9, top_matches=[("a", 1), ("b", 1)]
    )
    classifier = mock.Mock(classify_sync=mock.Mock(return_value=hit))
    with mock.patch.object(service, "_get_semantic_classifier", return_value=classifier):
        result = service.classify("zzqx blorp")
    assert result.matched_patterns == ["semantic:search", "a", "b"]

    classifier.classify_sync.return_value = SimpleNamespace(
        task_type=TaskType.SEARCH, confidence=0.1, top_matches=[]
    )
    with mock.patch.object(service, "_get_semantic_classifier", return_value=classifier):
        assert service._classify_semantic("x") is None
    classifier.classify_sync.side_effect = RuntimeError("boom")
    with mock.patch.object(service, "_get_semantic_classifier", return_value=classifier):
        assert service._classify_semantic("x") is None
    with mock.patch.object(service, "_get_semantic_classifier", return_value=None):
        assert service._classify_semantic("x") is None


def test_semantic_loader_caches_and_degrades_on_import_error():
    service = TaskComplexityService(use_semantic=True)
    loader = "victor.storage.embeddings.task_classifier.TaskTypeClassifier.get_instance"
    with mock.patch(loader, return_value="clf") as get_instance:
        assert service._get_semantic_classifier() == "clf"
        assert service._get_semantic_classifier() == "clf"
    get_instance.assert_called_once()

    degraded = TaskComplexityService(use_semantic=True)
    with mock.patch(loader, side_effect=ImportError):
        assert degraded._get_semantic_classifier() is None
    assert degraded.use_semantic is False


def _edge_service(decision=None, *, allowed=True, raises=False):
    llm = mock.Mock()
    if raises:
        llm.decide_sync.side_effect = RuntimeError("down")
    else:
        llm.decide_sync.return_value = decision
    service = _bare(llm_decision_service=llm)
    patcher = mock.patch("victor.agent.decisions.chain.should_use_llm", return_value=allowed)
    return service, patcher


@pytest.mark.parametrize(
    "task_type,expected",
    [("edit", TaskComplexity.ACTION), ("analysis", TaskComplexity.ANALYSIS), ("other", None)],
)
def test_edge_model_maps_confident_llm_decisions(task_type, expected):
    decision = SimpleNamespace(
        source="llm", confidence=0.8, result=SimpleNamespace(task_type=task_type)
    )
    service, gate = _edge_service(decision)
    with gate:
        result = service.classify("zzqx blorp")
    assert result.complexity is (expected or TaskComplexity.MEDIUM)
    assert result.matched_patterns == [f"edge:{task_type}"]


@pytest.mark.parametrize(
    "decision,allowed,raises",
    [
        (None, False, False),
        (SimpleNamespace(source="heuristic", confidence=0.9, result=None), True, False),
        (SimpleNamespace(source="llm", confidence=0.3, result=None), True, False),
        (SimpleNamespace(source="llm", confidence=0.9, result=SimpleNamespace()), True, False),
        (None, True, True),
    ],
)
def test_edge_model_declines_untrusted_decisions(decision, allowed, raises):
    service, gate = _edge_service(decision, allowed=allowed, raises=raises)
    with gate:
        assert service._classify_with_edge_model("zzqx") is None
    assert _bare()._classify_with_edge_model("zzqx") is None


def test_fast_path_uses_the_pattern_registry():
    service = TaskComplexityService(use_semantic=False)
    fast = service._classify_fast("list all files")
    assert fast is not None and fast.matched_patterns == ["fast:list_files"]
    with mock.patch("victor.classification.get_pattern_matcher", side_effect=RuntimeError):
        assert service._classify_fast("list all files") is None


def test_budget_helpers_and_convenience_functions():
    service = _bare()
    service.update_budget(TaskComplexity.SIMPLE, 3)
    assert service.get_budget(TaskComplexity.SIMPLE) == 3
    assert tc.ComplexityBudget.for_complexity(TaskComplexity.ANALYSIS) is (
        tc.COMPLEXITY_BUDGETS[TaskComplexity.ANALYSIS]
    )
    assert tc.get_budget_for_task("Reply with exactly: OK", 2.0) == 2
    assert tc.is_action_task("Reply with exactly: OK") is False
    assert tc.is_analysis_task("Reply with exactly: OK") is False

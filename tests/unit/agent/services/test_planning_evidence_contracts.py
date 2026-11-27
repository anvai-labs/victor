# Copyright 2026 Vijaykumar Singh <vijay@anaiops.com>
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

"""TDD coverage for the plan-mode evidence contracts and decision helpers.

Targets the regions the 680-test planning suite leaves uncovered in
``planning_runtime`` (53%): the degenerate-output detector, list
extraction, planning-gate signals, the step-evidence verdict tree, and
the agentic-retry predicate — plus the entirely untested ``PlanStore``
persistence layer.
"""

from pathlib import Path
from types import SimpleNamespace


from victor.agent.planning.store import PlanStore, get_plan_store, reset_plan_store
from victor.agent.services.planning_runtime import (
    PlanningConfig,
    PlanningCoordinator,
    PlanningMode,
)


def _coordinator(mode: PlanningMode = PlanningMode.AUTO) -> PlanningCoordinator:
    """A coordinator with just the decision state the helpers read."""
    coord = PlanningCoordinator.__new__(PlanningCoordinator)
    coord.config = PlanningConfig()
    coord._planning_mode = mode
    return coord


def _step(
    description="analyze the auth module",
    produces=None,
    execution="agent",
    allowed_tools=("read", "grep"),
):
    context = {}
    if produces is not None:
        context["produces"] = produces
    return SimpleNamespace(
        description=description,
        context=context,
        execution=execution,
        allowed_tools=list(allowed_tools),
    )


def _result(output="", tool_calls=0, artifacts=None, metadata=None, success=True, error=""):
    return SimpleNamespace(
        output=output,
        tool_calls_used=tool_calls,
        artifacts=list(artifacts or []),
        metadata=metadata or {},
        success=success,
        error=error,
    )


class TestIsDegenerateOutput:
    """The degenerate-sampling detector (guards the evidence contract)."""

    def test_unbroken_single_token_run_is_degenerate(self):
        output = "EventHandlerExecutorAction" * 2000  # ~50KB single token
        is_deg, reason = PlanningCoordinator._is_degenerate_output(output)
        assert is_deg
        assert "single unbroken" in reason

    def test_token_loop_with_three_unique_tokens(self):
        output = " ".join(["alpha", "beta", "gamma"] * 30)
        is_deg, reason = PlanningCoordinator._is_degenerate_output(output)
        assert is_deg
        assert "unique" in reason

    def test_single_short_token_dominating_byte_length(self):
        output = "x " * 300 + " and then a small amount of ordinary text follows here"
        is_deg, reason = PlanningCoordinator._is_degenerate_output(output)
        assert is_deg
        assert "consumes" in reason

    def test_consecutive_bigram_repetition_over_fifty(self):
        distinct = [f"word{i}" for i in range(80)]
        output = " ".join(distinct) + " " + " ".join(["snap"] * 55)
        is_deg, reason = PlanningCoordinator._is_degenerate_output(output)
        assert is_deg
        assert "repeated" in reason

    def test_repeated_pair_without_consecutive_run_is_not_degenerate(self):
        words = [f"tok{i}" for i in range(60)]
        interleaved = [w for pair in zip(words, ["and"] * 60) for w in pair]
        is_deg, _ = PlanningCoordinator._is_degenerate_output(" ".join(interleaved))
        assert not is_deg

    def test_normal_prose_is_not_degenerate(self):
        output = (
            "The auth module exposes JWTAuthenticator in auth.py which verifies "
            "tokens via verify(). Session management lives beside it and both are "
            "covered by tests in auth_test.py covering expiry and refresh paths."
        )
        is_deg, _ = PlanningCoordinator._is_degenerate_output(output)
        assert not is_deg

    def test_short_outputs_are_exempt(self):
        assert PlanningCoordinator._is_degenerate_output("done done done")[0] is False
        assert PlanningCoordinator._is_degenerate_output("")[0] is False


class TestExtractListFromOutput:
    """Produced-artifact list extraction (feeds loop/conditional routing)."""

    def test_json_array_preferred(self):
        output = '["a.py", "b.py", "c.py"]'
        assert PlanningCoordinator._extract_list_from_output(output) == [
            "a.py",
            "b.py",
            "c.py",
        ]

    def test_wrapped_json_array_falls_back_to_lines(self):
        # Only a bare leading "[" takes the JSON path; prose-wrapped arrays
        # use line extraction (a documented contract, not an accident).
        output = 'Here you go:\n["a.py", "b.py", "c.py"]\nend'
        result = PlanningCoordinator._extract_list_from_output(output)
        assert isinstance(result, list)

    def test_bullet_list_extraction(self):
        output = "Found the following:\n- src/one.py\n- src/two.py\n- src/three.py"
        assert PlanningCoordinator._extract_list_from_output(output) == [
            "src/one.py",
            "src/two.py",
            "src/three.py",
        ]

    def test_numbered_list_extraction(self):
        output = "1. alpha\n2. beta\n3. gamma"
        assert PlanningCoordinator._extract_list_from_output(output) == [
            "alpha",
            "beta",
            "gamma",
        ]

    def test_none_sentinel_returns_empty(self):
        assert PlanningCoordinator._extract_list_from_output("(none)") == []

    def test_compute_placeholder_returns_empty(self):
        assert (
            PlanningCoordinator._extract_list_from_output("Compute step: no registered handler")
            == []
        )

    def test_prose_sentence_guard_returns_empty(self):
        prose = "I looked at the repository and found that everything is fine."
        assert PlanningCoordinator._extract_list_from_output(prose) == []

    def test_empty_output_returns_empty(self):
        assert PlanningCoordinator._extract_list_from_output("") == []


class TestShouldUsePlanning:
    """The planning gate's signal resolution."""

    def test_always_and_never_modes_short_circuit(self):
        assert _coordinator(PlanningMode.ALWAYS)._should_use_planning("hi") is True
        assert (
            _coordinator(PlanningMode.NEVER)._should_use_planning(
                "refactor everything then migrate the database"
            )
            is False
        )

    def test_simple_message_does_not_trigger_auto_planning(self):
        assert _coordinator()._should_use_planning("what does this file do?") is False

    def test_multi_step_keyword_density_triggers(self):
        message = (
            "first analyze the module, then refactor the callers, and finally "
            "migrate the database and deploy after testing each stage"
        )
        assert _coordinator()._should_use_planning(message) is True

    def test_complexity_signal_triggers(self):
        from victor.agent.task_analyzer import TaskAnalysis, TaskComplexity

        analysis = TaskAnalysis(
            complexity=TaskComplexity.COMPLEX,
            tool_budget=10,
            complexity_confidence=0.9,
        )
        assert _coordinator()._should_use_planning("do the thing", analysis) is True


class TestAssessStepEvidence:
    """The step-evidence verdict tree."""

    def test_no_output_fails(self):
        ok, reason, _ = _coordinator()._assess_step_evidence(_step(), _result(output=""), {})
        assert not ok
        assert "no output" in reason

    def test_weak_completion_marker_fails(self):
        ok, reason, _ = _coordinator()._assess_step_evidence(
            _step(), _result(output="done", tool_calls=2), {}
        )
        assert not ok
        assert "generic completion marker" in reason

    def test_degenerate_spew_rejected_even_with_many_tool_calls(self):
        spew = "EventHandlerExecutorAction" * 2000
        ok, reason, evidence = _coordinator()._assess_step_evidence(
            _step(), _result(output=spew, tool_calls=12), {}
        )
        assert not ok
        assert "degenerate" in reason
        assert evidence["is_degenerate_output"] is True

    def test_artifacts_pass_without_tool_calls(self):
        ok, reason, _ = _coordinator()._assess_step_evidence(
            _step(),
            _result(output="wrote the summary", artifacts=[Path("out/summary.md")]),
            {},
        )
        assert ok

    def test_source_count_passes(self):
        ok, _, _ = _coordinator()._assess_step_evidence(
            _step(),
            _result(output="reviewed the sources", tool_calls=1),
            {"source_count": 4},
        )
        assert ok

    def test_zero_tool_calls_fail(self):
        ok, reason, _ = _coordinator()._assess_step_evidence(
            _step(), _result(output="looks fine overall"), {}
        )
        assert not ok
        assert "no tool-backed execution" in reason

    def test_intent_phrase_without_tools_fails(self):
        ok, reason, _ = _coordinator()._assess_step_evidence(
            _step(),
            _result(output="Let me use grep to find the callers", tool_calls=1),
            {},
        )
        assert not ok
        assert "intent statement" in reason

    def test_intent_phrase_with_five_tool_calls_passes(self):
        ok, reason, _ = _coordinator()._assess_step_evidence(
            _step(),
            _result(output="Let me summarize what I found across the crates", tool_calls=6),
            {},
        )
        assert ok
        assert "spawn artifact" in reason or "substantive" in reason

    def test_five_tool_calls_with_long_output_pass(self):
        output = (
            "Cross-crate review found three call sites; the retry policy lives in "
            "transport.rs while callers in operator and factory duplicate the "
            "backoff computation, and none of them honor the circuit breaker."
        )
        ok, _, _ = _coordinator()._assess_step_evidence(
            _step(), _result(output=output, tool_calls=5), {}
        )
        assert ok

    def test_write_step_with_one_tool_passes(self):
        step = _step(description="write the migration notes", allowed_tools=("write",))
        ok, reason, _ = _coordinator()._assess_step_evidence(
            step, _result(output="documented the plan in docs/migration.md", tool_calls=1), {}
        )
        assert ok
        assert "write tool" in reason.lower()

    def test_content_tool_signature_output_passes(self):
        output = "def verify(self, token: str) -> bool:\n    return len(token) > 10\n"
        ok, _, evidence = _coordinator()._assess_step_evidence(
            _step(), _result(output=output, tool_calls=2), {}
        )
        assert ok
        assert evidence["has_content_tool"] is True

    def test_manifest_dependency_audit_passes(self):
        output = '[dependencies]\nsandhi-gateway = "0.26.0"\nedition = "2021"\n'
        ok, _, evidence = _coordinator()._assess_step_evidence(
            _step(), _result(output=output, tool_calls=1), {}
        )
        assert ok
        assert evidence["has_content_tool"] is True

    def test_explicit_content_tool_names_in_metadata_pass(self):
        output = "Examined the relevant sections and confirmed the behavior."
        ok, _, evidence = _coordinator()._assess_step_evidence(
            _step(),
            _result(output=output, tool_calls=2),
            {"tool_names_used": ["read", "grep"]},
        )
        assert ok
        assert evidence["has_content_tool"] is True

    def test_directory_listing_only_fails_required_produces(self):
        listing = "src/\n  auth.py\n  session.py\n  tokens.py\n  utils.py"
        # Suffix-matched produces keys (findings/paths/items/...) require a
        # non-empty artifact; bare listings are not one.
        step = _step(produces="auth_paths")
        ok, reason, _ = _coordinator()._assess_step_evidence(
            step, _result(output=listing, tool_calls=3), {}
        )
        assert not ok
        assert "listing" in reason

    def test_unresolved_tool_markup_fails_required_produces(self):
        output = "<tool_call>\n<parameter>query</parameter>\n</tool_call>"
        step = _step(produces="findings")
        ok, reason, _ = _coordinator()._assess_step_evidence(
            step, _result(output=output, tool_calls=2), {}
        )
        assert not ok
        assert "tool-call markup" in reason


class TestShouldRetryAfterEvidenceFailure:
    """The one-shot agentic-retry predicate for hollow artifacts."""

    def _failed(self, produces="items", metadata=None, error="", execution="agent"):
        step = _step(produces=produces, execution=execution)
        step.context["requires_nonempty"] = True
        result = _result(
            output="partial", tool_calls=2, metadata=metadata, success=False, error=error
        )
        return step, result

    def test_successful_result_never_retries(self):
        step, result = self._failed()
        result.success = True
        assert PlanningCoordinator._should_retry_after_evidence_failure(step, result) is False

    def test_empty_required_produces_triggers_retry(self):
        step, result = self._failed(metadata={"empty_required_produces": {"items": 0}})
        assert PlanningCoordinator._should_retry_after_evidence_failure(step, result) is True

    def test_listing_only_validation_triggers_retry(self):
        step, result = self._failed(
            metadata={"evidence_validation": {"passed": False, "is_directory_listing_only": True}}
        )
        assert PlanningCoordinator._should_retry_after_evidence_failure(step, result) is True

    def test_unresolved_markup_validation_triggers_retry(self):
        step, result = self._failed(
            metadata={"evidence_validation": {"passed": False, "has_unresolved_tool_markup": True}}
        )
        assert PlanningCoordinator._should_retry_after_evidence_failure(step, result) is True

    def test_error_text_contract_phrases_trigger_retry(self):
        step, result = self._failed(
            error="Insufficient execution evidence for artifact",
            metadata={"evidence_validation": {"passed": False}},
        )
        assert PlanningCoordinator._should_retry_after_evidence_failure(step, result) is True

    def test_unrelated_failure_does_not_retry(self):
        step, result = self._failed(error="provider timeout")
        assert PlanningCoordinator._should_retry_after_evidence_failure(step, result) is False

    def test_compute_execution_never_retries(self):
        step, result = self._failed(
            metadata={"empty_required_produces": {"items": 0}}, execution="compute"
        )
        assert PlanningCoordinator._should_retry_after_evidence_failure(step, result) is False

    def test_step_without_produces_never_retries(self):
        result = _result(output="x", success=False, error="insufficient execution evidence")
        assert PlanningCoordinator._should_retry_after_evidence_failure(_step(), result) is False


class TestPlanStorePersistence:
    """The 0%-covered plan persistence layer."""

    def _plan(self, goal="test goal"):
        from victor.agent.planning.base import ExecutionPlan, PlanStep

        plan = ExecutionPlan(id=f"plan-{abs(hash(goal)) % 10**8}", goal=goal)
        plan.steps = [
            PlanStep(id="s1", description="step one"),
            PlanStep(id="s2", description="step two"),
        ]
        return plan

    def test_save_list_load_round_trip(self, tmp_path):
        store = PlanStore(project_root=tmp_path)
        plan = self._plan("round trip goal")
        path = store.save(plan)
        assert path.exists()

        listed = store.list_plans()
        assert len(listed) >= 1
        assert any("round trip goal" in str(entry.get("goal", "")) for entry in listed)

        loaded = store.load(path.stem)
        assert loaded is not None
        assert loaded.goal == "round trip goal"
        assert len(loaded.steps) == 2

    def test_get_latest_and_delete(self, tmp_path):
        store = PlanStore(project_root=tmp_path)
        # get_latest sorts filenames in reverse: control the order explicitly.
        store.save(self._plan("older"), filename="01_older")
        newest = self._plan("newest")
        store.save(newest, filename="02_newest")

        latest = store.get_latest()
        assert latest is not None
        assert latest.goal == "newest"

        # delete() matches by PLAN ID, not filename stem.
        assert store.delete(newest.id) is True
        assert store.get_latest().goal == "older"
        assert store.delete(newest.id) is False

    def test_unknown_load_returns_none(self, tmp_path):
        assert PlanStore(project_root=tmp_path).load("no-such-plan") is None

    def test_singleton_accessor_resets(self, tmp_path):
        reset_plan_store()
        store = get_plan_store(project_root=tmp_path)
        assert get_plan_store(project_root=tmp_path) is store
        reset_plan_store()
        assert get_plan_store(project_root=tmp_path) is not store
        reset_plan_store()


class TestComplexityRankExhaustiveness:
    """A future enum member must not silently rank below the threshold."""

    def test_rank_map_covers_every_planning_complexity_member(self):
        from victor.agent.planning.readable_schema import TaskComplexity
        from victor.agent.services.planning_runtime import PlanningRuntimeService

        rank = PlanningRuntimeService._PLANNING_COMPLEXITY_RANK
        members = list(TaskComplexity)
        assert set(rank.keys()) == set(members), (
            f"_PLANNING_COMPLEXITY_RANK is missing entries for {set(members) - set(rank)}; "
            "unmapped members default to rank 0 and would never trigger planning"
        )
        # Ranks are strictly ordered AND distinct — a tie (two members
        # sharing a rank) would silently let the lower-ordered member
        # trigger at the other's threshold.
        values = [rank[m] for m in members]
        assert values == sorted(values), "ranks must order with the enum"
        assert len(set(values)) == len(values), "ranks must be distinct"

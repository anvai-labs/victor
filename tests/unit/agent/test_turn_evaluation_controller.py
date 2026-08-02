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

"""Tests for the shared per-turn evaluation components (content-repetition, plateau, controller).

These lock the behavior that the headless (AgenticLoop) and streaming (StreamingChatExecutor)
loops both delegate to, so the consolidation is verifiably faithful.
"""

from victor.agent.turn_policy import (
    ContentRepetitionDetector,
    PlateauDetector,
    TurnEvaluationController,
    TurnObservation,
    evaluate_overlap_repetition,
)

# --------------------------------------------------------------------------- #
# evaluate_overlap_repetition (the exact thresholds the streaming loop used)
# --------------------------------------------------------------------------- #


def test_overlap_thresholds():
    assert evaluate_overlap_repetition(0.9, 0) == (1, "near_duplicate")
    assert evaluate_overlap_repetition(0.6, 0) == (1, "accumulating")
    assert evaluate_overlap_repetition(0.6, 1) == (2, "high_overlap")
    assert evaluate_overlap_repetition(0.2, 3) == (0, "reset")
    assert evaluate_overlap_repetition(0.4, 2) == (
        2,
        "hold",
    )  # ambiguous: hold, don't decay


# --------------------------------------------------------------------------- #
# ContentRepetitionDetector
# --------------------------------------------------------------------------- #


def test_identical_content_trips_near_duplicate_on_second_turn():
    # Matches the streaming loop: identical content gives overlap=1.0, so the second
    # turn is already a near-duplicate terminal (the exact-hash-3 path is a backstop).
    d = ContentRepetitionDetector()
    same = "I will now analyze the orchestrator and report what I find in detail."
    assert d.record(same) == "none"
    assert d.record(same) == "near_duplicate"


def test_near_duplicate_is_terminal_on_single_strong_repeat():
    d = ContentRepetitionDetector()
    a = "Let me drill into the orchestrator initialization phases and the service wiring details."
    b = "Let me drill into the orchestrator initialization phases and the service wiring details!!"
    assert d.record(a) == "none"
    assert d.record(b) == "near_duplicate"  # ~full overlap -> stop immediately


def test_distinct_content_does_not_trip():
    d = ContentRepetitionDetector()
    assert d.record("The orchestrator coordinates extracted components via a facade pattern.") in (
        "none",
        "reset",
    )
    assert d.record("Tool execution delegates to ToolPipeline with retry and caching logic.") in (
        "none",
        "reset",
        "hold",
    )


def test_short_content_ignored():
    d = ContentRepetitionDetector()
    assert d.record("ok") == "none"
    assert d.record("") == "none"
    assert d.record(None) == "none"


# --------------------------------------------------------------------------- #
# PlateauDetector (streaming productivity-weighted formula)
# --------------------------------------------------------------------------- #


def test_plateau_nudges_only_when_unproductive():
    d = PlateauDetector()
    # Three unproductive low-score turns -> plateau + nudge.
    d.record(0, 100)
    d.record(0, 100)
    r = d.record(0, 100)
    assert r.is_plateau and r.should_nudge

    # A productive turn flattening the score is a plateau but must NOT nudge.
    d2 = PlateauDetector()
    d2.record(2, 100)
    d2.record(2, 100)
    r2 = d2.record(2, 100)
    assert r2.is_plateau and not r2.should_nudge


def test_no_plateau_before_window():
    d = PlateauDetector()
    assert not d.record(0, 50).is_plateau
    assert not d.record(0, 50).is_plateau  # only 2 scores so far


# --------------------------------------------------------------------------- #
# TurnEvaluationController
# --------------------------------------------------------------------------- #


def test_controller_stops_on_content_repetition():
    c = TurnEvaluationController()
    same = "Reading the file in chunks and continuing the analysis as I go along here now."
    c.evaluate(TurnObservation(content=same, has_tool_calls=True, tool_count=1))
    c.evaluate(TurnObservation(content=same, has_tool_calls=True, tool_count=1))
    decision = c.evaluate(TurnObservation(content=same, has_tool_calls=True, tool_count=1))
    assert decision.stop is True
    assert decision.terminal_success is False
    assert decision.stop_reason == "content_repetition"
    assert "repetition" in (decision.stop_message or "").lower()


def test_controller_nudges_on_unproductive_plateau():
    c = TurnEvaluationController(enable_budget_warning=False)
    # Distinct short content each turn (no repetition stop), all unproductive -> plateau nudge.
    for n in range(3):
        decision = c.evaluate(
            TurnObservation(content=f"thinking step {n} about the task", productive_count=0)
        )
    assert decision.stop is False
    assert decision.nudge_message is not None
    assert decision.nudge_kind == "plateau"


def test_controller_write_intent_plateau_message():
    c = TurnEvaluationController(enable_budget_warning=False)
    for n in range(3):
        decision = c.evaluate(
            TurnObservation(
                content=f"looking at file {n}", productive_count=0, intent_is_write=True
            )
        )
    assert "edit(" in (decision.nudge_message or "")


def test_controller_clean_turn_continues():
    c = TurnEvaluationController(enable_budget_warning=False)
    decision = c.evaluate(
        TurnObservation(
            content="A clear, distinct, substantive answer.",
            productive_count=1,
            has_tool_calls=True,
            tool_count=1,
            iteration=1,
            max_iterations=8,
        )
    )
    assert decision.stop is False
    assert decision.nudge_message is None


def _saturating_search_obs(iteration):
    same_hits = [{"path": f"a{i}.py", "qualified_name": f"f{i}"} for i in range(5)]
    return TurnObservation(
        content=f"searching, iteration {iteration}",
        has_tool_calls=True,
        tool_count=1,
        tool_results=[{"tool_name": "code_search", "success": True, "result_items": same_hits}],
        iteration=iteration,
        max_iterations=12,
    )


def test_controller_force_completes_on_search_saturation():
    c = TurnEvaluationController(enable_budget_warning=False)
    decision = None
    for i in range(1, 7):
        decision = c.evaluate(_saturating_search_obs(i))
    assert decision.stop is True
    assert decision.terminal_success is True  # synthesize, not fail
    assert decision.stop_reason == "search_saturated"


def test_controller_nudges_to_synthesize_before_force_complete():
    c = TurnEvaluationController(enable_budget_warning=False)
    decisions = [c.evaluate(_saturating_search_obs(i)) for i in range(1, 5)]
    # A synthesize nudge appears before the force-complete.
    assert any(d.nudge_kind == "synthesize" for d in decisions if not d.stop)


def test_controller_distinct_searches_do_not_force_complete():
    c = TurnEvaluationController(enable_budget_warning=False)
    forced = False
    for i in range(1, 12):
        fresh = [{"path": f"f{i}_{j}.py", "qualified_name": f"s{i}_{j}"} for j in range(4)]
        d = c.evaluate(
            TurnObservation(
                content=f"distinct search {i}",
                has_tool_calls=True,
                tool_count=1,
                tool_results=[{"tool_name": "code_search", "success": True, "result_items": fresh}],
                iteration=i,
                max_iterations=12,
            )
        )
        forced = forced or (d.stop and d.stop_reason == "search_saturated")
    assert not forced


def _obs(files, content, iteration, editing=False):
    names = {"code_search"} | ({"edit"} if editing else set())
    results = [
        {
            "tool_name": "code_search",
            "success": True,
            "result_items": [{"path": p, "qualified_name": ""} for p in files],
        }
    ]
    if editing:
        results.append({"tool_name": "edit", "success": True})
    return TurnObservation(
        content=content,
        has_tool_calls=True,
        tool_count=1,
        tool_names=names,
        tool_results=results,
        iteration=iteration,
        max_iterations=20,
    )


_LONG_ANSWER = "The architecture is as follows. " * 40  # ~> 800 chars


# Two SUBSTANTIAL but lexically distinct answers (avoid tripping content-repetition,
# which compares consecutive turns, before reaching the fulfillment check).
_ANSWER_A = "The orchestrator coordinates services, tools, workflows, and shared state. " * 12
_ANSWER_B = (
    "Initialization runs nine phases wiring chat recovery session perception fulfillment. " * 12
)


def _run_fulfillment_scenario(answer, editing=False):
    c = TurnEvaluationController(enable_budget_warning=False)
    decisions = []
    decisions.append(c.evaluate(_obs(["a.py", "b.py", "c.py"], "short", 1)))  # warm-up
    decisions.append(c.evaluate(_obs(["d.py", "e.py", "f.py"], "short", 2)))  # warm-up
    a3 = answer if answer == "still just a short note" else _ANSWER_A
    a4 = answer if answer == "still just a short note" else _ANSWER_B
    decisions.append(c.evaluate(_obs(["a.py", "b.py", "c.py"], a3, 3, editing)))  # low-novelty
    decisions.append(c.evaluate(_obs(["a.py", "b.py", "c.py"], a4, 4, editing)))  # iteration >= 4
    return decisions


def test_fulfillment_completes_when_answer_and_findings():
    decisions = _run_fulfillment_scenario(_LONG_ANSWER)
    assert decisions[-1].stop is True
    assert decisions[-1].terminal_success is True
    assert decisions[-1].stop_reason == "fulfilled"


def test_fulfillment_does_not_fire_without_substantial_answer():
    decisions = _run_fulfillment_scenario("still just a short note")
    assert not any(d.stop and d.stop_reason == "fulfilled" for d in decisions)


def test_fulfillment_does_not_fire_while_editing():
    decisions = _run_fulfillment_scenario(_LONG_ANSWER, editing=True)
    assert not any(d.stop and d.stop_reason == "fulfilled" for d in decisions)


def test_fulfillment_respects_min_iterations():
    # Substantial answer + findings + low novelty, but all at early iterations (< 4) -> no fire.
    c = TurnEvaluationController(enable_budget_warning=False)
    c.evaluate(_obs(["a.py", "b.py", "c.py"], "short", 1))
    c.evaluate(_obs(["d.py", "e.py", "f.py"], "short", 2))
    d3 = c.evaluate(_obs(["a.py", "b.py", "c.py"], _LONG_ANSWER, 3))  # iteration 3 < 4
    assert not (d3.stop and d3.stop_reason == "fulfilled")


def test_controller_reset_clears_state():
    c = TurnEvaluationController()
    same = "Repeating the same narration sentence over and over and over again now."
    c.evaluate(TurnObservation(content=same))
    c.evaluate(TurnObservation(content=same))
    c.reset()
    # After reset, the same content starts fresh (no accumulated repetition).
    assert c.evaluate(TurnObservation(content=same)).stop is False


# --------------------------------------------------------------------------- #
# from_exploration_settings (tunable thresholds + off-switch)
# --------------------------------------------------------------------------- #


def test_from_exploration_settings_applies_thresholds():
    from types import SimpleNamespace

    settings = SimpleNamespace(
        search_novelty_guard_enabled=True,
        novelty_consecutive_low_limit=2,  # force-complete one turn sooner
        novelty_min_search_turns=1,
        fulfillment_completion_enabled=True,
    )
    c = TurnEvaluationController.from_exploration_settings(settings, enable_budget_warning=False)
    decision = None
    for i in range(1, 5):
        decision = c.evaluate(_saturating_search_obs(i))
    assert decision.stop and decision.stop_reason == "search_saturated"


def test_from_exploration_settings_off_switch_disables_novelty():
    from types import SimpleNamespace

    settings = SimpleNamespace(search_novelty_guard_enabled=False)
    c = TurnEvaluationController.from_exploration_settings(settings, enable_budget_warning=False)
    forced = False
    for i in range(1, 10):
        d = c.evaluate(_saturating_search_obs(i))
        forced = forced or (d.stop and d.stop_reason == "search_saturated")
    assert not forced  # guard disabled -> never force-completes


def test_from_exploration_settings_none_uses_defaults():
    c = TurnEvaluationController.from_exploration_settings(None, enable_budget_warning=False)
    decision = None
    for i in range(1, 7):
        decision = c.evaluate(_saturating_search_obs(i))
    assert decision.stop and decision.stop_reason == "search_saturated"

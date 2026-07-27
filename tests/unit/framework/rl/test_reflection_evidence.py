# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Reflection must see the whole prompt, and what actually failed.

Two truncations meant the model diagnosing a section could not read it:
``GEPAService.reflect`` passed ``current_text[:1000]`` and
``GEPAStrategy.reflect``'s LLM prompt passed ``current_text[:500]``. Four of the
seven evolvable sections exceed 1000 characters. Reflection also received only a
histogram of failure categories — "edit_mismatch: 7" — from which no rewrite can
be derived.

Asked to repair failures it could not see, in text it had only partly read, the
mutator returned approximately its input: ``COMPLETION_GUIDANCE`` collapsed to a
whitespace-only diff on two independent evolution runs.
"""

from unittest.mock import MagicMock, patch

from victor.framework.rl.gepa_service import GEPAService
from victor.framework.rl.learners.prompt_optimizer import (
    MAX_EXEMPLAR_CALLS_PER_TRACE,
    MAX_EXEMPLAR_CHARS,
    MAX_EXEMPLAR_TRACES,
    ExecutionTrace,
    ToolCallTrace,
    format_failing_exemplars,
)


def _trace(session_id="sess-abcdef123456", score=0.1, calls=None, **kw):
    return ExecutionTrace(
        session_id=session_id,
        task_type="default",
        provider="zai",
        model="glm-5.2",
        tool_calls=len(calls or []),
        tool_failures=kw.pop("tool_failures", {"edit_mismatch": 1}),
        success=kw.pop("success", False),
        completion_score=score,
        tokens_used=0,
        tool_call_details=calls or [],
        **kw,
    )


def _failed_call(tool="edit", error="old_str not found in a.py; 0 matches", **kw):
    return ToolCallTrace(
        tool_name=tool,
        arguments_summary=kw.pop("arguments_summary", "{'path': 'a.py'}"),
        reasoning_before=kw.pop("reasoning_before", "Replace the helper signature"),
        success=False,
        error_detail=error,
        **kw,
    )


class TestFailingExemplars:
    def test_renders_what_the_agent_did_and_what_came_back(self):
        out = format_failing_exemplars([_trace(calls=[_failed_call()])])
        assert "edit(" in out
        assert "old_str not found" in out
        assert "Replace the helper signature" in out

    def test_clean_traces_add_nothing(self):
        clean = _trace(
            calls=[ToolCallTrace(tool_name="read", success=True)],
            tool_failures={},
            success=True,
            score=1.0,
        )
        assert format_failing_exemplars([clean]) == ""

    def test_no_traces_at_all(self):
        assert format_failing_exemplars([]) == ""

    def test_traces_without_call_detail_are_skipped(self):
        # v1 collection yields aggregate-only traces; nothing to exemplify.
        assert format_failing_exemplars([_trace(calls=[])]) == ""

    def test_worst_traces_come_first(self):
        good = _trace(session_id="good-000000", score=0.9, calls=[_failed_call(error="minor")])
        bad = _trace(session_id="bad-0000000", score=0.0, calls=[_failed_call(error="severe")])
        out = format_failing_exemplars([good, bad])
        assert out.index("bad-00000") < out.index("good-0000")

    def test_respects_the_character_budget(self):
        """The budget is a safety valve for pathological blobs, not the shaper."""
        huge = [
            _trace(session_id=f"s{i:012d}", calls=[_failed_call(error="E" * 5000)])
            for i in range(10)
        ]
        assert len(format_failing_exemplars(huge)) <= MAX_EXEMPLAR_CHARS + 200

    def test_a_full_exemplar_set_is_not_truncated_by_the_budget(self):
        """Structural caps shape the output; the char budget must not bind first.

        At 4000 chars the budget silently dropped whole traces from a
        well-formed 3-trace set, which is the shaping job of MAX_EXEMPLAR_TRACES.
        """
        full = [
            _trace(
                session_id=f"sess{i:09d}",
                score=i / 10,
                calls=[
                    _failed_call(
                        arguments_summary="A" * 200,
                        reasoning_before="R" * 200,
                        error="E" * 300,
                    )
                    for _ in range(MAX_EXEMPLAR_CALLS_PER_TRACE)
                ],
            )
            for i in range(MAX_EXEMPLAR_TRACES)
        ]
        out = format_failing_exemplars(full)
        assert out.count("session ") == MAX_EXEMPLAR_TRACES
        assert len(out) < MAX_EXEMPLAR_CHARS

    def test_caps_the_number_of_traces(self):
        many = [
            _trace(session_id=f"sess{i:09d}", score=i / 100, calls=[_failed_call()])
            for i in range(10)
        ]
        out = format_failing_exemplars(many, max_traces=2)
        assert out.count("session ") == 2

    def test_successful_calls_are_not_reported_as_failures(self):
        mixed = _trace(
            calls=[
                ToolCallTrace(tool_name="read", success=True, result_summary="fine"),
                _failed_call(tool="edit"),
            ]
        )
        out = format_failing_exemplars([mixed])
        assert "edit(" in out
        assert "read(" not in out


class TestReflectSeesTheWholeSection:
    def _service(self):
        # Real constructor, not __new__: hand-built instances silently skip every
        # field added later, and the AttributeError surfaces as an unrelated
        # test failure in whichever method reads the new one.
        return GEPAService(
            provider=None, model="", tier="economic", max_prompt_chars=1500, max_tokens=800
        )

    def test_full_section_reaches_the_model(self):
        """A 1551-char section must arrive whole, not as its first 1000 chars."""
        from victor.agent.prompt_section_texts import COMPLETION_GUIDANCE

        assert len(COMPLETION_GUIDANCE) > 1000, "fixture premise: section exceeds the old cap"
        service = self._service()
        with patch.object(service, "_call_llm", return_value="ok") as call:
            service.reflect("summary", "COMPLETION_GUIDANCE", COMPLETION_GUIDANCE)
        sent = call.call_args[0][1]
        assert COMPLETION_GUIDANCE in sent
        # The tail is what the old cap cut away.
        assert COMPLETION_GUIDANCE[-200:] in sent

    def test_traces_summary_is_forwarded(self):
        service = self._service()
        with patch.object(service, "_call_llm", return_value="ok") as call:
            service.reflect("- edit_mismatch: 7", "SECTION", "short text")
        assert "- edit_mismatch: 7" in call.call_args[0][1]

    def test_failed_llm_call_degrades_to_a_marker(self):
        service = self._service()
        with patch.object(service, "_call_llm", return_value=""):
            out = service.reflect("s", "SECTION", "t")
        assert "Reflection unavailable" in out


class TestStrategyReflectionIncludesExemplars:
    def test_exemplars_appear_alongside_the_counts(self):
        from victor.framework.rl.learners.prompt_optimizer import GEPAStrategy

        strategy = GEPAStrategy.__new__(GEPAStrategy)
        strategy._provider_name = "test"
        strategy._model = "test"
        strategy._llm = None
        with patch.object(strategy, "_call_llm", return_value=""):
            reflection = strategy.reflect(
                section_name="COMPLETION_GUIDANCE",
                current_text="some guidance",
                traces=[_trace(calls=[_failed_call()])],
            )
        # Aggregate shape is retained...
        assert "Success rate" in reflection
        # ...and the specifics that make it actionable are now present.
        assert "Failing exemplars" in reflection
        assert "old_str not found" in reflection


class TestMutationFailureIsVisible:
    """A failed mutation must not masquerade as an evolved candidate.

    The mutate call was rate-limited (429), logged at DEBUG, and fell back to
    returning the seed. A downstream strategy then reformatted that seed, and
    the whitespace-only result was stored and reported as generation N with a
    healthy-looking score — for two full runs, on two different models, because
    neither model ever ran.
    """

    def _service(self):
        return GEPAService(
            provider=None,
            model="glm-5.2",
            tier="balanced",
            max_prompt_chars=1500,
            max_tokens=800,
        )

    def test_failed_mutation_returns_the_seed_and_warns(self, caplog):
        service = self._service()
        with patch.object(service, "_call_llm", return_value=None):
            with caplog.at_level("WARNING"):
                out = service.mutate("SEED TEXT", "reflection", "COMPLETION_GUIDANCE", 1500)
        assert out == "SEED TEXT"
        assert "COMPLETION_GUIDANCE" in caplog.text
        assert "unchanged" in caplog.text

    def test_provider_error_is_warned_not_buried(self, caplog):
        service = self._service()
        service._provider = MagicMock()
        with patch(
            "victor.framework.rl.gepa_service._get_background_loop",
            side_effect=RuntimeError("rate limited (429)"),
        ):
            with caplog.at_level("WARNING"):
                assert service._call_llm("sys", "user") is None
        assert "429" in caplog.text
        assert "NOT be mutated" in caplog.text


class TestPrefPOMergePreservesFormatting:
    """Merging guidance is not a licence to reformat the prompt.

    ``_rewrite_loser`` rebuilt the text from its non-blank lines, so every blank
    line vanished as a side effect of the join. When all additions were already
    present it discarded them for no gain: COMPLETION_GUIDANCE went 1551 -> 1545
    chars with zero semantic change, and that was the whole "evolution".
    """

    def _strategy(self):
        from victor.framework.rl.learners.strategies.prefpo_strategy import PrefPOStrategy

        return PrefPOStrategy.__new__(PrefPOStrategy)

    def test_nothing_to_add_leaves_the_text_byte_identical(self):
        from victor.agent.prompt_section_texts import COMPLETION_GUIDANCE as base

        out = self._strategy()._rewrite_loser(base, "Prefer challenger because it adds:\n", "S")
        assert out == base

    def test_additions_already_present_change_nothing(self):
        # Additions arrive bullet-prefixed, matching how guidance lines appear
        # in the prompt itself — so the containment check compares like with like.
        base = "Rules:\n\n- line one\n\n- line two"
        feedback = "Prefer challenger because it adds:\n- line two"
        assert self._strategy()._rewrite_loser(base, feedback, "S") == base

    def test_a_real_addition_is_appended_without_reflowing(self):
        from victor.agent.prompt_section_texts import COMPLETION_GUIDANCE as base

        feedback = "Prefer challenger because it adds:\n- Verify paths with ls() first."
        out = self._strategy()._rewrite_loser(base, feedback, "S")
        assert out.rstrip().endswith("- Verify paths with ls() first.")
        # The blank lines that structure the section survive.
        assert out.count("\n\n") == base.count("\n\n")
        assert len(out) > len(base)

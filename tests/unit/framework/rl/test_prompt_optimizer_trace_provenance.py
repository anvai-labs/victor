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

"""Evolution evidence must carry the provider it actually came from.

Two defects made prompt evolution provider-blind:

1. ``sessions[sid]["provider"]`` was initialised to ``""`` in both trace
   collectors and never assigned, so every ``ExecutionTrace`` reported
   ``provider="unknown"`` — even though ``session_start`` / ``stream_completed``
   events carry the real value.
2. ``evolve()`` reflected over the *global* ``~/.victor/logs/usage.jsonl`` pool
   while persisting the candidate under a single ``(section, provider)`` scope,
   so ZAI and Ollama failures were attributed to a Moonshot prompt.
"""

from victor.framework.rl.learners.prompt_optimizer import (
    ExecutionTrace,
    PromptOptimizerLearner,
)


def _trace(provider: str, sid: str = "s") -> ExecutionTrace:
    return ExecutionTrace(
        session_id=sid,
        task_type="default",
        provider=provider,
        model="m",
        tool_calls=4,
        tool_failures={},
        success=True,
        completion_score=0.9,
        tokens_used=0,
    )


class TestNormalizeProviderLabel:
    def test_strips_provider_suffix(self):
        assert PromptOptimizerLearner._normalize_provider_label("MoonshotProvider") == "moonshot"

    def test_strips_compat_suffix(self):
        assert (
            PromptOptimizerLearner._normalize_provider_label("MoonshotCompatProvider") == "moonshot"
        )

    def test_strips_gateway_prefix(self):
        # Sandhi fronts the upstream as transport; the prompt scope is the upstream.
        assert PromptOptimizerLearner._normalize_provider_label("SandhiOllamaProvider") == "ollama"

    def test_acronym_provider(self):
        assert PromptOptimizerLearner._normalize_provider_label("ZAIProvider") == "zai"

    def test_already_short_scope_is_idempotent(self):
        assert PromptOptimizerLearner._normalize_provider_label("moonshot") == "moonshot"

    def test_blank_stays_blank(self):
        assert PromptOptimizerLearner._normalize_provider_label("") == ""
        assert PromptOptimizerLearner._normalize_provider_label(None) == ""


class TestAbsorbSessionIdentity:
    def test_fills_provider_and_model_from_any_event(self):
        session = {"provider": "", "model": ""}
        PromptOptimizerLearner._absorb_session_identity(
            session, {"provider": "MoonshotProvider", "model": "kimi-k3"}
        )
        assert session == {"provider": "moonshot", "model": "kimi-k3"}

    def test_first_non_empty_wins(self):
        session = {"provider": "", "model": ""}
        PromptOptimizerLearner._absorb_session_identity(session, {"provider": "ZAIProvider"})
        PromptOptimizerLearner._absorb_session_identity(session, {"provider": "OllamaProvider"})
        assert session["provider"] == "zai"

    def test_events_without_identity_are_inert(self):
        session = {"provider": "", "model": ""}
        PromptOptimizerLearner._absorb_session_identity(session, {"tool_name": "read"})
        assert session == {"provider": "", "model": ""}

    def test_non_dict_payload_is_ignored(self):
        session = {"provider": "", "model": ""}
        PromptOptimizerLearner._absorb_session_identity(session, "not-a-dict")
        assert session == {"provider": "", "model": ""}


class TestScopeTracesToProvider:
    def test_default_scope_keeps_everything(self):
        traces = [_trace("moonshot"), _trace("zai")]
        assert PromptOptimizerLearner._scope_traces_to_provider(traces, "default") is traces

    def test_scopes_to_the_provider_being_evolved(self):
        traces = [_trace("MoonshotProvider", f"m{i}") for i in range(6)]
        traces += [_trace("ZAIProvider", f"z{i}") for i in range(6)]
        scoped = PromptOptimizerLearner._scope_traces_to_provider(traces, "moonshot")
        assert len(scoped) == 6
        assert {PromptOptimizerLearner._normalize_provider_label(t.provider) for t in scoped} == {
            "moonshot"
        }

    def test_falls_back_when_provider_pool_too_small(self):
        # Better a logged mixed pool than a stalled loop.
        traces = [_trace("moonshot", "m0")] + [_trace("zai", f"z{i}") for i in range(6)]
        scoped = PromptOptimizerLearner._scope_traces_to_provider(traces, "moonshot")
        assert scoped == traces

    def test_unknown_provider_traces_are_excluded_from_a_scoped_pool(self):
        traces = [_trace("MoonshotProvider", f"m{i}") for i in range(6)]
        traces += [_trace("unknown", f"u{i}") for i in range(6)]
        scoped = PromptOptimizerLearner._scope_traces_to_provider(traces, "moonshot")
        assert len(scoped) == 6


class TestAbsorbRunKind:
    """Run kind is read off the event, never re-derived from prompt text."""

    def test_reads_the_emitted_tag(self):
        session = {"run_kind": ""}
        PromptOptimizerLearner._absorb_run_kind(session, {"run_kind": "evaluation"})
        assert session["run_kind"] == "evaluation"

    def test_first_non_empty_wins(self):
        session = {"run_kind": ""}
        PromptOptimizerLearner._absorb_run_kind(session, {"run_kind": "evaluation"})
        PromptOptimizerLearner._absorb_run_kind(session, {"run_kind": "delegate"})
        assert session["run_kind"] == "evaluation"

    def test_normalizes_case_and_padding(self):
        session = {"run_kind": ""}
        PromptOptimizerLearner._absorb_run_kind(session, {"run_kind": "  DELEGATE "})
        assert session["run_kind"] == "delegate"

    def test_untagged_events_leave_it_empty(self):
        # Pre-existing log lines carry no tag; those sessions stay unknown
        # rather than being guessed from their prompts.
        session = {"run_kind": ""}
        PromptOptimizerLearner._absorb_run_kind(session, {"session_id": "s", "data": {}})
        assert session["run_kind"] == ""

    def test_non_dict_event_is_ignored(self):
        session = {"run_kind": ""}
        PromptOptimizerLearner._absorb_run_kind(session, "not-a-dict")
        assert session["run_kind"] == ""

    def test_trace_defaults_to_unknown(self):
        assert _trace("moonshot").run_kind == "unknown"

# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Tests for unified context-window resolution and compaction triggers.

Regression coverage for the 2026-11-04 wedged-session class of failures
(victor.log session ed737a8b and predecessors):

- The compaction trigger divided by a char budget decoupled from the model's
  real window while the TUI gauge used the catalog window — compaction never
  fired until long after the model window was exceeded.
- The session-prefill budget trigger was nested inside the utilization-gated
  path, so 4-13M cumulative prefill tokens (10-32x the 400K limit) never
  fired it.
- No log line ever recorded which window/source the trigger used.
"""

import logging
import time

import pytest

from victor.agent.context_compactor import (
    CompactionTrigger,
    CompactorConfig,
    ContextCompactor,
)
from victor.agent.conversation.controller import (
    ConversationConfig,
    ConversationController,
)
from victor.config.config_loaders import get_provider_limits
from victor.config.context_settings import ContextSettings
from victor.config.context_window import (
    DEFAULT_CONTEXT_WINDOW_TOKENS,
    resolve_context_window,
)


class FakeSettings:
    """Minimal settings stand-in exposing the nested context group."""

    def __init__(self, max_context_tokens=None, set_field=False):
        if set_field:
            self.context = ContextSettings(max_context_tokens=max_context_tokens)
        else:
            self.context = ContextSettings()


def _controller(
    messages=None,
    max_context_tokens=None,
    max_context_chars=800_000,
    max_session_prefill_tokens=400_000,
) -> ConversationController:
    controller = ConversationController(
        config=ConversationConfig(
            max_context_chars=max_context_chars,
            max_context_tokens=max_context_tokens,
            context_window_source="model-pattern" if max_context_tokens else "unresolved",
            max_session_prefill_tokens=max_session_prefill_tokens,
        )
    )
    for msg in messages or []:
        controller._history.add_message(msg["role"], msg["content"])
    return controller


# ============================================================================
# Context window resolution
# ============================================================================


class TestResolveContextWindow:
    def test_model_pattern_wins_over_provider_default(self):
        resolution = resolve_context_window("openai", "qwen3.7-plus")
        assert resolution.tokens == 262_144
        assert resolution.source == "model-pattern"
        assert resolution.pattern == "qwen3.7-plus*"

    def test_settings_value_wins_when_explicitly_set(self):
        settings = FakeSettings(max_context_tokens=6000, set_field=True)
        resolution = resolve_context_window("openai", "qwen3.7-plus", settings=settings)
        assert resolution.tokens == 6000
        assert resolution.source == "settings"

    def test_settings_default_does_not_shrink_model_window(self):
        """The nested 100000 default must be ignored — not override 262144."""
        settings = FakeSettings()  # default ContextSettings, field not set
        resolution = resolve_context_window("openai", "qwen3.7-plus", settings=settings)
        assert resolution.source == "model-pattern"
        assert resolution.tokens == 262_144

    def test_explicit_tokens_win_over_everything(self):
        settings = FakeSettings(max_context_tokens=6000, set_field=True)
        resolution = resolve_context_window(
            "openai", "qwen3.7-plus", settings=settings, explicit_tokens=999_999
        )
        assert resolution.tokens == 999_999
        assert resolution.source == "explicit"

    def test_provider_default_used_when_no_model_pattern(self):
        resolution = resolve_context_window("anthropic", "totally-unknown-model")
        assert resolution.source == "provider-default"
        assert resolution.tokens == 200_000

    def test_fallback_window(self):
        resolution = resolve_context_window("no-such-provider", "")
        assert resolution.source == "fallback"
        assert resolution.tokens == DEFAULT_CONTEXT_WINDOW_TOKENS

    def test_chars_budget_matches_legacy_conversion(self):
        resolution = resolve_context_window("openai", "qwen3.7-plus")
        assert resolution.chars_budget == int(262_144 * 2.8)

    def test_describe_includes_source_and_pattern(self):
        resolution = resolve_context_window("openai", "qwen3.7-plus")
        described = resolution.describe()
        assert "window_tokens=262,144" in described
        assert "source=model-pattern:qwen3.7-plus*" in described

    def test_yaml_pattern_ordering_specific_before_broad(self):
        """qwen3.7-plus must resolve to 262144, not the broad qwen3* 131072."""
        assert get_provider_limits("openai", "qwen3.7-plus").context_window == 262_144
        assert get_provider_limits("openai", "qwen3.7-plus-turbo").context_window == 262_144
        assert get_provider_limits("openai", "qwen3-30b").context_window == 131_072


# ============================================================================
# Token-space metrics (the wedge regression)
# ============================================================================


class TestTokenSpaceMetrics:
    def test_real_prompt_tokens_over_window_is_overflow_risk(self):
        """The exact wedge: 145.2K real tokens on a 128K window.

        Old behavior: char-based utilization read ~0.72 (denominator 800K
        chars) — below every threshold, no compaction, session wedged.
        """
        controller = _controller(max_context_tokens=131_072)
        controller._history.add_message("user", "x" * 400_000)
        controller.record_actual_usage(prompt_tokens=145_200, total_chars=400_000)

        metrics = controller.get_context_metrics()
        assert metrics.utilization == 1.0  # capped, but recognized as full
        assert metrics.is_overflow_risk is True

    def test_proactive_threshold_crosses_in_token_space(self):
        """75% proactive threshold in token space fires before the window."""
        controller = _controller(max_context_tokens=100_000)
        controller._history.add_message("user", "x" * 100)
        controller.record_actual_usage(prompt_tokens=80_000, total_chars=100)

        metrics = controller.get_context_metrics()
        assert metrics.utilization >= 0.75
        assert metrics.is_overflow_risk is False  # 80% < 85% overflow line

    def test_projection_adds_only_new_chars(self):
        """After a real reading, new messages project from the real anchor."""
        controller = _controller(max_context_tokens=131_072)
        controller._history.add_message("user", "y" * 200_000)
        controller.record_actual_usage(prompt_tokens=50_000, total_chars=200_000)
        # observed c/t stays 4.0 (200000/50000 == default 4.0)
        controller._history.add_message("assistant", "z" * 8_000)

        metrics = controller.get_context_metrics()
        assert metrics.estimated_tokens == 50_000 + 8_000 // 4

    def test_char_space_fallback_unchanged_without_window(self):
        """No token window → legacy char-space behavior is preserved."""
        controller = _controller(max_context_tokens=None, max_context_chars=100_000)
        controller._history.add_message("user", "x" * 90_000)

        metrics = controller.get_context_metrics()
        assert metrics.utilization == pytest.approx(0.9)
        assert metrics.max_context_tokens is None

    def test_metrics_expose_window_provenance(self):
        controller = _controller(max_context_tokens=262_144)
        metrics = controller.get_context_metrics()
        assert metrics.max_context_tokens == 262_144
        assert metrics.context_window_source == "model-pattern"
        assert metrics.session_prompt_tokens == 0


# ============================================================================
# Budget trigger no longer dead-gated
# ============================================================================


class TestBudgetTrigger:
    def test_budget_due_with_low_utilization_fires(self):
        """The dead-gating bug: prefill 32x over limit, utilization fine.

        Old behavior: check_and_compact returned NONE (utilization below
        threshold) and the budget check inside smart_compact_history was
        never reached. New behavior: BUDGET trigger fires.
        """
        messages = [{"role": "user", "content": "hello"}] * 12
        controller = _controller(
            messages=messages,
            max_context_tokens=1_000_000,
            max_session_prefill_tokens=400_000,
        )
        controller._session_prompt_tokens = 13_103_747  # the real incident value

        compactor = ContextCompactor(controller=controller)
        action = compactor.check_and_compact(current_query="hi")

        assert action.trigger is CompactionTrigger.BUDGET
        assert action.action_taken

    def test_budget_disabled_never_fires(self):
        messages = [{"role": "user", "content": "hello"}] * 12
        controller = _controller(messages=messages, max_session_prefill_tokens=None)
        controller._session_prompt_tokens = 13_103_747

        compactor = ContextCompactor(controller=controller)
        action = compactor.check_and_compact(current_query="hi")
        assert action.trigger is CompactionTrigger.NONE

    def test_should_compact_reports_budget_trigger(self):
        controller = _controller(max_context_tokens=1_000_000)
        controller._session_prompt_tokens = 500_000

        compactor = ContextCompactor(controller=controller)
        should, trigger = compactor.should_compact()
        assert should is True
        assert trigger is CompactionTrigger.BUDGET


# ============================================================================
# Stampede suppression
# ============================================================================


class TestStampedeSuppression:
    def _controller_near_threshold(self) -> ConversationController:
        # Token window 100K; real prompt 80K → THRESHOLD (75%) not OVERFLOW.
        messages = [{"role": "user", "content": "hello"}] * 12
        controller = _controller(messages=messages, max_context_tokens=100_000)
        controller.record_actual_usage(prompt_tokens=80_000, total_chars=10)
        return controller

    def test_duplicate_trigger_suppressed_after_recent_compaction(self):
        controller = self._controller_near_threshold()
        controller._last_compaction_time = time.time()  # compacted 0s ago

        compactor = ContextCompactor(controller=controller)
        action = compactor.check_and_compact(current_query="hi")

        assert action.trigger is CompactionTrigger.NONE
        assert any("Suppressed" in d for d in action.details)

    def test_overflow_not_suppressed_by_interval(self):
        """Real overflow must always pass through despite a recent run."""
        controller = self._controller_near_threshold()
        controller.record_actual_usage(prompt_tokens=99_000, total_chars=10)
        controller._last_compaction_time = time.time()

        compactor = ContextCompactor(controller=controller)
        should, trigger = compactor.should_compact()
        assert should is True
        assert trigger is CompactionTrigger.OVERFLOW


# ============================================================================
# Observability traces
# ============================================================================


class TestCompactionTraces:
    def test_skip_decision_logs_compaction_check_line(self, caplog):
        controller = _controller(
            messages=[{"role": "user", "content": "hi"}],
            max_context_tokens=131_072,
        )
        compactor = ContextCompactor(controller=controller)

        with caplog.at_level(logging.INFO, logger="victor.agent.context_compactor"):
            compactor.check_and_compact(current_query="hi")

        trace = [r.message for r in caplog.records if "[compaction-check]" in r.message]
        assert trace, "expected a [compaction-check] decision trace"
        line = trace[0]
        assert "decision=skip" in line
        assert "window=131,072 (model-pattern)" in line
        assert "util_token=" in line
        assert "budget=" in line

    def test_trigger_decision_logs_reason(self, caplog):
        messages = [{"role": "user", "content": "hello"}] * 12
        controller = _controller(messages=messages, max_context_tokens=100_000)
        controller._session_prompt_tokens = 999_999

        compactor = ContextCompactor(controller=controller)
        with caplog.at_level(logging.INFO, logger="victor.agent.context_compactor"):
            compactor.check_and_compact(current_query="hi")

        trace = [r.message for r in caplog.records if "[compaction-check]" in r.message]
        assert trace and "decision=trigger:budget" in trace[0]
        assert "DUE" in trace[0]

    def test_resolution_line_logged_by_char_calculator(self, caplog):
        from victor.agent.orchestrator_utils import calculate_max_context_chars

        resolution = resolve_context_window("openai", "qwen3.7-plus")
        with caplog.at_level(logging.INFO, logger="victor.agent.orchestrator_utils"):
            chars = calculate_max_context_chars(
                FakeSettings(), "openai", "qwen3.7-plus", window_resolution=resolution
            )
        assert chars == resolution.chars_budget
        logged = [r.message for r in caplog.records if "Model context" in r.message]
        assert logged and "262,144" in logged[0] and "source=model-pattern" in logged[0]

    def test_settings_char_override_now_logs(self, caplog):
        """The silent 800K early return must leave a trace."""
        from victor.agent.orchestrator_utils import calculate_max_context_chars

        class CharOverrideSettings(FakeSettings):
            max_context_chars = 800_000

        resolution = resolve_context_window("openai", "qwen3.7-plus")
        with caplog.at_level(logging.INFO, logger="victor.agent.orchestrator_utils"):
            chars = calculate_max_context_chars(
                CharOverrideSettings(), "openai", "qwen3.7-plus", window_resolution=resolution
            )
        assert chars == 800_000
        logged = [r.message for r in caplog.records if "char-override" in r.message]
        assert logged and "800,000" in logged[0]


class TestCompactorConfigDefaults:
    def test_compactor_config_accepts_new_fields_unchanged_defaults(self):
        config = CompactorConfig()
        assert config.proactive_threshold == 0.75
        assert config.enable_proactive is True

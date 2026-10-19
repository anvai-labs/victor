"""Tests for ConversationController budget-aware compaction (mirror of controller.py).

Cumulative session prefill volume (real provider prompt_tokens) crossing the
configured limit forces compaction at a turn boundary even under window
utilization — un-compacted history resend amplification is what burned token
budgets (measured 2026-10-18: 813k tokens across 24 calls of one turn).
"""

from __future__ import annotations

from victor.agent.conversation.controller import ConversationConfig, ConversationController
from victor.config.conversation_settings import DEFAULT_MAX_SESSION_PREFILL_TOKENS
from victor.config.settings import Settings


def _controller(limit=None) -> ConversationController:
    config = ConversationConfig(max_session_prefill_tokens=limit)
    return ConversationController(config=config)


def test_prefill_accumulates_from_recorded_usage():
    controller = _controller()
    controller.record_actual_usage(1_000, 4_000)
    controller.record_actual_usage(2_500, 4_000)
    assert controller._session_prompt_tokens == 3_500


def test_spend_threshold_triggers_despite_low_utilization():
    controller = _controller(limit=10_000)
    controller.record_actual_usage(10_000, 40_000)
    assert controller._should_compact(current_utilization=0.2) is True


def test_below_threshold_low_utilization_does_not_trigger():
    controller = _controller(limit=10_000)
    controller.record_actual_usage(9_999, 40_000)
    assert controller._should_compact(current_utilization=0.2) is False


def test_none_disables_spend_trigger():
    controller = _controller(limit=None)
    controller.record_actual_usage(10_000_000, 40_000_000)
    assert controller._should_compact(current_utilization=0.2) is False


def test_min_interval_still_suppresses_spend_trigger():
    controller = _controller(limit=10_000)
    controller.record_actual_usage(10_000, 40_000)
    # A compaction just happened: the interval guard suppresses even the
    # spend trigger (aggressive compaction loses context).
    import time

    controller._last_compaction_time = time.time()
    assert controller._should_compact(current_utilization=0.2) is False


def test_spend_trigger_logs_boundary_event(caplog):
    import logging

    controller = _controller(limit=5_000)
    controller.record_actual_usage(5_000, 20_000)
    with caplog.at_level(logging.INFO, logger="victor.agent.conversation.controller"):
        assert controller._should_compact(current_utilization=0.1) is True
    assert any("[budget-compaction]" in r.message for r in caplog.records)


def test_settings_and_config_defaults_agree():
    assert ConversationConfig().max_session_prefill_tokens == DEFAULT_MAX_SESSION_PREFILL_TOKENS
    assert Settings().max_session_prefill_tokens == DEFAULT_MAX_SESSION_PREFILL_TOKENS

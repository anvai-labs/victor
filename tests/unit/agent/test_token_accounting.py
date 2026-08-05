# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Token accounting unit tests.

Validates that actual API-returned prompt_tokens flow back to
ConversationController so get_context_metrics() uses real counts
instead of char/N estimation.
"""

from unittest.mock import MagicMock, patch

import pytest

from victor.agent.conversation.controller import (
    ConversationController,
    ConversationConfig,
)


class TestRecordActualUsage:
    """ConversationController.record_actual_usage() correctness."""

    def _controller(self):
        return ConversationController(config=ConversationConfig(chars_per_token_estimate=4))

    def test_initial_state_uses_char_estimate(self):
        """Before any actual usage is recorded, falls back to char division."""
        ctrl = self._controller()
        ctrl.add_user_message("hello world")  # 11 chars
        metrics = ctrl.get_context_metrics()
        # 11 chars // 4 estimate = 2 (plus system prompt if any)
        assert metrics.estimated_tokens >= 0  # No crash
        assert ctrl._last_actual_prompt_tokens is None

    def test_record_actual_usage_sets_last_tokens(self):
        """record_actual_usage stores prompt_tokens."""
        ctrl = self._controller()
        ctrl.record_actual_usage(prompt_tokens=150, total_chars=600)
        assert ctrl._last_actual_prompt_tokens == 150

    def test_get_context_metrics_uses_actual_when_set(self):
        """get_context_metrics uses calibrated ratio once actual usage is recorded."""
        ctrl = self._controller()
        ctrl.add_user_message("A" * 400)  # 400 chars

        # Record that the API said 100 tokens for 400 chars → ratio = 4.0
        ctrl.record_actual_usage(prompt_tokens=100, total_chars=400)

        metrics = ctrl.get_context_metrics()
        # Should use calibrated 4.0 ratio (same as default here, but via new path)
        # With 400 chars and ratio=4.0, estimated ~ 100
        assert metrics.estimated_tokens > 0
        # Must not use raw //4 anymore — uses float division via calibrated ratio
        assert isinstance(metrics.estimated_tokens, int)

    def test_calibrated_ratio_rolling_average(self):
        """Observed chars-per-token updates as rolling 80/20 average."""
        ctrl = self._controller()
        # Starting ratio = 4.0 (from config)
        # Observe 300 chars / 100 tokens = 3.0 chars/token
        ctrl.record_actual_usage(prompt_tokens=100, total_chars=300)
        # Expected: 0.8*4.0 + 0.2*3.0 = 3.8 (approx, due to initial init)
        # Initial _observed_chars_per_token = float(4) = 4.0
        expected = 0.8 * 4.0 + 0.2 * 3.0
        assert abs(ctrl._observed_chars_per_token - expected) < 0.01

    def test_calibrated_ratio_clamped_to_2_8(self):
        """Ratio stays within [2, 8] even for extreme observations."""
        ctrl = self._controller()
        # Extreme: 1 char / 100 tokens → 0.01 (should clamp to 2.0)
        ctrl.record_actual_usage(prompt_tokens=100, total_chars=1)
        assert ctrl._observed_chars_per_token >= 2.0

        ctrl2 = self._controller()
        # Extreme: 80000 chars / 100 tokens → 800 (should clamp to 8.0)
        ctrl2.record_actual_usage(prompt_tokens=100, total_chars=80000)
        assert ctrl2._observed_chars_per_token <= 8.0

    def test_zero_prompt_tokens_ignored(self):
        """record_actual_usage is a no-op for zero or negative tokens."""
        ctrl = self._controller()
        ctrl.record_actual_usage(prompt_tokens=0, total_chars=400)
        assert ctrl._last_actual_prompt_tokens is None

        ctrl.record_actual_usage(prompt_tokens=-5, total_chars=400)
        assert ctrl._last_actual_prompt_tokens is None

    def test_get_context_metrics_fallback_without_actual(self):
        """Falls back to integer char division when no actual usage recorded."""
        ctrl = self._controller()
        ctrl.add_user_message("hello")
        metrics = ctrl.get_context_metrics()
        total_chars = sum(len(m.content) for m in ctrl.messages)
        expected_fallback = total_chars // ctrl.config.chars_per_token_estimate
        assert metrics.estimated_tokens == expected_fallback

    def test_multiple_updates_converge_ratio(self):
        """Multiple record_actual_usage calls converge the calibrated ratio."""
        ctrl = self._controller()
        # Repeatedly observe 3.0 chars/token
        for _ in range(20):
            ctrl.record_actual_usage(prompt_tokens=100, total_chars=300)
        # After many updates, ratio should converge near 3.0
        assert abs(ctrl._observed_chars_per_token - 3.0) < 0.1


class TestTokenAccountingWiring:
    """TurnExecutor and ChatCoordinator wire record_actual_usage correctly."""

    def test_legacy_turn_executor_module_reexports_service_runtime(self):
        """Legacy coordinator import path should re-export the service runtime."""
        from victor.agent.coordinators.turn_executor import (
            TurnExecutor as legacy_turn_executor,
        )
        from victor.agent.services.turn_execution_runtime import (
            TurnExecutor as service_turn_executor,
        )

        assert legacy_turn_executor is service_turn_executor

    def test_accumulate_calls_record_actual_usage(self):
        """_accumulate_token_usage() calls record_actual_usage on the controller."""
        from victor.providers.base import CompletionResponse
        from victor.agent.services.turn_execution_runtime import TurnExecutor

        mock_ctrl = MagicMock()
        mock_ctrl.messages = [MagicMock(content="hello world")]

        mock_conversation = MagicMock()
        mock_conversation.conversation = mock_ctrl

        executor = MagicMock(spec=TurnExecutor)
        executor._chat_context = mock_conversation
        executor._token_tracker = None

        # Bind the real method to our mock instance
        TurnExecutor._accumulate_token_usage(
            executor,
            CompletionResponse(
                content="response",
                usage={
                    "prompt_tokens": 50,
                    "completion_tokens": 20,
                    "total_tokens": 70,
                },
            ),
        )

        mock_ctrl.record_actual_usage.assert_called_once()
        call_args = mock_ctrl.record_actual_usage.call_args
        # Called as record_actual_usage(prompt_tokens, total_chars) positionally
        assert call_args[0][0] == 50

    def test_accumulate_no_crash_when_controller_raises(self):
        """_accumulate_token_usage() never raises even if record_actual_usage fails."""
        from victor.providers.base import CompletionResponse
        from victor.agent.services.turn_execution_runtime import TurnExecutor

        mock_ctrl = MagicMock()
        mock_ctrl.record_actual_usage.side_effect = RuntimeError("DB down")
        mock_ctrl.messages = []

        mock_conversation = MagicMock()
        mock_conversation.conversation = mock_ctrl

        executor = MagicMock(spec=TurnExecutor)
        executor._chat_context = mock_conversation
        executor._token_tracker = None

        # Should not raise
        TurnExecutor._accumulate_token_usage(
            executor,
            CompletionResponse(
                content="ok",
                usage={
                    "prompt_tokens": 30,
                    "completion_tokens": 10,
                    "total_tokens": 40,
                },
            ),
        )

    def test_accumulate_skips_record_when_prompt_tokens_zero(self):
        """record_actual_usage is NOT called when prompt_tokens is 0."""
        from victor.providers.base import CompletionResponse
        from victor.agent.services.turn_execution_runtime import TurnExecutor

        mock_ctrl = MagicMock()
        mock_ctrl.messages = []
        mock_conversation = MagicMock()
        mock_conversation.conversation = mock_ctrl

        executor = MagicMock(spec=TurnExecutor)
        executor._chat_context = mock_conversation
        executor._token_tracker = None

        TurnExecutor._accumulate_token_usage(
            executor,
            CompletionResponse(
                content="ok",
                usage={"prompt_tokens": 0, "completion_tokens": 5, "total_tokens": 5},
            ),
        )

        mock_ctrl.record_actual_usage.assert_not_called()


class TestConversationStoreTokenSchema:
    """ConversationStore sessions table has token accounting columns."""

    def test_sessions_table_has_token_columns(self, tmp_path):
        """New DB must have prompt_tokens, completion_tokens, cached_tokens, etc."""
        import sqlite3
        from victor.agent.conversation.store import ConversationStore

        db_path = tmp_path / "test.db"
        store = ConversationStore(db_path=db_path)
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute("PRAGMA table_info(sessions)")
        cols = {row[1] for row in cursor.fetchall()}
        conn.close()

        for col in (
            "prompt_tokens",
            "completion_tokens",
            "cached_tokens",
            "reasoning_tokens",
            "cost_usd_micros",
        ):
            assert col in cols, f"sessions table missing column: {col}"

    def test_update_session_token_usage_accumulates(self, tmp_path):
        """update_session_token_usage adds to existing values on repeated calls."""
        import sqlite3
        from victor.agent.conversation.store import ConversationStore

        db_path = tmp_path / "test.db"
        store = ConversationStore(db_path=db_path)
        session_id = "test-session-token"
        store.create_session(session_id)

        store.update_session_token_usage(session_id, prompt_tokens=100, completion_tokens=50)
        store.update_session_token_usage(session_id, prompt_tokens=200, completion_tokens=80)

        conn = sqlite3.connect(str(db_path))
        row = conn.execute(
            "SELECT prompt_tokens, completion_tokens FROM sessions WHERE session_id=?",
            (session_id,),
        ).fetchone()
        conn.close()

        assert row[0] == 300, f"Expected 300 prompt_tokens, got {row[0]}"
        assert row[1] == 130, f"Expected 130 completion_tokens, got {row[1]}"

    def test_update_session_token_usage_no_crash_for_missing_session(self, tmp_path):
        """Calling update for a non-existent session_id must not raise."""
        from victor.agent.conversation.store import ConversationStore

        store = ConversationStore(db_path=tmp_path / "test.db")
        store.update_session_token_usage("nonexistent", prompt_tokens=50, completion_tokens=20)

    def test_migration_adds_token_columns_to_existing_db(self, tmp_path):
        """Existing DB without token columns gets them added via migration."""
        import sqlite3

        db_path = str(tmp_path / "old.db")
        # Create a minimal old sessions table without token columns
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                created_at TIMESTAMP NOT NULL,
                last_activity TIMESTAMP NOT NULL
            )
        """)
        conn.commit()
        conn.close()

        # Opening ConversationStore should migrate the table
        from pathlib import Path
        from victor.agent.conversation.store import ConversationStore

        ConversationStore(db_path=Path(db_path))

        conn = sqlite3.connect(db_path)
        cursor = conn.execute("PRAGMA table_info(sessions)")
        cols = {row[1] for row in cursor.fetchall()}
        conn.close()

        assert "prompt_tokens" in cols
        assert "completion_tokens" in cols


class TestExtendedTokenFieldsSourcing:
    """F1: extended token fields come from neutral usage + transport
    diagnostics; the raw native body is only a legacy fallback and the
    accumulator must behave identically when it is absent."""

    @staticmethod
    def _accumulate(response):
        from victor.agent.services.turn_execution_runtime import TurnExecutor

        cum = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        executor = MagicMock(spec=TurnExecutor)
        executor._chat_context = MagicMock()
        executor._chat_context._cumulative_token_usage = cum
        executor._token_tracker = None
        TurnExecutor._accumulate_token_usage(executor, response)
        return cum

    def test_neutral_usage_without_raw_body(self):
        """Typed transports with extensions absent lose nothing."""
        from victor.providers.base import CompletionResponse

        cum = self._accumulate(
            CompletionResponse(
                content="r",
                usage={
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                    "cache_read_input_tokens": 4,
                    "reasoning_tokens": 3,
                },
            )
        )
        assert cum["cached_tokens"] == 4
        assert cum["reasoning_tokens"] == 3
        assert cum["cache_miss_tokens"] == 0
        assert cum["cost_usd_micros"] == 0

    def test_diagnostics_supply_native_only_fields(self):
        from victor.providers.base import CompletionResponse

        cum = self._accumulate(
            CompletionResponse(
                content="r",
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                metadata={"sandhi_usage": {"cache_miss_tokens": 7, "cost_in_usd_ticks": 123}},
            )
        )
        assert cum["cache_miss_tokens"] == 7
        assert cum["cost_usd_micros"] == 123

    def test_legacy_raw_body_fallback_still_works(self):
        """Non-typed providers keep their raw-usage extraction unchanged."""
        from victor.providers.base import CompletionResponse

        cum = self._accumulate(
            CompletionResponse(
                content="r",
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                raw_response={
                    "usage": {
                        "prompt_tokens_details": {"cached_tokens": 6},
                        "completion_tokens_details": {"reasoning_tokens": 2},
                        "prompt_cache_miss_tokens": 9,
                        "cost_in_usd_ticks": 55,
                    }
                },
            )
        )
        assert cum["cached_tokens"] == 6
        assert cum["reasoning_tokens"] == 2
        assert cum["cache_miss_tokens"] == 9
        assert cum["cost_usd_micros"] == 55

    def test_neutral_usage_wins_over_raw_body(self):
        """No double counting: neutral values take precedence over the body."""
        from victor.providers.base import CompletionResponse

        cum = self._accumulate(
            CompletionResponse(
                content="r",
                usage={
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                    "cache_read_input_tokens": 4,
                    "reasoning_tokens": 3,
                },
                metadata={"sandhi_usage": {"cache_miss_tokens": 7}},
                raw_response={
                    "usage": {
                        "prompt_tokens_details": {"cached_tokens": 999},
                        "completion_tokens_details": {"reasoning_tokens": 999},
                        "prompt_cache_miss_tokens": 999,
                    }
                },
            )
        )
        assert cum["cached_tokens"] == 4
        assert cum["reasoning_tokens"] == 3
        assert cum["cache_miss_tokens"] == 7

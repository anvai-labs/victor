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

"""
Unit tests for provider improvement components.

Tests:
- ConversationStore (formerly ConversationStore)
- StreamingMetricsCollector
- CircuitBreaker and ProviderRetryStrategy
- RequestQueue (formerly RequestQueue)
"""

import asyncio
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

# Import components
from victor.agent.conversation.store import ConversationStore
from victor.agent.conversation.types import MessagePriority, MessageRole
from victor.observability.analytics.streaming_metrics import (
    StreamingMetricsCollector,
    StreamMetrics,
)
from victor.providers.concurrency import (
    ConcurrencyConfig,
    RequestQueue,
    RequestPriority,
    SlidingWindowRateLimiter,
    TokenBucketRateLimiter,
)
from victor.providers.base import CompletionResponse
from victor.providers.resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitOpenError,
    CircuitState,
    ResilientProvider,
    ProviderRetryConfig,
    RetryExhaustedError,
    ProviderRetryStrategy,
)

# =============================================================================
# Conversation Memory Tests
# =============================================================================


class TestConversationStore:
    """Tests for ConversationStore."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_conversations.db"
            yield db_path

    @pytest.fixture
    def manager(self, temp_db):
        """Create memory manager with temp database."""
        return ConversationStore(db_path=temp_db)

    def test_create_session(self, manager):
        """Test session creation."""
        session = manager.create_session(project_path="/test/project")

        assert session.session_id.startswith("session_")
        assert session.project_path == "/test/project"
        assert session.current_tokens == 0
        assert len(session.messages) == 0

    def test_add_message(self, manager):
        """Test adding messages."""
        session = manager.create_session()

        msg = manager.add_message(
            session.session_id,
            MessageRole.USER,
            "Hello, world!",
        )

        assert msg.role == MessageRole.USER
        assert msg.content == "Hello, world!"
        assert msg.token_count > 0

        # Check session updated
        retrieved = manager.get_session(session.session_id)
        assert len(retrieved.messages) == 1
        assert retrieved.current_tokens == msg.token_count

    def test_message_priority(self, manager):
        """Test message priority assignment."""
        session = manager.create_session()

        # System messages should be CRITICAL
        sys_msg = manager.add_system_message(session.session_id, "System prompt")
        assert sys_msg.priority == MessagePriority.CRITICAL

        # User messages should be HIGH
        user_msg = manager.add_message(
            session.session_id,
            MessageRole.USER,
            "User input",
        )
        assert user_msg.priority == MessagePriority.HIGH

    def test_get_context_messages(self, manager):
        """Test context message retrieval."""
        session = manager.create_session()

        # Add some messages
        manager.add_system_message(session.session_id, "System")
        manager.add_message(session.session_id, MessageRole.USER, "User 1")
        manager.add_message(session.session_id, MessageRole.ASSISTANT, "Assistant 1")
        manager.add_message(session.session_id, MessageRole.USER, "User 2")

        messages = manager.get_context_messages(session.session_id)

        assert len(messages) == 4
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_session_persistence(self, manager):
        """Test session persistence to database."""
        session = manager.create_session(project_path="/test")
        manager.add_message(session.session_id, MessageRole.USER, "Test message")

        # Create new manager with same database
        manager2 = ConversationStore(db_path=manager.db_path)

        # Load session
        loaded = manager2.get_session(session.session_id)

        assert loaded is not None
        assert loaded.project_path == "/test"
        assert len(loaded.messages) == 1

    def test_context_pruning(self, manager):
        """Test context pruning when tokens exceed limit."""
        # Create manager with low token limit
        small_manager = ConversationStore(
            db_path=manager.db_path,
            max_context_tokens=100,
            response_reserve=20,
        )

        session = small_manager.create_session()

        # Add many messages to exceed limit
        for i in range(20):
            small_manager.add_message(
                session.session_id,
                MessageRole.USER,
                f"Message {i} with some content to use tokens",
            )

        # Check that pruning occurred
        assert session.current_tokens < 100

    def test_session_stats(self, manager):
        """Test session statistics."""
        session = manager.create_session()

        manager.add_message(session.session_id, MessageRole.USER, "Hello")
        manager.add_message(session.session_id, MessageRole.ASSISTANT, "Hi there")
        manager.add_message(session.session_id, MessageRole.TOOL, "Tool output")

        stats = manager.get_session_stats(session.session_id)

        assert stats["message_count"] == 3
        assert stats["role_distribution"]["user"] == 1
        assert stats["role_distribution"]["assistant"] == 1


# =============================================================================
# Streaming Metrics Tests
# =============================================================================


class TestStreamingMetrics:
    """Tests for streaming metrics collection."""

    def test_stream_metrics_creation(self):
        """Test StreamMetrics creation."""
        metrics = StreamMetrics(
            request_id="test_123",
            model="claude-3-5-haiku",
            provider="anthropic",
            start_time=time.time(),
        )

        assert metrics.request_id == "test_123"
        assert metrics.total_chunks == 0
        assert metrics.ttft_ms is None

    def test_stream_metrics_calculations(self):
        """Test StreamMetrics calculations."""
        start = time.time()
        metrics = StreamMetrics(
            request_id="test",
            model="test",
            provider="test",
            start_time=start,
            first_token_time=start + 0.1,
            last_token_time=start + 1.0,
            total_tokens=100,
            chunk_intervals=[0.05, 0.06, 0.04, 0.05, 0.07],
        )

        # TTFT should be ~100ms
        assert metrics.ttft_ms is not None
        assert 90 < metrics.ttft_ms < 110

        # Duration should be ~1000ms
        assert metrics.total_duration_ms is not None
        assert 900 < metrics.total_duration_ms < 1100

        # Tokens per second
        assert metrics.tokens_per_second is not None
        assert 90 < metrics.tokens_per_second < 110

    def test_metrics_collector(self):
        """Test StreamingMetricsCollector."""
        collector = StreamingMetricsCollector(max_history=10)

        # Create and record metrics
        for i in range(5):
            metrics = collector.create_metrics(
                f"req_{i}",
                "claude-3-5-haiku",
                "anthropic",
            )
            metrics.first_token_time = metrics.start_time + 0.1
            metrics.last_token_time = metrics.start_time + 0.5
            metrics.total_tokens = 50
            collector.record_metrics_sync(metrics)

        # Get summary
        summary = collector.get_summary()

        assert summary.count == 5
        assert summary.ttft_ms["avg"] is not None

    def test_metrics_callback(self):
        """Test metrics callback notification."""
        collector = StreamingMetricsCollector()
        received = []

        def callback(metrics: StreamMetrics):
            received.append(metrics)

        collector.on_metrics(callback)

        metrics = collector.create_metrics("test", "model", "provider")
        collector.record_metrics_sync(metrics)

        assert len(received) == 1
        assert received[0].request_id == "test"


# =============================================================================
# Circuit Breaker Tests
# =============================================================================


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""

    @pytest.fixture
    def circuit(self):
        """Create circuit breaker with low thresholds for testing."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            success_threshold=2,
            timeout_seconds=0.1,
        )
        return CircuitBreaker("test", config)

    @pytest.mark.asyncio
    async def test_circuit_closed_success(self, circuit):
        """Test circuit stays closed on success."""

        async def success():
            return "ok"

        result = await circuit.execute(success)

        assert result == "ok"
        assert circuit.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_circuit_opens_on_failures(self, circuit):
        """Test circuit opens after failures."""

        async def fail():
            raise ConnectionError("Failed")

        # First failure
        with pytest.raises(ConnectionError):
            await circuit.execute(fail)

        assert circuit.state == CircuitState.CLOSED

        # Second failure - should open
        with pytest.raises(ConnectionError):
            await circuit.execute(fail)

        assert circuit.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_circuit_rejects_when_open(self, circuit):
        """Test circuit rejects requests when open."""

        async def fail():
            raise ConnectionError("Failed")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ConnectionError):
                await circuit.execute(fail)

        # Should reject immediately
        with pytest.raises(CircuitOpenError):
            await circuit.execute(fail)

    @pytest.mark.asyncio
    async def test_circuit_half_open_recovery(self, circuit):
        """Test circuit transitions to half-open and recovers."""

        async def fail():
            raise ConnectionError("Failed")

        async def success():
            return "ok"

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ConnectionError):
                await circuit.execute(fail)

        # Wait for timeout
        await asyncio.sleep(0.15)

        # Should transition to half-open and allow request
        result = await circuit.execute(success)
        assert result == "ok"

        # Another success should close it
        result = await circuit.execute(success)
        assert result == "ok"
        assert circuit.state == CircuitState.CLOSED


# =============================================================================
# Retry Strategy Tests
# =============================================================================


class TestRetryStrategy:
    """Tests for ProviderRetryStrategy."""

    @pytest.fixture
    def retry(self):
        """Create retry strategy with low delays for testing."""
        config = ProviderRetryConfig(
            max_retries=2,
            base_delay_seconds=0.01,
            max_delay_seconds=0.1,
        )
        return ProviderRetryStrategy(config)

    @pytest.mark.asyncio
    async def test_retry_success_first_try(self, retry):
        """Test success on first try."""

        async def success():
            return "ok"

        result = await retry.execute(success)
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_retry_success_after_failures(self, retry):
        """Test success after transient failures."""
        attempts = [0]

        async def flaky():
            attempts[0] += 1
            if attempts[0] < 2:
                raise ConnectionError("Transient failure")
            return "ok"

        result = await retry.execute(flaky)

        assert result == "ok"
        assert attempts[0] == 2

    @pytest.mark.asyncio
    async def test_retry_reports_retry_events_to_callback(self, retry):
        """Retry diagnostics should be available to callers that need metadata."""
        attempts = [0]
        events = []

        async def flaky():
            attempts[0] += 1
            if attempts[0] < 2:
                raise ConnectionError("Server disconnected without sending a response")
            return "ok"

        result = await retry.execute(flaky, retry_event_callback=events.append)

        assert result == "ok"
        assert len(events) == 1
        assert events[0]["attempt"] == 1
        assert events[0]["max_retries"] == 2
        assert events[0]["error_type"] == "ConnectionError"
        assert "Server disconnected" in events[0]["error"]

    @pytest.mark.asyncio
    async def test_retry_exhausted(self, retry):
        """Test retry exhaustion."""

        async def always_fail():
            raise ConnectionError("Always fails")

        with pytest.raises(RetryExhaustedError) as exc_info:
            await retry.execute(always_fail)

        assert exc_info.value.max_retries == 2

    @pytest.mark.asyncio
    async def test_non_retryable_error(self, retry):
        """Test non-retryable error is raised immediately."""

        async def value_error():
            raise ValueError("Not retryable")

        with pytest.raises(ValueError):
            await retry.execute(value_error)

    @pytest.mark.asyncio
    async def test_hard_quota_rate_limit_is_not_retried(self, retry):
        """Quota/billing 429 failures should fail fast without burning retries."""
        attempts = [0]
        request = httpx.Request("POST", "https://example.com/chat/completions")
        response = httpx.Response(
            429,
            request=request,
            text='{"error":{"message":"insufficient quota for this account"}}',
        )

        async def quota_exhausted():
            attempts[0] += 1
            raise httpx.HTTPStatusError(
                "429 Too Many Requests",
                request=request,
                response=response,
            )

        with pytest.raises(httpx.HTTPStatusError):
            await retry.execute(quota_exhausted)

        assert attempts[0] == 1

    @pytest.mark.asyncio
    async def test_retries_open_circuit_breaker(self, retry, monkeypatch):
        """Open circuits should be retried after their cooldown."""
        from victor.providers.circuit_breaker import (
            CircuitBreakerError as CanonicalCircuitBreakerError,
        )

        attempts = [0]
        sleeps: list[float] = []

        async def flaky():
            attempts[0] += 1
            if attempts[0] == 1:
                raise CanonicalCircuitBreakerError(
                    "Circuit breaker 'provider_OpenAIProvider' is OPEN. Retry after 0.5s",
                    state=CircuitState.OPEN,
                    retry_after=0.5,
                )
            return "ok"

        async def fake_sleep(delay: float) -> None:
            sleeps.append(delay)

        monkeypatch.setattr(asyncio, "sleep", fake_sleep)

        result = await retry.execute(flaky)

        assert result == "ok"
        assert attempts[0] == 2
        assert sleeps == [0.1]

    @pytest.mark.asyncio
    async def test_retries_openai_api_connection_error(self, retry, monkeypatch):
        """OpenAI API transport errors should be retried."""
        import httpx
        from openai import APIConnectionError

        attempts = [0]
        sleeps: list[float] = []

        async def flaky():
            attempts[0] += 1
            if attempts[0] == 1:
                error = APIConnectionError(
                    request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
                )
                error.__cause__ = OSError("connection reset by peer")
                raise error
            return "ok"

        async def fake_sleep(delay: float) -> None:
            sleeps.append(delay)

        monkeypatch.setattr(asyncio, "sleep", fake_sleep)

        result = await retry.execute(flaky)

        assert result == "ok"
        assert attempts[0] == 2
        assert len(sleeps) == 1


# =============================================================================
# Rate Limiter Tests
# =============================================================================


class TestTokenBucketRateLimiter:
    """Tests for TokenBucketRateLimiter."""

    @pytest.mark.asyncio
    async def test_acquire_available_tokens(self):
        """Test acquiring available tokens."""
        limiter = TokenBucketRateLimiter(
            tokens_per_second=100,
            burst_capacity=10,
        )

        wait_time = await limiter.acquire(5)
        assert wait_time == 0

        assert limiter.available_tokens < 6

    @pytest.mark.asyncio
    async def test_acquire_waits_when_empty(self):
        """Test waiting for token refill."""
        limiter = TokenBucketRateLimiter(
            tokens_per_second=100,
            burst_capacity=5,
        )

        # Exhaust tokens
        await limiter.acquire(5)

        # Should wait for refill
        start = time.monotonic()
        await limiter.acquire(1)
        elapsed = time.monotonic() - start

        assert elapsed >= 0.01  # Should have waited

    def test_try_acquire(self):
        """Test non-blocking acquire."""
        limiter = TokenBucketRateLimiter(
            tokens_per_second=10,
            burst_capacity=5,
        )

        assert limiter.try_acquire(3) is True
        assert limiter.try_acquire(3) is False  # Not enough tokens


class TestSlidingWindowRateLimiter:
    """Tests for SlidingWindowRateLimiter."""

    @pytest.mark.asyncio
    async def test_acquire_within_limit(self):
        """Test acquiring within rate limit."""
        limiter = SlidingWindowRateLimiter(
            max_requests=10,
            window_seconds=1.0,
        )

        for _ in range(5):
            wait_time = await limiter.acquire()
            assert wait_time == 0

        assert limiter.available_capacity == 5

    @pytest.mark.asyncio
    async def test_acquire_at_limit(self):
        """Test waiting at rate limit."""
        limiter = SlidingWindowRateLimiter(
            max_requests=3,
            window_seconds=0.1,
        )

        # Exhaust limit
        for _ in range(3):
            await limiter.acquire()

        # Should wait
        start = time.monotonic()
        await limiter.acquire()
        elapsed = time.monotonic() - start

        assert elapsed >= 0.05  # Should have waited


# =============================================================================
# Concurrent Request Manager Tests
# =============================================================================


class TestRequestQueue:
    """Tests for RequestQueue."""

    @pytest.fixture
    def manager(self):
        """Create manager with test configuration."""
        config = ConcurrencyConfig(
            max_concurrent_requests=3,
            requests_per_minute=100,
            tokens_per_minute=100000,
            max_queue_size=10,
            queue_timeout_seconds=5.0,
        )
        return RequestQueue(config)

    @pytest.mark.asyncio
    async def test_submit_single_request(self, manager):
        """Test submitting single request."""

        async def task():
            return "result"

        result = await manager.submit(task())
        assert result == "result"

        stats = manager.get_stats()
        assert stats["total_submitted"] == 1
        assert stats["total_completed"] == 1

    @pytest.mark.asyncio
    async def test_submit_parallel_requests(self, manager):
        """Test submitting parallel requests."""

        async def task(n):
            await asyncio.sleep(0.01)
            return n * 2

        results = await manager.submit_parallel(
            [
                task(1),
                task(2),
                task(3),
            ]
        )

        assert results == [2, 4, 6]

    @pytest.mark.asyncio
    async def test_priority_ordering(self):
        """Test that priority affects execution order.

        Uses a single worker to ensure sequential processing in priority order.
        """
        # Use single worker to guarantee sequential processing by priority
        config = ConcurrencyConfig(
            max_concurrent_requests=1,
            requests_per_minute=100,
            tokens_per_minute=100000,
            max_queue_size=10,
            queue_timeout_seconds=5.0,
        )
        single_worker_manager = RequestQueue(config, num_workers=1)

        order = []

        async def record(name):
            order.append(name)
            return name

        try:
            # Submit all requests quickly to queue them before processing
            # The priority queue will order them: CRITICAL (0) < NORMAL (2) < LOW (3)
            tasks = [
                single_worker_manager.submit(record("low"), priority=RequestPriority.LOW),
                single_worker_manager.submit(record("critical"), priority=RequestPriority.CRITICAL),
                single_worker_manager.submit(record("normal"), priority=RequestPriority.NORMAL),
            ]

            await asyncio.gather(*tasks)

            # Critical should be processed first (priority 0), then normal (2), then low (3)
            assert order[0] == "critical", f"Expected 'critical' first, got order: {order}"
            assert order[1] == "normal", f"Expected 'normal' second, got order: {order}"
            assert order[2] == "low", f"Expected 'low' third, got order: {order}"
        finally:
            await single_worker_manager.shutdown()

    @pytest.mark.asyncio
    async def test_tool_calls_parallel(self, manager):
        """Test parallel tool call execution."""

        async def executor(tool_call):
            await asyncio.sleep(0.01)
            return f"result_{tool_call['name']}"

        tool_calls = [
            {"name": "tool1"},
            {"name": "tool2"},
            {"name": "tool3"},
        ]

        results = await manager.execute_tool_calls_parallel(
            tool_calls,
            executor=executor,
        )

        assert len(results) == 3
        assert "result_tool1" in results


# =============================================================================
# Resilient Provider Tests
# =============================================================================


class TestResilientProvider:
    """Tests for ResilientProvider."""

    @pytest.fixture
    def mock_provider(self):
        """Create mock provider."""
        provider = MagicMock()
        provider.name = "test_provider"
        provider.chat = AsyncMock(return_value="response")
        provider.supports_tools = MagicMock(return_value=True)
        return provider

    @pytest.fixture
    def mock_fallback(self):
        """Create mock fallback provider."""
        provider = MagicMock()
        provider.name = "fallback_provider"
        provider.chat = AsyncMock(return_value="fallback_response")
        return provider

    @pytest.mark.asyncio
    async def test_successful_request(self, mock_provider):
        """Test successful request through resilient provider."""
        resilient = ResilientProvider(mock_provider)

        result = await resilient.chat(
            messages=[{"role": "user", "content": "test"}],
            model="test-model",
        )

        assert result == "response"
        mock_provider.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_retries_share_one_logical_call_id(self):
        """One ResilientProvider.chat = one logical call (sandhi TD-0021 P4): every
        retry re-enters the inner provider under the SAME call_id binding, so the
        sandhi transport stamps one Idempotency-Key and the gateway's meter counts
        the call once. TimeoutError is the retryable may-have-billed case."""
        from victor.core.context import get_call_id

        seen: list = []

        async def flaky_chat(messages, *, model, **kwargs):
            seen.append(get_call_id())
            if len(seen) == 1:
                raise asyncio.TimeoutError("simulated may-have-billed timeout")
            return "recovered"

        inner = MagicMock()
        inner.name = "flaky"
        inner.chat = flaky_chat
        resilient = ResilientProvider(
            inner,
            retry_config=ProviderRetryConfig(max_retries=1, base_delay_seconds=0.01),
        )

        result = await resilient.chat(messages=[], model="m")

        assert result == "recovered"
        assert len(seen) == 2
        assert seen[0] and seen[0] == seen[1], "retries of one logical call must share one id"

    @pytest.mark.asyncio
    async def test_fallback_shares_the_logical_call_id(self, mock_fallback):
        """A fallback is still the SAME logical call — the user asked once — so it
        re-enters under the primary's call_id, not a fresh one."""
        from victor.core.context import get_call_id

        primary_seen: list = []
        fallback_seen: list = []

        async def failing_chat(messages, *, model, **kwargs):
            primary_seen.append(get_call_id())
            raise ConnectionError("primary down")

        async def fallback_chat(messages, *, model, **kwargs):
            fallback_seen.append(get_call_id())
            return "fallback_response"

        mock_fallback.chat = fallback_chat
        primary = MagicMock()
        primary.name = "primary"
        primary.chat = failing_chat

        resilient = ResilientProvider(
            primary,
            fallback_providers=[mock_fallback],
            circuit_config=CircuitBreakerConfig(failure_threshold=1),
            retry_config=ProviderRetryConfig(max_retries=0),
        )

        result = await resilient.chat(messages=[], model="m")

        assert result == "fallback_response"
        assert primary_seen[0] == fallback_seen[0], "fallback is the same logical call"

    @pytest.mark.asyncio
    async def test_call_id_is_fresh_per_logical_call(self):
        """Two logical calls (two chat invocations) bind two distinct ids — a shared
        id would wrongly dedup two genuinely separate metered calls."""
        from victor.core.context import get_call_id

        seen: list = []

        async def recording_chat(messages, *, model, **kwargs):
            seen.append(get_call_id())
            return "ok"

        inner = MagicMock()
        inner.name = "recorder"
        inner.chat = recording_chat
        resilient = ResilientProvider(inner)

        await resilient.chat(messages=[], model="m")
        await resilient.chat(messages=[], model="m")

        assert len(seen) == 2
        assert seen[0] and seen[0] != seen[1]

    @pytest.mark.asyncio
    async def test_outermost_call_id_binding_wins(self):
        """A caller that already bound a call id (a stacked resilience layer) keeps
        the OUTER logical-call identity; the inner binding is a no-op and resets
        nothing."""
        import victor.core.context as ctx

        seen: list = []

        async def recording_chat(messages, *, model, **kwargs):
            seen.append(ctx.get_call_id())
            return "ok"

        inner = MagicMock()
        inner.name = "recorder"
        inner.chat = recording_chat
        resilient = ResilientProvider(inner)

        token = ctx.set_call_id("outer-logical-call")
        try:
            await resilient.chat(messages=[], model="m")
        finally:
            ctx.call_id.reset(token)

        assert seen == ["outer-logical-call"]
        assert ctx.get_call_id() == "", "inner layer must not reset the outer binding"

    @pytest.mark.asyncio
    async def test_stream_binds_the_logical_call_id(self):
        """The stream wrapper binds the same logical-call identity as chat."""
        from victor.core.context import get_call_id

        seen: list = []

        def recording_stream(messages, *, model, **kwargs):
            async def _chunks():
                seen.append(get_call_id())
                yield "chunk"

            return _chunks()

        inner = MagicMock()
        inner.name = "streamer"
        inner.stream = recording_stream
        resilient = ResilientProvider(inner)

        chunks = [c async for c in resilient.stream(messages=[], model="m")]

        assert chunks == ["chunk"]
        assert seen and seen[0], "stream must run under a bound call id"
        assert get_call_id() == "", "binding must be reset after the stream ends"

    @pytest.mark.asyncio
    async def test_fallback_on_failure(self, mock_provider, mock_fallback):
        """Test fallback provider is used on failure."""
        mock_provider.chat = AsyncMock(side_effect=ConnectionError("Failed"))

        resilient = ResilientProvider(
            mock_provider,
            fallback_providers=[mock_fallback],
            circuit_config=CircuitBreakerConfig(failure_threshold=1),
            retry_config=ProviderRetryConfig(max_retries=0),
        )

        result = await resilient.chat(
            messages=[{"role": "user", "content": "test"}],
            model="test-model",
        )

        assert result == "fallback_response"

    @pytest.mark.asyncio
    async def test_stats_tracking(self, mock_provider):
        """Test statistics tracking."""
        resilient = ResilientProvider(mock_provider)

        await resilient.chat(messages=[], model="test")
        await resilient.chat(messages=[], model="test")

        stats = resilient.get_stats()

        assert stats["total_requests"] == 2
        assert stats["primary_successes"] == 2

    @pytest.mark.asyncio
    async def test_recovered_primary_retry_attaches_response_diagnostics(self, mock_provider):
        """Recovered provider retries should be visible on response metadata."""
        mock_provider.chat = AsyncMock(
            side_effect=[
                ConnectionError("Server disconnected without sending a response"),
                CompletionResponse(content="recovered", metadata={"existing": "kept"}),
            ]
        )
        resilient = ResilientProvider(
            mock_provider,
            retry_config=ProviderRetryConfig(
                max_retries=1,
                base_delay_seconds=0.0,
                max_delay_seconds=0.0,
                jitter_factor=0.0,
            ),
        )

        result = await resilient.chat(
            messages=[{"role": "user", "content": "test"}],
            model="glm-5.1",
        )

        assert result.content == "recovered"
        assert result.metadata["existing"] == "kept"
        diagnostics = result.metadata["provider_retry_diagnostics"]
        assert diagnostics == [
            {
                "provider": "test_provider",
                "model": "glm-5.1",
                "retry_count": 1,
                "last_error": "Server disconnected without sending a response",
                "events": [
                    {
                        "attempt": 1,
                        "max_retries": 1,
                        "delay_seconds": 0.0,
                        "error": "Server disconnected without sending a response",
                        "error_type": "ConnectionError",
                    }
                ],
            }
        ]
        stats = resilient.get_stats()
        assert stats["retry_attempts"] == 1


# =============================================================================
# Enhanced Provider Factory Tests
# =============================================================================


class TestManagedProviderFactory:
    """Tests for ManagedProviderFactory."""

    @pytest.fixture
    def mock_base_provider(self):
        """Create mock base provider."""
        provider = MagicMock()
        provider.name = "test_provider"
        provider.chat = AsyncMock(return_value="response")
        provider.stream_chat = AsyncMock()
        provider.supports_tools = MagicMock(return_value=True)
        return provider

    def test_provider_config_defaults(self):
        """Test ProviderConfig default values."""
        from victor.providers.factory import ProviderConfig

        config = ProviderConfig(
            provider_name="anthropic",
            model="claude-3-5-haiku-20241022",
        )

        assert config.provider_name == "anthropic"
        assert config.model == "claude-3-5-haiku-20241022"
        assert config.enable_resilience is True
        assert config.enable_rate_limiting is True
        assert config.enable_metrics is True
        assert config.timeout == 120.0

    def test_enhanced_provider_properties(self, mock_base_provider):
        """Test ManagedProvider basic properties."""
        from victor.providers.factory import ManagedProvider, ProviderConfig

        config = ProviderConfig(
            provider_name="test",
            model="test-model",
        )
        enhanced = ManagedProvider(
            base_provider=mock_base_provider,
            config=config,
        )

        assert enhanced.name == "test_provider"
        assert enhanced.model == "test-model"
        assert enhanced.supports_tools() is True

    @pytest.mark.asyncio
    async def test_enhanced_provider_chat(self, mock_base_provider):
        """Test ManagedProvider chat method."""
        from victor.providers.factory import ManagedProvider, ProviderConfig

        config = ProviderConfig(
            provider_name="test",
            model="test-model",
        )
        enhanced = ManagedProvider(
            base_provider=mock_base_provider,
            config=config,
        )

        result = await enhanced.chat(
            messages=[{"role": "user", "content": "test"}],
        )

        assert result == "response"
        mock_base_provider.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_enhanced_provider_with_rate_limiting(self, mock_base_provider):
        """Test ManagedProvider with rate limiting enabled."""
        from victor.providers.factory import ManagedProvider, ProviderConfig
        from victor.providers.concurrency import RequestQueue, ConcurrencyConfig

        config = ProviderConfig(
            provider_name="test",
            model="test-model",
        )
        manager = RequestQueue(
            config=ConcurrencyConfig(max_concurrent_requests=2),
            num_workers=1,
        )

        enhanced = ManagedProvider(
            base_provider=mock_base_provider,
            request_manager=manager,
            config=config,
        )

        try:
            result = await enhanced.chat(
                messages=[{"role": "user", "content": "test"}],
            )
            assert result == "response"

            # Check rate limit stats
            stats = enhanced.get_rate_limit_stats()
            assert stats is not None
            assert "total_submitted" in stats
        finally:
            await enhanced.shutdown()

    @pytest.mark.asyncio
    async def test_enhanced_provider_with_metrics(self, mock_base_provider):
        """Test ManagedProvider with metrics collection."""
        from victor.providers.factory import ManagedProvider, ProviderConfig
        from victor.analytics.streaming_metrics import StreamingMetricsCollector

        config = ProviderConfig(
            provider_name="test",
            model="test-model",
        )
        collector = StreamingMetricsCollector()

        enhanced = ManagedProvider(
            base_provider=mock_base_provider,
            metrics_collector=collector,
            config=config,
        )

        # Make a request
        await enhanced.chat(messages=[{"role": "user", "content": "test"}])

        # Check metrics
        metrics = enhanced.get_metrics()
        assert metrics is not None
        assert "summary" in metrics

    @pytest.mark.asyncio
    async def test_enhanced_provider_with_resilience(self, mock_base_provider):
        """Test ManagedProvider with resilience enabled."""
        from victor.providers.factory import ManagedProvider, ProviderConfig
        from victor.providers.resilience import ResilientProvider

        config = ProviderConfig(
            provider_name="test",
            model="test-model",
        )
        resilient = ResilientProvider(mock_base_provider)

        enhanced = ManagedProvider(
            base_provider=mock_base_provider,
            resilient_provider=resilient,
            config=config,
        )

        result = await enhanced.chat(
            messages=[{"role": "user", "content": "test"}],
        )
        assert result == "response"

        # Check resilience stats
        stats = enhanced.get_resilience_stats()
        assert stats is not None
        assert "total_requests" in stats

    def test_metrics_returns_none_when_disabled(self, mock_base_provider):
        """Test that get_metrics returns None when metrics disabled."""
        from victor.providers.factory import ManagedProvider, ProviderConfig

        config = ProviderConfig(
            provider_name="test",
            model="test-model",
            enable_metrics=False,
        )
        enhanced = ManagedProvider(
            base_provider=mock_base_provider,
            config=config,
        )

        assert enhanced.get_metrics() is None
        assert enhanced.get_resilience_stats() is None
        assert enhanced.get_rate_limit_stats() is None


# =============================================================================
# End-to-End Integration Tests
# =============================================================================


class TestSharedInfrastructureIntegration:
    """End-to-end tests for shared infrastructure integration."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider that simulates real behavior."""
        provider = MagicMock()
        provider.name = "test_provider"
        provider.supports_tools = MagicMock(return_value=True)

        # Simulate chat with delay
        async def mock_chat(*args, **kwargs):
            await asyncio.sleep(0.05)  # Simulate network latency
            return MagicMock(
                content="Test response",
                tool_calls=[],
                usage={"prompt_tokens": 100, "completion_tokens": 50},
            )

        provider.chat = AsyncMock(side_effect=mock_chat)
        return provider

    @pytest.mark.asyncio
    async def test_full_stack_integration(self, mock_provider):
        """Test all components working together."""
        from victor.providers.factory import ManagedProvider, ProviderConfig
        from victor.providers.resilience import ResilientProvider
        from victor.providers.concurrency import RequestQueue, ConcurrencyConfig
        from victor.analytics.streaming_metrics import StreamingMetricsCollector
        from victor.agent.conversation.store import ConversationStore
        from victor.agent.conversation.types import MessageRole
        import tempfile
        from pathlib import Path

        # Create all components
        config = ProviderConfig(
            provider_name="test",
            model="test-model",
        )

        # Memory manager
        db_path = Path(tempfile.mktemp(suffix=".db"))
        memory_manager = ConversationStore(db_path=db_path)
        session = memory_manager.create_session(
            provider="test",
            model="test-model",
        )

        # Rate limiting manager
        concurrency_config = ConcurrencyConfig(
            max_concurrent_requests=2,
            requests_per_minute=100,
            tokens_per_minute=100000,
        )
        request_manager = RequestQueue(
            config=concurrency_config,
            num_workers=1,
        )

        # Metrics collector
        metrics_collector = StreamingMetricsCollector(max_history=100)

        # Resilient provider
        resilient = ResilientProvider(mock_provider)

        # Enhanced provider combining all
        enhanced = ManagedProvider(
            base_provider=mock_provider,
            resilient_provider=resilient,
            request_manager=request_manager,
            metrics_collector=metrics_collector,
            config=config,
        )

        try:
            # Test 1: Make a chat request through the full stack
            messages = [{"role": "user", "content": "Hello"}]
            result = await enhanced.chat(messages=messages)
            assert result is not None

            # Test 2: Record message in memory
            memory_manager.add_message(
                session.session_id,
                MessageRole.USER,
                "Hello",
            )
            memory_manager.add_message(
                session.session_id,
                MessageRole.ASSISTANT,
                "Test response",
            )

            # Test 3: Verify memory persistence
            context = memory_manager.get_context_messages(session.session_id)
            assert len(context) == 2

            # Test 4: Check resilience stats
            resilience_stats = enhanced.get_resilience_stats()
            assert resilience_stats is not None
            assert resilience_stats["total_requests"] >= 1

            # Test 5: Check rate limit stats
            rate_stats = enhanced.get_rate_limit_stats()
            assert rate_stats is not None
            assert "total_submitted" in rate_stats

            # Test 6: Multiple concurrent requests
            tasks = [
                enhanced.chat(messages=[{"role": "user", "content": f"Request {i}"}])
                for i in range(3)
            ]
            results = await asyncio.gather(*tasks)
            assert len(results) == 3

            # Test 7: Verify metrics after multiple requests
            final_stats = enhanced.get_rate_limit_stats()
            assert final_stats["total_submitted"] >= 4  # 1 initial + 3 concurrent

        finally:
            await enhanced.shutdown()
            db_path.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_priority_queue_under_load(self):
        """Test priority queue behavior under concurrent load."""
        from victor.providers.concurrency import (
            RequestQueue,
            ConcurrencyConfig,
            RequestPriority,
        )

        config = ConcurrencyConfig(
            max_concurrent_requests=1,  # Single worker for deterministic ordering
            requests_per_minute=1000,
            tokens_per_minute=1000000,
        )
        manager = RequestQueue(config, num_workers=1)

        execution_order = []

        async def track_execution(name: str, delay: float = 0.01):
            execution_order.append(name)
            await asyncio.sleep(delay)
            return name

        try:
            # Submit requests with different priorities
            # Using asyncio.gather to submit all at once
            tasks = [
                manager.submit(
                    track_execution("batch_1"),
                    priority=RequestPriority.BATCH,
                ),
                manager.submit(
                    track_execution("critical_1"),
                    priority=RequestPriority.CRITICAL,
                ),
                manager.submit(
                    track_execution("normal_1"),
                    priority=RequestPriority.NORMAL,
                ),
                manager.submit(
                    track_execution("high_1"),
                    priority=RequestPriority.HIGH,
                ),
                manager.submit(
                    track_execution("low_1"),
                    priority=RequestPriority.LOW,
                ),
            ]

            await asyncio.gather(*tasks)

            # Verify critical was processed first
            assert (
                execution_order[0] == "critical_1"
            ), f"Expected critical first, got: {execution_order}"
            # Verify batch was processed last
            assert execution_order[-1] == "batch_1", f"Expected batch last, got: {execution_order}"

        finally:
            await manager.shutdown()

    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self):
        """Test circuit breaker with retry and fallback."""
        from victor.providers.resilience import (
            ResilientProvider,
            CircuitBreakerConfig,
            ProviderRetryConfig,
        )

        # Primary provider that always fails
        failing_provider = MagicMock()
        failing_provider.name = "failing_provider"
        failing_provider.chat = AsyncMock(side_effect=ConnectionError("Primary failed"))

        # Fallback provider that succeeds
        fallback_provider = MagicMock()
        fallback_provider.name = "fallback_provider"
        fallback_provider.chat = AsyncMock(return_value="Fallback response")

        resilient = ResilientProvider(
            provider=failing_provider,
            circuit_config=CircuitBreakerConfig(
                failure_threshold=1,  # Open after 1 failure
                timeout_seconds=1.0,
            ),
            retry_config=ProviderRetryConfig(
                max_retries=0,  # No retries for faster test
            ),
            fallback_providers=[fallback_provider],
        )

        # First request - primary fails, fallback succeeds
        result = await resilient.chat(messages=[], model="test")
        assert result == "Fallback response"

        stats = resilient.get_stats()
        # Primary failed, fallback succeeded
        # total_failures only counts when ALL providers fail
        assert stats["total_requests"] >= 1
        assert stats["fallback_successes"] >= 1  # Fallback was used and succeeded
        assert stats["primary_successes"] == 0  # Primary never succeeded

    def test_settings_configuration(self):
        """Test that all new settings are properly configured."""
        from victor.config.settings import Settings, get_project_paths

        settings = Settings()

        # Conversation memory settings
        assert get_project_paths().project_db.name == "project.db"
        assert settings.conversation is not None
        assert settings.context is not None
        assert settings.response is not None
        assert settings.conversation.conversation_memory_enabled is not None
        assert settings.context.max_context_tokens is not None
        assert settings.response.response_token_reserve is not None

        # Resilience settings
        assert settings.resilience is not None
        assert settings.resilience.circuit_breaker_failure_threshold is not None
        assert settings.resilience.retry_max_attempts is not None

        # Rate limiting settings
        assert settings.resilience.rate_limiting_enabled is not None
        assert settings.resilience.rate_limit_requests_per_minute is not None
        assert settings.resilience.rate_limit_max_concurrent is not None

        # Streaming metrics settings
        assert settings.analytics is not None
        assert settings.analytics.streaming_metrics_enabled is not None
        assert settings.analytics.streaming_metrics_history_size is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

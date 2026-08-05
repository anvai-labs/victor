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

"""Tests for middleware pipeline module."""

import asyncio
import pytest

from victor.core.middleware import (
    CachingMiddleware,
    ContextKey,
    ErrorHandlingMiddleware,
    LoggingMiddleware,
    MetricsMiddleware,
    Middleware,
    MiddlewareContext,
    MiddlewarePipeline,
    PipelineBuilder,
    RetryMiddleware,
    TimeoutMiddleware,
    TimingMiddleware,
    ValidationMiddleware,
    create_default_pipeline,
    create_resilient_pipeline,
)

# =============================================================================
# MiddlewareContext Tests
# =============================================================================


class TestMiddlewareContext:
    """Tests for MiddlewareContext."""

    def test_basic_creation(self):
        """Test creating context."""
        context = MiddlewareContext(request={"key": "value"})

        assert context.request == {"key": "value"}
        assert context.cancelled is False
        assert context.error is None

    def test_set_and_get(self):
        """Test setting and getting values."""
        context = MiddlewareContext(request={})

        context.set("custom_key", "custom_value")

        assert context.get("custom_key") == "custom_value"
        assert context.has("custom_key") is True
        assert context.has("missing_key") is False

    def test_get_default(self):
        """Test get with default value."""
        context = MiddlewareContext(request={})

        assert context.get("missing", "default") == "default"

    def test_elapsed_ms(self):
        """Test elapsed time calculation."""
        import time

        context = MiddlewareContext(request={})
        time.sleep(0.01)

        assert context.elapsed_ms >= 10

    def test_cancel(self):
        """Test cancel method."""
        context = MiddlewareContext(request={})

        context.cancel()

        assert context.cancelled is True


# =============================================================================
# Basic Middleware Tests
# =============================================================================


class TestMiddleware:
    """Tests for base Middleware class."""

    @pytest.mark.asyncio
    async def test_custom_middleware(self):
        """Test creating custom middleware."""

        class CustomMiddleware(Middleware):
            def __init__(self):
                self.before_called = False
                self.after_called = False

            async def before(self, context):
                self.before_called = True
                return True

            async def after(self, context, response):
                self.after_called = True
                return response

        middleware = CustomMiddleware()
        context = MiddlewareContext(request={})

        async def next_handler():
            return "result"

        result = await middleware(context, next_handler)

        assert result == "result"
        assert middleware.before_called
        assert middleware.after_called

    @pytest.mark.asyncio
    async def test_middleware_short_circuit(self):
        """Test middleware can short-circuit chain."""

        class ShortCircuitMiddleware(Middleware):
            async def before(self, context):
                return False  # Don't continue

            async def short_circuit(self, context):
                return "short_circuited"

        middleware = ShortCircuitMiddleware()
        context = MiddlewareContext(request={})

        async def next_handler():
            return "should_not_reach"

        result = await middleware(context, next_handler)

        assert result == "short_circuited"


# =============================================================================
# MiddlewarePipeline Tests
# =============================================================================


class TestMiddlewarePipeline:
    """Tests for MiddlewarePipeline."""

    @pytest.mark.asyncio
    async def test_empty_pipeline(self):
        """Test empty pipeline just calls handler."""
        pipeline = MiddlewarePipeline()
        context = MiddlewareContext(request={})

        async def handler(ctx):
            return "result"

        result = await pipeline.execute(handler, context)

        assert result == "result"

    @pytest.mark.asyncio
    async def test_single_middleware(self):
        """Test pipeline with single middleware."""

        class TrackingMiddleware(Middleware):
            def __init__(self):
                super().__init__()
                self.called = False

            async def before(self, context):
                self.called = True
                return True

        middleware = TrackingMiddleware()
        pipeline = MiddlewarePipeline().use(middleware)
        context = MiddlewareContext(request={})

        async def handler(ctx):
            return "result"

        result = await pipeline.execute(handler, context)

        assert result == "result"
        assert middleware.called

    @pytest.mark.asyncio
    async def test_middleware_order(self):
        """Test middleware executes in correct order."""
        order = []

        class OrderMiddleware(Middleware):
            def __init__(self, name):
                super().__init__()
                self._name = name

            @property
            def name(self):
                return self._name

            async def before(self, context):
                order.append(f"before_{self._name}")
                return True

            async def after(self, context, response):
                order.append(f"after_{self._name}")
                return response

        pipeline = (
            MiddlewarePipeline()
            .use(OrderMiddleware("first"))
            .use(OrderMiddleware("second"))
            .use(OrderMiddleware("third"))
        )
        context = MiddlewareContext(request={})

        async def handler(ctx):
            order.append("handler")
            return "result"

        await pipeline.execute(handler, context)

        assert order == [
            "before_first",
            "before_second",
            "before_third",
            "handler",
            "after_third",
            "after_second",
            "after_first",
        ]

    @pytest.mark.asyncio
    async def test_cancelled_context_rejected(self):
        """Test cancelled context is rejected."""
        pipeline = MiddlewarePipeline()
        context = MiddlewareContext(request={})
        context.cancel()

        async def handler(ctx):
            return "result"

        with pytest.raises(RuntimeError, match="cancelled"):
            await pipeline.execute(handler, context)

    def test_use_if_condition_true(self):
        """Test conditional middleware add when true."""

        class TestMiddleware(Middleware):
            pass

        pipeline = MiddlewarePipeline().use_if(True, TestMiddleware())

        assert pipeline.middleware_count == 1

    def test_use_if_condition_false(self):
        """Test conditional middleware add when false."""

        class TestMiddleware(Middleware):
            pass

        pipeline = MiddlewarePipeline().use_if(False, TestMiddleware())

        assert pipeline.middleware_count == 0

    def test_freeze_prevents_modification(self):
        """Test frozen pipeline cannot be modified."""

        class TestMiddleware(Middleware):
            pass

        pipeline = MiddlewarePipeline().freeze()

        with pytest.raises(RuntimeError):
            pipeline.use(TestMiddleware())


# =============================================================================
# LoggingMiddleware Tests
# =============================================================================


class TestLoggingMiddleware:
    """Tests for LoggingMiddleware."""

    @pytest.mark.asyncio
    async def test_logs_request(self, caplog):
        """Test request logging."""
        import logging

        middleware = LoggingMiddleware(log_level=logging.INFO)
        context = MiddlewareContext(request={}, request_id="test-123")

        async def next_handler():
            return "result"

        # Capture logs from the specific middleware logger
        with caplog.at_level(logging.INFO, logger="victor.core.middleware"):
            await middleware(context, next_handler)

        assert "test-123" in caplog.text


# =============================================================================
# TimingMiddleware Tests
# =============================================================================


class TestTimingMiddleware:
    """Tests for TimingMiddleware."""

    @pytest.mark.asyncio
    async def test_records_timing(self):
        """Test timing is recorded."""
        middleware = TimingMiddleware()
        context = MiddlewareContext(request={})

        async def next_handler():
            await asyncio.sleep(0.01)
            return "result"

        await middleware(context, next_handler)

        assert "timing_ms" in context.metadata
        assert context.metadata["timing_ms"] >= 10


# =============================================================================
# ErrorHandlingMiddleware Tests
# =============================================================================


class TestErrorHandlingMiddleware:
    """Tests for ErrorHandlingMiddleware."""

    @pytest.mark.asyncio
    async def test_custom_handler(self):
        """Test custom error handler."""
        middleware = ErrorHandlingMiddleware(
            handlers={
                ValueError: lambda e: {"error": str(e)},
            }
        )
        context = MiddlewareContext(request={})

        async def next_handler():
            raise ValueError("test error")

        result = await middleware(context, next_handler)

        assert result == {"error": "test error"}
        assert context.error is not None

    @pytest.mark.asyncio
    async def test_default_handler(self):
        """Test default error handler."""
        middleware = ErrorHandlingMiddleware(
            default_handler=lambda e: "handled",
        )
        context = MiddlewareContext(request={})

        async def next_handler():
            raise RuntimeError("test")

        result = await middleware(context, next_handler)

        assert result == "handled"

    @pytest.mark.asyncio
    async def test_reraise_unhandled(self):
        """Test unhandled errors are reraised."""
        middleware = ErrorHandlingMiddleware(reraise=True)
        context = MiddlewareContext(request={})

        async def next_handler():
            raise RuntimeError("test")

        with pytest.raises(RuntimeError):
            await middleware(context, next_handler)


# =============================================================================
# RetryMiddleware Tests
# =============================================================================


class TestRetryMiddleware:
    """Tests for RetryMiddleware."""

    @pytest.mark.asyncio
    async def test_retries_on_failure(self):
        """Test retry on failure."""
        call_count = 0

        middleware = RetryMiddleware(
            max_retries=3,
            retry_on=(ValueError,),
            delay=0.01,
        )
        context = MiddlewareContext(request={})

        async def next_handler():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("transient")
            return "success"

        result = await middleware(context, next_handler)

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        """Test error raised after max retries."""
        middleware = RetryMiddleware(
            max_retries=2,
            retry_on=(ValueError,),
            delay=0.01,
        )
        context = MiddlewareContext(request={})

        async def next_handler():
            raise ValueError("always fails")

        with pytest.raises(ValueError):
            await middleware(context, next_handler)


# =============================================================================
# TimeoutMiddleware Tests
# =============================================================================


class TestTimeoutMiddleware:
    """Tests for TimeoutMiddleware."""

    @pytest.mark.asyncio
    async def test_completes_within_timeout(self):
        """Test successful completion within timeout."""
        middleware = TimeoutMiddleware(timeout=1.0)
        context = MiddlewareContext(request={})

        async def next_handler():
            return "fast"

        result = await middleware(context, next_handler)

        assert result == "fast"

    @pytest.mark.asyncio
    async def test_raises_on_timeout(self):
        """Test timeout error raised."""
        middleware = TimeoutMiddleware(timeout=0.1)
        context = MiddlewareContext(request={})

        async def next_handler():
            await asyncio.sleep(1.0)
            return "slow"

        with pytest.raises(TimeoutError):
            await middleware(context, next_handler)


# =============================================================================
# CachingMiddleware Tests
# =============================================================================


class TestCachingMiddleware:
    """Tests for CachingMiddleware."""

    @pytest.mark.asyncio
    async def test_caches_response(self):
        """Test response is cached."""
        cache = {}
        call_count = 0

        middleware = CachingMiddleware(
            cache=cache,
            key_fn=lambda ctx: ctx.request.get("id"),
            ttl=300,
        )

        async def next_handler():
            nonlocal call_count
            call_count += 1
            return "result"

        # First call
        context1 = MiddlewareContext(request={"id": "key1"})
        result1 = await middleware(context1, next_handler)

        # Second call (cached)
        context2 = MiddlewareContext(request={"id": "key1"})
        result2 = await middleware(context2, next_handler)

        assert result1 == "result"
        assert result2 == "result"
        assert call_count == 1  # Only called once
        assert context2.get("cache_hit") is True

    @pytest.mark.asyncio
    async def test_no_cache_on_none_key(self):
        """Test no caching when key is None."""
        cache = {}
        call_count = 0

        middleware = CachingMiddleware(
            cache=cache,
            key_fn=lambda ctx: None,  # No caching
            ttl=300,
        )

        async def next_handler():
            nonlocal call_count
            call_count += 1
            return "result"

        context = MiddlewareContext(request={})
        await middleware(context, next_handler)
        await middleware(context, next_handler)

        assert call_count == 2  # Called twice, no caching


# =============================================================================
# ValidationMiddleware Tests
# =============================================================================


class TestValidationMiddleware:
    """Tests for ValidationMiddleware."""

    @pytest.mark.asyncio
    async def test_valid_request(self):
        """Test valid request passes."""

        def validator(ctx):
            if not ctx.request.get("id"):
                raise ValueError("id required")

        middleware = ValidationMiddleware(validator=validator)
        context = MiddlewareContext(request={"id": "123"})

        async def next_handler():
            return "result"

        result = await middleware(context, next_handler)

        assert result == "result"

    @pytest.mark.asyncio
    async def test_invalid_request(self):
        """Test invalid request raises."""

        def validator(ctx):
            if not ctx.request.get("id"):
                raise ValueError("id required")

        middleware = ValidationMiddleware(validator=validator)
        context = MiddlewareContext(request={})

        async def next_handler():
            return "result"

        with pytest.raises(ValueError, match="id required"):
            await middleware(context, next_handler)


# =============================================================================
# MetricsMiddleware Tests
# =============================================================================


class TestMetricsMiddleware:
    """Tests for MetricsMiddleware."""

    @pytest.mark.asyncio
    async def test_tracks_metrics(self):
        """Test metrics are tracked."""
        middleware = MetricsMiddleware(prefix="test")
        context = MiddlewareContext(request={})

        async def next_handler():
            await asyncio.sleep(0.01)
            return "result"

        await middleware(context, next_handler)

        metrics = middleware.get_metrics()

        assert metrics["test_requests"] == 1
        assert metrics["test_errors"] == 0
        assert metrics["test_total_duration_ms"] > 0

    @pytest.mark.asyncio
    async def test_tracks_errors(self):
        """Test error tracking."""
        middleware = MetricsMiddleware(prefix="test")
        context = MiddlewareContext(request={})

        async def next_handler():
            raise RuntimeError("error")

        with pytest.raises(RuntimeError):
            await middleware(context, next_handler)

        metrics = middleware.get_metrics()

        assert metrics["test_errors"] == 1


# =============================================================================
# PipelineBuilder Tests
# =============================================================================


class TestPipelineBuilder:
    """Tests for PipelineBuilder."""

    def test_build_with_logging(self):
        """Test adding logging middleware."""
        pipeline = PipelineBuilder().with_logging().build()

        assert "LoggingMiddleware" in pipeline.middleware_names

    def test_build_with_timing(self):
        """Test adding timing middleware."""
        pipeline = PipelineBuilder().with_timing().build()

        assert "TimingMiddleware" in pipeline.middleware_names

    def test_build_with_error_handling(self):
        """Test adding error handling middleware."""
        pipeline = PipelineBuilder().with_error_handling().build()

        assert "ErrorHandlingMiddleware" in pipeline.middleware_names

    def test_build_with_retry(self):
        """Test adding retry middleware."""
        pipeline = PipelineBuilder().with_retry().build()

        assert "RetryMiddleware" in pipeline.middleware_names

    def test_build_with_timeout(self):
        """Test adding timeout middleware."""
        pipeline = PipelineBuilder().with_timeout().build()

        assert "TimeoutMiddleware" in pipeline.middleware_names

    def test_build_complete_pipeline(self):
        """Test building complete pipeline."""
        pipeline = (
            PipelineBuilder()
            .with_error_handling()
            .with_logging()
            .with_timing()
            .with_retry()
            .with_timeout()
            .build()
        )

        assert pipeline.middleware_count == 5


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_default_pipeline(self):
        """Test create_default_pipeline."""
        pipeline = create_default_pipeline()

        assert pipeline.middleware_count >= 2
        assert "LoggingMiddleware" in pipeline.middleware_names
        assert "TimingMiddleware" in pipeline.middleware_names

    def test_create_resilient_pipeline(self):
        """Test create_resilient_pipeline."""
        pipeline = create_resilient_pipeline(
            max_retries=5,
            timeout=60.0,
        )

        assert "RetryMiddleware" in pipeline.middleware_names
        assert "TimeoutMiddleware" in pipeline.middleware_names

    @pytest.mark.asyncio
    async def test_default_pipeline_executes(self):
        """Test default pipeline can execute."""
        pipeline = create_default_pipeline()
        context = MiddlewareContext(request={})

        async def handler(ctx):
            return "result"

        result = await pipeline.execute(handler, context)

        assert result == "result"

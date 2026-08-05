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

"""Generic request/response pipeline middleware (internal use only).

This module implements cross-cutting concerns (logging, metrics, error
handling, retry, timeout) for **HTTP/provider/workflow pipelines** using an
inheritance-based ``Middleware[TRequest, TResponse]`` ABC.

**Do NOT use this for tool execution middleware.** For tool-call interception,
implement ``MiddlewareProtocol`` from
``victor.core.verticals.protocols.middleware`` and register via a vertical's
``get_middleware()`` hook. Those middlewares are processed by
``victor.agent.middleware_chain.MiddlewareChain``, not by this pipeline.

The two systems are intentionally separate:
- This module: internal HTTP/provider pipelines (inheritance-based)
- ``victor.framework.middleware``: tool-call pipeline (protocol-based, SDK-blessed)

Design Patterns:
- Chain of Responsibility: Middleware chain for request processing
- Decorator Pattern: Middleware wrapping handlers
- Template Method: Base middleware with hooks
- Composite Pattern: Nested middleware composition

Example:
    from victor.core.middleware import (
        MiddlewarePipeline,
        LoggingMiddleware,
        MetricsMiddleware,
        TimingMiddleware,
        ErrorHandlingMiddleware,
    )

    # Build pipeline
    pipeline = (
        MiddlewarePipeline()
        .use(ErrorHandlingMiddleware())
        .use(LoggingMiddleware())
        .use(MetricsMiddleware())
        .use(TimingMiddleware())
    )

    # Execute with pipeline
    async def handler(context):
        return await process_request(context.request)

    result = await pipeline.execute(handler, context)
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Type,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Type Definitions
# =============================================================================

T = TypeVar("T")
TRequest = TypeVar("TRequest")
TResponse = TypeVar("TResponse")

# Handler types
Handler = Callable[[Any], Awaitable[Any]]
NextHandler = Callable[[], Awaitable[Any]]


# =============================================================================
# Context Pattern
# =============================================================================


class ContextKey(str, Enum):
    """Standard context keys."""

    REQUEST_ID = "request_id"
    START_TIME = "start_time"
    USER_ID = "user_id"
    TRACE_ID = "trace_id"
    SPAN_ID = "span_id"
    METADATA = "metadata"


@dataclass
class MiddlewareContext(Generic[TRequest]):
    """Context passed through middleware chain.

    Provides a shared state container that middleware can read from
    and write to during request processing.
    """

    request: TRequest
    request_id: str = ""
    start_time: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    cancelled: bool = False
    error: Optional[Exception] = None

    # Internal state
    _values: Dict[str, Any] = field(default_factory=dict)

    def set(self, key: str, value: Any) -> None:
        """Set a context value."""
        self._values[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get a context value."""
        return self._values.get(key, default)

    def has(self, key: str) -> bool:
        """Check if key exists."""
        return key in self._values

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return (time.time() - self.start_time) * 1000

    def cancel(self) -> None:
        """Mark context as cancelled."""
        self.cancelled = True


# =============================================================================
# Base Middleware (Template Method Pattern)
# =============================================================================


class Middleware(ABC, Generic[TRequest, TResponse]):
    """Abstract base class for middleware.

    Implements the Template Method pattern with hooks for:
    - before: Called before the next handler
    - after: Called after the next handler succeeds
    - on_error: Called when an error occurs

    Subclasses can override any of these hooks.
    """

    def __init__(self, priority: int = 100) -> None:
        """Initialize middleware.

        Args:
            priority: Execution priority (lower runs first). Default 100.
        """
        self.priority = priority

    @property
    def name(self) -> str:
        """Get middleware name."""
        return self.__class__.__name__

    async def __call__(
        self,
        context: MiddlewareContext[TRequest],
        next_handler: NextHandler,
    ) -> TResponse:
        """Execute middleware.

        Args:
            context: Request context.
            next_handler: Next handler in chain.

        Returns:
            Response from handler chain.
        """
        try:
            # Before hook
            should_continue = await self.before(context)
            if not should_continue:
                return await self.short_circuit(context)

            # Execute next handler
            response = await next_handler()

            # After hook
            response = await self.after(context, response)

            return response

        except Exception as e:
            # Error hook
            return await self.on_error(context, e)

    async def before(self, context: MiddlewareContext[TRequest]) -> bool:
        """Hook called before next handler.

        Args:
            context: Request context.

        Returns:
            True to continue, False to short-circuit.
        """
        return True

    async def after(
        self,
        context: MiddlewareContext[TRequest],
        response: TResponse,
    ) -> TResponse:
        """Hook called after successful response.

        Args:
            context: Request context.
            response: Response from next handler.

        Returns:
            Potentially modified response.
        """
        return response

    async def on_error(
        self,
        context: MiddlewareContext[TRequest],
        error: Exception,
    ) -> TResponse:
        """Hook called on error.

        Args:
            context: Request context.
            error: Exception that occurred.

        Returns:
            Error response or raises exception.
        """
        raise error

    async def short_circuit(
        self,
        context: MiddlewareContext[TRequest],
    ) -> TResponse:
        """Called when before() returns False.

        Args:
            context: Request context.

        Returns:
            Short-circuit response.
        """
        raise RuntimeError(f"Request short-circuited by {self.name}")


# =============================================================================
# Focused Middleware Protocols (ISP-Compliant)
# =============================================================================
# These protocols allow middleware to declare only the hooks they need,
# following the Interface Segregation Principle. The MiddlewarePipeline
# detects which protocols a middleware implements and calls only those hooks.


class RequestMiddleware(ABC):
    """Middleware that only needs to inspect/modify requests before handling."""

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    async def before(self, context: MiddlewareContext) -> bool:
        """Pre-handler hook. Return False to short-circuit."""
        ...


class ResponseMiddleware(ABC):
    """Middleware that only needs to inspect/modify responses after handling."""

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    async def after(self, context: MiddlewareContext, response: Any) -> Any:
        """Post-handler hook. Can transform the response."""
        ...


class ErrorMiddleware(ABC):
    """Middleware that only handles errors."""

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    async def on_error(self, context: MiddlewareContext, error: Exception) -> Any:
        """Error handler. Can recover or re-raise."""
        ...


class _FocusedMiddlewareAdapter(Middleware):
    """Internal adapter that wraps focused protocol middleware into the full interface."""

    def __init__(self, focused: Union[RequestMiddleware, ResponseMiddleware, ErrorMiddleware]):
        super().__init__()
        self._focused = focused

    @property
    def name(self) -> str:
        return self._focused.name

    async def before(self, context: MiddlewareContext) -> bool:
        if isinstance(self._focused, RequestMiddleware):
            return await self._focused.before(context)
        return True

    async def after(self, context: MiddlewareContext, response: Any) -> Any:
        if isinstance(self._focused, ResponseMiddleware):
            return await self._focused.after(context, response)
        return response

    async def on_error(self, context: MiddlewareContext, error: Exception) -> Any:
        if isinstance(self._focused, ErrorMiddleware):
            return await self._focused.on_error(context, error)
        raise error


# =============================================================================
# Middleware Pipeline (Chain of Responsibility)
# =============================================================================


class MiddlewarePipeline(Generic[TRequest, TResponse]):
    """Pipeline that chains middleware together.

    Implements the Chain of Responsibility pattern where each middleware
    can process the request, modify it, or short-circuit the chain.

    Example:
        pipeline = (
            MiddlewarePipeline()
            .use(AuthMiddleware())
            .use(LoggingMiddleware())
            .use(ValidationMiddleware())
        )

        result = await pipeline.execute(handler, context)
    """

    def __init__(self) -> None:
        """Initialize pipeline."""
        self._middleware: List[Middleware] = []
        self._frozen: bool = False

    def use(
        self,
        middleware: Union[
            Middleware[TRequest, TResponse],
            RequestMiddleware,
            ResponseMiddleware,
            ErrorMiddleware,
        ],
    ) -> "MiddlewarePipeline[TRequest, TResponse]":
        """Add middleware to pipeline.

        Accepts both full Middleware instances and focused protocol middleware
        (RequestMiddleware, ResponseMiddleware, ErrorMiddleware). Focused
        middleware is automatically wrapped with an adapter.

        Args:
            middleware: Middleware to add.

        Returns:
            Self for chaining.

        Raises:
            RuntimeError: If pipeline is frozen.
        """
        if self._frozen:
            raise RuntimeError("Cannot add middleware to frozen pipeline")

        if isinstance(middleware, (RequestMiddleware, ResponseMiddleware, ErrorMiddleware)):
            middleware = _FocusedMiddlewareAdapter(middleware)

        self._middleware.append(middleware)
        return self

    def use_if(
        self,
        condition: bool,
        middleware: Middleware[TRequest, TResponse],
    ) -> "MiddlewarePipeline[TRequest, TResponse]":
        """Conditionally add middleware.

        Args:
            condition: Whether to add middleware.
            middleware: Middleware to add.

        Returns:
            Self for chaining.
        """
        if condition:
            self.use(middleware)
        return self

    def freeze(self) -> "MiddlewarePipeline[TRequest, TResponse]":
        """Freeze pipeline to prevent modification.

        Returns:
            Self for chaining.
        """
        self._frozen = True
        return self

    async def execute(
        self,
        handler: Callable[[MiddlewareContext[TRequest]], Awaitable[TResponse]],
        context: MiddlewareContext[TRequest],
    ) -> TResponse:
        """Execute handler with middleware chain.

        Args:
            handler: Final handler to execute.
            context: Request context.

        Returns:
            Response from handler chain.
        """
        if context.cancelled:
            raise RuntimeError("Context is cancelled")

        # Build chain from end to start
        chain = self._build_chain(handler, context)

        # Execute chain
        return await chain()

    def _build_chain(
        self,
        handler: Callable[[MiddlewareContext[TRequest]], Awaitable[TResponse]],
        context: MiddlewareContext[TRequest],
    ) -> NextHandler:
        """Build middleware chain.

        Args:
            handler: Final handler.
            context: Request context.

        Returns:
            Chained handler.
        """

        async def final() -> TResponse:
            return await handler(context)

        chain = final

        # Sort by priority (stable sort preserves insertion order for equal priorities)
        sorted_middleware = sorted(self._middleware, key=lambda m: m.priority)

        # Wrap from end to start
        for middleware in reversed(sorted_middleware):
            current_chain = chain

            async def make_next(mw: Middleware, next_fn: NextHandler) -> TResponse:
                return await mw(context, next_fn)

            # Capture current middleware and chain in closure
            chain = self._make_chain_link(middleware, current_chain, context)

        return chain

    def _make_chain_link(
        self,
        middleware: Middleware,
        next_chain: NextHandler,
        context: MiddlewareContext[TRequest],
    ) -> NextHandler:
        """Create a chain link for middleware."""

        async def link() -> TResponse:
            return await middleware(context, next_chain)

        return link

    @property
    def middleware_count(self) -> int:
        """Get number of middleware in pipeline."""
        return len(self._middleware)

    @property
    def middleware_names(self) -> List[str]:
        """Get names of middleware in order."""
        return [m.name for m in self._middleware]


# =============================================================================
# Common Middleware Implementations
# =============================================================================


class LoggingMiddleware(Middleware):
    """Middleware that logs request/response.

    Example:
        pipeline.use(LoggingMiddleware(log_level=logging.DEBUG))
    """

    def __init__(
        self,
        log_level: int = logging.INFO,
        log_request: bool = True,
        log_response: bool = True,
        priority: int = 100,
    ) -> None:
        """Initialize logging middleware.

        Args:
            log_level: Logging level.
            log_request: Whether to log requests.
            log_response: Whether to log responses.
            priority: Execution priority (lower runs first).
        """
        super().__init__(priority=priority)
        self._level = log_level
        self._log_request = log_request
        self._log_response = log_response

    async def before(self, context: MiddlewareContext) -> bool:
        """Log request."""
        if self._log_request:
            logger.log(
                self._level,
                f"Request started: id={context.request_id}",
            )
        return True

    async def after(self, context: MiddlewareContext, response: Any) -> Any:
        """Log response."""
        if self._log_response:
            logger.log(
                self._level,
                f"Request completed: id={context.request_id} "
                f"elapsed={context.elapsed_ms:.2f}ms",
            )
        return response

    async def on_error(self, context: MiddlewareContext, error: Exception) -> Any:
        """Log error."""
        logger.error(
            f"Request failed: id={context.request_id} "
            f"error={error.__class__.__name__}: {error}",
        )
        raise error


class TimingMiddleware(Middleware):
    """Middleware that tracks execution timing.

    Records timing information in context metadata.
    """

    def __init__(self, metric_key: str = "timing", priority: int = 100) -> None:
        """Initialize timing middleware.

        Args:
            metric_key: Key to store timing in metadata.
            priority: Execution priority (lower runs first).
        """
        super().__init__(priority=priority)
        self._key = metric_key

    async def before(self, context: MiddlewareContext) -> bool:
        """Record start time."""
        context.set(f"{self._key}_start", time.perf_counter())
        return True

    async def after(self, context: MiddlewareContext, response: Any) -> Any:
        """Record end time and duration."""
        start = context.get(f"{self._key}_start", time.perf_counter())
        duration_ms = (time.perf_counter() - start) * 1000

        context.metadata[f"{self._key}_ms"] = duration_ms
        context.set(f"{self._key}_duration", duration_ms)

        return response


class ErrorHandlingMiddleware(Middleware):
    """Middleware that handles and transforms errors.

    Example:
        pipeline.use(ErrorHandlingMiddleware(
            handlers={
                ValueError: lambda e: {"error": str(e), "code": 400},
            }
        ))
    """

    def __init__(
        self,
        handlers: Optional[Dict[Type[Exception], Callable[[Exception], Any]]] = None,
        default_handler: Optional[Callable[[Exception], Any]] = None,
        reraise: bool = True,
        priority: int = 100,
    ) -> None:
        """Initialize error handling middleware.

        Args:
            handlers: Exception type to handler mapping.
            default_handler: Default handler for unhandled exceptions.
            reraise: Whether to reraise unhandled exceptions.
            priority: Execution priority (lower runs first).
        """
        super().__init__(priority=priority)
        self._handlers = handlers or {}
        self._default = default_handler
        self._reraise = reraise

    async def on_error(self, context: MiddlewareContext, error: Exception) -> Any:
        """Handle error."""
        context.error = error

        # Find matching handler
        for exc_type, handler in self._handlers.items():
            if isinstance(error, exc_type):
                return handler(error)

        # Use default handler if available
        if self._default:
            return self._default(error)

        # Reraise if configured
        if self._reraise:
            raise error

        # Return None if not reraising
        return None


class RetryMiddleware(Middleware):
    """Middleware that retries failed operations.

    Example:
        pipeline.use(RetryMiddleware(
            max_retries=3,
            retry_on=(ConnectionError, TimeoutError),
        ))
    """

    def __init__(
        self,
        max_retries: int = 3,
        retry_on: tuple = (Exception,),
        delay: float = 1.0,
        backoff: float = 2.0,
        priority: int = 100,
    ) -> None:
        """Initialize retry middleware.

        Args:
            max_retries: Maximum retry attempts.
            retry_on: Exception types to retry on.
            delay: Initial delay between retries.
            backoff: Backoff multiplier.
            priority: Execution priority (lower runs first).
        """
        super().__init__(priority=priority)
        self._max_retries = max_retries
        self._retry_on = retry_on
        self._delay = delay
        self._backoff = backoff

    async def __call__(
        self,
        context: MiddlewareContext,
        next_handler: NextHandler,
    ) -> Any:
        """Execute with retry logic."""
        last_error: Optional[Exception] = None
        delay = self._delay

        for attempt in range(self._max_retries + 1):
            try:
                return await next_handler()

            except self._retry_on as e:
                last_error = e
                if attempt < self._max_retries:
                    logger.warning(f"Retry {attempt + 1}/{self._max_retries} " f"after error: {e}")
                    await asyncio.sleep(delay)
                    delay *= self._backoff

        if last_error:
            raise last_error


class TimeoutMiddleware(Middleware):
    """Middleware that enforces execution timeout.

    Example:
        pipeline.use(TimeoutMiddleware(timeout=30.0))
    """

    def __init__(self, timeout: float = 30.0, priority: int = 100) -> None:
        """Initialize timeout middleware.

        Args:
            timeout: Timeout in seconds.
            priority: Execution priority (lower runs first).
        """
        super().__init__(priority=priority)
        self._timeout = timeout

    async def __call__(
        self,
        context: MiddlewareContext,
        next_handler: NextHandler,
    ) -> Any:
        """Execute with timeout."""
        try:
            return await asyncio.wait_for(
                next_handler(),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError:
            raise TimeoutError(f"Request timed out after {self._timeout}s")


class CachingMiddleware(Middleware):
    """Middleware that caches responses.

    Example:
        cache = {}
        pipeline.use(CachingMiddleware(
            cache=cache,
            key_fn=lambda ctx: ctx.request.get("id"),
            ttl=300,
        ))
    """

    def __init__(
        self,
        cache: Dict[str, Any],
        key_fn: Callable[[MiddlewareContext], Optional[str]],
        ttl: float = 300.0,
        priority: int = 100,
    ) -> None:
        """Initialize caching middleware.

        Args:
            cache: Cache dictionary.
            key_fn: Function to generate cache key.
            ttl: Cache TTL in seconds.
            priority: Execution priority (lower runs first).
        """
        super().__init__(priority=priority)
        self._cache = cache
        self._key_fn = key_fn
        self._ttl = ttl

    async def __call__(
        self,
        context: MiddlewareContext,
        next_handler: NextHandler,
    ) -> Any:
        """Execute with caching."""
        key = self._key_fn(context)

        if key is None:
            return await next_handler()

        # Check cache
        cached = self._cache.get(key)
        if cached:
            timestamp, value = cached
            if time.time() - timestamp < self._ttl:
                context.set("cache_hit", True)
                return value

        # Execute and cache
        result = await next_handler()
        self._cache[key] = (time.time(), result)
        context.set("cache_hit", False)

        return result


class ValidationMiddleware(Middleware):
    """Middleware that validates requests.

    Example:
        def validate(ctx):
            if not ctx.request.get("id"):
                raise ValueError("id is required")

        pipeline.use(ValidationMiddleware(validator=validate))
    """

    def __init__(
        self,
        validator: Callable[[MiddlewareContext], None],
        priority: int = 100,
    ) -> None:
        """Initialize validation middleware.

        Args:
            validator: Validation function that raises on invalid.
            priority: Execution priority (lower runs first).
        """
        super().__init__(priority=priority)
        self._validator = validator

    async def before(self, context: MiddlewareContext) -> bool:
        """Validate request."""
        self._validator(context)
        return True


class MetricsMiddleware(Middleware):
    """Middleware that collects metrics.

    Example:
        from victor.observability import MetricsRegistry

        registry = MetricsRegistry()
        pipeline.use(MetricsMiddleware(registry=registry))
    """

    def __init__(
        self,
        registry: Optional[Any] = None,
        prefix: str = "middleware",
        priority: int = 100,
    ) -> None:
        """Initialize metrics middleware.

        Args:
            registry: Metrics registry.
            prefix: Metric name prefix.
            priority: Execution priority (lower runs first).
        """
        super().__init__(priority=priority)
        self._registry = registry
        self._prefix = prefix
        self._request_count = 0
        self._error_count = 0
        self._total_duration = 0.0

    async def before(self, context: MiddlewareContext) -> bool:
        """Track request start."""
        self._request_count += 1
        return True

    async def after(self, context: MiddlewareContext, response: Any) -> Any:
        """Track request completion."""
        self._total_duration += context.elapsed_ms
        return response

    async def on_error(self, context: MiddlewareContext, error: Exception) -> Any:
        """Track error."""
        self._error_count += 1
        raise error

    def get_metrics(self) -> Dict[str, Any]:
        """Get collected metrics."""
        return {
            f"{self._prefix}_requests": self._request_count,
            f"{self._prefix}_errors": self._error_count,
            f"{self._prefix}_total_duration_ms": self._total_duration,
            f"{self._prefix}_avg_duration_ms": (
                self._total_duration / self._request_count if self._request_count > 0 else 0
            ),
        }


# =============================================================================
# Pipeline Builder (Builder Pattern)
# =============================================================================


class PipelineBuilder(Generic[TRequest, TResponse]):
    """Builder for creating middleware pipelines.

    Provides a fluent interface for building pipelines with
    common middleware configurations.

    Example:
        pipeline = (
            PipelineBuilder()
            .with_logging()
            .with_timing()
            .with_error_handling()
            .with_timeout(30.0)
            .build()
        )
    """

    def __init__(self) -> None:
        """Initialize builder."""
        self._pipeline: MiddlewarePipeline = MiddlewarePipeline()

    def with_logging(
        self,
        log_level: int = logging.INFO,
    ) -> "PipelineBuilder[TRequest, TResponse]":
        """Add logging middleware."""
        self._pipeline.use(LoggingMiddleware(log_level=log_level))
        return self

    def with_timing(
        self,
        metric_key: str = "timing",
    ) -> "PipelineBuilder[TRequest, TResponse]":
        """Add timing middleware."""
        self._pipeline.use(TimingMiddleware(metric_key=metric_key))
        return self

    def with_error_handling(
        self,
        handlers: Optional[Dict[Type[Exception], Callable]] = None,
    ) -> "PipelineBuilder[TRequest, TResponse]":
        """Add error handling middleware."""
        self._pipeline.use(ErrorHandlingMiddleware(handlers=handlers))
        return self

    def with_retry(
        self,
        max_retries: int = 3,
        retry_on: tuple = (Exception,),
    ) -> "PipelineBuilder[TRequest, TResponse]":
        """Add retry middleware."""
        self._pipeline.use(RetryMiddleware(max_retries=max_retries, retry_on=retry_on))
        return self

    def with_timeout(
        self,
        timeout: float = 30.0,
    ) -> "PipelineBuilder[TRequest, TResponse]":
        """Add timeout middleware."""
        self._pipeline.use(TimeoutMiddleware(timeout=timeout))
        return self

    def with_validation(
        self,
        validator: Callable[[MiddlewareContext], None],
    ) -> "PipelineBuilder[TRequest, TResponse]":
        """Add validation middleware."""
        self._pipeline.use(ValidationMiddleware(validator=validator))
        return self

    def with_metrics(
        self,
        registry: Optional[Any] = None,
        prefix: str = "middleware",
    ) -> "PipelineBuilder[TRequest, TResponse]":
        """Add metrics middleware."""
        self._pipeline.use(MetricsMiddleware(registry=registry, prefix=prefix))
        return self

    def with_custom(
        self,
        middleware: Middleware[TRequest, TResponse],
    ) -> "PipelineBuilder[TRequest, TResponse]":
        """Add custom middleware."""
        self._pipeline.use(middleware)
        return self

    def build(self) -> MiddlewarePipeline[TRequest, TResponse]:
        """Build the pipeline.

        Returns:
            Configured middleware pipeline.
        """
        return self._pipeline.freeze()


# =============================================================================
# Factory Functions
# =============================================================================


def create_default_pipeline() -> MiddlewarePipeline:
    """Create pipeline with default middleware.

    Returns:
        Pipeline with logging, timing, and error handling.
    """
    return PipelineBuilder().with_error_handling().with_logging().with_timing().build()


def create_resilient_pipeline(
    max_retries: int = 3,
    timeout: float = 30.0,
) -> MiddlewarePipeline:
    """Create pipeline with resilience features.

    Args:
        max_retries: Maximum retry attempts.
        timeout: Request timeout in seconds.

    Returns:
        Pipeline with retry, timeout, and error handling.
    """
    return (
        PipelineBuilder()
        .with_error_handling()
        .with_logging()
        .with_timeout(timeout)
        .with_retry(max_retries=max_retries)
        .with_timing()
        .build()
    )

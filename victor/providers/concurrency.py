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
Concurrent Request Handling for LLM Providers.

This module provides efficient concurrent request management:
- Token bucket rate limiting for API quota management
- Sliding window rate limiting for request counting
- Priority-based request queuing
- Parallel tool execution with concurrency control

Usage:
    from victor.providers.concurrency import ConcurrentRequestManager

    manager = ConcurrentRequestManager()

    # Submit single request
    result = await manager.submit(
        provider.chat(messages, model=model),
        priority=RequestPriority.HIGH,
    )

    # Submit parallel requests
    results = await manager.submit_parallel([
        provider.chat(msg1, model=model),
        provider.chat(msg2, model=model),
    ])

    # Execute tool calls in parallel
    results = await manager.execute_tool_calls_parallel(
        tool_calls,
        executor=tool_executor,
    )
"""

import asyncio
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, Generic, List, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RequestPriority(Enum):
    """Priority levels for request scheduling.

    Lower values indicate higher priority.
    """

    CRITICAL = 0  # User-facing, immediate response needed
    HIGH = 1  # Important operations
    NORMAL = 2  # Standard requests
    LOW = 3  # Background tasks
    BATCH = 4  # Bulk operations


@dataclass(order=True)
class PrioritizedRequest(Generic[T]):
    """A request with priority for queue ordering.

    Ordering is by (priority, timestamp) so higher priority
    requests are processed first, with FIFO ordering within
    the same priority level.
    """

    priority: int
    timestamp: float = field(compare=True)
    request_id: str = field(compare=False)
    coroutine: Awaitable[T] = field(compare=False, repr=False)
    future: asyncio.Future = field(compare=False, repr=False)
    estimated_tokens: int = field(compare=False, default=1000)
    metadata: Dict[str, Any] = field(compare=False, default_factory=dict)


class TokenBucketRateLimiter:
    """
    Token bucket rate limiter for API quota management.

    Features:
    - Configurable tokens per second refill rate
    - Burst capacity for handling spikes
    - Async-friendly waiting
    - Non-blocking availability check

    Usage:
        limiter = TokenBucketRateLimiter(
            tokens_per_second=10.0,  # 10 tokens per second
            burst_capacity=50,       # Can burst up to 50 tokens
        )

        # Wait for tokens
        wait_time = await limiter.acquire(tokens=5)

        # Check availability without waiting
        if limiter.try_acquire(tokens=5):
            # Tokens acquired
            pass
    """

    def __init__(
        self,
        tokens_per_second: float,
        burst_capacity: int,
    ):
        """Initialize token bucket.

        Args:
            tokens_per_second: Token refill rate
            burst_capacity: Maximum token capacity
        """
        self.tokens_per_second = tokens_per_second
        self.burst_capacity = burst_capacity

        self._tokens = float(burst_capacity)
        self._last_update = time.monotonic()
        self._lock = asyncio.Lock()

        logger.debug(
            f"TokenBucketRateLimiter initialized. "
            f"Rate: {tokens_per_second}/s, Burst: {burst_capacity}"
        )

    async def acquire(self, tokens: int = 1) -> float:
        """Acquire tokens, waiting if necessary.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            Time waited in seconds
        """
        async with self._lock:
            self._refill()

            if self._tokens >= tokens:
                self._tokens -= tokens
                return 0.0

            # Calculate wait time
            tokens_needed = tokens - self._tokens
            wait_time = tokens_needed / self.tokens_per_second

            # Wait and then acquire
            await asyncio.sleep(wait_time)

            self._refill()
            self._tokens = max(0, self._tokens - tokens)

            return wait_time

    def try_acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens without waiting.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            True if tokens were acquired, False otherwise
        """
        self._refill()

        if self._tokens >= tokens:
            self._tokens -= tokens
            return True

        return False

    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_update
        self._last_update = now

        self._tokens = min(
            self.burst_capacity,
            self._tokens + elapsed * self.tokens_per_second,
        )

    @property
    def available_tokens(self) -> float:
        """Get current available tokens (approximate)."""
        self._refill()
        return self._tokens

    @property
    def is_full(self) -> bool:
        """Check if bucket is full."""
        return self._tokens >= self.burst_capacity


class SlidingWindowRateLimiter:
    """
    Sliding window rate limiter for request counting.

    More accurate than token bucket for API rate limits
    that count requests per time window.

    Features:
    - Precise request counting within time window
    - Async-friendly waiting
    - Automatic cleanup of old entries

    Usage:
        limiter = SlidingWindowRateLimiter(
            max_requests=50,    # 50 requests
            window_seconds=60,  # per minute
        )

        # Wait for permission
        wait_time = await limiter.acquire()

        # Check capacity
        available = limiter.available_capacity
    """

    def __init__(
        self,
        max_requests: int,
        window_seconds: float,
    ):
        """Initialize sliding window limiter.

        Args:
            max_requests: Maximum requests per window
            window_seconds: Window duration in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds

        self._requests: List[float] = []
        self._lock = asyncio.Lock()

        logger.debug(
            f"SlidingWindowRateLimiter initialized. " f"Max: {max_requests} per {window_seconds}s"
        )

    async def acquire(self) -> float:
        """Acquire permission to make a request.

        Returns:
            Time waited in seconds
        """
        async with self._lock:
            now = time.monotonic()

            # Remove old requests outside window
            cutoff = now - self.window_seconds
            self._requests = [t for t in self._requests if t > cutoff]

            if len(self._requests) < self.max_requests:
                self._requests.append(now)
                return 0.0

            # Calculate wait time until oldest request expires
            oldest = self._requests[0]
            wait_time = oldest + self.window_seconds - now + 0.01  # Small buffer

            if wait_time > 0:
                await asyncio.sleep(wait_time)
                now = time.monotonic()

            # Clean up and add new request
            cutoff = now - self.window_seconds
            self._requests = [t for t in self._requests if t > cutoff]
            self._requests.append(now)

            return max(0, wait_time)

    def try_acquire(self) -> bool:
        """Try to acquire permission without waiting.

        Returns:
            True if permission granted, False otherwise
        """
        now = time.monotonic()
        cutoff = now - self.window_seconds
        self._requests = [t for t in self._requests if t > cutoff]

        if len(self._requests) < self.max_requests:
            self._requests.append(now)
            return True

        return False

    @property
    def available_capacity(self) -> int:
        """Get current available capacity."""
        now = time.monotonic()
        cutoff = now - self.window_seconds
        current_count = sum(1 for t in self._requests if t > cutoff)
        return max(0, self.max_requests - current_count)

    @property
    def time_until_available(self) -> float:
        """Get time until next request is available."""
        if self.available_capacity > 0:
            return 0.0

        now = time.monotonic()
        cutoff = now - self.window_seconds
        valid_requests = [t for t in self._requests if t > cutoff]

        if not valid_requests:
            return 0.0

        oldest = min(valid_requests)
        return max(0, oldest + self.window_seconds - now)


@dataclass
class ConcurrencyConfig:
    """Configuration for concurrent request handling.

    Attributes:
        max_concurrent_requests: Maximum parallel API requests
        max_concurrent_tool_calls: Maximum parallel tool executions
        requests_per_minute: Rate limit for requests per minute
        tokens_per_minute: Rate limit for tokens per minute
        max_queue_size: Maximum pending requests in queue
        queue_timeout_seconds: Timeout for queued requests
    """

    max_concurrent_requests: int = 10
    max_concurrent_tool_calls: int = 5

    # Rate limiting
    requests_per_minute: int = 50
    tokens_per_minute: int = 50000

    # Queue settings
    max_queue_size: int = 100
    queue_timeout_seconds: float = 300.0

    # Backpressure
    enable_backpressure: bool = True
    backpressure_threshold: float = 0.8  # 80% queue utilization


class QueueFullError(Exception):
    """Raised when request queue is full."""

    def __init__(self, queue_size: int, max_size: int):
        self.queue_size = queue_size
        self.max_size = max_size
        super().__init__(f"Request queue full ({queue_size}/{max_size})")


class RequestTimeoutError(Exception):
    """Raised when request times out in queue."""

    def __init__(self, request_id: str, timeout: float):
        self.request_id = request_id
        self.timeout = timeout
        super().__init__(f"Request {request_id} timed out after {timeout}s")


class RequestQueue:
    """
    Manages concurrent API requests with rate limiting and queuing.

    Features:
    - Priority-based request scheduling (true priority queue)
    - Dual rate limiting (requests + tokens)
    - Concurrent execution with semaphores
    - Request timeout handling
    - Backpressure support
    - Worker-based processing for priority ordering

    Note: Previously named 'ConcurrentRequestManager'. Renamed for clarity.

    Usage:
        queue = RequestQueue()

        # Submit request
        result = await queue.submit(
            provider.chat(messages, model=model),
            priority=RequestPriority.HIGH,
            estimated_tokens=1000,
        )

        # Submit parallel requests
        results = await queue.submit_parallel([
            provider.chat(msg1, model=model),
            provider.chat(msg2, model=model),
        ])

        # Get statistics
        stats = queue.get_stats()

        # Cleanup when done
        await queue.shutdown()
    """

    def __init__(
        self,
        config: Optional[ConcurrencyConfig] = None,
        num_workers: int = 3,
    ):
        """Initialize concurrent request manager.

        Args:
            config: Concurrency configuration
            num_workers: Number of worker tasks processing the queue
        """
        self.config = config or ConcurrencyConfig()
        self._num_workers = num_workers

        # Semaphores for concurrency control
        self._request_semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        self._tool_semaphore = asyncio.Semaphore(self.config.max_concurrent_tool_calls)

        # Rate limiters
        self._request_limiter = SlidingWindowRateLimiter(
            max_requests=self.config.requests_per_minute,
            window_seconds=60.0,
        )
        self._token_limiter = TokenBucketRateLimiter(
            tokens_per_second=self.config.tokens_per_minute / 60,
            burst_capacity=self.config.tokens_per_minute // 2,
        )

        # Priority queue for request ordering
        # Using asyncio.PriorityQueue for thread-safe priority scheduling
        self._request_queue: asyncio.PriorityQueue = asyncio.PriorityQueue(
            maxsize=self.config.max_queue_size
        )

        # Request tracking
        self._pending_requests: Dict[str, PrioritizedRequest] = {}
        self._active_requests: Dict[str, PrioritizedRequest] = {}
        self._lock = asyncio.Lock()

        # Worker management
        self._workers: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()
        self._started = False

        # Statistics
        self._stats = {
            "total_submitted": 0,
            "total_completed": 0,
            "total_failed": 0,
            "total_timeouts": 0,
            "total_wait_time": 0.0,
            "total_tokens": 0,
        }
        # Rolling window of recent wait times for percentile tracking
        self._wait_times: deque = deque(maxlen=1000)

        logger.info(
            f"ConcurrentRequestManager initialized. "
            f"Max concurrent: {self.config.max_concurrent_requests}, "
            f"RPM: {self.config.requests_per_minute}, "
            f"Workers: {num_workers}"
        )

    def _ensure_started(self) -> None:
        """Ensure worker tasks are running (lazy initialization)."""
        if not self._started:
            self._started = True
            for i in range(self._num_workers):
                worker = asyncio.create_task(
                    self._worker_loop(f"worker_{i}"),
                    name=f"request_worker_{i}",
                )
                self._workers.append(worker)
            logger.debug(f"Started {self._num_workers} request workers")

    async def _worker_loop(self, worker_name: str) -> None:
        """Worker loop that processes requests from the priority queue.

        Args:
            worker_name: Name of this worker for logging
        """
        logger.debug(f"{worker_name} started")

        while not self._shutdown_event.is_set():
            try:
                # Wait for a request with timeout to allow checking shutdown
                try:
                    request = await asyncio.wait_for(
                        self._request_queue.get(),
                        timeout=0.5,
                    )
                except asyncio.TimeoutError:
                    continue

                # Process the request
                await self._process_request(request)
                self._request_queue.task_done()

            except asyncio.CancelledError:
                logger.debug(f"{worker_name} cancelled")
                break
            except Exception as e:
                logger.error(f"{worker_name} error: {e}")

        logger.debug(f"{worker_name} stopped")

    async def shutdown(self) -> None:
        """Shutdown the manager and stop all workers."""
        logger.info("Shutting down ConcurrentRequestManager...")
        self._shutdown_event.set()

        # Cancel all workers
        for worker in self._workers:
            worker.cancel()

        # Wait for workers to finish
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)

        self._workers.clear()
        self._started = False
        logger.info("ConcurrentRequestManager shutdown complete")

    async def __aenter__(self) -> "RequestQueue":
        """Enable async context manager usage."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Shutdown on context exit."""
        await self.shutdown()

    def __del__(self) -> None:
        if self._started:
            import warnings

            warnings.warn(
                "RequestQueue was not shut down. " "Call shutdown() or use 'async with'.",
                ResourceWarning,
                stacklevel=2,
            )

    async def submit(
        self,
        coroutine: Awaitable[T],
        priority: RequestPriority = RequestPriority.NORMAL,
        estimated_tokens: int = 1000,
        request_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> T:
        """Submit a request for execution.

        Requests are queued and processed in priority order by worker tasks.
        Higher priority requests (lower enum value) are processed first.

        Args:
            coroutine: Async function to execute
            priority: Request priority (CRITICAL > HIGH > NORMAL > LOW > BATCH)
            estimated_tokens: Estimated token usage for rate limiting
            request_id: Optional request identifier
            metadata: Optional request metadata

        Returns:
            Result of the coroutine

        Raises:
            QueueFullError: If queue is full
            RequestTimeoutError: If request times out
        """
        # Ensure workers are running (lazy start)
        self._ensure_started()

        if request_id is None:
            request_id = f"req_{uuid.uuid4().hex[:8]}"

        # Check queue capacity and backpressure
        async with self._lock:
            current_size = self._request_queue.qsize()
            if current_size >= self.config.max_queue_size:
                raise QueueFullError(current_size, self.config.max_queue_size)

            # Check backpressure
            if self.config.enable_backpressure:
                utilization = current_size / self.config.max_queue_size
                if utilization > self.config.backpressure_threshold:
                    logger.warning(
                        f"Queue utilization high: {utilization:.1%}. "
                        "Consider reducing request rate."
                    )

        # Create prioritized request
        try:
            loop = asyncio.get_running_loop()
            future: asyncio.Future = loop.create_future()
        except RuntimeError:
            future = asyncio.get_event_loop().create_future()

        request = PrioritizedRequest(
            priority=priority.value,
            timestamp=time.monotonic(),
            request_id=request_id,
            coroutine=coroutine,
            future=future,
            estimated_tokens=estimated_tokens,
            metadata=metadata or {},
        )

        # Track request and add to priority queue
        async with self._lock:
            self._pending_requests[request_id] = request
            self._stats["total_submitted"] += 1

        # Add to priority queue - workers will process in priority order
        await self._request_queue.put(request)

        # Wait for result with timeout
        try:
            return await asyncio.wait_for(
                future,
                timeout=self.config.queue_timeout_seconds,
            )
        except asyncio.TimeoutError:
            self._stats["total_timeouts"] += 1
            async with self._lock:
                self._pending_requests.pop(request_id, None)
            raise RequestTimeoutError(request_id, self.config.queue_timeout_seconds)

    async def submit_parallel(
        self,
        coroutines: List[Awaitable[T]],
        priority: RequestPriority = RequestPriority.NORMAL,
        return_exceptions: bool = True,
    ) -> List[T]:
        """Submit multiple requests for parallel execution.

        Args:
            coroutines: List of async functions to execute
            priority: Priority for all requests
            return_exceptions: If True, exceptions are returned in results

        Returns:
            List of results in same order as input
        """
        tasks = [self.submit(coro, priority=priority) for coro in coroutines]

        return await asyncio.gather(*tasks, return_exceptions=return_exceptions)

    async def execute_tool_calls_parallel(
        self,
        tool_calls: List[Dict[str, Any]],
        executor: Callable[[Dict[str, Any]], Awaitable[Any]],
        max_concurrency: Optional[int] = None,
    ) -> List[Any]:
        """Execute multiple tool calls in parallel with concurrency control.

        Args:
            tool_calls: List of tool call definitions
            executor: Function to execute each tool call
            max_concurrency: Override max concurrent tool calls

        Returns:
            List of results in same order as input
        """
        semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else self._tool_semaphore

        async def execute_with_semaphore(tool_call: Dict[str, Any]) -> Any:
            async with semaphore:
                return await executor(tool_call)

        tasks = [execute_with_semaphore(tc) for tc in tool_calls]

        return await asyncio.gather(*tasks, return_exceptions=True)

    async def _process_request(self, request: PrioritizedRequest):
        """Process a single request with rate limiting."""
        try:
            # Acquire rate limit tokens
            request_wait = await self._request_limiter.acquire()
            token_wait = await self._token_limiter.acquire(request.estimated_tokens)
            total_wait = request_wait + token_wait

            self._stats["total_wait_time"] += total_wait
            self._wait_times.append(total_wait)

            if total_wait > 0:
                logger.debug(
                    f"Request {request.request_id} waited {total_wait:.2f}s " f"for rate limit"
                )

            # Acquire concurrency semaphore
            async with self._request_semaphore:
                # Move from pending to active
                async with self._lock:
                    self._pending_requests.pop(request.request_id, None)
                    self._active_requests[request.request_id] = request

                try:
                    result = await request.coroutine
                    request.future.set_result(result)
                    self._stats["total_completed"] += 1
                    self._stats["total_tokens"] += request.estimated_tokens

                except Exception as e:
                    if not request.future.done():
                        request.future.set_exception(e)
                    self._stats["total_failed"] += 1

                finally:
                    async with self._lock:
                        self._active_requests.pop(request.request_id, None)

        except Exception as e:
            if not request.future.done():
                request.future.set_exception(e)
            self._stats["total_failed"] += 1

            async with self._lock:
                self._pending_requests.pop(request.request_id, None)

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        return {
            **self._stats,
            "pending_requests": len(self._pending_requests),
            "active_requests": len(self._active_requests),
            "request_capacity": self._request_limiter.available_capacity,
            "token_capacity": self._token_limiter.available_tokens,
            "queue_utilization": len(self._pending_requests) / self.config.max_queue_size,
            "avg_wait_time": (
                self._stats["total_wait_time"] / self._stats["total_submitted"]
                if self._stats["total_submitted"] > 0
                else 0
            ),
            "success_rate": (
                self._stats["total_completed"]
                / (self._stats["total_completed"] + self._stats["total_failed"])
                if (self._stats["total_completed"] + self._stats["total_failed"]) > 0
                else 0
            ),
            **self._compute_wait_time_percentiles(),
        }

    def _compute_wait_time_percentiles(self) -> Dict[str, float]:
        """Compute wait time percentiles from recent requests."""
        if not self._wait_times:
            return {"wait_time_p50": 0.0, "wait_time_p95": 0.0, "wait_time_p99": 0.0}
        sorted_times = sorted(self._wait_times)
        n = len(sorted_times)
        return {
            "wait_time_p50": sorted_times[int(n * 0.50)],
            "wait_time_p95": sorted_times[min(int(n * 0.95), n - 1)],
            "wait_time_p99": sorted_times[min(int(n * 0.99), n - 1)],
        }

    def reset_stats(self):
        """Reset statistics."""
        self._stats = {
            "total_submitted": 0,
            "total_completed": 0,
            "total_failed": 0,
            "total_timeouts": 0,
            "total_wait_time": 0.0,
            "total_tokens": 0,
        }
        self._wait_times.clear()


# Pre-configured rate limits for different providers
PROVIDER_RATE_LIMITS: Dict[str, Dict[str, ConcurrencyConfig]] = {
    "anthropic": {
        "claude-3-5-haiku-20241022": ConcurrencyConfig(
            max_concurrent_requests=10,
            requests_per_minute=50,
            tokens_per_minute=50000,
        ),
        "claude-sonnet-4-5-20250929": ConcurrencyConfig(
            max_concurrent_requests=5,
            requests_per_minute=50,
            tokens_per_minute=40000,
        ),
        "claude-opus-4-5-20251101": ConcurrencyConfig(
            max_concurrent_requests=3,
            requests_per_minute=50,
            tokens_per_minute=20000,
        ),
        "default": ConcurrencyConfig(
            max_concurrent_requests=5,
            requests_per_minute=50,
            tokens_per_minute=40000,
        ),
    },
    "openai": {
        "gpt-4": ConcurrencyConfig(
            max_concurrent_requests=5,
            requests_per_minute=60,
            tokens_per_minute=40000,
        ),
        "gpt-4-turbo": ConcurrencyConfig(
            max_concurrent_requests=10,
            requests_per_minute=500,
            tokens_per_minute=150000,
        ),
        "default": ConcurrencyConfig(
            max_concurrent_requests=10,
            requests_per_minute=100,
            tokens_per_minute=90000,
        ),
    },
    "google": {
        "default": ConcurrencyConfig(
            max_concurrent_requests=10,
            requests_per_minute=60,
            tokens_per_minute=100000,
        ),
    },
    "ollama": {
        "default": ConcurrencyConfig(
            max_concurrent_requests=3,
            requests_per_minute=1000,  # Local, no real limit
            tokens_per_minute=1000000,
        ),
    },
}


def get_provider_config(
    provider: str,
    model: Optional[str] = None,
) -> ConcurrencyConfig:
    """Get rate limit configuration for a provider/model.

    Args:
        provider: Provider name
        model: Optional model name

    Returns:
        ConcurrencyConfig for the provider/model
    """
    provider_configs = PROVIDER_RATE_LIMITS.get(provider.lower(), {})

    if model and model in provider_configs:
        return provider_configs[model]

    return provider_configs.get("default", ConcurrencyConfig())

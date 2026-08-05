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

"""Batch workflow executor for running workflows over multiple inputs.

Provides parallel batch processing with configurable batching, delays,
rate limiting, progress tracking, and retry logic.

Example:
    executor = BatchWorkflowExecutor(workflow_executor, batch_config)

    # Define inputs
    inputs = [
        {"symbol": "AAPL", "date": "2024-01-01"},
        {"symbol": "GOOGL", "date": "2024-01-01"},
        {"symbol": "MSFT", "date": "2024-01-01"},
    ]

    # Run batch with progress callback
    results = await executor.execute_batch(
        workflow=workflow,
        inputs=inputs,
        on_progress=lambda p: print(f"Progress: {p.completed}/{p.total}"),
    )
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    TypeVar,
)

from victor.core.retry import compute_backoff_delay

if TYPE_CHECKING:
    from victor.workflows.definition import WorkflowDefinition
    from victor.workflows.context import WorkflowResult
    from victor.workflows.protocols import IWorkflowRuntimeExecutor

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BatchRetryStrategy(Enum):
    """Retry strategy for failed batch items.

    Renamed from RetryStrategy to be semantically distinct:
    - BatchRetryStrategy (here): Enum for batch retry modes
    - BaseRetryStrategy (victor.core.retry): Abstract base with should_retry(), get_delay()
    - ProviderRetryStrategy (victor.providers.resilience): Concrete provider retry with execute()
    """

    NONE = "none"  # No retries
    IMMEDIATE = "immediate"  # Retry immediately
    END_OF_BATCH = "end_of_batch"  # Retry at end of batch
    EXPONENTIAL_BACKOFF = "exponential_backoff"  # Retry with backoff


@dataclass
class BatchConfig:
    """Configuration for batch execution.

    Attributes:
        batch_size: Number of items to process per batch
        max_concurrent: Maximum concurrent workflow executions
        delay_seconds: Delay between batches in seconds
        retry_strategy: How to handle failed items
        max_retries: Maximum retry attempts per item
        retry_delay_seconds: Initial delay between retries
        progress_interval: How often to emit progress updates (items)
        timeout_per_item: Timeout per workflow execution in seconds
        fail_fast: Stop entire batch on first failure
    """

    batch_size: int = 5
    max_concurrent: int = 3
    delay_seconds: float = 1.0
    retry_strategy: BatchRetryStrategy = BatchRetryStrategy.END_OF_BATCH
    max_retries: int = 2
    retry_delay_seconds: float = 5.0
    progress_interval: int = 1
    timeout_per_item: Optional[float] = None
    fail_fast: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BatchConfig":
        """Create config from dictionary (e.g., from YAML metadata)."""
        retry_str = data.get("retry_strategy", "end_of_batch")
        retry_strategy = (
            BatchRetryStrategy(retry_str) if retry_str else BatchRetryStrategy.END_OF_BATCH
        )

        return cls(
            batch_size=data.get("batch_size", 5),
            max_concurrent=data.get("max_concurrent", 3),
            delay_seconds=data.get("delay_seconds", 1.0),
            retry_strategy=retry_strategy,
            max_retries=data.get("max_retries", 2),
            retry_delay_seconds=data.get("retry_delay_seconds", 5.0),
            progress_interval=data.get("progress_interval", 1),
            timeout_per_item=data.get("timeout_per_item"),
            fail_fast=data.get("fail_fast", False),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "batch_size": self.batch_size,
            "max_concurrent": self.max_concurrent,
            "delay_seconds": self.delay_seconds,
            "retry_strategy": self.retry_strategy.value,
            "max_retries": self.max_retries,
            "retry_delay_seconds": self.retry_delay_seconds,
            "progress_interval": self.progress_interval,
            "timeout_per_item": self.timeout_per_item,
            "fail_fast": self.fail_fast,
        }


class ItemStatus(Enum):
    """Status of a batch item."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    SKIPPED = "skipped"


@dataclass
class BatchItemResult(Generic[T]):
    """Result for a single batch item.

    Attributes:
        input: The input that was processed
        status: Execution status
        result: Workflow result if completed
        error: Error message if failed
        attempts: Number of execution attempts
        duration_seconds: Total execution time
    """

    input: T
    status: ItemStatus
    result: Optional["WorkflowResult"] = None
    error: Optional[str] = None
    attempts: int = 1
    duration_seconds: float = 0.0

    @property
    def success(self) -> bool:
        """Check if item completed successfully."""
        return self.status == ItemStatus.COMPLETED


@dataclass
class BatchProgress:
    """Progress information for batch execution.

    Attributes:
        total: Total number of items
        completed: Number of completed items
        failed: Number of failed items
        pending: Number of pending items
        current_batch: Current batch number
        total_batches: Total number of batches
        elapsed_seconds: Elapsed time
        estimated_remaining_seconds: Estimated time remaining
    """

    total: int
    completed: int = 0
    failed: int = 0
    pending: int = 0
    current_batch: int = 0
    total_batches: int = 0
    elapsed_seconds: float = 0.0
    estimated_remaining_seconds: Optional[float] = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate (0-1)."""
        processed = self.completed + self.failed
        if processed == 0:
            return 0.0
        return self.completed / processed

    @property
    def percent_complete(self) -> float:
        """Calculate percentage complete (0-100)."""
        if self.total == 0:
            return 100.0
        return (self.completed + self.failed) / self.total * 100

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "total": self.total,
            "completed": self.completed,
            "failed": self.failed,
            "pending": self.pending,
            "current_batch": self.current_batch,
            "total_batches": self.total_batches,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "estimated_remaining_seconds": (
                round(self.estimated_remaining_seconds, 2)
                if self.estimated_remaining_seconds
                else None
            ),
            "success_rate": round(self.success_rate, 3),
            "percent_complete": round(self.percent_complete, 1),
        }


@dataclass
class BatchResult(Generic[T]):
    """Result from batch execution.

    Attributes:
        items: Results for each input item
        total_duration: Total execution time
        total_successful: Count of successful items
        total_failed: Count of failed items
        aborted: Whether execution was aborted
        error: Error message if aborted
    """

    items: List[BatchItemResult[T]] = field(default_factory=list)
    total_duration: float = 0.0
    total_successful: int = 0
    total_failed: int = 0
    aborted: bool = False
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        """Check if batch completed successfully (all items succeeded)."""
        return not self.aborted and self.total_failed == 0

    @property
    def partial_success(self) -> bool:
        """Check if batch had any successful items."""
        return self.total_successful > 0

    def get_successful_results(self) -> List["WorkflowResult"]:
        """Get all successful workflow results."""
        return [item.result for item in self.items if item.success and item.result]

    def get_failed_inputs(self) -> List[T]:
        """Get inputs that failed."""
        return [item.input for item in self.items if item.status == ItemStatus.FAILED]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "total_duration": round(self.total_duration, 2),
            "total_successful": self.total_successful,
            "total_failed": self.total_failed,
            "success": self.success,
            "partial_success": self.partial_success,
            "aborted": self.aborted,
            "error": self.error,
            "item_count": len(self.items),
        }


# Progress callback protocol
class ProgressCallback(Protocol):
    """Protocol for progress callbacks."""

    def __call__(self, progress: BatchProgress) -> None:
        """Called when progress is updated."""
        ...


class BatchWorkflowExecutor(Generic[T]):
    """Executor for running workflows over batches of inputs.

    Provides:
    - Parallel execution with configurable concurrency
    - Batch-based processing with delays between batches
    - Retry logic with configurable strategies
    - Progress callbacks for monitoring
    - Graceful cancellation

    Example:
        executor = BatchWorkflowExecutor(
            workflow_executor=executor,
            config=BatchConfig(batch_size=5, max_concurrent=3),
        )

        results = await executor.execute_batch(
            workflow=my_workflow,
            inputs=[{"id": 1}, {"id": 2}, {"id": 3}],
            on_progress=lambda p: print(f"{p.percent_complete:.0f}% complete"),
        )

        print(f"Success: {results.total_successful}/{len(inputs)}")
    """

    def __init__(
        self,
        workflow_executor: "IWorkflowRuntimeExecutor",
        config: Optional[BatchConfig] = None,
    ):
        """Initialize batch executor.

        Args:
            workflow_executor: Underlying workflow executor
            config: Batch configuration (uses defaults if not provided)
        """
        self.workflow_executor = workflow_executor
        self.config = config or BatchConfig()
        self._cancelled = False
        self._active_tasks: Dict[str, asyncio.Task] = {}

    async def execute_batch(
        self,
        workflow: "WorkflowDefinition",
        inputs: List[T],
        *,
        on_progress: Optional[ProgressCallback] = None,
        config_override: Optional[BatchConfig] = None,
    ) -> BatchResult[T]:
        """Execute workflow over a batch of inputs.

        Args:
            workflow: Workflow to execute for each input
            inputs: List of input dictionaries
            on_progress: Optional callback for progress updates
            config_override: Override batch config for this execution

        Returns:
            BatchResult with all item results
        """
        config = config_override or self.config
        self._cancelled = False

        # Extract batch config from workflow metadata if present
        if "batch_config" in workflow.metadata and config_override is None:
            config = BatchConfig.from_dict(workflow.metadata["batch_config"])

        start_time = time.time()
        total = len(inputs)
        batch_size = config.batch_size

        # Calculate total batches
        total_batches = (total + batch_size - 1) // batch_size

        # Initialize progress
        progress = BatchProgress(
            total=total,
            pending=total,
            total_batches=total_batches,
        )

        # Initialize results
        item_results: List[BatchItemResult[T]] = [
            BatchItemResult(input=inp, status=ItemStatus.PENDING) for inp in inputs
        ]

        retry_queue: List[int] = []  # Indices of items to retry
        semaphore = asyncio.Semaphore(config.max_concurrent)

        logger.info(
            f"Starting batch execution: {total} items, "
            f"{total_batches} batches, max_concurrent={config.max_concurrent}"
        )

        try:
            for batch_idx in range(total_batches):
                if self._cancelled:
                    break

                progress.current_batch = batch_idx + 1
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, total)
                batch_indices = list(range(batch_start, batch_end))

                logger.debug(
                    f"Processing batch {batch_idx + 1}/{total_batches} "
                    f"(items {batch_start + 1}-{batch_end})"
                )

                # Execute batch items concurrently
                await self._execute_batch_items(
                    workflow=workflow,
                    item_results=item_results,
                    indices=batch_indices,
                    semaphore=semaphore,
                    config=config,
                    progress=progress,
                    on_progress=on_progress,
                    retry_queue=retry_queue,
                )

                # Update progress
                progress.elapsed_seconds = time.time() - start_time
                self._update_estimated_remaining(progress, start_time)

                if on_progress and (batch_idx + 1) % config.progress_interval == 0:
                    on_progress(progress)

                # Delay between batches (except for last batch)
                if batch_idx < total_batches - 1 and config.delay_seconds > 0:
                    await asyncio.sleep(config.delay_seconds)

            # Process retry queue at end if using END_OF_BATCH strategy
            if retry_queue and config.retry_strategy == BatchRetryStrategy.END_OF_BATCH:
                logger.info(f"Retrying {len(retry_queue)} failed items")
                await self._process_retry_queue(
                    workflow=workflow,
                    item_results=item_results,
                    retry_queue=retry_queue,
                    semaphore=semaphore,
                    config=config,
                    progress=progress,
                    on_progress=on_progress,
                )

        except asyncio.CancelledError:
            logger.warning("Batch execution cancelled")
            self._cancelled = True

        except Exception as e:
            logger.error(f"Batch execution failed: {e}", exc_info=True)
            return BatchResult(
                items=item_results,
                total_duration=time.time() - start_time,
                total_successful=progress.completed,
                total_failed=progress.failed,
                aborted=True,
                error=str(e),
            )

        # Build final result
        total_duration = time.time() - start_time
        result = BatchResult(
            items=item_results,
            total_duration=total_duration,
            total_successful=progress.completed,
            total_failed=progress.failed,
            aborted=self._cancelled,
        )

        logger.info(
            f"Batch execution complete: {result.total_successful}/{total} successful "
            f"in {total_duration:.1f}s"
        )

        return result

    async def _execute_batch_items(
        self,
        workflow: "WorkflowDefinition",
        item_results: List[BatchItemResult[T]],
        indices: List[int],
        semaphore: asyncio.Semaphore,
        config: BatchConfig,
        progress: BatchProgress,
        on_progress: Optional[ProgressCallback],
        retry_queue: List[int],
    ) -> None:
        """Execute a batch of items concurrently.

        Args:
            workflow: Workflow to execute
            item_results: List of all item results (updated in place)
            indices: Indices of items to process in this batch
            semaphore: Concurrency limiter
            config: Batch configuration
            progress: Progress tracker
            on_progress: Progress callback
            retry_queue: Queue for items to retry later
        """

        async def execute_item(idx: int) -> None:
            async with semaphore:
                if self._cancelled:
                    return

                item_result = item_results[idx]
                item_result.status = ItemStatus.RUNNING
                item_start = time.time()

                try:
                    # Execute workflow with this input
                    result = await self.workflow_executor.execute(
                        workflow,
                        initial_context=(
                            dict(item_result.input)
                            if isinstance(item_result.input, dict)
                            else {"input": item_result.input}
                        ),
                        timeout=config.timeout_per_item,
                    )

                    item_result.duration_seconds = time.time() - item_start
                    item_result.result = result

                    if result.success:
                        item_result.status = ItemStatus.COMPLETED
                        progress.completed += 1
                        progress.pending -= 1
                    else:
                        # Handle failure
                        await self._handle_failure(
                            idx=idx,
                            item_result=item_result,
                            error=result.error or "Unknown error",
                            config=config,
                            progress=progress,
                            retry_queue=retry_queue,
                        )

                except asyncio.TimeoutError:
                    item_result.duration_seconds = time.time() - item_start
                    await self._handle_failure(
                        idx=idx,
                        item_result=item_result,
                        error=f"Timed out after {config.timeout_per_item}s",
                        config=config,
                        progress=progress,
                        retry_queue=retry_queue,
                    )

                except Exception as e:
                    item_result.duration_seconds = time.time() - item_start
                    await self._handle_failure(
                        idx=idx,
                        item_result=item_result,
                        error=str(e),
                        config=config,
                        progress=progress,
                        retry_queue=retry_queue,
                    )

                # Emit progress after each item if configured
                if on_progress and progress.completed % config.progress_interval == 0:
                    on_progress(progress)

                # Fail fast if configured
                if config.fail_fast and item_result.status == ItemStatus.FAILED:
                    self._cancelled = True

        # Execute all items in this batch
        tasks = [execute_item(idx) for idx in indices]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _handle_failure(
        self,
        idx: int,
        item_result: BatchItemResult[T],
        error: str,
        config: BatchConfig,
        progress: BatchProgress,
        retry_queue: List[int],
    ) -> None:
        """Handle a failed item execution.

        Args:
            idx: Index of the failed item
            item_result: Result object to update
            error: Error message
            config: Batch configuration
            progress: Progress tracker
            retry_queue: Queue for items to retry later
        """
        item_result.error = error
        item_result.attempts += 1

        if item_result.attempts <= config.max_retries:
            if config.retry_strategy == BatchRetryStrategy.IMMEDIATE:
                # Retry immediately with backoff
                delay = compute_backoff_delay(item_result.attempts - 1, config.retry_delay_seconds)
                logger.debug(f"Retrying item {idx} after {delay}s (attempt {item_result.attempts})")
                item_result.status = ItemStatus.RETRYING
                await asyncio.sleep(delay)
                # Re-queue for immediate retry in current batch
                retry_queue.append(idx)

            elif config.retry_strategy == BatchRetryStrategy.END_OF_BATCH:
                # Queue for retry at end
                logger.debug(f"Queueing item {idx} for retry at end of batch")
                item_result.status = ItemStatus.RETRYING
                retry_queue.append(idx)

            elif config.retry_strategy == BatchRetryStrategy.EXPONENTIAL_BACKOFF:
                # Queue with exponential backoff marker
                item_result.status = ItemStatus.RETRYING
                retry_queue.append(idx)

            else:
                # No retry
                item_result.status = ItemStatus.FAILED
                progress.failed += 1
                progress.pending -= 1
        else:
            # Max retries exceeded
            item_result.status = ItemStatus.FAILED
            progress.failed += 1
            progress.pending -= 1
            logger.warning(f"Item {idx} failed after {item_result.attempts} attempts: {error}")

    async def _process_retry_queue(
        self,
        workflow: "WorkflowDefinition",
        item_results: List[BatchItemResult[T]],
        retry_queue: List[int],
        semaphore: asyncio.Semaphore,
        config: BatchConfig,
        progress: BatchProgress,
        on_progress: Optional[ProgressCallback],
    ) -> None:
        """Process items in the retry queue.

        Args:
            workflow: Workflow to execute
            item_results: List of all item results
            retry_queue: Queue of item indices to retry
            semaphore: Concurrency limiter
            config: Batch configuration
            progress: Progress tracker
            on_progress: Progress callback
        """
        # Process retries in batches
        batch_size = config.batch_size
        retry_indices = list(set(retry_queue))  # Deduplicate

        for i in range(0, len(retry_indices), batch_size):
            if self._cancelled:
                break

            batch = retry_indices[i : i + batch_size]

            # Apply backoff delay for exponential strategy
            if config.retry_strategy == BatchRetryStrategy.EXPONENTIAL_BACKOFF:
                max_attempts = max(item_results[idx].attempts for idx in batch)
                delay = compute_backoff_delay(max_attempts - 1, config.retry_delay_seconds)
                await asyncio.sleep(delay)

            # Clear retry queue and re-execute
            await self._execute_batch_items(
                workflow=workflow,
                item_results=item_results,
                indices=batch,
                semaphore=semaphore,
                config=config,
                progress=progress,
                on_progress=on_progress,
                retry_queue=[],  # Don't queue further retries
            )

    def _update_estimated_remaining(
        self,
        progress: BatchProgress,
        start_time: float,
    ) -> None:
        """Update estimated remaining time.

        Args:
            progress: Progress to update
            start_time: Start time of batch execution
        """
        processed = progress.completed + progress.failed
        if processed == 0:
            progress.estimated_remaining_seconds = None
            return

        elapsed = time.time() - start_time
        rate = processed / elapsed  # items per second

        if rate > 0:
            remaining_items = progress.pending
            progress.estimated_remaining_seconds = remaining_items / rate
        else:
            progress.estimated_remaining_seconds = None

    def cancel(self) -> None:
        """Cancel the batch execution gracefully."""
        logger.info("Cancelling batch execution")
        self._cancelled = True

    @property
    def is_cancelled(self) -> bool:
        """Check if execution has been cancelled."""
        return self._cancelled


__all__ = [
    "BatchConfig",
    "BatchItemResult",
    "BatchProgress",
    "BatchResult",
    "BatchWorkflowExecutor",
    "ItemStatus",
    "ProgressCallback",
    "BatchRetryStrategy",
]

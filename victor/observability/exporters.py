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

"""Event exporters implementing the Strategy pattern.

Exporters handle persisting or transmitting events to external systems.
Each exporter implements a specific strategy for event handling.

Available exporters:
- JsonLineExporter: Writes events to JSONL file
- CallbackExporter: Calls user-defined callback
- CompositeExporter: Combines multiple exporters
- FilteringExporter: Filters events before delegation

Example:
    from victor.core.events import get_observability_bus
    from victor.observability import JsonLineExporter

    bus = get_observability_bus()
    bus.add_exporter(JsonLineExporter("events.jsonl"))
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union

from victor.core.events import MessagingEvent

logger = logging.getLogger(__name__)


class BaseExporter(ABC):
    """Abstract base class for event exporters.

    Implements the Strategy pattern - each exporter defines its own
    strategy for handling events.

    Subclasses must implement:
        - export(): Synchronous event handling
        - close(): Cleanup resources

    Optional:
        - export_async(): Asynchronous event handling
    """

    @abstractmethod
    def export(self, event: MessagingEvent) -> None:
        """Export an event synchronously.

        Args:
            event: MessagingEvent to export.
        """
        pass

    async def export_async(self, event: MessagingEvent) -> None:
        """Export an event asynchronously.

        Default implementation calls sync export.

        Args:
            event: MessagingEvent to export.
        """
        self.export(event)

    @abstractmethod
    def close(self) -> None:
        """Close the exporter and release resources."""
        pass

    def __enter__(self) -> "BaseExporter":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class JsonLineExporter(BaseExporter):
    """Exports events to a JSONL (JSON Lines) file.

    Each event is written as a single line of JSON, making it
    easy to process with standard tools (jq, grep, etc.).

    Attributes:
        path: Path to output file.
        buffer_size: Number of events to buffer before flushing.
        include_categories: Optional set of categories to include.
        exclude_categories: Optional set of categories to exclude.

    Example:
        exporter = JsonLineExporter("events.jsonl")
        exporter.export(event)
        exporter.close()
    """

    def __init__(
        self,
        path: Union[str, Path],
        *,
        buffer_size: int = 10,
        flush_interval_seconds: int = 60,
        include_categories: Optional[Set[str]] = None,
        exclude_categories: Optional[Set[str]] = None,
        append: bool = True,
    ) -> None:
        """Initialize the JSONL exporter.

        Args:
            path: Path to output file.
            buffer_size: Events to buffer before flush.
            flush_interval_seconds: Flush at least this often (default: 60 seconds).
            include_categories: Topic prefixes to include (None = all).
            exclude_categories: Topic prefixes to exclude.
            append: Whether to append to existing file.
        """
        self.path = Path(path)
        self.buffer_size = buffer_size
        self.flush_interval_seconds = flush_interval_seconds
        self.include_categories = include_categories
        self.exclude_categories = exclude_categories or set()

        # Ensure parent directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Open file
        mode = "a" if append else "w"
        self._file = open(self.path, mode, encoding="utf-8")
        self._buffer: List[str] = []
        self._event_count = 0
        self._last_flush_time = time.time()

    def export(self, event: MessagingEvent) -> None:
        """Export event to JSONL file.

        Args:
            event: MessagingEvent to export.
        """
        # Filter by topic prefix (was category, now topic-based)
        if self.include_categories:
            if not any(event.topic.startswith(prefix) for prefix in self.include_categories):
                return
        if any(event.topic.startswith(prefix) for prefix in self.exclude_categories):
            return

        try:
            # Serialize to dict and handle non-JSON-serializable types
            event_dict = event.to_dict()

            # Convert sets to lists for JSON serialization
            event_dict = self._make_json_serializable(event_dict)

            # Serialize to single-line JSON with no extra whitespace
            # separators=(',', ':') removes spaces after commas and colons
            # ensure_ascii=False handles unicode characters properly
            line = json.dumps(
                event_dict,
                separators=(",", ":"),  # Compact single-line output
                ensure_ascii=False,  # Preserve unicode
                default=str,  # Fallback for unknown types
            )

            # Ensure the JSON is single-line (no embedded newlines)
            line = line.replace("\n", " ").replace("\r", " ") + "\n"

            # Buffer the line
            self._buffer.append(line)
            self._event_count += 1

            # Flush if buffer is full OR time interval elapsed
            time_since_flush = time.time() - self._last_flush_time
            if (
                len(self._buffer) >= self.buffer_size
                or time_since_flush >= self.flush_interval_seconds
            ):
                self.flush()

        except (TypeError, ValueError) as e:
            # Log serialization error but don't crash
            logger.warning(
                f"Failed to serialize event {event.topic}: {e}. "
                f"Event data keys: {list(event.data.keys()) if event.data else 'N/A'}"
            )

    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert non-JSON-serializable types to JSON-compatible types.

        Args:
            obj: Object to convert.

        Returns:
            JSON-serializable version of the object.
        """
        if isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            # For other types, convert to string
            return str(obj)

    def flush(self) -> None:
        """Flush buffered events to file."""
        if self._buffer:
            self._file.writelines(self._buffer)
            self._file.flush()
            self._buffer.clear()
            self._last_flush_time = time.time()

    def close(self) -> None:
        """Close the file, flushing any remaining events."""
        self.flush()
        self._file.close()

    @property
    def event_count(self) -> int:
        """Get total number of events exported."""
        return self._event_count


class LoggingExporter(BaseExporter):
    """Exports events to Python's logging system.

    This allows observability events to be written to log files alongside
    regular log messages, using the existing logging infrastructure.

    Events are logged at INFO level by default, with ERROR events at ERROR level.

    Example:
        from victor.core.events import get_observability_bus
        from victor.observability import LoggingExporter

        bus = get_observability_bus()
        bus.add_exporter(LoggingExporter("victor.events"))

        # Events will now appear in logs:
        # 2025-01-01 12:00:00 - victor.events - INFO - [TOOL] tool.read.start
    """

    def __init__(
        self,
        logger_name: str = "victor.events",
        *,
        include_categories: Optional[Set[str]] = None,
        exclude_categories: Optional[Set[str]] = None,
        log_level: int = logging.INFO,
        include_data: bool = True,
    ) -> None:
        """Initialize the logging exporter.

        Args:
            logger_name: Name of the logger to use.
            include_categories: Topic prefixes to include (None = all).
            exclude_categories: Topic prefixes to exclude.
            log_level: Default log level for events.
            include_data: Whether to include event data in log message.
        """
        self._logger = logging.getLogger(logger_name)
        self.include_categories = include_categories
        self.exclude_categories = exclude_categories or set()
        self.log_level = log_level
        self.include_data = include_data
        self._event_count = 0

    def export(self, event: MessagingEvent) -> None:
        """Export event to logging system.

        Args:
            event: MessagingEvent to export.
        """
        # Filter by topic prefix
        if self.include_categories:
            if not any(event.topic.startswith(prefix) for prefix in self.include_categories):
                return
        if any(event.topic.startswith(prefix) for prefix in self.exclude_categories):
            return

        # Determine log level
        level = self.log_level
        if event.topic.startswith("error."):
            level = logging.ERROR
        elif event.topic.startswith("audit."):
            level = logging.WARNING

        # Format message
        # Extract category from topic prefix (e.g., "tool.start" -> "TOOL")
        topic_parts = event.topic.split(".", 1)
        category_name = topic_parts[0].upper() if topic_parts else "UNKNOWN"
        message = f"[{category_name}] {event.topic}"

        if self.include_data and event.data:
            # Include key data fields (avoid very long output)
            data_preview = {}
            for k, v in event.data.items():
                if isinstance(v, str) and len(v) > 100:
                    data_preview[k] = v[:100] + "..."
                elif isinstance(v, (list, set)) and len(v) > 5:
                    data_preview[k] = f"[{len(v)} items]"
                else:
                    data_preview[k] = v
            message += f": {data_preview}"

        self._logger.log(level, message)
        self._event_count += 1

    def close(self) -> None:
        """Close the exporter (no-op for logging)."""
        pass

    @property
    def event_count(self) -> int:
        """Get total number of events exported."""
        return self._event_count


class CallbackExporter(BaseExporter):
    """Exports events by calling a user-defined callback.

    Useful for integrating with existing logging systems,
    metrics collectors, or custom handlers.

    Example:
        def my_handler(event):
            print(f"Event: {event.topic}")

        exporter = CallbackExporter(my_handler)
    """

    def __init__(
        self,
        callback: Callable[[MessagingEvent], None],
        *,
        async_callback: Optional[Callable[[MessagingEvent], Any]] = None,
        error_handler: Optional[Callable[[Exception, MessagingEvent], None]] = None,
    ) -> None:
        """Initialize callback exporter.

        Args:
            callback: Sync callback function.
            async_callback: Optional async callback.
            error_handler: Optional error handler.
        """
        self._callback = callback
        self._async_callback = async_callback
        self._error_handler = error_handler

    def export(self, event: MessagingEvent) -> None:
        """Export event via callback.

        Args:
            event: MessagingEvent to export.
        """
        try:
            self._callback(event)
        except Exception as e:
            if self._error_handler:
                self._error_handler(e, event)
            else:
                logger.warning(f"Callback error: {e}")

    async def export_async(self, event: MessagingEvent) -> None:
        """Export event via async callback.

        Args:
            event: MessagingEvent to export.
        """
        if self._async_callback:
            try:
                await self._async_callback(event)
            except Exception as e:
                if self._error_handler:
                    self._error_handler(e, event)
                else:
                    logger.warning(f"Async callback error: {e}")
        else:
            self.export(event)

    def close(self) -> None:
        """No-op for callback exporter."""
        pass


class CompositeExporter(BaseExporter):
    """Combines multiple exporters into one.

    Implements the Composite pattern - events are sent to all
    child exporters.

    Example:
        composite = CompositeExporter([
            JsonLineExporter("events.jsonl"),
            CallbackExporter(my_handler),
        ])
    """

    def __init__(self, exporters: List[BaseExporter]) -> None:
        """Initialize composite exporter.

        Args:
            exporters: List of exporters to delegate to.
        """
        self._exporters = list(exporters)

    def add(self, exporter: BaseExporter) -> None:
        """Add an exporter.

        Args:
            exporter: Exporter to add.
        """
        self._exporters.append(exporter)

    def remove(self, exporter: BaseExporter) -> None:
        """Remove an exporter.

        Args:
            exporter: Exporter to remove.
        """
        if exporter in self._exporters:
            self._exporters.remove(exporter)

    def export(self, event: MessagingEvent) -> None:
        """Export event to all child exporters.

        Args:
            event: MessagingEvent to export.
        """
        for exporter in self._exporters:
            try:
                exporter.export(event)
            except Exception as e:
                logger.warning(f"Exporter {type(exporter).__name__} error: {e}")

    async def export_async(self, event: MessagingEvent) -> None:
        """Export event to all child exporters asynchronously.

        Args:
            event: MessagingEvent to export.
        """
        tasks = []
        for exporter in self._exporters:
            if hasattr(exporter, "export_async"):
                tasks.append(exporter.export_async(event))
            else:
                exporter.export(event)

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"Async exporter error: {result}")

    def close(self) -> None:
        """Close all child exporters."""
        for exporter in self._exporters:
            try:
                exporter.close()
            except Exception as e:
                logger.warning(f"Exporter close error: {e}")


class FilteringExporter(BaseExporter):
    """Wraps another exporter with filtering logic.

    Implements the Decorator pattern - adds filtering capability
    to any existing exporter.

    Example:
        # Only export tool events
        filtered = FilteringExporter(
            JsonLineExporter("tools.jsonl"),
            categories={"tool."},  # Topic prefix
        )
    """

    def __init__(
        self,
        exporter: BaseExporter,
        *,
        categories: Optional[Set[str]] = None,
        names: Optional[Set[str]] = None,
        predicate: Optional[Callable[[MessagingEvent], bool]] = None,
    ) -> None:
        """Initialize filtering exporter.

        Args:
            exporter: Underlying exporter.
            categories: Set of topic prefixes to include.
            names: Set of event topics to include.
            predicate: Custom filter function.
        """
        self._exporter = exporter
        self._categories = categories
        self._names = names
        self._predicate = predicate

    def _should_export(self, event: MessagingEvent) -> bool:
        """Check if event should be exported.

        Args:
            event: MessagingEvent to check.

        Returns:
            True if event should be exported.
        """
        if self._categories:
            if not any(event.topic.startswith(prefix) for prefix in self._categories):
                return False
        if self._names and event.topic not in self._names:
            return False
        if self._predicate and not self._predicate(event):
            return False
        return True

    def export(self, event: MessagingEvent) -> None:
        """Export event if it passes filters.

        Args:
            event: MessagingEvent to export.
        """
        if self._should_export(event):
            self._exporter.export(event)

    async def export_async(self, event: MessagingEvent) -> None:
        """Export event asynchronously if it passes filters.

        Args:
            event: MessagingEvent to export.
        """
        if self._should_export(event):
            await self._exporter.export_async(event)

    def close(self) -> None:
        """Close underlying exporter."""
        self._exporter.close()


class BufferedExporter(BaseExporter):
    """Buffers events and exports in batches.

    Useful for reducing I/O overhead when exporting many events.

    Example:
        buffered = BufferedExporter(
            JsonLineExporter("events.jsonl"),
            batch_size=100,
            flush_interval=5.0,
        )
    """

    def __init__(
        self,
        exporter: BaseExporter,
        *,
        batch_size: int = 100,
        flush_interval: float = 5.0,
    ) -> None:
        """Initialize buffered exporter.

        Args:
            exporter: Underlying exporter.
            batch_size: Events to buffer before flush.
            flush_interval: Seconds between auto-flushes.
        """
        self._exporter = exporter
        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self._buffer: List[MessagingEvent] = []
        self._last_flush = datetime.now(timezone.utc)

    def export(self, event: MessagingEvent) -> None:
        """Buffer event for later export.

        Args:
            event: MessagingEvent to buffer.
        """
        self._buffer.append(event)

        # Check if we should flush
        should_flush = (
            len(self._buffer) >= self._batch_size
            or (datetime.now(timezone.utc) - self._last_flush).total_seconds()
            >= self._flush_interval
        )

        if should_flush:
            self.flush()

    def flush(self) -> None:
        """Flush all buffered events."""
        for event in self._buffer:
            self._exporter.export(event)
        self._buffer.clear()
        self._last_flush = datetime.now(timezone.utc)

    def close(self) -> None:
        """Flush remaining events and close."""
        self.flush()
        self._exporter.close()

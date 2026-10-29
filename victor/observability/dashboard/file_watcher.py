"""File watcher for real-time event streaming from Victor JSONL event files.

This component monitors the Victor event file (~/.victor/metrics/victor.jsonl)
and emits events to the dashboard's EventBus as new events are written.

This enables cross-process event streaming when the dashboard and
agent run in separate processes.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from textual import work
from textual.widgets import Static

from victor.core.events import MessagingEvent, get_observability_bus

logger = logging.getLogger(__name__)


class EventFileWatcher(Static):
    """Watches Victor JSONL event file for new events and emits them to EventBus.

    This enables cross-process event streaming by tailing the event file
    that is written by another process (e.g., the Victor agent).

    The event format is JSONL (one JSON object per line):
        {"timestamp": "2025-01-06T12:00:00.000Z", "category": "TOOL", "name": "tool.start", ...}

    Example:
        watcher = EventFileWatcher()
        await watcher.watch("/path/to/victor.jsonl")
    """

    def __init__(self, file_path: Optional[Path] = None, **kwargs):
        """Initialize the file watcher.

        Args:
            file_path: Path to Victor log file to watch.
            **kwargs: Additional arguments for Static widget.
        """
        super().__init__(**kwargs)
        self._file_path = file_path
        self._watching = False
        self._last_position = 0
        self._loaded_initial = False  # Track if initial load is done
        self._max_initial_load = 100  # Only load last 100 events at startup

    def on_mount(self) -> None:
        """Start watching when widget is mounted."""
        logger.info(f"[EventFileWatcher] Mounting with file_path: {self._file_path}")

        if self._file_path:
            # Load existing events if file exists
            if self._file_path.exists():
                logger.info("[EventFileWatcher] File exists, loading historical events")
                self._load_existing_events()
            else:
                logger.info("[EventFileWatcher] File does not exist yet, will wait for creation")

            # Always start watching (file may be created later)
            logger.info("[EventFileWatcher] Starting file watcher")
            self._start_watching()

    def on_unmount(self) -> None:
        """Stop watching when widget is unmounted."""
        self._stop_watching()

    def _load_existing_events(self) -> None:
        """Load the last 100 events from the JSONL file at startup.

        This loads only the most recent events for faster startup.
        After initial load, the file watcher continues monitoring for new events.
        """
        logger.debug("[EventFileWatcher] _load_existing_events() START")

        if not self._file_path or not self._file_path.exists():
            logger.debug(f"[EventFileWatcher] No file or file doesn't exist: {self._file_path}")
            return

        event_bus = get_observability_bus()
        events_loaded = 0
        events_parsed = 0
        events_failed = 0

        try:
            # Read all lines into memory, then take last N
            with open(self._file_path, "r", encoding="utf-8") as f:
                all_lines = f.readlines()

            logger.debug(f"[EventFileWatcher] Read {len(all_lines)} lines from file")

            # Take only the last max_initial_load lines
            lines_to_process = (
                all_lines[-self._max_initial_load :]
                if len(all_lines) > self._max_initial_load
                else all_lines
            )

            logger.debug(
                f"[EventFileWatcher] Processing last {len(lines_to_process)} lines (max: {self._max_initial_load})"
            )

            # Process each line IN REVERSE ORDER (newest first) for correct display
            # Reverse the lines so newest events are emitted first
            reversed_lines = list(reversed(lines_to_process))
            for idx, line in enumerate(reversed_lines):
                line = line.strip()
                if not line:
                    logger.debug(f"[EventFileWatcher] Line {idx}: Empty, skipping")
                    continue

                logger.debug(f"[EventFileWatcher] Line {idx}: Processing (length: {len(line)})")

                try:
                    # Parse JSONL line and create VictorEvent
                    event = self._parse_jsonl_line(line)

                    if event:
                        events_parsed += 1
                        logger.debug(
                            f"[EventFileWatcher] Line {idx}: PARSED [{event.topic.split('.')[0]}/{event.topic}]"
                        )

                        # Emit to EventBus
                        event_bus.publish(event)
                        events_loaded += 1
                        logger.debug(
                            f"[EventFileWatcher] Line {idx}: PUBLISHED to EventBus (total: {events_loaded})"
                        )
                    else:
                        events_failed += 1
                        logger.warning(
                            f"[EventFileWatcher] Line {idx}: Failed to parse event (returned None)"
                        )

                except Exception as e:
                    # Skip invalid lines
                    events_failed += 1
                    logger.warning(f"[EventFileWatcher] Line {idx}: Exception during parse: {e}")
                    logger.debug(f"[EventFileWatcher] Line {idx} content: {line[:200]}")
                    continue

            # Update position to end of file
            self._last_position = self._file_path.stat().st_size
            self._loaded_initial = True

            logger.info(
                f"[EventFileWatcher] Loaded {events_loaded} events (parsed: {events_parsed}, failed: {events_failed}, total_lines: {len(all_lines)})"
            )

        except Exception as e:
            logger.error(f"[EventFileWatcher] Error loading events: {e}")
            import traceback

            logger.error(f"[EventFileWatcher] Traceback: {traceback.format_exc()}")
            # If loading fails, just start watching from current position
            self._last_position = self._file_path.stat().st_size

    def _start_watching(self) -> None:
        """Start the file watching task.

        The @work decorator on _watch_file() automatically manages the worker.
        We just need to set the flag.
        """
        if not self._watching and self._file_path:
            self._watching = True

    def _stop_watching(self) -> None:
        """Stop the file watching task.

        The @work decorator will handle cleanup when the widget is unmounted.
        """
        self._watching = False

    @work(exclusive=True)
    async def _watch_file(self) -> None:
        """Watch the file for new lines and emit events.

        This worker runs in the background, polling the file for changes.
        When new lines are detected, they're parsed and emitted to the EventBus.
        """
        event_bus = get_observability_bus()
        logger.info(f"[EventFileWatcher] File watcher started, monitoring: {self._file_path}")

        while self._watching and self._file_path:
            try:
                if not self._file_path.exists():
                    await asyncio.sleep(0.5)
                    continue

                current_size = self._file_path.stat().st_size

                # Check if file has new content
                if current_size > self._last_position:
                    logger.debug(
                        f"[EventFileWatcher] File grew from {self._last_position} to {current_size} bytes"
                    )

                    # Read new content with explicit cleanup
                    new_lines = []
                    file_handle = None
                    try:
                        file_handle = open(self._file_path, "r", encoding="utf-8")
                        file_handle.seek(self._last_position)
                        new_lines = file_handle.readlines()
                        self._last_position = file_handle.tell()
                    except Exception as e:
                        logger.error(f"[EventFileWatcher] Error reading file: {e}")
                    finally:
                        # Ensure file handle is always closed
                        if file_handle is not None:
                            try:
                                file_handle.close()
                            except Exception:
                                pass

                    logger.debug(f"[EventFileWatcher] Read {len(new_lines)} new lines")

                    # Process each new line IN REVERSE ORDER (newest first) for correct display
                    events_loaded = 0
                    reversed_new_lines = list(reversed(new_lines))
                    for line in reversed_new_lines:
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            # Parse JSONL line and create VictorEvent
                            event = self._parse_jsonl_line(line)

                            if event:
                                # Emit to EventBus
                                event_bus.publish(event)
                                events_loaded += 1

                        except Exception as e:
                            # Skip invalid lines
                            logger.debug(f"[EventFileWatcher] Failed to parse line: {e}")
                            continue

                    if events_loaded > 0:
                        logger.info(
                            f"[EventFileWatcher] Loaded {events_loaded} new events from file"
                        )

                # Small delay to avoid busy-waiting
                await asyncio.sleep(0.1)

            except Exception as e:
                # Log error but continue watching
                logger.error(f"[EventFileWatcher] Error watching file: {e}")
                await asyncio.sleep(1)

    def _parse_jsonl_line(self, line: str) -> Optional[MessagingEvent]:
        """Parse a JSONL event line into an Event.

        Args:
            line: JSONL line to parse (JSON object as string).

        Returns: Event or None if parsing fails.

        Example JSONL line:
            {"id": "...", "timestamp": "2025-01-06T12:00:00Z", "topic": "tool.start", ...}
        """
        try:
            logger.debug(f"[EventFileWatcher._parse_jsonl_line] Parsing line (len={len(line)})")

            # Parse JSON
            data = json.loads(line)
            logger.debug(
                f"[EventFileWatcher._parse_jsonl_line] JSON parsed: keys={list(data.keys())}"
            )

            # Extract timestamp
            timestamp_str = data.get("timestamp")
            if timestamp_str:
                # Handle ISO format with or without microseconds
                if "T" in timestamp_str:
                    # ISO format: 2025-01-06T12:00:00.000Z
                    timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                else:
                    # Fallback for other formats
                    timestamp = datetime.fromisoformat(timestamp_str)
                logger.debug(f"[EventFileWatcher._parse_jsonl_line] Timestamp: {timestamp}")
            else:
                timestamp = datetime.now()
                logger.debug("[EventFileWatcher._parse_jsonl_line] No timestamp, using now")

            # Extract or construct topic
            topic = data.get("topic")
            if not topic:
                # Backward compatibility: construct from old category + name
                category_str = data.get("category", "custom")
                name = data.get("name", "unknown")
                topic = f"{category_str.lower()}.{name.lower()}"
            logger.debug(f"[EventFileWatcher._parse_jsonl_line] Topic: '{topic}'")

            # Create MessagingEvent from dict
            event = MessagingEvent(
                id=data.get("id"),
                timestamp=timestamp,
                topic=topic,
                data=data.get("data", {}),
                source=data.get("source"),
                correlation_id=data.get("trace_id"),
                headers=(
                    {"session_id": data.get("session_id")} if data.get("session_id") else None
                ),
            )

            logger.debug(
                f"[EventFileWatcher._parse_jsonl_line] Created event: {event.topic.split('.')[0]}/{event.topic}"
            )
            return event

        except Exception as e:
            logger.error(f"[EventFileWatcher._parse_jsonl_line] Exception: {e}")
            logger.debug(f"[EventFileWatcher._parse_jsonl_line] Line content: {line[:300]}")
            import traceback

            logger.debug(
                f"[EventFileWatcher._parse_jsonl_line] Traceback: {traceback.format_exc()}"
            )
            return None

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

"""Export Service for observability data.

Provides streaming export of events in multiple formats (JSON, CSV, JSONL).
Reuses exporter strategy pattern from victor/observability/exporters.py.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import AsyncIterator, Optional, List
import csv
import io
import json
import logging

from victor.core.events import MessagingEvent
from victor.observability.query_service import QueryService

logger = logging.getLogger(__name__)


@dataclass
class ExportServiceConfig:
    """Configuration for ExportService."""

    max_events_per_export: int = 50000
    chunk_size: int = 1000


class ExportService:
    """Service for exporting observability data in multiple formats."""

    def __init__(self, config: Optional[ExportServiceConfig] = None):
        """Initialize ExportService.

        Args:
            config: Service configuration
        """
        self.config = config or ExportServiceConfig()
        self._query_service = QueryService()

    async def export_events(
        self,
        format: str = "json",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        topic_pattern: Optional[str] = None,
    ) -> AsyncIterator[bytes]:
        """Export events in specified format.

        Args:
            format: Export format ("json", "csv", "jsonl")
            start_time: Start time filter
            end_time: End time filter
            topic_pattern: Topic pattern filter (e.g., "tool.*")

        Yields:
            Chunks of exported data as bytes

        Raises:
            ValueError: If format is not supported
        """
        if format not in ("json", "csv", "jsonl"):
            raise ValueError(f"Unsupported format: {format}")

        # Query events
        events = await self._query_service.get_recent_events(
            start_time=start_time,
            end_time=end_time,
            topic_pattern=topic_pattern,
            limit=self.config.max_events_per_export,
        )

        # Export in format
        if format == "json":
            async for chunk in self._export_json(events):
                yield chunk
        elif format == "csv":
            async for chunk in self._export_csv(events):
                yield chunk
        elif format == "jsonl":
            async for chunk in self._export_jsonl(events):
                yield chunk

    async def _export_json(self, events: List[MessagingEvent]) -> AsyncIterator[bytes]:
        """Export events as JSON array.

        Args:
            events: List of events to export

        Yields:
            JSON chunks as bytes
        """
        output = io.StringIO()

        # Start JSON array
        output.write("[\n")

        for i, event in enumerate(events):
            # Convert event to dict
            event_dict = event.to_dict()

            # Write event
            json.dump(event_dict, output, indent=2)

            # Add comma if not last
            if i < len(events) - 1:
                output.write(",\n")
            else:
                output.write("\n")

            # Yield chunk if buffer is large
            if output.tell() > 10240:  # 10KB chunks
                yield output.getvalue().encode("utf-8")
                output.seek(0)
                output.truncate(0)

        # End JSON array
        output.write("]\n")

        # Yield remaining data
        if output.tell() > 0:
            yield output.getvalue().encode("utf-8")

    async def _export_csv(self, events: List[MessagingEvent]) -> AsyncIterator[bytes]:
        """Export events as CSV.

        Args:
            events: List of events to export

        Yields:
            CSV chunks as bytes
        """
        output = io.StringIO()

        # Create CSV writer
        writer = csv.writer(output)

        # Write header
        writer.writerow(
            [
                "timestamp",
                "topic",
                "category",
                "source",
                "correlation_id",
                "data",
            ]
        )

        yield output.getvalue().encode("utf-8")
        output.seek(0)
        output.truncate(0)

        # Write events
        for event in events:
            writer.writerow(
                [
                    event.datetime.isoformat(),
                    event.topic,
                    event.category,
                    event.source,
                    event.correlation_id or "",
                    json.dumps(event.data) if event.data else "",
                ]
            )

            # Yield chunk if buffer is large
            if output.tell() > 10240:  # 10KB chunks
                yield output.getvalue().encode("utf-8")
                output.seek(0)
                output.truncate(0)

        # Yield remaining data
        if output.tell() > 0:
            yield output.getvalue().encode("utf-8")

    async def _export_jsonl(self, events: List[MessagingEvent]) -> AsyncIterator[bytes]:
        """Export events as JSONL (one JSON object per line).

        Args:
            events: List of events to export

        Yields:
            JSONL chunks as bytes
        """
        output = io.StringIO()

        for event in events:
            # Convert event to dict
            event_dict = event.to_dict()

            # Write event as JSON line
            json.dump(event_dict, output)
            output.write("\n")

            # Yield chunk if buffer is large
            if output.tell() > 10240:  # 10KB chunks
                yield output.getvalue().encode("utf-8")
                output.seek(0)
                output.truncate(0)

        # Yield remaining data
        if output.tell() > 0:
            yield output.getvalue().encode("utf-8")

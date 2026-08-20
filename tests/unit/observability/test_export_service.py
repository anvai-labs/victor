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

"""Unit tests for ExportService."""

import pytest
import json
import csv
from io import StringIO
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from victor.observability.export_service import ExportService, ExportServiceConfig
from victor.core.events import MessagingEvent


@pytest.fixture
def export_service():
    """Create ExportService instance for testing."""
    return ExportService()


@pytest.fixture
def mock_events():
    """Create mock events for testing."""
    events = []
    base_time = datetime.now()

    for i in range(10):
        event = MagicMock(spec=MessagingEvent)
        event.topic = f"topic_{i % 3}"
        event.category = "test"
        event.datetime = base_time
        event.source = "test_source"
        event.correlation_id = f"corr_{i}"
        event.data = {"index": i, "value": f"test_{i}"}

        # Mock to_dict method
        event.to_dict.return_value = {
            "topic": event.topic,
            "category": event.category,
            "datetime": event.datetime.isoformat(),
            "source": event.source,
            "correlation_id": event.correlation_id,
            "data": event.data,
        }

        events.append(event)

    return events


class TestExportServiceConfig:
    """Tests for ExportServiceConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ExportServiceConfig()
        assert config.max_events_per_export == 50000
        assert config.chunk_size == 1000

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ExportServiceConfig(
            max_events_per_export=10000,
            chunk_size=500,
        )
        assert config.max_events_per_export == 10000
        assert config.chunk_size == 500


class TestExportService:
    """Tests for ExportService."""

    def test_initialization(self, export_service):
        """Test service initialization."""
        assert export_service.config is not None
        assert export_service._query_service is not None

    @pytest.mark.asyncio
    async def test_export_json(self, export_service, mock_events):
        """Test JSON export format."""
        with patch.object(
            export_service._query_service,
            "get_recent_events",
            return_value=mock_events,
        ):
            chunks = []
            async for chunk in export_service.export_events(format="json"):
                chunks.append(chunk)

            # Combine chunks
            data = b"".join(chunks).decode("utf-8")

            # Verify JSON structure
            exported_data = json.loads(data)
            assert isinstance(exported_data, list)
            assert len(exported_data) == 10

            # Verify event structure
            for event in exported_data:
                assert "topic" in event
                assert "category" in event
                assert "data" in event

    @pytest.mark.asyncio
    async def test_export_csv(self, export_service, mock_events):
        """Test CSV export format."""
        with patch.object(
            export_service._query_service,
            "get_recent_events",
            return_value=mock_events,
        ):
            chunks = []
            async for chunk in export_service.export_events(format="csv"):
                chunks.append(chunk)

            # Combine chunks
            data = b"".join(chunks).decode("utf-8")

            # Verify CSV structure
            reader = csv.reader(StringIO(data))
            rows = list(reader)

            # Check header
            assert rows[0] == [
                "timestamp",
                "topic",
                "category",
                "source",
                "correlation_id",
                "data",
            ]

            # Check data rows (should have 10 events + 1 header)
            assert len(rows) == 11

            # Verify data row structure
            for row in rows[1:]:
                assert len(row) == 6

    @pytest.mark.asyncio
    async def test_export_jsonl(self, export_service, mock_events):
        """Test JSONL export format."""
        with patch.object(
            export_service._query_service,
            "get_recent_events",
            return_value=mock_events,
        ):
            chunks = []
            async for chunk in export_service.export_events(format="jsonl"):
                chunks.append(chunk)

            # Combine chunks
            data = b"".join(chunks).decode("utf-8")

            # Verify JSONL structure (one JSON object per line)
            lines = data.strip().split("\n")
            assert len(lines) == 10

            # Verify each line is valid JSON
            for line in lines:
                event = json.loads(line)
                assert "topic" in event
                assert "category" in event
                assert "data" in event

    @pytest.mark.asyncio
    async def test_export_invalid_format(self, export_service, mock_events):
        """Test that invalid format raises ValueError."""
        with patch.object(
            export_service._query_service,
            "get_recent_events",
            return_value=mock_events,
        ):
            with pytest.raises(ValueError, match="Unsupported format"):
                async for _ in export_service.export_events(format="invalid"):
                    pass

    @pytest.mark.asyncio
    async def test_export_with_time_filters(self, export_service, mock_events):
        """Test export with time range filters."""
        start_time = datetime.now() - timedelta(hours=1)
        end_time = datetime.now()

        with patch.object(
            export_service._query_service,
            "get_recent_events",
            return_value=mock_events,
        ) as mock_get_events:
            async for _ in export_service.export_events(
                format="jsonl",
                start_time=start_time,
                end_time=end_time,
            ):
                pass

            # Verify filters were passed to query service
            mock_get_events.assert_called_once()
            call_kwargs = mock_get_events.call_args.kwargs
            assert "start_time" in call_kwargs
            assert "end_time" in call_kwargs

    @pytest.mark.asyncio
    async def test_export_with_topic_filter(self, export_service, mock_events):
        """Test export with topic pattern filter."""
        with patch.object(
            export_service._query_service,
            "get_recent_events",
            return_value=mock_events,
        ) as mock_get_events:
            async for _ in export_service.export_events(
                format="jsonl",
                topic_pattern="tool.*",
            ):
                pass

            # Verify filter was passed to query service
            mock_get_events.assert_called_once()
            call_kwargs = mock_get_events.call_args.kwargs
            assert "topic_pattern" in call_kwargs
            assert call_kwargs["topic_pattern"] == "tool.*"

    @pytest.mark.asyncio
    async def test_export_streaming_large_dataset(self, export_service):
        """Test that export streams data in chunks for large datasets."""
        # Create many events to trigger chunking
        large_events = []
        for i in range(1000):
            event = MagicMock(spec=MessagingEvent)
            event.topic = "test"
            event.category = "test"
            event.datetime = datetime.now()
            event.source = "test"
            event.correlation_id = f"corr_{i}"
            event.data = {"index": i}
            event.to_dict.return_value = {
                "topic": event.topic,
                "category": event.category,
                "datetime": event.datetime.isoformat(),
                "source": event.source,
                "correlation_id": event.correlation_id,
                "data": event.data,
            }
            large_events.append(event)

        with patch.object(
            export_service._query_service,
            "get_recent_events",
            return_value=large_events,
        ):
            chunk_count = 0
            async for chunk in export_service.export_events(format="jsonl"):
                chunk_count += 1
                # Verify chunks are reasonable size (allow slight overflow for safety)
                assert len(chunk) <= 11264  # 10KB + 10% tolerance

            # Verify multiple chunks were produced
            assert chunk_count > 1

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
# specific language governing permissions and
# limitations under the License.

"""Observability API routes for web UI.

Provides REST endpoints for querying events, sessions, and metrics.
Includes web-based dashboard UI with real-time updates.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from victor.observability.query_service import (
    QueryService,
    EventFilters,
    Event,
    SessionInfo,
    MetricsSnapshot,
)
from victor.observability.aggregation_service import AggregationService
from victor.observability.export_service import ExportService


# Request/Response Models
class EventsResponse(BaseModel):
    """Response model for events endpoint."""

    events: List[dict] = Field(description="List of events")
    total: int = Field(description="Total count (for pagination)")
    limit: int = Field(description="Page size")
    offset: int = Field(description="Current offset")


class SessionsResponse(BaseModel):
    """Response model for sessions endpoint."""

    sessions: List[dict] = Field(description="List of sessions")
    total: int = Field(description="Total count")
    limit: int = Field(description="Page size")
    offset: int = Field(description="Current offset")


class MetricsResponse(BaseModel):
    """Response model for metrics endpoint."""

    metrics: dict = Field(description="Current metrics snapshot")
    timestamp: str = Field(description="Metrics timestamp")


# Create router
router = APIRouter(
    prefix="/obs",
    tags=["observability"],
)

# Initialize query service (will be injected by FastAPI dependency injection in production)
_query_service: Optional[QueryService] = None
_aggregation_service: Optional[AggregationService] = None
_export_service: Optional[ExportService] = None


def get_query_service() -> QueryService:
    """Get or create query service instance.

    Returns:
        QueryService instance
    """
    global _query_service
    if _query_service is None:
        _query_service = QueryService()
    return _query_service


def get_aggregation_service() -> AggregationService:
    """Get or create aggregation service instance.

    Returns:
        AggregationService instance
    """
    global _aggregation_service
    if _aggregation_service is None:
        _aggregation_service = AggregationService()
    return _aggregation_service


def get_export_service() -> ExportService:
    """Get or create export service instance.

    Returns:
        ExportService instance
    """
    global _export_service
    if _export_service is None:
        _export_service = ExportService()
    return _export_service


# =============================================================================
# Event Endpoints
# =============================================================================


@router.get("/events/recent", response_model=EventsResponse)
async def get_recent_events(
    limit: int = Query(100, ge=1, le=1000, description="Maximum events to return"),
    offset: int = Query(0, ge=0, description="Number of events to skip"),
    event_types: Optional[str] = Query(None, description="Comma-separated event types to filter"),
    session_id: Optional[str] = Query(None, description="Filter by session ID"),
    tool_name: Optional[str] = Query(None, description="Filter by tool name"),
    severity: Optional[str] = Query(None, description="Filter by severity (error, warning, info)"),
) -> EventsResponse:
    """Get recent events with pagination.

    Returns the most recent events from the observability system,
    with optional filtering by event type, session, tool, or severity.
    """
    service = get_query_service()

    # Build filters
    filters = None
    if event_types or session_id or tool_name or severity:
        filters = EventFilters(
            event_types=event_types.split(",") if event_types else None,
            session_ids=[session_id] if session_id else None,
            tool_names=[tool_name] if tool_name else None,
            severity=severity,
        )

    # Query events
    events = await service.get_recent_events(
        limit=limit,
        offset=offset,
        filters=filters,
    )

    return EventsResponse(
        events=[e.to_dict() for e in events],
        total=len(events),  # Will be enhanced with true count in AggregationService
        limit=limit,
        offset=offset,
    )


# =============================================================================
# Session Endpoints
# =============================================================================


@router.get("/sessions", response_model=SessionsResponse)
async def get_sessions(
    limit: int = Query(20, ge=1, le=100, description="Maximum sessions to return"),
    offset: int = Query(0, ge=0, description="Number of sessions to skip"),
) -> SessionsResponse:
    """Get list of sessions.

    Returns all sessions with basic metadata, sorted by
    most recently updated.
    """
    service = get_query_service()

    sessions = await service.get_sessions(limit=limit, offset=offset)

    return SessionsResponse(
        sessions=[s.to_dict() for s in sessions],
        total=len(sessions),  # Will be enhanced with true count
        limit=limit,
        offset=offset,
    )


@router.get("/sessions/{session_id}")
async def get_session_details(
    session_id: str,
) -> dict:
    """Get detailed session information.

    Args:
        session_id: Session ID

    Returns:
        Session details with messages, metrics, etc.
    """
    service = get_query_service()

    # Get session info
    session_info = await service.get_session(session_id)

    if session_info is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    # Get events for this session
    events = await service.get_recent_events(
        limit=1000,
        filters=EventFilters(session_ids=[session_id]),
    )

    # Calculate session metrics from events
    tool_calls = sum(1 for e in events if e.event_type and "tool" in e.event_type.lower())
    errors = sum(1 for e in events if e.severity == "error")

    # Calculate duration
    if events:
        first_event = min(events, key=lambda e: e.timestamp)
        last_event = max(events, key=lambda e: e.timestamp)
        duration_seconds = (last_event.timestamp - first_event.timestamp).total_seconds()
    else:
        duration_seconds = 0

    return {
        "session": session_info.to_dict(),
        "metrics": {
            "tool_calls": tool_calls,
            "errors": errors,
            "duration_seconds": duration_seconds,
            "total_events": len(events),
        },
        "events": [e.to_dict() for e in events[:100]],  # First 100 events
    }


# =============================================================================
# Metrics Endpoints
# =============================================================================


@router.get("/metrics/summary", response_model=MetricsResponse)
async def get_metrics_summary() -> MetricsResponse:
    """Get current metrics snapshot.

    Returns real-time metrics including tool calls, errors,
    token usage, and active sessions.
    """
    service = get_query_service()

    metrics = await service.get_metrics_summary()

    return MetricsResponse(
        metrics=metrics.to_dict(),
        timestamp=datetime.now().isoformat(),
    )


@router.get("/metrics/history")
async def get_metrics_history(
    time_window: str = Query("1h", description="Time window (e.g., 1h, 24h, 7d)"),
) -> dict:
    """Get historical metrics data.

    Args:
        time_window: Time window for metrics

    Returns:
        Time-series metrics data
    """
    service = get_aggregation_service()
    return await service.get_metrics_history(time_window=time_window)


@router.get("/metrics/tools")
async def get_tool_metrics() -> dict:
    """Get tool usage statistics.

    Returns tool execution counts, success rates, and
    performance metrics.
    """
    service = get_aggregation_service()
    return await service.get_tool_statistics()


@router.get("/metrics/performance")
async def get_performance_metrics() -> dict:
    """Get performance metrics.

    Returns latency percentiles (p50, p95, p99), throughput,
    and resource utilization.
    """
    service = get_aggregation_service()
    return await service.get_performance_metrics()


# =============================================================================
# Trace Endpoints
# =============================================================================


@router.get("/traces")
async def list_traces(
    limit: int = Query(100, ge=1, le=1000, description="Maximum traces to return"),
    offset: int = Query(0, ge=0, description="Number of traces to skip"),
) -> dict:
    """List execution traces.

    A trace is a collection of related events identified by correlation_id.
    Each trace represents a complete execution flow.

    Returns:
        List of traces with metadata
    """
    service = get_query_service()

    # Get recent events
    all_events = await service.get_recent_events(limit=10000, offset=0)

    # Group events by correlation_id to form traces
    traces_dict: Dict[str, List[Event]] = {}

    for event in all_events:
        trace_id = event.session_id  # Use session_id as trace identifier
        if trace_id not in traces_dict:
            traces_dict[trace_id] = []
        traces_dict[trace_id].append(event)

    # Build trace metadata
    traces = []
    for trace_id, events in sorted(
        traces_dict.items(),
        key=lambda x: max(x[1], key=lambda e: e.timestamp),
        reverse=True,
    ):
        first_event = min(events, key=lambda e: e.timestamp)
        last_event = max(events, key=lambda e: e.timestamp)

        traces.append(
            {
                "trace_id": trace_id,
                "event_count": len(events),
                "start_time": first_event.timestamp.isoformat(),
                "end_time": last_event.timestamp.isoformat(),
                "duration_seconds": (last_event.timestamp - first_event.timestamp).total_seconds(),
                "has_errors": any(e.severity == "error" for e in events),
            }
        )

    # Apply pagination
    paginated_traces = traces[offset : offset + limit]

    return {
        "traces": paginated_traces,
        "total": len(traces),
        "limit": limit,
        "offset": offset,
    }


@router.get("/traces/{trace_id}")
async def get_trace_details(trace_id: str) -> dict:
    """Get detailed execution trace.

    Args:
        trace_id: Trace ID (correlation_id or session_id)

    Returns:
        Execution trace with spans (events)
    """
    service = get_query_service()

    # Get events for this trace
    events = await service.get_recent_events(
        limit=10000,
        filters=EventFilters(session_ids=[trace_id]),
    )

    if not events:
        raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found")

    # Sort events by timestamp
    sorted_events = sorted(events, key=lambda e: e.timestamp)

    # Calculate trace statistics
    tool_calls = [e for e in sorted_events if e.event_type and "tool" in e.event_type.lower()]
    errors = [e for e in sorted_events if e.severity == "error"]

    # Build span tree (simplified - just list events in order)
    spans = []
    for event in sorted_events:
        span = {
            "span_id": str(event.id),
            "parent_span_id": None,  # Could be derived from event hierarchy
            "operation": event.event_type or "unknown",
            "start_time": event.timestamp.isoformat(),
            "duration_ms": None,  # Could be calculated from paired events
            "status": "error" if event.severity == "error" else "success",
            "tags": {
                "severity": event.severity,
                "tool_name": event.tool_name,
            },
            "data": event.data,
        }
        spans.append(span)

    return {
        "trace_id": trace_id,
        "spans": spans,
        "span_count": len(spans),
        "tool_calls": len(tool_calls),
        "errors": len(errors),
        "start_time": sorted_events[0].timestamp.isoformat() if sorted_events else None,
        "end_time": sorted_events[-1].timestamp.isoformat() if sorted_events else None,
        "duration_seconds": (
            (sorted_events[-1].timestamp - sorted_events[0].timestamp).total_seconds()
            if len(sorted_events) >= 2
            else 0
        ),
    }


# =============================================================================
# Export Endpoints
# =============================================================================


@router.get("/export/events")
async def export_events(
    format: str = Query("json", description="Export format (json, csv, jsonl)"),
    start_time: Optional[str] = Query(None, description="Start time (ISO format)"),
    end_time: Optional[str] = Query(None, description="End time (ISO format)"),
    topic_pattern: Optional[str] = Query(None, description="Topic pattern filter"),
) -> StreamingResponse:
    """Export events data.

    Args:
        format: Export format (json, csv, jsonl)
        start_time: Start time filter
        end_time: End time filter
        topic_pattern: Topic pattern filter

    Returns:
        Streaming response with exported data
    """
    # Parse time filters
    start_dt = datetime.fromisoformat(start_time) if start_time else None
    end_dt = datetime.fromisoformat(end_time) if end_time else None

    service = get_export_service()

    # Determine media type
    media_types = {
        "json": "application/json",
        "csv": "text/csv",
        "jsonl": "application/jsonl",
    }
    media_type = media_types.get(format, "application/json")

    # Create filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"victor_events_{timestamp}.{format}"

    # Stream export
    async def event_stream():
        async for chunk in service.export_events(
            format=format,
            start_time=start_dt,
            end_time=end_dt,
            topic_pattern=topic_pattern,
        ):
            yield chunk

    return StreamingResponse(
        event_stream(),
        media_type=media_type,
        headers={
            "Content-Disposition": f"attachment; filename={filename}",
        },
    )


# =============================================================================
# Web UI Endpoints
# =============================================================================


@router.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard() -> HTMLResponse:
    """Serve the web-based observability dashboard.

    Returns:
        HTML page for the dashboard
    """
    static_dir = Path(__file__).parent.parent / "static"
    html_file = static_dir / "observability.html"

    if not html_file.exists():
        return HTMLResponse(
            content="<h1>Dashboard not found</h1><p>Static files not available.</p>",
            status_code=404,
        )

    return HTMLResponse(content=html_file.read_text())


@router.get("/static/{file_path:path}")
async def serve_static(file_path: str) -> FileResponse:
    """Serve static files (CSS, JS) for the dashboard.

    Args:
        file_path: Path to the static file

    Returns:
        Static file response
    """
    static_dir = Path(__file__).parent.parent / "static"
    file_path = file_path.lstrip("/")
    file_location = static_dir / file_path

    if not file_location.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(file_location)


# =============================================================================
# Additional Endpoints for Web UI
# =============================================================================


class ToolStatsResponse(BaseModel):
    """Tool execution statistics response."""

    tool_name: str
    total_calls: int
    successful_calls: int
    failed_calls: int
    avg_duration_ms: float
    last_called: Optional[str] = None


class TokenUsageResponse(BaseModel):
    """Token usage response."""

    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    by_session: Dict[str, int] = Field(default_factory=dict)
    by_model: Dict[str, int] = Field(default_factory=dict)


@router.get("/tools/stats", response_model=List[ToolStatsResponse])
async def get_tool_statistics(
    limit: int = Query(20, ge=1, le=100),
) -> List[ToolStatsResponse]:
    """Get tool execution statistics.

    Args:
        limit: Maximum number of tools to return

    Returns:
        List of tool statistics
    """
    service = get_query_service()
    stats = await service.get_tool_statistics(limit=limit)

    return [
        ToolStatsResponse(
            tool_name=stat["tool_name"],
            total_calls=stat["total_calls"],
            successful_calls=stat["successful_calls"],
            failed_calls=stat["failed_calls"],
            avg_duration_ms=stat["avg_duration_ms"],
            last_called=stat.get("last_called"),
        )
        for stat in stats
    ]


@router.get("/tokens/usage", response_model=TokenUsageResponse)
async def get_token_usage() -> TokenUsageResponse:
    """Get token usage statistics.

    Returns:
        Token usage breakdown
    """
    service = get_query_service()
    usage = await service.get_token_usage()

    return TokenUsageResponse(
        total_tokens=usage.get("total_tokens", 0),
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        by_session=usage.get("by_session", {}),
        by_model=usage.get("by_model", {}),
    )

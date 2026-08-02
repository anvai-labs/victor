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

"""HITL API Service for Remote Deployments.

Provides a REST API and web UI for Human-in-the-Loop approvals
when workflows are deployed to remote environments (Docker, K8s,
Lambda, etc.) where CLI-based interaction isn't possible.

Features:
- REST API endpoints for HITL request management
- Web-based approval UI (embedded HTML/JS)
- WebSocket support for real-time updates
- Token-based authentication (optional)
- Request expiration handling

Usage:
    # Standalone server
    from victor.workflows.hitl_api import create_hitl_app, run_hitl_server

    app = create_hitl_app()
    await run_hitl_server(app, host="0.0.0.0", port=8080)

    # As router in existing FastAPI app
    from victor.workflows.hitl_api import create_hitl_router

    app = FastAPI()
    app.include_router(create_hitl_router(hitl_store), prefix="/hitl")

Example API calls:
    # List pending requests
    GET /hitl/requests

    # Get specific request
    GET /hitl/requests/{request_id}

    # Submit response (approve)
    POST /hitl/respond/{request_id}
    {
        "approved": true,
        "reason": "Looks good"
    }

    # Submit response (reject)
    POST /hitl/respond/{request_id}
    {
        "approved": false,
        "reason": "Needs review"
    }
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from victor.config.settings import get_project_paths
from victor.workflows.hitl import (
    HITLFallback,
    HITLNodeType,
    HITLRequest,
    HITLResponse,
    HITLStatus,
)

# Union type for store implementations
HITLStoreProtocol = Any  # Both HITLStore and SQLiteHITLStore implement same interface

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    """Return current UTC time (timezone-aware)."""
    return datetime.now(timezone.utc)


def get_default_hitl_db_path() -> Path:
    """Resolve the default persistent HITL database path."""
    return get_project_paths().global_victor_dir / "hitl.db"


# ============================================================================
# Request Store
# ============================================================================


@dataclass
class StoredRequest:
    """A stored HITL request with metadata.

    Attributes:
        request: The original HITL request
        workflow_id: ID of the workflow that created this request
        thread_id: Thread ID for checkpointing
        status: Current status
        response: Response if available
        created_at: When request was stored
        expires_at: When request expires
        notified: Whether notifications have been sent
    """

    request: HITLRequest
    workflow_id: str = ""
    thread_id: str = ""
    status: str = "pending"
    response: Optional[HITLResponse] = None
    created_at: datetime = field(default_factory=_utc_now)
    expires_at: Optional[datetime] = None
    notified: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for API response."""
        data = {
            "request_id": self.request.request_id,
            "workflow_id": self.workflow_id,
            "thread_id": self.thread_id,
            "node_id": self.request.node_id,
            "hitl_type": self.request.hitl_type.value,
            "prompt": self.request.prompt,
            "context": self.request.context,
            "choices": self.request.choices,
            "default_value": self.request.default_value,
            "timeout": self.request.timeout,
            "fallback": self.request.fallback.value,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }
        if self.response:
            data["response"] = self.response.to_dict()
        return data


class HITLStore:
    """In-memory store for HITL requests.

    Thread-safe storage for pending HITL requests with
    event-based notification when responses arrive.

    For production, consider Redis or database-backed store.
    """

    def __init__(self):
        self._requests: Dict[str, StoredRequest] = {}
        self._events: Dict[str, asyncio.Event] = {}
        self._subscribers: Dict[str, Set[Callable]] = {}
        self._lock = asyncio.Lock()

    async def store_request(
        self,
        request: HITLRequest,
        workflow_id: str = "",
        thread_id: str = "",
    ) -> StoredRequest:
        """Store a new HITL request.

        Args:
            request: The HITL request
            workflow_id: Associated workflow ID
            thread_id: Thread ID for checkpointing

        Returns:
            StoredRequest with metadata
        """
        async with self._lock:
            now = _utc_now()
            stored = StoredRequest(
                request=request,
                workflow_id=workflow_id,
                thread_id=thread_id,
                status="pending",
                created_at=now,
                expires_at=datetime.fromtimestamp(
                    now.timestamp() + request.timeout, tz=timezone.utc
                ),
            )
            self._requests[request.request_id] = stored
            self._events[request.request_id] = asyncio.Event()

            logger.info(f"Stored HITL request {request.request_id}")
            await self._notify_subscribers("new_request", stored)

            return stored

    async def get_request(self, request_id: str) -> Optional[StoredRequest]:
        """Get a stored request by ID.

        Args:
            request_id: The request ID

        Returns:
            StoredRequest or None if not found
        """
        return self._requests.get(request_id)

    async def list_pending(self) -> List[StoredRequest]:
        """List all pending requests.

        Returns:
            List of pending StoredRequests
        """
        now = _utc_now()
        pending = []
        for stored in self._requests.values():
            if stored.status == HITLStatus.PENDING.value:
                # Check if expired
                if stored.expires_at and now > stored.expires_at:
                    stored.status = "expired"
                else:
                    pending.append(stored)
        return pending

    async def submit_response(
        self,
        request_id: str,
        approved: bool,
        value: Optional[Any] = None,
        modifications: Optional[Dict[str, Any]] = None,
        reason: Optional[str] = None,
        responder: Optional[str] = None,
    ) -> Optional[HITLResponse]:
        """Submit a response to a HITL request.

        Args:
            request_id: The request ID
            approved: Whether approved
            value: Value for choice/input types
            modifications: Modifications for review type
            reason: Optional reason/comment
            responder: Identity of the human who responded (delegated/SSO flow)

        Returns:
            HITLResponse or None if request not found
        """
        async with self._lock:
            stored = self._requests.get(request_id)
            if not stored:
                logger.warning(f"HITL request {request_id} not found")
                return None

            if stored.status != "pending":
                logger.warning(f"HITL request {request_id} already {stored.status}")
                return None

            # Determine status
            if modifications:
                status = HITLStatus.MODIFIED
            elif approved:
                status = HITLStatus.APPROVED
            else:
                status = HITLStatus.REJECTED

            response = HITLResponse(
                request_id=request_id,
                status=status,
                approved=approved,
                value=value,
                modifications=modifications,
                reason=reason,
                responder=responder,
            )

            stored.response = response
            stored.status = status.value

            # Signal waiting coroutines
            event = self._events.get(request_id)
            if event:
                event.set()

            logger.info(f"HITL response submitted for {request_id}: {status.value}")
            await self._notify_subscribers("response", stored)

            return response

    async def wait_for_response(
        self,
        request_id: str,
        timeout: Optional[float] = None,
    ) -> Optional[HITLResponse]:
        """Wait for a response to a HITL request.

        Args:
            request_id: The request ID
            timeout: Timeout in seconds (uses request timeout if not specified)

        Returns:
            HITLResponse or None if timeout/not found
        """
        stored = self._requests.get(request_id)
        if not stored:
            return None

        event = self._events.get(request_id)
        if not event:
            return None

        effective_timeout = timeout or stored.request.timeout

        try:
            await asyncio.wait_for(event.wait(), timeout=effective_timeout)
            return stored.response
        except asyncio.TimeoutError:
            # Mark as timed out
            async with self._lock:
                if stored.status == HITLStatus.PENDING.value:
                    stored.status = HITLStatus.TIMEOUT.value
            return None

    async def subscribe(
        self,
        event_type: str,
        callback: Callable[[StoredRequest], Any],
    ) -> None:
        """Subscribe to request events.

        Args:
            event_type: Event type ("new_request", "response", "expired")
            callback: Callback function
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = set()
        self._subscribers[event_type].add(callback)

    async def unsubscribe(
        self,
        event_type: str,
        callback: Callable[[StoredRequest], Any],
    ) -> None:
        """Unsubscribe from request events.

        Args:
            event_type: Event type
            callback: Callback function
        """
        if event_type in self._subscribers:
            self._subscribers[event_type].discard(callback)

    async def _notify_subscribers(
        self,
        event_type: str,
        stored: StoredRequest,
    ) -> None:
        """Notify subscribers of an event."""
        callbacks = self._subscribers.get(event_type, set())
        for callback in callbacks:
            try:
                result = callback(stored)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Subscriber callback error: {e}")

    async def cleanup_expired(self) -> int:
        """Remove expired requests.

        Returns:
            Number of requests cleaned up
        """
        now = _utc_now()
        to_remove = []

        async with self._lock:
            for request_id, stored in self._requests.items():
                if stored.expires_at and now > stored.expires_at:
                    if stored.status == HITLStatus.PENDING.value:
                        stored.status = "expired"
                        event = self._events.get(request_id)
                        if event:
                            event.set()
                    # Mark for removal after 1 hour
                    time_since_expiry = (now - stored.expires_at).total_seconds()
                    if time_since_expiry > 3600:
                        to_remove.append(request_id)

            for request_id in to_remove:
                del self._requests[request_id]
                self._events.pop(request_id, None)

        return len(to_remove)


# ============================================================================
# SQLite-backed Persistent Store
# ============================================================================


class SQLiteHITLStore:
    """SQLite-backed persistent store for HITL requests.

    Provides durable storage with UUID-based request IDs,
    audit trail, and multi-server support.

    Database schema:
        hitl_requests:
            - id: UUID primary key
            - workflow_id: Workflow identifier
            - thread_id: Thread for checkpointing
            - node_id: HITL node identifier
            - hitl_type: Type of interaction
            - prompt: User prompt
            - context: JSON context data
            - choices: JSON choices (if applicable)
            - default_value: JSON default value
            - timeout: Timeout in seconds
            - fallback: Fallback behavior
            - status: Current status
            - response: JSON response data
            - created_at: Creation timestamp
            - expires_at: Expiration timestamp
            - responded_at: Response timestamp
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        table_name: str = "hitl_requests",
    ):
        """Initialize SQLite store.

        Args:
            db_path: Path to SQLite database (default: ~/.victor/hitl.db)
            table_name: Table name for HITL requests
        """
        if db_path is None:
            default_db_path = get_default_hitl_db_path()
            default_db_path.parent.mkdir(parents=True, exist_ok=True)
            db_path = str(default_db_path)

        self.db_path = db_path
        # SQLite cannot parameterize identifiers, and the table name is
        # interpolated into DDL/DML below. Validate it as a plain SQL
        # identifier so the f-string interpolation cannot be used for
        # injection (the only non-parameterized value in every query).
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table_name):
            raise ValueError(
                f"Invalid table_name {table_name!r}: must be a valid SQL "
                "identifier (letters, digits, underscores; not starting with a digit)"
            )
        self.table_name = table_name
        self._events: Dict[str, asyncio.Event] = {}
        self._subscribers: Dict[str, Set[Callable]] = {}
        self._lock = asyncio.Lock()

        # Initialize database
        self._init_db()

    def _get_connection(self):
        """Get a database connection."""
        import sqlite3

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self._get_connection()
        try:
            # table_name validated as a SQL identifier in __init__; no other
            # value is interpolated (all data uses bound ? params).
            # nosemgrep
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id TEXT PRIMARY KEY,
                    workflow_id TEXT,
                    thread_id TEXT,
                    node_id TEXT,
                    hitl_type TEXT,
                    prompt TEXT,
                    context TEXT,
                    choices TEXT,
                    default_value TEXT,
                    timeout REAL,
                    fallback TEXT,
                    status TEXT DEFAULT 'pending',
                    response TEXT,
                    created_at TEXT,
                    expires_at TEXT,
                    responded_at TEXT
                )
            """)
            # nosemgrep
            conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_status
                ON {self.table_name}(status)
            """)
            # nosemgrep
            conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_workflow
                ON {self.table_name}(workflow_id)
            """)
            conn.commit()
        finally:
            conn.close()

    def _row_to_stored_request(self, row) -> StoredRequest:
        """Convert database row to StoredRequest."""
        import json

        context = json.loads(row["context"]) if row["context"] else {}
        choices = json.loads(row["choices"]) if row["choices"] else None
        default_value = json.loads(row["default_value"]) if row["default_value"] else None

        request = HITLRequest(
            request_id=row["id"],
            node_id=row["node_id"],
            hitl_type=HITLNodeType(row["hitl_type"]),
            prompt=row["prompt"],
            context=context,
            choices=choices,
            default_value=default_value,
            timeout=row["timeout"],
            fallback=HITLFallback(row["fallback"]),
            created_at=datetime.fromisoformat(row["created_at"]),
        )

        response = None
        if row["response"]:
            resp_data = json.loads(row["response"])
            response = HITLResponse(
                request_id=resp_data["request_id"],
                status=HITLStatus(resp_data["status"]),
                approved=resp_data.get("approved", False),
                value=resp_data.get("value"),
                modifications=resp_data.get("modifications"),
                reason=resp_data.get("reason"),
                responded_at=(
                    datetime.fromisoformat(resp_data["responded_at"])
                    if resp_data.get("responded_at")
                    else _utc_now()
                ),
            )

        return StoredRequest(
            request=request,
            workflow_id=row["workflow_id"] or "",
            thread_id=row["thread_id"] or "",
            status=row["status"],
            response=response,
            created_at=datetime.fromisoformat(row["created_at"]),
            expires_at=(datetime.fromisoformat(row["expires_at"]) if row["expires_at"] else None),
        )

    async def store_request(
        self,
        request: HITLRequest,
        workflow_id: str = "",
        thread_id: str = "",
    ) -> StoredRequest:
        """Store a new HITL request.

        Args:
            request: The HITL request
            workflow_id: Associated workflow ID
            thread_id: Thread ID for checkpointing

        Returns:
            StoredRequest with metadata
        """
        import json

        async with self._lock:
            now = _utc_now()
            expires_at = datetime.fromtimestamp(now.timestamp() + request.timeout, tz=timezone.utc)

            conn = self._get_connection()
            try:
                conn.execute(  # nosemgrep  # table_name validated as identifier; values bound
                    f"""
                    INSERT INTO {self.table_name} (
                        id, workflow_id, thread_id, node_id, hitl_type,
                        prompt, context, choices, default_value, timeout,
                        fallback, status, created_at, expires_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        request.request_id,
                        workflow_id,
                        thread_id,
                        request.node_id,
                        request.hitl_type.value,
                        request.prompt,
                        json.dumps(request.context),
                        json.dumps(request.choices) if request.choices else None,
                        (
                            json.dumps(request.default_value)
                            if request.default_value is not None
                            else None
                        ),
                        request.timeout,
                        request.fallback.value,
                        "pending",
                        now.isoformat(),
                        expires_at.isoformat(),
                    ),
                )
                conn.commit()
            finally:
                conn.close()

            stored = StoredRequest(
                request=request,
                workflow_id=workflow_id,
                thread_id=thread_id,
                status="pending",
                created_at=now,
                expires_at=expires_at,
            )

            self._events[request.request_id] = asyncio.Event()
            logger.info(f"Stored HITL request {request.request_id} to SQLite")
            await self._notify_subscribers("new_request", stored)

            return stored

    async def get_request(self, request_id: str) -> Optional[StoredRequest]:
        """Get a stored request by ID.

        Args:
            request_id: The request ID

        Returns:
            StoredRequest or None if not found
        """
        conn = self._get_connection()
        try:
            row = conn.execute(  # nosemgrep  # table_name validated as identifier; values bound
                f"SELECT * FROM {self.table_name} WHERE id = ?",
                (request_id,),
            ).fetchone()
            if row:
                return self._row_to_stored_request(row)
            return None
        finally:
            conn.close()

    async def list_pending(self) -> List[StoredRequest]:
        """List all pending requests.

        Returns:
            List of pending StoredRequests
        """
        now = _utc_now()
        conn = self._get_connection()
        try:
            # Update expired requests
            conn.execute(  # nosemgrep  # table_name validated as identifier; values bound
                f"""
                UPDATE {self.table_name}
                SET status = 'expired'
                WHERE status = 'pending' AND expires_at < ?
            """,
                (now.isoformat(),),
            )
            conn.commit()

            # Fetch pending
            rows = conn.execute(  # nosemgrep  # table_name validated as identifier; values bound
                f"SELECT * FROM {self.table_name} WHERE status = 'pending' ORDER BY created_at"
            ).fetchall()
            return [self._row_to_stored_request(row) for row in rows]
        finally:
            conn.close()

    async def submit_response(
        self,
        request_id: str,
        approved: bool,
        value: Optional[Any] = None,
        modifications: Optional[Dict[str, Any]] = None,
        reason: Optional[str] = None,
        responder: Optional[str] = None,
    ) -> Optional[HITLResponse]:
        """Submit a response to a HITL request.

        Args:
            request_id: The request ID
            approved: Whether approved
            value: Value for choice/input types
            modifications: Modifications for review type
            reason: Optional reason/comment
            responder: Identity of the human who responded (delegated/SSO flow)

        Returns:
            HITLResponse or None if request not found
        """
        import json

        async with self._lock:
            conn = self._get_connection()
            try:
                # Check request exists and is pending
                row = conn.execute(  # nosemgrep  # table_name validated as identifier; values bound
                    f"SELECT * FROM {self.table_name} WHERE id = ? AND status = 'pending'",
                    (request_id,),
                ).fetchone()

                if not row:
                    logger.warning(f"HITL request {request_id} not found or not pending")
                    return None

                # Determine status
                if modifications:
                    status = HITLStatus.MODIFIED
                elif approved:
                    status = HITLStatus.APPROVED
                else:
                    status = HITLStatus.REJECTED

                now = _utc_now()
                response = HITLResponse(
                    request_id=request_id,
                    status=status,
                    approved=approved,
                    value=value,
                    modifications=modifications,
                    reason=reason,
                    responder=responder,
                    responded_at=now,
                )

                # Update database
                conn.execute(  # nosemgrep  # table_name validated as identifier; values bound
                    f"""
                    UPDATE {self.table_name}
                    SET status = ?, response = ?, responded_at = ?
                    WHERE id = ?
                """,
                    (
                        status.value,
                        json.dumps(response.to_dict()),
                        now.isoformat(),
                        request_id,
                    ),
                )
                conn.commit()

                # Signal waiting coroutines
                event = self._events.get(request_id)
                if event:
                    event.set()

                logger.info(f"HITL response submitted for {request_id}: {status.value}")

                stored = self._row_to_stored_request(
                    conn.execute(  # nosemgrep  # table_name validated as identifier; values bound
                        f"SELECT * FROM {self.table_name} WHERE id = ?",
                        (request_id,),
                    ).fetchone()
                )
                await self._notify_subscribers("response", stored)

                return response

            finally:
                conn.close()

    async def wait_for_response(
        self,
        request_id: str,
        timeout: Optional[float] = None,
    ) -> Optional[HITLResponse]:
        """Wait for a response to a HITL request.

        Args:
            request_id: The request ID
            timeout: Timeout in seconds

        Returns:
            HITLResponse or None if timeout/not found
        """
        stored = await self.get_request(request_id)
        if not stored:
            return None

        # If already has response, return it
        if stored.response:
            return stored.response

        event = self._events.get(request_id)
        if not event:
            event = asyncio.Event()
            self._events[request_id] = event

        effective_timeout = timeout or stored.request.timeout

        try:
            await asyncio.wait_for(event.wait(), timeout=effective_timeout)
            # Fetch fresh response from database
            stored = await self.get_request(request_id)
            return stored.response if stored else None
        except asyncio.TimeoutError:
            # Mark as timed out in database
            async with self._lock:
                conn = self._get_connection()
                try:
                    # table_name validated as identifier in __init__; id is bound.
                    sql = (
                        f"UPDATE {self.table_name} "  # nosemgrep
                        "SET status = 'timeout' WHERE id = ? AND status = 'pending'"
                    )
                    # nosemgrep
                    conn.execute(sql, (request_id,))
                    conn.commit()
                finally:
                    conn.close()
            return None

    async def subscribe(
        self,
        event_type: str,
        callback: Callable[[StoredRequest], Any],
    ) -> None:
        """Subscribe to request events."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = set()
        self._subscribers[event_type].add(callback)

    async def unsubscribe(
        self,
        event_type: str,
        callback: Callable[[StoredRequest], Any],
    ) -> None:
        """Unsubscribe from request events."""
        if event_type in self._subscribers:
            self._subscribers[event_type].discard(callback)

    async def _notify_subscribers(
        self,
        event_type: str,
        stored: StoredRequest,
    ) -> None:
        """Notify subscribers of an event."""
        callbacks = self._subscribers.get(event_type, set())
        for callback in callbacks:
            try:
                result = callback(stored)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Subscriber callback error: {e}")

    async def cleanup_expired(self) -> int:
        """Remove old expired/completed requests.

        Returns:
            Number of requests cleaned up
        """
        # Keep records for 24 hours after completion
        cutoff = datetime.fromtimestamp(_utc_now().timestamp() - 86400, tz=timezone.utc)

        conn = self._get_connection()
        try:
            cursor = conn.execute(  # nosemgrep  # table_name validated as identifier; values bound
                f"""
                DELETE FROM {self.table_name}
                WHERE status != 'pending'
                AND (responded_at < ? OR (expires_at < ? AND responded_at IS NULL))
            """,
                (cutoff.isoformat(), cutoff.isoformat()),
            )
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()

    async def get_request_history(
        self,
        workflow_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[StoredRequest]:
        """Get request history for audit purposes.

        Args:
            workflow_id: Filter by workflow ID (optional)
            limit: Maximum number of records

        Returns:
            List of StoredRequests
        """
        conn = self._get_connection()
        try:
            if workflow_id:
                rows = conn.execute(  # nosemgrep  # table_name validated as identifier; values bound
                    f"""
                    SELECT * FROM {self.table_name}
                    WHERE workflow_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                """,
                    (workflow_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(  # nosemgrep  # table_name validated as identifier; values bound
                    f"""
                    SELECT * FROM {self.table_name}
                    ORDER BY created_at DESC
                    LIMIT ?
                """,
                    (limit,),
                ).fetchall()
            return [self._row_to_stored_request(row) for row in rows]
        finally:
            conn.close()


# ============================================================================
# Global Store Management
# ============================================================================


# Global store instance
_global_store: Optional[HITLStore] = None
_global_sqlite_store: Optional[SQLiteHITLStore] = None


def get_global_store(persistent: bool = False) -> HITLStore:
    """Get or create the global HITL store.

    Args:
        persistent: If True, use SQLite-backed persistent store

    Returns:
        The global HITLStore instance
    """
    global _global_store, _global_sqlite_store

    if persistent:
        if _global_sqlite_store is None:
            _global_sqlite_store = SQLiteHITLStore()
        return _global_sqlite_store  # type: ignore

    if _global_store is None:
        _global_store = HITLStore()
    return _global_store


def get_sqlite_store(db_path: Optional[str] = None) -> SQLiteHITLStore:
    """Get or create a SQLite HITL store.

    Args:
        db_path: Optional custom database path

    Returns:
        SQLiteHITLStore instance
    """
    global _global_sqlite_store

    if db_path:
        return SQLiteHITLStore(db_path=db_path)

    if _global_sqlite_store is None:
        _global_sqlite_store = SQLiteHITLStore()
    return _global_sqlite_store


# ============================================================================
# API Handler (implements HITLHandler protocol)
# ============================================================================


class APIHITLHandler:
    """HITL handler that uses API for remote approvals.

    Stores requests in HITLStore and waits for API responses.
    Use this handler when deploying to Docker, Kubernetes, Lambda, etc.
    """

    def __init__(
        self,
        store: Optional[HITLStore] = None,
        workflow_id: str = "",
        thread_id: str = "",
    ):
        """Initialize API handler.

        Args:
            store: HITL store (uses global if not provided)
            workflow_id: Workflow ID for tracking
            thread_id: Thread ID for checkpointing
        """
        self.store = store or get_global_store()
        self.workflow_id = workflow_id
        self.thread_id = thread_id

    async def request_human_input(self, request: HITLRequest) -> HITLResponse:
        """Request input via API.

        Stores the request and waits for API response.

        Args:
            request: The HITL request

        Returns:
            HITLResponse from API
        """
        # Store request
        await self.store.store_request(
            request,
            workflow_id=self.workflow_id,
            thread_id=self.thread_id,
        )

        logger.info(
            f"HITL request {request.request_id} waiting for API response "
            f"(timeout: {request.timeout}s)"
        )

        # Wait for response
        response = await self.store.wait_for_response(request.request_id)

        if response:
            return response

        # Handle timeout based on fallback
        if request.fallback == HITLFallback.CONTINUE:
            return HITLResponse(
                request_id=request.request_id,
                status=HITLStatus.TIMEOUT,
                approved=True,
                value=request.default_value,
                reason="Timed out, continuing with default",
            )
        elif request.fallback == HITLFallback.SKIP:
            return HITLResponse(
                request_id=request.request_id,
                status=HITLStatus.SKIPPED,
                approved=True,
                reason="Timed out, skipping",
            )
        else:
            return HITLResponse(
                request_id=request.request_id,
                status=HITLStatus.TIMEOUT,
                approved=False,
                reason=f"Timed out after {request.timeout}s",
            )


# ============================================================================
# FastAPI Router and Endpoints
# ============================================================================

# Import Pydantic at module level for model definitions
try:
    from pydantic import BaseModel as PydanticBaseModel

    class ResponseSubmit(PydanticBaseModel):
        """Request body for submitting HITL response."""

        approved: bool
        value: Optional[Any] = None
        modifications: Optional[Dict[str, Any]] = None
        reason: Optional[str] = None

except ImportError:
    # Define a placeholder if pydantic not installed
    ResponseSubmit = None  # type: ignore


def _render_decision_page(title: str, message: str, *, ok: bool) -> str:
    """Render a minimal confirmation page shown after a signed approve/reject click."""
    import html as _html

    accent = "#16a34a" if ok else "#dc2626"
    icon = "✓" if ok else "✕"
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_html.escape(title)}</title>
<style>
  body {{ font-family: system-ui, -apple-system, sans-serif; background: #0b0f17;
         color: #e5e7eb; display: flex; min-height: 100vh; margin: 0;
         align-items: center; justify-content: center; }}
  .card {{ background: #111827; border: 1px solid #1f2937; border-radius: 14px;
          padding: 40px 48px; max-width: 460px; text-align: center; }}
  .icon {{ font-size: 44px; color: {accent}; }}
  h1 {{ font-size: 20px; margin: 16px 0 8px; }}
  p {{ color: #9ca3af; line-height: 1.5; }}
</style></head>
<body><div class="card">
  <div class="icon">{icon}</div>
  <h1>{_html.escape(title)}</h1>
  <p>{_html.escape(message)}</p>
</div></body></html>"""


def create_hitl_router(
    store: Optional[HITLStore] = None,
    require_auth: bool = False,
    auth_token: Optional[str] = None,
):
    """Create FastAPI router for HITL endpoints.

    Args:
        store: HITL store (uses global if not provided)
        require_auth: Whether to require authentication
        auth_token: Bearer token for authentication

    Returns:
        FastAPI APIRouter
    """
    try:
        from fastapi import APIRouter, Body, HTTPException, Header, Query
        from fastapi.responses import HTMLResponse, RedirectResponse
    except ImportError:
        raise ImportError("FastAPI is required for HITL API. Install with: pip install fastapi")

    router = APIRouter(tags=["hitl"])
    hitl_store = store or get_global_store()
    expected_token = auth_token or os.environ.get("HITL_AUTH_TOKEN")

    # Delegated/SSO identity capture is opt-in: configured -> approve/reject links
    # route through an Entra sign-in so the decision records *who* approved.
    from victor.core.identity import OidcConfig

    oidc_config = OidcConfig.from_env()

    def _encode_state(request_id: str, action: str, token: Optional[str]) -> str:
        return f"{request_id}|{action}|{token or ''}"

    def _decode_state(state: str) -> tuple:
        parts = state.split("|", 2)
        if len(parts) != 3:
            return "", "", ""
        return parts[0], parts[1], parts[2]

    async def _record_decision(
        request_id: str,
        action: str,
        reason: Optional[str],
        responder: Optional[str],
    ):
        """Record an approve/reject and render the confirmation/error page."""
        response = await hitl_store.submit_response(
            request_id=request_id,
            approved=(action == "approve"),
            reason=reason or f"Responded via signed link ({action})",
            responder=responder,
        )
        if not response:
            return HTMLResponse(
                status_code=409,
                content=_render_decision_page(
                    "Already decided",
                    "This request was not found or has already been responded to.",
                    ok=False,
                ),
            )
        verb = "approved" if action == "approve" else "rejected"
        by = f" by {responder}" if responder else ""
        return HTMLResponse(
            content=_render_decision_page(
                f"Request {verb}",
                f"Request {request_id[:12]}… {verb}{by}. You may close this tab.",
                ok=True,
            )
        )

    async def verify_auth(authorization: Optional[str] = Header(None)):
        """Verify authentication if required."""
        if not require_auth:
            return
        if not expected_token:
            return  # No token configured, skip auth
        if not authorization:
            raise HTTPException(status_code=401, detail="Authorization required")
        if not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Invalid authorization format")
        token = authorization[7:]
        if token != expected_token:
            raise HTTPException(status_code=401, detail="Invalid token")

    @router.get("/requests")
    async def list_requests(authorization: Optional[str] = Header(None)):
        """List all pending HITL requests."""
        await verify_auth(authorization)
        pending = await hitl_store.list_pending()
        return {"requests": [r.to_dict() for r in pending], "count": len(pending)}

    @router.get("/requests/{request_id}")
    async def get_request(request_id: str, authorization: Optional[str] = Header(None)):
        """Get a specific HITL request."""
        await verify_auth(authorization)
        stored = await hitl_store.get_request(request_id)
        if not stored:
            raise HTTPException(status_code=404, detail="Request not found")
        return stored.to_dict()

    @router.post("/respond/{request_id}")
    async def submit_response(
        request_id: str,
        body: ResponseSubmit = Body(...),
        authorization: Optional[str] = Header(None),
    ):
        """Submit a response to a HITL request."""
        await verify_auth(authorization)
        response = await hitl_store.submit_response(
            request_id=request_id,
            approved=body.approved,
            value=body.value,
            modifications=body.modifications,
            reason=body.reason,
        )
        if not response:
            raise HTTPException(status_code=404, detail="Request not found or already processed")
        return {"success": True, "response": response.to_dict()}

    @router.get("/respond/{request_id}", response_class=HTMLResponse)
    async def submit_response_via_link(
        request_id: str,
        action: str = Query(..., pattern="^(approve|reject)$"),
        token: Optional[str] = Query(None),
        reason: Optional[str] = Query(None),
    ):
        """Browser-clickable approve/reject from a signed message link.

        This is the receiving end of the approve/reject buttons embedded in
        Slack/Teams/email approval messages. Authentication is the **signed
        token** (not the bearer header used by the POST endpoint): it binds
        (request_id, action, expiry) with an HMAC so a link cannot be forged or
        flipped. Replay of an already-decided request is rejected by the store.
        """
        from victor.workflows.hitl_signing import get_signing_secret, verify_action

        secret = get_signing_secret()
        if secret and not verify_action(request_id, action, token, secret=secret):
            return HTMLResponse(
                status_code=403,
                content=_render_decision_page(
                    "Link invalid or expired",
                    "This approval link is invalid, expired, or has been tampered with.",
                    ok=False,
                ),
            )

        # Delegated/SSO: send the approver to Entra sign-in first so the decision
        # records their identity. The signed token rides along as `state` and is
        # re-verified on return.
        if oidc_config is not None:
            from victor.core.identity import build_authorize_url

            authorize_url = build_authorize_url(
                oidc_config, state=_encode_state(request_id, action, token)
            )
            return RedirectResponse(url=authorize_url, status_code=302)

        # No SSO configured -> record the decision anonymously.
        return await _record_decision(request_id, action, reason, responder=None)

    @router.get("/auth/callback", response_class=HTMLResponse)
    async def auth_callback(
        code: Optional[str] = Query(None),
        state: str = Query(...),
        error: Optional[str] = Query(None),
        error_description: Optional[str] = Query(None),
    ):
        """OIDC redirect target: resolve the signed-in approver, record the decision.

        Entra redirects here after the approver authenticates. We re-verify the
        signed token carried in ``state`` (so the decision is still bound to the
        original request/action), exchange the auth ``code`` for the approver's
        identity via Microsoft Graph ``/me``, and record the decision with that
        identity attached.
        """
        from victor.workflows.hitl_signing import get_signing_secret, verify_action

        if error or not code:
            return HTMLResponse(
                status_code=400,
                content=_render_decision_page(
                    "Sign-in failed",
                    error_description or error or "No authorization code was returned.",
                    ok=False,
                ),
            )

        request_id, action, token = _decode_state(state)
        if action not in ("approve", "reject"):
            return HTMLResponse(
                status_code=400,
                content=_render_decision_page(
                    "Invalid request", "The sign-in state was malformed.", ok=False
                ),
            )

        secret = get_signing_secret()
        if secret and not verify_action(request_id, action, token, secret=secret):
            return HTMLResponse(
                status_code=403,
                content=_render_decision_page(
                    "Link invalid or expired",
                    "This approval link is invalid, expired, or has been tampered with.",
                    ok=False,
                ),
            )

        responder = None
        if oidc_config is not None:
            from victor.core.identity import resolve_identity_from_code

            try:
                identity = await resolve_identity_from_code(oidc_config, code)
                responder = identity.label()
            except Exception as exc:  # pragma: no cover - network/identity failure
                logger.error(f"HITL SSO identity resolution failed: {exc}")
                return HTMLResponse(
                    status_code=502,
                    content=_render_decision_page(
                        "Could not verify identity",
                        "Sign-in succeeded but your identity could not be resolved.",
                        ok=False,
                    ),
                )

        return await _record_decision(request_id, action, reason=None, responder=responder)

    @router.get("/ui", response_class=HTMLResponse)
    async def hitl_ui():
        """Serve the HITL approval web UI."""
        return HTMLResponse(content=get_hitl_ui_html())

    @router.get("/ui/request/{request_id}", response_class=HTMLResponse)
    async def hitl_request_ui(request_id: str):
        """Serve the HITL approval UI for a specific request."""
        return HTMLResponse(content=get_hitl_request_ui_html(request_id))

    @router.get("/history")
    async def get_history(
        workflow_id: Optional[str] = None,
        limit: int = 50,
        authorization: Optional[str] = Header(None),
    ):
        """Get approval history for audit purposes."""
        await verify_auth(authorization)
        if hasattr(hitl_store, "get_request_history"):
            history = await hitl_store.get_request_history(workflow_id=workflow_id, limit=limit)
            return {"history": [r.to_dict() for r in history], "count": len(history)}
        return {"history": [], "count": 0, "note": "History requires SQLite store"}

    return router


def get_hitl_ui_html() -> str:
    """Get HTML for the HITL approval dashboard."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Victor - Workflow Approvals</title>
    <style>
        :root {
            --primary: #6366f1;
            --primary-dark: #4f46e5;
            --primary-light: #818cf8;
            --success: #10b981;
            --success-light: #34d399;
            --danger: #ef4444;
            --danger-light: #f87171;
            --warning: #f59e0b;
            --info: #0ea5e9;
            --bg: #0f172a;
            --bg-elevated: #1e293b;
            --bg-card: #1e293b;
            --bg-input: #0f172a;
            --text: #f1f5f9;
            --text-secondary: #cbd5e1;
            --text-muted: #64748b;
            --border: #334155;
            --border-light: #475569;
            --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -2px rgba(0, 0, 0, 0.2);
            --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.3), 0 4px 6px -4px rgba(0, 0, 0, 0.2);
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Inter', sans-serif;
            background: linear-gradient(135deg, var(--bg) 0%, #1a1f35 100%);
            color: var(--text);
            min-height: 100vh;
            line-height: 1.6;
        }

        /* Navigation */
        .navbar {
            background: rgba(30, 41, 59, 0.8);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--border);
            padding: 0.75rem 2rem;
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .nav-content {
            max-width: 1400px;
            margin: 0 auto;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .logo-section {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .logo {
            width: 36px;
            height: 36px;
            background: linear-gradient(135deg, var(--primary), var(--primary-light));
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 1.25rem;
            color: white;
            box-shadow: 0 2px 8px rgba(99, 102, 241, 0.3);
        }

        .logo-text {
            font-size: 1.125rem;
            font-weight: 600;
            color: var(--text);
        }

        .logo-text span {
            color: var(--text-muted);
            font-weight: 400;
        }

        .nav-actions {
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .nav-tabs {
            display: flex;
            gap: 0.25rem;
            background: var(--bg);
            padding: 0.25rem;
            border-radius: 8px;
        }

        .nav-tab {
            padding: 0.5rem 1rem;
            border: none;
            background: transparent;
            color: var(--text-muted);
            font-size: 0.875rem;
            font-weight: 500;
            cursor: pointer;
            border-radius: 6px;
            transition: all 0.2s;
        }

        .nav-tab:hover { color: var(--text); }
        .nav-tab.active {
            background: var(--primary);
            color: white;
        }

        .status-indicator {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.3);
            border-radius: 20px;
            font-size: 0.8125rem;
            color: var(--success-light);
        }

        .status-dot {
            width: 8px;
            height: 8px;
            background: var(--success);
            border-radius: 50%;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.4); }
            50% { opacity: 0.8; box-shadow: 0 0 0 4px rgba(16, 185, 129, 0); }
        }

        .refresh-btn {
            background: var(--bg);
            border: 1px solid var(--border);
            color: var(--text-secondary);
            padding: 0.5rem 1rem;
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.875rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            transition: all 0.2s;
        }

        .refresh-btn:hover {
            border-color: var(--primary);
            color: var(--primary-light);
        }

        /* Main Content */
        .main-content {
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }

        /* Stats Bar */
        .stats-bar {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }

        .stat-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.25rem;
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .stat-icon {
            width: 48px;
            height: 48px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
        }

        .stat-icon.pending { background: rgba(245, 158, 11, 0.15); }
        .stat-icon.approved { background: rgba(16, 185, 129, 0.15); }
        .stat-icon.rejected { background: rgba(239, 68, 68, 0.15); }

        .stat-info h3 {
            font-size: 1.5rem;
            font-weight: 600;
            color: var(--text);
        }

        .stat-info p {
            font-size: 0.8125rem;
            color: var(--text-muted);
        }

        /* Section Header */
        .section-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1.5rem;
        }

        .section-title {
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--text);
        }

        /* Request Cards */
        .requests-grid {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }

        .request-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 16px;
            overflow: hidden;
            box-shadow: var(--shadow);
            transition: all 0.3s;
        }

        .request-card:hover {
            border-color: var(--primary);
            box-shadow: var(--shadow-lg), 0 0 0 1px var(--primary);
        }

        .card-header {
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(139, 92, 246, 0.05));
            border-bottom: 1px solid var(--border);
            padding: 1.25rem 1.5rem;
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
        }

        .card-header-left {
            display: flex;
            align-items: flex-start;
            gap: 1rem;
        }

        .type-icon {
            width: 48px;
            height: 48px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
            flex-shrink: 0;
        }

        .type-icon.approval { background: rgba(99, 102, 241, 0.2); }
        .type-icon.review { background: rgba(245, 158, 11, 0.2); }
        .type-icon.choice { background: rgba(6, 182, 212, 0.2); }
        .type-icon.input { background: rgba(16, 185, 129, 0.2); }
        .type-icon.confirmation { background: rgba(168, 85, 247, 0.2); }

        .card-title-section h3 {
            font-size: 1rem;
            font-weight: 600;
            color: var(--text);
            margin-bottom: 0.25rem;
        }

        .card-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 0.75rem;
            font-size: 0.75rem;
            color: var(--text-muted);
        }

        .meta-item {
            display: flex;
            align-items: center;
            gap: 0.25rem;
        }

        .badge {
            display: inline-flex;
            align-items: center;
            padding: 0.375rem 0.75rem;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.025em;
        }

        .badge-approval { background: rgba(99, 102, 241, 0.2); color: #a5b4fc; }
        .badge-review { background: rgba(245, 158, 11, 0.2); color: #fcd34d; }
        .badge-choice { background: rgba(6, 182, 212, 0.2); color: #67e8f9; }
        .badge-input { background: rgba(16, 185, 129, 0.2); color: #6ee7b7; }
        .badge-confirmation { background: rgba(168, 85, 247, 0.2); color: #c4b5fd; }

        .timer-badge {
            display: flex;
            align-items: center;
            gap: 0.375rem;
            padding: 0.375rem 0.75rem;
            background: rgba(245, 158, 11, 0.15);
            border: 1px solid rgba(245, 158, 11, 0.3);
            border-radius: 6px;
            font-size: 0.75rem;
            color: var(--warning);
        }

        .card-body {
            padding: 1.5rem;
        }

        .prompt-section {
            margin-bottom: 1.5rem;
        }

        .prompt-label {
            font-size: 0.6875rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
        }

        .prompt-text {
            background: var(--bg-input);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 1rem 1.25rem;
            font-size: 0.9375rem;
            color: var(--text);
            line-height: 1.6;
        }

        .context-section {
            margin-bottom: 1.5rem;
        }

        .context-content {
            background: var(--bg-input);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 1rem 1.25rem;
            font-family: 'SF Mono', 'Fira Code', Monaco, monospace;
            font-size: 0.8125rem;
            color: var(--text-secondary);
            max-height: 200px;
            overflow-y: auto;
            white-space: pre-wrap;
            word-break: break-word;
        }

        /* Choices */
        .choices-section {
            margin-bottom: 1.5rem;
        }

        .choices-grid {
            display: grid;
            gap: 0.75rem;
        }

        .choice-option {
            background: var(--bg-input);
            border: 2px solid var(--border);
            border-radius: 10px;
            padding: 1rem 1.25rem;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .choice-option:hover {
            border-color: var(--primary);
            background: rgba(99, 102, 241, 0.05);
        }

        .choice-option.selected {
            border-color: var(--primary);
            background: rgba(99, 102, 241, 0.1);
        }

        .choice-radio {
            width: 20px;
            height: 20px;
            border: 2px solid var(--border);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
            transition: all 0.2s;
        }

        .choice-option.selected .choice-radio {
            border-color: var(--primary);
            background: var(--primary);
        }

        .choice-option.selected .choice-radio::after {
            content: '';
            width: 8px;
            height: 8px;
            background: white;
            border-radius: 50%;
        }

        .choice-text {
            font-size: 0.9375rem;
            color: var(--text);
        }

        /* Input Field */
        .input-section {
            margin-bottom: 1.5rem;
        }

        .text-input {
            width: 100%;
            background: var(--bg-input);
            border: 2px solid var(--border);
            border-radius: 10px;
            padding: 1rem 1.25rem;
            font-size: 0.9375rem;
            color: var(--text);
            transition: all 0.2s;
        }

        .text-input:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15);
        }

        .text-input::placeholder {
            color: var(--text-muted);
        }

        /* Feedback Section */
        .feedback-section {
            margin-bottom: 1.5rem;
            padding-top: 1.5rem;
            border-top: 1px solid var(--border);
        }

        .feedback-label {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.8125rem;
            font-weight: 500;
            color: var(--text-secondary);
            margin-bottom: 0.75rem;
        }

        .feedback-textarea {
            width: 100%;
            background: var(--bg-input);
            border: 2px solid var(--border);
            border-radius: 10px;
            padding: 1rem 1.25rem;
            font-size: 0.875rem;
            color: var(--text);
            resize: vertical;
            min-height: 80px;
            font-family: inherit;
            transition: all 0.2s;
        }

        .feedback-textarea:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15);
        }

        .feedback-textarea::placeholder {
            color: var(--text-muted);
        }

        .feedback-hint {
            margin-top: 0.5rem;
            font-size: 0.75rem;
            color: var(--text-muted);
        }

        /* Action Buttons */
        .card-actions {
            display: flex;
            gap: 1rem;
            padding-top: 1rem;
        }

        .btn {
            flex: 1;
            padding: 0.875rem 1.5rem;
            border: none;
            border-radius: 10px;
            font-size: 0.9375rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        }

        .btn-approve {
            background: linear-gradient(135deg, var(--success), #059669);
            color: white;
            box-shadow: 0 2px 8px rgba(16, 185, 129, 0.3);
        }

        .btn-approve:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);
        }

        .btn-reject {
            background: linear-gradient(135deg, var(--danger), #dc2626);
            color: white;
            box-shadow: 0 2px 8px rgba(239, 68, 68, 0.3);
        }

        .btn-reject:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(239, 68, 68, 0.4);
        }

        .btn-submit {
            background: linear-gradient(135deg, var(--primary), var(--primary-dark));
            color: white;
            box-shadow: 0 2px 8px rgba(99, 102, 241, 0.3);
        }

        .btn-submit:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
        }

        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none !important;
        }

        /* Empty State */
        .empty-state {
            text-align: center;
            padding: 4rem 2rem;
            background: var(--bg-card);
            border: 1px dashed var(--border);
            border-radius: 16px;
        }

        .empty-icon {
            width: 80px;
            height: 80px;
            background: rgba(99, 102, 241, 0.1);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2.5rem;
            margin: 0 auto 1.5rem;
        }

        .empty-title {
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--text);
            margin-bottom: 0.5rem;
        }

        .empty-description {
            font-size: 0.9375rem;
            color: var(--text-muted);
            max-width: 400px;
            margin: 0 auto;
        }

        /* History Table */
        .history-section {
            display: none;
        }

        .history-section.active {
            display: block;
        }

        .history-table {
            width: 100%;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
        }

        .history-table th,
        .history-table td {
            padding: 1rem 1.25rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }

        .history-table th {
            background: rgba(99, 102, 241, 0.1);
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
        }

        .history-table td {
            font-size: 0.875rem;
            color: var(--text-secondary);
        }

        .history-table tr:last-child td {
            border-bottom: none;
        }

        .status-approved {
            color: var(--success);
            display: flex;
            align-items: center;
            gap: 0.375rem;
        }

        .status-rejected {
            color: var(--danger);
            display: flex;
            align-items: center;
            gap: 0.375rem;
        }

        /* Toast */
        .toast {
            position: fixed;
            bottom: 2rem;
            right: 2rem;
            background: var(--bg-elevated);
            border: 1px solid var(--border);
            padding: 1rem 1.5rem;
            border-radius: 12px;
            display: flex;
            align-items: center;
            gap: 0.75rem;
            box-shadow: var(--shadow-lg);
            transform: translateY(100px);
            opacity: 0;
            transition: all 0.3s;
            z-index: 1000;
        }

        .toast.show {
            transform: translateY(0);
            opacity: 1;
        }

        .toast.success {
            border-color: var(--success);
            background: linear-gradient(135deg, var(--bg-elevated), rgba(16, 185, 129, 0.1));
        }

        .toast.error {
            border-color: var(--danger);
            background: linear-gradient(135deg, var(--bg-elevated), rgba(239, 68, 68, 0.1));
        }

        .toast-icon {
            font-size: 1.25rem;
        }

        .toast-message {
            font-size: 0.9375rem;
            color: var(--text);
        }

        /* Loading */
        .loading-spinner {
            width: 24px;
            height: 24px;
            border: 3px solid var(--border);
            border-top-color: var(--primary);
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        /* Responsive */
        @media (max-width: 768px) {
            .navbar { padding: 0.75rem 1rem; }
            .main-content { padding: 1rem; }
            .stats-bar { grid-template-columns: 1fr; }
            .card-header { flex-direction: column; gap: 1rem; }
            .card-actions { flex-direction: column; }
        }
    </style>
</head>
<body>
    <nav class="navbar">
        <div class="nav-content">
            <div class="logo-section">
                <div class="logo">V</div>
                <div class="logo-text">Victor <span>Approvals</span></div>
            </div>
            <div class="nav-actions">
                <div class="nav-tabs">
                    <button class="nav-tab active" onclick="showTab('pending')">Pending</button>
                    <button class="nav-tab" onclick="showTab('history')">History</button>
                </div>
                <div class="status-indicator">
                    <span class="status-dot"></span>
                    <span id="status-text">Connected</span>
                </div>
                <button class="refresh-btn" onclick="loadData()">
                    <span>↻</span> Refresh
                </button>
            </div>
        </div>
    </nav>

    <main class="main-content">
        <div class="stats-bar">
            <div class="stat-card">
                <div class="stat-icon pending">⏳</div>
                <div class="stat-info">
                    <h3 id="pending-count">0</h3>
                    <p>Pending Approvals</p>
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-icon approved">✓</div>
                <div class="stat-info">
                    <h3 id="approved-count">0</h3>
                    <p>Approved Today</p>
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-icon rejected">✕</div>
                <div class="stat-info">
                    <h3 id="rejected-count">0</h3>
                    <p>Rejected Today</p>
                </div>
            </div>
        </div>

        <div id="pending-section">
            <div class="section-header">
                <h2 class="section-title">Pending Approval Requests</h2>
            </div>
            <div id="requests-container" class="requests-grid">
                <div class="empty-state">
                    <div class="empty-icon">📋</div>
                    <h3 class="empty-title">Loading...</h3>
                    <p class="empty-description">Fetching pending requests</p>
                </div>
            </div>
        </div>

        <div id="history-section" class="history-section">
            <div class="section-header">
                <h2 class="section-title">Approval History</h2>
            </div>
            <table class="history-table">
                <thead>
                    <tr>
                        <th>Request ID</th>
                        <th>Workflow</th>
                        <th>Node</th>
                        <th>Type</th>
                        <th>Status</th>
                        <th>Feedback</th>
                        <th>Time</th>
                    </tr>
                </thead>
                <tbody id="history-body">
                </tbody>
            </table>
        </div>
    </main>

    <div id="toast" class="toast">
        <span class="toast-icon"></span>
        <span class="toast-message"></span>
    </div>

    <script>
        const API_BASE = window.location.pathname.replace(/\\/ui$/, '');
        let requests = [];
        let history = [];
        let selectedChoices = {};

        async function loadData() {
            await Promise.all([loadRequests(), loadHistory()]);
        }

        async function loadRequests() {
            try {
                const response = await fetch(`${API_BASE}/requests`);
                if (!response.ok) throw new Error('Failed to fetch');
                const data = await response.json();
                requests = data.requests || [];
                document.getElementById('pending-count').textContent = requests.length;
                renderRequests();
            } catch (error) {
                console.error('Failed to load requests:', error);
                showToast('Failed to load requests', 'error');
            }
        }

        async function loadHistory() {
            try {
                const response = await fetch(`${API_BASE}/history?limit=50`);
                if (!response.ok) return;
                const data = await response.json();
                history = data.history || [];

                const today = new Date().toDateString();
                const todayApproved = history.filter(h =>
                    h.status === 'approved' && new Date(h.created_at).toDateString() === today
                ).length;
                const todayRejected = history.filter(h =>
                    h.status === 'rejected' && new Date(h.created_at).toDateString() === today
                ).length;

                document.getElementById('approved-count').textContent = todayApproved;
                document.getElementById('rejected-count').textContent = todayRejected;
                renderHistory();
            } catch (error) {
                console.log('History not available');
            }
        }

        function showTab(tab) {
            document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
            document.querySelector(`.nav-tab[onclick="showTab('${tab}')"]`).classList.add('active');

            document.getElementById('pending-section').style.display = tab === 'pending' ? 'block' : 'none';
            document.getElementById('history-section').style.display = tab === 'history' ? 'block' : 'none';
            document.getElementById('history-section').classList.toggle('active', tab === 'history');
        }

        function renderRequests() {
            const container = document.getElementById('requests-container');

            if (!requests || requests.length === 0) {
                container.innerHTML = `
                    <div class="empty-state">
                        <div class="empty-icon">✅</div>
                        <h3 class="empty-title">All caught up!</h3>
                        <p class="empty-description">No pending approval requests. New workflow requests will appear here automatically.</p>
                    </div>
                `;
                return;
            }

            container.innerHTML = requests.map(req => renderCard(req)).join('');
        }

        function renderCard(req) {
            const typeIcons = {
                approval: '🔐',
                review: '📝',
                choice: '🔘',
                input: '✏️',
                confirmation: '✓'
            };

            const expiresIn = req.expires_at ? getTimeRemaining(req.expires_at) : '';
            let contentHtml = '';

            // Context section
            if (req.context && Object.keys(req.context).length > 0) {
                contentHtml += `
                    <div class="context-section">
                        <div class="prompt-label">📋 Context Data</div>
                        <div class="context-content">${formatContext(req.context)}</div>
                    </div>
                `;
            }

            // Choices
            if (req.hitl_type === 'choice' && req.choices) {
                contentHtml += `
                    <div class="choices-section">
                        <div class="prompt-label">Select an option</div>
                        <div class="choices-grid">
                            ${req.choices.map((choice, i) => `
                                <div class="choice-option" onclick="selectChoice('${req.request_id}', '${escapeHtml(choice)}', this)">
                                    <div class="choice-radio"></div>
                                    <span class="choice-text">${escapeHtml(choice)}</span>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                `;
            }

            // Input field
            if (req.hitl_type === 'input') {
                contentHtml += `
                    <div class="input-section">
                        <div class="prompt-label">Your Response</div>
                        <input type="text" class="text-input" id="input-${req.request_id}"
                               placeholder="Enter your response here...">
                    </div>
                `;
            }

            // Actions
            let actionsHtml = '';
            if (req.hitl_type === 'approval' || req.hitl_type === 'confirmation') {
                actionsHtml = `
                    <button class="btn btn-reject" onclick="respond('${req.request_id}', false)">
                        ✕ Reject
                    </button>
                    <button class="btn btn-approve" onclick="respond('${req.request_id}', true)">
                        ✓ Approve
                    </button>
                `;
            } else if (req.hitl_type === 'review') {
                actionsHtml = `
                    <button class="btn btn-reject" onclick="respond('${req.request_id}', false)">
                        ✕ Request Changes
                    </button>
                    <button class="btn btn-approve" onclick="respond('${req.request_id}', true)">
                        ✓ Approve
                    </button>
                `;
            } else {
                actionsHtml = `
                    <button class="btn btn-reject" onclick="respond('${req.request_id}', false)">
                        ✕ Cancel
                    </button>
                    <button class="btn btn-submit" onclick="respond('${req.request_id}', true)">
                        Submit Response
                    </button>
                `;
            }

            return `
                <div class="request-card" id="card-${req.request_id}">
                    <div class="card-header">
                        <div class="card-header-left">
                            <div class="type-icon ${req.hitl_type}">${typeIcons[req.hitl_type] || '📋'}</div>
                            <div class="card-title-section">
                                <h3>${escapeHtml(req.node_id)}</h3>
                                <div class="card-meta">
                                    <span class="meta-item">📁 ${escapeHtml(req.workflow_id || 'Unknown Workflow')}</span>
                                    <span class="meta-item">🔑 ${req.request_id.substring(0, 12)}...</span>
                                </div>
                            </div>
                        </div>
                        <div style="display: flex; flex-direction: column; align-items: flex-end; gap: 0.5rem;">
                            <span class="badge badge-${req.hitl_type}">${req.hitl_type}</span>
                            ${expiresIn ? `<div class="timer-badge">⏱ ${expiresIn}</div>` : ''}
                        </div>
                    </div>
                    <div class="card-body">
                        <div class="prompt-section">
                            <div class="prompt-label">📣 Request</div>
                            <div class="prompt-text">${escapeHtml(req.prompt)}</div>
                        </div>
                        ${contentHtml}
                        <div class="feedback-section">
                            <label class="feedback-label">
                                💬 Feedback / Comments
                            </label>
                            <textarea class="feedback-textarea" id="feedback-${req.request_id}"
                                      placeholder="Add your feedback, notes, or reason for your decision. This will be logged for audit purposes..."></textarea>
                            <div class="feedback-hint">Your feedback will be recorded with your decision for future reference.</div>
                        </div>
                        <div class="card-actions">${actionsHtml}</div>
                    </div>
                </div>
            `;
        }

        function renderHistory() {
            const tbody = document.getElementById('history-body');
            if (!history || history.length === 0) {
                tbody.innerHTML = '<tr><td colspan="7" style="text-align: center; color: var(--text-muted);">No history available</td></tr>';
                return;
            }

            tbody.innerHTML = history.map(h => {
                const date = new Date(h.created_at);
                const statusClass = h.status === 'approved' ? 'status-approved' : 'status-rejected';
                const statusIcon = h.status === 'approved' ? '✓' : '✕';
                const feedback = h.response?.reason || '-';

                return `
                    <tr>
                        <td style="font-family: monospace; font-size: 0.8125rem;">${h.request_id.substring(0, 12)}...</td>
                        <td>${escapeHtml(h.workflow_id || '-')}</td>
                        <td>${escapeHtml(h.node_id)}</td>
                        <td><span class="badge badge-${h.hitl_type}">${h.hitl_type}</span></td>
                        <td><span class="${statusClass}">${statusIcon} ${h.status}</span></td>
                        <td style="max-width: 200px; overflow: hidden; text-overflow: ellipsis;">${escapeHtml(feedback)}</td>
                        <td style="white-space: nowrap;">${date.toLocaleString()}</td>
                    </tr>
                `;
            }).join('');
        }

        function escapeHtml(str) {
            if (!str) return '';
            const div = document.createElement('div');
            div.textContent = str;
            return div.innerHTML;
        }

        function formatContext(context) {
            if (typeof context === 'string') return escapeHtml(context);
            try {
                return escapeHtml(JSON.stringify(context, null, 2));
            } catch {
                return escapeHtml(String(context));
            }
        }

        function getTimeRemaining(expiresAt) {
            const now = new Date();
            const expires = new Date(expiresAt);
            const diff = expires - now;

            if (diff <= 0) return 'Expired';

            const hours = Math.floor(diff / 3600000);
            const minutes = Math.floor((diff % 3600000) / 60000);
            const seconds = Math.floor((diff % 60000) / 1000);

            if (hours > 0) return `${hours}h ${minutes}m`;
            if (minutes > 0) return `${minutes}m ${seconds}s`;
            return `${seconds}s`;
        }

        function selectChoice(requestId, choice, element) {
            selectedChoices[requestId] = choice;
            const card = document.getElementById(`card-${requestId}`);
            card.querySelectorAll('.choice-option').forEach(opt => opt.classList.remove('selected'));
            element.classList.add('selected');
        }

        async function respond(requestId, approved) {
            const request = requests.find(r => r.request_id === requestId);
            const feedback = document.getElementById(`feedback-${requestId}`)?.value || null;

            let value = null;

            if (request.hitl_type === 'choice') {
                value = selectedChoices[requestId];
                if (approved && !value) {
                    showToast('Please select an option first', 'error');
                    return;
                }
            } else if (request.hitl_type === 'input') {
                value = document.getElementById(`input-${requestId}`)?.value;
                if (approved && !value) {
                    showToast('Please enter a response', 'error');
                    return;
                }
            }

            // Disable buttons
            const card = document.getElementById(`card-${requestId}`);
            card.querySelectorAll('.btn').forEach(btn => btn.disabled = true);

            try {
                const response = await fetch(`${API_BASE}/respond/${requestId}`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        approved,
                        value,
                        reason: feedback
                    })
                });

                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Failed to submit response');
                }

                showToast(approved ? 'Request approved!' : 'Request rejected', 'success');

                // Animate card removal
                card.style.transition = 'all 0.3s ease';
                card.style.opacity = '0';
                card.style.transform = 'translateX(20px)';

                setTimeout(() => {
                    loadData();
                }, 300);
            } catch (error) {
                console.error('Failed to submit response:', error);
                showToast(error.message, 'error');
                card.querySelectorAll('.btn').forEach(btn => btn.disabled = false);
            }
        }

        function showToast(message, type = 'info') {
            const toast = document.getElementById('toast');
            const icon = toast.querySelector('.toast-icon');
            const msg = toast.querySelector('.toast-message');

            icon.textContent = type === 'success' ? '✓' : '✕';
            msg.textContent = message;
            toast.className = `toast show ${type}`;

            setTimeout(() => {
                toast.className = 'toast';
            }, 4000);
        }

        // Initial load and auto-refresh
        loadData();
        setInterval(loadData, 10000);
    </script>
</body>
</html>"""


def get_hitl_request_ui_html(request_id: str) -> str:
    """Get HTML for a specific HITL request page."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Victor - Approval Request</title>
    <style>
        :root {{
            --primary: #6366f1;
            --success: #10b981;
            --danger: #ef4444;
            --bg: #0f172a;
            --bg-card: #1e293b;
            --text: #e2e8f0;
            --text-muted: #94a3b8;
            --border: #334155;
        }}

        * {{ box-sizing: border-box; margin: 0; padding: 0; }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 2rem;
        }}

        .card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 2rem;
            max-width: 500px;
            width: 100%;
        }}

        .header {{
            text-align: center;
            margin-bottom: 1.5rem;
        }}

        .logo {{
            width: 48px;
            height: 48px;
            background: linear-gradient(135deg, var(--primary), #818cf8);
            border-radius: 12px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 1.5rem;
            margin-bottom: 1rem;
        }}

        h1 {{ font-size: 1.25rem; margin-bottom: 0.25rem; }}

        .subtitle {{ color: var(--text-muted); font-size: 0.875rem; }}

        .prompt {{
            background: var(--bg);
            padding: 1.25rem;
            border-radius: 10px;
            margin-bottom: 1.5rem;
            line-height: 1.5;
        }}

        .actions {{
            display: flex;
            gap: 1rem;
        }}

        .btn {{
            flex: 1;
            padding: 1rem;
            border: none;
            border-radius: 10px;
            font-size: 1rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
        }}

        .btn-approve {{ background: var(--success); color: white; }}
        .btn-approve:hover {{ background: #059669; }}

        .btn-reject {{ background: var(--danger); color: white; }}
        .btn-reject:hover {{ background: #dc2626; }}

        .success-state, .error-state {{
            text-align: center;
            padding: 2rem;
        }}

        .success-icon {{ font-size: 3rem; margin-bottom: 1rem; }}
    </style>
</head>
<body>
    <div class="card" id="main-card">
        <div class="header">
            <div class="logo">V</div>
            <h1>Approval Required</h1>
            <div class="subtitle" id="request-id">{request_id}</div>
        </div>
        <div id="content">Loading...</div>
    </div>

    <script>
        const API_BASE = window.location.pathname.replace(/\\/ui\\/request\\/.*$/, '');
        const requestId = '{request_id}';

        async function loadRequest() {{
            try {{
                const response = await fetch(`${{API_BASE}}/requests/${{requestId}}`);
                if (!response.ok) throw new Error('Request not found');

                const req = await response.json();
                renderRequest(req);
            }} catch (error) {{
                document.getElementById('content').innerHTML = `
                    <div class="error-state">
                        <div class="success-icon">❌</div>
                        <p>Request not found or already processed</p>
                    </div>
                `;
            }}
        }}

        function renderRequest(req) {{
            document.getElementById('content').innerHTML = `
                <div class="prompt">${{req.prompt}}</div>
                <div class="actions">
                    <button class="btn btn-reject" onclick="respond(false)">
                        ✕ Reject
                    </button>
                    <button class="btn btn-approve" onclick="respond(true)">
                        ✓ Approve
                    </button>
                </div>
            `;
        }}

        async function respond(approved) {{
            try {{
                const response = await fetch(`${{API_BASE}}/respond/${{requestId}}`, {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ approved }})
                }});

                if (!response.ok) throw new Error('Failed to submit');

                document.getElementById('content').innerHTML = `
                    <div class="success-state">
                        <div class="success-icon">${{approved ? '✅' : '❌'}}</div>
                        <p>${{approved ? 'Approved!' : 'Rejected'}}</p>
                        <p style="color: var(--text-muted); margin-top: 0.5rem;">
                            You can close this window
                        </p>
                    </div>
                `;
            }} catch (error) {{
                alert('Failed to submit response');
            }}
        }}

        loadRequest();
    </script>
</body>
</html>"""


# ============================================================================
# Standalone Server
# ============================================================================


def create_hitl_app(
    store: Optional[HITLStore] = None,
    require_auth: bool = False,
    auth_token: Optional[str] = None,
    title: str = "Victor HITL API",
    allowed_origins: Optional[List[str]] = None,
):
    """Create a standalone FastAPI app for HITL service.

    Args:
        store: HITL store (uses global if not provided)
        require_auth: Whether to require authentication
        auth_token: Bearer token for authentication
        title: API title
        allowed_origins: Explicit list of CORS origins permitted to call the
            API. Defaults to localhost dev origins. A wildcard ``"*"`` is only
            honored when credentials are not required; combining ``"*"`` with
            credentialed requests is insecure and disallowed by browsers.

    Returns:
        FastAPI application
    """
    try:
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError:
        raise ImportError("FastAPI is required for HITL API. Install with: pip install fastapi")

    hitl_store = store or get_global_store()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup: start cleanup task
        cleanup_task = asyncio.create_task(cleanup_loop(hitl_store))
        yield
        # Shutdown: cancel cleanup task
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass

    app = FastAPI(
        title=title,
        description="Human-in-the-Loop API for Victor workflow approvals",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS for web UI. Default to localhost dev origins rather than a wildcard.
    # The wildcard "*" combined with credentials is rejected by browsers and is
    # a security risk, so only enable credentials for an explicit origin list.
    origins = allowed_origins or [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]
    allow_credentials = "*" not in origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include HITL router
    router = create_hitl_router(
        store=hitl_store,
        require_auth=require_auth,
        auth_token=auth_token,
    )
    app.include_router(router, prefix="/hitl")

    @app.get("/")
    async def root():
        return {
            "name": "Victor HITL API",
            "version": "1.0.0",
            "ui": "/hitl/ui",
            "docs": "/docs",
        }

    @app.get("/health")
    async def health():
        pending = await hitl_store.list_pending()
        return {"status": "healthy", "pending_requests": len(pending)}

    return app


async def cleanup_loop(store: HITLStore, interval: float = 60.0):
    """Background task to cleanup expired requests.

    Args:
        store: HITL store
        interval: Cleanup interval in seconds
    """
    while True:
        try:
            await asyncio.sleep(interval)
            removed = await store.cleanup_expired()
            if removed > 0:
                logger.info(f"Cleaned up {removed} expired HITL requests")
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


async def run_hitl_server(
    app=None,
    host: str = "0.0.0.0",
    port: int = 8080,
    **kwargs,
):
    """Run the HITL server.

    Args:
        app: FastAPI app (creates default if not provided)
        host: Host to bind to
        port: Port to bind to
        **kwargs: Additional uvicorn arguments
    """
    try:
        import uvicorn
    except ImportError:
        raise ImportError("uvicorn is required for HITL server. Install with: pip install uvicorn")

    if app is None:
        app = create_hitl_app()

    config = uvicorn.Config(app, host=host, port=port, **kwargs)
    server = uvicorn.Server(config)
    await server.serve()


__all__ = [
    # Store (in-memory)
    "StoredRequest",
    "HITLStore",
    "get_global_store",
    # Store (SQLite persistent)
    "SQLiteHITLStore",
    "get_sqlite_store",
    # Handler
    "APIHITLHandler",
    # Router/App
    "create_hitl_router",
    "create_hitl_app",
    "run_hitl_server",
    # UI HTML
    "get_hitl_ui_html",
    "get_hitl_request_ui_html",
]

"""Injectable session store for the web server (P0-B of the UX action plan).

Replaces the module-level ``SESSION_AGENTS``/``SESSION_TOKENS``/``SESSION_LOCK``
dicts in ``main.py`` with a typed, self-locking store behind a small protocol,
so alternative backends (service-backed, Redis) can drop in without touching
endpoint code.

Design notes (derived from the previous inline implementation's usage sites):

- Sessions are a typed entity (:class:`WebSession`), not stringly-keyed dicts.
- The store owns its lock. Critical sections cover only state mutation —
  callers construct agents *before* ``add()`` and shut them down *after*
  ``pop_idle()``/``pop_all()`` return. The previous implementation held the
  global lock across ``await agent.initialize()`` (one cold init blocked every
  other connect and heartbeat) and across ``await agent.shutdown()`` in the
  cleanup loop; the protocol shape makes that mistake impossible to repeat.
- The session cap is enforced atomically inside ``add()`` — pre-checks via
  ``has_capacity()`` are advisory UX, ``add()`` is the authoritative gate.
- The token index lives with the store so removing a session always revokes
  its token in the same critical section.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable


@dataclass
class WebSession:
    """State for one logical chat session served over the websocket."""

    session_id: str
    agent: Any
    session_token: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    connection_count: int = 0


class SessionLimitReached(RuntimeError):
    """Raised by ``add()`` when the store is at its session cap."""


@runtime_checkable
class SessionStore(Protocol):
    """Minimal async protocol the web endpoints program against."""

    async def get(self, session_id: str) -> Optional[WebSession]: ...

    async def contains(self, session_id: str) -> bool: ...

    async def count(self) -> int: ...

    async def has_capacity(self) -> bool: ...

    async def touch(self, session_id: str) -> None: ...

    async def acquire_connection(self, session_id: str) -> Optional[WebSession]: ...

    async def release_connection(self, session_id: str) -> None: ...

    async def add(self, session: WebSession) -> Tuple[WebSession, bool]: ...

    async def pop_idle(self, idle_timeout: float) -> List[WebSession]: ...

    async def pop_all(self) -> List[WebSession]: ...

    async def bind_token(self, token: str, session_id: str) -> None: ...

    async def revoke_token(self, token: str) -> None: ...


class InMemorySessionStore:
    """Single-process store with the same semantics the inline dicts had."""

    def __init__(self, max_sessions: int) -> None:
        self._max_sessions = max_sessions
        self._sessions: Dict[str, WebSession] = {}
        self._tokens: Dict[str, str] = {}
        self._lock = asyncio.Lock()

    async def get(self, session_id: str) -> Optional[WebSession]:
        async with self._lock:
            return self._sessions.get(session_id)

    async def contains(self, session_id: str) -> bool:
        async with self._lock:
            return session_id in self._sessions

    async def count(self) -> int:
        async with self._lock:
            return len(self._sessions)

    async def has_capacity(self) -> bool:
        async with self._lock:
            return len(self._sessions) < self._max_sessions

    async def touch(self, session_id: str) -> None:
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is not None:
                session.last_activity = time.time()

    async def acquire_connection(self, session_id: str) -> Optional[WebSession]:
        """Atomically touch + increment connection count for an existing session."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            session.last_activity = time.time()
            session.connection_count += 1
            return session

    async def release_connection(self, session_id: str) -> None:
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is not None:
                session.connection_count = max(0, session.connection_count - 1)

    async def add(self, session: WebSession) -> Tuple[WebSession, bool]:
        """Insert a new session, enforcing the cap atomically.

        Returns ``(session, True)`` on insert, or ``(existing, False)`` if a
        concurrent connect already created this session id (the caller should
        discard its duplicate agent). Raises :class:`SessionLimitReached` at cap.
        """
        async with self._lock:
            existing = self._sessions.get(session.session_id)
            if existing is not None:
                return existing, False
            if len(self._sessions) >= self._max_sessions:
                raise SessionLimitReached(f"Session limit reached ({self._max_sessions})")
            self._sessions[session.session_id] = session
            if session.session_token:
                self._tokens[session.session_token] = session.session_id
            return session, True

    async def pop_idle(self, idle_timeout: float) -> List[WebSession]:
        """Remove and return sessions idle longer than ``idle_timeout`` seconds.

        Token revocation happens in the same critical section; agent shutdown
        is the caller's job, outside the lock.
        """
        now = time.time()
        async with self._lock:
            expired = [
                session
                for session in self._sessions.values()
                if now - session.last_activity > idle_timeout
            ]
            for session in expired:
                del self._sessions[session.session_id]
                if session.session_token:
                    self._tokens.pop(session.session_token, None)
            return expired

    async def pop_all(self) -> List[WebSession]:
        async with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
            self._tokens.clear()
            return sessions

    async def bind_token(self, token: str, session_id: str) -> None:
        async with self._lock:
            self._tokens[token] = session_id

    async def revoke_token(self, token: str) -> None:
        async with self._lock:
            self._tokens.pop(token, None)


_session_store: Optional[SessionStore] = None


def get_session_store() -> SessionStore:
    """Return the process-wide session store (created lazily)."""
    global _session_store
    if _session_store is None:
        raise RuntimeError("Session store not configured — call set_session_store() at startup")
    return _session_store


def set_session_store(store: SessionStore) -> SessionStore:
    """Install the session store. Single mutation point for tests/backends."""
    global _session_store
    _session_store = store
    return store

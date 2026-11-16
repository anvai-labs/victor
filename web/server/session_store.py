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
- Scoped admission (ADR-034 / VAS-06b) reserves capacity before agent
  construction. A reservation is a time-bounded capability: only the exact
  immutable object returned by the store can advance RESERVED → INITIALIZING →
  ACTIVE. Scope keys hide a known session ID from another principal or
  workspace, while connection ownership and the single turn lease remain
  independent. Process-local scoped token indexes also record the store epoch;
  signing and the wire envelope remain a later VAS-06d concern.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, runtime_checkable


@dataclass(frozen=True, slots=True)
class SessionScope:
    """Server-owned identity for one in-process chat session."""

    principal_id: str
    workspace_id: str
    session_id: str


class SessionState(Enum):
    """Admission states used by the scoped in-memory store."""

    RESERVED = "reserved"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    DRAINING = "draining"
    REMOVED = "removed"


@dataclass(frozen=True, slots=True)
class SessionReservation:
    """A capacity reservation created before an agent is constructed."""

    reservation_id: str
    scope: SessionScope
    state: SessionState
    lease_expires_at: float


@dataclass(frozen=True, slots=True)
class SessionTurnLease:
    """The single admitted turn for one scoped session."""

    scope: SessionScope
    turn_id: str


@dataclass(frozen=True, slots=True)
class ScopedSessionToken:
    """Process-local token index fields; signing and wire format remain VAS-06d."""

    token: str
    token_nonce: str
    scope: SessionScope
    store_epoch: str
    issued_at: float
    expires_at: float


@dataclass
class WebSession:
    """State for one logical chat session served over the websocket."""

    session_id: str
    agent: Any
    session_token: Optional[str] = None
    scope: Optional[SessionScope] = None
    state: SessionState = SessionState.ACTIVE
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    connection_count: int = 0


class SessionLimitReached(RuntimeError):
    """Raised by ``add()`` when the store is at its session cap."""


class PrincipalSessionLimitReached(RuntimeError):
    """Raised when one principal already holds the configured session budget."""


class SessionReservationExpired(RuntimeError):
    """Raised when initialization is attempted after its bounded lease expired."""


class SessionReservationInvalid(RuntimeError):
    """Raised when a reservation token does not match current server state."""


class SessionConnectionLimitReached(RuntimeError):
    """Raised when a session already holds its configured connection bound."""


class SessionDraining(RuntimeError):
    """Raised when a draining session cannot admit more work or connections."""


class SessionTurnConflict(RuntimeError):
    """Raised when a session already owns an active turn."""


class SessionTokenRejected(RuntimeError):
    """Raised when a process-local scoped token is invalid or no longer admitted."""


@runtime_checkable
class SessionStore(Protocol):
    """Minimal async protocol the web endpoints program against."""

    async def get(self, session_id: str) -> Optional[WebSession]: ...

    async def contains(self, session_id: str) -> bool: ...

    async def count(self) -> int: ...

    async def has_capacity(self, principal_id: Optional[str] = None) -> bool: ...

    async def touch(self, session_id: str) -> None: ...

    async def acquire_connection(self, session_id: str) -> Optional[WebSession]: ...

    async def release_connection(self, session_id: str) -> None: ...

    async def add(self, session: WebSession) -> Tuple[WebSession, bool]: ...

    async def pop_idle(self, idle_timeout: float) -> List[WebSession]: ...

    async def pop_all(self) -> List[WebSession]: ...

    async def bind_token(self, token: str, session_id: str) -> None: ...

    async def revoke_token(self, token: str) -> None: ...

    async def reserve_scope(self, principal_id: str, workspace_id: str) -> SessionReservation: ...

    async def begin_initialization(self, reservation: SessionReservation) -> SessionReservation: ...

    async def complete_initialization(
        self, reservation: SessionReservation, agent: Any
    ) -> WebSession: ...

    async def fail_initialization(self, reservation: SessionReservation) -> None: ...

    async def get_scoped(self, scope: SessionScope) -> Optional[WebSession]: ...

    async def contains_scoped(self, scope: SessionScope) -> bool: ...

    async def touch_scoped(self, scope: SessionScope) -> None: ...

    async def acquire_scoped_connection(self, scope: SessionScope) -> Optional[WebSession]: ...

    async def release_scoped_connection(self, scope: SessionScope) -> None: ...

    async def acquire_turn(self, scope: SessionScope, turn_id: str) -> SessionTurnLease: ...

    async def release_turn(self, scope: SessionScope, turn_id: str) -> None: ...

    async def begin_drain(self, scope: SessionScope) -> WebSession: ...

    async def pop_scope(self, scope: SessionScope) -> Optional[WebSession]: ...

    async def bind_scoped_token(
        self,
        scope: SessionScope,
        token: str,
        token_nonce: str,
        issued_at: float,
        expires_at: float,
    ) -> ScopedSessionToken: ...

    async def resolve_scoped_token(
        self, token: ScopedSessionToken, expected_scope: SessionScope
    ) -> WebSession: ...

    async def revoke_scoped_token(self, token: str) -> None: ...


class InMemorySessionStore:
    """Single-process store with the same semantics the inline dicts had."""

    def __init__(
        self,
        max_sessions: int,
        *,
        max_sessions_per_principal: Optional[int] = None,
        max_connections_per_session: Optional[int] = None,
        reservation_lease_seconds: float = 30.0,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        if max_sessions < 1:
            raise ValueError("max_sessions must be positive")
        if max_sessions_per_principal is not None and max_sessions_per_principal < 1:
            raise ValueError("max_sessions_per_principal must be positive")
        if max_connections_per_session is not None and max_connections_per_session < 1:
            raise ValueError("max_connections_per_session must be positive")
        if reservation_lease_seconds <= 0:
            raise ValueError("reservation_lease_seconds must be positive")

        self._max_sessions = max_sessions
        self._max_sessions_per_principal = max_sessions_per_principal
        self._max_connections_per_session = max_connections_per_session
        self._reservation_lease_seconds = reservation_lease_seconds
        self._clock = clock or time.monotonic
        self.store_epoch = uuid.uuid4().hex
        self._sessions: Dict[str, WebSession] = {}
        self._scoped_sessions: Dict[SessionScope, WebSession] = {}
        self._reservations: Dict[str, SessionReservation] = {}
        self._turn_leases: Dict[SessionScope, str] = {}
        self._scoped_tokens: Dict[str, ScopedSessionToken] = {}
        self._token_nonces: Dict[Tuple[SessionScope, str], str] = {}
        self._scope_tokens: Dict[SessionScope, set[str]] = {}
        self._tokens: Dict[str, str] = {}
        self._lock = asyncio.Lock()

    def _active_count(self) -> int:
        return len(self._scoped_sessions)

    def _principal_count(self, principal_id: str) -> int:
        return sum(
            1 for scope in self._scoped_sessions if scope.principal_id == principal_id
        ) + sum(
            1
            for reservation in self._reservations.values()
            if reservation.scope.principal_id == principal_id
        )

    def _expire_reservations(self) -> None:
        now = self._clock()
        expired = [
            reservation_id
            for reservation_id, reservation in self._reservations.items()
            if now >= reservation.lease_expires_at
        ]
        for reservation_id in expired:
            self._reservations.pop(reservation_id, None)

    def _assert_capacity(self, principal_id: str) -> None:
        if self._active_count() + len(self._reservations) >= self._max_sessions:
            raise SessionLimitReached(f"Session limit reached ({self._max_sessions})")
        if (
            self._max_sessions_per_principal is not None
            and self._principal_count(principal_id) >= self._max_sessions_per_principal
        ):
            raise PrincipalSessionLimitReached(
                f"Principal session limit reached ({self._max_sessions_per_principal})"
            )

    def _new_session_id(self) -> str:
        while True:
            session_id = uuid.uuid4().hex
            if session_id not in self._sessions and session_id not in self._reservations:
                return session_id

    def _admitted_reservation(
        self, reservation: SessionReservation, expected_state: SessionState
    ) -> SessionReservation:
        stored = self._reservations.get(reservation.reservation_id)
        if stored != reservation or stored.state is not expected_state:
            raise SessionReservationInvalid(
                f"Reservation {reservation.reservation_id} is no longer admitted"
            )
        if self._clock() >= stored.lease_expires_at:
            self._reservations.pop(stored.reservation_id, None)
            raise SessionReservationExpired(f"Reservation {stored.reservation_id} expired")
        return stored

    def _remove_session_locked(self, session: WebSession) -> None:
        scope = session.scope
        if scope is None:
            self._sessions.pop(session.session_id, None)
        else:
            self._scoped_sessions.pop(scope, None)
            if self._sessions.get(session.session_id) is session:
                self._sessions.pop(session.session_id, None)
            self._turn_leases.pop(scope, None)
            for token in self._scope_tokens.pop(scope, set()):
                record = self._scoped_tokens.pop(token, None)
                if record is not None:
                    self._token_nonces.pop((record.scope, record.token_nonce), None)
        if session.session_token:
            self._tokens.pop(session.session_token, None)
        session.state = SessionState.REMOVED

    async def get(self, session_id: str) -> Optional[WebSession]:
        async with self._lock:
            return self._sessions.get(session_id)

    async def contains(self, session_id: str) -> bool:
        async with self._lock:
            return session_id in self._sessions

    async def count(self) -> int:
        async with self._lock:
            self._expire_reservations()
            return self._active_count() + len(self._reservations)

    async def has_capacity(self, principal_id: Optional[str] = None) -> bool:
        async with self._lock:
            self._expire_reservations()
            if self._active_count() + len(self._reservations) >= self._max_sessions:
                return False
            if principal_id is not None:
                limit = self._max_sessions_per_principal
                return limit is None or self._principal_count(principal_id) < limit
            return True

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
            if session.state is SessionState.DRAINING:
                raise SessionDraining(f"Session {session_id} is draining")
            if (
                self._max_connections_per_session is not None
                and session.connection_count >= self._max_connections_per_session
            ):
                raise SessionConnectionLimitReached(
                    f"Session connection limit reached ({self._max_connections_per_session})"
                )
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
            self._expire_reservations()
            existing = self._sessions.get(session.session_id)
            if existing is not None:
                return existing, False
            principal = session.scope.principal_id if session.scope is not None else "legacy:shared"
            self._assert_capacity(principal)
            if session.scope is None:
                session.scope = SessionScope(
                    principal_id="legacy:shared",
                    workspace_id="legacy:default",
                    session_id=session.session_id,
                )
                self._scoped_sessions[session.scope] = session
                self._sessions[session.session_id] = session
            else:
                self._scoped_sessions[session.scope] = session
                self._sessions[session.session_id] = session
            session.state = SessionState.ACTIVE
            if session.session_token:
                self._tokens[session.session_token] = session.session_id
            return session, True

    async def reserve_scope(self, principal_id: str, workspace_id: str) -> SessionReservation:
        if not principal_id:
            raise ValueError("principal_id must not be empty")
        if not workspace_id:
            raise ValueError("workspace_id must not be empty")

        async with self._lock:
            self._expire_reservations()
            self._assert_capacity(principal_id)
            session_id = self._new_session_id()
            scope = SessionScope(principal_id, workspace_id, session_id)
            reservation = SessionReservation(
                reservation_id=session_id,
                scope=scope,
                state=SessionState.RESERVED,
                lease_expires_at=self._clock() + self._reservation_lease_seconds,
            )
            self._reservations[reservation.reservation_id] = reservation
            return reservation

    async def begin_initialization(self, reservation: SessionReservation) -> SessionReservation:
        async with self._lock:
            stored = self._admitted_reservation(reservation, SessionState.RESERVED)
            initializing = SessionReservation(
                reservation_id=stored.reservation_id,
                scope=stored.scope,
                state=SessionState.INITIALIZING,
                lease_expires_at=stored.lease_expires_at,
            )
            self._reservations[initializing.reservation_id] = initializing
            return initializing

    async def complete_initialization(
        self, reservation: SessionReservation, agent: Any
    ) -> WebSession:
        async with self._lock:
            stored = self._admitted_reservation(reservation, SessionState.INITIALIZING)

            session = WebSession(
                session_id=stored.scope.session_id,
                agent=agent,
                scope=stored.scope,
                state=SessionState.ACTIVE,
            )
            self._reservations.pop(stored.reservation_id, None)
            self._scoped_sessions[stored.scope] = session
            self._sessions[session.session_id] = session
            return session

    async def fail_initialization(self, reservation: SessionReservation) -> None:
        async with self._lock:
            self._admitted_reservation(reservation, SessionState.INITIALIZING)
            self._reservations.pop(reservation.reservation_id, None)

    async def get_scoped(self, scope: SessionScope) -> Optional[WebSession]:
        async with self._lock:
            self._expire_reservations()
            return self._scoped_sessions.get(scope)

    async def contains_scoped(self, scope: SessionScope) -> bool:
        async with self._lock:
            self._expire_reservations()
            return scope in self._scoped_sessions

    async def touch_scoped(self, scope: SessionScope) -> None:
        async with self._lock:
            session = self._scoped_sessions.get(scope)
            if session is not None:
                session.last_activity = time.time()

    async def acquire_scoped_connection(self, scope: SessionScope) -> Optional[WebSession]:
        async with self._lock:
            session = self._scoped_sessions.get(scope)
            if session is None:
                return None
            if session.state is SessionState.DRAINING:
                raise SessionDraining(f"Session {scope.session_id} is draining")
            if (
                self._max_connections_per_session is not None
                and session.connection_count >= self._max_connections_per_session
            ):
                raise SessionConnectionLimitReached(
                    f"Session connection limit reached ({self._max_connections_per_session})"
                )
            session.last_activity = time.time()
            session.connection_count += 1
            return session

    async def release_scoped_connection(self, scope: SessionScope) -> None:
        async with self._lock:
            session = self._scoped_sessions.get(scope)
            if session is not None:
                session.connection_count = max(0, session.connection_count - 1)

    async def acquire_turn(self, scope: SessionScope, turn_id: str) -> SessionTurnLease:
        if not turn_id:
            raise ValueError("turn_id must not be empty")
        async with self._lock:
            session = self._scoped_sessions.get(scope)
            if session is None:
                raise SessionTurnConflict(f"Session {scope.session_id} was not found")
            if session.state is SessionState.DRAINING:
                raise SessionDraining(f"Session {scope.session_id} is draining")
            if scope in self._turn_leases:
                raise SessionTurnConflict(f"Session {scope.session_id} already has a turn")
            self._turn_leases[scope] = turn_id
            return SessionTurnLease(scope=scope, turn_id=turn_id)

    async def release_turn(self, scope: SessionScope, turn_id: str) -> None:
        async with self._lock:
            if self._turn_leases.get(scope) == turn_id:
                self._turn_leases.pop(scope, None)

    async def begin_drain(self, scope: SessionScope) -> WebSession:
        async with self._lock:
            session = self._scoped_sessions.get(scope)
            if session is None:
                raise SessionDraining(f"Session {scope.session_id} was not found")
            session.state = SessionState.DRAINING
            return session

    async def pop_scope(self, scope: SessionScope) -> Optional[WebSession]:
        async with self._lock:
            session = self._scoped_sessions.get(scope)
            if session is None:
                return None
            self._remove_session_locked(session)
            return session

    async def pop_idle(self, idle_timeout: float) -> List[WebSession]:
        """Remove and return sessions idle longer than ``idle_timeout`` seconds.

        Token revocation happens in the same critical section; agent shutdown
        is the caller's job, outside the lock.
        """
        now = time.time()
        async with self._lock:
            self._expire_reservations()
            expired = [
                session
                for session in self._scoped_sessions.values()
                if now - session.last_activity > idle_timeout
            ]
            for session in expired:
                self._remove_session_locked(session)
            return expired

    async def pop_all(self) -> List[WebSession]:
        async with self._lock:
            sessions = list(self._scoped_sessions.values())
            for session in sessions:
                self._remove_session_locked(session)
            self._reservations.clear()
            self._tokens.clear()
            self._scoped_tokens.clear()
            self._token_nonces.clear()
            self._scope_tokens.clear()
            self._turn_leases.clear()
            return sessions

    async def bind_token(self, token: str, session_id: str) -> None:
        async with self._lock:
            self._tokens[token] = session_id

    async def revoke_token(self, token: str) -> None:
        async with self._lock:
            self._tokens.pop(token, None)

    async def bind_scoped_token(
        self,
        scope: SessionScope,
        token: str,
        token_nonce: str,
        issued_at: float,
        expires_at: float,
    ) -> ScopedSessionToken:
        if not token or not token_nonce:
            raise ValueError("token and token_nonce must not be empty")
        if expires_at <= issued_at:
            raise ValueError("expires_at must be later than issued_at")

        async with self._lock:
            if scope not in self._scoped_sessions:
                raise SessionTokenRejected("session token scope mismatch")
            if token in self._scoped_tokens:
                raise ValueError("token is already bound")
            if (scope, token_nonce) in self._token_nonces:
                raise ValueError("token_nonce is already bound")
            record = ScopedSessionToken(
                token=token,
                token_nonce=token_nonce,
                scope=scope,
                store_epoch=self.store_epoch,
                issued_at=issued_at,
                expires_at=expires_at,
            )
            self._scoped_tokens[token] = record
            self._token_nonces[(scope, token_nonce)] = token
            self._scope_tokens.setdefault(scope, set()).add(token)
            return record

    async def resolve_scoped_token(
        self, token: ScopedSessionToken, expected_scope: SessionScope
    ) -> WebSession:
        async with self._lock:
            self._expire_reservations()
            if token.store_epoch != self.store_epoch:
                raise SessionTokenRejected("session token store epoch mismatch")
            if token.scope != expected_scope:
                raise SessionTokenRejected("session token scope mismatch")
            if self._clock() >= token.expires_at:
                raise SessionTokenRejected("session token expired")
            indexed = self._scoped_tokens.get(token.token)
            if (
                indexed is None
                or indexed != token
                or self._token_nonces.get((token.scope, token.token_nonce)) != token.token
            ):
                raise SessionTokenRejected("session token revoked")
            session = self._scoped_sessions.get(token.scope)
            if session is None:
                raise SessionTokenRejected("session token revoked")
            return session

    async def revoke_scoped_token(self, token: str) -> None:
        async with self._lock:
            record = self._scoped_tokens.pop(token, None)
            if record is None:
                return
            self._token_nonces.pop((record.scope, record.token_nonce), None)
            tokens = self._scope_tokens.get(record.scope)
            if tokens is not None:
                tokens.discard(token)
                if not tokens:
                    self._scope_tokens.pop(record.scope, None)


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

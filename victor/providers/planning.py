"""Request-scoped correlation for applications that embed Victor providers."""

import re
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_planning_session_id: ContextVar[str] = ContextVar("planning_session_id", default="")


@contextmanager
def planning_session(session_id: str) -> Iterator[None]:
    """Scope provider calls to an opaque planning session, including async calls.

    Reuse the identifier for one materialized conversation. Use a different
    identifier for an independent job. Keep the scope open while consuming a
    stream. The identifier is correlation metadata, never an authentication
    credential.
    """
    if not isinstance(session_id, str) or not re.fullmatch(r"[A-Za-z0-9._-]{1,128}", session_id):
        raise ValueError(
            "planning session ID must contain 1–128 ASCII letters, digits, dots, underscores or hyphens"
        )
    token = _planning_session_id.set(session_id)
    try:
        yield
    finally:
        _planning_session_id.reset(token)

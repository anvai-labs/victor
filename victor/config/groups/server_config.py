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

"""Server configuration settings.

Extracted from victor/config/settings.py to improve maintainability.
Contains configuration for API server, WebSocket, and diagram rendering.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator


class ServerSettings(BaseModel):
    """Server configuration for API and WebSocket endpoints.

    Includes FastAPI/WebSocket security, session management,
    and diagram rendering limits.

    Disposition of every advertised control (Codex finding 22 — the
    effective-posture surface on the server reports the enforced state of
    each at startup and via ``/status``):

    - ``server_api_key`` — ENFORCED: merged into the FastAPI server's
      credential map at startup (its actual owner is the FastAPI auth
      gate) AND honored by the web UI server (``web/server/main.py``).
    - ``server_session_secret`` — ENFORCED by the web UI serving path
      (``web/server/main.py``): HMAC session-token signing/verification.
      The validator AUTO-GENERATES a per-process secret when unconfigured
      (dev/test convenience; ``server_session_secret_auto_generated``
      records that it happened) — a multi-worker PRODUCTION deployment
      MUST set it explicitly, or session tokens will not verify across
      processes; the web server warns loudly and the effective posture
      reports the source at startup.
    - ``server_max_sessions`` / ``server_session_ttl_seconds`` — ENFORCED
      by the web UI serving path: the in-memory session store cap and the
      session-token expiry.
    - ``server_max_message_bytes`` — ENFORCED: inbound WebSocket payload
      cap applied by the ``/ws`` and ``/ws/events`` routes (FastAPI) and
      the web UI server.
    - ``server_ws_delivery_queue_size`` / ``server_ws_send_deadline_seconds``
      — ENFORCED: bounded per-client delivery (VAS-06/10, TD-03).
    """

    # Server Security (FastAPI/WebSocket layer)
    # When set, API key is required for HTTP + WebSocket requests (Authorization: Bearer <token>)
    server_api_key: Optional[SecretStr] = None
    # HMAC secret for session-token signing in the web UI serving path.
    # Auto-generated per process when unconfigured (dev/test); production
    # multi-worker deployments must set it explicitly (see class docstring).
    server_session_secret: Optional[SecretStr] = Field(default=None, validate_default=True)
    # True only when the validator generated the session secret above —
    # the honest marker that a real deployment is NOT using an operator-
    # supplied secret (surfaced by the web server warning + posture).
    server_session_secret_auto_generated: bool = False
    # Hard cap on simultaneous sessions (web UI session store).
    server_max_sessions: int = 100
    # Maximum inbound message payload size (bytes) for WebSocket messages
    server_max_message_bytes: int = 32768
    # Session token time-to-live in seconds (web UI session tokens).
    server_session_ttl_seconds: int = 86400

    # Diagram rendering limits
    render_max_payload_bytes: int = 20000
    render_timeout_seconds: int = 10
    render_max_concurrency: int = 2

    # Bounded per-client WebSocket delivery (VAS-06/10, TD-03): each /ws
    # socket gets an outbound queue of this depth drained by its own sender
    # task, so one slow socket cannot block the broadcast loop for every
    # recipient. Overflow sheds the oldest nonessential progress event;
    # terminal/approval events are never dropped (the slow client is
    # disconnected and the gap recorded instead).
    server_ws_delivery_queue_size: int = 256
    # Per-send deadline for a client's sender task: a socket wedged past
    # this bound is disconnected instead of stalling delivery forever.
    server_ws_send_deadline_seconds: float = 10.0

    @field_validator("server_max_sessions")
    @classmethod
    def validate_max_sessions(cls, v: int) -> int:
        """Validate max sessions is positive.

        Args:
            v: Max sessions value

        Returns:
            Validated max sessions

        Raises:
            ValueError: If max sessions is not positive
        """
        if v <= 0:
            raise ValueError("server_max_sessions must be positive")
        return v

    @field_validator("server_max_message_bytes")
    @classmethod
    def validate_max_message_bytes(cls, v: int) -> int:
        """Validate max message bytes is positive.

        Args:
            v: Max message bytes

        Returns:
            Validated max message bytes

        Raises:
            ValueError: If max message bytes is not positive
        """
        if v <= 0:
            raise ValueError("server_max_message_bytes must be positive")
        return v

    @field_validator("server_session_ttl_seconds")
    @classmethod
    def validate_session_ttl(cls, v: int) -> int:
        """Validate session TTL is positive.

        Args:
            v: Session TTL in seconds

        Returns:
            Validated session TTL

        Raises:
            ValueError: If session TTL is not positive
        """
        if v <= 0:
            raise ValueError("server_session_ttl_seconds must be positive")
        return v

    @field_validator("server_ws_delivery_queue_size")
    @classmethod
    def validate_ws_delivery_queue_size(cls, v: int) -> int:
        """Validate the per-client delivery queue depth is positive.

        Args:
            v: Queue size value

        Returns:
            Validated queue size

        Raises:
            ValueError: If queue size is not positive
        """
        if v <= 0:
            raise ValueError("server_ws_delivery_queue_size must be positive")
        return v

    @field_validator("server_ws_send_deadline_seconds")
    @classmethod
    def validate_ws_send_deadline(cls, v: float) -> float:
        """Validate the per-send deadline is positive.

        Args:
            v: Deadline value in seconds

        Returns:
            Validated deadline

        Raises:
            ValueError: If deadline is not positive
        """
        if v <= 0:
            raise ValueError("server_ws_send_deadline_seconds must be positive")
        return v

    @model_validator(mode="before")
    @classmethod
    def _autogenerate_session_secret(cls, data: Any) -> Any:
        """Auto-generate a per-process session secret when unconfigured.

        This is the DEV/TEST fallback that keeps single-process runs (and
        the web UI server's fail-fast) working without configuration. It
        is deliberately VISIBLE: the generation is recorded on
        ``server_session_secret_auto_generated`` so the web server can
        warn and the effective posture can report that an operator has
        NOT supplied the secret. A multi-worker production deployment
        MUST set ``server.server_session_secret`` explicitly — per-
        process secrets cannot verify tokens issued by another worker.
        """
        if isinstance(data, dict) and data.get("server_session_secret") is None:
            import secrets as _secrets

            data["server_session_secret"] = SecretStr(_secrets.token_urlsafe(32))
            data["server_session_secret_auto_generated"] = True
        return data

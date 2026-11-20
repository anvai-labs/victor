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

from typing import Optional

from pydantic import BaseModel, SecretStr, field_validator


class ServerSettings(BaseModel):
    """Server configuration for API and WebSocket endpoints.

    Includes FastAPI/WebSocket security, session management,
    and diagram rendering limits.

    Disposition of every advertised control (Codex finding 22 — the
    effective-posture surface on the server reports the enforced state of
    each at startup and via ``/status``):

    - ``server_api_key`` — ENFORCED: merged into the server's credential
      map at startup (its actual owner is the FastAPI auth gate).
    - ``server_max_message_bytes`` — ENFORCED: inbound WebSocket payload
      cap applied by the ``/ws`` and ``/ws/events`` routes.
    - ``server_ws_delivery_queue_size`` / ``server_ws_send_deadline_seconds``
      — ENFORCED: bounded per-client delivery (VAS-06/10, TD-03).
    - ``server_session_secret`` — DECLARED, NOT ENFORCED (deprecated):
      aiohttp-era session-token layer; no serving-path consumer. The
      former auto-generation validator was REMOVED — it fabricated a
      secret nothing read, which made the dead control look alive. Kept
      declared so existing configs keep loading; scheduled for retirement
      with the VAS-06 session contract.
    - ``server_max_sessions`` / ``server_session_ttl_seconds`` — DECLARED,
      NOT ENFORCED (deprecated): same aiohttp-era session-token layer; no
      session registry or expiry consumer exists. Kept declared for config
      compatibility; scheduled for retirement with VAS-06.
    """

    # Server Security (FastAPI/WebSocket layer)
    # When set, API key is required for HTTP + WebSocket requests (Authorization: Bearer <token>)
    server_api_key: Optional[SecretStr] = None
    # DEPRECATED, not enforced: aiohttp-era session-token HMAC secret with
    # no serving-path consumer (see class docstring). Defaults to None —
    # the previous validator fabricated an unused secret per process.
    server_session_secret: Optional[SecretStr] = None
    # DEPRECATED, not enforced: aiohttp-era session cap with no consumer.
    server_max_sessions: int = 100
    # Maximum inbound message payload size (bytes) for WebSocket messages
    server_max_message_bytes: int = 32768
    # DEPRECATED, not enforced: aiohttp-era session-token TTL with no consumer.
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

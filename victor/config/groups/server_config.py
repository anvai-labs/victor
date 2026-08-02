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

from pydantic import BaseModel, Field, SecretStr, field_validator


class ServerSettings(BaseModel):
    """Server configuration for API and WebSocket endpoints.

    Includes FastAPI/WebSocket security, session management,
    and diagram rendering limits.
    """

    # Server Security (FastAPI/WebSocket layer)
    # When set, API key is required for HTTP + WebSocket requests (Authorization: Bearer <token>)
    server_api_key: Optional[SecretStr] = None
    # HMAC secret for issuing/verifying session tokens (defaults to random per-process secret)
    server_session_secret: Optional[SecretStr] = Field(default=None, validate_default=True)
    # Hard cap on simultaneous sessions to avoid resource exhaustion
    server_max_sessions: int = 100
    # Maximum inbound message payload size (bytes) for WebSocket messages
    server_max_message_bytes: int = 32768
    # Session token time-to-live in seconds
    server_session_ttl_seconds: int = 86400

    # Diagram rendering limits
    render_max_payload_bytes: int = 20000
    render_timeout_seconds: int = 10
    render_max_concurrency: int = 2

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

    @field_validator("server_session_secret", mode="before")
    @classmethod
    def _autogenerate_session_secret(
        cls,
        v: Optional[SecretStr],
    ) -> Optional[SecretStr]:
        """Auto-generate a cryptographically secure session secret when None.

        Args:
            v: Session secret value (may be None)

        Returns:
            SecretStr with generated secret if input was None, otherwise original value

        """
        if v is None:
            import secrets as _secrets

            return SecretStr(_secrets.token_urlsafe(32))
        return v

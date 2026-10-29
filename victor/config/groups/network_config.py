"""Network and HTTP configuration for tool execution.

This module contains settings for:
- Tool retry logic
- HTTP connection pooling
- Request timeouts and retries
"""

from pydantic import BaseModel, field_validator


class NetworkSettings(BaseModel):
    """Network and HTTP settings for tool execution.

    Controls retry logic, connection pooling, and timeout behavior
    for tool execution and HTTP requests.
    """

    # ==========================================================================
    # Tool Retry Settings
    # ==========================================================================
    # Enable automatic retry for failed tool executions with exponential backoff.
    # Helps recover from transient network failures and temporary unavailability.
    tool_retry_enabled: bool = True
    tool_retry_max_attempts: int = 3  # Maximum retry attempts per tool call
    tool_retry_base_delay: float = 1.0  # Base delay in seconds for exponential backoff
    tool_retry_max_delay: float = 10.0  # Maximum delay in seconds between retries

    @field_validator("tool_retry_max_attempts")
    @classmethod
    def validate_max_attempts(cls, v: int) -> int:
        """Validate max retry attempts is positive.

        Args:
            v: Max attempts

        Returns:
            Validated max attempts

        Raises:
            ValueError: If max attempts is not positive
        """
        if v < 1:
            raise ValueError("tool_retry_max_attempts must be >= 1")
        return v

    @field_validator("tool_retry_base_delay")
    @classmethod
    def validate_base_delay(cls, v: float) -> int:
        """Validate base delay is non-negative.

        Args:
            v: Base delay in seconds

        Returns:
            Validated base delay

        Raises:
            ValueError: If base delay is negative
        """
        if v < 0:
            raise ValueError("tool_retry_base_delay must be >= 0")
        return v

    @field_validator("tool_retry_max_delay")
    @classmethod
    def validate_max_delay(cls, v: float) -> int:
        """Validate max delay is non-negative and >= base delay.

        Args:
            v: Max delay in seconds

        Returns:
            Validated max delay

        Raises:
            ValueError: If max delay is negative
        """
        if v < 0:
            raise ValueError("tool_retry_max_delay must be >= 0")
        return v

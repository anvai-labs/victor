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

"""
Structured logging for Victor provider operations.

This module provides structured logging events for:
- Provider initialization
- API key resolution
- API calls (start, success, error)
- Performance metrics

All events are logged with structured extra data for JSON logging
and easy parsing by log aggregators.

Usage:
    from victor.providers.logging import ProviderLogger

    logger = ProviderLogger("deepseek", __name__)
    logger.log_provider_init(
        model="deepseek-chat",
        key_source="DEEPSEEK_API_KEY environment variable",
        non_interactive=True,
        config={"timeout": 120},
    )
"""

import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional

from victor.providers.resolution import APIKeyResult

logger = logging.getLogger(__name__)


class ProviderLogger:
    """
    Structured logger for provider operations.

    Provides consistent logging events across all providers:
    - PROVIDER_INIT: Provider initialization
    - API_KEY_RESOLUTION: API key resolution attempt
    - API_CALL_START: API call started
    - API_CALL_SUCCESS: API call succeeded
    - API_CALL_ERROR: API call failed

    All events include structured extra data for JSON logging.
    """

    def __init__(self, provider_name: str, logger_name: str):
        """
        Initialize provider logger.

        Args:
            provider_name: Provider name (e.g., "deepseek", "anthropic")
            logger_name: Logger name (typically __name__ of calling module)
        """
        self.provider = provider_name
        self.logger = logging.getLogger(logger_name)
        self._initialized = False

    def _initialize_structured_logging(self):
        """Configure logger for structured output."""
        if self._initialized:
            return

        # Add a structured formatter if not already present
        # This is a no-op in production - actual config done in logging config
        self._initialized = True

    def log_provider_init(
        self,
        model: str,
        key_source: Optional[str],
        non_interactive: bool,
        config: Dict[str, Any],
    ):
        """
        Log provider initialization.

        Args:
            model: Model identifier
            key_source: Where API key came from (or None)
            non_interactive: Whether running in non-interactive mode
            config: Provider configuration (sanitized)
        """
        self.logger.info(
            f"PROVIDER_INIT provider={self.provider} model={model}",
            extra={
                "event": "PROVIDER_INIT",
                "provider": self.provider,
                "model": model,
                "key_source": key_source,
                "non_interactive": non_interactive,
                "config": self._sanitize_config(config),
            },
        )

    def log_api_key_resolution(
        self,
        result: APIKeyResult,
        latency_ms: Optional[float] = None,
    ):
        """
        Log API key resolution result.

        Args:
            result: API key resolution result
            latency_ms: Time taken to resolve key (optional)
        """
        sources_summary = [
            {
                "source": s.source,
                "description": s.description,
                "found": s.found,
            }
            for s in result.sources_attempted
        ]

        self.logger.info(
            f"API_KEY_RESOLUTION provider={self.provider} source={result.source} success={result.key is not None}",
            extra={
                "event": "API_KEY_RESOLUTION",
                "provider": self.provider,
                "source": result.source,
                "source_detail": result.source_detail,
                "success": result.key is not None,
                "confidence": result.confidence,
                "non_interactive": result.non_interactive,
                "latency_ms": latency_ms,
                "sources_attempted": sources_summary,
            },
        )

    @contextmanager
    def log_api_call(
        self,
        endpoint: str,
        model: str,
        operation: str = "chat",
        **extra_context: Any,
    ):
        """
        Context manager for logging API call lifecycle.

        Automatically logs API_CALL_SUCCESS on normal exit. Providers can
        optionally call the yielded callback to attach token counts to
        the success log — if the callback is invoked, it replaces the
        automatic log so there is never a duplicate.

        Args:
            endpoint: API endpoint being called
            model: Model being used
            operation: Operation type (chat, stream, etc.)
            **extra_context: Additional context to log

        Yields:
            Function to call with success details (tokens, etc.)
        """
        start_time = time.time()
        call_id = f"{self.provider}_{model}_{int(start_time * 1000)}"
        # Track whether the provider explicitly called the callback
        _logged = {"done": False}

        self.logger.info(
            f"API_CALL_START provider={self.provider} model={model} operation={operation}",
            extra={
                "event": "API_CALL_START",
                "call_id": call_id,
                "provider": self.provider,
                "model": model,
                "operation": operation,
                "endpoint": endpoint,
                **extra_context,
            },
        )

        def _log_success(**kwargs: Any) -> None:
            _logged["done"] = True
            self._log_api_call_success(call_id, endpoint, model, start_time, **kwargs)

        try:
            yield _log_success
        except Exception as e:
            self._log_api_call_error(call_id, endpoint, model, start_time, e, **extra_context)
            raise
        else:
            # Auto-log success if the provider didn't call the callback
            if not _logged["done"]:
                self._log_api_call_success(
                    call_id,
                    endpoint,
                    model,
                    start_time,
                )

    def _log_api_call_success(
        self,
        call_id: str,
        endpoint: str,
        model: str,
        start_time: float,
        tokens: Optional[int] = None,
        **extra: Any,
    ):
        """Log successful API call."""
        duration_ms = (time.time() - start_time) * 1000

        self.logger.info(
            f"API_CALL_SUCCESS provider={self.provider} model={model} duration_ms={round(duration_ms, 2)}",
            extra={
                "event": "API_CALL_SUCCESS",
                "call_id": call_id,
                "provider": self.provider,
                "model": model,
                "endpoint": endpoint,
                "duration_ms": round(duration_ms, 2),
                "tokens": tokens,
                **extra,
            },
        )

    def _log_api_call_error(
        self,
        call_id: str,
        endpoint: str,
        model: str,
        start_time: float,
        error: Exception,
        **extra: Any,
    ):
        """Log failed API call."""
        duration_ms = (time.time() - start_time) * 1000

        # Determine error type
        error_type = type(error).__name__
        error_code = getattr(error, "code", None)
        retryable = self._is_retryable_error(error)

        self.logger.error(
            f"API_CALL_ERROR provider={self.provider} model={model} error_type={error_type} retryable={retryable}",
            extra={
                "event": "API_CALL_ERROR",
                "call_id": call_id,
                "provider": self.provider,
                "model": model,
                "endpoint": endpoint,
                "duration_ms": round(duration_ms, 2),
                "error_type": error_type,
                "error_code": error_code,
                "error": str(error),
                "retryable": retryable,
                **extra,
            },
            exc_info=not retryable,
        )

    def _is_retryable_error(self, error: Exception) -> bool:
        """Determine if an error is retryable.

        Uses a three-tier classification strategy:
        1. Exception type checks (most reliable)
        2. HTTP status code checks
        3. String pattern fallback (for third-party exceptions)
        """
        # Tier 1: Type-based check (reliable)
        try:
            from victor.core.errors import ProviderRateLimitError, ProviderTimeoutError

            if isinstance(error, (ProviderRateLimitError, ProviderTimeoutError)):
                return True
        except ImportError:
            pass

        # Tier 2: Status code check
        status = getattr(error, "status_code", None) or getattr(error, "code", None)
        if isinstance(status, int) and status in (429, 500, 502, 503, 504):
            return True

        # Tier 3: String fallback (for third-party exceptions)
        error_str = str(error).lower()
        return any(
            p in error_str
            for p in (
                "timeout",
                "rate limit",
                "connection reset",
                "connection refused",
                "internal server error",
                "gateway timeout",
                "server error",
                "service unavailable",
                "too many requests",
            )
        )

    def _sanitize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize config for logging (remove secrets).

        Args:
            config: Raw configuration dict

        Returns:
            Sanitized config with secrets redacted
        """
        sanitized = {}
        sensitive_keys = {
            "api_key",
            "apikey",
            "api-key",
            "token",
            "authorization",
            "auth",
            "password",
            "secret",
            "credential",
        }

        for key, value in config.items():
            if key.lower() in sensitive_keys:
                if isinstance(value, str) and len(value) > 0:
                    # Show first few chars for debugging, but hide most
                    if len(value) <= 2:
                        sanitized[key] = "(empty)"
                    elif len(value) <= 4:
                        # Show first char only
                        sanitized[key] = f"{value[:1]}***"
                    else:
                        # Show first 2 chars for short keys, up to 8 for longer keys
                        prefix_len = min(len(value) - 2, 8)
                        sanitized[key] = f"{value[:prefix_len]}..."
                else:
                    sanitized[key] = "***"
            else:
                sanitized[key] = value

        return sanitized


@contextmanager
def log_provider_operation(
    operation: str,
    provider: str,
    model: str,
    logger_instance: Optional[logging.Logger] = None,
):
    """
    Context manager for logging any provider operation.

    Args:
        operation: Operation name (e.g., "chat", "stream", "init")
        provider: Provider name
        model: Model name
        logger_instance: Custom logger (uses module logger if None)

    Yields:
        None

    Example:
        with log_provider_operation("chat", "deepseek", "deepseek-chat"):
            response = await provider.chat(messages)
    """
    op_logger = logger_instance or logging.getLogger(__name__)
    start_time = time.time()

    op_logger.info(
        f"PROVIDER_OPERATION_START operation={operation} provider={provider} model={model}",
        extra={
            "event": "PROVIDER_OPERATION_START",
            "operation": operation,
            "provider": provider,
            "model": model,
            "timestamp": start_time,
        },
    )

    try:
        yield
    except Exception as e:
        duration = time.time() - start_time
        op_logger.error(
            f"PROVIDER_OPERATION_ERROR operation={operation} provider={provider} error_type={type(e).__name__}",
            extra={
                "event": "PROVIDER_OPERATION_ERROR",
                "operation": operation,
                "provider": provider,
                "model": model,
                "duration_s": round(duration, 2),
                "error_type": type(e).__name__,
                "error": str(e),
            },
            exc_info=True,
        )
        raise
    else:
        duration = time.time() - start_time
        op_logger.info(
            f"PROVIDER_OPERATION_SUCCESS operation={operation} provider={provider} duration_s={round(duration, 2)}",
            extra={
                "event": "PROVIDER_OPERATION_SUCCESS",
                "operation": operation,
                "provider": provider,
                "model": model,
                "duration_s": round(duration, 2),
            },
        )

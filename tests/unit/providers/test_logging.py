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

"""Tests for victor.providers.logging module."""

import logging
from unittest.mock import patch, MagicMock
import time

import pytest

from victor.providers.logging import (
    ProviderLogger,
    log_provider_operation,
)


class TestProviderLogger:
    """Tests for ProviderLogger class."""

    def test_initialization(self):
        """Test logger initialization."""
        logger = ProviderLogger("deepseek", __name__)
        assert logger.provider == "deepseek"
        assert logger.logger.name == __name__

    def test_log_provider_init(self, caplog):
        """Test logging provider initialization."""
        with caplog.at_level(logging.INFO):
            logger = ProviderLogger("deepseek", "test_logger")
            logger.log_provider_init(
                model="deepseek-chat",
                key_source="DEEPSEEK_API_KEY environment variable",
                non_interactive=True,
                config={"timeout": 120, "temperature": 0.1},
            )

            assert "PROVIDER_INIT" in caplog.text
            assert "deepseek" in caplog.text
            assert "deepseek-chat" in caplog.text

    def test_sanitize_config_removes_secrets(self):
        """Test that sensitive config values are sanitized."""
        logger = ProviderLogger("test", __name__)

        config = {
            "api_key": "sk-sensitive-key-12345",
            "token": "secret-token",
            "model": "deepseek-chat",
            "timeout": 120,
        }

        sanitized = logger._sanitize_config(config)

        assert sanitized["model"] == "deepseek-chat"
        assert sanitized["timeout"] == 120
        assert "api_key" in sanitized
        assert sanitized["api_key"] != "sk-sensitive-key-12345"
        assert sanitized["api_key"] != "secret-token"
        assert "..." in sanitized["api_key"] or "***" in sanitized["api_key"]

    def test_sanitize_config_with_empty_key(self):
        """Test sanitization with empty key value."""
        logger = ProviderLogger("test", __name__)

        config = {"api_key": "", "model": "test"}

        sanitized = logger._sanitize_config(config)

        assert sanitized["api_key"] in ("***", "(empty)")

    def test_sanitize_config_with_short_key(self):
        """Test sanitization with short key."""
        logger = ProviderLogger("test", __name__)

        config = {"api_key": "sk-ab", "model": "test"}

        sanitized = logger._sanitize_config(config)

        assert sanitized["api_key"] != "sk-ab"
        assert "sk" in sanitized["api_key"]

    def test_log_api_call_success(self, caplog):
        """Test logging successful API call."""
        with caplog.at_level(logging.INFO):
            logger = ProviderLogger("deepseek", "test_logger")
            logger._log_api_call_success(
                call_id="test-call-123",
                endpoint="https://api.deepseek.com/v1/chat",
                model="deepseek-chat",
                start_time=time.time(),
                tokens=150,
            )

            assert "API_CALL_SUCCESS" in caplog.text

    def test_log_api_call_error(self, caplog):
        """Test logging failed API call."""
        with caplog.at_level(logging.ERROR):
            logger = ProviderLogger("deepseek", "test_logger")
            logger._log_api_call_error(
                call_id="test-call-123",
                endpoint="https://api.deepseek.com/v1/chat",
                model="deepseek-chat",
                start_time=time.time(),
                error=Exception("Connection timeout"),
            )

            assert "API_CALL_ERROR" in caplog.text

    def test_is_retryable_error(self):
        """Test retryable error detection."""
        logger = ProviderLogger("test", __name__)

        # Retryable errors
        assert logger._is_retryable_error(Exception("Request timeout"))
        assert logger._is_retryable_error(Exception("Connection refused"))
        assert logger._is_retryable_error(Exception("Rate limit exceeded"))
        assert logger._is_retryable_error(Exception("500 Internal Server Error"))
        assert logger._is_retryable_error(Exception("503 Service Unavailable"))
        assert logger._is_retryable_error(Exception("504 Gateway Timeout"))
        assert logger._is_retryable_error(Exception("429 Too Many Requests"))

        # Non-retryable errors
        assert not logger._is_retryable_error(Exception("Invalid API key"))
        assert not logger._is_retryable_error(Exception("400 Bad Request"))


class TestLogProviderOperation:
    """Tests for log_provider_operation context manager."""

    def test_success_logging(self, caplog):
        """Test successful operation logging."""
        with caplog.at_level(logging.INFO):
            with log_provider_operation("chat", "deepseek", "deepseek-chat"):
                pass  # Success

            assert "PROVIDER_OPERATION_START" in caplog.text
            assert "PROVIDER_OPERATION_SUCCESS" in caplog.text
            assert "chat" in caplog.text

    def test_error_logging(self, caplog):
        """Test error logging in context manager."""
        with caplog.at_level(logging.INFO):
            with pytest.raises(ValueError):
                with log_provider_operation("chat", "deepseek", "deepseek-chat"):
                    raise ValueError("Test error")

            assert "PROVIDER_OPERATION_START" in caplog.text
            assert "PROVIDER_OPERATION_ERROR" in caplog.text
            assert "Test error" in caplog.text

    def test_custom_logger(self, caplog):
        """Test custom logger parameter."""
        custom_logger = logging.getLogger("custom.test")
        with caplog.at_level(logging.INFO):
            with log_provider_operation(
                "init", "test", "test-model", logger_instance=custom_logger
            ):
                pass

        # Should have logged to custom logger
        assert any("PROVIDER_OPERATION_START" in record.message for record in caplog.records)


class TestStructuredLogging:
    """Tests for structured logging extra data."""

    def test_provider_init_has_structured_data(self, caplog):
        """Test PROVIDER_INIT includes structured extra data."""
        with caplog.at_level(logging.INFO):
            logger = ProviderLogger("deepseek", "test_logger")
            logger.log_provider_init(
                model="deepseek-chat",
                key_source="DEEPSEEK_API_KEY",
                non_interactive=False,
                config={"timeout": 120},
            )

            for record in caplog.records:
                if hasattr(record, "event") and record.event == "PROVIDER_INIT":
                    assert record.provider == "deepseek"
                    assert record.model == "deepseek-chat"
                    assert record.key_source == "DEEPSEEK_API_KEY"
                    assert record.non_interactive is False
                    assert "config" in record.__dict__

    def test_api_key_resolution_logging(self, caplog):
        """Test API key resolution includes structured data."""
        from victor.providers.resolution import APIKeyResult, KeySource

        with caplog.at_level(logging.INFO):
            logger = ProviderLogger("deepseek", "test_logger")
            result = APIKeyResult(
                key="sk-test",
                source="environment",
                source_detail="DEEPSEEK_API_KEY",
                sources_attempted=[
                    KeySource(
                        source="environment",
                        description="DEEPSEEK_API_KEY",
                        found=True,
                    )
                ],
                non_interactive=True,
                confidence="high",
            )
            logger.log_api_key_resolution(result, latency_ms=5.2)

            for record in caplog.records:
                if hasattr(record, "event") and record.event == "API_KEY_RESOLUTION":
                    assert record.provider == "deepseek"
                    assert record.source == "environment"
                    assert record.success is True
                    assert record.latency_ms == 5.2
                    assert "sources_attempted" in record.__dict__

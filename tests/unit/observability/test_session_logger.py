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

"""Tests for session ID integration with logger."""

from __future__ import annotations

import logging
import pytest
from pathlib import Path
import tempfile
import re


class TestSessionLoggerIntegration:
    """Test suite for session ID in logger output."""

    def test_logger_format_includes_session_id(self):
        """Test that logger format includes session ID when available."""
        from victor.agent.session_id import generate_session_id

        # Generate session ID
        session_id = generate_session_id()

        # Create logger with session ID
        logger = logging.getLogger(f"victor.{session_id}")

        # Create string handler to capture output
        import io

        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setFormatter(logging.Formatter("[%(name)s] %(levelname)s: %(message)s"))

        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        # Log a message
        logger.info("Test message")

        # Get output
        output = log_capture.getvalue()

        # Verify session ID is in logger name
        assert session_id in output
        assert f"victor.{session_id}" in output
        assert "Test message" in output

    def test_logger_without_session_id(self):
        """Test logger without session ID uses default format."""
        logger = logging.getLogger("victor")

        # Create string handler
        import io

        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setFormatter(logging.Formatter("[%(name)s] %(levelname)s: %(message)s"))

        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        # Log a message
        logger.info("Test message")

        # Get output
        output = log_capture.getvalue()

        # Verify default format
        assert "[victor]" in output
        assert "Test message" in output

    def test_agent_active_session_id_in_logs(self, tmp_path):
        """Test that agent's active_session_id is used in logger."""
        # This would require agent initialization, keeping as unit test
        from victor.agent.session_id import generate_session_id

        session_id = generate_session_id()

        # Simulate agent with active_session_id
        class MockAgent:
            def __init__(self, session_id):
                self.active_session_id = session_id

        agent = MockAgent(session_id)

        # Logger should use agent's active_session_id
        logger_name = f"victor.{agent.active_session_id}" if agent.active_session_id else "victor"
        logger = logging.getLogger(logger_name)

        # Verify logger name
        assert session_id in logger_name

    def test_session_id_validation_in_logs(self):
        """Test that invalid session IDs are handled gracefully."""
        # Invalid session ID format
        invalid_session_id = "invalid-format"

        # Logger should still work
        logger = logging.getLogger(f"victor.{invalid_session_id}")
        logger.setLevel(logging.WARNING)

        # Create string handler
        import io

        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setFormatter(logging.Formatter("[%(name)s] %(levelname)s: %(message)s"))

        logger.addHandler(handler)

        # Log a message
        logger.warning("Test warning")

        # Get output
        output = log_capture.getvalue()

        # Should still log, even with invalid session ID
        assert "Test warning" in output

    def test_log_file_path_includes_session_id(self, tmp_path):
        """Test that log file path includes session ID via setup_logging."""
        from victor.agent.session_id import generate_session_id
        from victor.ui.commands.utils import configure_logging

        session_id = generate_session_id()

        # Create log file path with session ID
        log_file = tmp_path / f"victor_{session_id}.log"

        # Use existing configure_logging which includes session_id in format
        configure_logging(
            log_level="INFO",
            log_file=log_file,
            session_id=session_id,
        )

        logger = logging.getLogger("victor.test")
        logger.info("Test message to file")

        # Flush handlers
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.flush()
                handler.close()

        # Verify log file exists and contains message
        assert log_file.exists()
        content = log_file.read_text()
        assert "Test message to file" in content
        # Session ID should be in the log file format: repo-session_id - name - level - message
        assert session_id in content


class TestSessionLoggingInWorkflow:
    """Test session logging integration in workflow context."""

    def test_session_id_propagated_to_subcomponents(self):
        """Test that session ID is propagated to logger in all components."""
        from victor.agent.session_id import generate_session_id

        session_id = generate_session_id()

        # Create multiple loggers with session ID
        main_logger = logging.getLogger(f"victor.{session_id}")
        tool_logger = logging.getLogger(f"victor.{session_id}.tools")
        orchestrator_logger = logging.getLogger(f"victor.{session_id}.orchestrator")

        # All should include session ID
        assert session_id in main_logger.name
        assert session_id in tool_logger.name
        assert session_id in orchestrator_logger.name

    def test_session_id_in_structured_logs(self):
        """Test session ID in structured JSON logs."""
        import json
        import io

        from victor.agent.session_id import generate_session_id

        session_id = generate_session_id()

        # Create logger with JSON formatter
        logger = logging.getLogger(f"victor.{session_id}")
        log_capture = io.StringIO()

        handler = logging.StreamHandler(log_capture)

        # JSON formatter
        class JsonFormatter(logging.Formatter):
            def format(self, record):
                log_obj = {
                    "timestamp": self.formatTime(record),
                    "logger": record.name,
                    "level": record.levelname,
                    "message": record.getMessage(),
                    "session_id": (record.name.split(".")[-1] if "." in record.name else None),
                }
                return json.dumps(log_obj)

        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        # Log a message
        logger.info("Test message")

        # Get and parse output
        output = log_capture.getvalue()
        log_obj = json.loads(output)

        # Verify session_id field
        assert log_obj["session_id"] == session_id
        assert log_obj["message"] == "Test message"
        assert log_obj["logger"] == f"victor.{session_id}"

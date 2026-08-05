# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Tests for the Live-aware console log handler.

Covers the regression where raw ``StreamHandler(sys.stderr)`` records tore
through the Rich Live display mid-frame, and tool-execution errors rendered
multiple times (raw log line + service print + renderer error line).
"""

import io
import logging

import pytest
from rich.console import Console

from victor.core.errors import ErrorHandler
from victor.ui.rendering.live_renderer import LiveDisplayRenderer
from victor.ui.rendering.log_handler import (
    LiveAwareLogHandler,
    get_live_console,
    register_live_console,
    unregister_live_console,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    unregister_live_console()
    yield
    unregister_live_console()


def _record(msg="hello", level=logging.ERROR, **extra_attrs) -> logging.LogRecord:
    record = logging.LogRecord(
        name="victor", level=level, pathname=__file__, lineno=1, msg=msg, args=(), exc_info=None
    )
    for key, value in extra_attrs.items():
        setattr(record, key, value)
    return record


class TestLiveAwareLogHandler:
    def test_plain_stream_when_no_live_console(self):
        stream = io.StringIO()
        handler = LiveAwareLogHandler(stream)
        handler.emit(_record("plain path"))
        assert "plain path" in stream.getvalue()

    def test_routes_through_live_console_when_registered(self):
        stream = io.StringIO()
        console = Console(record=True, width=100)
        register_live_console(console)
        handler = LiveAwareLogHandler(stream)
        handler.emit(_record("routed above live region"))
        assert "routed above live region" in console.export_text()
        assert stream.getvalue() == ""

    def test_tool_execution_records_dropped_only_while_live(self):
        stream = io.StringIO()
        console = Console(record=True, width=100)
        handler = LiveAwareLogHandler(stream)

        register_live_console(console)
        handler.emit(_record("dup error", victor_category="tool_execution"))
        assert "dup error" not in console.export_text()
        assert stream.getvalue() == ""

        unregister_live_console()
        handler.emit(_record("headless error", victor_category="tool_execution"))
        assert "headless error" in stream.getvalue()

    def test_other_categories_still_print_while_live(self):
        console = Console(record=True, width=100)
        register_live_console(console)
        handler = LiveAwareLogHandler(io.StringIO())
        handler.emit(_record("provider blew up", victor_category="provider"))
        assert "provider blew up" in console.export_text()


class TestRendererRegistration:
    def test_start_registers_and_cleanup_unregisters(self):
        console = Console(record=True, width=100)
        renderer = LiveDisplayRenderer(console)
        renderer.start()
        assert get_live_console() is console
        renderer.cleanup()
        assert get_live_console() is None


class TestErrorHandlerTagging:
    def test_log_error_attaches_victor_category(self):
        captured: list[logging.LogRecord] = []

        class Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                captured.append(record)

        from victor.core.errors import ToolExecutionError

        error_handler = ErrorHandler()
        error_handler.logger.addHandler(capture := Capture())
        try:
            error_handler.handle(
                ToolExecutionError("not allowed in readonly mode", tool_name="shell")
            )
        finally:
            error_handler.logger.removeHandler(capture)
        assert captured
        assert any(getattr(r, "victor_category", None) == "tool_execution" for r in captured)


class TestServiceDuplicateSuppression:
    def test_live_display_active_reflects_registry(self):
        from victor.runtime.live_console import live_display_active

        assert live_display_active() is False
        register_live_console(Console(width=80))
        assert live_display_active() is True

    def test_print_tool_error_suppressed_while_live(self):
        from unittest.mock import MagicMock

        from victor.agent.services.tool_error_display import print_tool_error_once

        ctx = MagicMock()
        ctx.shown_tool_errors = set()
        register_live_console(Console(width=80))
        print_tool_error_once(ctx, "shell", "boom", skipped=False, elapsed_ms=2.0)
        ctx.console.print.assert_not_called()

        unregister_live_console()
        print_tool_error_once(ctx, "shell", "boom", skipped=False, elapsed_ms=2.0)
        ctx.console.print.assert_called_once()

    def test_not_found_errors_deduplicated(self):
        from unittest.mock import MagicMock

        from victor.agent.services.tool_error_display import print_tool_error_once

        ctx = MagicMock()
        ctx.shown_tool_errors = set()
        print_tool_error_once(ctx, "ghost", "tool not found", skipped=True, elapsed_ms=1.0)
        print_tool_error_once(ctx, "ghost", "tool not found", skipped=True, elapsed_ms=1.0)
        assert ctx.console.print.call_count == 1

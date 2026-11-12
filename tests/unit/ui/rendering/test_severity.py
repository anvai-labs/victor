# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for the structured Severity seam (audit P3).

Provider errors used to be glued to "❌" strings and run through a substring
classifier that ignored that marker — terminal errors rendered as dim neutral
status while any prose containing "failed" rendered yellow.
"""

from __future__ import annotations

import io

from rich.console import Console

from victor.ui.rendering.utils import (
    Severity,
    classify_severity,
    render_severity,
    render_status_message,
)


def _capture() -> tuple[Console, io.StringIO]:
    # force_terminal keeps the ANSI styling so the tests can pin colors too;
    # a plain StringIO capture strips codes entirely.
    buffer = io.StringIO()
    return Console(file=buffer, force_terminal=True, width=100), buffer


class TestStructuredSeverity:
    def test_error_renders_red_with_cross_icon(self):
        console, buffer = _capture()
        render_severity(console, Severity.ERROR, "Model not found: gpt-oss")
        out = buffer.getvalue()
        assert "✗" in out and "\x1b[31m" in out  # red
        assert "Model not found" in out

    def test_warning_strips_legacy_alert_prefix(self):
        console, buffer = _capture()
        render_severity(console, Severity.WARNING, "⚠ cache is cold")
        out = buffer.getvalue()
        assert "!" in out and "\x1b[33m" in out  # yellow
        assert "cache is cold" in out
        assert "⚠" not in out

    def test_success_and_info_styles(self):
        console, buffer = _capture()
        render_severity(console, Severity.SUCCESS, "done")
        assert "\x1b[32m" in buffer.getvalue()  # green
        console, buffer = _capture()
        render_severity(console, Severity.INFO, "working")
        assert "\x1b[2m" in buffer.getvalue()  # dim


class TestLegacyClassifier:
    def test_classifies_historical_smuggled_markers(self):
        # The ❌ glue the event dispatcher used to prepend — previously
        # IGNORED by the sniffer, now maps to ERROR.
        assert classify_severity("❌ upstream status 400") is Severity.ERROR
        assert classify_severity("⚠ cache is cold") is Severity.WARNING
        assert classify_severity("compilation failed somewhere") is Severity.WARNING
        assert classify_severity("thinking about it") is Severity.THINKING
        assert classify_severity("all normal") is Severity.INFO

    def test_prose_mentioning_failed_still_classifies_warning(self):
        # Documented legacy behavior of the shim: prose sniffing remains for
        # untyped callers. Structured callers bypass it entirely.
        assert classify_severity("the retry policy failed open") is Severity.WARNING


class TestShimParity:
    def test_render_status_message_matches_classified_severity(self):
        shim_console, shim_buffer = _capture()
        render_status_message(shim_console, "❌ boom")
        direct_console, direct_buffer = _capture()
        render_severity(direct_console, Severity.ERROR, "❌ boom")
        assert shim_buffer.getvalue() == direct_buffer.getvalue()

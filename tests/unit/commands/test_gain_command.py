"""Tests for the /gain slash command (condensation savings dashboard)."""

from __future__ import annotations

import io

from rich.console import Console
from unittest.mock import MagicMock

from victor.agent.usage_analytics import UsageAnalytics
from victor.ui.slash.commands.metrics import GainCommand
from victor.ui.slash.protocol import CommandContext


def _run_gain() -> str:
    buf = io.StringIO()
    console = Console(file=buf, width=120)
    ctx = CommandContext(console=console, settings=MagicMock())
    GainCommand().execute(ctx)
    return buf.getvalue()


class TestGainCommand:
    def setup_method(self):
        UsageAnalytics.reset_instance()

    def teardown_method(self):
        UsageAnalytics.reset_instance()

    def test_metadata(self):
        meta = GainCommand().metadata
        assert meta.name == "gain"
        assert set(meta.aliases) == {"savings", "condensation"}
        assert meta.category == "metrics"

    def test_empty_stats_message(self):
        output = _run_gain()
        assert "No condensation events yet" in output

    def test_populated_stats_render(self):
        analytics = UsageAnalytics.get_instance()
        analytics.record_output_condensation("pytest", 100_000, 5_000)
        analytics.record_output_condensation("git-status", 10_000, 4_000)

        output = _run_gain()
        assert "pytest" in output
        assert "git-status" in output
        assert "Condensation Gain" in output
        # Total saved: 101,000 chars → ~25,250 tokens
        assert "101,000" in output
        assert "25,250" in output

    def test_error_path_is_graceful(self, monkeypatch):
        monkeypatch.setattr(
            UsageAnalytics,
            "get_instance",
            classmethod(lambda cls, config=None: (_ for _ in ()).throw(RuntimeError("boom"))),
        )
        output = _run_gain()
        assert "Error fetching condensation stats" in output

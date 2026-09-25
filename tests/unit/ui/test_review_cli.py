"""Unit tests for the `victor review` CLI surface.

The panel itself is faked at the module seam (``_run_review``), so these tests
pin argument parsing, reviewer-spec coercion, verdict printing, and the
CI-able exit codes without any LLM or network.
"""

from __future__ import annotations

import asyncio

import pytest
from typer.testing import CliRunner

import victor.ui.commands.review as review_cli
from victor.framework.review import APPROVE, REQUEST_CHANGES, REVIEW_INCOMPLETE, ReviewVerdict
from victor.ui.commands.review import (
    _default_reviewers,
    _parse_reviewers,
    review_app,
)

runner = CliRunner()


class TestParseReviewers:
    def test_valid_pairs(self):
        specs = _parse_reviewers("zai:glm-5.3, inferflux:qwen3-coder-30b")
        assert [(s.provider, s.model) for s in specs] == [
            ("zai", "glm-5.3"),
            ("inferflux", "qwen3-coder-30b"),
        ]
        assert specs[0].display_name == "zai/glm-5.3"

    def test_empty_string_yields_no_specs(self):
        assert _parse_reviewers("") == []

    def test_missing_model_is_rejected(self):
        with pytest.raises(review_cli.typer.BadParameter):
            _parse_reviewers("zai")


class TestDefaultReviewers:
    def test_explicit_provider_and_model_win(self):
        from victor.config.settings import load_settings

        specs = _default_reviewers("anthropic", "claude-x")
        assert [(s.provider, s.model) for s in specs] == [("anthropic", "claude-x")]

    def test_falls_back_to_configured_defaults(self, monkeypatch):
        class _Provider:
            default_provider = "zai"
            default_model = "glm-5.3"

        class _Settings:
            provider = _Provider()

        monkeypatch.setattr("victor.config.settings.load_settings", lambda fresh=False: _Settings())
        specs = _default_reviewers(None, None)
        assert [(s.provider, s.model) for s in specs] == [("zai", "glm-5.3")]

    def test_unconfigured_defaults_are_rejected(self, monkeypatch):
        class _Settings:
            provider = None

        monkeypatch.setattr("victor.config.settings.load_settings", lambda fresh=False: _Settings())
        with pytest.raises(review_cli.typer.BadParameter):
            _default_reviewers(None, None)


class TestPrintVerdict:
    def test_prints_reviewers_findings_and_summary(self, capsys):
        verdict = ReviewVerdict(verdict=REQUEST_CHANGES)
        verdict.reviewer_verdicts = {"zai/glm": REQUEST_CHANGES, "inf/qwen": APPROVE}
        verdict.blocked_by = ["zai/glm"]
        verdict.reviewer_summaries = {"zai/glm": "race on shared state"}
        from victor.framework.review import Finding

        verdict.findings = [
            Finding(severity="major", file="a.py", line=12, summary="race", reviewer="zai/glm")
        ]

        review_cli._print_verdict(verdict)
        out = capsys.readouterr().out
        assert "VERDICT: request_changes" in out
        assert "zai/glm" in out and "blocked by: zai/glm" in out
        assert "a.py:12 — race" in out


def _patch_panel(monkeypatch, result: int, calls: dict):
    async def fake_run(specs, **kwargs):
        calls["specs"] = specs
        calls["kwargs"] = kwargs
        return result

    def factory(*args, **kwargs):
        return fake_run(*args, **kwargs)

    monkeypatch.setattr(review_cli, "_run_review", factory)


def test_cli_pr_approve_exit_zero(monkeypatch, tmp_path):
    calls: dict = {}
    _patch_panel(monkeypatch, 0, calls)
    diff_file = tmp_path / "d.diff"
    diff_file.write_text("diff --git a/x b/x\n")

    result = runner.invoke(
        review_app,
        [
            "pr",
            "1187",
            "--reviewers",
            "zai:glm-5.3",
            "--diff-file",
            str(diff_file),
            "--intent",
            "fix mock",
        ],
    )
    assert result.exit_code == 0
    assert calls["kwargs"]["diff_text"].startswith("diff --git")
    assert calls["kwargs"]["intent"] == "fix mock"
    assert calls["specs"][0].provider == "zai"


@pytest.mark.parametrize("panel_result,expected_exit", [(0, 0), (1, 1), (2, 2)])
def test_cli_exit_codes_mirror_verdict(monkeypatch, tmp_path, panel_result, expected_exit):
    calls: dict = {}
    _patch_panel(monkeypatch, panel_result, calls)
    diff_file = tmp_path / "d.diff"
    diff_file.write_text("x\n")
    result = runner.invoke(review_app, ["pr", "1", "--diff-file", str(diff_file)])
    assert result.exit_code == expected_exit


def test_cli_diff_command_reviews_file(monkeypatch, tmp_path):
    calls: dict = {}
    _patch_panel(monkeypatch, 2, calls)
    diff_file = tmp_path / "d.diff"
    diff_file.write_text("unified diff\n")
    result = runner.invoke(review_app, ["diff", str(diff_file)])
    assert result.exit_code == 2  # REVIEW_INCOMPLETE
    assert calls["kwargs"]["diff_text"] == "unified diff\n"


def test_cli_timeout_reports_exit_two(monkeypatch, tmp_path):
    def factory(*args, **kwargs):
        async def forever(*a, **k):
            raise asyncio.TimeoutError()

        return forever()

    monkeypatch.setattr(review_cli, "_run_review", factory)
    monkeypatch.setattr(review_cli.asyncio, "run", lambda coro: _drain(coro))
    diff_file = tmp_path / "d.diff"
    diff_file.write_text("x\n")
    result = runner.invoke(review_app, ["pr", "1", "--diff-file", str(diff_file), "--timeout", "1"])
    assert result.exit_code == 2


def _drain(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(asyncio.wait_for(coro, timeout=2))
    except (asyncio.TimeoutError, TimeoutError):
        raise asyncio.TimeoutError()
    finally:
        loop.close()


def test_cli_review_incomplete_prints(monkeypatch, tmp_path):
    # Print content is covered by TestPrintVerdict; here we pin that the
    # incomplete verdict flows through the command's exit code.
    calls: dict = {}
    _patch_panel(monkeypatch, 2, calls)
    diff_file = tmp_path / "d.diff"
    diff_file.write_text("x\n")
    result = runner.invoke(review_app, ["pr", "1", "--diff-file", str(diff_file)])
    assert result.exit_code == 2


def test_review_incomplete_constant_is_fail_closed():
    assert REVIEW_INCOMPLETE != APPROVE

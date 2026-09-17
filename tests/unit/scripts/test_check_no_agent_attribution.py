"""Tests for the commit-message attribution policy."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "ci" / "check_no_agent_attribution.py"
SPEC = importlib.util.spec_from_file_location("check_no_agent_attribution", SCRIPT)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_dependabot_service_trailer_is_allowed() -> None:
    message = (
        "fix: update dependency\n\n"
        "Co-authored-by: dependabot[bot] "
        "<49699333+dependabot[bot]@users.noreply.github.com>\n"
    )

    assert MODULE.scan(message, "commit") == []


def test_untrusted_bot_coauthor_remains_blocked() -> None:
    message = "Co-authored-by: release-helper[bot] <helper@example.com>\n"

    violations = MODULE.scan(message, "commit")

    assert len(violations) == 1
    assert "bot/AI account" in violations[0]


def test_dependabot_name_with_untrusted_email_remains_blocked() -> None:
    message = "Co-authored-by: dependabot[bot] <dependabot@example.com>\n"

    violations = MODULE.scan(message, "commit")

    assert len(violations) == 1
    assert "bot/AI account" in violations[0]


def test_third_party_ai_coauthor_remains_blocked() -> None:
    message = "Co-authored-by: Claude Code <noreply@anthropic.com>\n"

    violations = MODULE.scan(message, "commit")

    assert violations
    assert any("AI agent" in violation for violation in violations)

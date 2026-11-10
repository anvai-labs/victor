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

"""Opt-in live smoke tests for session interop (never run in CI).

Enabled only when ``INTEROP_SMOKE=1`` is set:

    INTEROP_SMOKE=1 python -m pytest tests/integration/agent/conversation/ \\
        -m integration -k interop_smoke -q

Tier 1 (always when enabled): product-path file mechanics — CLI import →
export → re-import through a real temp project DB.

Tier 2 (``INTEROP_SMOKE_LIVE=1``): hands the exported transcript to the real
``claude`` CLI and asserts it recalls source context (requires local auth).
The codex leg is mechanically covered by tier 1's writer assertions; its
live model call is quota-gated and intentionally not automated.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("INTEROP_SMOKE") != "1",
        reason="opt-in live smoke: set INTEROP_SMOKE=1 (never in CI)",
    ),
]

REPO_ROOT = Path(__file__).parents[4]
# Real home captured at import time — the integration conftest redirects HOME
# during tests, but the live claude leg needs the genuine ~/.claude (auth+store).
_REAL_HOME = Path(os.environ.get("HOME", str(Path.home()))).resolve()
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "interop"
RECALL_TOKEN = "login bug"


@pytest.fixture
def smoke_project(tmp_path, monkeypatch):
    """Temp victor project DB addressed through the real CLI module."""
    db = tmp_path / "project.db"
    monkeypatch.setenv("VICTOR_CONVERSATION_DB", str(db))
    monkeypatch.chdir(tmp_path)
    yield {"db": db, "cwd": tmp_path}


def _cli(*args):
    # Run from the repo root so ``python -m`` resolves THIS checkout's victor
    # package (a bare cwd would import the stale editable install).
    result = subprocess.run(
        [os.sys.executable, "-m", "victor.ui.cli", *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=120,
    )
    assert result.returncode == 0, f"CLI failed: {result.stdout}\n{result.stderr}"
    return result.stdout


def _cli_session_id(*args) -> str:
    """Run a CLI import with --json and pull session_id from the report."""

    stdout = _cli(*args, "--json")
    start = stdout.find("{")
    assert start >= 0, f"no JSON report in output: {stdout}"
    report, _ = json.JSONDecoder().raw_decode(stdout[start:])
    return report["session_id"]


class TestInteropSmokeFiles:
    def test_cli_import_export_reimport_round_trip(self, smoke_project):
        """Import a claude fixture → export victor json → re-import intact."""
        source = FIXTURES / "claude_tools_thinking.jsonl"
        out = smoke_project["cwd"] / "rt.json"

        session_id = _cli_session_id("data", "session", "import", str(source))

        _cli(
            "data",
            "session",
            "export",
            "--session-id",
            session_id,
            "--format",
            "victor",
            "--output",
            str(out),
        )
        assert out.exists()

        # Re-import into a second, independent project DB
        second_db = smoke_project["cwd"] / "second.db"
        result = subprocess.run(
            [os.sys.executable, "-m", "victor.ui.cli", "data", "session", "import", str(out)],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            timeout=120,
            env=dict(os.environ, VICTOR_CONVERSATION_DB=str(second_db)),
        )
        assert result.returncode == 0, result.stderr

        from victor.agent.conversation.store import ConversationStore

        fresh = ConversationStore(db_path=second_db)
        loaded = fresh.load_session(session_id)
        assert loaded is not None
        messages = loaded["conversation"]["messages"]
        assert any(
            RECALL_TOKEN in " ".join(str(m.get("content", "")) for m in messages)
            or "tests" in str(m.get("content", ""))
            for m in messages
        )
        roles = {m["role"] for m in messages}
        assert "tool_call" in roles and "tool" in roles

    def test_export_claude_shape_is_resumable(self, smoke_project):
        """The claude export is a well-formed resumable transcript."""
        source = FIXTURES / "codex_minimal.jsonl"
        out = smoke_project["cwd"] / "out.claude.jsonl"
        session_id = _cli_session_id("data", "session", "import", str(source))
        _cli(
            "data",
            "session",
            "export",
            "--session-id",
            session_id,
            "--format",
            "claude",
            "--output",
            str(out),
        )
        entries = [json.loads(ln) for ln in out.read_text().splitlines()]
        assert entries[0]["parentUuid"] is None
        for prev, cur in zip(entries, entries[1:]):
            assert cur["parentUuid"] == prev["uuid"]
        assert all(e["sessionId"] for e in entries)
        assert "thinking" not in out.read_text()


class TestInteropSmokeLive:
    """Full loop through the real claude CLI (auth + one small model call)."""

    pytestmark = pytest.mark.skipif(
        os.environ.get("INTEROP_SMOKE_LIVE") != "1",
        reason="live leg: set INTEROP_SMOKE_LIVE=1 (spends one small call)",
    )

    def test_claude_resume_recalls_imported_context(self, smoke_project, tmp_path):
        claude = shutil.which("claude")
        if not claude:
            pytest.skip("claude CLI not on PATH")
        # The integration conftest redirects HOME for isolation; claude needs
        # its REAL home for auth and its projects store. Captured at module
        # import (before the autouse fixture swaps it).
        real_home = _REAL_HOME
        if not (real_home / ".claude").is_dir():
            pytest.skip(f"no claude config under {real_home}")

        # Work in an isolated cwd whose slug dir we own.
        workdir = tmp_path / "liveproj"
        workdir.mkdir()
        source = FIXTURES / "claude_tools_thinking.jsonl"
        out = workdir / "session.claude.jsonl"
        session_id = _cli_session_id("data", "session", "import", str(source))
        _cli(
            "data",
            "session",
            "export",
            "--session-id",
            session_id,
            "--format",
            "claude",
            "--output",
            str(out),
        )

        # Place the transcript where claude --resume finds it for this cwd.
        # claude matches the INTERNAL sessionId, not the filename.
        import re

        entries = [json.loads(ln) for ln in out.read_text().splitlines()]
        claude_session_id = entries[0]["sessionId"]
        slug = re.sub(r"[^A-Za-z0-9]", "-", str(workdir))
        projects_dir = real_home / ".claude" / "projects" / slug
        projects_dir.mkdir(parents=True, exist_ok=True)
        target = projects_dir / f"{claude_session_id}.jsonl"
        shutil.copy(out, target)
        try:
            probe = (
                "Reply in ONE short sentence: what did the user ask to be run"
                " in this conversation?"
            )
            result = subprocess.run(
                [claude, "--resume", claude_session_id, "-p", probe],
                capture_output=True,
                text=True,
                cwd=workdir,
                timeout=240,
                env=dict(os.environ, HOME=str(real_home)),
            )
            answer = (result.stdout or "").strip()
            assert answer, f"empty answer (stderr: {result.stderr[-400:]})"
            lowered = answer.lower()
            assert (
                "test" in lowered or "pytest" in lowered
            ), f"recall probe failed; answer: {answer[:300]}"
        finally:
            target.unlink(missing_ok=True)
            # We own this slug dir (derived from our unique tmp workdir);
            # the claude run may have created side files (e.g. memory/).
            shutil.rmtree(projects_dir, ignore_errors=True)

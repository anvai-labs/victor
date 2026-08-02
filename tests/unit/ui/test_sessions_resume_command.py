# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
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

"""FEP-0029 Phase 3b: `victor session resume` CLI subcommand + from_cli_flags durable arming."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from typer.testing import CliRunner

from victor.framework.session_config import SessionConfig
from victor.ui.commands import sessions as sessions_cmd

# ── framework: from_cli_flags arms durable ────────────────────────


def test_from_cli_flags_threads_durable_approval() -> None:
    assert SessionConfig.from_cli_flags(durable_approval=True).tool_approval.durable is True
    # Default off — byte-identical for existing callers.
    assert SessionConfig.from_cli_flags().tool_approval.durable is False


# ── CLI: `victor chat resume` ─────────────────────────────────────


class _FakeClient:
    def __init__(self) -> None:
        self.decisions: list[Any] = []
        self.closed = False

    async def resume(self, run_id: str, decision: Any) -> Any:
        self.decisions.append((run_id, decision.approved, decision.response, decision.responder))
        return SimpleNamespace(status="ok", content="continued answer", run_id=None)

    async def close(self) -> None:
        self.closed = True


class _FakeRunner:
    last_client: "_FakeClient | None" = None

    def __init__(self, settings: Any, config: Any) -> None:
        pass

    def create_client(self, config: Any) -> _FakeClient:
        client = _FakeClient()
        _FakeRunner.last_client = client
        return client

    async def initialize_client(self, client: Any, **kw: Any) -> Any:
        return client


def _patch(monkeypatch: Any, runner_cls: Any = _FakeRunner) -> None:
    # The subcommand imports these locally, so patch them at their source modules.
    monkeypatch.setattr("victor.config.settings.load_settings", lambda: SimpleNamespace())
    monkeypatch.setattr("victor.framework.session_runner.FrameworkSessionRunner", runner_cls)


def test_resume_approve_forwards_decision_and_prints_answer(monkeypatch: Any) -> None:
    _patch(monkeypatch)
    result = CliRunner().invoke(sessions_cmd.sessions_app, ["resume", "run-1", "--approve"])

    assert result.exit_code == 0, result.output
    assert "approved" in result.output.lower()
    assert "continued answer" in result.output
    client = _FakeRunner.last_client
    assert client is not None and client.closed is True
    assert client.decisions == [("run-1", True, None, "cli")]


def test_resume_reject_with_note(monkeypatch: Any) -> None:
    _patch(monkeypatch)
    result = CliRunner().invoke(
        sessions_cmd.sessions_app, ["resume", "run-2", "--reject", "--note", "too risky"]
    )
    assert result.exit_code == 0, result.output
    assert "rejected" in result.output.lower()
    assert _FakeRunner.last_client.decisions == [("run-2", False, "too risky", "cli")]


def test_resume_unknown_run_exits_nonzero(monkeypatch: Any) -> None:
    class _BadClient(_FakeClient):
        async def resume(self, run_id: str, decision: Any) -> Any:
            raise ValueError(f"Unknown paused run: {run_id}")

    class _BadRunner(_FakeRunner):
        def create_client(self, config: Any) -> Any:
            client = _BadClient()
            _BadRunner.last_client = client
            return client

    _patch(monkeypatch, runner_cls=_BadRunner)
    result = CliRunner().invoke(sessions_cmd.sessions_app, ["resume", "nope", "--approve"])
    assert result.exit_code == 1
    assert "Unknown paused run" in result.output

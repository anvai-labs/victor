"""Wiring tests for ``run_interactive``'s TUI bootstrap hand-off.

Regression suite for the shell-first TUI startup (#1080):

* ``run_interactive`` must hand ``VictorTUIApp`` an *async callable* — the app
  invokes ``await bootstrap()`` in-app. Shipping #1080 passed the invoked
  coroutine instead (``return _run()``), so every TUI-capable interactive
  session died with ``'coroutine' object is not callable`` and an orphaned
  ``never awaited`` coroutine.
* Intentional ``typer.Exit`` stops (failed bootstrap, ``InitializationError``)
  must propagate — the broad ``except Exception`` handler reformatted them as
  an opaque ``Error during operation: 1`` after the real reason had already
  been shown.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any, Optional

import pytest
import typer

import victor.ui.commands.chat as chat_module
from victor.framework.agent_factory import InitializationError


class _Recording:
    """Shared mutable state between the patched helpers and the test."""

    def __init__(self) -> None:
        self.bootstrap: Any = None
        self.tui_agent: Any = None
        self.init_calls: int = 0


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tui_result: Optional[str] = "await",
    create_client_error: Optional[Exception] = None,
) -> _Recording:
    """Patch ``run_interactive``'s collaborators around the TUI branch.

    ``tui_result`` selects the fake ``_run_tui_app`` behavior: ``"await"``
    invokes the bootstrap exactly like ``VictorTUIApp._run_bootstrap`` does and
    returns the agent; any other value is returned verbatim (``None`` models a
    bootstrap that already failed in-app and asked the CLI to exit).
    """
    recorded = _Recording()

    class FakeSessionRunner:
        def __init__(self, settings: Any, config: Any) -> None:
            self._config = config

        def prepare_state(self, **kwargs: Any) -> Any:
            return SimpleNamespace(config=self._config, show_reasoning=False, use_streaming=False)

        def create_client(self, config: Any) -> Any:
            if create_client_error is not None:
                raise create_client_error
            return SimpleNamespace(name="fake-client")

        async def initialize_client(self, client: Any, planning_model: Optional[str] = None) -> Any:
            recorded.init_calls += 1
            return SimpleNamespace(name="fake-agent", active_session_id=None)

    async def fake_run_tui_app(
        client: Any,
        agent: Any,
        settings: Any,
        *,
        mode: Optional[str] = None,
        tool_budget: Optional[int] = None,
        theme: str = "dark",
        bootstrap: Any = None,
    ) -> Any:
        recorded.bootstrap = bootstrap
        if tui_result == "await":
            # The exact call convention of VictorTUIApp._run_bootstrap.
            recorded.tui_agent = await bootstrap()
            return recorded.tui_agent
        return tui_result

    async def fake_graceful_shutdown(agent: Any) -> None:
        return None

    async def fake_cleanup_session() -> None:
        return None

    monkeypatch.setattr(chat_module, "FrameworkSessionRunner", FakeSessionRunner)
    monkeypatch.setattr(chat_module, "_tui_capable", lambda: True)
    monkeypatch.setattr(chat_module, "_run_tui_app", fake_run_tui_app)
    monkeypatch.setattr(
        chat_module,
        "_require_existing_non_default_profile",
        lambda settings, profile: ({}, None),
    )
    monkeypatch.setattr(
        chat_module,
        "_resolve_profile_display",
        lambda **kwargs: SimpleNamespace(provider="fake-provider", model="fake-model"),
    )
    monkeypatch.setattr(
        chat_module, "_should_start_file_watchers_on_startup", lambda settings: False
    )
    monkeypatch.setattr(chat_module, "graceful_shutdown", fake_graceful_shutdown)
    monkeypatch.setattr(
        "victor.core.indexing.watcher_initializer.cleanup_session",
        fake_cleanup_session,
    )
    return recorded


async def test_tui_receives_async_callable_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    """The bootstrap handed to the TUI is a callable, and it initializes once."""
    recorded = _install_fakes(monkeypatch)
    settings = SimpleNamespace()

    await chat_module.run_interactive(
        settings, "default", True, False, surface="tui", graph_watch=False
    )

    boot = recorded.bootstrap
    # Positive: the app's documented contract (an async callable it can invoke).
    assert callable(boot)
    assert not inspect.iscoroutine(boot)  # regression: #1080 passed the coroutine
    assert recorded.init_calls == 1  # session initialized exactly once, in-app
    assert recorded.tui_agent is not None


async def test_failed_tui_bootstrap_exits_cleanly(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A bootstrap that already failed in-app exits 1 without opaque reformatting."""
    _install_fakes(monkeypatch, tui_result=None)
    settings = SimpleNamespace()

    with pytest.raises(typer.Exit) as exc_info:
        await chat_module.run_interactive(
            settings, "default", True, False, surface="tui", graph_watch=False
        )

    assert exc_info.value.exit_code == 1
    output = capsys.readouterr().out
    # Negative: the swallowed-Exit reformat buried the real reason ("1").
    assert "Error during operation" not in output


async def test_initialization_error_exits_without_duplicate_error_block(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """InitializationError prints its own guidance once; no second error block."""
    _install_fakes(
        monkeypatch,
        create_client_error=InitializationError(
            stage="credentials",
            message="api key missing",
            suggestions=["Run victor keys check"],
        ),
    )
    settings = SimpleNamespace()

    with pytest.raises(typer.Exit) as exc_info:
        await chat_module.run_interactive(
            settings, "default", True, False, surface="tui", graph_watch=False
        )

    assert exc_info.value.exit_code == 1
    output = capsys.readouterr().out
    assert "credentials" in output  # the structured message is shown
    assert "api key missing" in output
    # Negative: the Exit raised by the handler itself must not be re-caught
    # and reformatted as a second, content-free error block.
    assert output.count("Error during operation") == 0

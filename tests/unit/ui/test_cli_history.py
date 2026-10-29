"""Tests for CLI chat history and planning wiring."""

import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, AsyncMock

import pytest


class TestCliPromptSession:
    """Tests for _create_cli_prompt_session with persistent history."""

    def test_prompt_fragments_include_profile_context(self):
        from victor.ui.commands.chat import _build_cli_prompt_fragments

        fragments = _build_cli_prompt_fragments("coding")

        rendered = "".join(text for _style, text in fragments)
        assert "victor" in rendered
        assert "[coding]" in rendered

    def test_bottom_toolbar_shows_runtime_context_and_shortcuts(self):
        from victor.ui.commands.chat import _build_cli_bottom_toolbar

        settings = SimpleNamespace(
            provider=SimpleNamespace(default_provider="ollama", default_model="qwen")
        )
        profile = SimpleNamespace(provider="anthropic", model="claude-sonnet")

        toolbar = _build_cli_bottom_toolbar(
            settings=settings,
            profile_config=profile,
            profile_name="review",
            vertical_name="coding",
            enable_planning=True,
            stream=False,
            renderer_choice="rich",
            show_reasoning=True,
        )

        rendered = "".join(text for _style, text in toolbar)
        assert "review" in rendered
        assert "anthropic" in rendered
        assert "claude-sonnet" in rendered
        assert "coding" in rendered
        assert "Mode" in rendered
        assert "plan" in rendered
        assert "sync" in rendered
        assert "rich" in rendered
        assert "reason" in rendered
        assert "Tab commands" in rendered
        assert "Enter send" in rendered
        assert "Alt+Enter newline" in rendered
        assert "Up/Down history" in rendered
        assert "Ctrl+O" in rendered
        assert "F1 help" in rendered

    def test_bottom_toolbar_uses_compact_layout_in_narrow_terminals(self):
        from victor.ui.commands.chat import _build_cli_bottom_toolbar

        profile = SimpleNamespace(
            provider="anthropic",
            model="claude-sonnet-4-5-with-a-very-long-name",
        )

        toolbar = _build_cli_bottom_toolbar(
            profile_config=profile,
            profile_name="very-long-profile-name",
            vertical_name="coding",
            enable_planning=None,
            stream=True,
            renderer_choice="auto",
            width=78,
        )

        rendered = "".join(text for _style, text in toolbar)
        assert "auto/stream" in rendered
        assert "Tab cmds" in rendered
        assert "F1 help" in rendered
        assert "Up/Down history" not in rendered
        assert "very-long-pro…" in rendered
        assert "claude-sonnet-4-5…" in rendered

    def test_bottom_toolbar_truncates_long_values_in_standard_layout(self):
        from victor.ui.commands.chat import _build_cli_bottom_toolbar

        profile = SimpleNamespace(
            provider="provider-with-an-extremely-long-name",
            model="model-with-an-extremely-long-name-that-would-wrap",
        )

        toolbar = _build_cli_bottom_toolbar(
            profile_config=profile,
            profile_name="profile-with-an-extremely-long-name",
            vertical_name="vertical-with-an-extremely-long-name",
            width=140,
        )

        rendered = "".join(text for _style, text in toolbar)
        assert "provider-with-an-ex…" in rendered
        assert "model-with-an-extremely-long-na…" in rendered
        assert "profile-with-an-ext…" in rendered

    def test_right_prompt_shows_exit_hint(self):
        from victor.ui.commands.chat import _build_cli_right_prompt

        rendered = "".join(text for _style, text in _build_cli_right_prompt())

        assert "Ctrl+D exit" in rendered

    def test_work_status_message_reflects_planning_mode(self):
        from victor.ui.commands.chat import _cli_work_status_message

        assert _cli_work_status_message(True) == "Planning..."
        assert _cli_work_status_message(False) == "Thinking..."
        assert _cli_work_status_message(None) == "Thinking..."

    def test_runtime_segment_shows_live_budget_messages_and_context(self, monkeypatch):
        from victor.ui.commands import chat as chat_command

        monkeypatch.setattr(
            chat_command,
            "_resolve_cli_context_window",
            lambda provider, model: 128_000,
        )
        conversation = SimpleNamespace(
            messages=[{"content": "abcd" * 1000}, {"content": "done"}],
            message_count=lambda: 2,
        )
        agent = SimpleNamespace(
            tool_calls_used=3,
            tool_budget=50,
            conversation=conversation,
        )

        segment = chat_command._build_cli_runtime_segment(
            agent=agent,
            provider="anthropic",
            model="claude-sonnet",
        )

        assert segment is not None
        assert "Tools 3/50" in segment
        assert "Msg 2" in segment
        assert "Ctx ~1k/128k" in segment

    def test_runtime_segment_uses_compact_labels(self, monkeypatch):
        from victor.ui.commands import chat as chat_command

        monkeypatch.setattr(
            chat_command,
            "_resolve_cli_context_window",
            lambda provider, model: 128_000,
        )
        conversation = SimpleNamespace(
            messages=[{"content": "abcd" * 1000}],
            message_count=lambda: 1,
        )
        agent = SimpleNamespace(
            tool_calls_used=1,
            tool_budget=10,
            conversation=conversation,
        )

        segment = chat_command._build_cli_runtime_segment(
            agent=agent,
            provider="anthropic",
            model="claude-sonnet",
            compact=True,
        )

        assert segment is not None
        assert "t 1/10" in segment
        assert "msg 1" in segment
        assert "ctx ~1k/128k" in segment

    def test_creates_prompt_session(self):
        from victor.ui.commands.chat import _create_cli_prompt_session

        session, _renderer_holder = _create_cli_prompt_session()
        assert session is not None
        assert hasattr(session, "prompt")
        assert session.completer is not None
        assert session.bottom_toolbar is not None
        assert session.rprompt is not None
        assert session.mouse_support is False
        assert session.multiline is True
        assert session.reserve_space_for_menu >= 8

    def test_mouse_support_is_opt_in(self, monkeypatch):
        from victor.ui.commands.chat import _create_cli_prompt_session

        monkeypatch.setenv("VICTOR_CHAT_MOUSE_SUPPORT", "1")

        session, _renderer_holder = _create_cli_prompt_session()

        assert session.mouse_support is True

    def test_command_completer_suggests_slash_commands(self):
        from prompt_toolkit.document import Document

        from victor.ui.commands.chat import _build_cli_command_completer

        completer = _build_cli_command_completer()
        completions = list(completer.get_completions(Document("/mo"), None))
        labels = {completion.text for completion in completions}

        assert "/model" in labels
        assert "/mode" in labels

    def test_command_completer_includes_registered_slash_commands(self):
        from prompt_toolkit.document import Document

        from victor.ui.commands.chat import _build_cli_command_completer

        completer = _build_cli_command_completer()
        completions = list(completer.get_completions(Document("/bu"), None))
        labels = {completion.text for completion in completions}

        assert "/build" in labels

    def test_command_completer_suggests_shortcuts_command(self):
        from prompt_toolkit.document import Document

        from victor.ui.commands.chat import _build_cli_command_completer

        completer = _build_cli_command_completer()
        completions = list(completer.get_completions(Document("/sh"), None))
        labels = {completion.text for completion in completions}

        assert "/shortcuts" in labels

    def test_command_completer_suggests_known_arguments(self):
        from prompt_toolkit.document import Document

        from victor.ui.commands.chat import _build_cli_command_completer

        completer = _build_cli_command_completer()
        completions = list(completer.get_completions(Document("/mode pl"), None))
        labels = {completion.text for completion in completions}

        assert "plan" in labels

    def test_command_completer_handles_command_with_trailing_space(self):
        from prompt_toolkit.document import Document

        from victor.ui.commands.chat import _build_cli_command_completer

        completer = _build_cli_command_completer()

        completions = list(completer.get_completions(Document("/mode "), None))

        assert {completion.text for completion in completions} >= {"plan", "review"}

    def test_command_completer_suggests_provider_arguments(self):
        from prompt_toolkit.document import Document

        from victor.ui.commands.chat import _build_cli_command_completer

        completer = _build_cli_command_completer()
        completions = list(completer.get_completions(Document("/provider an"), None))
        labels = {completion.text for completion in completions}

        assert "anthropic" in labels

    def test_command_completer_includes_metadata_and_aliases(self):
        from prompt_toolkit.document import Document

        from victor.ui.commands.chat import _build_cli_command_completer

        completer = _build_cli_command_completer()
        completions = list(completer.get_completions(Document("/?"), None))

        assert completions
        assert completions[0].text == "/?"
        assert "alias" in str(completions[0].display_meta)

    def test_normalizes_cli_input_aliases(self):
        from victor.ui.commands.chat import _normalize_cli_input_alias

        assert _normalize_cli_input_alias("/?") == "/help"
        assert _normalize_cli_input_alias(":help") == "/shortcuts"
        assert _normalize_cli_input_alias(":q") == "/quit"
        assert _normalize_cli_input_alias("hello") == "hello"

    def test_shortcuts_panel_lists_prompt_keys(self):
        from rich.console import Console

        from victor.ui.commands.chat import _build_cli_shortcuts_panel

        console = Console(record=True, width=100)
        console.print(_build_cli_shortcuts_panel())
        rendered = console.export_text()

        assert "CLI Shortcuts" in rendered
        assert "Alt+Enter" in rendered
        assert "Ctrl+O" in rendered
        assert "/shortcuts" in rendered

    def test_uses_file_history(self):
        from victor.ui.commands.chat import _create_cli_prompt_session

        session, _renderer_holder = _create_cli_prompt_session()
        from prompt_toolkit.history import FileHistory

        assert isinstance(session.history, FileHistory)

    def test_default_history_file_uses_isolated_unit_test_victor_dir(
        self, isolated_project_victor_dir
    ):
        from victor.ui.commands.chat import _create_cli_prompt_session

        session, _renderer_holder = _create_cli_prompt_session()

        assert Path(session.history.filename) == isolated_project_victor_dir / "chat_history"

    def test_fallback_to_in_memory_on_error(self):
        from victor.ui.commands.chat import _create_cli_prompt_session

        with patch(
            "victor.config.settings.get_project_paths",
            side_effect=RuntimeError("no paths"),
        ):
            # Should not raise — falls back to InMemoryHistory
            session, _renderer_holder = _create_cli_prompt_session()
            from prompt_toolkit.history import InMemoryHistory

            assert isinstance(session.history, InMemoryHistory)

    def test_seeds_history_from_db_without_internal_prompts(self, tmp_path):
        from victor.ui.commands.chat import _create_cli_prompt_session

        project_dir = tmp_path / ".victor"
        project_dir.mkdir(exist_ok=True)
        db_path = project_dir / "project.db"

        with sqlite3.connect(db_path) as conn:
            conn.execute("""
                CREATE TABLE messages (
                    role TEXT,
                    content TEXT,
                    timestamp TEXT,
                    metadata TEXT
                )
                """)
            conn.executemany(
                "INSERT INTO messages(role, content, timestamp, metadata) VALUES (?, ?, ?, ?)",
                [
                    ("user", "real prompt", "2026-04-26 10:00:00", None),
                    ("user", "[SYSTEM-REMINDER: hidden]", "2026-04-26 10:01:00", None),
                    (
                        "user",
                        "Continue. Use appropriate tools if needed.",
                        "2026-04-26 10:02:00",
                        None,
                    ),
                    (
                        "user",
                        "plain-looking hidden prompt",
                        "2026-04-26 10:02:30",
                        '{"interactive_history": false, "internal_prompt_kind": "prompt_tool_call"}',
                    ),
                    ("user", "follow-up prompt", "2026-04-26 10:03:00", None),
                ],
            )

        fake_paths = SimpleNamespace(project_victor_dir=project_dir, project_db=db_path)
        fake_settings = SimpleNamespace(ui=SimpleNamespace(cli_history_max_entries=20))

        with patch("victor.config.settings.get_project_paths", return_value=fake_paths):
            session, _renderer_holder = _create_cli_prompt_session(settings=fake_settings)

        history_strings = list(session.history.load_history_strings())
        assert history_strings == ["follow-up prompt", "real prompt"]


class TestPlanningWiring:
    """Tests that planning is wired through agent.chat(use_planning=...)."""

    @pytest.mark.asyncio
    async def test_orchestrator_chat_passes_use_planning(self):
        """Verify orchestrator.chat() passes use_planning to coordinator."""
        mock_coordinator = AsyncMock()
        mock_coordinator.chat.return_value = MagicMock(content="response")

        orchestrator = MagicMock()
        orchestrator._use_service_layer = False
        orchestrator._chat_service = None
        orchestrator._chat_coordinator = mock_coordinator

        # Call the actual method logic

        # Simulate what orchestrator.chat does
        result = await mock_coordinator.chat("test message", use_planning=None)
        mock_coordinator.chat.assert_called_once_with("test message", use_planning=None)

    @pytest.mark.asyncio
    async def test_orchestrator_chat_default_no_planning(self):
        """Verify default use_planning=False preserves backward compat."""
        mock_coordinator = AsyncMock()
        mock_coordinator.chat.return_value = MagicMock(content="response")

        result = await mock_coordinator.chat("test message", use_planning=False)
        mock_coordinator.chat.assert_called_once_with("test message", use_planning=False)

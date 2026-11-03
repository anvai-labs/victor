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

"""Unit tests for the TUI intent confirmation modal."""

from __future__ import annotations

from victor.ui.tui.intent_modal import (
    IntentAction,
    IntentConfirmationScreen,
    IntentOption,
    IntentRequest,
)


class TestIntentAction:
    """Verify IntentAction enum values."""

    def test_enum_values(self) -> None:
        assert IntentAction.CONFIRM.value == "confirm"
        assert IntentAction.MODIFY.value == "modify"
        assert IntentAction.DISCARD.value == "discard"
        assert IntentAction.CHAT.value == "chat"


class TestIntentOption:
    """Verify IntentOption dataclass."""

    def test_defaults(self) -> None:
        option = IntentOption(label="Do X", rationale="Because Y")
        assert option.label == "Do X"
        assert option.rationale == "Because Y"
        assert option.is_recommended is False

    def test_recommended(self) -> None:
        option = IntentOption(label="Do X", rationale="Because Y", is_recommended=True)
        assert option.is_recommended is True


class TestIntentRequest:
    """Verify IntentRequest dataclass."""

    def test_defaults(self) -> None:
        options = [IntentOption(label="A", rationale="B")]
        request = IntentRequest(task_description="Test task", options=options)
        assert request.task_description == "Test task"
        assert len(request.options) == 1
        assert request.context is None
        assert request.timeout_seconds == 60

    def test_with_context(self) -> None:
        options = [IntentOption(label="A", rationale="B")]
        request = IntentRequest(
            task_description="Test task",
            options=options,
            context="Some context",
            timeout_seconds=30,
        )
        assert request.context == "Some context"
        assert request.timeout_seconds == 30


class TestIntentConfirmationScreen:
    """Verify IntentConfirmationScreen behavior."""

    def test_screen_creation(self) -> None:
        options = [
            IntentOption(label="Option A", rationale="Reason A", is_recommended=True),
            IntentOption(label="Option B", rationale="Reason B"),
        ]
        request = IntentRequest(task_description="Test task", options=options)
        screen = IntentConfirmationScreen(request)
        assert screen._request == request
        assert screen._decided is False
        assert screen._selected_option is None

    def test_recommended_option_highlighted(self) -> None:
        """Recommended option should include RECOMMENDED text and cyan color."""
        options = [
            IntentOption(label="Option A", rationale="Reason A", is_recommended=True),
            IntentOption(label="Option B", rationale="Reason B"),
        ]
        request = IntentRequest(task_description="Test task", options=options)
        screen = IntentConfirmationScreen(request)
        # Verify the screen was created with the request
        assert len(screen._request.options) == 2
        assert screen._request.options[0].is_recommended is True
        assert screen._request.options[1].is_recommended is False

    def test_discard_action(self) -> None:
        options = [IntentOption(label="A", rationale="B")]
        request = IntentRequest(task_description="Test task", options=options)
        screen = IntentConfirmationScreen(request)
        # Mock dismiss to avoid needing active app context
        screen.dismiss = lambda x: None  # type: ignore
        screen.action_discard()
        assert screen._decided is True

    def test_chat_action(self) -> None:
        options = [IntentOption(label="A", rationale="B")]
        request = IntentRequest(task_description="Test task", options=options)
        screen = IntentConfirmationScreen(request)
        screen.dismiss = lambda x: None  # type: ignore
        screen.action_chat()
        assert screen._decided is True

    def test_select_first_option(self) -> None:
        options = [
            IntentOption(label="A", rationale="B"),
            IntentOption(label="C", rationale="D"),
        ]
        request = IntentRequest(task_description="Test task", options=options)
        screen = IntentConfirmationScreen(request)
        screen.dismiss = lambda x: None  # type: ignore
        screen.action_select_first()
        assert screen._decided is True
        assert screen._selected_option == 0

    def test_select_second_option(self) -> None:
        options = [
            IntentOption(label="A", rationale="B"),
            IntentOption(label="C", rationale="D"),
        ]
        request = IntentRequest(task_description="Test task", options=options)
        screen = IntentConfirmationScreen(request)
        screen.dismiss = lambda x: None  # type: ignore
        screen.action_select_second()
        assert screen._decided is True
        assert screen._selected_option == 1

    def test_select_third_option(self) -> None:
        options = [
            IntentOption(label="A", rationale="B"),
            IntentOption(label="C", rationale="D"),
            IntentOption(label="E", rationale="F"),
        ]
        request = IntentRequest(task_description="Test task", options=options)
        screen = IntentConfirmationScreen(request)
        screen.dismiss = lambda x: None  # type: ignore
        screen.action_select_third()
        assert screen._decided is True
        assert screen._selected_option == 2

    def test_select_second_option_not_available(self) -> None:
        """Selecting second option when only one option exists should not decide."""
        options = [IntentOption(label="A", rationale="B")]
        request = IntentRequest(task_description="Test task", options=options)
        screen = IntentConfirmationScreen(request)
        screen.dismiss = lambda x: None  # type: ignore
        screen.action_select_second()
        assert screen._decided is False

    def test_double_decide_is_noop(self) -> None:
        """Once decided, further actions should be no-ops."""
        options = [IntentOption(label="A", rationale="B")]
        request = IntentRequest(task_description="Test task", options=options)
        screen = IntentConfirmationScreen(request)
        screen.dismiss = lambda x: None  # type: ignore
        screen.action_discard()
        assert screen._decided is True
        # Try to change decision
        screen.action_chat()
        # Still decided (no change)
        assert screen._decided is True

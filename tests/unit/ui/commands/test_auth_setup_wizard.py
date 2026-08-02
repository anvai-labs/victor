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

"""AuthSetupWizard: menu index mapping, retry loop, and first-run mode.

Regression coverage for the 1-based menu → 0-based list off-by-one (selecting
"1. Anthropic" used to configure OpenAI, and the last menu entry crashed with
IndexError), the connection-test retry loop ("Try again?" used to abort), and
the ``first_run`` mode used by the onboarding flow.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from victor.ui.commands.auth import POPULAR_MODELS, AuthSetupWizard


def _make_wizard(first_run=False):
    console = Console(record=True, force_terminal=False, width=100)
    with patch("victor.ui.commands.auth.get_account_manager"):
        return AuthSetupWizard(console, first_run=first_run)


class TestProviderMenuMapping:
    """Menus are displayed 1-based; the selection must map back correctly."""

    @pytest.mark.parametrize(
        "choice,expected",
        [("1", "anthropic"), ("2", "openai"), ("3", "google"), ("4", "zai")],
    )
    def test_provider_choice_maps_to_displayed_entry(self, choice, expected):
        wizard = _make_wizard()
        wizard.state["provider_type"] = "cloud"

        # Second answer selects model "1" from POPULAR_MODELS.
        with patch("victor.ui.commands.auth.Prompt.ask", side_effect=[choice, "1"]):
            assert wizard._select_provider_and_model() is True

        assert wizard.state["selected_provider"] == expected
        assert wizard.state["selected_model"] == POPULAR_MODELS[expected][0]

    def test_last_model_entry_does_not_crash(self):
        wizard = _make_wizard()
        wizard.state["provider_type"] = "cloud"
        last = str(len(POPULAR_MODELS["anthropic"]))

        with patch("victor.ui.commands.auth.Prompt.ask", side_effect=["1", last]):
            assert wizard._select_provider_and_model() is True

        assert wizard.state["selected_model"] == POPULAR_MODELS["anthropic"][-1]

    def test_zero_prompts_for_manual_model(self):
        wizard = _make_wizard()
        wizard.state["provider_type"] = "cloud"

        with patch(
            "victor.ui.commands.auth.Prompt.ask",
            side_effect=["1", "0", "claude-custom"],
        ):
            assert wizard._select_provider_and_model() is True

        assert wizard.state["selected_model"] == "claude-custom"


class TestConnectionTestOutcomes:
    def _run_test(self, success, retry_answer=False):
        wizard = _make_wizard()
        wizard.state.update(
            {
                "selected_provider": "anthropic",
                "selected_model": "claude-sonnet-4-6",
                "auth_method": "api_key",
                "api_key": "sk-test",
            }
        )
        result = MagicMock(success=success, error=None if success else "bad key")
        with patch("victor.ui.commands.auth.ConnectionValidator") as mock_validator:
            mock_validator.return_value.test_account_sync.return_value = result
            with patch("victor.ui.commands.auth.Confirm.ask", return_value=retry_answer):
                return wizard._test_connection(), wizard

    def test_success_returns_ok(self):
        outcome, _ = self._run_test(success=True)
        assert outcome == "ok"

    def test_failure_with_retry_returns_retry(self):
        outcome, _ = self._run_test(success=False, retry_answer=True)
        assert outcome == "retry"

    def test_failure_without_retry_aborts_with_doctor_hint(self):
        outcome, wizard = self._run_test(success=False, retry_answer=False)
        assert outcome == "abort"
        assert "victor doctor" in wizard.console.export_text()


class TestRetryLoop:
    def test_retry_loops_back_to_credential_entry(self):
        wizard = _make_wizard()

        with (
            patch.object(wizard, "_check_migration", return_value=False),
            patch.object(wizard, "_detect_environment"),
            patch.object(wizard, "_choose_provider_type", return_value=True),
            patch.object(wizard, "_select_provider_and_model", return_value=True),
            patch.object(wizard, "_configure_authentication", return_value=True) as mock_auth,
            patch.object(wizard, "_name_account", return_value=True),
            patch.object(wizard, "_test_connection", side_effect=["retry", "ok"]),
            patch.object(wizard, "_save_account") as mock_save,
        ):
            assert wizard.run() == 0

        assert mock_auth.call_count == 2
        mock_save.assert_called_once()

    def test_abort_exits_without_saving(self):
        wizard = _make_wizard()

        with (
            patch.object(wizard, "_check_migration", return_value=False),
            patch.object(wizard, "_detect_environment"),
            patch.object(wizard, "_choose_provider_type", return_value=True),
            patch.object(wizard, "_select_provider_and_model", return_value=True),
            patch.object(wizard, "_configure_authentication", return_value=True),
            patch.object(wizard, "_name_account", return_value=True),
            patch.object(wizard, "_test_connection", return_value="abort"),
            patch.object(wizard, "_save_account") as mock_save,
        ):
            assert wizard.run() == 0

        mock_save.assert_not_called()


class TestFirstRunMode:
    def test_name_account_auto_accepts_suggestion(self):
        wizard = _make_wizard(first_run=True)
        wizard.state["selected_provider"] = "anthropic"
        wizard.state["selected_model"] = "claude-sonnet-4-6"

        with patch("victor.ui.commands.auth.Prompt.ask") as mock_prompt:
            assert wizard._name_account() is True

        mock_prompt.assert_not_called()
        assert wizard.state["account_name"] == "anthropic"
        assert wizard.state["tags"] == []

    def test_name_account_uses_variant_suffix(self):
        wizard = _make_wizard(first_run=True)
        wizard.state["selected_provider"] = "zai"
        wizard.state["selected_model"] = "glm-4.6:coding"

        assert wizard._name_account() is True
        assert wizard.state["account_name"] == "zai-coding"

    def test_first_run_skips_welcome_and_completion(self):
        wizard = _make_wizard(first_run=True)

        with (
            patch.object(wizard, "_show_welcome") as mock_welcome,
            patch.object(wizard, "_check_migration", return_value=False),
            patch.object(wizard, "_detect_environment"),
            patch.object(wizard, "_choose_provider_type", return_value=True),
            patch.object(wizard, "_select_provider_and_model", return_value=True),
            patch.object(wizard, "_configure_authentication", return_value=True),
            patch.object(wizard, "_name_account", return_value=True),
            patch.object(wizard, "_test_connection", return_value="ok"),
            patch.object(wizard, "_save_account"),
            patch.object(wizard, "_show_completion") as mock_completion,
        ):
            assert wizard.run() == 0

        mock_welcome.assert_not_called()
        mock_completion.assert_not_called()

    def test_default_mode_shows_welcome_and_completion(self):
        wizard = _make_wizard(first_run=False)

        with (
            patch.object(wizard, "_show_welcome") as mock_welcome,
            patch.object(wizard, "_check_migration", return_value=False),
            patch.object(wizard, "_detect_environment"),
            patch.object(wizard, "_choose_provider_type", return_value=True),
            patch.object(wizard, "_select_provider_and_model", return_value=True),
            patch.object(wizard, "_configure_authentication", return_value=True),
            patch.object(wizard, "_name_account", return_value=True),
            patch.object(wizard, "_test_connection", return_value="ok"),
            patch.object(wizard, "_save_account"),
            patch.object(wizard, "_show_completion") as mock_completion,
        ):
            assert wizard.run() == 0

        mock_welcome.assert_called_once()
        mock_completion.assert_called_once()


class TestSavedAccountExposure:
    def test_save_account_stashes_saved_account_in_state(self):
        wizard = _make_wizard(first_run=True)
        wizard.state.update(
            {
                "selected_provider": "anthropic",
                "selected_model": "claude-sonnet-4-6",
                "auth_method": "api_key",
                "api_key": "sk-test",
                "account_name": "anthropic",
                "tags": [],
            }
        )
        wizard.account_manager = MagicMock()
        wizard.account_manager.list_accounts.return_value = [MagicMock()]

        with (
            patch("victor.ui.commands.auth._sync_profile_from_account", return_value=True),
            patch("victor.config.api_keys._set_key_in_keyring"),
        ):
            wizard._save_account()

        saved = wizard.state["saved_account"]
        assert saved is not None
        assert saved.provider == "anthropic"
        assert saved.model == "claude-sonnet-4-6"

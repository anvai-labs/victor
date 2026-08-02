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

"""UX P4: first-run onboarding sequences auth setup → default profile → next steps.

The wizard no longer owns provider/model/credential logic — that is delegated
to ``AuthSetupWizard`` (the ``victor auth setup`` core). Onboarding frames it,
installs the recommended default profile pointing at the chosen provider,
writes the completion marker, and shows example prompts. Failures surface a
``victor doctor`` hint inline.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from victor.ui.commands.onboarding import (
    EXAMPLE_PROMPTS,
    OnboardingWizard,
    run_onboarding,
)


def _make_wizard(tmp_path, offer_chat=False):
    """Build a wizard with an isolated config dir and recording console."""
    console = Console(record=True, force_terminal=False, width=100)
    wizard = OnboardingWizard(console, offer_chat=offer_chat)
    wizard.config_dir = tmp_path
    return wizard


def _fake_auth_wizard(exit_code=0, provider="anthropic", model="claude-sonnet-4-6", account=...):
    """Build a stand-in for AuthSetupWizard with a preloaded result state."""
    if account is ...:
        account = SimpleNamespace(name=provider, provider=provider, model=model, auth=None)
    fake = MagicMock()
    fake.run.return_value = exit_code
    fake.state = {
        "selected_provider": provider,
        "selected_model": model,
        "saved_account": account,
    }
    return fake


class TestOnboardingWizardInit:
    def test_init_defaults(self):
        from victor.ui.commands.onboarding import get_project_paths

        wizard = OnboardingWizard()
        assert wizard.console is not None
        assert wizard.offer_chat is True
        # config_dir resolves through centralized Victor paths (secure,
        # $HOME-independent), not Path.home() directly.
        assert wizard.config_dir == get_project_paths().global_victor_dir

    def test_init_offer_chat_false(self):
        wizard = OnboardingWizard(offer_chat=False)
        assert wizard.offer_chat is False


class TestRunFlow:
    @patch("victor.ui.commands.onboarding.Confirm.ask", return_value=True)
    @patch("victor.ui.commands.auth.AuthSetupWizard")
    def test_full_run_success(self, mock_wizard_cls, _confirm, tmp_path):
        fake = _fake_auth_wizard()
        mock_wizard_cls.return_value = fake
        wizard = _make_wizard(tmp_path)

        with patch.object(wizard, "_install_default_profile") as mock_install:
            result = wizard.run()

        assert result == 0
        mock_wizard_cls.assert_called_once_with(wizard.console, first_run=True)
        fake.run.assert_called_once()
        mock_install.assert_called_once_with(
            "anthropic", "claude-sonnet-4-6", fake.state["saved_account"]
        )
        assert (tmp_path / ".onboarding_completed").exists()

    @patch("victor.ui.commands.onboarding.Confirm.ask", return_value=False)
    def test_cancel_at_confirm(self, _confirm, tmp_path):
        wizard = _make_wizard(tmp_path)
        result = wizard.run()
        assert result == 0
        assert not (tmp_path / ".onboarding_completed").exists()

    @patch("victor.ui.commands.onboarding.Confirm.ask", return_value=True)
    @patch("victor.ui.commands.auth.AuthSetupWizard")
    def test_auth_wizard_failure_shows_doctor_hint(self, mock_wizard_cls, _confirm, tmp_path):
        fake = _fake_auth_wizard(exit_code=1)
        mock_wizard_cls.return_value = fake
        wizard = _make_wizard(tmp_path)

        with patch.object(wizard, "_show_doctor_hint") as mock_hint:
            result = wizard.run()

        assert result == 1
        mock_hint.assert_called_once()
        assert not (tmp_path / ".onboarding_completed").exists()

    @patch("victor.ui.commands.onboarding.Confirm.ask", return_value=True)
    @patch("victor.ui.commands.auth.AuthSetupWizard")
    def test_auth_wizard_backed_out_is_clean_cancel(self, mock_wizard_cls, _confirm, tmp_path):
        # Auth wizard returns 0 for user-cancel, with nothing saved.
        fake = _fake_auth_wizard(provider=None, account=None)
        mock_wizard_cls.return_value = fake
        wizard = _make_wizard(tmp_path)

        result = wizard.run()

        assert result == 0
        assert not (tmp_path / ".onboarding_completed").exists()

    @patch("victor.ui.commands.onboarding.Confirm.ask", return_value=True)
    @patch("victor.ui.commands.auth.AuthSetupWizard")
    def test_exception_shows_doctor_hint(self, mock_wizard_cls, _confirm, tmp_path):
        mock_wizard_cls.return_value.run.side_effect = RuntimeError("boom")
        wizard = _make_wizard(tmp_path)

        with patch.object(wizard, "_show_doctor_hint") as mock_hint:
            result = wizard.run()

        assert result == 1
        mock_hint.assert_called_once()


class TestDefaultProfileInstall:
    @patch("victor.ui.commands.onboarding.ProfileManager")
    @patch("victor.ui.commands.onboarding.install_profile")
    @patch("victor.ui.commands.onboarding.get_recommended_profile")
    def test_installs_recommended_with_overrides(
        self, mock_recommended, mock_install, mock_pm, tmp_path
    ):
        mock_recommended.return_value = SimpleNamespace(name="basic")
        mock_install.return_value = tmp_path / "profiles.yaml"
        wizard = _make_wizard(tmp_path)
        account = SimpleNamespace(name="anthropic")

        wizard._install_default_profile("anthropic", "claude-sonnet-4-6", account)

        mock_install.assert_called_once_with(
            mock_recommended.return_value,
            config_dir=tmp_path,
            provider_override="anthropic",
            model_override="claude-sonnet-4-6",
        )
        # Account-named chat profile is re-synced after the wholesale rewrite.
        mock_pm.for_config_dir.assert_called_once_with(tmp_path)
        mock_pm.for_config_dir.return_value.upsert_account_profile.assert_called_once_with(account)

    @patch("victor.ui.commands.onboarding.ProfileManager")
    @patch("victor.ui.commands.onboarding.install_profile")
    @patch("victor.ui.commands.onboarding.get_recommended_profile")
    def test_local_placeholder_model_not_overridden(
        self, mock_recommended, mock_install, _mock_pm, tmp_path
    ):
        mock_recommended.return_value = SimpleNamespace(name="basic")
        mock_install.return_value = tmp_path / "profiles.yaml"
        wizard = _make_wizard(tmp_path)

        wizard._install_default_profile("ollama", "default", SimpleNamespace(name="ollama"))

        assert mock_install.call_args.kwargs["model_override"] is None


class TestCompletionMarker:
    def test_marker_records_provider_and_model(self, tmp_path):
        wizard = _make_wizard(tmp_path)
        wizard._write_completion_marker("anthropic", "claude-sonnet-4-6")

        content = (tmp_path / ".onboarding_completed").read_text()
        assert "anthropic" in content
        assert "claude-sonnet-4-6" in content


class TestCompletionScreen:
    def test_shows_example_prompts_and_next_steps(self, tmp_path):
        wizard = _make_wizard(tmp_path, offer_chat=False)
        wizard._show_completion("anthropic", "claude-sonnet-4-6")

        output = wizard.console.export_text()
        for prompt in EXAMPLE_PROMPTS:
            assert prompt in output
        assert "victor examples" in output
        assert "victor doctor" in output

    @patch("victor.ui.commands.onboarding.Confirm.ask", return_value=True)
    def test_offer_chat_starts_first_chat(self, _confirm, tmp_path):
        wizard = _make_wizard(tmp_path, offer_chat=True)
        with patch.object(wizard, "_start_first_chat") as mock_chat:
            wizard._show_completion("anthropic", "claude-sonnet-4-6")
        mock_chat.assert_called_once()

    def test_offer_chat_false_skips_prompt(self, tmp_path):
        wizard = _make_wizard(tmp_path, offer_chat=False)
        with patch.object(wizard, "_start_first_chat") as mock_chat:
            with patch("victor.ui.commands.onboarding.Confirm.ask") as mock_confirm:
                wizard._show_completion("anthropic", "claude-sonnet-4-6")
        mock_chat.assert_not_called()
        mock_confirm.assert_not_called()


class TestDoctorHint:
    def test_hint_mentions_doctor(self, tmp_path):
        wizard = _make_wizard(tmp_path)
        wizard._show_doctor_hint()
        assert "victor doctor" in wizard.console.export_text()

    def test_hint_survives_doctor_failure(self, tmp_path):
        wizard = _make_wizard(tmp_path)
        with patch("victor.ui.commands.doctor.DoctorChecks", side_effect=RuntimeError("no doctor")):
            wizard._show_doctor_hint()
        assert "victor doctor" in wizard.console.export_text()


class TestRunOnboarding:
    @patch("victor.ui.commands.onboarding.OnboardingWizard")
    def test_entry_point_passes_offer_chat(self, mock_wizard_cls):
        mock_wizard_cls.return_value.run.return_value = 0

        assert run_onboarding(offer_chat=False) == 0

        _, kwargs = mock_wizard_cls.call_args
        assert kwargs["offer_chat"] is False

    @patch("victor.ui.commands.onboarding.OnboardingWizard")
    def test_entry_point_handles_wizard_init_exception(self, mock_wizard_cls):
        mock_wizard_cls.side_effect = RuntimeError("init failed")
        assert run_onboarding() == 1


class TestFirstTimeUserDetection:
    """is_first_time_user must treat keyring-only auth-setup users as configured."""

    @pytest.fixture
    def fake_home(self, tmp_path, monkeypatch):
        from pathlib import Path

        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        victor_dir = tmp_path / ".victor"
        victor_dir.mkdir(exist_ok=True)
        return victor_dir

    def _is_first_time(self):
        from victor.config.settings import is_first_time_user

        return is_first_time_user()

    def test_marker_wins(self, fake_home):
        (fake_home / ".onboarding_completed").touch()
        assert self._is_first_time() is False

    def test_accounts_config_counts_as_configured(self, fake_home):
        # `victor auth setup` writes config.yaml + keyring, no env keys.
        (fake_home / "config.yaml").write_text("accounts: []\n")
        assert self._is_first_time() is False

    def test_empty_home_is_first_time(self, fake_home):
        assert self._is_first_time() is True

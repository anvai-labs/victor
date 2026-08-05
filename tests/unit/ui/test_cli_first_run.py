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

"""UX P4: bare ``victor`` first-run wiring.

First-time users get the onboarding wizard (with ``offer_chat=False`` since
this path always drops into chat afterwards); ``--no-setup`` /
``--skip-onboarding`` bypass it; configured users go straight to chat.
"""

from __future__ import annotations

from unittest.mock import patch

from typer.testing import CliRunner

from victor.ui.cli import app

runner = CliRunner()


class TestFirstRunFlow:
    @patch("victor.ui.cli._run_default_interactive")
    @patch("victor.ui.commands.onboarding.run_onboarding", return_value=0)
    @patch("victor.config.settings.is_first_time_user", return_value=True)
    def test_first_time_user_gets_wizard_then_chat(self, _first, mock_onboard, mock_chat):
        result = runner.invoke(app, [])
        assert result.exit_code == 0
        mock_onboard.assert_called_once_with(offer_chat=False)
        mock_chat.assert_called_once()

    @patch("victor.ui.cli._run_default_interactive")
    @patch("victor.ui.commands.onboarding.run_onboarding")
    @patch("victor.config.settings.is_first_time_user", return_value=False)
    def test_configured_user_goes_straight_to_chat(self, _first, mock_onboard, mock_chat):
        result = runner.invoke(app, [])
        assert result.exit_code == 0
        mock_onboard.assert_not_called()
        mock_chat.assert_called_once()

    @patch("victor.ui.cli._run_default_interactive")
    @patch("victor.ui.commands.onboarding.run_onboarding")
    @patch("victor.config.settings.is_first_time_user", return_value=True)
    def test_no_setup_flag_skips_wizard(self, _first, mock_onboard, mock_chat):
        result = runner.invoke(app, ["--no-setup"])
        assert result.exit_code == 0
        mock_onboard.assert_not_called()
        mock_chat.assert_called_once()

    @patch("victor.ui.cli._run_default_interactive")
    @patch("victor.ui.commands.onboarding.run_onboarding")
    @patch("victor.config.settings.is_first_time_user", return_value=True)
    def test_skip_onboarding_flag_still_works(self, _first, mock_onboard, mock_chat):
        result = runner.invoke(app, ["--skip-onboarding"])
        assert result.exit_code == 0
        mock_onboard.assert_not_called()
        mock_chat.assert_called_once()

    @patch("victor.ui.cli._run_default_interactive")
    @patch("victor.ui.commands.onboarding.run_onboarding", return_value=1)
    @patch("victor.config.settings.is_first_time_user", return_value=True)
    def test_interrupted_wizard_still_starts_chat(self, _first, mock_onboard, mock_chat):
        result = runner.invoke(app, [])
        assert result.exit_code == 0
        mock_chat.assert_called_once()

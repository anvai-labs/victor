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

"""SessionConfig must thread the --headless CLI flag through to settings.headless.

This covers the config-threading seam of headless mode (CLI flag -> SessionConfig ->
settings.headless.headless_mode); prompt-section inclusion from the resulting builder is
covered separately in tests/unit/agent/test_headless_prompt.py.
"""

from types import SimpleNamespace

from victor.config.groups.headless_config import HeadlessSettings
from victor.framework.session_config import SessionConfig


def test_headless_flag_threads_into_settings_headless_group():
    settings = SimpleNamespace(headless=HeadlessSettings())
    assert settings.headless.headless_mode is False

    SessionConfig.from_cli_flags(headless_mode=True).apply_to_settings(settings)

    assert settings.headless.headless_mode is True


def test_headless_default_does_not_force_mode_on():
    settings = SimpleNamespace(headless=HeadlessSettings())
    SessionConfig.from_cli_flags().apply_to_settings(settings)
    assert settings.headless.headless_mode is False


def test_apply_is_safe_when_headless_group_absent():
    """If the headless settings group is not initialized (default None), applying a
    headless SessionConfig must not raise — it degrades to a no-op."""
    settings = SimpleNamespace(headless=None)
    # Should not raise.
    SessionConfig.from_cli_flags(headless_mode=True).apply_to_settings(settings)

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

"""SessionConfig human-in-the-loop tool-approval wiring."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from victor.core.feature_flags import FeatureFlag, disable_feature, is_feature_enabled
from victor.framework.session_config import SessionConfig, ToolApprovalConfig


@pytest.fixture(autouse=True)
def _restore_policy_engine_flag():
    """Keep the process-global USE_POLICY_ENGINE flag from leaking across tests."""
    was_enabled = is_feature_enabled(FeatureFlag.USE_POLICY_ENGINE)
    yield
    if not was_enabled:
        disable_feature(FeatureFlag.USE_POLICY_ENGINE)


def _governance_settings(**kw):
    gov = SimpleNamespace(enabled=False, ask_on_tools=["preexisting"], ask_fallback="deny", **kw)
    return SimpleNamespace(governance=gov)


def test_default_session_config_leaves_governance_untouched():
    settings = _governance_settings()
    SessionConfig.from_cli_flags().apply_to_settings(settings)
    assert settings.governance.enabled is False
    assert settings.governance.ask_on_tools == ["preexisting"]


def test_tool_approval_enables_governance_and_unions_tools():
    settings = _governance_settings()
    config = SessionConfig.from_cli_flags(
        tool_approval_enabled=True,
        ask_on_tools=["bash", "preexisting", "git_push"],
        ask_fallback="allow",
    )
    config.apply_to_settings(settings)

    assert settings.governance.enabled is True
    assert settings.governance.ask_fallback == "allow"
    # Pre-existing tools are preserved; new ones appended without duplicates.
    assert settings.governance.ask_on_tools == ["preexisting", "bash", "git_push"]
    # Requesting approval implies the policy engine flag.
    assert is_feature_enabled(FeatureFlag.USE_POLICY_ENGINE) is True


def test_from_cli_flags_populates_tool_approval_config():
    config = SessionConfig.from_cli_flags(
        tool_approval_enabled=True, ask_on_tools=["bash"], ask_fallback="deny"
    )
    assert isinstance(config.tool_approval, ToolApprovalConfig)
    assert config.tool_approval.enabled is True
    assert config.tool_approval.ask_on_tools == ("bash",)
    assert config.tool_approval.ask_fallback == "deny"


def test_tool_approval_is_noop_when_settings_lack_governance():
    # Should not raise when the settings object has no governance group.
    settings = SimpleNamespace()
    SessionConfig.from_cli_flags(
        tool_approval_enabled=True, ask_on_tools=["bash"]
    ).apply_to_settings(settings)

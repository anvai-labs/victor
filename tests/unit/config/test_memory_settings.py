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

"""Memory settings group (FEP-0048): confirm-by-default posture."""

from victor.config.memory_settings import MemorySettings
from victor.config.settings import Settings


def test_auto_capture_default_off():
    """Wrong memories are worse than missing ones — capture is opt-in."""
    assert MemorySettings().memory_auto_capture is False


def test_settings_nested_group_syncs_flat_field():
    settings = Settings(**{"env_file": None})
    assert settings.memory_auto_capture is False
    assert settings.memory is not None
    assert settings.memory.memory_auto_capture is False


def test_auto_capture_optin_flows_to_nested_group():
    settings = Settings(**{"env_file": None, "memory_auto_capture": True})
    assert settings.memory_auto_capture is True
    assert settings.memory.memory_auto_capture is True


def test_env_var_optin(monkeypatch):
    monkeypatch.setenv("VICTOR_MEMORY_AUTO_CAPTURE", "true")
    settings = Settings(**{"env_file": None})
    assert settings.memory.memory_auto_capture is True


# =============================================================================
# Read-path injection settings (FOU-A2c): default-on with off-when-empty
# =============================================================================


def test_injection_defaults_on_with_off_when_empty_semantics():
    """Default ON per FEP-0048; an empty store still injects nothing (A2c)."""
    memory = MemorySettings()
    assert memory.memory_enabled is True
    assert memory.user_scope_enabled is True
    assert memory.project_scope_enabled is True
    assert memory.memory_injection_max_tokens == 500


def test_settings_flat_fields_sync_into_memory_group():
    settings = Settings(**{"env_file": None})
    assert settings.memory_enabled is True
    assert settings.memory is not None
    assert settings.memory.memory_enabled is True
    assert settings.memory.user_scope_enabled is True
    assert settings.memory.project_scope_enabled is True
    assert settings.memory.memory_injection_max_tokens == 500


def test_env_opt_out_flows_to_nested_group(monkeypatch):
    monkeypatch.setenv("VICTOR_MEMORY_ENABLED", "false")
    settings = Settings(**{"env_file": None})
    assert settings.memory_enabled is False
    assert settings.memory.memory_enabled is False


def test_per_scope_env_opt_outs(monkeypatch):
    monkeypatch.setenv("VICTOR_USER_SCOPE_ENABLED", "false")
    settings = Settings(**{"env_file": None})
    assert settings.memory.user_scope_enabled is False
    assert settings.memory.project_scope_enabled is True


def test_injection_cap_env_override(monkeypatch):
    monkeypatch.setenv("VICTOR_MEMORY_INJECTION_MAX_TOKENS", "200")
    settings = Settings(**{"env_file": None})
    assert settings.memory.memory_injection_max_tokens == 200

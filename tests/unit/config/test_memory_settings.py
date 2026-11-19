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

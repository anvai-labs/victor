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

"""Regression tests for victor.config package exports."""

from __future__ import annotations

import importlib
import sys
from unittest.mock import patch


def test_config_package_lazily_resolves_api_keys_after_package_reimport(
    monkeypatch,
) -> None:
    """Reloading victor.config should not orphan already-imported submodules."""

    api_keys_module = importlib.import_module("victor.config.api_keys")

    monkeypatch.delitem(sys.modules, "victor.config", raising=False)
    config_pkg = importlib.import_module("victor.config")

    assert config_pkg.api_keys is api_keys_module


def test_patch_target_resolution_survives_config_package_reimport(monkeypatch) -> None:
    """Fixtures patching victor.config.api_keys should still work after reloads."""

    importlib.import_module("victor.config.api_keys")
    monkeypatch.delitem(sys.modules, "victor.config", raising=False)
    importlib.import_module("victor.config")

    with patch("victor.config.api_keys.get_api_key", return_value=None) as mock_get_api_key:
        from victor.config.api_keys import get_api_key

        assert get_api_key("openai") is None
        mock_get_api_key.assert_called_once_with("openai")

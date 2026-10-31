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

"""Unit tests for optional, name-based provider creation through the registry."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from victor.providers import registry as registry_module
from victor.providers.registry import create_provider_or_none


def test_creates_the_registered_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = MagicMock()
    monkeypatch.setattr(registry_module, "get_provider_registry", lambda: fake)
    assert create_provider_or_none("zai", purpose="test") is fake.create.return_value
    fake.create.assert_called_once_with("zai")


@pytest.mark.parametrize("name", [None, ""])
def test_falsy_name_disables_creation(monkeypatch: pytest.MonkeyPatch, name: object) -> None:
    fake = MagicMock()
    monkeypatch.setattr(registry_module, "get_provider_registry", lambda: fake)
    assert create_provider_or_none(name, purpose="test") is None  # type: ignore[arg-type]
    fake.create.assert_not_called()


def test_creation_failure_returns_none_and_logs_purpose(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    fake = MagicMock()
    fake.create.side_effect = KeyError("no such provider")
    monkeypatch.setattr(registry_module, "get_provider_registry", lambda: fake)
    with caplog.at_level(logging.DEBUG, logger="victor.providers.registry"):
        assert create_provider_or_none("nope", purpose="LLM-as-Critic") is None
    assert "Failed to create nope provider for LLM-as-Critic" in caplog.text


def test_exported() -> None:
    assert "create_provider_or_none" in registry_module.__all__

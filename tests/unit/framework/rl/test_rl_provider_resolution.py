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

"""RL components build their LLM provider through the provider registry.

GEPAStrategy and the LLM-as-Critic credit assigner derived the provider class
from ``f"{name.title()}Provider"``, which only exists for ollama and anthropic:
openai (OpenAIProvider), zai (ZAIProvider), deepseek and lmstudio silently
produced no provider.
"""

from unittest.mock import MagicMock

import pytest

from victor.framework.rl.credit_assignment import LLMCriticCreditAssigner
from victor.framework.rl.learners.gepa_strategy import GEPAStrategy


@pytest.fixture(params=["gepa", "critic"])
def component(request):
    cls = GEPAStrategy if request.param == "gepa" else LLMCriticCreditAssigner
    obj = cls.__new__(cls)
    obj._provider = None
    return obj


@pytest.mark.parametrize("name", ["openai", "zai", "deepseek", "lmstudio", "ollama"])
def test_provider_comes_from_the_registry(component, monkeypatch, name):
    registry = MagicMock()
    monkeypatch.setattr("victor.providers.registry.get_provider_registry", lambda: registry)
    component._provider_name = name
    assert component._get_provider() is registry.create.return_value
    registry.create.assert_called_once_with(name)
    assert component._get_provider() is registry.create.return_value  # cached
    registry.create.assert_called_once()


def test_real_registry_resolves_names_the_title_convention_missed():
    """openai/zai/deepseek/lmstudio have registered classes (no instantiation here)."""
    from victor.providers.registry import get_provider_registry

    registry = get_provider_registry()
    for name in ("openai", "zai", "deepseek", "lmstudio"):
        assert registry.get_or_raise(name) is not None, name


def test_disabled_or_unknown_provider_yields_none(component, monkeypatch):
    component._provider_name = None
    assert component._get_provider() is None
    registry = MagicMock()
    registry.create.side_effect = KeyError("no such provider")
    monkeypatch.setattr("victor.providers.registry.get_provider_registry", lambda: registry)
    component._provider_name = "nope"
    assert component._get_provider() is None

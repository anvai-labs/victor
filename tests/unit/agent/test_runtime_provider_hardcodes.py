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

"""Pins against hardcoded providers in runtime paths.

The chat default is configurable (InferFlux as of the defaults flip);
runtime subsystems must derive from configuration, not restate Ollama.
"""

import asyncio
import os
from typing import Any, Optional

import pytest

from victor.agent.edge_model import EdgeModelConfig


def test_edge_model_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VICTOR_EDGE_MODEL_PROVIDER", "llamacpp")
    monkeypatch.setenv("VICTOR_EDGE_MODEL", "tiny-instruct")
    monkeypatch.setenv("VICTOR_EDGE_MODEL_BASE_URL", "http://localhost:8081/v1")
    config = EdgeModelConfig()
    assert (config.provider, config.model, config.base_url) == (
        "llamacpp",
        "tiny-instruct",
        "http://localhost:8081/v1",
    )


def test_edge_model_defaults_stay_local_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    """Edge decisions want a tiny local model: defaults unchanged by the
    InferFlux flip (a 30B coder over a tunnel is the wrong edge shape)."""
    for var in ("VICTOR_EDGE_MODEL_PROVIDER", "VICTOR_EDGE_MODEL", "VICTOR_EDGE_MODEL_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    config = EdgeModelConfig()
    assert (config.provider, config.model) == ("ollama", "qwen3.5:2b")


class _StubProviderSettings:
    default_provider = "inferflux"
    providers = {}


class _StubSettings:
    provider = _StubProviderSettings()


def test_completion_provider_follows_configured_default(monkeypatch: pytest.MonkeyPatch):
    """The completion subsystem resolves the configured default provider via
    the real registry API (the old code called nonexistent get_provider)."""
    from victor.processing.completion.providers import ai as ai_mod

    created = {}

    class _Provider:
        def __init__(self, **kwargs: Any) -> None:
            created.update(kwargs)

    captured = {}

    def _fake_create(name: str, **kwargs: Any):
        captured["name"] = name
        captured["kwargs"] = kwargs
        return _Provider()

    monkeypatch.setattr("victor.config.settings.load_settings", lambda: _StubSettings())
    monkeypatch.setattr(
        "victor.providers.registry.ProviderRegistry.create", staticmethod(_fake_create)
    )
    provider = ai_mod.AICompletionProvider()._get_provider()
    assert isinstance(provider, _Provider)
    assert captured["name"] == "inferflux"


def test_completion_provider_returns_none_when_unresolvable(
    monkeypatch: pytest.MonkeyPatch,
):
    from victor.processing.completion.providers import ai as ai_mod

    def _raise(name: str, **kwargs: Any):
        raise RuntimeError("no provider")

    monkeypatch.setattr("victor.config.settings.load_settings", lambda: _StubSettings())
    monkeypatch.setattr("victor.providers.registry.ProviderRegistry.create", staticmethod(_raise))
    provider = ai_mod.AICompletionProvider()._get_provider()
    assert provider is None


def test_skill_matcher_attaches_before_initialize_completes():
    """Matcher readiness is asynchronous by design: bootstrap no longer pays
    the ~6s embedding-model load; consumers' readiness guards degrade."""
    from unittest.mock import MagicMock

    from victor.framework.agent_factory import AgentFactory

    factory = AgentFactory.__new__(AgentFactory)
    orchestrator = MagicMock()
    factory._orchestrator = orchestrator

    class _SlowMatcher:
        def __init__(self) -> None:
            self._initialized = False

        @property
        def initialized(self) -> bool:
            return self._initialized

        async def initialize(self, registry: Any) -> None:
            await asyncio.sleep(0.05)
            self._initialized = True

    slow = _SlowMatcher()

    async def run() -> None:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("victor.framework.skill_matcher.SkillMatcher", lambda: slow)
            import victor.framework.skills as skills_mod

            class _Registry:
                _external_loaded = False

                def from_entry_points(self) -> None: ...

                def from_user_skills(self) -> None: ...

            mp.setattr(skills_mod, "get_skill_registry", lambda: _Registry())
            await factory._initialize_skill_matcher()
            # attached synchronously, BEFORE initialize finished
            assert orchestrator._skill_matcher is slow
            assert slow.initialized is False  # still initializing
            await asyncio.sleep(0.1)

    asyncio.run(run())
    assert slow.initialized is True

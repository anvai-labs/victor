# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Pinned prompt-strategy classification across the provider matrix.

Prompt strategy decides how much scaffolding a model gets, and getting it wrong
is not cosmetic: in session ``sandhi-cdfbc589`` a frontier model was classified as
a weak local one and handed a read-only "code analyst" prompt, and it stopped
working. #700 replaced the hand-maintained provider-name lists that caused it.

But #700 then shipped its own version of the same mistake: `_is_remotely_hosted`
keyed on the OpenAI-compat spec's ``prompt_caching`` flag, which is a *pricing*
fact rather than a hosting one, and silently demoted `moonshot/kimi-k3` from
MINIMAL to STRUCTURED (#706). That went unnoticed because the tests checked one
provider at a time.

**This table is the deliverable.** Changing how strategy is derived should force
this file to change, showing the full blast radius in the diff rather than one
row at a time. When a row moves, decide whether the move is intended — do not
reflexively re-pin it.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from victor.agent.intelligent_prompt_builder import (
    IntelligentPromptBuilder,
    ProfileLearningStore,
    PromptContext,
    PromptStrategy,
)

# (provider, model, expected strategy) with the catalog facts that produce it.
# Regenerate deliberately, never blindly.
STRATEGY_MATRIX = [
    # Hosted + native + not strict -> MINIMAL
    ("anthropic", "claude-opus-4", PromptStrategy.MINIMAL),  # native=T strict=F
    ("openai", "gpt-5", PromptStrategy.MINIMAL),  # native=T strict=F
    ("google", "gemini-3", PromptStrategy.MINIMAL),  # native=T strict=F
    ("xai", "grok-4", PromptStrategy.MINIMAL),  # native=T strict=F
    ("deepseek", "deepseek-v4pro", PromptStrategy.MINIMAL),  # native=T strict=F
    ("openrouter", "claude-3", PromptStrategy.MINIMAL),  # native=T strict=F
    ("together", "mixtral-8x7b", PromptStrategy.MINIMAL),  # native=T strict=F
    ("fireworks", "firefunction-v2", PromptStrategy.MINIMAL),  # native=T strict=F
    ("cerebras", "llama3.1-8b", PromptStrategy.MINIMAL),  # native=T strict=F
    # The two that motivated this file:
    ("zai", "glm-5.2", PromptStrategy.MINIMAL),  # was STRICT before #700
    ("moonshot", "kimi-k3", PromptStrategy.MINIMAL),  # was STRUCTURED after #700
    # Local + native + not strict -> STRUCTURED
    ("ollama", "qwen2.5-coder:7b", PromptStrategy.STRUCTURED),  # native=T strict=F
    ("lmstudio", "mistral-7b", PromptStrategy.STRUCTURED),  # native=T strict=F
    ("vllm", "qwen3-32b", PromptStrategy.STRUCTURED),  # native=T strict=F
    # Catalog says the model needs strict prompting -> STRICT, wherever it runs
    ("ollama", "codellama:7b", PromptStrategy.STRICT),  # native=F strict=T
    ("ollama", "llama-3.3-70b", PromptStrategy.STRICT),  # native=F strict=T
]


@pytest.fixture
def learning_store():
    """Cold profile store — pins the *derived* strategy, not a learned one."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield ProfileLearningStore(db_path=Path(tmpdir) / "parity.db")


def _strategy(store, provider: str, model: str) -> PromptStrategy:
    builder = IntelligentPromptBuilder(
        provider_name=provider,
        model=model,
        profile_name=f"{provider}:{model}",
        learning_store=store,
    )
    return builder._determine_strategy(
        PromptContext(
            task="t",
            task_type="analysis",
            profile_name=f"{provider}:{model}",
            provider=provider,
            model=model,
        )
    )


class TestStrategyMatrix:
    """The pinned table. A diff here is a behaviour change for real sessions."""

    @pytest.mark.parametrize("provider,model,expected", STRATEGY_MATRIX)
    def test_strategy_is_pinned(self, learning_store, provider, model, expected):
        assert _strategy(learning_store, provider, model) is expected

    def test_no_hosted_frontier_model_is_demoted_to_strict(self, learning_store):
        """STRICT on a capable hosted model is the sandhi-cdfbc589 failure.

        It produces the read-only "code analyst / plain English only" identity,
        which contradicts any write-capable mode.
        """
        demoted = [
            f"{p}/{m}"
            for p, m, expected in STRATEGY_MATRIX
            if p not in {"ollama", "lmstudio", "vllm"} and expected is PromptStrategy.STRICT
        ]

        assert not demoted, f"hosted models classified STRICT: {demoted}"


class TestClassificationSources:
    """Classification must derive from hosting + declared capability. Nothing else."""

    def test_pricing_does_not_influence_classification(self, learning_store):
        """#706: `prompt_caching` is a pricing fact, not a hosting one.

        Moonshot is a hosted provider that declares ``prompt_caching: False``.
        Keying hosting on that flag demoted kimi-k3 to STRUCTURED. This asserts
        the flag is genuinely False *and* that classification ignores it — so the
        test still fails if someone reintroduces the proxy.
        """
        from victor.providers.openai_compat_model_policy import (
            get_openai_compat_provider_spec,
        )

        spec = get_openai_compat_provider_spec("moonshot")
        assert spec.capabilities.prompt_caching is False, (
            "premise changed: moonshot now declares prompt caching, so this test no "
            "longer exercises the pricing-vs-hosting distinction — pick another provider"
        )

        assert _strategy(learning_store, "moonshot", "kimi-k3") is PromptStrategy.MINIMAL

    def test_hosting_comes_from_the_canonical_local_provider_set(self, learning_store):
        """Every declared local provider is treated as local, and only those."""
        from victor.config.api_keys import LOCAL_PROVIDERS

        builder = IntelligentPromptBuilder(
            provider_name="ollama",
            model="qwen2.5-coder:7b",
            profile_name="p",
            learning_store=learning_store,
        )
        for provider in LOCAL_PROVIDERS:
            builder.provider_name = provider
            assert builder._is_remotely_hosted() is False, provider

        for provider in ("anthropic", "openai", "zai", "moonshot", "together"):
            builder.provider_name = provider
            assert builder._is_remotely_hosted() is True, provider

    def test_capability_comes_from_the_shared_catalog(self, learning_store):
        """Not from a private list — the adapters read the same source."""
        from victor.agent.tool_calling.capabilities import get_model_capabilities

        builder = IntelligentPromptBuilder(
            provider_name="zai",
            model="glm-5.2",
            profile_name="p",
            learning_store=learning_store,
        )

        assert builder._tool_calling_capabilities() == get_model_capabilities("zai", "glm-5.2")

    def test_no_provider_name_sets_reintroduced(self):
        """The lists #700 removed must not come back in any form."""
        assert not hasattr(IntelligentPromptBuilder, "CLOUD_PROVIDERS")
        assert not hasattr(IntelligentPromptBuilder, "LOCAL_PROVIDERS")
        assert not hasattr(IntelligentPromptBuilder, "_has_native_tool_support")

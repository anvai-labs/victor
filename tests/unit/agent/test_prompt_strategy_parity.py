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

import pathlib
import tempfile

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
    ("anthropic", "claude-opus-4", PromptStrategy.MINIMAL),  # claude-*
    ("openai", "gpt-5.4", PromptStrategy.MINIMAL),  # gpt-5.4*
    ("google", "gemini-3", PromptStrategy.MINIMAL),  # gemini-3*
    ("xai", "grok-4", PromptStrategy.MINIMAL),  # grok-4*
    ("openrouter", "claude-3", PromptStrategy.MINIMAL),  # claude-*
    ("together", "mixtral-8x7b", PromptStrategy.MINIMAL),  # mixtral*
    ("fireworks", "firefunction-v2", PromptStrategy.MINIMAL),  # firefunction*
    ("cerebras", "llama3.1-8b", PromptStrategy.MINIMAL),  # llama3.1*
    # The two that motivated this file:
    ("zai", "glm-5.2", PromptStrategy.MINIMAL),  # glm-5*   — was STRICT before #700
    ("moonshot", "kimi-k3", PromptStrategy.MINIMAL),  # kimi-k3* — was STRUCTURED after #700
    # Local + native + not strict -> STRUCTURED
    ("ollama", "qwen2.5-coder:7b", PromptStrategy.STRUCTURED),  # qwen2.5*
    ("lmstudio", "mistral-7b", PromptStrategy.STRUCTURED),  # mistral*
    ("vllm", "qwen3:32b", PromptStrategy.STRUCTURED),  # qwen3:*
    # requires_strict_prompting -> STRICT regardless of native tool support.
    # llama3.3 DOES support native tool calling (native=True); it is pinned STRICT
    # because the catalog marks it as needing strict prompting on a local runtime.
    ("ollama", "llama3.3:70b", PromptStrategy.STRICT),  # llama3.3*  native=T strict=T
    ("ollama", "codellama:7b", PromptStrategy.STRICT),  # codellama* native=F strict=T
]


@pytest.fixture
def learning_store():
    """Cold profile store — pins the *derived* strategy, not a learned one."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield ProfileLearningStore(db_path=pathlib.Path(tmpdir) / "parity.db")


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


class TestMatrixUsesRealModelIdentifiers:
    """Every pinned row must name a model the capability catalog recognises.

    The first version of this file pinned ``llama-3.3-70b`` and ``qwen3-32b``.
    Neither matches any catalog pattern (the real identifiers are ``llama3.3:70b``
    and ``qwen3:32b``), so those rows silently exercised the unknown-model
    fallback while *reading* as statements about Llama 3.3 and Qwen3 — and led to
    Llama 3.3 being described as lacking native tool calling, which is false: it
    supports it, and is STRICT here only because the catalog marks it as needing
    strict prompting on a local runtime.

    A parity table whose rows name models that do not exist pins nothing.
    """

    @staticmethod
    def _catalog_patterns() -> list[str]:
        import yaml

        config = pathlib.Path(__file__).resolve().parents[3] / (
            "victor/config/model_capabilities.yaml"
        )
        return sorted(yaml.safe_load(config.read_text()).get("models", {}))

    @staticmethod
    def _matches(model: str, patterns: list[str]) -> str | None:
        import fnmatch

        hits = [p for p in patterns if fnmatch.fnmatch(model.lower(), p.lower())]
        return max(hits, key=len) if hits else None

    def test_every_row_matches_a_catalog_pattern(self):
        patterns = self._catalog_patterns()

        unmatched = [
            f"{provider}/{model}"
            for provider, model, _ in STRATEGY_MATRIX
            if self._matches(model, patterns) is None
        ]

        assert not unmatched, (
            "these rows name models the catalog does not recognise, so they pin the "
            f"unknown-model fallback rather than the model: {unmatched}. Use the real "
            "identifier (e.g. 'llama3.3:70b', not 'llama-3.3-70b')."
        )

    def test_strict_rows_are_strict_for_a_stated_catalog_reason(self):
        """A STRICT pin must come from declared capability, not a lookup miss."""
        from victor.agent.tool_calling.capabilities import get_model_capabilities

        for provider, model, expected in STRATEGY_MATRIX:
            if expected is not PromptStrategy.STRICT:
                continue
            caps = get_model_capabilities(provider, model)
            assert caps.requires_strict_prompting or not caps.native_tool_calls, (
                f"{provider}/{model} is pinned STRICT but the catalog declares "
                f"native={caps.native_tool_calls} strict={caps.requires_strict_prompting} "
                "— that is a fallback, not a decision"
            )

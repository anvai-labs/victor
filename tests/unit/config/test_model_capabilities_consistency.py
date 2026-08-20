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

"""Internal consistency of the model capability catalog.

``model_capabilities.yaml`` is the single source of truth for prompt strategy and
tool-calling adaptation. It carries two related facts per model:

* ``training.tool_calling`` — whether the *weights* were trained for tool calling
* ``providers.<name>.native_tool_calls`` — whether that runtime exposes them

The second cannot exceed the first. A runtime can supply a chat template that
*formats* tools, but it cannot give a model a capability it was never trained
for; the model will emit prose or malformed JSON regardless of who serves it.

An entry that claims otherwise produces exactly the defect this catalog exists to
prevent — the wrong prompt strategy for the model's real ability, which is what
made a frontier model get a read-only prompt in session sandhi-cdfbc589.
"""

from __future__ import annotations

import pathlib

import pytest
import yaml

CAPABILITIES_PATH = (
    pathlib.Path(__file__).resolve().parents[3] / "victor/config/model_capabilities.yaml"
)


@pytest.fixture(scope="module")
def catalog() -> dict:
    return yaml.safe_load(CAPABILITIES_PATH.read_text())


def _provider_overrides(entry: dict) -> dict:
    providers = entry.get("providers")
    return providers if isinstance(providers, dict) else {}


class TestTrainingBoundsProviderClaims:
    """No runtime may claim a capability the weights do not have."""

    def test_untrained_models_claim_no_native_tool_calls(self, catalog: dict) -> None:
        violations = []
        for pattern, entry in (catalog.get("models") or {}).items():
            if not isinstance(entry, dict):
                continue
            if (entry.get("training") or {}).get("tool_calling") is not False:
                continue
            for provider, override in _provider_overrides(entry).items():
                if isinstance(override, dict) and override.get("native_tool_calls") is True:
                    violations.append(f"{pattern} -> providers.{provider}")

        assert not violations, (
            "these entries declare training.tool_calling=false yet claim "
            "native_tool_calls=true for a runtime:\n  "
            + "\n  ".join(violations)
            + "\n\nA runtime can format tools; it cannot train the model to emit them. "
            "The effect is a weaker prompt strategy for a model that cannot call tools."
        )

    def test_a_model_is_not_native_on_one_runtime_and_not_another_without_reason(
        self, catalog: dict
    ) -> None:
        """Split verdicts are allowed, but only when training says tools work.

        If the weights support tools, runtimes legitimately differ (one wires up
        the template, another does not). If the weights do not, every runtime
        must agree — there is nothing to differ about.
        """
        split = []
        for pattern, entry in (catalog.get("models") or {}).items():
            if not isinstance(entry, dict):
                continue
            trained = (entry.get("training") or {}).get("tool_calling")
            verdicts = {
                provider: override.get("native_tool_calls")
                for provider, override in _provider_overrides(entry).items()
                if isinstance(override, dict) and "native_tool_calls" in override
            }
            if trained is False and len(set(verdicts.values())) > 1:
                split.append(f"{pattern}: {verdicts}")

        assert not split, (
            "untrained models must be non-native on every runtime, but these "
            "disagree across providers:\n  " + "\n  ".join(split)
        )


class TestStrategyFollowsTraining:
    """The end-to-end consequence: an untrained model always gets STRICT."""

    @pytest.mark.parametrize(
        "provider,model",
        [
            ("ollama", "codellama:7b"),
            ("vllm", "codellama:7b"),
            ("lmstudio", "codellama:7b"),
            ("ollama", "gemma:7b"),
            ("vllm", "gemma:7b"),
            ("lmstudio", "gemma:7b"),
        ],
    )
    def test_models_without_tool_training_get_strict_everywhere(
        self, provider: str, model: str
    ) -> None:
        """Where it runs must not change whether it can call tools.

        Before this guard, codellama and gemma resolved STRICT on ollama but
        STRUCTURED on vllm/lmstudio — the same weights getting *less* scaffolding
        purely because of a runtime override the training data contradicts.
        """
        import tempfile

        from victor.agent.intelligent_prompt_builder import (
            IntelligentPromptBuilder,
            ProfileLearningStore,
            PromptContext,
            PromptStrategy,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            store = ProfileLearningStore(db_path=pathlib.Path(tmpdir) / "c.db")
            builder = IntelligentPromptBuilder(
                provider_name=provider,
                model=model,
                profile_name=f"{provider}:{model}",
                learning_store=store,
            )
            strategy = builder._determine_strategy(
                PromptContext(
                    task="t",
                    task_type="analysis",
                    profile_name=f"{provider}:{model}",
                    provider=provider,
                    model=model,
                )
            )

        assert strategy is PromptStrategy.STRICT, (
            f"{provider}/{model} resolved {strategy.name}; a model whose weights "
            "were not trained for tool calling must get STRICT on every runtime"
        )

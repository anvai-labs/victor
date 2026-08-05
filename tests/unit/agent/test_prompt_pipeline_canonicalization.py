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

"""Regression coverage for canonical model-facing tool names in prompt prefixes."""

from unittest.mock import MagicMock, patch

from victor.agent.services.runtime_intelligence import PromptOptimizationBundle


def _make_provider(api_cache: bool = False, kv_cache: bool = False) -> MagicMock:
    provider = MagicMock()
    provider.supports_prompt_caching.return_value = api_cache
    provider.supports_kv_prefix_caching.return_value = kv_cache
    return provider


def _make_builder(base_prompt: str = "You are a helpful assistant.") -> MagicMock:
    builder = MagicMock()
    builder.build.return_value = base_prompt
    builder.provider_name = "test"
    builder.model = "test-model"
    return builder


def _make_registry() -> MagicMock:
    registry = MagicMock()
    registry.get_all.return_value = []
    registry.get_by_category.return_value = []
    return registry


def _make_optimizer(evolved_sections=None, few_shots=None, failure_hint=None) -> MagicMock:
    optimizer = MagicMock()
    optimizer.get_evolved_sections.return_value = evolved_sections or []
    optimizer.get_few_shots.return_value = few_shots
    optimizer.get_failure_hint.return_value = failure_hint
    optimizer.clear_session_cache.return_value = None
    return optimizer


def _make_pipeline(**kwargs):
    from victor.agent.prompt_pipeline import UnifiedPromptPipeline

    defaults = {
        "provider": _make_provider(api_cache=True, kv_cache=True),
        "builder": _make_builder(),
        "registry": _make_registry(),
        "optimizer": None,
        "task_analyzer": None,
        "get_context_window": lambda: 128000,
        "session_id": "test-session",
    }
    defaults.update(kwargs)
    return UnifiedPromptPipeline(**defaults)


def _make_turn_context(**overrides):
    from victor.agent.prompt_pipeline import TurnContext

    defaults = {
        "provider_name": "test",
        "model": "test-model",
        "task_type": "default",
    }
    defaults.update(overrides)
    return TurnContext(**defaults)


class TestPromptPipelineCanonicalization:
    """Dynamic prefix content should use canonical core-tool names."""

    def test_optimizer_content_is_canonicalized_before_injection(self):
        optimizer = _make_optimizer(
            evolved_sections=[
                "Prefer read_file over cat.",
                "Use execute_bash for git status after list_directory.",
            ],
            few_shots="Example: read_file('app.py') then edit_file(...)",
            failure_hint="Retry write_file after verifying list_directory output.",
        )
        pipeline = _make_pipeline(optimizer=optimizer)
        ctx = _make_turn_context(last_turn_failed=True, last_failure_category="file_not_found")

        prefix = pipeline.compose_turn_prefix("Fix the bug", ctx)

        assert "read_file" not in prefix
        assert "execute_bash" not in prefix
        assert "list_directory" not in prefix
        assert "edit_file" not in prefix
        assert "write_file" not in prefix
        assert "Prefer read over cat." in prefix
        assert "Use shell for git status after ls." in prefix
        assert "Example: read('app.py') then edit(...)" in prefix
        assert "Retry write after verifying ls output." in prefix

    def test_credit_and_reputation_guidance_are_canonicalized(self):
        pipeline = _make_pipeline()
        ctx = _make_turn_context()

        with (
            patch.object(
                type(pipeline),
                "_get_credit_guidance",
                return_value="Tool effectiveness:\n- read_file: high effectiveness\n- bash: low effectiveness",
            ),
            patch.object(
                type(pipeline),
                "_get_tool_reputation_guidance",
                return_value="Mid-turn tool reputation:\n- list_directory: reliable\n- execute_bash: unreliable",
            ),
        ):
            prefix = pipeline.compose_turn_prefix("Inspect the repo", ctx)

        assert "read_file" not in prefix
        assert "list_directory" not in prefix
        assert "execute_bash" not in prefix
        assert "\n- read: high effectiveness" in prefix
        assert "\n- shell: low effectiveness" in prefix
        assert "\n- ls: reliable" in prefix
        assert "\n- shell: unreliable" in prefix

    def test_experiment_memory_guidance_is_canonicalized_before_injection(self):
        runtime_intelligence = MagicMock()
        runtime_intelligence.get_prompt_optimization_bundle.return_value = PromptOptimizationBundle(
            experiment_guidance=[
                "Experiment-guided next candidate: Use read_file after list_directory, then execute_bash for pytest.",
            ],
            experiment_memory_hints={"experiment_memory_match_count": 1},
        )
        pipeline = _make_pipeline(runtime_intelligence=runtime_intelligence)
        ctx = _make_turn_context()

        prefix = pipeline.compose_turn_prefix("Fix the bug", ctx)

        assert "read_file" not in prefix
        assert "list_directory" not in prefix
        assert "execute_bash" not in prefix
        assert "Use read after ls, then shell for pytest." in prefix

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

"""A/B hygiene + Graph-RAG setup wiring (victor#907 slice 2).

The first backend A/B was confounded by the harness auto-evolving its
prompt between arms (gen-3 -> gen-4); and benchmark setup had no way to
build the Graph-RAG codegraph the marker-aware graph tooling reads.
"""

import inspect
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def test_frozen_prompts_serves_static_fallback(monkeypatch):
    from victor.agent.evolved_content_resolver import EvolvedContentResolver

    class ExplodingInjector:
        def __getattr__(self, name):  # pragma: no cover - must never be reached
            raise AssertionError("injector must not be consulted when frozen")

    monkeypatch.setenv("VICTOR_FROZEN_PROMPTS", "1")
    resolver = EvolvedContentResolver(optimization_injector=ExplodingInjector())
    resolved = resolver.resolve_section(
        "ASI_TOOL_EFFECTIVENESS_GUIDANCE", fallback_text="baseline text"
    )
    assert resolved.text == "baseline text"
    assert resolved.source == "static"


def test_unfrozen_resolver_still_consults_injector(monkeypatch):
    from victor.agent.evolved_content_resolver import EvolvedContentResolver

    monkeypatch.delenv("VICTOR_FROZEN_PROMPTS", raising=False)
    injector = Mock()
    injector.get_section_payload.return_value = {"text": "selected candidate"}
    resolver = EvolvedContentResolver(optimization_injector=injector)
    resolved = resolver.resolve_section("X", fallback_text="fb")
    assert resolved.text == "selected candidate"
    assert resolved.source == "evolved"
    injector.get_section_payload.assert_called_once()


@pytest.mark.parametrize("warm_cache", [False, True])
def test_frozen_turn_guidance_excludes_learned_content_and_attribution(monkeypatch, warm_cache):
    from victor.agent.optimization_injector import OptimizationInjector
    from victor.agent.services.runtime_intelligence import (
        PromptOptimizationIdentity,
        RuntimeIntelligenceService,
    )

    monkeypatch.delenv("VICTOR_FROZEN_PROMPTS", raising=False)
    injector = OptimizationInjector()
    section = "ASI_TOOL_EFFECTIVENESS_GUIDANCE"
    sample = Mock(
        return_value={
            "text": "changing candidate",
            "section_name": section,
            "prompt_candidate_hash": "candidate-1",
            "source": "candidate",
        }
    )
    monkeypatch.setattr(injector, "_sample_evolved_payload", sample)
    if warm_cache:
        injector.get_section_payload(section, "test", "test", "default")
        injector._few_shot_payload_cache[("query", "test", "test", "default")] = {
            "text": "changing example",
            "prompt_candidate_hash": "example-1",
        }
    service = RuntimeIntelligenceService(
        task_analyzer=None,
        perception_integration=None,
        optimization_injector=injector,
        decision_service=None,
    )
    memory = Mock(return_value={"experiment_memory_next_candidate_hints": ["changing hint"]})
    monkeypatch.setattr(service, "get_experiment_routing_context", memory)
    context = SimpleNamespace(
        provider_name="test",
        model="test",
        task_type="default",
        last_turn_failed=True,
        last_failure_category="file_not_found",
        last_failure_error="missing file",
    )
    # Simulate an earlier unfrozen turn so stale reward identities must be replaced.
    service._last_served_prompt_identities = [
        PromptOptimizationIdentity(prompt_candidate_hash="previous-candidate")
    ]
    sample.reset_mock()
    memory.reset_mock()
    monkeypatch.setenv("VICTOR_FROZEN_PROMPTS", "1")

    bundle = service.get_prompt_optimization_bundle("query", context)

    static = injector._build_static_fallback_payload(section, "test")
    assert static is not None
    assert bundle.evolved_sections == [static["text"]]
    assert bundle.few_shots is None
    assert bundle.experiment_guidance == []
    assert bundle.experiment_memory_hints == {}
    assert bundle.failure_hint == injector.get_failure_hint("file_not_found", "missing file")
    assert bundle.failure_hint
    assert all(identity.prompt_candidate_hash is None for identity in bundle.identities)
    assert service._last_served_prompt_identities == []
    sample.assert_not_called()
    memory.assert_not_called()


def test_frozen_baseline_rejects_explicit_candidate_binding(monkeypatch):
    from victor.agent.optimization_injector import OptimizationInjector

    monkeypatch.setenv("VICTOR_FROZEN_PROMPTS", "1")
    with pytest.raises(ValueError, match="frozen"):
        OptimizationInjector().bind_prompt_candidate(
            section_name="COMPLETION_GUIDANCE",
            prompt_candidate_hash="candidate-1",
        )


def test_setup_threads_build_graph_rag():
    from victor.evaluation.swe_bench_loader import SWEBenchWorkspaceManager

    sig = inspect.signature(SWEBenchWorkspaceManager.setup_repo_with_indexes)
    assert "build_graph_rag" in sig.parameters
    assert hasattr(SWEBenchWorkspaceManager, "_build_graph_rag")


def test_cli_flags_exist():
    from victor.ui.commands import benchmark as bench_cmd

    setup_sig = inspect.signature(bench_cmd.setup_benchmark)
    assert "graph_rag" in setup_sig.parameters
    run_sig = inspect.signature(bench_cmd.run_benchmark)
    assert "frozen_prompts" in run_sig.parameters

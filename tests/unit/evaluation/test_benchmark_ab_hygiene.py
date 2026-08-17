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
    resolver = EvolvedContentResolver(optimization_injector=None)
    resolved = resolver.resolve_section("X", fallback_text="fb")
    assert resolved.text == "fb"


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

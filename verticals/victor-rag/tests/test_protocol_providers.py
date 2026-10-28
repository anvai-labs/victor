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

"""Pre-move characterization for victor-rag protocol providers (dedup F3)."""

from victor_rag.protocols import (
    RAGPromptProvider,
    RAGSafetyProvider,
    RAGToolProvider,
    RAGToolSelectionStrategy,
    RAGWorkflowProvider,
)


class TestRAGToolProvider:
    def test_tool_list_contract(self):
        tools = RAGToolProvider().get_tools()
        assert len(tools) == len(set(tools)), "duplicate tools"
        for expected in ("read", "vector_search", "vector_add"):
            assert expected in tools

    def test_selection_strategy_stage_and_fallback(self):
        strategy = RAGToolSelectionStrategy()
        stages = strategy.get_tools_for_stage("index", "rag")
        assert isinstance(stages, list)
        # rag's fallback is its search pair — per-vertical fallback is data.
        assert strategy.get_tools_for_stage("unknown_stage", "any") == [
            "semantic_search", "knowledge_query",
        ]


class TestRAGSafetyProvider:
    def test_pattern_backed_surface(self):
        provider = RAGSafetyProvider()
        assert provider.get_bash_patterns(), "no dangerous patterns"
        assert provider.get_file_patterns() == []
        assert provider.get_extensions() == []
        assert isinstance(provider.get_tool_restrictions(), dict)


class TestRAGPromptAndWorkflowProviders:
    def test_prompt_sections(self):
        sections = RAGPromptProvider().get_system_prompt_sections()
        assert isinstance(sections, dict) and sections
        assert isinstance(RAGPromptProvider().get_prompt_contributors(), list)

    def test_workflow_registry(self):
        provider = RAGWorkflowProvider()
        names = provider.list_workflows()
        assert names == list(provider.get_workflows().keys())
        for name in names:
            assert provider.get_workflow(name) is not None
        assert provider.get_workflow("no_such_workflow") is None

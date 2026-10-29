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

"""Pre-move characterization for victor-coding protocol providers (dedup F3)."""

from victor_coding.protocols import (
    CodingPromptProvider,
    CodingSafetyProvider,
    CodingToolProvider,
    CodingToolSelectionStrategy,
    CodingWorkflowProvider,
)


class TestCodingToolProvider:
    def test_tool_list_contract(self):
        tools = CodingToolProvider().get_tools()
        assert len(tools) == len(set(tools)), "duplicate tools"
        for expected in ("read", "write", "shell", "grep", "code_search"):
            assert expected in tools

    def test_selection_strategy_stage_and_fallback(self):
        strategy = CodingToolSelectionStrategy()
        assert "grep" in strategy.get_tools_for_stage("understand", "coding")
        # stage map membership (order is nondeterministic — set-built)
        assert set(strategy.get_tools_for_stage("implement", "coding")) == {
            "write", "shell", "edit",
        }
        # coding's fallback is [] (plus any task-type enhancements) — NOT the
        # devops [read, shell]; per-vertical fallback is kernel data.
        assert strategy.get_tools_for_stage("unknown_stage", "any") == []


class TestCodingSafetyProvider:
    def test_extension_backed_surface(self):
        provider = CodingSafetyProvider()
        # coding's provider is extension-backed (not the devops pattern shape)
        assert isinstance(provider.get_extensions(), list)
        assert isinstance(provider.get_bash_patterns(), list)
        assert isinstance(provider.get_file_patterns(), list)
        assert isinstance(provider.get_tool_restrictions(), dict)


class TestCodingPromptAndWorkflowProviders:
    def test_prompt_sections(self):
        sections = CodingPromptProvider().get_system_prompt_sections()
        assert isinstance(sections, dict) and sections
        assert isinstance(CodingPromptProvider().get_prompt_contributors(), list)

    def test_workflow_registry(self):
        provider = CodingWorkflowProvider()
        names = provider.list_workflows()
        assert names == list(provider.get_workflows().keys())
        assert sorted(names) == sorted(set(names)), "duplicate workflow names"
        for name in names:
            assert provider.get_workflow(name) is not None
        assert provider.get_workflow("no_such_workflow") is None

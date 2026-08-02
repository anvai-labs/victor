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

"""Description-contract guard tests (Tool Reliability P9).

Telemetry mining (13 days, 4,612 tool results) showed the top tool-error
causes are avoidable first-call mistakes the LLM makes because the tool
description does not warn against them (grep-style flags to ``code``,
porcelain flags to ``git``, readonly-pinned ``shell`` calls, type-less
``edit`` ops, argument-less ``write``). The fixes fold each lesson into the
docstring that reaches the model via the ``@tool`` decorator.

These tests are the cheap regression net: each tool's LLM-facing description
must keep its load-bearing phrase so future docstring edits don't silently
drop the lesson. Phrases whose docstring changes land in other PRs of the
release train (R3/R4/R5) are marked ``xfail(strict=False)`` with a TODO to
tighten once the train merges.
"""

import pytest


def _llm_description(tool_func) -> str:
    """The description the LLM actually sees (short + long docstring).

    Mirrors ``victor/tools/decorators.py::_create_tool_class`` — content after
    the first ``Args:`` section does NOT reach the model, so asserting on the
    decorator-built ``Tool`` instance's description (not the raw ``__doc__``)
    also guards against a phrase drifting below the Args cutoff.
    """
    return tool_func.Tool.description


class TestShellDescriptionContract:
    """shell: 17× readonly-mode friction — the model must learn up front that
    read-intent turns pin readonly and how to declare mutating intent."""

    def test_shell_docstring_mentions_action_exec(self):
        from victor.tools.bash import shell

        assert "action='exec'" in (shell.__doc__ or "")

    def test_shell_llm_description_carries_readonly_pin_warning(self):
        from victor.tools.bash import shell

        description = _llm_description(shell)
        assert "action='exec'" in description
        assert "pinned readonly" in description

    def test_shell_readonly_rejection_mentions_turn_classification(self):
        """The runtime rejection must explain WHY readonly was on (the turn
        may have been intent-pinned by the pipeline, not chosen by the model)."""
        import inspect

        from victor.tools import bash

        source = inspect.getsource(bash)
        assert "may have been classified read-only" in source


class TestAsiGuidanceContract:
    def test_asi_guidance_says_code_git_are_not_shells(self):
        from victor.agent.prompt_section_texts import ASI_TOOL_EFFECTIVENESS_GUIDANCE

        assert "NOT shells" in ASI_TOOL_EFFECTIVENESS_GUIDANCE
        assert "action='exec'" in ASI_TOOL_EFFECTIVENESS_GUIDANCE


class TestPeerToolDescriptionContracts:
    """Contracts for docstrings that land with other PRs of the R-train.

    TODO(tool-reliability): flip each xfail to a firm assertion once the
    corresponding PR (R3 code, R4 git, R5 edit/write) merges to develop.
    """

    @pytest.mark.xfail(strict=False, reason="code docstring update lands with R3")
    def test_code_docstring_says_grep_flags_not_needed(self):
        from victor.tools.unified.code_tool import code_tool

        assert "grep flags you do NOT need" in _llm_description(code_tool)

    @pytest.mark.xfail(strict=False, reason="git docstring update lands with R4")
    def test_git_docstring_points_to_shell_fallback(self):
        from victor.tools.unified.git_tool import git_tool

        assert "shell(cmd='git" in _llm_description(git_tool)

    @pytest.mark.xfail(strict=False, reason="edit docstring update lands with R5")
    def test_edit_docstring_says_each_op_requires_type(self):
        from victor.tools.file_editor_tool import edit

        assert "REQUIRES `type`" in _llm_description(edit)

    @pytest.mark.xfail(strict=False, reason="write docstring update lands with R5")
    def test_write_docstring_says_path_and_content_required(self):
        from victor.tools.filesystem import write

        assert "requires `path` AND `content`" in _llm_description(write)

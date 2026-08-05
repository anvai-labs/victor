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

"""Coding-specific prompt contributions.

This module provides task type hints and system prompt sections
specific to software development tasks. These are injected into
the framework via the PromptContributorProtocol.
"""

from __future__ import annotations

from typing import Dict

from victor_contracts.verticals import PromptContributorProtocol, TaskTypeHint

# Task-type-specific prompt hints for coding tasks
# These guide the model's approach based on detected task type
CODING_TASK_TYPE_HINTS: Dict[str, TaskTypeHint] = {
    "code_generation": TaskTypeHint(
        task_type="code_generation",
        hint="[GENERATE] Write code directly. No exploration needed. Complete implementation.",
        tool_budget=3,
        priority_tools=["write"],
    ),
    "create_simple": TaskTypeHint(
        task_type="create_simple",
        hint="[CREATE] Write file immediately. Skip codebase exploration. One tool call max.",
        tool_budget=2,
        priority_tools=["write"],
    ),
    "create": TaskTypeHint(
        task_type="create",
        hint="[CREATE+CONTEXT] Read 1-2 relevant files, then create. Follow existing patterns.",
        tool_budget=5,
        priority_tools=["read", "write"],
    ),
    "edit": TaskTypeHint(
        task_type="edit",
        hint="[EDIT] Read target file first, then modify. Focused changes only.",
        tool_budget=5,
        priority_tools=["read", "edit"],
    ),
    "search": TaskTypeHint(
        task_type="search",
        hint="[SEARCH] Use grep/ls for exploration. Summarize after 2-4 calls.",
        tool_budget=6,
        priority_tools=["grep", "ls", "code_search"],
    ),
    "action": TaskTypeHint(
        task_type="action",
        hint="[ACTION] Execute git/test/build operations. Multiple tool calls allowed. Continue until complete.",
        tool_budget=15,
        priority_tools=["shell", "git", "test"],
    ),
    "analysis_deep": TaskTypeHint(
        task_type="analysis_deep",
        hint="[ANALYSIS] Thorough codebase exploration. Read all relevant modules. Comprehensive output.",
        tool_budget=25,
        priority_tools=["read", "grep", "code_search", "ls"],
    ),
    "analyze": TaskTypeHint(
        task_type="analyze",
        hint="[ANALYZE] Examine code carefully. Read related files. Structured findings.",
        tool_budget=12,
        priority_tools=["read", "grep", "symbol"],
    ),
    "design": TaskTypeHint(
        task_type="design",
        hint="""[ARCHITECTURE] For architecture/component questions:
USE STRUCTURED GRAPH FIRST:
- Call architecture_summary to get module pagerank/centrality with edge_counts + 2–3 callsites (runtime-only). Avoid ad-hoc graph/find hops unless data is missing.
- Keep modules vs symbols separate; cite CALLS/INHERITS/IMPORTS counts and callsites (file:line) per hotspot.
- Prefer runtime code; ignore tests/venv/build outputs unless explicitly requested.
DOC-FIRST STRATEGY (mandatory order):
1. FIRST: Read architecture docs if they exist:
   - read_file CLAUDE.md, .victor/init.md, README.md, ARCHITECTURE.md
   - These contain component lists, named implementations, and key relationships
2. SECOND: Explore implementation directories systematically:
   - list_directory on src/, lib/, engines/, impls/, modules/, core/, services/
   - Directory names under impls/ or engines/ are often named implementations
   - Look for ALL-CAPS directory/file names - these are typically named engines/components
3. THIRD: Read key implementation files for each component found
4. FOURTH: Look for benchmark/test files (benches/, *_bench*, *_test*) for performance insights

DISCOVERY PATTERNS - Look for:
- Named implementations: Directories with ALL-CAPS names (engines, stores, protocols)
- Factories/registries: Files named *_factory.*, *_registry.*, mod.rs, index.ts
- Core abstractions: base.py, interface.*, trait definitions
- Configuration: *.yaml, *.toml in config/ directories

Output requirements:
- Use discovered component names (not generic descriptions like "storage module")
- Include file:line references (e.g., "src/engines/impl.rs:42")
- Verify improvements reference ACTUAL code patterns (grep first)
Use 15-20 tool calls minimum. Prioritize by architectural importance.""",
        tool_budget=25,
        priority_tools=["read", "ls", "grep", "code_search"],
    ),
    "refactor": TaskTypeHint(
        task_type="refactor",
        hint="[REFACTOR] Analyze code structure first. Use refactoring tools. Verify with tests.",
        tool_budget=15,
        priority_tools=["read", "rename", "test"],
    ),
    "debug": TaskTypeHint(
        task_type="debug",
        hint="[DEBUG] Read error context. Trace execution flow. Find root cause before fixing.",
        tool_budget=12,
        priority_tools=["read", "grep", "shell"],
    ),
    "test": TaskTypeHint(
        task_type="test",
        hint="[TEST] Run tests first. Analyze failures. Fix issues incrementally.",
        tool_budget=10,
        priority_tools=["test", "read", "edit"],
    ),
    "general": TaskTypeHint(
        task_type="general",
        hint="[GENERAL] Moderate exploration. 3-6 tool calls. Answer concisely.",
        tool_budget=8,
        priority_tools=["read", "grep", "ls"],
    ),
    "bug_fix": TaskTypeHint(
        task_type="bug_fix",
        hint="""[BUG FIX] Resolve a GitHub issue or bug report. CRITICAL WORKFLOW:

PHASE 1 - UNDERSTAND (max 5 file reads):
1. Read the file(s) mentioned in the error traceback/issue
2. Read related imports and dependencies (1-2 files max)
3. Identify the root cause from the code

PHASE 2 - FIX (MANDATORY after Phase 1):
4. Use edit_file or write_file to make the fix
5. The fix should be minimal and surgical - only change what's necessary
6. If the issue suggests a fix (e.g., "add quiet=True"), implement exactly that

PHASE 3 - VERIFY (optional):
7. If tests exist, run them to verify the fix

CRITICAL RULES:
- DO NOT read more than 5-7 files before making an edit
- After reading the traceback/error location, you have enough context to edit
- Prefer SMALL, FOCUSED changes over large refactors
- If unsure, make the minimal fix that addresses the reported issue
- Say "Fix applied" when done editing

ANTI-PATTERNS TO AVOID:
- Reading the entire codebase before editing
- Exploring tangential files not in the error trace
- Waiting for "perfect understanding" before acting
- Re-reading files you've already read""",
        tool_budget=12,
        priority_tools=["read", "edit", "test", "shell"],
    ),
    "issue_resolution": TaskTypeHint(
        task_type="issue_resolution",
        hint="[ISSUE] Same as bug_fix - resolve GitHub issue with focused edits after minimal exploration.",
        tool_budget=12,
        priority_tools=["read", "edit", "test", "shell"],
    ),
}


# Coding-specific grounding rules
CODING_GROUNDING_RULES = """
GROUNDING: Base ALL responses on tool output only. Never invent file paths or content.
Quote code exactly from tool output. If more info needed, call another tool.
""".strip()


# Extended grounding for local models
CODING_GROUNDING_EXTENDED = """
CRITICAL - TOOL OUTPUT GROUNDING:
When you receive tool output in <TOOL_OUTPUT> tags:
1. The content between ═══ markers is ACTUAL file/command output - NEVER ignore it
2. You MUST base your analysis ONLY on this actual content
3. NEVER fabricate, invent, or imagine file contents that differ from tool output
4. If you need more information, call another tool - do NOT guess
5. When citing code, quote EXACTLY from the tool output
6. If tool output is empty or truncated, acknowledge this limitation

VIOLATION OF THESE RULES WILL RESULT IN INCORRECT ANALYSIS.
""".strip()


# Coding-specific system prompt section
CODING_SYSTEM_PROMPT_SECTION = """
When exploring code:
- Use semantic_code_search for conceptual queries ("authentication logic")
- Use code_search for exact patterns ("def authenticate")
- Use overview to understand file structure

When modifying code:
- Use edit for surgical changes to existing code
- Use write only for new files or complete rewrites
- Always verify changes compile/pass tests when possible

Code quality guidelines:
1. **Understand before modifying**: Always read and understand code before making changes
2. **Incremental changes**: Make small, focused changes rather than large rewrites
3. **Verify changes**: Run tests or validation after modifications
4. **Explain reasoning**: Briefly explain your approach when making non-trivial changes
5. **Preserve style**: Match existing code style and patterns
6. **Handle errors gracefully**: If something fails, diagnose and recover
""".strip()


class CodingPromptContributor(PromptContributorProtocol):
    """Prompt contributor for coding vertical.

    Provides coding-specific task type hints and system prompt sections
    for integration with the framework's prompt builder.
    """

    def __init__(self, use_extended_grounding: bool = False):
        """Initialize the prompt contributor.

        Args:
            use_extended_grounding: Whether to use extended grounding rules
                                   (typically for local models)
        """
        self._use_extended_grounding = use_extended_grounding

    def get_task_type_hints(self) -> Dict[str, TaskTypeHint]:
        """Get coding-specific task type hints.

        Returns:
            Dict mapping task types to their hints
        """
        return CODING_TASK_TYPE_HINTS.copy()

    def get_system_prompt_section(self) -> str:
        """Get coding-specific system prompt section.

        Returns:
            System prompt text for coding tasks
        """
        return CODING_SYSTEM_PROMPT_SECTION

    def get_grounding_rules(self) -> str:
        """Get coding-specific grounding rules.

        Returns:
            Grounding rules text
        """
        if self._use_extended_grounding:
            return CODING_GROUNDING_EXTENDED
        return CODING_GROUNDING_RULES

    def get_priority(self) -> int:
        """Get priority for prompt section ordering.

        Returns:
            Priority value (coding is primary, so high priority)
        """
        return 10


def get_task_type_hint(task_type: str) -> str:
    """Get prompt hint for a specific task type.

    Convenience function for backward compatibility.

    Args:
        task_type: The detected task type (e.g., "create_simple", "edit")

    Returns:
        Task-specific prompt hint or empty string if not found
    """
    hint = CODING_TASK_TYPE_HINTS.get(task_type.lower())
    return hint.hint if hint else ""


__all__ = [
    "CodingPromptContributor",
    "CODING_TASK_TYPE_HINTS",
    "CODING_GROUNDING_RULES",
    "CODING_GROUNDING_EXTENDED",
    "CODING_SYSTEM_PROMPT_SECTION",
    "get_task_type_hint",
]

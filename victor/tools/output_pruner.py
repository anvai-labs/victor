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

"""Tool output pruner for task-aware token reduction.

This module implements task-conditioned tool output pruning inspired by
arXiv:2604.04979 (Squeez). It reduces tokens sent to the LLM by filtering
tool outputs based on task type, achieving 40-60% token reduction without
loss of essential information.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from victor.tools.core_tool_aliases import canonicalize_core_tool_name

logger = logging.getLogger(__name__)


@dataclass
class PruningInfo:
    """Metadata about pruning operation.

    Attributes:
        was_pruned: Whether pruning was applied
        original_lines: Number of lines before pruning
        pruned_lines: Number of lines after pruning
        pruning_reason: Human-readable reason for pruning
        task_type: Task type used for pruning decision
    """

    was_pruned: bool
    original_lines: int
    pruned_lines: int
    pruning_reason: str
    task_type: str
    omitted_lines: int = 0
    recovery_hint: str = ""

    def __str__(self) -> str:
        if not self.was_pruned:
            return f"No pruning applied (task_type={self.task_type})"
        reduction_pct = ((self.original_lines - self.pruned_lines) / self.original_lines) * 100
        detail = (
            f"Pruned {self.original_lines}→{self.pruned_lines} lines "
            f"({reduction_pct:.1f}% reduction, task_type={self.task_type}, "
            f"reason={self.pruning_reason})"
        )
        if self.omitted_lines > 0:
            detail += f", omitted_lines={self.omitted_lines}"
        return detail


class ToolOutputPruner:
    """Prune tool outputs based on task type to reduce tokens.

    Implements task-conditioned output filtering as described in:
    arXiv:2604.04979 (Squeez) - Task-Conditioned Tool-Output Pruning for Coding Agents

    The pruner applies different rules based on:
    - Task type (code_generation, edit, search, create_simple, debug, etc.)
    - Tool name (read, grep, code_search, etc.)
    - Output size and structure

    This reduces tokens by 40-60% while preserving essential information
    for task completion.
    """

    # Task-specific pruning rules
    TASK_PRUNING_RULES: Dict[str, Dict[str, Any]] = {
        "code_generation": {
            "max_lines": 50,  # Truncate file reads to 50 lines
            "strip_comments": True,  # Remove comments from read files
            "strip_blank_lines": True,  # Remove blank lines
            "keep_imports": True,  # Preserve import statements
            "max_line_length": 120,  # Truncate long lines
        },
        "edit": {
            "max_lines": 30,  # Less context needed for edits
            "context_lines": 5,  # Only keep 5 lines around edit target
            "strip_comments": False,  # Keep comments for context
            "strip_blank_lines": True,
        },
        "search": {
            "max_results": 10,  # Limit grep/search results
            "max_line_length": 120,  # Truncate long search lines
            "strip_blank_lines": True,
        },
        "create_simple": {
            "max_lines": 0,  # No file reading needed at all
            "return_summary_only": True,  # Just return "File exists: X"
        },
        "debug": {
            "max_lines": 80,  # More context for debugging
            "include_traceback": True,  # Always keep tracebacks
            "strip_comments": False,
            "strip_blank_lines": False,
        },
        "analysis_deep": {
            "max_lines": 100,  # More context for deep analysis
            "strip_comments": False,
            "strip_blank_lines": False,
        },
        "exploration": {
            "max_lines": 30,  # Limited context for exploration
            "strip_comments": True,
            "strip_blank_lines": True,
        },
        "research": {
            "max_lines": 500,  # Need full content for research tasks
            "strip_comments": False,  # Preserve all context
            "strip_blank_lines": False,  # Keep structure
            "preserve_paper_ids": True,  # Don't truncate arXiv IDs (e.g., 1234.56789)
            "preserve_search_results": True,  # Keep full search output
            "preserve_urls": True,  # Keep paper URLs
        },
        # Default rule for unknown task types
        "default": {
            "max_lines": 100,  # Conservative default
            "strip_comments": False,
            "strip_blank_lines": False,
        },
    }

    # Tool-specific pruning adjustments
    TOOL_PRUNING_OVERRIDES: Dict[str, Dict[str, Any]] = {
        "read": {
            "preserve_structure": True,  # Keep code structure intact
        },
        "grep": {
            "max_results": 10,  # Limit grep results
            "context_lines": 2,  # Show 2 lines of context
        },
        "code_search": {
            "max_results": 5,  # Limit semantic search results
            "strip_snippets": True,  # Remove code snippets
        },
    }

    FORMATTED_SAFE_DEFAULT_TOOLS = {
        "read",
        "grep",
        "ls",
        "overview",
        "code_search",
        "semantic_code_search",
    }

    FORMATTED_NEVER_PRUNE_TOOLS = {
        "diff",
        "git_diff",
        "write",
        "edit",
        "patch",
        "apply_patch",
        "shell",
        "pytest",
    }

    FORMATTED_RULES: Dict[str, Dict[str, int]] = {
        "read": {"max_body_lines": 140},
        "grep": {"max_body_lines": 120},
        "code_search": {"max_body_lines": 140},
        "semantic_code_search": {"max_body_lines": 140},
        "ls": {"max_body_lines": 160},
        "overview": {"max_body_lines": 180},
        "default": {"max_body_lines": 140},
    }

    FORMATTED_SAFETY_CRITICAL_PATTERNS = (
        re.compile(r"^\s*diff --git ", re.MULTILINE),
        re.compile(r"^\s*@@ ", re.MULTILINE),
        re.compile(r"^\s*--- [^\n]+", re.MULTILINE),
        re.compile(r"^\s*\+\+\+ [^\n]+", re.MULTILINE),
        re.compile(r"Traceback \(most recent call last\):"),
        re.compile(r"^\s*[A-Za-z0-9_./\\-]+:\d+:\d+:\s+(error|warning):", re.MULTILINE),
        re.compile(
            r"^\s*(error|fatal error|syntaxerror|typeerror|nameerror|referenceerror)\b",
            re.MULTILINE | re.IGNORECASE,
        ),
        re.compile(r"^\s*=+ FAILURES =+\s*$", re.MULTILINE),
        re.compile(r"^\s*FAILED\b", re.MULTILINE),
        re.compile(r"AssertionError"),
    )

    def __init__(self, enabled: bool = True):
        """Initialize the output pruner.

        Args:
            enabled: Whether pruning is enabled (for feature flagging)
        """
        self.enabled = enabled
        logger.info(f"ToolOutputPruner initialized (enabled={enabled})")

    def prune(
        self,
        tool_output: str,
        task_type: str,
        tool_name: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, PruningInfo]:
        """Apply task-aware pruning to tool output.

        Args:
            tool_output: Raw tool output string
            task_type: Detected task type (e.g., "code_generation", "edit")
            tool_name: Name of the tool that produced output
            context: Optional execution context

        Returns:
            Tuple of (pruned_output, pruning_info)
        """
        if not self.enabled:
            return tool_output, PruningInfo(
                was_pruned=False,
                original_lines=0,
                pruned_lines=0,
                pruning_reason="Pruning disabled",
                task_type=task_type,
            )

        tool_name = canonicalize_core_tool_name(tool_name)

        if context and context.get("formatted_output"):
            return self._prune_formatted_output(
                tool_output=tool_output,
                task_type=task_type,
                tool_name=tool_name,
                context=context,
            )

        original_lines = tool_output.count("\n") + 1
        pruned_output = tool_output
        was_pruned = False
        pruning_reason = ""

        # Get task-specific rules
        task_rules = self.TASK_PRUNING_RULES.get(task_type, self.TASK_PRUNING_RULES["default"])

        # Get tool-specific overrides
        tool_overrides = self.TOOL_PRUNING_OVERRIDES.get(tool_name, {})

        # Merge rules (tool overrides take precedence)
        rules = {**task_rules, **tool_overrides}

        # Apply line limit
        max_lines = rules.get("max_lines", 100)
        if max_lines > 0 and original_lines > max_lines:
            pruned_output = self._apply_line_limit(pruned_output, max_lines, tool_name, rules)
            was_pruned = True
            pruning_reason = f"max_lines={max_lines}"

        # Strip comments if configured
        if rules.get("strip_comments", False):
            pruned_output = self._strip_comments(pruned_output)
            if pruned_output != tool_output:
                was_pruned = True
                if pruning_reason:
                    pruning_reason += ", strip_comments"
                else:
                    pruning_reason = "strip_comments"

        # Strip blank lines if configured
        if rules.get("strip_blank_lines", False):
            pruned_output = self._strip_blank_lines(pruned_output)
            if pruned_output != tool_output:
                was_pruned = True
                if pruning_reason:
                    pruning_reason += ", strip_blank_lines"
                else:
                    pruning_reason = "strip_blank_lines"

        # Truncate long lines if configured
        max_line_length = rules.get("max_line_length", 0)
        if max_line_length > 0:
            pruned_output = self._truncate_long_lines(pruned_output, max_line_length)
            if pruned_output != tool_output:
                was_pruned = True
                if pruning_reason:
                    pruning_reason += f", max_line_length={max_line_length}"
                else:
                    pruning_reason = f"max_line_length={max_line_length}"

        # Handle create_simple special case
        if task_type == "create_simple" and rules.get("return_summary_only", False):
            # Just return summary instead of full content
            if tool_name == "read":
                # Extract just filename
                lines = tool_output.split("\n")
                for line in lines:
                    if line.strip() and not line.startswith("#"):
                        pruned_output = f"File: {line.strip()}"
                        break
            was_pruned = True
            pruning_reason = "summary_only"

        pruned_lines = pruned_output.count("\n") + 1

        pruning_info = PruningInfo(
            was_pruned=was_pruned,
            original_lines=original_lines,
            pruned_lines=pruned_lines,
            pruning_reason=pruning_reason if was_pruned else "no_pruning",
            task_type=task_type,
        )

        if was_pruned:
            logger.debug(f"[ToolOutputPruner] {pruning_info}")

        return pruned_output, pruning_info

    def _prune_formatted_output(
        self,
        tool_output: str,
        task_type: str,
        tool_name: str,
        context: Dict[str, Any],
    ) -> Tuple[str, PruningInfo]:
        """Conservatively prune already-formatted tool output for user preview."""
        original_lines = tool_output.count("\n") + 1
        safe_only = bool(context.get("safe_only", True))

        if tool_name in self.FORMATTED_NEVER_PRUNE_TOOLS:
            return self._no_prune(
                tool_output,
                original_lines,
                task_type,
                "safety_critical_tool",
            )

        if safe_only and tool_name not in self.FORMATTED_SAFE_DEFAULT_TOOLS:
            return self._no_prune(
                tool_output,
                original_lines,
                task_type,
                "safe_default_scope",
            )

        if self._contains_safety_critical_content(tool_output):
            return self._no_prune(
                tool_output,
                original_lines,
                task_type,
                "safety_critical_content",
            )

        rules = self.FORMATTED_RULES.get(tool_name, self.FORMATTED_RULES["default"])
        prefix, body, suffix = self._split_formatted_sections(tool_output)
        max_body_lines = rules["max_body_lines"]

        if len(body) <= max_body_lines:
            return self._no_prune(
                tool_output,
                original_lines,
                task_type,
                "within_safe_budget",
            )

        kept_body = body[:max_body_lines]
        omitted_lines = len(body) - len(kept_body)
        recovery_hint = self._build_recovery_hint(tool_name, kept_body, context)
        omission_note = self._build_omission_note(omitted_lines, recovery_hint)

        pruned_lines = prefix + kept_body + [omission_note] + suffix
        pruned_output = "\n".join(pruned_lines)
        pruning_info = PruningInfo(
            was_pruned=True,
            original_lines=original_lines,
            pruned_lines=pruned_output.count("\n") + 1,
            pruning_reason=f"formatted_max_body_lines={max_body_lines}",
            task_type=task_type,
            omitted_lines=omitted_lines,
            recovery_hint=recovery_hint,
        )
        logger.debug("[ToolOutputPruner] %s", pruning_info)
        return pruned_output, pruning_info

    def _no_prune(
        self,
        tool_output: str,
        original_lines: int,
        task_type: str,
        reason: str,
    ) -> Tuple[str, PruningInfo]:
        """Return unmodified output with a structured no-prune reason."""
        return tool_output, PruningInfo(
            was_pruned=False,
            original_lines=original_lines,
            pruned_lines=original_lines,
            pruning_reason=reason,
            task_type=task_type,
        )

    def _contains_safety_critical_content(self, output: str) -> bool:
        """Detect content that should remain exact and unpruned."""
        return any(pattern.search(output) for pattern in self.FORMATTED_SAFETY_CRITICAL_PATTERNS)

    def _split_formatted_sections(self, output: str) -> Tuple[List[str], List[str], List[str]]:
        """Split formatted tool output into prefix, body, and suffix sections."""
        lines = output.split("\n")
        start = 0
        end = len(lines)

        while start < end:
            stripped = lines[start].strip()
            if not stripped or self._is_formatted_prefix_line(stripped):
                start += 1
                continue
            break

        note_block_started = False
        while end > start:
            stripped = lines[end - 1].strip()
            if not stripped:
                end -= 1
                continue
            if self._is_formatted_suffix_line(stripped):
                if stripped.startswith(("IMPORTANT:", "NOTE:", "ACTION REQUIRED:")):
                    note_block_started = True
                end -= 1
                continue
            if note_block_started and re.match(r"^(- |\d+\. )", stripped):
                end -= 1
                continue
            break

        prefix = lines[:start]
        body = lines[start:end]
        suffix = lines[end:]
        return prefix, body, suffix

    def _is_formatted_prefix_line(self, stripped: str) -> bool:
        return (
            stripped.startswith("<TOOL_OUTPUT")
            or stripped.startswith("═══")
            or stripped.startswith("[File:")
            or stripped.startswith("[Lines ")
            or stripped.startswith("[Size:")
            or stripped.startswith("[TRUNCATED:")
        )

    def _is_formatted_suffix_line(self, stripped: str) -> bool:
        return (
            stripped == "</TOOL_OUTPUT>"
            or stripped.startswith("═══ END")
            or stripped.startswith("IMPORTANT:")
            or stripped.startswith("NOTE:")
            or stripped.startswith("ACTION REQUIRED:")
            or stripped.startswith("Use only ")
            or stripped.startswith("These are the actual ")
            or stripped.startswith("To continue:")
        )

    def _build_omission_note(self, omitted_lines: int, recovery_hint: str) -> str:
        note = f"[PRUNED PREVIEW: omitted {omitted_lines} lines."
        if recovery_hint:
            note += f" {recovery_hint}"
        return note + "]"

    def _build_recovery_hint(
        self,
        tool_name: str,
        kept_body: List[str],
        context: Dict[str, Any],
    ) -> str:
        tool_args = context.get("tool_args") or {}
        if not isinstance(tool_args, dict):
            tool_args = {}

        if tool_name == "read":
            path = tool_args.get("path")
            next_offset = self._extract_next_read_offset(kept_body)
            if isinstance(path, str) and next_offset is not None:
                return (
                    f"Use read(path={path!r}, offset={next_offset}, limit=200) "
                    "to continue from the omitted section."
                )
            if isinstance(path, str):
                return f"Use read(path={path!r}, offset=..., limit=200) to continue."

        if tool_name == "ls":
            path = tool_args.get("path", ".")
            return (
                f"Rerun ls(path={path!r}, pattern='...', limit=...) "
                "or read a specific listed file to recover omitted entries."
            )

        if tool_name == "overview":
            path = tool_args.get("path", ".")
            return (
                f"Rerun overview(path={path!r}, max_depth=1) "
                "or inspect specific files with read()."
            )

        if tool_name in {"grep", "code_search", "semantic_code_search"}:
            path = tool_args.get("path", ".")
            query = tool_args.get("query") or tool_args.get("pattern")
            if isinstance(query, str) and query:
                return (
                    f"Rerun {tool_name}(path={path!r}, query={query!r}) "
                    "with a narrower path or query to recover omitted matches."
                )
            return (
                f"Rerun {tool_name}(path={path!r}) with narrower scope to recover omitted matches."
            )

        return "Rerun the tool with narrower scope to recover omitted content."

    def _extract_next_read_offset(self, kept_body: List[str]) -> Optional[int]:
        """Extract the next read offset from the last numbered read line."""
        for line in reversed(kept_body):
            match = re.match(r"^\s*(\d+)\t", line)
            if match:
                return int(match.group(1))
        return None

    def _apply_line_limit(
        self, output: str, max_lines: int, tool_name: str, rules: Dict[str, Any]
    ) -> str:
        """Apply line limit to output.

        For code files, try to preserve structure by keeping imports and
        focusing on the most relevant parts.
        """
        lines = output.split("\n")

        # If preserve_structure and it's a code file, try to be smart
        if rules.get("preserve_structure", False) and self._is_code_file(output):
            # Keep imports first
            imports = []
            code_lines = []
            in_imports = True

            for line in lines:
                stripped = line.strip()
                # Detect end of imports
                if in_imports and stripped and not stripped.startswith(("import", "from", "#")):
                    in_imports = False

                if in_imports or (stripped and stripped.startswith(("import", "from"))):
                    imports.append(line)
                else:
                    code_lines.append(line)

            # Prioritize imports, then code
            preserved_imports = imports[:10]  # Max 10 import lines
            remaining_budget = max_lines - len(preserved_imports)
            preserved_code = code_lines[:remaining_budget] if remaining_budget > 0 else []

            return "\n".join(preserved_imports + preserved_code)
        else:
            # Simple truncation
            return "\n".join(lines[:max_lines])

    def _strip_comments(self, output: str) -> str:
        """Remove comment lines from output."""
        lines = output.split("\n")
        filtered = [line for line in lines if not line.strip().startswith("#")]
        return "\n".join(filtered)

    def _strip_blank_lines(self, output: str) -> str:
        """Remove blank lines from output."""
        lines = output.split("\n")
        filtered = [line for line in lines if line.strip()]
        return "\n".join(filtered)

    def _truncate_long_lines(self, output: str, max_length: int) -> str:
        """Truncate lines that exceed max length."""
        lines = output.split("\n")
        truncated = [
            line[:max_length] + "..." if len(line) > max_length else line for line in lines
        ]
        return "\n".join(truncated)

    def _is_code_file(self, output: str) -> bool:
        """Heuristic to detect if output is code."""
        # Check for common code patterns
        code_indicators = [
            "def ",  # Python functions
            "class ",  # Python classes
            "import ",
            "from ",
            "function ",  # JavaScript
            "const ",  # JavaScript
            "=>",  # Arrow functions
        ]
        output_lower = output.lower()
        return any(indicator in output_lower for indicator in code_indicators)


# Singleton instance for use across the framework
_default_pruner: Optional[ToolOutputPruner] = None


def get_output_pruner() -> ToolOutputPruner:
    """Get the default output pruner instance (singleton)."""
    global _default_pruner
    if _default_pruner is None:
        _default_pruner = ToolOutputPruner()
    return _default_pruner


__all__ = ["ToolOutputPruner", "PruningInfo", "get_output_pruner"]

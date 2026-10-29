# Copyright 2025 Vijaykumar Singh
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

"""Consolidated Prompt Builder for Victor Framework.

This module provides a unified, composable prompt building system that consolidates
duplicate prompt construction logic from across the codebase. It supports:

- Section-based composition with priority ordering
- Tool hints integration
- Safety rules injection
- Contextual information merging
- Grounding rules (minimal vs extended)
- Task-type-specific hints

The PromptBuilder uses a fluent API pattern for ergonomic construction:

    prompt = (
        PromptBuilder()
        .add_section("identity", "You are an expert code analyst.", priority=10)
        .add_section("guidelines", "Follow best practices.", priority=20)
        .add_tool_hints({"read": "Use to read file contents"})
        .add_safety_rules(["Never expose credentials"])
        .add_context("Working in Python codebase")
        .build()
    )

Integration with verticals:

    # Prompt contributors are provided by external vertical packages
    # (e.g., victor-coding) via victor.prompt_contributors entry points.
    builder = PromptBuilder()
    builder.add_from_contributor(prompt_contributor)
    prompt = builder.build()
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from textwrap import dedent
from enum import Enum
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
)

# Python 3.11+ has typing.Self, 3.10 needs typing_extensions
try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

if TYPE_CHECKING:
    from victor.core.verticals.protocols import PromptContributorProtocol

from victor.agent.prompting.document import (
    PromptBlock,
    PromptDeduplicationProcessor,
    PromptDocument,
    PromptPriorityTrimProcessor,
)
from victor.context.instruction_discovery import (
    DEFAULT_INSTRUCTION_FILENAMES,
    discover_instruction_files as discover_shared_instruction_files,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Enums for Prompt Building
# =============================================================================


class PromptScope(str, Enum):
    """Scope level for prompt sections.

    Determines where prompt sections are visible:
    - WORKSPACE: All agents in the workspace
    - PROJECT: All agents in the project
    - USER: User-specific configuration
    """

    WORKSPACE = "workspace"  # All agents in the workspace
    PROJECT = "project"  # All agents in the project
    USER = "user"  # User-specific configuration


# Backward compatibility
_PromptScopeLiteral = Literal["workspace", "project", "user"]


@dataclass
class PromptSection(PromptBlock):
    """A section of the system prompt.

    Sections are composed together to form the final prompt, ordered by priority.
    Lower priority values appear earlier in the final prompt.

    Attributes:
        name: Unique identifier for the section (e.g., "identity", "grounding")
        content: The actual prompt text for this section
        priority: Ordering priority (lower = earlier). Default is 50.
        enabled: Whether this section is included in the final prompt
        header: Optional header format. If None, uses "## {name}". Set to "" to disable.
    """

    pass


@dataclass
class ToolHint:
    """A hint for using a specific tool.

    Attributes:
        tool_name: The canonical name of the tool
        hint: Usage guidance for the tool
        priority_boost: Boost to apply when selecting this tool (0.0-1.0)
    """

    tool_name: str
    hint: str
    priority_boost: float = 0.0


class PromptBuilder:
    """Builds system prompts from composable sections.

    This class consolidates prompt building logic from across the codebase,
    providing a unified API for constructing system prompts with:

    - Named sections with priority ordering
    - Tool usage hints
    - Safety rules
    - Contextual information
    - Grounding rules

    The builder uses a fluent API pattern, allowing method chaining:

        prompt = (
            PromptBuilder()
            .add_section("identity", "You are an AI assistant.")
            .add_tool_hints({"read": "Read file contents"})
            .build()
        )

    Attributes:
        _sections: Dictionary of named prompt sections
        _tool_hints: Dictionary of tool name to hint text
        _safety_rules: List of safety rules to include
        _context: List of contextual information strings
        _grounding_mode: "minimal" or "extended" grounding rules
    """

    # Standard priority levels for common sections
    PRIORITY_IDENTITY = 10  # Who the assistant is
    PRIORITY_CAPABILITIES = 20  # What it can do
    PRIORITY_GUIDELINES = 30  # How to behave
    PRIORITY_TASK_HINTS = 40  # Task-specific guidance
    PRIORITY_TOOL_GUIDANCE = 50  # Tool usage hints
    PRIORITY_SAFETY = 60  # Safety rules
    PRIORITY_CONTEXT = 70  # Contextual information
    PRIORITY_GROUNDING = 80  # Grounding rules (last)

    def __init__(self) -> None:
        """Initialize a new PromptBuilder."""
        self._document = PromptDocument()
        self._sections: Dict[str, PromptSection] = self._document.blocks  # Backward compat
        self._tool_hints: Dict[str, ToolHint] = {}
        self._safety_rules: List[str] = []
        self._context: List[str] = []
        self._grounding_mode: str = "minimal"
        self._custom_grounding: Optional[str] = None

    def add_section(
        self,
        name: str,
        content: str,
        priority: int = 50,
        header: Optional[str] = None,
    ) -> Self:
        """Add a named section to the prompt.

        Sections are ordered by priority when building the final prompt.
        Lower priority values appear earlier.

        Args:
            name: Unique identifier for the section
            content: The prompt text for this section
            priority: Ordering priority (default 50, lower = earlier)
            header: Optional custom header. None = auto-generate, "" = no header

        Returns:
            Self for method chaining
        """
        self._document.upsert(
            PromptSection(
                name=name,
                content=content,
                priority=priority,
                enabled=True,
                header=header,
            )
        )
        return self

    def add_tool_hints(self, hints: Dict[str, str]) -> Self:
        """Add tool usage hints.

        Tool hints provide guidance on how to use specific tools effectively.
        These are merged into the prompt in the tool guidance section.

        Args:
            hints: Dictionary mapping tool names to hint text

        Returns:
            Self for method chaining
        """
        for tool_name, hint in hints.items():
            self._tool_hints[tool_name] = ToolHint(tool_name=tool_name, hint=hint)
        return self

    def add_tool_hint(
        self,
        tool_name: str,
        hint: str,
        priority_boost: float = 0.0,
    ) -> Self:
        """Add a single tool hint with optional priority boost.

        Args:
            tool_name: The canonical name of the tool
            hint: Usage guidance for the tool
            priority_boost: Boost to apply when selecting this tool (0.0-1.0)

        Returns:
            Self for method chaining
        """
        self._tool_hints[tool_name] = ToolHint(
            tool_name=tool_name,
            hint=hint,
            priority_boost=priority_boost,
        )
        return self

    def add_safety_rules(self, rules: List[str]) -> Self:
        """Add safety rules.

        Safety rules are constraints the model should follow to ensure
        safe operation (e.g., "Never expose credentials").

        Args:
            rules: List of safety rule strings

        Returns:
            Self for method chaining
        """
        self._safety_rules.extend(rules)
        return self

    def add_safety_rule(self, rule: str) -> Self:
        """Add a single safety rule.

        Args:
            rule: Safety rule string

        Returns:
            Self for method chaining
        """
        self._safety_rules.append(rule)
        return self

    def add_context(self, context: str) -> Self:
        """Add contextual information.

        Context provides situational information like project details,
        file types being worked with, or current task state.

        Args:
            context: Contextual information string

        Returns:
            Self for method chaining
        """
        normalized = context.strip()
        if not normalized:
            return self

        if normalized not in self._context:
            self._context.append(normalized)
        return self

    def set_grounding_mode(self, mode: str) -> Self:
        """Set the grounding rules mode.

        Args:
            mode: Either "minimal" or "extended"

        Returns:
            Self for method chaining

        Raises:
            ValueError: If mode is not "minimal" or "extended"
        """
        if mode not in ("minimal", "extended"):
            raise ValueError(f"Invalid grounding mode: {mode}. Use 'minimal' or 'extended'.")
        self._grounding_mode = mode
        return self

    def set_custom_grounding(self, grounding: str) -> Self:
        """Set custom grounding rules (overrides mode).

        Args:
            grounding: Custom grounding rules text

        Returns:
            Self for method chaining
        """
        self._custom_grounding = grounding
        return self

    def remove_section(self, name: str) -> Self:
        """Remove a section by name.

        Args:
            name: The name of the section to remove

        Returns:
            Self for method chaining
        """
        self._sections.pop(name, None)
        return self

    def disable_section(self, name: str) -> Self:
        """Disable a section without removing it.

        Args:
            name: The name of the section to disable

        Returns:
            Self for method chaining
        """
        if name in self._sections:
            self._sections[name].enabled = False
        return self

    def enable_section(self, name: str) -> Self:
        """Enable a previously disabled section.

        Args:
            name: The name of the section to enable

        Returns:
            Self for method chaining
        """
        if name in self._sections:
            self._sections[name].enabled = True
        return self

    def has_section(self, name: str) -> bool:
        """Check if a section exists.

        Args:
            name: The name of the section to check

        Returns:
            True if the section exists, False otherwise
        """
        return name in self._sections and bool(self._sections[name].content.strip())

    def get_section(self, name: str) -> Optional[PromptSection]:
        """Get a section by name.

        Args:
            name: The name of the section to get

        Returns:
            The PromptSection if found, None otherwise
        """
        return self._sections.get(name)

    def ensure_section(
        self,
        name: str,
        content: str,
        priority: int = 50,
        header: Optional[str] = None,
    ) -> Self:
        """Ensure a section exists with non-empty content.

        Adds the section if it is missing or blank.

        Args:
            name: Section name
            content: Section content
            priority: Section priority ordering
            header: Optional header override

        Returns:
            Self for method chaining
        """
        if not self.has_section(name):
            self.add_section(name, content, priority=priority, header=header)
        return self

    def iter_sections(self) -> List[PromptSection]:
        """Return a list of current sections."""
        return [
            section
            for section in self._document.iter_blocks()
            if isinstance(section, PromptSection)
        ]

    def iter_named_sections(self) -> List[Tuple[str, PromptSection]]:
        """Return (name, section) tuples for all sections."""
        return [
            (name, section)
            for name, section in self._document.iter_named_blocks()
            if isinstance(section, PromptSection)
        ]

    def estimate_section_length(self) -> int:
        """Estimate total character length of all sections."""
        return self._document.estimate_length()

    def trim_sections_by_priority(
        self,
        max_total_chars: int,
        protected_sections: Optional[Iterable[str]] = None,
        min_priority: int = 0,
    ) -> None:
        """Trim lower-priority sections to respect a character budget.

        Args:
            max_total_chars: Maximum allowed characters for sections
            protected_sections: Section names that should never be trimmed
            min_priority: Minimum priority threshold eligible for trimming
        """
        if max_total_chars <= 0:
            return

        self._document = PromptPriorityTrimProcessor(
            max_total_chars=max_total_chars,
            protected_blocks=protected_sections or (),
            min_priority=min_priority,
        ).process(self._document)
        self._sections = self._document.blocks

    def deduplicate_sections(self) -> None:
        """Deduplicate or remove empty sections through the canonical processor."""
        self._document = PromptDeduplicationProcessor().process(self._document)
        self._sections = self._document.blocks

    def clear(self) -> Self:
        """Clear all sections, hints, rules, and context.

        Returns:
            Self for method chaining
        """
        self._document = PromptDocument()
        self._sections = self._document.blocks
        self._tool_hints.clear()
        self._safety_rules.clear()
        self._context.clear()
        self._grounding_mode = "minimal"
        self._custom_grounding = None
        return self

    def add_from_contributor(
        self,
        contributor: "PromptContributorProtocol",
        task_type: Optional[str] = None,
    ) -> Self:
        """Add sections from a PromptContributorProtocol implementation.

        This integrates with vertical-specific prompt contributors like
        CodingPromptContributor, DevOpsPromptContributor, etc.

        Args:
            contributor: A PromptContributorProtocol implementation
            task_type: Optional task type to get specific hints for

        Returns:
            Self for method chaining
        """
        from victor.core.verticals.protocols.prompt_provider import (
            collect_prompt_section_contributions,
        )

        # Get priority from contributor
        priority_offset = contributor.get_priority() * 10

        # Add system prompt section(s)
        for contribution in collect_prompt_section_contributions(contributor):
            self.add_section(
                name=contribution.name.lower(),
                content=contribution.text,
                priority=self.PRIORITY_GUIDELINES + priority_offset,
            )

        # Add grounding rules
        grounding = contributor.get_grounding_rules()
        if grounding:
            self.set_custom_grounding(grounding)

        # Add task-type-specific hint if task_type provided
        if task_type:
            hints = contributor.get_task_type_hints()
            if task_type.lower() in hints:
                task_hint = hints[task_type.lower()]
                hint_text = task_hint.hint if hasattr(task_hint, "hint") else str(task_hint)
                self.add_section(
                    name="task_hint",
                    content=hint_text,
                    priority=self.PRIORITY_TASK_HINTS,
                    header="",  # Task hints usually have their own markers like [EDIT]
                )

                # Add tool hints from task type if available
                if hasattr(task_hint, "priority_tools") and task_hint.priority_tools:
                    for tool in task_hint.priority_tools:
                        self.add_tool_hint(tool, f"Prioritized for {task_type}", 0.2)

                # Add execution guidance based on skip flags (arXiv:2604.01681)
                execution_guidance = []
                if hasattr(task_hint, "skip_planning") and task_hint.skip_planning:
                    execution_guidance.append("Execute directly without extensive planning")
                if hasattr(task_hint, "skip_evaluation") and task_hint.skip_evaluation:
                    execution_guidance.append("No need to explicitly verify results")
                if hasattr(task_hint, "token_budget") and task_hint.token_budget:
                    execution_guidance.append(
                        f"Keep response concise (target ~{task_hint.token_budget} tokens)"
                    )

                if execution_guidance:
                    guidance_text = "Execution constraints:\n- " + "\n- ".join(execution_guidance)
                    self.add_section(
                        name="task_execution_guidance",
                        content=guidance_text,
                        priority=self.PRIORITY_GUIDELINES + 5,  # Just after guidelines
                        header="",
                    )

        return self

    def merge(self, other: "PromptBuilder") -> Self:
        """Merge another PromptBuilder into this one.

        Sections from the other builder override sections with the same name.
        Tool hints, safety rules, and context are merged.

        Args:
            other: Another PromptBuilder to merge from

        Returns:
            Self for method chaining
        """
        # Merge sections (other overrides same-named sections)
        for name, section in other.iter_named_sections():
            self._document.upsert(
                PromptSection(
                    name=name,
                    content=section.content,
                    priority=section.priority,
                    enabled=section.enabled,
                    header=section.header,
                )
            )

        # Merge tool hints
        self._tool_hints.update(other._tool_hints)

        # Merge safety rules (deduplicate)
        for rule in other._safety_rules:
            if rule not in self._safety_rules:
                self._safety_rules.append(rule)

        # Merge context
        for ctx in other._context:
            if ctx not in self._context:
                self._context.append(ctx)

        # Take grounding from other if it has custom grounding
        if other._custom_grounding:
            self._custom_grounding = other._custom_grounding
        else:
            # Otherwise prefer "extended" mode
            if other._grounding_mode == "extended":
                self._grounding_mode = "extended"

        return self

    def _get_grounding_rules(self) -> str:
        """Get the appropriate grounding rules.

        Returns:
            Grounding rules text based on mode or custom setting
        """
        # Import here to avoid circular imports
        from victor.agent.prompting.sections import (
            GROUNDING_RULES_MINIMAL,
            GROUNDING_RULES_EXTENDED,
        )

        if self._custom_grounding:
            return self._custom_grounding

        if self._grounding_mode == "extended":
            return GROUNDING_RULES_EXTENDED
        return GROUNDING_RULES_MINIMAL

    def build_document(self) -> PromptDocument:
        """Build the canonical prompt document."""
        document = self._document.clone()

        if self._tool_hints:
            hints_lines = [f"- {hint.tool_name}: {hint.hint}" for hint in self._tool_hints.values()]
            hints_text = "\n".join(hints_lines)
            document.upsert(
                PromptBlock(
                    name="tool_hints",
                    content=hints_text,
                    priority=self.PRIORITY_TOOL_GUIDANCE,
                    header="## Tool Hints",
                    kind="tool_hints",
                )
            )

        document.upsert(
            PromptBlock(
                name="analysis_efficiency",
                content=dedent("""\
                    Before starting any structural or architectural analysis of the codebase:
                    1. Call `analysis_checkpoint(mode="list", topic="")` to see available cached analyses.
                    2. Call `analysis_checkpoint(mode="read", topic="<topic>")` for the relevant topic.
                    3. If status is "current" — use the cached content directly. Skip tool-intensive analysis.
                    4. If status is "stale" — re-analyze only the changed files listed in stale_files, then
                       update: `analysis_checkpoint(mode="write", topic="...", source_files=[...], content="...")`.
                    5. If status is "not_found" — run analysis, then write a checkpoint on completion.

                    Checkpoints live in `.victor/analysis/{slug}-analysis.md` with YAML frontmatter tracking
                    source file mtimes. Staleness is auto-detected via the graph_file_mtime table.
                    Reading a checkpoint costs ~500 tokens vs 8,000–15,000 for a fresh deep analysis.\
                """),
                priority=self.PRIORITY_TOOL_GUIDANCE,
                header="## Analysis Efficiency",
                kind="analysis_efficiency",
            )
        )

        if self._safety_rules:
            rules_lines = [f"- {rule}" for rule in self._safety_rules]
            rules_text = "\n".join(rules_lines)
            document.upsert(
                PromptBlock(
                    name="safety_rules",
                    content=rules_text,
                    priority=self.PRIORITY_SAFETY,
                    header="## Safety Rules",
                    kind="safety_rules",
                )
            )

        if self._context:
            context_text = "\n\n".join(self._context)
            document.upsert(
                PromptBlock(
                    name="context",
                    content=context_text,
                    priority=self.PRIORITY_CONTEXT,
                    header="## Context",
                    kind="context",
                )
            )

        grounding = self._get_grounding_rules()
        if grounding:
            document.upsert(
                PromptBlock(
                    name="grounding",
                    content=grounding,
                    priority=self.PRIORITY_GROUNDING,
                    header="",
                    kind="grounding",
                )
            )

        return document

    def build(self) -> str:
        """Build the final prompt string from the canonical document."""
        return self.build_document().render()

    def add_evolved_section(
        self,
        section_name: str,
        provider: str = "",
        model: str = "",
        task_type: str = "default",
        priority: int = 50,
    ) -> "PromptBuilder":
        """Add an evolvable section with automatic content resolution.

        This method checks for evolved content via OptimizationInjector
        and falls back to static defaults if no evolved version exists.

        Args:
            section_name: Name of the section (e.g., "ASI_TOOL_EFFECTIVENESS_GUIDANCE")
            provider: Provider name for evolution lookup
            model: Model name for evolution lookup
            task_type: Task type for context
            priority: Section priority for ordering

        Returns:
            Self for method chaining
        """
        from victor.agent.evolved_content_resolver import EvolvedContentResolver
        from victor.agent.optimization_injector import OptimizationInjector

        # Get resolver (create if needed)
        if not hasattr(self, "_evolved_resolver"):
            self._evolved_resolver = EvolvedContentResolver(
                optimization_injector=OptimizationInjector()
            )

        # Get fallback text
        fallback = self._get_section_fallback(section_name)

        # Resolve evolved content
        resolved = self._evolved_resolver.resolve_section(
            section_name=section_name,
            provider=provider,
            model=model,
            task_type=task_type,
            fallback_text=fallback,
        )

        # Add to document
        if resolved.text:
            self.add_section(
                name=section_name.lower(),
                content=resolved.text,
                priority=priority,
            )
            logger.debug(
                f"Added evolved section '{section_name}' "
                f"(source={resolved.source}, priority={priority})"
            )

        return self

    def _get_section_fallback(self, section_name: str) -> str:
        """Get static fallback text for a section.

        Args:
            section_name: Section name

        Returns:
            Static fallback text
        """
        from victor.agent.prompt_section_registry import build_fallback_map

        return build_fallback_map([section_name]).get(section_name, "")

    def __repr__(self) -> str:
        """Return a string representation of the builder state."""
        return (
            f"PromptBuilder("
            f"sections={len(self._sections)}, "
            f"tool_hints={len(self._tool_hints)}, "
            f"safety_rules={len(self._safety_rules)}, "
            f"context={len(self._context)}, "
            f"grounding='{self._grounding_mode}'"
            f")"
        )


# Convenience factory functions


def create_coding_prompt_builder() -> PromptBuilder:
    """Create a PromptBuilder pre-configured for coding tasks.

    Returns:
        A PromptBuilder with coding-specific defaults
    """
    from victor.agent.prompting.sections import (
        CODING_IDENTITY,
        CODING_GUIDELINES,
        CODING_TOOL_USAGE,
    )

    return (
        PromptBuilder()
        .add_section("identity", CODING_IDENTITY, priority=PromptBuilder.PRIORITY_IDENTITY)
        .add_section("guidelines", CODING_GUIDELINES, priority=PromptBuilder.PRIORITY_GUIDELINES)
        .add_section(
            "tool_usage",
            CODING_TOOL_USAGE,
            priority=PromptBuilder.PRIORITY_TOOL_GUIDANCE,
        )
    )


def create_devops_prompt_builder() -> PromptBuilder:
    """Create a PromptBuilder pre-configured for DevOps tasks.

    Returns:
        A PromptBuilder with DevOps-specific defaults
    """
    from victor.agent.prompting.sections import (
        DEVOPS_IDENTITY,
        DEVOPS_SECURITY_CHECKLIST,
        DEVOPS_COMMON_PITFALLS,
    )

    return (
        PromptBuilder()
        .add_section("identity", DEVOPS_IDENTITY, priority=PromptBuilder.PRIORITY_IDENTITY)
        .add_section(
            "security",
            DEVOPS_SECURITY_CHECKLIST,
            priority=PromptBuilder.PRIORITY_SAFETY,
        )
        .add_section(
            "pitfalls",
            DEVOPS_COMMON_PITFALLS,
            priority=PromptBuilder.PRIORITY_GUIDELINES,
        )
    )


def create_research_prompt_builder() -> PromptBuilder:
    """Create a PromptBuilder pre-configured for research tasks.

    Returns:
        A PromptBuilder with research-specific defaults
    """
    from victor.agent.prompting.sections import (
        RESEARCH_IDENTITY,
        RESEARCH_QUALITY_CHECKLIST,
        RESEARCH_SOURCE_HIERARCHY,
    )

    return (
        PromptBuilder()
        .add_section("identity", RESEARCH_IDENTITY, priority=PromptBuilder.PRIORITY_IDENTITY)
        .add_section(
            "quality",
            RESEARCH_QUALITY_CHECKLIST,
            priority=PromptBuilder.PRIORITY_GUIDELINES,
        )
        .add_section(
            "sources",
            RESEARCH_SOURCE_HIERARCHY,
            priority=PromptBuilder.PRIORITY_GUIDELINES + 5,
        )
    )


def create_data_analysis_prompt_builder() -> PromptBuilder:
    """Create a PromptBuilder pre-configured for data analysis tasks.

    Returns:
        A PromptBuilder with data analysis-specific defaults
    """
    from victor.agent.prompting.sections import (
        DATA_ANALYSIS_IDENTITY,
        DATA_ANALYSIS_LIBRARIES,
        DATA_ANALYSIS_OPERATIONS,
    )

    return (
        PromptBuilder()
        .add_section("identity", DATA_ANALYSIS_IDENTITY, priority=PromptBuilder.PRIORITY_IDENTITY)
        .add_section(
            "libraries",
            DATA_ANALYSIS_LIBRARIES,
            priority=PromptBuilder.PRIORITY_CAPABILITIES,
        )
        .add_section(
            "operations",
            DATA_ANALYSIS_OPERATIONS,
            priority=PromptBuilder.PRIORITY_GUIDELINES,
        )
    )


# ---------------------------------------------------------------------------
# Dynamic prompt builder with workspace context discovery
# ---------------------------------------------------------------------------


@dataclass
class ContextFile:
    """An instruction file discovered in the workspace.

    Attributes:
        path: Absolute or relative path to the file.
        content: Raw text content of the file.
        scope: One of "workspace", "project", or "user".
    """

    path: str
    content: str
    scope: str  # "workspace", "project", "user"


@dataclass
class ProjectContext:
    """Snapshot of the current workspace context.

    Attributes:
        cwd: Current working directory.
        current_date: ISO-formatted date string.
        git_status: Short git status output, if available.
        git_diff_summary: Condensed diff stat, if available.
        instruction_files: Discovered instruction/context files.
        is_git_repo: Whether the cwd is inside a git repository.
        branch_name: Active git branch name, if available.
    """

    cwd: Path
    current_date: str
    git_status: Optional[str] = None
    git_diff_summary: Optional[str] = None
    instruction_files: List[ContextFile] = field(default_factory=list)
    is_git_repo: bool = False
    branch_name: Optional[str] = None


@dataclass
class PromptBudget:
    """Character budgets for dynamic prompt sections.

    Attributes:
        max_per_file_chars: Max chars kept per instruction file.
        max_total_instruction_chars: Aggregate cap across files.
        max_git_status_chars: Max chars for git status output.
        max_git_diff_chars: Max chars for git diff summary.
    """

    max_per_file_chars: int = 4000
    max_total_instruction_chars: int = 12000
    max_git_status_chars: int = 2000
    max_git_diff_chars: int = 3000


def _truncate(text: str, limit: int, label: str = "content") -> str:
    """Truncate *text* to *limit* chars with a notice."""
    if len(text) <= limit:
        return text
    truncated = text[:limit]
    notice = f"\n... [{label} truncated to {limit} chars]"
    return truncated + notice


class WorkspaceContextBuilder:
    """Builder for constructing system prompts with dynamic context.

    Sections are assembled in a deterministic order so that the
    most important static instructions appear first and
    budget-managed dynamic content appears after a clearly-marked
    boundary.

    Example:
        builder = (
            WorkspaceContextBuilder()
            .with_base_prompt("You are a coding assistant.")
            .with_os_info("Darwin", "25.2.0")
            .with_model_info("claude-opus-4-6")
            .with_project_context(context)
            .with_tools(["filesystem", "git"])
        )
        prompt = builder.build()
    """

    def __init__(self, budget: Optional[PromptBudget] = None) -> None:
        self._budget = budget or PromptBudget()
        self._base_prompt: Optional[str] = None
        self._output_style_name: Optional[str] = None
        self._output_style_prompt: Optional[str] = None
        self._os_name: Optional[str] = None
        self._os_version: Optional[str] = None
        self._model_name: Optional[str] = None
        self._context: Optional[ProjectContext] = None
        self._tool_names: List[str] = []
        self._extra_sections: List[str] = []

    # -- fluent setters ------------------------------------------------

    def with_base_prompt(self, prompt: str) -> "WorkspaceContextBuilder":
        """Set the static base/vertical prompt."""
        self._base_prompt = prompt
        return self

    def with_output_style(self, name: str, prompt: str) -> "WorkspaceContextBuilder":
        """Set an output style directive."""
        self._output_style_name = name
        self._output_style_prompt = prompt
        return self

    def with_os_info(self, os_name: str, os_version: str) -> "WorkspaceContextBuilder":
        """Set operating system metadata."""
        self._os_name = os_name
        self._os_version = os_version
        return self

    def with_model_info(self, model_name: str) -> "WorkspaceContextBuilder":
        """Set the model identifier shown in the prompt."""
        self._model_name = model_name
        return self

    def with_project_context(self, context: ProjectContext) -> "WorkspaceContextBuilder":
        """Attach a discovered project context snapshot."""
        self._context = context
        return self

    def with_tools(self, tool_names: List[str]) -> "WorkspaceContextBuilder":
        """Declare which tools are available to the agent."""
        self._tool_names = list(tool_names)
        return self

    def append_section(self, section: str) -> "WorkspaceContextBuilder":
        """Append a free-form section after all standard parts."""
        self._extra_sections.append(section)
        return self

    # -- assembly ------------------------------------------------------

    def build(self) -> str:
        """Assemble all sections into the final system prompt.

        Assembly order:
            1. Output style section (if provided)
            2. Base prompt
            3. System capabilities (tools available)
            4. Environment section
            5. Dynamic boundary marker
            6. Instruction files (budget-truncated)
            7. Git status context (budget-truncated)
            8. Appended sections
        """
        parts: List[str] = []

        # 1. Output style
        if self._output_style_name and self._output_style_prompt:
            parts.append(
                f"# Output Style: {self._output_style_name}\n" f"{self._output_style_prompt}"
            )

        # 2. Base prompt
        if self._base_prompt:
            parts.append(self._base_prompt)

        # 3. System capabilities
        if self._tool_names:
            tool_list = ", ".join(sorted(self._tool_names))
            parts.append("# System Capabilities\n" f"Tools available: {tool_list}")

        # 4. Environment
        env_lines = self._build_environment_section()
        if env_lines:
            parts.append(env_lines)

        # 5. Dynamic boundary
        parts.append("\n# --- DYNAMIC CONTEXT BELOW ---\n")

        # 6. Instruction files
        instr = self._build_instruction_section()
        if instr:
            parts.append(instr)

        # 7. Git status context
        git_section = self._build_git_context_section()
        if git_section:
            parts.append(git_section)

        # 8. Extra appended sections
        for section in self._extra_sections:
            parts.append(section)

        return "\n\n".join(parts)

    # -- private helpers -----------------------------------------------

    def _build_environment_section(self) -> str:
        """Render the ``# Environment`` section."""
        lines: List[str] = ["# Environment"]

        if self._model_name:
            lines.append(f"Model: {self._model_name}")

        if self._context:
            lines.append(f"Working directory: {self._context.cwd}")
            lines.append(f"Current date: {self._context.current_date}")

        if self._os_name and self._os_version:
            lines.append(f"Platform: {self._os_name} " f"{self._os_version}")

        if self._context:
            if self._context.is_git_repo:
                lines.append("Git repository: yes")
            if self._context.branch_name:
                lines.append(f"Git branch: " f"{self._context.branch_name}")

        if len(lines) <= 1:
            return ""
        return "\n".join(lines)

    def _build_instruction_section(self) -> str:
        """Render budget-truncated instruction files."""
        if not self._context or not self._context.instruction_files:
            return ""

        budget = self._budget
        parts: List[str] = []
        total_chars = 0

        for ctx_file in self._context.instruction_files:
            remaining = budget.max_total_instruction_chars - total_chars
            if remaining <= 0:
                parts.append("... [instruction budget exhausted, " "remaining files skipped]")
                break

            per_file = min(budget.max_per_file_chars, remaining)
            content = _truncate(ctx_file.content, per_file, ctx_file.path)
            total_chars += len(content)

            header = f"## {ctx_file.path} " f"(scope: {ctx_file.scope})"
            parts.append(f"{header}\n{content}")

        if not parts:
            return ""
        return "# Project Instructions\n\n" + "\n\n".join(parts)

    def _build_git_context_section(self) -> str:
        """Render budget-truncated git status / diff."""
        if not self._context:
            return ""

        budget = self._budget
        parts: List[str] = []

        if self._context.git_status:
            status = _truncate(
                self._context.git_status,
                budget.max_git_status_chars,
                "git status",
            )
            parts.append(f"## Git Status\n```\n{status}\n```")

        if self._context.git_diff_summary:
            diff = _truncate(
                self._context.git_diff_summary,
                budget.max_git_diff_chars,
                "git diff",
            )
            parts.append(f"## Git Diff Summary\n```\n{diff}\n```")

        if not parts:
            return ""
        return "# Git Context\n\n" + "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Project context discovery
# ---------------------------------------------------------------------------


def _classify_scope(file_dir: Path, workspace_dir: Path) -> PromptScope:
    """Determine the scope label for a discovered file.

    Args:
        file_dir: Directory the file was found in.
        workspace_dir: The original working directory.

    Returns:
        One of "workspace", "project", or "user".
    """
    if file_dir == workspace_dir:
        return "workspace"
    if (file_dir / ".git").exists():
        return "project"
    return "user"


async def _run_git(cmd: List[str], cwd: Path) -> Optional[str]:
    """Execute a git command asynchronously.

    Args:
        cmd: Command and arguments list.
        cwd: Working directory for the subprocess.

    Returns:
        Stripped stdout on success, or None on any error.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        if proc.returncode != 0:
            return None
        return stdout.decode("utf-8", errors="replace")
    except (OSError, ValueError):
        logger.debug("Git command failed: %s", " ".join(cmd))
        return None


class ProjectContextDiscovery:
    """Discovers workspace context for prompt construction.

    All methods are static or class-level so the class acts as
    a namespace; no instance state is required.

    Example:
        ctx = await ProjectContextDiscovery.discover(
            Path.cwd()
        )
    """

    _INSTRUCTION_FILENAMES: List[str] = list(DEFAULT_INSTRUCTION_FILENAMES)

    @staticmethod
    async def discover(
        cwd: Path,
        current_date: Optional[str] = None,
    ) -> ProjectContext:
        """Build a full ProjectContext for *cwd*.

        Args:
            cwd: Working directory to inspect.
            current_date: Override for the current date
                string. Defaults to today in ISO format.

        Returns:
            A populated ProjectContext dataclass.
        """
        from datetime import date as _date

        if current_date is None:
            current_date = _date.today().isoformat()

        is_git = (cwd / ".git").exists()

        if is_git:
            branch, status, diff_summary = await asyncio.gather(
                ProjectContextDiscovery._get_branch_name(cwd),
                ProjectContextDiscovery._get_git_status(cwd),
                ProjectContextDiscovery._get_git_diff_summary(cwd),
            )
        else:
            branch = None
            status = None
            diff_summary = None

        instruction_files = ProjectContextDiscovery._discover_instruction_files(cwd)

        return ProjectContext(
            cwd=cwd,
            current_date=current_date,
            git_status=status,
            git_diff_summary=diff_summary,
            instruction_files=instruction_files,
            is_git_repo=is_git,
            branch_name=branch,
        )

    @staticmethod
    def _discover_instruction_files(
        cwd: Path,
    ) -> List[ContextFile]:
        """Walk ancestor chain looking for instruction files.

        Searches from *cwd* upward to the filesystem root.
        Each discovered file is labelled with a scope:

        - ``workspace`` -- lives in *cwd* itself
        - ``project`` -- lives in a parent containing ``.git``
        - ``user`` -- everything else (e.g. home directory)

        Files with identical content are deduplicated; only the
        first occurrence is kept.

        Returns:
            Ordered list of ContextFile instances.
        """
        results: List[ContextFile] = []
        for item in discover_shared_instruction_files(
            cwd,
            filenames=ProjectContextDiscovery._INSTRUCTION_FILENAMES,
        ):
            results.append(
                ContextFile(
                    path=str(item.path),
                    content=item.content,
                    scope=item.scope,
                )
            )
        return results

    @staticmethod
    async def _get_git_status(
        cwd: Path,
    ) -> Optional[str]:
        """Run ``git status --short --branch`` async.

        Returns:
            Status output string, or None on failure.
        """
        return await _run_git(
            [
                "git",
                "--no-optional-locks",
                "status",
                "--short",
                "--branch",
            ],
            cwd,
        )

    @staticmethod
    async def _get_git_diff_summary(
        cwd: Path,
    ) -> Optional[str]:
        """Run ``git diff --stat`` async.

        Returns:
            Diff stat output string, or None on failure.
        """
        return await _run_git(
            ["git", "diff", "--stat"],
            cwd,
        )

    @staticmethod
    async def _get_branch_name(
        cwd: Path,
    ) -> Optional[str]:
        """Get the current branch via ``git rev-parse``.

        Returns:
            Branch name string, or None on failure.
        """
        result = await _run_git(
            [
                "git",
                "rev-parse",
                "--abbrev-ref",
                "HEAD",
            ],
            cwd,
        )
        return result.strip() if result else None


__all__ = [
    "PromptSection",
    "ToolHint",
    "PromptBuilder",
    "ContextFile",
    "ProjectContext",
    "PromptBudget",
    "WorkspaceContextBuilder",
    "ProjectContextDiscovery",
    "create_coding_prompt_builder",
    "create_devops_prompt_builder",
    "create_research_prompt_builder",
    "create_data_analysis_prompt_builder",
]

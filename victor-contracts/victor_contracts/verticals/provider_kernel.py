"""Provider factories shared by every vertical's ``protocols`` module.

Each vertical's ToolProvider / ToolSelectionStrategy / SafetyProvider /
PromptProvider / WorkflowProvider is the same class shape returning its own
literals: a tool list, a stage map with a fallback, a pattern table, prompt
sections, a workflow dict. These factories build those classes once; verticals
keep their tables and assign name-preserving results::

    from victor_contracts.verticals import make_tool_provider

    DevOpsToolProvider = make_tool_provider(
        "DevOpsToolProvider",
        DEVOPS_TOOLS,
        "Tool provider for victor-devops.",
    )

The built classes duck-type the same interfaces the hand-written ones did
(the contracts protocols are nominal parents only — no runtime conformance
check exists, by design).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

# Direct submodule imports (never the package __init__) to avoid cycles.
from victor_contracts.verticals.protocols.prompts import PromptProvider
from victor_contracts.verticals.protocols.safety import SafetyProvider
from victor_contracts.verticals.protocols.tools import (
    ToolProvider,
    ToolSelectionStrategy,
)
from victor_contracts.verticals.protocols.workflows import WorkflowProvider


def make_tool_provider(name: str, tools: Sequence[str], doc: str = "") -> type:
    """A ToolProvider whose ``get_tools`` returns a copy of ``tools``."""
    tool_list = list(tools)

    def get_tools(self) -> List[str]:
        return list(tool_list)

    return type(
        name,
        (ToolProvider,),
        {"__doc__": doc or f"Tool provider ({name}).", "get_tools": get_tools},
    )


def make_stage_tool_strategy(
    name: str,
    stage_tools: Dict[str, List[str]],
    fallback: List[str],
    doc: str = "",
) -> type:
    """A ToolSelectionStrategy over a stage->tools map with a shared fallback.

    Per-vertical fallbacks are data: devops falls back to ``["read", "shell"]``,
    coding to ``[]``, rag to its search pair.
    """
    stage_map = {stage: list(tools) for stage, tools in stage_tools.items()}
    default = list(fallback)

    def get_tools_for_stage(self, stage: str, task_type: str) -> List[str]:
        return stage_map.get(stage, list(default))

    return type(
        name,
        (ToolSelectionStrategy,),
        {
            "__doc__": doc or f"Stage-aware tool selection ({name}).",
            "get_tools_for_stage": get_tools_for_stage,
        },
    )


def make_pattern_safety_provider(
    name: str,
    dangerous_patterns: Sequence[str],
    tool_restrictions: Dict[str, List[str]],
    *,
    file_patterns: Sequence[str] = (),
    include_self: bool = False,
    extensions: Optional[List[Any]] = None,
    doc: str = "",
) -> type:
    """A pattern-table SafetyProvider.

    ``include_self`` preserves the historical fork where devops's
    ``get_extensions`` returned ``[self]`` and the others returned ``[]``;
    ``extensions`` (when given) is returned as-is for providers that surface
    extension instances instead.
    """

    def get_extensions(self) -> List[Any]:
        if extensions is not None:
            return list(extensions)
        return [self] if include_self else []

    def get_bash_patterns(self) -> List[Any]:
        return list(dangerous_patterns)

    def get_file_patterns(self) -> List[Any]:
        return list(file_patterns)

    def get_tool_restrictions(self) -> Dict[str, List[str]]:
        return {tool: list(args) for tool, args in tool_restrictions.items()}

    return type(
        name,
        (SafetyProvider,),
        {
            "__doc__": doc or f"Pattern-backed safety provider ({name}).",
            "get_extensions": get_extensions,
            "get_bash_patterns": get_bash_patterns,
            "get_file_patterns": get_file_patterns,
            "get_tool_restrictions": get_tool_restrictions,
        },
    )


def make_prompt_provider(
    name: str,
    sections: Dict[str, str],
    task_type_hints: Dict[str, Any],
    doc: str = "",
) -> type:
    """A PromptProvider over static sections and task-type hints."""

    def get_system_prompt_sections(self) -> Dict[str, str]:
        return dict(sections)

    def get_task_type_hints(self) -> Dict[str, Any]:
        return dict(task_type_hints)

    def get_prompt_contributors(self) -> List[Any]:
        return []

    return type(
        name,
        (PromptProvider,),
        {
            "__doc__": doc or f"Prompt provider ({name}).",
            "get_system_prompt_sections": get_system_prompt_sections,
            "get_task_type_hints": get_task_type_hints,
            "get_prompt_contributors": get_prompt_contributors,
        },
    )


def make_workflow_provider(name: str, workflows: Dict[str, Any], doc: str = "") -> type:
    """A WorkflowProvider over a static workflow spec dict."""
    spec = dict(workflows)

    def get_workflows(self) -> Dict[str, Any]:
        return dict(spec)

    def get_workflow(self, workflow_name: str) -> Optional[Any]:
        return spec.get(workflow_name)

    def list_workflows(self) -> List[str]:
        return list(spec.keys())

    return type(
        name,
        (WorkflowProvider,),
        {
            "__doc__": doc or f"Workflow provider ({name}).",
            "get_workflows": get_workflows,
            "get_workflow": get_workflow,
            "list_workflows": list_workflows,
        },
    )

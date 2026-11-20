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

"""Subagent-spawn tool — the Task-tool pattern (FOU-A1, foreground mode).

``spawn_subagent(prompt, tools?, model?, max_depth?)`` delegates a bounded,
single-task run to a child agent and returns the child's final result to the
calling loop as a structured dict: ``success`` / ``content`` / ``tool_calls_used``
/ ``duration_seconds`` / ``error`` / ``child_session_id``.

Substrate: the team coordinator's member-execution path
(``victor.agent.subagents.SubAgentOrchestrator``), chosen over raw
``AgentFactory.create()`` because it already provides the lifecycle a spawn
needs — constrained per-member tool registries, per-role tool budgets and
context sizing, member session-id derivation, timeout wrapping, and the
``SubAgentResult`` hand-off shape. The tool never touches that module
directly: the agent factory binds a *runner callable* into the tool
executor's context (``spawn_subagent_runner``, see
``victor.agent.factory.tool_builders.ToolBuildersMixin.build_spawn_subagent_runner``
and the ``ComponentAssembler.assemble_tools`` wiring), keeping this module
free of runtime-layer imports and the session's parent context explicit.

Depth-1 recursion guard, enforced mechanically:
- ``max_depth`` beyond 1 is REFUSED (phase 1 ships foreground depth 1 only;
  deeper trees are a follow-up increment);
- ``spawn_subagent`` is stripped from any requested ``tools`` list before it
  reaches the substrate (``sanitize_subagent_tool_list``), so a recursing
  child tool set cannot be requested, let alone registered;
- the bound runner always constructs the member with
  ``can_spawn_subagents=False`` and no role authorizes the spawn tool.

Budget: the spawn itself is one call against the parent's live tool-call
budget (ToolPipeline budget enforcement, unchanged). The child's own tool
budget comes from ``ToolSettings.spawn_subagent_tool_budget`` (None = the
substrate's role default). ``task_tool_config.yaml`` was verified to carry
only exploration-iteration budgets and stage tool lists — no per-tool
budget mechanism — so the live settings path is used.

Permissions: workspace-write tier (``DEFAULT_TOOL_PERMISSIONS``), so the
tool inherits the parent's PermissionMode through the normal gate —
read-only sessions deny, workspace-write and above allow.

Escalation: phase 1 default role is RESEARCHER (read-only tool set), so a
foreground spawn cannot silently widen write access beyond what the caller
explicitly passes in ``tools``; HITL escalation inside a subagent surfaces
through the parent's approval path (member approval context).
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, Iterable, List, Optional

from victor.tools.base import AccessMode, DangerLevel, ExecutionCategory, Priority
from victor.tools.decorators import resolve_tool_name, tool

logger = logging.getLogger(__name__)

SPAWN_SUBAGENT_TOOL_NAME = "spawn_subagent"

# Phase 1 ships exactly one level of delegation.
MAX_SPAWN_DEPTH = 1

RunnerResult = Dict[str, Any]
SpawnRunner = Any  # async callable(prompt, tools, model, tool_budget) -> RunnerResult|None


def sanitize_subagent_tool_list(tools: Optional[List[str]]) -> Optional[List[str]]:
    """Strip the spawn tool (alias-resolved) from a requested child tool set.

    This is the mechanical depth-1 guard at the tool boundary: whatever the
    model asks for, the substrate never receives a tool list that could
    re-enter ``spawn_subagent``.
    """
    if tools is None:
        return None
    return [t for t in tools if resolve_tool_name(t) != SPAWN_SUBAGENT_TOOL_NAME]


def compute_child_tool_set(
    requested: Optional[List[str]],
    role_grants: Iterable[str],
    parent_admitted: Iterable[str],
) -> List[str]:
    """Final child tool set: requested ∩ role grants ∩ parent-admitted.

    Finding (delegation grant widening): a restrict-list must never REPLACE
    the role's grant list, and delegation must never widen the session's
    enabled tool set. Every candidate name is alias-resolved to its canonical
    form first, then must appear in BOTH the effective role grants and the
    parent's admitted capabilities to survive. Requested order is preserved.

    Args:
        requested: Alias-resolved-sanitized restrict-list from the caller
            (``None`` = no explicit list; role defaults then apply, still
            bounded by the parent's admitted set).
        role_grants: The effective role-provider grant list for the child role.
        parent_admitted: Canonical names the parent session actually admits.

    Returns:
        The final allowed_tools list for the child (never contains the spawn
        tool — depth 1).
    """
    role_set = {resolve_tool_name(t) for t in role_grants}
    parent_set = {resolve_tool_name(t) for t in parent_admitted}

    final: List[str] = []
    seen: set[str] = set()

    def _admit(name: str) -> None:
        canonical = resolve_tool_name(name)
        if canonical == SPAWN_SUBAGENT_TOOL_NAME or canonical in seen:
            return
        if canonical not in role_set or canonical not in parent_set:
            return
        seen.add(canonical)
        final.append(canonical)

    if requested is None:
        for name in role_grants:
            _admit(name)
    else:
        for name in requested:
            _admit(name)
    return final


def mint_child_session_id(parent_session_id: Optional[str]) -> str:
    """Mint a UNIQUE child session id for one spawn invocation.

    Finding (spawn session ids): every foreground spawn used to share one
    member id (``spawn_subagent``), so the substrate's session derivation
    collapsed to ``{parent}-spawn_subagent`` for concurrent children — a
    shared upstream session/cache key — and the tool's ``child_session_id``
    read an optional input that was never supplied. The minted id is passed
    as the explicit ``child_session_id``, which is precedence 1 in
    ``SubAgentConfig.resolve_member_session_id``: the SAME value then keys
    the provider session binding, the result metadata and the audit record.
    """
    if not parent_session_id or not str(parent_session_id).strip():
        from victor.core.context import get_session_id

        parent_session_id = get_session_id() or "subagent"
    return f"{parent_session_id}-spawn-{uuid.uuid4().hex[:12]}"


def _refusal(error: str) -> Dict[str, Any]:
    logger.info("spawn_subagent refused: %s", error)
    return {"success": False, "error": error}


def _map_subagent_result(
    raw: RunnerResult,
    tool_budget: Optional[int],
) -> Dict[str, Any]:
    """Map ``SubAgentResult.to_dict()`` into the tool's structured result."""
    details = raw.get("details") or {}
    if not isinstance(details, dict):
        details = {}
    return {
        "success": bool(raw.get("success")),
        "content": raw.get("summary") or "",
        "tool_calls_used": int(raw.get("tool_calls_used") or 0),
        "duration_seconds": float(raw.get("duration_seconds") or 0.0),
        "error": raw.get("error"),
        "child_session_id": details.get("child_session_id"),
        "role": details.get("role"),
        "tool_budget": tool_budget,
    }


@tool(
    name=SPAWN_SUBAGENT_TOOL_NAME,
    category="agent",
    priority=Priority.LOW,
    access_mode=AccessMode.WRITE,  # workspace-write tier per FOU-A1
    danger_level=DangerLevel.MEDIUM,
    execution_category=ExecutionCategory.WRITE,
    keywords=["subagent", "delegate", "spawn", "hand off", "parallel task"],
    task_types=["action"],
    # Foreground spawn runs a full child loop; match the substrate's default
    # member timeout instead of the pipeline's 60s tool default.
    timeout=330.0,
)
async def spawn_subagent(
    prompt: str,
    tools: Optional[List[str]] = None,
    model: Optional[str] = None,
    max_depth: int = MAX_SPAWN_DEPTH,
    _exec_ctx: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Delegate a bounded single task to a child agent and await its result.

    Spawns a constrained sub-agent (default read-only researcher role) that
    runs the prompt to completion with its own tool budget and isolated
    context, then returns the child's final result. The child cannot spawn
    further subagents (depth 1), cannot exceed this session's permission
    mode, and is budget-capped per the session configuration.

    Args:
        prompt: The self-contained task for the sub-agent. It sees only this
                prompt and its own tool results — pass file paths and
                context explicitly.
        tools: Optional restrict-list of tool names for the child (default:
               the role's read-only set). ``spawn_subagent`` is always
               stripped — children cannot recurse.
        model: Optional model override for the child (resolved through the
               parent's provider; failures fall back to the parent model).
        max_depth: Delegation depth. Only 1 is supported; deeper values are
                   refused.

    Returns:
        Dict with success flag, the child's final content, tool call count,
        duration, and the child's session id; on refusal/failure an
        actionable error.
    """
    if not prompt or not prompt.strip():
        return _refusal("spawn_subagent requires a non-empty prompt describing the task")

    if max_depth != MAX_SPAWN_DEPTH:
        return _refusal(
            f"max_depth {max_depth} exceeds the supported delegation depth "
            f"({MAX_SPAWN_DEPTH}); a subagent cannot spawn subagents"
        )

    context = _exec_ctx or {}
    runner: SpawnRunner = context.get("spawn_subagent_runner")
    if runner is None:
        return _refusal(
            "spawn_subagent is not wired in this session (no substrate runner "
            "bound by the agent factory)"
        )

    tool_budget = context.get("spawn_subagent_budget")
    sanitized_tools = sanitize_subagent_tool_list(tools)

    try:
        raw = await runner(
            prompt=prompt,
            tools=sanitized_tools,
            model=model,
            tool_budget=tool_budget,
        )
    except Exception as exc:
        logger.warning("spawn_subagent substrate failed: %s", type(exc).__name__)
        return {
            "success": False,
            "error": f"subagent execution failed: {exc}",
            "content": "",
            "tool_calls_used": 0,
            "duration_seconds": 0.0,
            "tool_budget": tool_budget,
        }

    if raw is None:
        return _refusal(
            "subagent orchestration is disabled in this session "
            "(subagent_orchestration_enabled=false)"
        )

    return _map_subagent_result(raw, tool_budget)


__all__ = [
    "spawn_subagent",
    "sanitize_subagent_tool_list",
    "compute_child_tool_set",
    "mint_child_session_id",
    "SPAWN_SUBAGENT_TOOL_NAME",
    "MAX_SPAWN_DEPTH",
]

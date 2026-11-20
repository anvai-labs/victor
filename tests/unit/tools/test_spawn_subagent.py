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

"""FOU-A1 unit suite: the ``spawn_subagent`` tool (foreground mode).

Covers the increment's gates: the structured foreground result returned to
the calling loop (success/content/tool_calls_used/duration), the depth-1
recursion guard enforced MECHANICALLY (a subagent's tool list can never
contain ``spawn_subagent``; ``max_depth`` beyond 1 is refused, not clamped),
the workspace-write permission gating in both directions, per-spawn tool
budget forwarding, and the substrate invariants (role tool defaults and
constrained member registries exclude the spawn tool; ``can_spawn_subagents``
defaults False).

The subagent execution substrate (``SubAgentOrchestrator``, the live member
execution path) is injected as a runner callable via the tool executor
context — tests stub it, so no provider, database, or network is touched.
The WIRING itself (the real AgentOrchestrator construction path binding the
runner) is pinned in ``tests/unit/agent/test_spawn_subagent_wiring.py``.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.core.shared_types import SubAgentRole
from victor.security.permissions import (
    DEFAULT_TOOL_PERMISSIONS,
    PermissionMode,
    PermissionPolicy,
)
from victor.tools.enums import AccessMode
from victor.tools.spawn_subagent import (
    SPAWN_SUBAGENT_TOOL_NAME,
    compute_child_tool_set,
    mint_child_session_id,
    sanitize_subagent_tool_list,
    spawn_subagent,
)
from victor.agent.subagents.base import SubAgent, SubAgentConfig

# A SubAgentResult.to_dict() shape, as the real substrate returns it.
_SUBAGENT_RESULT_DICT = {
    "success": True,
    "summary": "Found 3 auth middleware call sites in victor/agent.",
    "details": {"role": "researcher", "child_session_id": "sess-abc-spawn-1"},
    "tool_calls_used": 6,
    "context_size": 12000,
    "duration_seconds": 4.2,
    "error": None,
}


@pytest.fixture
def runner() -> AsyncMock:
    """A stub substrate runner bound into the executor context."""
    stub = AsyncMock(return_value=dict(_SUBAGENT_RESULT_DICT))
    return stub


@pytest.fixture
def exec_ctx(runner: AsyncMock) -> dict:
    """Executor context the way the factory wiring provides it."""
    return {
        "spawn_subagent_runner": runner,
        "spawn_subagent_budget": None,
    }


# =============================================================================
# Foreground result shape
# =============================================================================


async def test_spawn_returns_structured_subagent_result(exec_ctx, runner):
    """Foreground spawn awaits the substrate and maps SubAgentResult into
    the documented structured result for the calling loop."""
    result = await spawn_subagent(
        prompt="Find all auth middleware call sites",
        _exec_ctx=exec_ctx,
    )

    runner.assert_awaited_once()
    assert result["success"] is True
    assert result["content"] == _SUBAGENT_RESULT_DICT["summary"]
    assert result["tool_calls_used"] == 6
    assert result["duration_seconds"] == 4.2
    assert result["error"] is None
    assert result["child_session_id"] == "sess-abc-spawn-1"


async def test_spawn_forwards_prompt_tools_model_budget(exec_ctx, runner):
    """The runner receives the documented signature: prompt, sanitized tools,
    model, and the per-spawn tool budget from the executor context."""
    await spawn_subagent(
        prompt="Summarize the provider adapters",
        tools=["read", "grep"],
        model="qwen3:32b",
        _exec_ctx={**exec_ctx, "spawn_subagent_budget": 12},
    )

    kwargs = runner.await_args.kwargs
    assert kwargs["prompt"] == "Summarize the provider adapters"
    assert kwargs["tools"] == ["read", "grep"]
    assert kwargs["model"] == "qwen3:32b"
    assert kwargs["tool_budget"] == 12


async def test_spawn_failure_result_carries_error(exec_ctx):
    """A failed subagent run maps to success=False with the substrate error,
    but still returns the structured shape (never raises to the loop)."""
    failed = {
        **_SUBAGENT_RESULT_DICT,
        "success": False,
        "summary": "",
        "error": "Timeout after 300s",
    }
    exec_ctx = {
        **exec_ctx,
        "spawn_subagent_runner": AsyncMock(return_value=failed),
    }

    result = await spawn_subagent(prompt="doomed task", _exec_ctx=exec_ctx)

    assert result["success"] is False
    assert result["error"] == "Timeout after 300s"
    assert "content" in result and "tool_calls_used" in result


async def test_runner_exception_surfaced_as_structured_failure(exec_ctx):
    """A substrate crash is a structured failure, never a raise into the loop."""
    exec_ctx = {
        **exec_ctx,
        "spawn_subagent_runner": AsyncMock(side_effect=RuntimeError("substrate boom")),
    }

    result = await spawn_subagent(prompt="task", _exec_ctx=exec_ctx)

    assert result["success"] is False
    assert "substrate boom" in result["error"]


# =============================================================================
# Dormancy guard: unwired session refuses (FEP-0041 lesson)
# =============================================================================


async def test_spawn_without_runner_refuses():
    """No runner bound in the executor context -> structured refusal."""
    result = await spawn_subagent(prompt="task", _exec_ctx={})

    assert result["success"] is False
    assert "spawn_subagent" in result["error"]
    assert "runner" in result["error"] or "wired" in result["error"]


async def test_spawn_without_context_refuses():
    """Even a missing _exec_ctx (bare execution) refuses rather than raise."""
    result = await spawn_subagent(prompt="task")

    assert result["success"] is False
    assert "error" in result


async def test_spawn_with_disabled_orchestration_refuses(exec_ctx):
    """The real runner returns None when subagent orchestration is disabled;
    the tool must surface that as a refusal, not a fake success."""
    exec_ctx = {**exec_ctx, "spawn_subagent_runner": AsyncMock(return_value=None)}

    result = await spawn_subagent(prompt="task", _exec_ctx=exec_ctx)

    assert result["success"] is False
    assert "orchestration" in result["error"].lower()


# =============================================================================
# Depth-1 recursion guard (enforced mechanically, not documented-only)
# =============================================================================


async def test_max_depth_beyond_one_is_refused(exec_ctx, runner):
    """Phase 1 supports exactly depth 1; a deeper request is refused."""
    result = await spawn_subagent(prompt="task", max_depth=2, _exec_ctx=exec_ctx)

    assert result["success"] is False
    assert "depth" in result["error"].lower()
    runner.assert_not_awaited()


async def test_requested_tool_list_cannot_include_spawn_subagent(exec_ctx, runner):
    """Even if the model asks for a recursing subagent, the tool list handed
    to the substrate has the spawn tool stripped — depth 1 is structural."""
    await spawn_subagent(
        prompt="task",
        tools=["read", SPAWN_SUBAGENT_TOOL_NAME, "grep"],
        _exec_ctx=exec_ctx,
    )

    assert runner.await_args.kwargs["tools"] == ["read", "grep"]


def test_sanitize_strips_spawn_tool_and_aliases():
    """The sanitizer strips the canonical name (alias-resolved), preserving
    order and dedup-free behavior for the remaining tools."""
    assert sanitize_subagent_tool_list(["read", "spawn_subagent"]) == ["read"]
    assert sanitize_subagent_tool_list(["read", "grep"]) == ["read", "grep"]
    assert sanitize_subagent_tool_list(None) is None
    assert sanitize_subagent_tool_list([SPAWN_SUBAGENT_TOOL_NAME]) == []


# =============================================================================
# Substrate invariants: a subagent cannot spawn subagents
# =============================================================================


def test_every_role_tool_default_excludes_spawn_subagent():
    """No role's default tool set authorizes recursion."""
    from victor.agent.subagents.protocols import get_role_tool_provider

    provider = get_role_tool_provider()
    for role in SubAgentRole:
        tools = provider.get_tools_for_role(role.value, None)
        assert SPAWN_SUBAGENT_TOOL_NAME not in tools, role.value


def test_subagent_config_defaults_cannot_spawn():
    config = SubAgentConfig(
        role=SubAgentRole.RESEARCHER,
        task="t",
        allowed_tools=["read"],
        tool_budget=5,
        context_limit=1000,
    )
    assert config.can_spawn_subagents is False


def test_constrained_member_registry_never_registers_spawn_tool():
    """The member-execution tool boundary is allowed_tools: a parent registry
    that HAS the spawn tool does not leak it into a child whose allowed_tools
    exclude it (negative test for the depth-1 enforcement)."""
    parent = MagicMock()
    parent.settings = MagicMock()
    parent.settings.tool_budget = 50
    parent.settings.max_context_chars = 100000
    parent.provider_name = "mock_provider"
    parent.model = "mock_model"
    parent.temperature = 0.7
    parent.tool_registry = MagicMock()

    def _get(name):
        tool = MagicMock()
        tool.name = name
        return tool  # parent HAS every tool, including spawn_subagent

    parent.tool_registry.get.side_effect = _get
    # The SubAgentContextAdapter reads the parent registry via `tools`.
    parent.tools = MagicMock()
    parent.tools.get.side_effect = _get

    config = SubAgentConfig(
        role=SubAgentRole.RESEARCHER,
        task="child task",
        allowed_tools=["read", "grep"],  # spawn_subagent NOT authorized
        tool_budget=5,
        context_limit=1000,
    )
    subagent = SubAgent(config, parent)

    child_registry = MagicMock()
    child_orchestrator = MagicMock()
    child_orchestrator.tool_registry = child_registry
    subagent._configure_allowed_tools(child_orchestrator)

    registered_names = [call.args[0].name for call in child_registry.register.call_args_list]
    assert SPAWN_SUBAGENT_TOOL_NAME not in registered_names
    assert set(registered_names) == {"read", "grep"}


# =============================================================================
# Permission gating (both directions) + write-tier registration
# =============================================================================


def test_declared_permission_is_workspace_write():
    """The tool is registered in DEFAULT_TOOL_PERMISSIONS mirroring the
    write-tier tools, and its metadata access mode agrees."""
    assert DEFAULT_TOOL_PERMISSIONS[SPAWN_SUBAGENT_TOOL_NAME] == PermissionMode.WORKSPACE_WRITE
    assert spawn_subagent.Tool.name == SPAWN_SUBAGENT_TOOL_NAME
    assert spawn_subagent.Tool.access_mode is AccessMode.WRITE


def test_read_only_parent_cannot_spawn():
    policy = PermissionPolicy(active_mode=PermissionMode.READ_ONLY)
    decision = policy.authorize(SPAWN_SUBAGENT_TOOL_NAME)
    assert not decision.allowed


def test_workspace_write_parent_can_spawn():
    policy = PermissionPolicy(active_mode=PermissionMode.WORKSPACE_WRITE)
    assert policy.authorize(SPAWN_SUBAGENT_TOOL_NAME).allowed


def test_full_access_parent_can_spawn():
    policy = PermissionPolicy(active_mode=PermissionMode.DANGER_FULL_ACCESS)
    assert policy.authorize(SPAWN_SUBAGENT_TOOL_NAME).allowed


# =============================================================================
# Input validation
# =============================================================================


async def test_empty_prompt_refused(exec_ctx, runner):
    result = await spawn_subagent(prompt="   ", _exec_ctx=exec_ctx)

    assert result["success"] is False
    assert "prompt" in result["error"].lower()
    runner.assert_not_awaited()


# =============================================================================
# Finding (delegation grant widening): requested ∩ role grants ∩ parent
# =============================================================================


def test_compute_child_tool_set_intersects_all_three_grant_sources():
    """A restrict-list never REPLACES the role list: only tools present in
    the requested list AND the role grants AND the parent's admitted set
    survive (alias-resolved, requested order preserved)."""
    final = compute_child_tool_set(
        requested=["grep", "read", "write"],
        role_grants=["read", "grep", "search"],
        parent_admitted=["read", "grep"],
    )
    assert final == ["grep", "read"]


def test_compute_child_tool_set_none_requested_uses_bounded_role_defaults():
    """No explicit list -> role defaults, still intersected with the parent's
    admitted set (role-provider defaults are selected after the request and
    must be bounded too)."""
    final = compute_child_tool_set(
        requested=None,
        role_grants=["read", "web_search", "code_search"],
        parent_admitted=["read", "web_search", "shell"],
    )
    assert final == ["read", "web_search"]


def test_compute_child_tool_set_resolves_aliases_before_intersecting():
    """Alias names resolve to canonical form before the intersection."""
    final = compute_child_tool_set(
        requested=["execute_bash"],  # alias of "shell"
        role_grants=["shell"],
        parent_admitted=["shell"],
    )
    assert final == ["shell"]


def test_compute_child_tool_set_never_re_admits_spawn_tool():
    """The spawn tool cannot survive from any source — depth 1 is final."""
    final = compute_child_tool_set(
        requested=[SPAWN_SUBAGENT_TOOL_NAME, "read"],
        role_grants=["read"],
        parent_admitted=["read", SPAWN_SUBAGENT_TOOL_NAME],
    )
    assert final == ["read"]


# =============================================================================
# Finding (spawn session ids): one minted child session id per invocation
# =============================================================================


def test_mint_child_session_id_is_unique_per_invocation():
    """Overlapping spawns and retries under one parent never share an
    upstream session key."""
    a = mint_child_session_id("sess-parent")
    b = mint_child_session_id("sess-parent")
    assert a != b
    assert a.startswith("sess-parent-spawn-")
    assert b.startswith("sess-parent-spawn-")


def test_mint_child_session_id_falls_back_to_context_session():
    from victor.core.context import get_session_id, session_id, set_session_id

    token = set_session_id("ctx-session")
    try:
        minted = mint_child_session_id(None)
        assert minted.startswith("ctx-session-spawn-")
        assert get_session_id() == "ctx-session"
    finally:
        session_id.reset(token)


def test_explicit_child_session_id_wins_over_member_id_derivation():
    """The minted id is precedence 1: it — not ``{parent}-{member_id}`` —
    keys the provider session binding."""
    config = SubAgentConfig(
        role=SubAgentRole.RESEARCHER,
        task="t",
        allowed_tools=["read"],
        tool_budget=5,
        context_limit=1000,
        member_id="spawn_subagent",
        agent_id="agent_1",
        parent_session_id="sess-parent",
        child_session_id="sess-parent-spawn-abc",
    )
    assert config.resolve_member_session_id() == "sess-parent-spawn-abc"
    assert config.to_runtime_context().session_id == "sess-parent-spawn-abc"


def test_derivation_without_explicit_id_still_uses_member_identity():
    """Legacy members (no minted id) keep the documented derivation."""
    config = SubAgentConfig(
        role=SubAgentRole.RESEARCHER,
        task="t",
        allowed_tools=["read"],
        tool_budget=5,
        context_limit=1000,
        member_id="spawn_subagent",
        agent_id="agent_1",
        parent_session_id="sess-parent",
    )
    assert config.resolve_member_session_id() == "sess-parent-spawn_subagent"


def test_identity_metadata_reports_resolved_session_id():
    """The returned child-id field reports the RESOLVED session id, not the
    optional input (which used to read None for every foreground spawn)."""
    config = SubAgentConfig(
        role=SubAgentRole.RESEARCHER,
        task="t",
        allowed_tools=["read"],
        tool_budget=5,
        context_limit=1000,
        member_id="spawn_subagent",
        parent_session_id="sess-parent",
    )
    member = SubAgent(config, MagicMock())
    member._resolved_session_id = "sess-parent-spawn-abc"
    metadata = member._identity_metadata()
    assert metadata["child_session_id"] == "sess-parent-spawn-abc"
    assert metadata["session_id"] == "sess-parent-spawn-abc"


def test_identity_metadata_falls_back_to_configured_child_session_id():
    """Before execution binds a session (e.g. construction-time audit), the
    configured id is still surfaced rather than a null field."""
    config = SubAgentConfig(
        role=SubAgentRole.RESEARCHER,
        task="t",
        allowed_tools=["read"],
        tool_budget=5,
        context_limit=1000,
        member_id="spawn_subagent",
        parent_session_id="sess-parent",
        child_session_id="sess-parent-spawn-abc",
    )
    member = SubAgent(config, MagicMock())
    assert member._identity_metadata()["child_session_id"] == "sess-parent-spawn-abc"


# =============================================================================
# Finding (grant widening, admission layer): the child registry consults the
# spawn boolean and never admits an out-of-parent tool
# =============================================================================


def test_constrained_registry_skips_spawn_tool_even_when_listed():
    """A member that cannot spawn is never REGISTERED the spawn tool, even if
    a caller smuggled it into allowed_tools past the upstream guards."""
    parent = MagicMock()
    parent.settings = MagicMock()
    parent.settings.tool_budget = 50
    parent.settings.max_context_chars = 100000
    parent.provider_name = "mock_provider"
    parent.model = "mock_model"
    parent.temperature = 0.7

    def _get(name):
        tool = MagicMock()
        tool.name = name
        return tool

    parent.tool_registry = MagicMock()
    parent.tool_registry.get.side_effect = _get
    parent.tools = MagicMock()
    parent.tools.get.side_effect = _get

    config = SubAgentConfig(
        role=SubAgentRole.RESEARCHER,
        task="child task",
        allowed_tools=["read", SPAWN_SUBAGENT_TOOL_NAME],
        tool_budget=5,
        context_limit=1000,
        can_spawn_subagents=False,
    )
    subagent = SubAgent(config, parent)

    child_registry = MagicMock()
    child_orchestrator = MagicMock()
    child_orchestrator.tool_registry = child_registry
    subagent._configure_allowed_tools(child_orchestrator)

    registered_names = [call.args[0].name for call in child_registry.register.call_args_list]
    assert registered_names == ["read"]

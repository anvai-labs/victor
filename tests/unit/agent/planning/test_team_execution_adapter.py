from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.agent.planning.base import PlanStep, StepType
from victor.agent.planning.readable_schema import ReadableTaskPlan, TaskComplexity
from victor.agent.planning.team_execution import PlanningTeamExecutionAdapter
from victor.teams.types import MemberResult, TeamFormation, TeamResult


class RecordingCoordinator:
    def __init__(self):
        self.members = []
        self.manager = None
        self.formation = None
        self.context = None

    def add_member(self, member):
        self.members.append(member)
        return self

    def set_manager(self, manager):
        return self.set_supervisor(manager)

    def set_supervisor(self, manager):
        self.manager = manager
        if manager not in self.members:
            self.members.insert(0, manager)
        return self

    def set_formation(self, formation):
        self.formation = formation
        return self

    async def execute_task(self, task, context):
        self.context = context
        return TeamResult(
            success=True,
            final_output="team complete",
            formation=self.formation,
            total_tool_calls=3,
            member_results={
                member.id: MemberResult(
                    member_id=member.id,
                    success=True,
                    output=f"{member.id} output",
                    metadata={"member_id": member.id},
                    tool_calls_used=1,
                )
                for member in self.members
            },
        )


class FallbackWorkerCoordinator(RecordingCoordinator):
    async def execute_task(self, task, context):
        self.context = context
        return TeamResult(
            success=False,
            final_output="inventory complete",
            formation=self.formation,
            total_tool_calls=1,
            member_results={
                "plan_manager": MemberResult(
                    member_id="plan_manager",
                    success=False,
                    output="",
                    error="Unknown or disabled tool: shell",
                ),
                "step_2_researcher": MemberResult(
                    member_id="step_2_researcher",
                    success=True,
                    output="inventory complete",
                    tool_calls_used=1,
                ),
            },
        )


def _complex_plan() -> ReadableTaskPlan:
    return ReadableTaskPlan(
        name="Rust Arc Review",
        complexity=TaskComplexity.COMPLEX,
        desc="Review Rust Arc and readonly usage by workspace",
        steps=[
            ["1", "research", "Map Rust workspaces", "ls,read"],
            ["2", "review", "Review Arc usage", "grep,read", ["1"]],
        ],
    )


def test_subagent_output_for_parent_appends_tool_evidence_for_thin_final_response():
    result = SimpleNamespace(
        summary="Done.",
        tool_calls_used=6,
        details={
            "full_response": "Done.",
            "tool_evidence": {
                "tool_names": ["ls", "read"],
                "summary": "ls: crates/core Cargo.toml clients/python/pyproject.toml\n"
                'read: [workspace] members = ["crates/core"]',
            },
        },
    )

    output = PlanningTeamExecutionAdapter._subagent_output_for_parent(result)

    assert "Done." in output
    assert "Tool-backed evidence digest" in output
    assert "crates/core" in output


@pytest.mark.asyncio
async def test_adapter_executes_plan_step_with_direct_worker():
    coordinator = RecordingCoordinator()
    parent = SimpleNamespace(active_session_id="session_root")
    subagents = MagicMock()
    subagents.spawn = AsyncMock(
        return_value=SimpleNamespace(
            success=True,
            summary="Read Cargo.toml and found 2 workspace crates.",
            error=None,
            details={"full_response": "Read Cargo.toml and clients/rust/Cargo.toml."},
            tool_calls_used=2,
            duration_seconds=0.1,
        )
    )
    adapter = PlanningTeamExecutionAdapter(
        orchestrator=parent,
        sub_agent_orchestrator=subagents,
        coordinator_factory=lambda _orchestrator: coordinator,
    )
    plan = _complex_plan()
    execution_plan = plan.to_execution_plan()

    result = await adapter.execute_step(
        plan=plan,
        execution_plan=execution_plan,
        step=execution_plan.steps[0],
        root_session_id="session_root",
    )

    assert result.success is True
    assert result.output == "Read Cargo.toml and found 2 workspace crates."
    assert coordinator.formation is None
    assert coordinator.manager is None
    spawn_kwargs = subagents.spawn.await_args.kwargs
    assert spawn_kwargs["member_id"] == "step_1_researcher"
    assert spawn_kwargs["parent_session_id"] == "session_root"
    assert spawn_kwargs["child_session_id"] == (
        f"session_root:team_{execution_plan.id}_1:step_1_researcher"
    )
    assert result.metadata["execution_mode"] == "direct_step_worker"
    assert result.artifacts == []


@pytest.mark.asyncio
async def test_adapter_member_executor_spawns_isolated_child_session_with_handoff_metadata():
    parent = SimpleNamespace(active_session_id="session_root")
    subagents = MagicMock()
    subagents.spawn = AsyncMock(
        return_value=SimpleNamespace(
            success=True,
            summary="bounded child handoff",
            error=None,
            details={
                "agent_id": "team_plan_1_step_1_step_1_researcher",
                "parent_handoff": {"summary": "bounded child handoff"},
            },
            tool_calls_used=2,
            duration_seconds=0.2,
        )
    )
    adapter = PlanningTeamExecutionAdapter(
        orchestrator=parent,
        sub_agent_orchestrator=subagents,
    )
    plan = _complex_plan()
    execution_plan = plan.to_execution_plan()
    team_id = "team_plan_1_step_1"
    members = adapter._build_members(execution_plan, team_id)

    payload = await members["step_1_researcher"].execute_task(
        "Map Rust workspaces",
        {
            "root_session_id": "session_root",
            "parent_session_id": "session_root",
            "plan_step_id": "1",
        },
    )

    assert payload["success"] is True
    assert payload["metadata"]["parent_handoff"]["summary"] == "bounded child handoff"
    spawn_kwargs = subagents.spawn.await_args.kwargs
    assert spawn_kwargs["member_id"] == "step_1_researcher"
    assert spawn_kwargs["agent_id"] == "team_plan_1_step_1_step_1_researcher"
    assert spawn_kwargs["team_id"] == team_id
    assert spawn_kwargs["parent_session_id"] == "session_root"
    assert spawn_kwargs["child_session_id"] == "session_root:team_plan_1_step_1:step_1_researcher"
    assert spawn_kwargs["plan_id"] == execution_plan.id
    assert spawn_kwargs["plan_step_id"] == "1"
    assert spawn_kwargs["allowed_tools"] == ["read", "ls", "grep"]


@pytest.mark.asyncio
async def test_adapter_preserves_shell_from_plan_step_tool_hints():
    parent = SimpleNamespace(active_session_id="session_root")
    subagents = MagicMock()
    subagents.spawn = AsyncMock(
        return_value=SimpleNamespace(
            success=True,
            summary="inventory complete",
            error=None,
            details={},
            tool_calls_used=1,
            duration_seconds=0.1,
        )
    )
    adapter = PlanningTeamExecutionAdapter(
        orchestrator=parent,
        sub_agent_orchestrator=subagents,
    )
    plan = ReadableTaskPlan(
        name="Rust Inventory",
        complexity=TaskComplexity.COMPLEX,
        desc="Inventory Rust source files",
        steps=[["3", "analyze", "Inventory all Rust source files", "grep,shell"]],
    )
    execution_plan = plan.to_execution_plan()
    members = adapter._build_members(execution_plan, "team_plan_step")

    await members["step_3_researcher"].execute_task(
        "Inventory all Rust source files",
        {"root_session_id": "session_root", "parent_session_id": "session_root"},
    )

    spawn_kwargs = subagents.spawn.await_args.kwargs
    assert spawn_kwargs["role"].value == "researcher"
    assert spawn_kwargs["allowed_tools"] == ["read", "ls", "grep", "shell"]


@pytest.mark.asyncio
async def test_adapter_filters_unavailable_shell_from_plan_step_tool_hints():
    """Regression: plans can mention shell even when the active runtime lacks it.

    The sub-agent allowlist must reflect the parent runtime's actual enabled tools;
    otherwise the model receives an unusable shell schema and tool execution is
    skipped as ``Unknown or disabled tool: shell``.
    """

    parent = SimpleNamespace(
        active_session_id="session_root",
        get_enabled_tools=lambda: {"read", "ls", "grep", "code_search"},
    )
    subagents = MagicMock()
    subagents.spawn = AsyncMock(
        return_value=SimpleNamespace(
            success=True,
            summary="inventory complete",
            error=None,
            details={},
            tool_calls_used=1,
            duration_seconds=0.1,
        )
    )
    adapter = PlanningTeamExecutionAdapter(
        orchestrator=parent,
        sub_agent_orchestrator=subagents,
    )
    plan = ReadableTaskPlan(
        name="Rust Inventory",
        complexity=TaskComplexity.COMPLEX,
        desc="Inventory Rust source files",
        steps=[
            [
                "3",
                "analyze",
                "Inventory all Rust source files",
                "grep,shell,code_search",
            ]
        ],
    )
    execution_plan = plan.to_execution_plan()
    members = adapter._build_members(execution_plan, "team_plan_step")

    await members["step_3_researcher"].execute_task(
        "Inventory all Rust source files",
        {"root_session_id": "session_root", "parent_session_id": "session_root"},
    )

    spawn_kwargs = subagents.spawn.await_args.kwargs
    assert spawn_kwargs["role"].value == "researcher"
    assert spawn_kwargs["allowed_tools"] == ["read", "ls", "grep", "code_search"]


@pytest.mark.asyncio
async def test_adapter_executes_compute_node_step_without_provider():
    """Explicit exec='compute' steps run deterministically — no subagent, no model call."""
    from victor.agent.planning.base import StepResult

    parent = SimpleNamespace(active_session_id="session_root")
    subagents = MagicMock()
    subagents.spawn = AsyncMock()

    # Register a language-specific compute node (simulates what victor-coding does at
    # plugin init — the framework itself ships with no language-specific content).
    PlanningTeamExecutionAdapter.register_compute_node(
        "rust_best_practices_checklist",
        lambda step: StepResult(
            success=True,
            output="Rust best-practices checklist:\n1. Ownership ...\n2. Arc ...",
            tool_calls_used=0,
            metadata={
                "execution_mode": "compute_node",
                "compute_node": "rust_best_practices_checklist",
                "node_type": "deterministic_planning_step",
            },
        ),
    )
    adapter = PlanningTeamExecutionAdapter(
        orchestrator=parent,
        sub_agent_orchestrator=subagents,
    )
    # Use the rich dict format with explicit exec + node so dispatch is unambiguous.
    plan = ReadableTaskPlan(
        name="Rust Checklist",
        complexity=TaskComplexity.COMPLEX,
        desc="Create Rust review checklist",
        steps=[
            {
                "id": "4",
                "type": "doc",
                "desc": "Create a master Rust best practices checklist",
                "tools": [],
                "deps": [],
                "exec": "compute",
                "node": "rust_best_practices_checklist",
                "exit": ["checklist covers Arc and immutability"],
            }
        ],
    )
    execution_plan = plan.to_execution_plan()

    result = await adapter.execute_step(
        plan=plan,
        execution_plan=execution_plan,
        step=execution_plan.steps[0],
        root_session_id="session_root",
    )

    assert result.success is True
    assert "Rust best-practices checklist" in result.output
    assert "Arc" in result.output
    assert result.metadata["execution_mode"] == "compute_node"
    assert result.metadata["compute_node"] == "rust_best_practices_checklist"
    assert result.metadata["node_type"] == "deterministic_planning_step"
    subagents.spawn.assert_not_awaited()

    # Clean up registry so other tests are not affected.
    PlanningTeamExecutionAdapter._COMPUTE_NODES.pop("rust_best_practices_checklist", None)


@pytest.mark.asyncio
async def test_checklist_produces_step_uses_builtin_compute_without_subagent():
    """Checklist artifacts should be deterministic planning outputs, not 300s agent loops."""
    parent = SimpleNamespace(active_session_id="session_root")
    subagents = MagicMock()
    subagents.spawn = AsyncMock()
    adapter = PlanningTeamExecutionAdapter(
        orchestrator=parent,
        sub_agent_orchestrator=subagents,
    )
    plan = ReadableTaskPlan(
        name="Rust checklist",
        complexity=TaskComplexity.COMPLEX,
        desc="Create Rust review checklist",
        steps=[
            {
                "id": "5",
                "type": "doc",
                "desc": (
                    "Create comprehensive Rust best practices checklist covering: "
                    "Arc vs Rc selection, unnecessary clone elimination, zero-copy patterns"
                ),
                "tools": [],
                "deps": [],
                "exec": "compute",
                "produces": "best_practices_checklist",
            }
        ],
    )
    execution_plan = plan.to_execution_plan()

    result = await adapter.execute_step(
        plan=plan,
        execution_plan=execution_plan,
        step=execution_plan.steps[0],
        root_session_id="session_root",
    )

    assert result.success is True
    assert result.tool_calls_used == 0
    assert result.metadata["compute_node"] == "_checklist_artifact"
    assert "Arc vs Rc selection" in result.output
    assert "unnecessary clone elimination" in result.output
    assert "Record file:line evidence" in result.output
    subagents.spawn.assert_not_awaited()


def test_compute_node_for_step_does_not_treat_analysis_against_checklist_as_checklist():
    """Review steps mentioning a checklist must still use tool-backed analysis."""
    step = PlanStep(
        id="8a",
        description=(
            "Deep review of each workspace crate one-by-one. Evaluate each module "
            "and file against the checklist."
        ),
        step_type=StepType.RESEARCH,
        context={"produces": "per_crate_findings"},
    )

    assert PlanningTeamExecutionAdapter._compute_node_for_step(step) is None


@pytest.mark.asyncio
async def test_cross_crate_findings_step_uses_agent_when_not_explicit_compute(
    tmp_path, monkeypatch
):
    """Cross-crate analysis must not be downgraded to a zero-tool pattern count."""
    rust = tmp_path / "rust"
    crate_a = rust / "crates" / "protocol"
    crate_b = rust / "crates" / "state"
    (crate_a / "src").mkdir(parents=True)
    (crate_b / "src").mkdir(parents=True)
    (rust).mkdir(exist_ok=True)
    (rust / "Cargo.toml").write_text('[workspace]\nmembers = ["crates/protocol", "crates/state"]\n')
    (crate_a / "Cargo.toml").write_text(
        '[package]\nname = "protocol"\nversion = "0.1.0"\n' '[dependencies]\nserde = "1"\n'
    )
    (crate_a / "src" / "lib.rs").write_text(
        "use std::sync::Arc;\npub struct Shared(pub Arc<String>);\n"
    )
    (crate_b / "Cargo.toml").write_text(
        '[package]\nname = "state"\nversion = "0.1.0"\n'
        '[dependencies]\nprotocol = { path = "../protocol" }\nserde = "1.0"\n'
    )
    (crate_b / "src" / "lib.rs").write_text(
        "use std::sync::{Arc, Mutex};\npub fn clone_it(v: Arc<String>) { let _ = v.clone(); }\n"
    )
    monkeypatch.chdir(tmp_path)

    parent = SimpleNamespace(active_session_id="session_root")
    subagents = MagicMock()
    subagents.spawn = AsyncMock(
        return_value=SimpleNamespace(
            success=True,
            summary=(
                "Reviewed rust/crates/protocol/src/lib.rs and "
                "rust/crates/state/src/lib.rs. state depends on protocol; Arc is local."
            ),
            error=None,
            details={"tool_names_used": ["read", "code_search"]},
            tool_calls_used=6,
            duration_seconds=0.2,
        )
    )
    adapter = PlanningTeamExecutionAdapter(
        orchestrator=parent,
        sub_agent_orchestrator=subagents,
    )
    plan = ReadableTaskPlan(
        name="Cross crate",
        complexity=TaskComplexity.COMPLEX,
        desc="Review workspace",
        steps=[
            {
                "id": "8",
                "type": "analyze",
                "desc": "Cross-crate analysis: shared Arc patterns across crate boundaries",
                "tools": ["read", "code_search"],
                "deps": [],
                "produces": "cross_crate_findings",
            }
        ],
    )
    execution_plan = plan.to_execution_plan()

    result = await adapter.execute_step(
        plan=plan,
        execution_plan=execution_plan,
        step=execution_plan.steps[0],
        root_session_id="session_root",
        plan_state={"workspace_members": ["rust/crates/protocol", "rust/crates/state"]},
    )

    assert result.success is True
    assert result.tool_calls_used == 6
    assert "rust/crates/state/src/lib.rs" in result.output
    assert "compute_node" not in result.metadata
    subagents.spawn.assert_awaited_once()


@pytest.mark.asyncio
async def test_rust_crate_review_uses_agent_when_not_explicit_compute(tmp_path, monkeypatch):
    """Deep Rust reviews should use a worker so file reads are visible in the graph."""
    crate = tmp_path / "rust" / "crates" / "python-bindings"
    src = crate / "src"
    src.mkdir(parents=True)
    (crate / "Cargo.toml").write_text('[package]\nname = "victor_native"\nversion = "0.1.0"\n')
    (src / "lib.rs").write_text(
        "use std::sync::Arc;\n"
        "pub struct Matcher { inner: Arc<String> }\n"
        "impl Matcher {\n"
        "  pub fn new(value: String) -> Self { Self { inner: Arc::new(value) } }\n"
        '  pub fn label(&self) -> String { format!("{}", self.inner.clone()) }\n'
        "}\n"
    )
    monkeypatch.chdir(tmp_path)

    parent = SimpleNamespace(active_session_id="session_root")
    subagents = MagicMock()
    subagents.spawn = AsyncMock(
        return_value=SimpleNamespace(
            success=True,
            summary=(
                "Reviewed rust/crates/python-bindings/src/lib.rs:2 and :4; "
                "Arc wraps immutable state and format! clones should be checked."
            ),
            error=None,
            details={"tool_names_used": ["read", "code_search"]},
            tool_calls_used=5,
            duration_seconds=0.2,
        )
    )
    adapter = PlanningTeamExecutionAdapter(
        orchestrator=parent,
        sub_agent_orchestrator=subagents,
    )
    plan = ReadableTaskPlan(
        name="Crate review",
        complexity=TaskComplexity.COMPLEX,
        desc="Review Rust crate",
        steps=[
            {
                "id": "10",
                "type": "analyze",
                "desc": (
                    "Review python-bindings crate: Arc usage for shared Python state, "
                    "immutable data transfer, pyclass design"
                ),
                "tools": ["read", "code_search"],
                "deps": [],
                "produces": "findings_python_bindings",
            }
        ],
    )
    execution_plan = plan.to_execution_plan()

    result = await adapter.execute_step(
        plan=plan,
        execution_plan=execution_plan,
        step=execution_plan.steps[0],
        root_session_id="session_root",
        plan_state={"workspace_members": ["rust/crates/python-bindings"]},
    )

    assert result.success is True
    assert result.tool_calls_used == 5
    assert "rust/crates/python-bindings/src/lib.rs:2" in result.output
    assert "compute_node" not in result.metadata
    subagents.spawn.assert_awaited_once()


def test_compute_node_for_step_does_not_auto_detect_rust_crate_review():
    step = PlanStep(
        id="6",
        description="Review edge-runtime crate: Arc usage patterns and async correctness",
        step_type=StepType.RESEARCH,
        context={"produces": "findings_edge_runtime"},
    )

    assert PlanningTeamExecutionAdapter._compute_node_for_step(step) is None


def test_compute_node_for_step_does_not_infer_rust_crate_review_from_compute_only():
    step = PlanStep(
        id="6",
        description="Review edge-runtime crate: Arc usage patterns and async correctness",
        step_type=StepType.RESEARCH,
        execution="compute",
        context={"produces": "findings_edge_runtime"},
    )

    assert PlanningTeamExecutionAdapter._compute_node_for_step(step) is None


def test_compute_node_for_step_does_not_auto_detect_ranked_rust_findings_report():
    step = PlanStep(
        id="10",
        description="Aggregate and rank all findings by impact and effort for Rust Arc review",
        step_type=StepType.RESEARCH,
        context={
            "produces": "ranked_findings",
            "inputs": ["per_crate_findings", "cross_crate_findings"],
        },
    )

    assert PlanningTeamExecutionAdapter._compute_node_for_step(step) is None


def test_compute_node_for_step_does_not_auto_detect_rust_final_report():
    step = PlanStep(
        id="10",
        description=(
            "Synthesize all findings into a prioritized report: per-crate findings "
            "with file:line references and cross-crate themes"
        ),
        step_type=StepType.RESEARCH,
        context={
            "produces": "final_report",
            "inputs": ["per_crate_findings", "cross_crate_findings"],
        },
    )

    assert PlanningTeamExecutionAdapter._compute_node_for_step(step) is None


def test_compute_node_for_step_does_not_infer_ranked_rust_findings_from_compute_only():
    step = PlanStep(
        id="10",
        description="Aggregate and rank all findings by impact and effort for Rust Arc review",
        step_type=StepType.RESEARCH,
        execution="compute",
        context={
            "produces": "ranked_findings",
            "inputs": ["per_crate_findings", "cross_crate_findings"],
        },
    )

    assert PlanningTeamExecutionAdapter._compute_node_for_step(step) is None


@pytest.mark.asyncio
async def test_compute_node_receives_plan_state():
    """Compute nodes registered with fn(step, plan_state) can read prior step outputs."""
    from victor.agent.planning.base import StepResult

    received_plan_state: dict = {}

    def _checklist_with_context(step, plan_state: dict) -> StepResult:
        received_plan_state.update(plan_state)
        members = plan_state.get("workspace_members", [])
        return StepResult(
            success=True,
            output=f"Checklist for {len(members)} crate(s)",
            tool_calls_used=0,
            metadata={
                "execution_mode": "compute_node",
                "compute_node": "lang_checklist",
            },
        )

    PlanningTeamExecutionAdapter.register_compute_node("lang_checklist", _checklist_with_context)
    parent = SimpleNamespace(active_session_id="session_root")
    subagents = MagicMock()
    subagents.spawn = AsyncMock()
    adapter = PlanningTeamExecutionAdapter(orchestrator=parent, sub_agent_orchestrator=subagents)

    plan = ReadableTaskPlan(
        name="Checklist test",
        complexity=TaskComplexity.MODERATE,
        desc="Test plan_state pass-through to compute node",
        steps=[
            {
                "id": "1",
                "type": "analyze",
                "desc": "Checklist",
                "exec": "compute",
                "node": "lang_checklist",
            }
        ],
    )
    exec_plan = plan.to_execution_plan()
    step = exec_plan.steps[0]

    result = await adapter.execute_step(
        plan=plan,
        execution_plan=exec_plan,
        step=step,
        plan_state={"workspace_members": ["core", "util", "cli"]},
    )

    assert result.success is True
    assert "3 crate(s)" in result.output
    assert received_plan_state.get("workspace_members") == ["core", "util", "cli"]
    subagents.spawn.assert_not_awaited()

    PlanningTeamExecutionAdapter._COMPUTE_NODES.pop("lang_checklist", None)


# ---------------------------------------------------------------------------
# Team-routing / dict-format parsing
# ---------------------------------------------------------------------------


def test_should_use_team_handles_dict_format_steps():
    """should_use_team must not crash on rich dict steps (KeyError: 1 guard)."""
    adapter = PlanningTeamExecutionAdapter(orchestrator=SimpleNamespace())

    # Dict-format steps with exploratory types → exploratory count >= 2
    plan_moderate = ReadableTaskPlan(
        name="Dict plan",
        complexity=TaskComplexity.MODERATE,
        desc="Dict step team detection",
        steps=[
            {"id": "1", "type": "analyze", "desc": "Analyze codebase"},
            {"id": "2", "type": "review", "desc": "Review patterns"},
            {"id": "3", "type": "feature", "desc": "Apply fixes"},
        ],
    )
    assert adapter.should_use_team(plan_moderate) is True

    # Complex plans always use team regardless of step format
    plan_complex = ReadableTaskPlan(
        name="Complex plan",
        complexity=TaskComplexity.COMPLEX,
        desc="Complex",
        steps=[{"id": "1", "type": "analyze", "desc": "step"}],
    )
    assert adapter.should_use_team(plan_complex) is True

    # Mixed list + dict steps
    plan_mixed = ReadableTaskPlan(
        name="Mixed",
        complexity=TaskComplexity.MODERATE,
        desc="Mixed step formats",
        steps=[
            ["1", "analyze", "List step"],
            {"id": "2", "type": "analyze", "desc": "Dict step"},
        ],
    )
    assert adapter.should_use_team(plan_mixed) is True


def test_should_use_team_forces_team_for_advanced_execution_types():
    """Any step with compute/loop/conditional/approval forces team routing."""
    adapter = PlanningTeamExecutionAdapter(orchestrator=SimpleNamespace())

    for exec_type in (
        "compute",
        "loop",
        "conditional",
        "approval",
        "checkpoint",
        "tool",
    ):
        plan = ReadableTaskPlan(
            name=f"Plan with {exec_type}",
            complexity=TaskComplexity.SIMPLE,
            desc=f"Single {exec_type} step",
            steps=[{"id": "1", "type": "feature", "desc": "Step", "exec": exec_type}],
        )
        assert adapter.should_use_team(plan) is True, f"Expected True for exec={exec_type!r}"

    # Compact list format with 6th element
    plan_list = ReadableTaskPlan(
        name="List format compute",
        complexity=TaskComplexity.SIMPLE,
        desc="List step with exec",
        steps=[["1", "feature", "Compute step", "read", [], "compute"]],
    )
    assert adapter.should_use_team(plan_list) is True

    # Plain agent step on a SIMPLE plan → only 1 exploratory step → False
    plan_simple = ReadableTaskPlan(
        name="Simple plan",
        complexity=TaskComplexity.SIMPLE,
        desc="Simple single step",
        steps=[{"id": "1", "type": "feature", "desc": "Write code"}],
    )
    assert adapter.should_use_team(plan_simple) is False


def test_adapter_gives_step_tools_to_hierarchical_manager():
    adapter = PlanningTeamExecutionAdapter(orchestrator=SimpleNamespace())
    plan = ReadableTaskPlan(
        name="Rust Inventory",
        complexity=TaskComplexity.COMPLEX,
        desc="Inventory Rust source files",
        steps=[["2", "analyze", "Enumerate all Rust source files", "shell,read"]],
    )
    execution_plan = plan.to_execution_plan()

    members = adapter._build_members(
        execution_plan,
        "team_plan_step",
        current_step=execution_plan.steps[0],
    )

    assert members["plan_manager"].member.allowed_tools == [
        "read",
        "ls",
        "grep",
        "shell",
    ]


def test_adapter_treats_successful_fallback_worker_as_step_success():
    result = TeamResult(
        success=False,
        final_output="worker output",
        formation=TeamFormation.HIERARCHICAL,
        total_tool_calls=4,
        member_results={
            "plan_manager": MemberResult(
                member_id="plan_manager",
                success=False,
                output="",
                error="Unknown or disabled tool: shell",
                tool_calls_used=0,
            ),
            "step_2_researcher": MemberResult(
                member_id="step_2_researcher",
                success=True,
                output="inventory complete",
                tool_calls_used=1,
            ),
        },
    )

    step_result = PlanningTeamExecutionAdapter._team_result_to_step_result(result)

    assert step_result.success is True
    assert step_result.output == "inventory complete"
    assert step_result.error is None
    assert step_result.tool_calls_used == 4


def test_adapter_treats_dict_shaped_successful_fallback_worker_as_step_success():
    result = {
        "success": False,
        "final_output": "worker output",
        "error": "manager failed",
        "total_tool_calls": 1,
        "member_results": {
            "plan_manager": {
                "success": False,
                "output": "",
                "error": "shell unavailable",
            },
            "step_2_researcher": {"success": True, "output": "inventory complete"},
        },
    }

    step_result = PlanningTeamExecutionAdapter._team_result_to_step_result(result)

    assert step_result.success is True
    assert step_result.output == "inventory complete"
    assert step_result.error is None


@pytest.mark.asyncio
async def test_adapter_execute_step_does_not_start_hierarchical_manager_for_single_step():
    coordinator = FallbackWorkerCoordinator()
    subagents = MagicMock()
    subagents.spawn = AsyncMock(
        return_value=SimpleNamespace(
            success=True,
            summary="Enumerated 192 Rust files including src/lib.rs.",
            error=None,
            details={},
            tool_calls_used=1,
            duration_seconds=0.1,
        )
    )
    adapter = PlanningTeamExecutionAdapter(
        orchestrator=SimpleNamespace(active_session_id="session_root"),
        sub_agent_orchestrator=subagents,
        coordinator_factory=lambda _orchestrator: coordinator,
    )
    plan = ReadableTaskPlan(
        name="Rust Inventory",
        complexity=TaskComplexity.COMPLEX,
        desc="Inventory Rust source files",
        steps=[["2", "analyze", "Enumerate all Rust source files", "shell,read"]],
    )
    execution_plan = plan.to_execution_plan()

    result = await adapter.execute_step(
        plan=plan,
        execution_plan=execution_plan,
        step=execution_plan.steps[0],
        root_session_id="session_root",
    )

    assert result.success is True
    assert result.output == "Enumerated 192 Rust files including src/lib.rs."
    assert coordinator.manager is None
    spawn_kwargs = subagents.spawn.await_args.kwargs
    assert spawn_kwargs["allowed_tools"] == ["read", "ls", "grep", "shell"]


def test_adapter_only_uses_team_for_complex_exploratory_plans():
    adapter = PlanningTeamExecutionAdapter(orchestrator=SimpleNamespace())
    complex_plan = _complex_plan()
    simple_plan = ReadableTaskPlan(
        name="Small Fix",
        complexity=TaskComplexity.SIMPLE,
        desc="Fix one typo",
        steps=[["1", "feature", "Fix typo", "edit"]],
    )

    assert adapter.should_use_team(complex_plan) is True
    assert adapter.should_use_team(simple_plan) is False


# ---------------------------------------------------------------------------
# Loop node tests
# ---------------------------------------------------------------------------


def _spawn_result(item: str, success: bool = True):
    from types import SimpleNamespace as NS

    return NS(
        success=success,
        summary=f"Reviewed {item}.",
        error=None if success else f"Failed on {item}",
        tool_calls_used=3,
        duration_seconds=1.0,
        details={},
    )


def _make_loop_plan(items=None, loop_over=None, tool_calls=None, exit_criteria=None):
    step_dict: dict = {
        "id": "5",
        "type": "analyze",
        "desc": "Review each workspace member",
        "tools": ["read", "grep"],
        "deps": [],
        "exec": "loop",
    }
    if items is not None:
        step_dict["items"] = items
    if loop_over is not None:
        step_dict["loop_over"] = loop_over
    if tool_calls is not None:
        step_dict["tool_calls"] = tool_calls
    if exit_criteria is not None:
        step_dict["exit"] = exit_criteria

    return ReadableTaskPlan(
        name="Loop test",
        complexity=TaskComplexity.COMPLEX,
        desc="Loop over crates",
        steps=[step_dict],
    )


@pytest.mark.asyncio
async def test_loop_node_iterates_over_static_items():
    """Loop node spawns one subagent per static item, aggregates output."""
    calls = []
    tasks = []

    async def fake_spawn(**kwargs):
        item = kwargs["display_name"].split(": ", 1)[-1]
        calls.append(item)
        tasks.append(kwargs["task"])
        return _spawn_result(item)

    subagents = MagicMock()
    subagents.spawn = AsyncMock(side_effect=fake_spawn)
    adapter = PlanningTeamExecutionAdapter(
        orchestrator=SimpleNamespace(active_session_id="sess"),
        sub_agent_orchestrator=subagents,
    )
    plan = _make_loop_plan(items=["protocol", "state", "tools"])
    execution_plan = plan.to_execution_plan()

    result = await adapter.execute_step(
        plan=plan,
        execution_plan=execution_plan,
        step=execution_plan.steps[0],
        root_session_id="sess",
        plan_state={},
    )

    assert result.success is True
    assert calls == ["protocol", "state", "tools"]
    assert result.metadata["execution_mode"] == "loop_node"
    assert result.metadata["loop_items_count"] == 3
    assert "[protocol]" in result.output
    assert "[state]" in result.output
    assert "[tools]" in result.output
    assert all("Evidence requirements" in task for task in tasks)
    assert all("file:line references" in task for task in tasks)
    assert all("do not stop after inventory" in task for task in tasks)


@pytest.mark.asyncio
async def test_loop_node_resolves_items_from_plan_state():
    """Loop node reads item list from plan_state when loop_over is set."""
    calls = []

    async def fake_spawn(**kwargs):
        item = kwargs["display_name"].split(": ", 1)[-1]
        calls.append(item)
        return _spawn_result(item)

    subagents = MagicMock()
    subagents.spawn = AsyncMock(side_effect=fake_spawn)
    adapter = PlanningTeamExecutionAdapter(
        orchestrator=SimpleNamespace(active_session_id="sess"),
        sub_agent_orchestrator=subagents,
    )
    plan = _make_loop_plan(loop_over="workspace_members")
    execution_plan = plan.to_execution_plan()

    plan_state = {"workspace_members": ["alpha", "beta"]}
    result = await adapter.execute_step(
        plan=plan,
        execution_plan=execution_plan,
        step=execution_plan.steps[0],
        root_session_id="sess",
        plan_state=plan_state,
    )

    assert result.success is True
    assert calls == ["alpha", "beta"]
    assert result.metadata["loop_items_count"] == 2


@pytest.mark.asyncio
async def test_loop_node_no_items_returns_success_with_skip_message():
    """Loop node with no items or empty plan_state succeeds with a skip notice."""
    subagents = MagicMock()
    subagents.spawn = AsyncMock()
    adapter = PlanningTeamExecutionAdapter(
        orchestrator=SimpleNamespace(active_session_id="sess"),
        sub_agent_orchestrator=subagents,
    )
    plan = _make_loop_plan(loop_over="missing_key")
    execution_plan = plan.to_execution_plan()

    result = await adapter.execute_step(
        plan=plan,
        execution_plan=execution_plan,
        step=execution_plan.steps[0],
        root_session_id="sess",
        plan_state={},
    )

    assert result.success is True
    assert "no items" in result.output.lower()
    subagents.spawn.assert_not_awaited()


@pytest.mark.asyncio
async def test_loop_node_uses_15_tool_calls_per_iteration_by_default():
    """Loop iterations must use 15 tool_budget by default (not 10) for adequate per-crate coverage."""
    spawn_budgets = []

    async def fake_spawn(**kwargs):
        spawn_budgets.append(kwargs.get("tool_budget"))
        item = kwargs["display_name"].split(": ", 1)[-1]
        return _spawn_result(item)

    subagents = MagicMock()
    subagents.spawn = AsyncMock(side_effect=fake_spawn)
    adapter = PlanningTeamExecutionAdapter(
        orchestrator=SimpleNamespace(active_session_id="sess"),
        sub_agent_orchestrator=subagents,
    )
    plan = _make_loop_plan(items=["protocol", "state", "tools"])
    execution_plan = plan.to_execution_plan()

    await adapter.execute_step(
        plan=plan,
        execution_plan=execution_plan,
        step=execution_plan.steps[0],
        root_session_id="sess",
        plan_state={},
    )

    assert spawn_budgets == [
        15,
        15,
        15,
    ], f"Each loop iteration must use 15 tool_budget, got {spawn_budgets}"


@pytest.mark.asyncio
async def test_loop_node_explicit_tool_calls_overrides_default():
    """Explicit tool_calls on a loop step overrides the 15 default."""
    spawn_budgets = []

    async def fake_spawn(**kwargs):
        spawn_budgets.append(kwargs.get("tool_budget"))
        item = kwargs["display_name"].split(": ", 1)[-1]
        return _spawn_result(item)

    subagents = MagicMock()
    subagents.spawn = AsyncMock(side_effect=fake_spawn)
    adapter = PlanningTeamExecutionAdapter(
        orchestrator=SimpleNamespace(active_session_id="sess"),
        sub_agent_orchestrator=subagents,
    )
    plan = _make_loop_plan(items=["alpha", "beta"], tool_calls=25)
    execution_plan = plan.to_execution_plan()

    await adapter.execute_step(
        plan=plan,
        execution_plan=execution_plan,
        step=execution_plan.steps[0],
        root_session_id="sess",
        plan_state={},
    )

    assert spawn_budgets == [
        25,
        25,
    ], f"Explicit tool_calls=25 must be used as per-iteration budget, got {spawn_budgets}"


@pytest.mark.asyncio
async def test_loop_node_appends_exit_criteria_to_item_task():
    """Loop iteration task must include the step's exit criteria so the sub-agent
    knows it cannot complete until those criteria are satisfied.

    Root cause: sub-agents terminated after 3 tool calls (ls + 2 reads) even with
    budget=25 because they had no explicit completion requirements. Appending exit
    criteria to the task string provides the sub-agent with gating conditions.
    """
    spawned_tasks = []

    async def fake_spawn(**kwargs):
        spawned_tasks.append(kwargs.get("task", ""))
        item = kwargs["display_name"].split(": ", 1)[-1]
        return _spawn_result(item)

    subagents = MagicMock()
    subagents.spawn = AsyncMock(side_effect=fake_spawn)
    adapter = PlanningTeamExecutionAdapter(
        orchestrator=SimpleNamespace(active_session_id="sess"),
        sub_agent_orchestrator=subagents,
    )
    plan = _make_loop_plan(
        items=["protocol", "state"],
        exit_criteria=["grep for Arc<T> patterns", "record findings per crate"],
    )
    execution_plan = plan.to_execution_plan()

    await adapter.execute_step(
        plan=plan,
        execution_plan=execution_plan,
        step=execution_plan.steps[0],
        root_session_id="sess",
        plan_state={},
    )

    assert len(spawned_tasks) == 2
    for task in spawned_tasks:
        assert (
            "grep for Arc<T> patterns" in task
        ), f"Exit criteria must appear in iteration task; got: {task!r}"
        assert (
            "record findings per crate" in task
        ), f"Exit criteria must appear in iteration task; got: {task!r}"


@pytest.mark.asyncio
async def test_loop_node_no_exit_criteria_task_unchanged():
    """If the step has no exit criteria, the iteration task should not be modified."""
    spawned_tasks = []

    async def fake_spawn(**kwargs):
        spawned_tasks.append(kwargs.get("task", ""))
        item = kwargs["display_name"].split(": ", 1)[-1]
        return _spawn_result(item)

    subagents = MagicMock()
    subagents.spawn = AsyncMock(side_effect=fake_spawn)
    adapter = PlanningTeamExecutionAdapter(
        orchestrator=SimpleNamespace(active_session_id="sess"),
        sub_agent_orchestrator=subagents,
    )
    plan = _make_loop_plan(items=["protocol"])
    execution_plan = plan.to_execution_plan()

    await adapter.execute_step(
        plan=plan,
        execution_plan=execution_plan,
        step=execution_plan.steps[0],
        root_session_id="sess",
        plan_state={},
    )

    assert len(spawned_tasks) == 1
    assert (
        "criteria" not in spawned_tasks[0].lower()
    ), f"Task with no exit criteria should not mention criteria; got: {spawned_tasks[0]!r}"


@pytest.mark.asyncio
async def test_loop_node_partial_failure_marks_result_failed():
    """Loop node is failed when any item fails; failed items listed in error."""

    async def fake_spawn(**kwargs):
        item = kwargs["display_name"].split(": ", 1)[-1]
        return _spawn_result(item, success=(item != "beta"))

    subagents = MagicMock()
    subagents.spawn = AsyncMock(side_effect=fake_spawn)
    adapter = PlanningTeamExecutionAdapter(
        orchestrator=SimpleNamespace(active_session_id="sess"),
        sub_agent_orchestrator=subagents,
    )
    plan = _make_loop_plan(items=["alpha", "beta", "gamma"])
    execution_plan = plan.to_execution_plan()

    result = await adapter.execute_step(
        plan=plan,
        execution_plan=execution_plan,
        step=execution_plan.steps[0],
        plan_state={},
    )

    assert result.success is False
    assert "beta" in result.metadata["failed_items"]
    assert result.metadata["loop_items_count"] == 3


# ---------------------------------------------------------------------------
# Plan state / produces key tests
# ---------------------------------------------------------------------------


def test_resolve_loop_items_falls_back_to_newline_string():
    """When plan_state value is a raw string, split it into lines."""
    from victor.agent.planning.base import PlanStep, StepType

    step = PlanStep(
        id="5",
        description="loop step",
        step_type=StepType.RESEARCH,
        execution="loop",
        context={"loop_over": "members"},
    )
    plan_state = {"members": "protocol\nstate\ntools"}
    items = PlanningTeamExecutionAdapter._resolve_loop_items(step, plan_state)
    assert items == ["protocol", "state", "tools"]


def test_step_dict_with_produces_and_loop_over_parsed_into_context():
    """Rich dict steps carry loop_over and produces through to PlanStep.context."""
    plan = ReadableTaskPlan(
        name="state passing",
        complexity=TaskComplexity.COMPLEX,
        desc="state test",
        steps=[
            {
                "id": 2,
                "type": "analyze",
                "desc": "Discover crates",
                "tools": ["shell"],
                "deps": [],
                "exec": "tool",
                "produces": "workspace_members",
            },
            {
                "id": 3,
                "type": "analyze",
                "desc": "Review each crate",
                "tools": ["grep"],
                "deps": [2],
                "exec": "loop",
                "loop_over": "workspace_members",
            },
        ],
    )
    ep = plan.to_execution_plan()
    producer = ep.steps[0]
    looper = ep.steps[1]

    assert producer.context["produces"] == "workspace_members"
    assert producer.execution == "tool"
    assert looper.context["loop_over"] == "workspace_members"
    assert looper.execution == "loop"


# ---------------------------------------------------------------------------
# Conditional node tests
# ---------------------------------------------------------------------------


def _conditional_step(condition_on, condition="non_empty", branches=None, produces=""):
    from victor.agent.planning.base import PlanStep, StepType

    ctx = {
        "condition_on": condition_on,
        "condition": condition,
        "execution": "conditional",
    }
    if branches:
        ctx["branches"] = branches
    if produces:
        ctx["produces"] = produces
    return PlanStep(
        id="cond",
        description="conditional test step",
        step_type=StepType.RESEARCH,
        execution="conditional",
        context=ctx,
    )


def test_conditional_node_non_empty_true():
    # branches["true"] = steps to RUN when True → skip branches["false"] when True
    adapter = PlanningTeamExecutionAdapter(orchestrator=SimpleNamespace())
    step = _conditional_step(
        "members", "non_empty", branches={"true": ["yes_step"], "false": ["7"]}
    )
    result = adapter._execute_conditional_node(step, {"members": ["a", "b"]})

    assert result.success is True
    assert result.metadata["condition_result"] is True
    assert result.metadata["active_branch"] == "true"
    assert result.metadata["skip_step_ids"] == ["7"]  # skip the "false" branch


def test_conditional_node_non_empty_false_skips_true_branch():
    adapter = PlanningTeamExecutionAdapter(orchestrator=SimpleNamespace())
    # branches["true"] = loop step; when condition is False, skip "true" branch (loop)
    step = _conditional_step("members", "non_empty", branches={"true": ["6a"], "false": ["6b"]})
    result = adapter._execute_conditional_node(step, {"members": []})

    assert result.metadata["condition_result"] is False
    assert result.metadata["active_branch"] == "false"
    assert result.metadata["skip_step_ids"] == ["6a"]  # skip the "true" branch


def test_conditional_node_multiple_condition():
    adapter = PlanningTeamExecutionAdapter(orchestrator=SimpleNamespace())
    # True (multiple crates) → run loop "6a", skip single-agent "6b"
    # branches["true"] = loop step "6a"; branches["false"] = single-agent "6b"
    # When True: inactive="false" → skip branches["false"] = ["6b"] ✓
    step = _conditional_step(
        "crates",
        "multiple",
        branches={"true": ["6a"], "false": ["6b"]},
        produces="is_workspace",
    )
    plan_state: dict = {"crates": ["alpha", "beta", "gamma"]}
    result = adapter._execute_conditional_node(step, plan_state)

    assert result.metadata["condition_result"] is True
    assert result.metadata["skip_step_ids"] == ["6b"]  # skip single-agent when multi-crate
    assert plan_state["is_workspace"] is True


def test_conditional_node_single_item_multiple_false():
    adapter = PlanningTeamExecutionAdapter(orchestrator=SimpleNamespace())
    # branches["true"] = loop "6a"; branches["false"] = single-agent "6b"
    # single item → multiple=False → inactive="true" → skip ["6a"] (loop)
    step = _conditional_step("crates", "multiple", branches={"true": ["6a"], "false": ["6b"]})
    result = adapter._execute_conditional_node(step, {"crates": ["only_one"]})

    assert result.metadata["condition_result"] is False
    assert result.metadata["skip_step_ids"] == ["6a"]  # skip loop when single crate


def test_conditional_node_applies_skip_via_runtime(tmp_path):
    """_skip_specific_steps marks PENDING steps SKIPPED; already-running steps are unaffected."""
    from victor.agent.planning.base import PlanStep, StepStatus, StepType, ExecutionPlan
    from victor.agent.services.planning_runtime import PlanningRuntimeService

    steps = [
        PlanStep(id="loop_step", description="loop", step_type=StepType.RESEARCH),
        PlanStep(id="single_step", description="single", step_type=StepType.RESEARCH),
        PlanStep(
            id="synth_step",
            description="synth",
            step_type=StepType.RESEARCH,
            status=StepStatus.IN_PROGRESS,
        ),
    ]
    plan = ExecutionPlan(id="p", goal="test", steps=steps)

    PlanningRuntimeService._skip_specific_steps(plan, ["loop_step"])

    assert steps[0].status == StepStatus.SKIPPED
    assert steps[1].status == StepStatus.PENDING
    assert steps[2].status == StepStatus.IN_PROGRESS  # not touched


def test_conditional_step_dict_parses_branches_and_condition_on():
    plan = ReadableTaskPlan(
        name="cond test",
        complexity=TaskComplexity.COMPLEX,
        desc="branch routing",
        steps=[
            {
                "id": 5,
                "type": "analyze",
                "desc": "Route by workspace size",
                "deps": [],
                "exec": "conditional",
                "condition_on": "workspace_members",
                "condition": "multiple",
                "produces": "is_workspace",
                "branches": {"true": ["6b"], "false": ["6a"]},
            }
        ],
    )
    ep = plan.to_execution_plan()
    step = ep.steps[0]

    assert step.execution == "conditional"
    assert step.context["condition_on"] == "workspace_members"
    assert step.context["condition"] == "multiple"
    assert step.context["produces"] == "is_workspace"
    assert step.context["branches"] == {"true": ["6b"], "false": ["6a"]}


# ---------------------------------------------------------------------------
# Exit criteria enforcement tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_loop_node_stops_early_when_exit_criteria_met():
    """Sequential loop stops after the first iteration that satisfies exit_criteria."""
    calls = []

    async def fake_spawn(**kwargs):
        item = kwargs["display_name"].split(": ", 1)[-1]
        calls.append(item)
        summary = "done reviewing" if item == "beta" else "still working"
        return _spawn_result(item)

    subagents = MagicMock()
    subagents.spawn = AsyncMock(side_effect=fake_spawn)
    adapter = PlanningTeamExecutionAdapter(
        orchestrator=SimpleNamespace(active_session_id="sess"),
        sub_agent_orchestrator=subagents,
    )

    from victor.agent.planning.base import PlanStep, StepType

    step = PlanStep(
        id="L",
        description="review each crate",
        step_type=StepType.RESEARCH,
        execution="loop",
        context={"items": ["alpha", "beta", "gamma"]},
        exit_criteria=["reviewed alpha"],  # satisfied after first item's output
    )
    plan = _make_loop_plan(items=["alpha", "beta", "gamma"])
    ep = plan.to_execution_plan()
    # Inject exit_criteria into the step
    ep.steps[0].exit_criteria = ["Reviewed alpha"]

    result = await adapter.execute_step(
        plan=plan,
        execution_plan=ep,
        step=ep.steps[0],
        plan_state={},
    )

    assert result.success is True
    assert result.metadata["early_stopped"] is True
    assert len(calls) == 1  # stopped after "alpha"


@pytest.mark.asyncio
async def test_loop_node_no_early_stop_without_exit_criteria():
    """Loop runs all items when no exit_criteria are set."""
    calls = []

    async def fake_spawn(**kwargs):
        item = kwargs["display_name"].split(": ", 1)[-1]
        calls.append(item)
        return _spawn_result(item)

    subagents = MagicMock()
    subagents.spawn = AsyncMock(side_effect=fake_spawn)
    adapter = PlanningTeamExecutionAdapter(
        orchestrator=SimpleNamespace(active_session_id="sess"),
        sub_agent_orchestrator=subagents,
    )
    plan = _make_loop_plan(items=["x", "y", "z"])
    ep = plan.to_execution_plan()

    result = await adapter.execute_step(
        plan=plan, execution_plan=ep, step=ep.steps[0], plan_state={}
    )
    assert calls == ["x", "y", "z"]
    assert result.metadata.get("early_stopped") is False


# ---------------------------------------------------------------------------
# Approval node tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_approval_node_auto_approves_when_no_callback():
    """Approval node succeeds immediately (auto-approve) when no callback is set."""
    from victor.agent.planning.base import PlanStep, StepType

    adapter = PlanningTeamExecutionAdapter(orchestrator=SimpleNamespace())
    step = PlanStep(
        id="A",
        description="Review findings before applying fixes",
        step_type=StepType.REVIEW,
        execution="approval",
    )
    plan = ReadableTaskPlan(
        name="ap",
        complexity=TaskComplexity.SIMPLE,
        desc="d",
        steps=[["A", "review", "Review findings before applying fixes", ""]],
    )
    ep = plan.to_execution_plan()

    result = await adapter._execute_approval_node(step, {})

    assert result.success is True
    assert result.metadata["approved"] is True
    assert result.metadata["auto_approved"] is True


@pytest.mark.asyncio
async def test_approval_node_calls_callback_and_uses_decision():
    """Approval node respects callback decision: rejected → failure."""
    from victor.agent.planning.base import PlanStep, StepType

    async def reject_callback(step, context):
        return False, "Not ready yet"

    adapter = PlanningTeamExecutionAdapter(
        orchestrator=SimpleNamespace(),
        approval_callback=reject_callback,
    )
    step = PlanStep(
        id="A",
        description="Apply patches",
        step_type=StepType.REVIEW,
        execution="approval",
    )
    result = await adapter._execute_approval_node(step, {})

    assert result.success is False
    assert result.metadata["approved"] is False
    assert "Not ready yet" in (result.metadata["feedback"] or "")


@pytest.mark.asyncio
async def test_approval_node_approved_via_callback():
    from victor.agent.planning.base import PlanStep, StepType

    async def approve_callback(step, context):
        return True, "Looks good"

    adapter = PlanningTeamExecutionAdapter(
        orchestrator=SimpleNamespace(),
        approval_callback=approve_callback,
    )
    step = PlanStep(
        id="A",
        description="Deploy",
        step_type=StepType.DEPLOYMENT,
        execution="approval",
    )
    result = await adapter._execute_approval_node(step, {})

    assert result.success is True
    assert result.metadata["approved"] is True
    assert "Looks good" in result.output


# ---------------------------------------------------------------------------
# Fallback resolution tests
# ---------------------------------------------------------------------------


def test_conditional_node_falls_back_to_word_overlap_when_exact_key_missing():
    """When condition_on key is absent, the node searches plan_state by word overlap."""
    from types import SimpleNamespace
    from victor.agent.planning.base import PlanStep, StepType

    adapter = PlanningTeamExecutionAdapter(orchestrator=SimpleNamespace())
    step = PlanStep(
        id="7",
        description="Route: multi vs single",
        step_type=StepType.RESEARCH,
        execution="conditional",
        context={
            "condition_on": "workspace_members",  # key is absent from plan_state
            "condition": "multiple",
            "branches": {"true": ["8a"], "false": ["8b"]},
        },
    )
    # plan_state uses a different key but with overlapping words
    plan_state = {"workspace_member_crates": ["crates/protocol", "crates/state", "crates/tools"]}

    result = adapter._execute_conditional_node(step, plan_state)

    # Should resolve the list via fallback and evaluate True (multiple items)
    assert result.success is True
    assert result.metadata["condition_result"] is True
    assert "8b" in result.metadata["skip_step_ids"]  # false branch skipped


def test_resolve_loop_items_falls_back_when_key_name_differs():
    """_resolve_loop_items resolves via word-overlap when loop_over key is absent."""
    from types import SimpleNamespace
    from victor.agent.planning.base import PlanStep, StepType

    adapter = PlanningTeamExecutionAdapter(orchestrator=SimpleNamespace())
    step = PlanStep(
        id="8a",
        description="Loop over each workspace member crate",
        step_type=StepType.RESEARCH,
        execution="loop",
        context={"loop_over": "workspace_members"},  # exact key absent
    )
    # plan_state uses a slightly different key
    plan_state = {"workspace_member_crates": ["crates/protocol", "crates/state"]}

    items = adapter._resolve_loop_items(step, plan_state)
    assert items == ["crates/protocol", "crates/state"]


# ---------------------------------------------------------------------------
# Regression: _task_description_for_step output-format injection
# ---------------------------------------------------------------------------


def test_task_description_for_step_appends_format_instruction_when_produces_key_present():
    """Regression: steps with a 'produces' key must carry explicit output-format instructions.

    Bug: LLM sub-agents for list-producing steps returned prose summaries instead of
    structured lists, causing _extract_list_from_output to return [] due to its prose
    guard, which in turn left plan_state unpopulated and broke conditional routing.
    Fix: _task_description_for_step appends a required output-format block whenever
    the step carries a 'produces' context key.
    """
    from victor.agent.planning.base import PlanStep, StepType

    step = PlanStep(
        id="2",
        description="Discover Rust workspace members",
        step_type=StepType.RESEARCH,
        context={"produces": "workspace_members"},
    )

    task = PlanningTeamExecutionAdapter._task_description_for_step(step)

    assert task.startswith("Discover Rust workspace members")
    assert "OUTPUT FORMAT" in task
    assert "workspace_members" in task
    assert "(none)" in task


def test_task_description_for_step_uses_artifact_contract_for_checklist_produces():
    """Checklist/report/findings producers must return content, not progress narration."""
    from victor.agent.planning.base import PlanStep, StepType

    step = PlanStep(
        id="4",
        description="Build comprehensive Rust best practices checklist",
        step_type=StepType.RESEARCH,
        execution="compute",
        context={"produces": "best_practices_checklist"},
    )

    task = PlanningTeamExecutionAdapter._task_description_for_step(step)

    assert "OUTPUT FORMAT" in task
    assert "best_practices_checklist" in task
    assert "as Markdown" in task
    assert "concrete artifact" in task
    assert "Do not return a status update" in task
    assert "plain list" not in task
    assert "knowledge generation step" in task


def test_task_description_for_step_adds_evidence_guidance_for_review_without_produces_key():
    """Review steps get evidence guidance even when they do not write plan_state."""
    from victor.agent.planning.base import PlanStep, StepType

    step = PlanStep(
        id="3",
        description="Review Arc usage across all crates",
        step_type=StepType.RESEARCH,
        context={},
    )

    task = PlanningTeamExecutionAdapter._task_description_for_step(step)

    assert task.startswith("Review Arc usage across all crates")
    assert "OUTPUT FORMAT" not in task
    assert "Evidence requirements" in task
    assert "read the relevant source files" in task
    assert "file:line references" in task
    assert "do not stop after inventory" in task


def test_task_description_for_step_does_not_add_evidence_guidance_for_inventory_step():
    """Inventory/discovery steps can legitimately complete from directory listings."""
    from victor.agent.planning.base import PlanStep, StepType

    step = PlanStep(
        id="3",
        description="Map file inventory",
        step_type=StepType.RESEARCH,
        context={},
        exit_criteria=[],
    )

    task = PlanningTeamExecutionAdapter._task_description_for_step(step)

    assert task == "Map file inventory"
    assert "Evidence requirements" not in task


@pytest.mark.asyncio
async def test_execute_step_passes_format_instruction_to_subagent_for_produces_step():
    """The format instruction reaches the sub-agent spawn call when produces is set."""
    captured_task: list[str] = []

    async def fake_spawn(**kwargs):
        captured_task.append(kwargs.get("task", ""))
        return SimpleNamespace(
            success=True,
            summary="crates/protocol\ncrates/state\ncrates/tools",
            error=None,
            details={},
            tool_calls_used=1,
            duration_seconds=0.1,
        )

    subagents = MagicMock()
    subagents.spawn = AsyncMock(side_effect=fake_spawn)
    adapter = PlanningTeamExecutionAdapter(
        orchestrator=SimpleNamespace(active_session_id="sess"),
        sub_agent_orchestrator=subagents,
    )
    plan = ReadableTaskPlan(
        name="Workspace discovery",
        complexity=TaskComplexity.COMPLEX,
        desc="Discover crates",
        steps=[
            {
                "id": "2",
                "type": "analyze",
                "desc": "Discover Rust workspace members",
                "tools": ["shell"],
                "deps": [],
                "exec": "tool",
                "produces": "workspace_members",
            }
        ],
    )
    execution_plan = plan.to_execution_plan()

    await adapter.execute_step(
        plan=plan,
        execution_plan=execution_plan,
        step=execution_plan.steps[0],
        root_session_id="sess",
    )

    assert len(captured_task) == 1
    assert "OUTPUT FORMAT" in captured_task[0]
    assert "workspace_members" in captured_task[0]
    assert "(none)" in captured_task[0]


def test_task_description_injects_plan_state_list_context():
    """plan_state list values appear as 'Context from prior steps' in the task."""
    from victor.agent.planning.base import PlanStep, StepType

    step = PlanStep(
        id="3",
        description="Inventory Rust source files in each crate",
        step_type=StepType.RESEARCH,
        context={"produces": "crate_file_inventory"},
    )
    plan_state = {
        "workspace_members": [
            "rust/crates/protocol",
            "rust/crates/state",
            "rust/crates/tools",
            "rust/crates/edge-runtime",
            "rust/crates/python-bindings",
        ]
    }

    task = PlanningTeamExecutionAdapter._task_description_for_step(step, plan_state)

    assert "Context from prior steps" in task
    assert "workspace_members: rust/crates/protocol" in task
    assert "rust/crates/edge-runtime" in task
    assert "OUTPUT FORMAT" in task
    assert task.index("Context from prior steps") < task.index("OUTPUT FORMAT")


def test_task_description_excludes_raw_step_dumps_from_plan_state():
    """step_N keys (raw step output text) must not bleed into the task description."""
    from victor.agent.planning.base import PlanStep, StepType

    step = PlanStep(
        id="4",
        description="Analyze crate dependencies",
        step_type=StepType.RESEARCH,
        context={},
    )
    plan_state = {
        "step_1": "verified git repo at /Users/vijaysingh/code/codingagent",
        "step_2": "rust/crates/protocol\nrust/crates/state",
        "workspace_members": ["rust/crates/protocol", "rust/crates/state"],
    }

    task = PlanningTeamExecutionAdapter._task_description_for_step(step, plan_state)

    assert "step_1" not in task
    assert "step_2" not in task
    assert "workspace_members" in task


def test_task_description_discloses_partial_failed_dependencies():
    """Continuable partial steps must tell the worker what coverage is missing."""
    from victor.agent.planning.base import PlanStep, StepType

    step = PlanStep(
        id="8",
        description="Cross-crate analysis of Arc usage",
        step_type=StepType.RESEARCH,
        context={"partial_failed_dependencies": ["7a", "7a", "7b"]},
    )

    task = PlanningTeamExecutionAdapter._task_description_for_step(step, {})

    assert "Partial execution context" in task
    assert "7a, 7b" in task
    assert "do not invent missing findings" in task


def test_task_description_unchanged_when_plan_state_empty():
    """An empty plan_state dict must not alter the task description."""
    from victor.agent.planning.base import PlanStep, StepType

    step = PlanStep(
        id="2",
        description="Discover workspace members",
        step_type=StepType.RESEARCH,
        context={"produces": "workspace_members"},
    )

    task_without = PlanningTeamExecutionAdapter._task_description_for_step(step)
    task_with_empty = PlanningTeamExecutionAdapter._task_description_for_step(step, {})

    assert task_without == task_with_empty


@pytest.mark.asyncio
async def test_execute_step_tool_node_injects_plan_state_into_task():
    """Tool node path must forward plan_state context to the sub-agent task string."""
    captured_task: list[str] = []

    async def fake_spawn(**kwargs):
        captured_task.append(kwargs.get("task", ""))
        return SimpleNamespace(
            success=True,
            summary="rust/crates/protocol/src/lib.rs\nrust/crates/state/src/lib.rs",
            error=None,
            details={},
            tool_calls_used=5,
            duration_seconds=1.0,
        )

    subagents = MagicMock()
    subagents.spawn = AsyncMock(side_effect=fake_spawn)
    adapter = PlanningTeamExecutionAdapter(
        orchestrator=SimpleNamespace(active_session_id="sess"),
        sub_agent_orchestrator=subagents,
    )
    plan = ReadableTaskPlan(
        name="Crate inventory",
        complexity=TaskComplexity.COMPLEX,
        desc="Map Rust crate files",
        steps=[
            {
                "id": "3",
                "type": "analyze",
                "desc": "Inventory all Rust source files in each workspace crate",
                "tools": ["shell"],
                "deps": [],
                "exec": "tool",
                "produces": "crate_file_inventory",
            }
        ],
    )
    execution_plan = plan.to_execution_plan()

    await adapter.execute_step(
        plan=plan,
        execution_plan=execution_plan,
        step=execution_plan.steps[0],
        root_session_id="sess",
        plan_state={
            "workspace_members": [
                "rust/crates/protocol",
                "rust/crates/state",
                "rust/crates/tools",
            ]
        },
    )

    assert len(captured_task) == 1
    assert "Context from prior steps" in captured_task[0]
    assert "workspace_members" in captured_task[0]
    assert "rust/crates/protocol" in captured_task[0]
    assert "OUTPUT FORMAT" in captured_task[0]


def test_synthesis_step_includes_step_n_content_in_task():
    """Regression: synthesis/write steps need prior analysis content injected.

    Bug: step 9 (doc/synthesize) got a fresh sub-agent with no memory of steps 7a/8
    analysis findings. Without step_N content injected, it used all 10 tool budget
    re-reading files and produced only 95 chars output → evidence contract FAIL.
    Fix: for synthesis steps (write in allowed_tools or synthesis keywords in desc),
    step_N keys are also injected, truncated to 600 chars each.
    """
    from victor.agent.planning.base import PlanStep, StepType

    step = PlanStep(
        id="9",
        description="Synthesize all findings into a prioritized report",
        step_type=StepType.RESEARCH,
        allowed_tools=["write"],
        context={},
    )
    plan_state = {
        "workspace_members": ["rust/crates/protocol", "rust/crates/state"],
        "step_7a": "Arc usage: protocol crate uses Arc<Config> in 3 places. Unnecessary cloning at line 45.",
        "step_8": "Cross-crate: state and tools both use Arc<Mutex<T>> pattern, could share a helper.",
    }

    task = PlanningTeamExecutionAdapter._task_description_for_step(step, plan_state)

    assert "Context from prior steps" in task
    assert "workspace_members" in task
    # step_N keys must be present for synthesis steps
    assert "step_7a" in task
    assert "Arc usage: protocol crate" in task
    assert "step_8" in task
    assert "Cross-crate:" in task
    # No output-format block (no produces key)
    assert "OUTPUT FORMAT" not in task


def test_non_synthesis_step_does_not_include_step_n_content():
    """Non-synthesis steps must NOT get step_N raw dumps to avoid noise."""
    from victor.agent.planning.base import PlanStep, StepType

    step = PlanStep(
        id="3",
        description="Inventory Rust source files",
        step_type=StepType.RESEARCH,
        allowed_tools=["shell", "read"],
        context={"produces": "crate_file_inventory"},
    )
    plan_state = {
        "workspace_members": ["rust/crates/protocol", "rust/crates/state"],
        "step_1": "verified git repo at /Users/vijaysingh/code/codingagent",
        "step_2": "rust/crates/protocol\nrust/crates/state",
    }

    task = PlanningTeamExecutionAdapter._task_description_for_step(step, plan_state)

    assert "workspace_members" in task
    # step_N keys must NOT be present for non-synthesis steps
    assert "step_1" not in task
    assert "step_2" not in task


def test_inferred_inputs_inject_only_declared_keys():
    """When inputs are inferred via _enrich_step_dicts, only those keys are injected."""
    from victor.agent.planning.base import PlanStep, StepType

    # Simulate a plan where step 3 consumes workspace_members (inferred via description)
    step = PlanStep(
        id="3",
        description="Map module tree for each workspace member",
        step_type=StepType.RESEARCH,
        allowed_tools=["shell"],
        context={"produces": "module_tree"},
        inputs=["workspace_members"],  # set as if inferred by _enrich_step_dicts
    )
    plan_state = {
        "workspace_members": ["rust/crates/protocol", "rust/crates/state"],
        "other_key": "irrelevant value",
        "step_1": "raw output from step 1",
    }

    task = PlanningTeamExecutionAdapter._task_description_for_step(step, plan_state)

    assert "Context from prior steps" in task
    assert "workspace_members" in task
    assert "rust/crates/protocol" in task
    # other keys must NOT be injected (precise routing)
    assert "other_key" not in task
    assert "step_1" not in task
    assert "irrelevant value" not in task


def test_enrich_step_dicts_infers_inputs_from_produces_keywords():
    """Pass 5 of _enrich_step_dicts infers inputs by matching produces keys against descriptions."""
    plan = ReadableTaskPlan(
        name="Rust review",
        complexity=TaskComplexity.COMPLEX,
        desc="Review Rust codebase",
        steps=[
            {
                "id": "2",
                "type": "analyze",
                "desc": "Inventory workspace members",
                "produces": "workspace_members",
            },
            {
                "id": "3",
                "type": "analyze",
                "desc": "Map module tree for each workspace member",
                "tools": ["shell"],
            },
            {
                "id": "9",
                "type": "doc",
                "desc": "Synthesize findings into report",
                "tools": ["write"],
            },
        ],
    )
    execution_plan = plan.to_execution_plan()
    step2, step3, step9 = execution_plan.steps

    # Step 2 produces workspace_members, step 3's description contains "workspace member"
    assert (
        "workspace_members" in step3.inputs
    ), f"Expected inputs inferred for step 3, got {step3.inputs}"
    # Step 9 is synthesis — no auto-inferred inputs from keyword matching (synthesis uses fallback)
    # Step 2 must not have workspace_members as input (it produces it)
    assert "workspace_members" not in step2.inputs


def test_synthesis_step_truncates_long_step_n_content():
    """Long step_N values are truncated to 600 chars to avoid overwhelming the context."""
    from victor.agent.planning.base import PlanStep, StepType

    step = PlanStep(
        id="9",
        description="Synthesize findings",
        step_type=StepType.RESEARCH,
        allowed_tools=["write"],
        context={},
    )
    long_analysis = "A" * 2000  # much longer than 600-char limit
    plan_state = {"step_7a": long_analysis}

    task = PlanningTeamExecutionAdapter._task_description_for_step(step, plan_state)

    # Should be truncated — step_7a appears but capped
    assert "step_7a" in task
    assert long_analysis not in task  # full 2000 chars must NOT appear
    # At most 600 chars of the value injected
    idx = task.index("step_7a: ") + len("step_7a: ")
    injected = task[idx : idx + 700]
    assert "A" * 601 not in injected


# ──────────────────────────────────────────────────────────────────────────────
# Pass 6: minimize depends_on for parallel dispatch
# ──────────────────────────────────────────────────────────────────────────────


def test_pass6_trims_sequential_chain_to_parallel_siblings():
    """Steps that share the same data dependency should have deps trimmed to enable parallel dispatch."""
    plan = ReadableTaskPlan(
        name="Parallel test",
        complexity=TaskComplexity.COMPLEX,
        desc="Test parallel planning",
        steps=[
            {
                "id": "1",
                "type": "analyze",
                "desc": "Inventory workspace",
                "produces": "workspace_members",
            },
            # LLM erroneously chains these sequentially
            {
                "id": "2",
                "type": "analyze",
                "desc": "Analyze module A",
                "deps": ["1"],
                "inputs": ["workspace_members"],
                "produces": "findings_A",
            },
            {
                "id": "3",
                "type": "analyze",
                "desc": "Analyze module B",
                "deps": ["2"],
                "inputs": ["workspace_members"],
                "produces": "findings_B",
            },
            {
                "id": "4",
                "type": "analyze",
                "desc": "Analyze module C",
                "deps": ["3"],
                "inputs": ["workspace_members"],
                "produces": "findings_C",
            },
            {
                "id": "5",
                "type": "doc",
                "desc": "Synthesize all findings",
                "deps": ["4"],
                "inputs": ["findings_A", "findings_B", "findings_C"],
                "tools": ["write"],
            },
        ],
    )
    execution_plan = plan.to_execution_plan()
    steps = {s.id: s for s in execution_plan.steps}

    # Steps 2, 3, 4 all declare inputs=['workspace_members'] (produced by step 1).
    # Pass 6 must trim their deps to [1] so they can run in parallel.
    assert steps["2"].depends_on == ["1"], f"step 2 deps={steps['2'].depends_on}"
    assert steps["3"].depends_on == ["1"], f"step 3 deps={steps['3'].depends_on}"
    assert steps["4"].depends_on == ["1"], f"step 4 deps={steps['4'].depends_on}"

    # Step 5 originally has deps=['4'] and inputs=['findings_A','findings_B','findings_C'].
    # Pass 6 is trim-only: it never grows the dep set, so step 5 stays deps=['4'].
    # (Adding steps 2 and 3 would require topology analysis beyond Pass 6's scope.)
    assert (
        "4" in steps["5"].depends_on
    ), f"step 5 must keep dep on step 4; got {steps['5'].depends_on}"
    assert (
        "1" not in steps["5"].depends_on
    ), f"step 5 must not gain dep on step 1; got {steps['5'].depends_on}"


def test_pass6_keeps_control_flow_deps_for_approval_steps():
    """Non-data-producing deps (approval/checkpoint gates) must survive Pass 6 trimming."""
    plan = ReadableTaskPlan(
        name="Approval gate test",
        complexity=TaskComplexity.COMPLEX,
        desc="Test approval gate retained",
        steps=[
            {
                "id": "1",
                "type": "analyze",
                "desc": "Inventory workspace",
                "produces": "workspace_members",
            },
            # Step 2 is an approval gate — it produces nothing named
            {
                "id": "2",
                "type": "review",
                "desc": "Approve analysis scope",
                "deps": ["1"],
                "exec": "approval",
            },
            # Step 3 declares inputs but also must wait for the approval gate
            {
                "id": "3",
                "type": "analyze",
                "desc": "Analyze workspace members",
                "deps": ["1", "2"],
                "inputs": ["workspace_members"],
                "produces": "findings",
            },
        ],
    )
    execution_plan = plan.to_execution_plan()
    steps = {s.id: s for s in execution_plan.steps}

    # Step 3 must retain dep on approval gate "2" even though it doesn't produce a named key.
    assert (
        "1" in steps["3"].depends_on
    ), f"step 3 must still depend on data producer: {steps['3'].depends_on}"
    assert (
        "2" in steps["3"].depends_on
    ), f"step 3 must retain approval gate dep: {steps['3'].depends_on}"


def test_pass6_does_not_touch_steps_without_inputs():
    """Steps that declare no inputs have their deps left unchanged by Pass 6."""
    plan = ReadableTaskPlan(
        name="No-inputs test",
        complexity=TaskComplexity.COMPLEX,
        desc="Test no-inputs steps unchanged",
        steps=[
            {"id": "1", "type": "analyze", "desc": "Read workspace"},
            # Step 2 has explicit deps but no inputs — Pass 6 must not modify it.
            {"id": "2", "type": "analyze", "desc": "Check config", "deps": ["1"]},
        ],
    )
    execution_plan = plan.to_execution_plan()
    steps = {s.id: s for s in execution_plan.steps}

    assert steps["2"].depends_on == ["1"], f"step 2 deps={steps['2'].depends_on}"


@pytest.mark.asyncio
async def test_parallel_steps_dispatched_simultaneously():
    """When multiple steps are ready, they should be dispatched in a single asyncio.gather call."""
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, patch

    from victor.agent.planning.base import StepResult
    from victor.agent.planning.readable_schema import ReadableTaskPlan, TaskComplexity
    from victor.agent.services.planning_runtime import PlanningRuntimeService

    # Build a plan where steps 2, 3, 4 all depend only on step 1
    plan = ReadableTaskPlan(
        name="Parallel dispatch test",
        complexity=TaskComplexity.COMPLEX,
        desc="Verify parallel dispatch",
        steps=[
            {
                "id": "1",
                "type": "analyze",
                "desc": "Inventory workspace",
                "produces": "workspace_members",
            },
            {
                "id": "2",
                "type": "analyze",
                "desc": "Analyze module A",
                "deps": ["1"],
                "inputs": ["workspace_members"],
                "produces": "findings_A",
            },
            {
                "id": "3",
                "type": "analyze",
                "desc": "Analyze module B",
                "deps": ["1"],
                "inputs": ["workspace_members"],
                "produces": "findings_B",
            },
            {
                "id": "4",
                "type": "doc",
                "desc": "Synthesize all findings",
                "deps": ["2", "3"],
                "inputs": ["findings_A", "findings_B"],
                "tools": ["write"],
            },
        ],
    )

    dispatch_batches: list[list[str]] = []

    svc = PlanningRuntimeService(SimpleNamespace(active_session_id="s"))

    async def fake_execute_step(**kwargs) -> StepResult:
        step = kwargs["step"]
        output_map = {
            "1": "workspace/core workspace/util",
            "2": "findings for A",
            "3": "findings for B",
            "4": "synthesized report written",
        }
        return StepResult(
            success=True,
            output=output_map.get(step.id, "done"),
            tool_calls_used=2,
        )

    mock_adapter = MagicMock(spec=PlanningTeamExecutionAdapter)
    mock_adapter.execute_step = AsyncMock(side_effect=fake_execute_step)

    # Capture batch sizes by patching asyncio.gather to record the call count per batch
    original_gather = asyncio.gather
    gather_call_sizes: list[int] = []

    async def recording_gather(*coros, **kwargs):
        gather_call_sizes.append(len(coros))
        return await original_gather(*coros, **kwargs)

    with (
        patch(
            "victor.agent.services.planning_runtime.asyncio.gather",
            side_effect=recording_gather,
        ),
        patch.object(
            svc,
            "_apply_step_evidence_contract",
            side_effect=lambda step, r, *a, **kw: r,
        ),
    ):
        result = await svc._execute_plan_via_team_adapter(plan, mock_adapter)

    # 3 batches: [step 1], [steps 2+3 in parallel], [step 4]
    assert len(gather_call_sizes) == 3, f"Expected 3 gather calls, got {gather_call_sizes}"
    assert gather_call_sizes[0] == 1, "Batch 1: only step 1"
    assert (
        gather_call_sizes[1] == 2
    ), f"Batch 2: steps 2+3 in parallel, got size {gather_call_sizes[1]}"
    assert gather_call_sizes[2] == 1, "Batch 3: step 4 alone"
    assert result.steps_completed == 4


def test_pass6_keeps_conditional_gate_in_branch_deps():
    """A conditional gate that also produces a named output must stay in downstream branch deps.

    This is the critical regression: a conditional node (exec=conditional) that sets
    ``is_multi_crate`` was being treated as a pure data-producer and removed from branch
    deps when the branches didn't consume ``is_multi_crate`` as an input.  Both branches
    then ran in parallel before the conditional fired, producing the wrong routing.
    """
    plan = ReadableTaskPlan(
        name="Conditional gate test",
        complexity=TaskComplexity.COMPLEX,
        desc="Conditional routing test",
        steps=[
            {
                "id": "1",
                "type": "analyze",
                "desc": "Scan repository",
                "produces": "workspace_members",
            },
            {
                "id": "2",
                "type": "analyze",
                "desc": "Parse crates",
                "deps": ["1"],
                "inputs": ["workspace_members"],
                "produces": "crate_list",
            },
            # Conditional gate — produces is_multi_crate but exec=conditional
            {
                "id": "3",
                "type": "compute",
                "desc": "Determine project type",
                "deps": ["2"],
                "exec": "conditional",
                "produces": "is_multi_crate",
            },
            # Branch A and B depend on step 3 but only declare inputs from step 2
            {
                "id": "4",
                "type": "analyze",
                "desc": "Multi-crate analysis",
                "deps": ["3"],
                "inputs": ["crate_list"],
                "produces": "multi_findings",
            },
            {
                "id": "5",
                "type": "analyze",
                "desc": "Single-crate analysis",
                "deps": ["3"],
                "inputs": ["crate_list"],
                "produces": "single_findings",
            },
        ],
    )
    execution_plan = plan.to_execution_plan()
    steps = {s.id: s for s in execution_plan.steps}

    # Steps 4 and 5 must still depend on step 3 (the conditional gate).
    # Pass 6 may trim the data dep on step 2 (since step 3 is the producer of crate_list...
    # wait — crate_list is produced by step 2, so that dep is kept as a data dep).
    # The key assertion: step 3 must NOT be removed from deps of steps 4 and 5.
    assert (
        "3" in steps["4"].depends_on
    ), f"branch A must retain dep on conditional gate 3; got {steps['4'].depends_on}"
    assert (
        "3" in steps["5"].depends_on
    ), f"branch B must retain dep on conditional gate 3; got {steps['5'].depends_on}"


# Pass 6.5: enforce conditional data-flow deps
# ──────────────────────────────────────────────────────────────────────────────


# Self-dep removal at parse time
# ──────────────────────────────────────────────────────────────────────────────


def test_parse_step_removes_self_referential_dep():
    """A step that lists its own ID in deps must have the self-dep stripped.

    Reproduces the production issue (session 89e99624):
      - LLM wrote step 8 with deps=[7, 8]
      - Pass 7 expanded '7' → ['7a','7b'], kept '8'
      - Result: depends_on=['7a','7b','8'] — step 8 depends on itself
      - get_ready_steps() never returned step 8 (unmet dep = '8' → itself)

    The fix strips the self-ref at parse time in _parse_step_dict and _parse_step_list.
    """
    plan = ReadableTaskPlan(
        name="Self-dep removal test",
        complexity=TaskComplexity.COMPLEX,
        desc="Step with self-referential dep",
        steps=[
            {
                "id": "1",
                "type": "analyze",
                "desc": "Step 1",
                "produces": "workspace_members",
            },
            {
                "id": "6",
                "type": "analyze",
                "desc": "Conditional",
                "exec": "conditional",
                "deps": ["1"],
                "condition_on": "workspace_members",
                "branches": {"true": ["7a"], "false": ["7b"]},
            },
            {"id": "7a", "type": "analyze", "desc": "Branch A", "deps": ["6"]},
            {"id": "7b", "type": "analyze", "desc": "Branch B", "deps": ["6"]},
            # Step 8 mistakenly lists its own ID '8' in deps (LLM wrote deps=[7, 8])
            # After Pass 7 expands '7' → ['7a','7b'] the deps would be ['7a','7b','8']
            # The self-ref '8' must be stripped at parse time.
            {
                "id": "8",
                "type": "analyze",
                "desc": "Cross-crate analysis",
                "deps": ["7", "8"],
            },  # '7' → phantom expanded; '8' → self-dep
        ],
    )
    execution_plan = plan.to_execution_plan()
    steps = {s.id: s for s in execution_plan.steps}

    assert (
        "8" not in steps["8"].depends_on
    ), f"Self-dep '8' must be stripped from step 8's depends_on; got {steps['8'].depends_on}"
    # '7' was expanded to ['7a','7b'] by Pass 7
    assert (
        "7a" in steps["8"].depends_on
    ), f"Branch variant '7a' must be in step 8's deps; got {steps['8'].depends_on}"
    assert (
        "7b" in steps["8"].depends_on
    ), f"Branch variant '7b' must be in step 8's deps; got {steps['8'].depends_on}"


def test_parse_step_list_removes_self_referential_dep():
    """Self-dep is also stripped from the compact list step format."""
    plan = ReadableTaskPlan(
        name="List self-dep test",
        complexity=TaskComplexity.SIMPLE,
        desc="List step with self-dep",
        steps=[
            # Compact list format: [id, type, desc, tools, deps]
            ["1", "analyze", "Step 1", "read", []],
            ["2", "analyze", "Step 2 with self-dep", "read", ["1", "2"]],
        ],
    )
    execution_plan = plan.to_execution_plan()
    steps = {s.id: s for s in execution_plan.steps}

    assert (
        "2" not in steps["2"].depends_on
    ), f"Self-dep '2' must be stripped from list step; got {steps['2'].depends_on}"
    assert "1" in steps["2"].depends_on, f"Valid dep '1' must be kept; got {steps['2'].depends_on}"


def test_pass6_5_adds_missing_dep_on_condition_on_producer():
    """A conditional step missing a dep on its condition_on producer gets it added.

    Reproduces the production race condition (session 9eecc53c):
      - Step 2 produces workspace_members
      - Step 6 is conditional on workspace_members but LLM wrote deps=[] (no dep on step 2)
      - Steps 2 and 6 dispatched in same parallel batch
      - workspace_members missing from plan_state → evaluates to False → wrong branch

    Pass 6.5 must add "2" to step 6's deps so it runs AFTER step 2 completes.
    """
    plan = ReadableTaskPlan(
        name="Conditional dep enforcement test",
        complexity=TaskComplexity.COMPLEX,
        desc="Enforce conditional deps",
        steps=[
            {"id": "1", "type": "analyze", "desc": "Read workspace root manifest"},
            # Step 2 produces workspace_members but step 6 has no dep on it
            {
                "id": "2",
                "type": "analyze",
                "desc": "Extract workspace member crates",
                "deps": ["1"],
                "produces": "workspace_members",
            },
            # Conditional with missing dep on step 2 — LLM forgot to declare it
            {
                "id": "6",
                "type": "analyze",
                "desc": "Route strategy based on workspace",
                "exec": "conditional",
                "deps": [],
                "condition_on": "workspace_members",
                "branches": {"true": ["7a"], "false": ["7b"]},
            },
            {"id": "7a", "type": "analyze", "desc": "Multi-crate path", "deps": ["6"]},
            {"id": "7b", "type": "analyze", "desc": "Single-crate path", "deps": ["6"]},
        ],
    )
    execution_plan = plan.to_execution_plan()
    steps = {s.id: s for s in execution_plan.steps}

    assert (
        "2" in steps["6"].depends_on
    ), f"Pass 6.5 must add dep on '2' (produces workspace_members); got {steps['6'].depends_on}"


def test_pass6_5_does_not_add_dep_when_already_present():
    """Pass 6.5 must not add a duplicate dep when it's already there."""
    plan = ReadableTaskPlan(
        name="No duplicate dep test",
        complexity=TaskComplexity.MODERATE,
        desc="No duplicate dep",
        steps=[
            {"id": "1", "type": "analyze", "desc": "Step 1", "produces": "data"},
            {
                "id": "2",
                "type": "analyze",
                "desc": "Conditional",
                "exec": "conditional",
                "deps": ["1"],
                "condition_on": "data",
                "branches": {"true": ["3"], "false": ["4"]},
            },
            {"id": "3", "type": "analyze", "desc": "Branch A", "deps": ["2"]},
            {"id": "4", "type": "analyze", "desc": "Branch B", "deps": ["2"]},
        ],
    )
    execution_plan = plan.to_execution_plan()
    steps = {s.id: s for s in execution_plan.steps}

    # dep on "1" already present — must appear exactly once
    assert (
        steps["2"].depends_on.count("1") == 1
    ), f"Dep '1' must appear exactly once; got {steps['2'].depends_on}"


def test_pass6_5_no_change_when_condition_on_key_not_produced_by_any_step():
    """When no step produces the condition_on key, Pass 6.5 makes no change."""
    plan = ReadableTaskPlan(
        name="Unknown condition_on test",
        complexity=TaskComplexity.SIMPLE,
        desc="No producer for condition key",
        steps=[
            # No step produces 'some_flag' — conditional has no producer to depend on
            {
                "id": "1",
                "type": "analyze",
                "desc": "Conditional on unproduced key",
                "exec": "conditional",
                "deps": [],
                "condition_on": "some_flag",
                "branches": {"true": ["2"], "false": []},
            },
            {"id": "2", "type": "analyze", "desc": "True branch", "deps": ["1"]},
        ],
    )
    execution_plan = plan.to_execution_plan()
    steps = {s.id: s for s in execution_plan.steps}

    # No producer for 'some_flag' → deps stay empty
    assert (
        steps["1"].depends_on == []
    ), f"Deps must remain empty when condition_on has no producer; got {steps['1'].depends_on}"


def test_task_description_knowledge_note_for_compute_step_with_produces():
    """compute steps that produce named output get a 'no tools needed' knowledge note.

    When execution=compute and produces is set but no handler is registered, the step
    falls through to the agent path.  The model may hallucinate tool names (e.g.
    'FirstResponderTool') without an explicit note that this is a knowledge task.
    """
    from victor.agent.planning.base import PlanStep, StepType

    step = PlanStep(
        id="4",
        step_type=StepType.IMPLEMENTATION,
        description="Generate best practices checklist for Rust async code",
        execution="compute",
        context={"produces": "best_practices_checklist"},
    )
    task = PlanningTeamExecutionAdapter._task_description_for_step(step, plan_state={})
    assert "knowledge generation step" in task, f"Expected knowledge note, got: {task[:300]}"
    assert "no tool calls are required" in task.lower() or "no tools" in task.lower()


def test_task_description_knowledge_note_absent_for_regular_agent_step():
    """Non-compute agent steps must not get the knowledge-generation note."""
    from victor.agent.planning.base import PlanStep, StepType

    step = PlanStep(
        id="4",
        step_type=StepType.IMPLEMENTATION,
        description="Read Cargo.toml and extract dependencies",
        execution="agent",
        context={},
    )
    task = PlanningTeamExecutionAdapter._task_description_for_step(step, plan_state={})
    assert "knowledge generation step" not in task


def test_task_description_knowledge_note_absent_for_compute_step_without_produces():
    """compute steps that produce nothing fall back to a generic placeholder — no note needed."""
    from victor.agent.planning.base import PlanStep, StepType

    step = PlanStep(
        id="4",
        step_type=StepType.IMPLEMENTATION,
        description="Validate configuration",
        execution="compute",
        context={},  # no produces key
    )
    task = PlanningTeamExecutionAdapter._task_description_for_step(step, plan_state={})
    assert "knowledge generation step" not in task


def test_pass6_never_adds_deps_not_in_original():
    """Pass 6 must only TRIM deps — it must never ADD new step IDs.

    Regression: when a step declares inputs that are produced by steps NOT already
    in its deps, Pass 6 was adding those producer step IDs to deps.  This made dep
    graphs wider than intended (step 6 deps grew from ['5'] to ['2','3','5']) and
    emitted a misleading "trimmed" log when deps actually grew.

    Example from failing run:
      Step 6 deps=['5'], inputs=['workspace_members','crate_file_inventory','best_practices_checklist']
      workspace_members produced by step 2, crate_file_inventory by step 3 (not in deps)
      Pass 6 wrongly added ['2','3'] → deps=['2','3','5']
    """
    plan = ReadableTaskPlan(
        name="No-dep-addition test",
        complexity=TaskComplexity.COMPLEX,
        desc="Pass 6 must not add deps",
        steps=[
            {
                "id": "1",
                "type": "analyze",
                "desc": "Discover workspace",
                "produces": "workspace_members",
            },
            {
                "id": "2",
                "type": "analyze",
                "desc": "Read file inventory",
                "deps": ["1"],
                "inputs": ["workspace_members"],
                "produces": "crate_file_inventory",
            },
            {
                "id": "3",
                "type": "analyze",
                "desc": "Read dependencies",
                "deps": ["2"],
                "inputs": ["workspace_members"],
                "produces": "crate_dependency_graph",
            },
            {
                "id": "4",
                "type": "doc",
                "desc": "Create checklist",
                "deps": ["3"],
                "produces": "best_practices_checklist",
            },
            # Step 5 only has dep on step 4, but declares inputs that include outputs
            # from steps 1, 2 — those producers are NOT in current deps.
            # Pass 6 must NOT add step 1 or step 2 to deps.
            {
                "id": "5",
                "type": "review",
                "desc": "Present checklist and inventory",
                "deps": ["4"],
                "inputs": [
                    "workspace_members",
                    "crate_file_inventory",
                    "best_practices_checklist",
                ],
            },
        ],
    )
    execution_plan = plan.to_execution_plan()
    steps = {s.id: s for s in execution_plan.steps}

    # Step 5 must NOT gain deps on step 1 or step 2 — they weren't in original deps.
    assert (
        "1" not in steps["5"].depends_on
    ), f"Pass 6 must not add step 1 to step 5 deps; got {steps['5'].depends_on}"
    assert (
        "2" not in steps["5"].depends_on
    ), f"Pass 6 must not add step 2 to step 5 deps; got {steps['5'].depends_on}"
    # Step 4 (in original deps and produces best_practices_checklist consumed by step 5)
    # should be kept.
    assert (
        "4" in steps["5"].depends_on
    ), f"Step 5 must keep dep on step 4; got {steps['5'].depends_on}"


# Pass 7: phantom dep normalization
# ──────────────────────────────────────────────────────────────────────────────


# Pass 6.6: anchor approval/review steps to their predecessor
# ──────────────────────────────────────────────────────────────────────────────


def test_pass6_6_anchors_approval_step_with_no_deps_to_predecessor():
    """An approval step with no deps is anchored to the preceding non-approval step.

    Reproduces session 89e99624: step 10 (final user review, exec=approval, deps=[])
    dispatched in the first batch and auto-approved before any analysis ran.
    Pass 6.6 must add a dep on step 9 (the preceding non-approval step).
    """
    plan = ReadableTaskPlan(
        name="Approval anchor test",
        complexity=TaskComplexity.COMPLEX,
        desc="Final review anchored to preceding step",
        steps=[
            {"id": "1", "type": "analyze", "desc": "Step 1"},
            {
                "id": "9",
                "type": "doc",
                "desc": "Synthesize findings",
                "deps": ["1"],
                "produces": "final_report",
            },
            # Review step with no deps — LLM forgot to write deps=['9']
            {
                "id": "10",
                "type": "review",
                "desc": "Present report to user",
                "exec": "approval",
                "deps": [],
            },
        ],
    )
    execution_plan = plan.to_execution_plan()
    steps = {s.id: s for s in execution_plan.steps}

    assert (
        "9" in steps["10"].depends_on
    ), f"Pass 6.6 must anchor approval step 10 to preceding step 9; got {steps['10'].depends_on}"


def test_pass6_6_does_not_override_existing_deps():
    """Pass 6.6 must not change approval steps that already have explicit deps."""
    plan = ReadableTaskPlan(
        name="No override test",
        complexity=TaskComplexity.MODERATE,
        desc="Approval with explicit dep kept",
        steps=[
            {"id": "1", "type": "analyze", "desc": "Step 1"},
            {"id": "2", "type": "doc", "desc": "Step 2", "deps": ["1"]},
            # Already has explicit dep on step 1 — must not be changed
            {
                "id": "3",
                "type": "review",
                "desc": "Review step 1 output",
                "exec": "approval",
                "deps": ["1"],
            },
        ],
    )
    execution_plan = plan.to_execution_plan()
    steps = {s.id: s for s in execution_plan.steps}

    assert steps["3"].depends_on == [
        "1"
    ], f"Existing deps must not be overridden; got {steps['3'].depends_on}"


def test_pass6_6_skips_first_step_with_no_predecessor():
    """If an approval step is the first (no predecessor), Pass 6.6 makes no change."""
    plan = ReadableTaskPlan(
        name="First step approval",
        complexity=TaskComplexity.SIMPLE,
        desc="Approval first in plan",
        steps=[
            # Approval is the very first step — no preceding non-approval step
            {
                "id": "1",
                "type": "review",
                "desc": "Initial approval gate",
                "exec": "approval",
                "deps": [],
            },
            {
                "id": "2",
                "type": "analyze",
                "desc": "Step after approval",
                "deps": ["1"],
            },
        ],
    )
    execution_plan = plan.to_execution_plan()
    steps = {s.id: s for s in execution_plan.steps}

    assert (
        steps["1"].depends_on == []
    ), f"First approval step with no predecessor must keep empty deps; got {steps['1'].depends_on}"


def test_pass7_replaces_phantom_dep_with_branch_variants():
    """LLM may write deps=['7'] when actual steps are '7a' and '7b'.

    Pass 7 must replace the phantom '7' with ['7a','7b'] so downstream steps
    are not permanently blocked after both branches terminate.
    """
    plan = ReadableTaskPlan(
        name="Phantom dep test",
        complexity=TaskComplexity.COMPLEX,
        desc="Replace phantom dep with branch variants",
        steps=[
            {
                "id": "1",
                "type": "analyze",
                "desc": "Discover workspace",
                "produces": "workspace_members",
            },
            {
                "id": "6",
                "type": "analyze",
                "desc": "Route to branch",
                "exec": "conditional",
                "deps": ["1"],
                "condition_on": "workspace_members",
            },
            {
                "id": "7a",
                "type": "analyze",
                "desc": "Multi-crate analysis",
                "exec": "loop",
                "deps": ["6"],
                "loop_over": "workspace_members",
                "produces": "per_crate_findings",
            },
            {
                "id": "7b",
                "type": "analyze",
                "desc": "Single crate deep-dive",
                "deps": ["6"],
                "produces": "per_crate_findings",
            },
            # LLM wrote deps=['7'] — phantom ID (plan only has 7a and 7b)
            {
                "id": "8",
                "type": "doc",
                "desc": "Cross-crate synthesis",
                "deps": ["7"],
                "inputs": ["per_crate_findings"],
            },
        ],
    )
    execution_plan = plan.to_execution_plan()
    steps = {s.id: s for s in execution_plan.steps}

    # Pass 7 must expand phantom '7' → ['7a', '7b']
    assert (
        "7" not in steps["8"].depends_on
    ), f"Phantom dep '7' must be replaced; got {steps['8'].depends_on}"
    assert (
        "7a" in steps["8"].depends_on
    ), f"Branch variant '7a' must replace phantom '7'; got {steps['8'].depends_on}"
    assert (
        "7b" in steps["8"].depends_on
    ), f"Branch variant '7b' must replace phantom '7'; got {steps['8'].depends_on}"


def test_pass7_preserves_phantom_dep_without_branch_variants():
    """A phantom dep with no matching branch variants is preserved unchanged.

    Pass 7 only replaces phantom deps when branch variants exist (e.g. "7" → ["7a","7b"]).
    A dep like "999" with no variants is left intact; the is_ready() fallback in
    base.py treats non-existent step IDs as satisfied at runtime, preventing BLOCKED.
    """
    plan = ReadableTaskPlan(
        name="Preserve phantom test",
        complexity=TaskComplexity.MODERATE,
        desc="Phantom dep without branch variants is preserved",
        steps=[
            {"id": "1", "type": "analyze", "desc": "Discover", "produces": "data"},
            # deps=['999'] references a step that doesn't exist and has no branch variants
            {
                "id": "2",
                "type": "doc",
                "desc": "Synthesize",
                "deps": ["1", "999"],
                "inputs": ["data"],
            },
        ],
    )
    execution_plan = plan.to_execution_plan()
    steps = {s.id: s for s in execution_plan.steps}

    # Pass 7 leaves "999" intact (no branch variants found)
    assert (
        "999" in steps["2"].depends_on
    ), f"Unresolvable phantom dep '999' must be preserved (not dropped); got {steps['2'].depends_on}"
    assert "1" in steps["2"].depends_on, f"Valid dep '1' must be kept; got {steps['2'].depends_on}"


def test_pass7_preserves_real_deps_unchanged():
    """Pass 7 must not touch deps that reference existing step IDs."""
    plan = ReadableTaskPlan(
        name="Real dep preservation",
        complexity=TaskComplexity.SIMPLE,
        desc="Real deps must not be modified",
        steps=[
            {"id": "1", "type": "analyze", "desc": "Step 1", "produces": "data"},
            {"id": "2", "type": "analyze", "desc": "Step 2", "deps": ["1"]},
        ],
    )
    execution_plan = plan.to_execution_plan()
    steps = {s.id: s for s in execution_plan.steps}

    assert steps["2"].depends_on == [
        "1"
    ], f"Real dep '1' must be preserved unchanged; got {steps['2'].depends_on}"


# get_ready_steps: phantom dep fallback (belt-and-suspenders for runtime)
# ──────────────────────────────────────────────────────────────────────────────


def test_get_ready_steps_treats_phantom_dep_as_satisfied():
    """Steps with a phantom dep (ID not in plan) must be returned as ready.

    This is the belt-and-suspenders runtime fallback.  Pass 7 normalizes phantom
    deps at parse time; this test ensures the runtime also handles them gracefully
    in case a phantom dep slips through (e.g. loaded from a persisted plan).
    """
    from victor.agent.planning.base import ExecutionPlan, PlanStep, StepStatus

    steps = [
        PlanStep(id="1", description="Step 1", status=StepStatus.COMPLETED),
        # Step 2 has a phantom dep on '999' which doesn't exist in this plan.
        PlanStep(id="2", description="Step 2", depends_on=["1", "999"]),
    ]
    plan = ExecutionPlan(id="test", goal="test", steps=steps)

    ready = plan.get_ready_steps()
    assert any(
        s.id == "2" for s in ready
    ), "Step 2 must be ready — dep '1' is completed, '999' is phantom (non-existent)"


def test_get_ready_steps_blocks_on_real_unsatisfied_dep():
    """Steps with a real unsatisfied dep must NOT be returned as ready."""
    from victor.agent.planning.base import ExecutionPlan, PlanStep

    steps = [
        PlanStep(id="1", description="Step 1"),  # PENDING — not completed
        PlanStep(id="2", description="Step 2", depends_on=["1"]),
    ]
    plan = ExecutionPlan(id="test", goal="test", steps=steps)

    ready = plan.get_ready_steps()
    assert not any(
        s.id == "2" for s in ready
    ), "Step 2 must NOT be ready — dep '1' is a real step and not yet completed"


def test_pass8_adds_findings_deps_to_synthesis_step():
    """Synthesis step (produces='final_report') must depend on all findings-producing
    steps even when the LLM omitted those deps.  Prevents the report from being generated
    before per-crate analysis (7a) and cross-crate analysis (8) have run.
    """
    plan_dict = {
        "name": "rust-review",
        "desc": "Rust Arc & best practices review",
        "complexity": "complex",
        "steps": [
            {
                "id": "1",
                "type": "analyze",
                "description": "Read Cargo.toml",
                "deps": [],
                "tools": ["read"],
                "context": {"produces": "workspace_members"},
            },
            {
                "id": "7a",
                "type": "analyze",
                "description": "Deep review each crate",
                "deps": ["1"],
                "tools": ["read", "grep"],
                "context": {"produces": "per_crate_findings"},
            },
            {
                "id": "8",
                "type": "analyze",
                "description": "Cross-crate analysis",
                "deps": ["7a"],
                "tools": ["read"],
                "context": {"produces": "cross_crate_findings"},
            },
            {
                "id": "9",
                "type": "analyze",
                "description": "Dependency audit",
                "deps": ["1"],
                "tools": ["read"],
                "context": {"produces": "dependency_findings"},
            },
            {
                # LLM only specified dep on 9 — missing 7a and 8
                "id": "10",
                "type": "doc",
                "description": "Synthesize all findings into report",
                "deps": ["9"],
                "tools": ["write"],
                "context": {"produces": "final_report"},
            },
        ],
    }
    plan = ReadableTaskPlan(**plan_dict)
    enriched = plan._enrich_step_dicts(plan.steps)

    step10 = next(s for s in enriched if str(s.get("id")) == "10")
    deps_10 = {str(d) for d in (step10.get("deps") or step10.get("depends_on") or [])}

    assert (
        "7a" in deps_10
    ), f"Pass 8 must add dep on 7a (per_crate_findings producer); deps={deps_10}"
    assert (
        "8" in deps_10
    ), f"Pass 8 must add dep on 8 (cross_crate_findings producer); deps={deps_10}"
    assert "9" in deps_10, "Original dep on 9 must be preserved"


def test_pass8_adds_findings_deps_to_report_step_without_produces_key():
    """Doc/report steps without produces=final_report must still wait on findings."""
    plan_dict = {
        "name": "rust-review",
        "desc": "Rust Arc & best practices review",
        "complexity": "complex",
        "steps": [
            {
                "id": "7a",
                "type": "analyze",
                "description": "Deep review each crate",
                "deps": [],
                "tools": ["read", "grep"],
                "context": {"produces": "per_crate_findings"},
            },
            {
                "id": "8",
                "type": "analyze",
                "description": "Cross-crate analysis",
                "deps": ["7a"],
                "tools": ["read"],
                "context": {"produces": "cross_crate_findings"},
            },
            {
                "id": "9",
                "type": "doc",
                "description": "Synthesize all findings into a prioritized report",
                "deps": ["7b"],
                "tools": ["write"],
            },
        ],
    }
    plan = ReadableTaskPlan(**plan_dict)
    enriched = plan._enrich_step_dicts(plan.steps)

    step9 = next(s for s in enriched if str(s.get("id")) == "9")
    deps_9 = {str(d) for d in (step9.get("deps") or step9.get("depends_on") or [])}

    assert "7a" in deps_9
    assert "8" in deps_9
    assert "7b" in deps_9


def test_schema_wires_performance_findings_into_final_report():
    """Performance hotspot analysis should be a named input to the final report."""
    plan = ReadableTaskPlan(
        name="Rust best practices review - Arc, immutability, performance",
        complexity=TaskComplexity.COMPLEX,
        desc="Review Rust workspace",
        steps=[
            {
                "id": "8a",
                "type": "analyze",
                "desc": "Deep review each workspace member crate with full module coverage",
                "tools": ["read", "grep", "code_search"],
                "produces": "per_crate_findings",
            },
            {
                "id": "9",
                "type": "analyze",
                "desc": "Cross-crate analysis: shared Arc patterns across crate boundaries",
                "tools": ["read", "grep", "code_search"],
                "produces": "cross_crate_findings",
            },
            {
                "id": "10",
                "type": "analyze",
                "desc": (
                    "Performance hotspot analysis: identify high-frequency paths, "
                    "find allocation-heavy loops, detect unnecessary String/Vec creation"
                ),
                "tools": ["read", "grep", "code_search"],
            },
            {
                "id": "11",
                "type": "doc",
                "desc": (
                    "Synthesize all findings into prioritized report: executive summary, "
                    "per-crate detailed findings, cross-crate optimization opportunities, "
                    "performance hotspot rankings"
                ),
                "tools": ["write"],
            },
        ],
    )

    steps = {step.id: step for step in plan.to_execution_plan().steps}

    assert steps["10"].context["produces"] == "performance_hotspot_findings"
    assert "performance_hotspot_findings" in steps["11"].inputs
    assert "10" in steps["11"].depends_on


def test_pass8_does_not_touch_steps_without_synthesis_produces():
    """Non-synthesis steps (produces='per_crate_findings') should NOT get their
    deps modified by Pass 8."""
    plan_dict = {
        "name": "rust-review",
        "desc": "Rust Arc & best practices review",
        "complexity": "complex",
        "steps": [
            {
                "id": "1",
                "type": "analyze",
                "description": "Read workspace",
                "deps": [],
                "tools": ["read"],
                "context": {"produces": "workspace_members"},
            },
            {
                "id": "7a",
                "type": "analyze",
                "description": "Deep review each crate",
                "deps": ["1"],
                "tools": ["read", "grep"],
                "context": {"produces": "per_crate_findings"},
            },
            {
                "id": "8",
                "type": "analyze",
                "description": "Cross-crate analysis",
                "deps": ["7a"],
                "tools": ["read"],
                "context": {"produces": "cross_crate_findings"},
            },
        ],
    }
    plan = ReadableTaskPlan(**plan_dict)
    enriched = plan._enrich_step_dicts(plan.steps)

    step7a = next(s for s in enriched if str(s.get("id")) == "7a")
    deps_7a = {str(d) for d in (step7a.get("deps") or step7a.get("depends_on") or [])}

    # 7a produces per_crate_findings (a findings key) — Pass 8 should NOT modify it
    assert "1" in deps_7a, "Original dep on 1 must be preserved"
    assert "8" not in deps_7a, "Pass 8 must not add cross-crate dep to an analysis step"


def test_pass8_no_op_when_no_findings_producers():
    """When no step produces a *_findings key, Pass 8 must leave all deps untouched."""
    plan_dict = {
        "name": "simple",
        "desc": "Simple plan",
        "complexity": "simple",
        "steps": [
            {"id": "1", "type": "analyze", "description": "Step 1", "deps": []},
            {
                "id": "2",
                "type": "doc",
                "description": "Write report",
                "deps": ["1"],
                "context": {"produces": "final_report"},
            },
        ],
    }
    plan = ReadableTaskPlan(**plan_dict)
    enriched = plan._enrich_step_dicts(plan.steps)

    step2 = next(s for s in enriched if str(s.get("id")) == "2")
    deps_2 = {str(d) for d in (step2.get("deps") or step2.get("depends_on") or [])}

    assert deps_2 == {"1"}, f"No findings producers → deps unchanged; got {deps_2}"


def test_get_ready_steps_after_branch_completion_unblocks_downstream():
    """Reproduces the production BLOCKED scenario: steps 8-10 must become ready
    once 7a COMPLETES and 7b is SKIPPED (both in satisfied).

    Root cause: LLM wrote deps=['7'] (phantom) for step 8.  Pass 7 expands it to
    ['7a','7b'].  After 7a COMPLETES and 7b SKIPPED, both in satisfied → step 8 ready.
    """
    from victor.agent.planning.base import ExecutionPlan, PlanStep, StepStatus

    steps = [
        PlanStep(id="6", description="Conditional", status=StepStatus.COMPLETED),
        PlanStep(
            id="7a",
            description="Loop branch",
            status=StepStatus.COMPLETED,
            depends_on=["6"],
        ),
        PlanStep(
            id="7b",
            description="Single branch",
            status=StepStatus.SKIPPED,
            depends_on=["6"],
        ),
        # After Pass 7 normalization, phantom '7' becomes ['7a','7b']
        PlanStep(id="8", description="Synthesis", depends_on=["7a", "7b"]),
        PlanStep(id="9", description="Report", depends_on=["8"]),
        PlanStep(id="10", description="Review", depends_on=["9"]),
    ]
    plan = ExecutionPlan(id="test", goal="test", steps=steps)

    ready = plan.get_ready_steps()
    ready_ids = {s.id for s in ready}
    assert (
        "8" in ready_ids
    ), f"Step 8 must be ready (7a COMPLETED, 7b SKIPPED → both satisfied); got ready={ready_ids}"
    assert "9" not in ready_ids, "Step 9 is not yet ready (step 8 not completed)"
    assert "10" not in ready_ids, "Step 10 is not yet ready (step 9 not completed)"


# ---------------------------------------------------------------------------
# Fix #1: synthesis step default budget
# ---------------------------------------------------------------------------


def test_task_description_appends_exit_criteria_for_tool_step():
    """Non-loop tool-execution steps with exit criteria must include them in the
    task string so the sub-agent cannot self-terminate before satisfying them.

    Root cause: step 8 (cross-crate analysis, exec=agent) self-terminated after
    4 tool calls because no completion requirements were present in the task.
    Only loop iterations previously got exit criteria appended.
    """
    from victor.agent.planning.base import PlanStep, StepType

    step = PlanStep(
        id="8",
        description="Cross-crate analysis: identify shared Arc patterns",
        step_type=StepType.RESEARCH,
        context={},
        exit_criteria=["grep for Arc<T> in all crates", "document lock ordering"],
    )

    task = PlanningTeamExecutionAdapter._task_description_for_step(step)

    assert (
        "grep for Arc<T> in all crates" in task
    ), f"Exit criteria must appear in task; got: {task!r}"
    assert "document lock ordering" in task, f"Exit criteria must appear in task; got: {task!r}"
    assert (
        "Verification criteria" in task
    ), f"Task must include a 'Verification criteria' header; got: {task!r}"


def test_task_description_no_exit_criteria_keeps_inventory_step_without_verification():
    """Inventory steps without exit criteria avoid extra verification text."""
    from victor.agent.planning.base import PlanStep, StepType

    step = PlanStep(
        id="3",
        description="Map file inventory",
        step_type=StepType.RESEARCH,
        context={},
        exit_criteria=[],
    )

    task = PlanningTeamExecutionAdapter._task_description_for_step(step)

    assert task == "Map file inventory"
    assert "Verification criteria" not in task
    assert "Evidence requirements" not in task


def test_task_description_exit_criteria_appended_after_context_and_format():
    """Exit criteria appear as the last section — after plan_state context and format hints."""
    from victor.agent.planning.base import PlanStep, StepType

    step = PlanStep(
        id="8",
        description="Cross-crate analysis",
        step_type=StepType.RESEARCH,
        context={"produces": "cross_crate_findings"},
        exit_criteria=["grep for shared Arc patterns"],
    )

    task = PlanningTeamExecutionAdapter._task_description_for_step(step)

    # FORMAT block should be present (due to produces key)
    assert "OUTPUT FORMAT" in task
    # Exit criteria come after format
    format_pos = task.index("OUTPUT FORMAT")
    criteria_pos = task.index("Verification criteria")
    assert criteria_pos > format_pos, "Verification criteria must appear after OUTPUT FORMAT"


# ---------------------------------------------------------------------------
# Fix #1: synthesis / doc step default budget = 8
# ---------------------------------------------------------------------------


def test_synthesis_doc_step_default_tool_calls_is_8():
    """Synthesis (doc/write) steps without explicit tool_calls default to 8.

    Root cause: schema hint showed 'tool_calls: 5' for synthesis steps. With
    577 items of per_crate_findings to synthesize, budget=5 is insufficient
    for reading findings and writing the report. Default raised to 8.
    """
    plan = ReadableTaskPlan(
        name="Synthesis budget test",
        complexity=TaskComplexity.COMPLEX,
        desc="Test",
        steps=[
            {
                "id": "9",
                "type": "doc",
                "desc": "Synthesize all findings into a prioritized report",
                "tools": ["write"],
                "deps": ["8"],
                "exec": "agent",
                "inputs": ["per_crate_findings", "cross_crate_findings"],
                "produces": "final_report",
            }
        ],
    )
    step = plan.to_execution_plan().steps[0]
    assert (
        step.estimated_tool_calls == 8
    ), f"Doc/synthesis steps must default to 8 tool calls (not 5 or 10); got {step.estimated_tool_calls}"


def test_synthesis_explicit_tool_calls_respected():
    """Explicit tool_calls on a doc step overrides the 8 default."""
    plan = ReadableTaskPlan(
        name="Synthesis explicit budget test",
        complexity=TaskComplexity.COMPLEX,
        desc="Test",
        steps=[
            {
                "id": "9",
                "type": "doc",
                "desc": "Synthesize findings",
                "tools": ["write"],
                "tool_calls": 12,
            }
        ],
    )
    step = plan.to_execution_plan().steps[0]
    assert step.estimated_tool_calls == 12


def test_non_synthesis_step_default_remains_10():
    """Non-doc, non-loop steps keep the existing default of 10."""
    plan = ReadableTaskPlan(
        name="Non-synthesis default test",
        complexity=TaskComplexity.COMPLEX,
        desc="Test",
        steps=[
            {
                "id": "8",
                "type": "analyze",
                "desc": "Cross-crate analysis",
                "tools": ["read", "grep"],
                "exec": "agent",
            }
        ],
    )
    step = plan.to_execution_plan().steps[0]
    assert step.estimated_tool_calls == 10

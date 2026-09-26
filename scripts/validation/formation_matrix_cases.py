"""Explicit artifact scenarios for the canonical formation and aggregation registries.

These configure public presets; execution always uses the existing coordinator.
The legacy six-case battery remains unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from scripts.validation.formation_matrix_oracle import CONTRACT_VERSION, NUMERIC_DOMAIN
from victor.coordination.formations.ensemble import MODES
from victor.framework.teams import AgentTeam, TeamFormation, TeamMemberSpec

TASK_PROFILES = ("standard", "single-file")


def numeric_artifact(function: str, task_profile: str) -> str:
    """Derive the implementation path once for prompts, artifacts and the oracle."""
    if task_profile not in TASK_PROFILES:
        raise ValueError(f"Unknown matrix task profile: {task_profile}")
    return f"test_{function}.py" if task_profile == "single-file" else f"{function}.py"


CASE_NAMES = tuple(formation.value for formation in TeamFormation) + tuple(
    f"ensemble_{mode}" for mode in sorted(MODES)
)


@dataclass(frozen=True)
class MatrixCase:
    team: AgentTeam
    artifacts: dict[str, tuple[str, ...]]
    pytest_names: tuple[str, ...]
    isolated: bool = False
    task_profile: str = "standard"

    @property
    def executed_names(self) -> tuple[str, ...]:
        return tuple(self.artifacts)


async def build_case(
    orchestrator: Any,
    name: str,
    root: Path,
    python: Path,
    timeout: int,
    *,
    task_profile: str = "standard",
) -> MatrixCase:
    """Build one bounded scenario with explicit expected participants and artifacts."""
    if task_profile not in TASK_PROFILES:
        raise ValueError(f"Unknown matrix task profile: {task_profile}")
    if name not in CASE_NAMES:
        raise ValueError(f"Unknown matrix case: {name}")
    shared: dict[str, Any] = {
        "parent_session_id": root.name,
        "capture_member_usage": True,
        "member_task_binding": "structured-v1",
        "consensus_max_rounds": 3,
    }
    kwargs: dict[str, Any] = {"shared_context": shared, "timeout_seconds": timeout}

    def files(member: str) -> tuple[str, ...]:
        return tuple(dict.fromkeys((numeric_artifact(member, task_profile), f"test_{member}.py")))

    def single_file_goal(member: str, directory: Path | None) -> str:
        path = numeric_artifact(member, task_profile)
        target = str(directory / path) if directory else path
        return (
            NUMERIC_DOMAIN + f"Write one file {target} containing both "
            f"def {member}(x): return x * 2 and "
            f"def test_{member}(): assert {member}(4) == 8. "
            f"Run {python} -m pytest {target} -q using shell readonly=False. "
            "Do not repeat successful writes. The file must exist and pytest must pass. "
        )

    def spec(member: str) -> TeamMemberSpec:
        return TeamMemberSpec(
            role="executor",
            name=member,
            goal=(
                single_file_goal(member, root)
                + 'Your final output must be exactly {"status":"ready"}.'
                if task_profile == "single-file"
                else NUMERIC_DOMAIN + f"Assigned member: {member}. Complete in order: "
                f"1. Write {root}/{member}.py with def {member}(x): return x * 2. "
                f"2. Write {root}/test_{member}.py importing that function and defining "
                f"def test_{member}(): assert {member}(4) == 8. "
                f"3. Run {python} -m pytest {root}/test_{member}.py -q using shell readonly=False. "
                "Do not repeat successful writes. Both files must exist and pytest must pass. "
                'Your final output must be exactly {"status":"ready"}.'
            ),
            allowed_tools=["read", "write", "shell"],
            tool_budget=12,
            max_iterations=10,
        )

    members = [spec("first"), spec("second")]
    artifacts: dict[str, tuple[str, ...]] = {}
    for member in members:
        assert member.name is not None
        artifacts[member.name] = files(member.name)
    goal = "Complete your assigned deliverable and test."
    isolated = False
    pytest_names: tuple[str, ...] = ("first", "second")
    if name.startswith("ensemble_"):
        mode = name.removeprefix("ensemble_")
        isolated = True
        shared.update(
            parallel_worktree_isolation=True,
            repo_root=str(root),
            worktree_parent=str(root / "members"),
            branch_prefix="feat/matrix-member",
        )
        goal = (
            NUMERIC_DOMAIN + "Independently work in your assigned isolated workspace. "
            "Write member.py with def member(x): return x * 2. "
            "Write test_member.py importing member and defining def test_member(): assert member(4) == 8. "
            f"Run {python} -m pytest test_member.py -q using shell readonly=False. "
            "Use relative paths so each member writes only in its own workspace. "
            "Do not repeat successful writes. After both files exist and tests pass, return only "
            'JSON {"vote_key":"double","answer":"<absolute path to your member.py>"}.'
        )
        if task_profile == "single-file":
            goal = (
                "Independently work in your assigned isolated workspace. "
                + single_file_goal("member", None)
                + "Use relative paths so each member writes only in its own workspace. "
                'Return only JSON {"vote_key":"double","answer":"<absolute path to your test_member.py>"}.'
            )
        artifacts = {"first": files("member"), "second": files("member")}
        aggregator = None
        if mode != "vote":
            contract = (
                '{"selected_member_id":"<one candidate ID from the task>"}'
                if mode == "judge"
                else '{"answer":"<combined findings with artifact references>"}'
            )
            aggregator = spec(mode)
            aggregator.goal = (
                "Read the supplied candidate artifact paths and inspect the corresponding tests. "
                f"Run {python} -m pytest on each candidate's test_member.py separately using shell readonly=False. "
                f"Write decision.json in your assigned workspace with exactly {contract}. "
                "Use the candidate evidence to make your one-shot decision. Return only that same JSON."
            )
            artifacts[mode] = ("decision.json",)
        team = await AgentTeam.create_ensemble_team(
            orchestrator, name, goal, members, mode=mode, aggregator=aggregator, **kwargs
        )
    elif name == "reflection":
        members[1].goal = (
            NUMERIC_DOMAIN
            + (
                f"Read {root}/first.py and {root}/test_first.py. "
                if task_profile == "standard"
                else f"Read {root}/test_first.py. "
            )
            + f"Run {python} -m pytest {root}/test_first.py -q using shell readonly=False. "
            f"Write {root}/review.json with exactly verdict and feedback keys: "
            "verdict=satisfied only if the function doubles input and its test passes, "
            "otherwise verdict=needs_work; feedback must be a string. Return only that JSON."
        )
        artifacts["second"] = ("review.json",)
        pytest_names = ("first",)
        team = await AgentTeam.create_reflection_team(
            orchestrator,
            name,
            goal,
            generator=members[0],
            critic=members[1],
            verdict_format="json",
            rounds=3,
            **kwargs,
        )
    elif name in {"group_chat", "debate", "handoff"}:
        for index, member in enumerate(members):
            member.goal = member.goal.replace(
                'Your final output must be exactly {"status":"ready"}.',
                "After writing your files and passing pytest, return ONLY the conversation JSON "
                '{"content":"<artifact references and test result>","done":<boolean>,"handoff_to":<ID or null>}. '
                + (
                    "Set done=false and handoff_to to the other peer ID supplied in your task."
                    if name == "handoff" and index == 0
                    else f"Set done={'false' if index == 0 else 'true'} and handoff_to=null."
                ),
            )
        factory = getattr(AgentTeam, f"create_{name}_team")
        options: dict[str, Any] = {"max_turns": 2}
        if name == "handoff":
            options["start_member"] = "first"
        elif name == "debate":
            judge = spec("judge")
            judge.goal = (
                "Review the transcript and the speakers' artifacts. "
                f"Run {python} -m pytest {root} -q using shell readonly=False. "
                f"Write {root}/decision.json with exactly selected_member_id (one speaker's ID) "
                "and verdict (nonempty string). Return only that same JSON as your one-shot verdict."
            )
            options["judge"] = judge
            artifacts["judge"] = ("decision.json",)
        team = await factory(orchestrator, name, goal, members, **options, **kwargs)
    elif name == "adaptive":
        team = await AgentTeam.create_adaptive_team(orchestrator, name, goal, members, **kwargs)
    elif name == "dynamic_router":
        team = await AgentTeam.create_router_team(
            orchestrator, name, "artifact: " + goal, members, routes={"artifact": "first"}, **kwargs
        )
        artifacts.pop("second")
        pytest_names = ("first",)
    elif name == "multi_level_hierarchy":
        members.append(spec("third"))
        artifacts["third"] = files("third")
        pytest_names = ("first", "second", "third")
        team = await AgentTeam.create_multi_level_hierarchy_team(
            orchestrator, name, goal, members, max_depth=3, **kwargs
        )
        first, second, third = team._config.members
        # A three-level chain retains the complete structured objective at every
        # level; the default branching splitter partitions raw text, not tasks.
        team._config.shared_context["hierarchy"] = {
            "member_id": first.id,
            "children": [{"member_id": second.id, "children": [{"member_id": third.id}]}],
        }
    elif name in {"sequential", "pipeline", "parallel", "hierarchical", "consensus"}:
        team = await AgentTeam.create(
            orchestrator, name, goal, members, formation=TeamFormation(name), **kwargs
        )
    else:
        raise ValueError(f"Registered formation needs an explicit validation scenario: {name}")
    # Formations can replace the member task (routing, critique, delegation).
    # Carry assignments in the structured team objective so those payloads retain
    # the task contract; bind IDs only after the canonical preset created them.
    task_contract = {"version": CONTRACT_VERSION, "numeric_domain": NUMERIC_DOMAIN}
    if task_profile != "standard":
        task_contract["task_profile"] = task_profile
    team._config.goal = json.dumps(
        {
            "objective": team._config.goal,
            "task_contract": task_contract,
            "instructions": "Perform only your assigned task, identified by member_id, name, or role. Preserve the response_contract supplied by your formation.",
            "assignments": {
                member.id: {"name": member.name, "task": member.goal}
                for member in team._config.members
            },
        }
    )
    return MatrixCase(
        team=team,
        artifacts=artifacts,
        pytest_names=pytest_names,
        isolated=isolated,
        task_profile=task_profile,
    )

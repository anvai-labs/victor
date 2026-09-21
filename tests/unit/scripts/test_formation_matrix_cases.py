"""Exercise matrix configuration, leaving strategy semantics to their existing suites."""

import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import MagicMock

import pytest

_ROOT = Path(__file__).resolve().parents[3]
# Console pytest does not put the repository on sys.path. This file-loaded
# scenario module imports sibling validation scripts, independently of order.
sys.path.insert(0, str(_ROOT))

from victor.coordination.formations.ensemble import MODES
from victor.teams.types import TeamFormation

_PATH = _ROOT / "scripts/validation/formation_matrix_cases.py"
_SPEC = importlib.util.spec_from_file_location("formation_matrix_cases", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
matrix = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = matrix
_SPEC.loader.exec_module(matrix)


@pytest.mark.parametrize(
    "name", [f.value for f in TeamFormation] + [f"ensemble_{m}" for m in sorted(MODES)]
)
async def test_every_registered_formation_has_an_explicit_artifact_case(name, tmp_path):
    case = await matrix.build_case(
        MagicMock(), name, tmp_path, Path("/work/.venv-codesign/bin/python"), 240
    )
    config = case.team._config
    expected_formation = "parallel" if name.startswith("ensemble_") else name
    assert config.formation.value == expected_formation
    names = {member.name for member in config.members}
    assert set(case.artifacts) <= names
    assert set(case.executed_names) == set(case.artifacts)
    assert all(case.artifacts.values())
    assert config.shared_context["capture_member_usage"] is True
    assert config.shared_context["member_task_binding"] == "structured-v1"
    assert config.shared_context["parent_session_id"] == tmp_path.name
    assert config.timeout_seconds == 240
    task = json.loads(config.goal)
    assert task["task_contract"]["version"] == 2
    assert "multiples of 0.25" in task["task_contract"]["numeric_domain"]
    assert set(task["assignments"]) == {member.id for member in config.members}
    for member in config.members:
        assert task["assignments"][member.id] == {"name": member.name, "task": member.goal}
    if name == "dynamic_router":
        assert case.executed_names == ("first",)
        assert len(config.members) == 2
        assert set(config.shared_context["router_routes"].values()) == {config.members[0].id}
    else:
        assert set(case.executed_names) == names
    if name.startswith("ensemble_"):
        assert case.isolated
        assert config.shared_context["parallel_worktree_isolation"] is True
        assert config.shared_context["ensemble_mode"] == name.removeprefix("ensemble_")
        assert config.members[0].goal == config.members[1].goal
        assert case.artifacts["first"] == ("member.py", "test_member.py")
    if name in {"group_chat", "debate", "handoff"}:
        assert config.shared_context["conversation_max_turns"] == 2
    if name == "reflection":
        assert config.shared_context["reflection_verdict_format"] == "json"
        assert case.artifacts["second"] == ("review.json",)


async def test_unknown_cases_fail_explicitly(tmp_path):
    with pytest.raises(ValueError):
        await matrix.build_case(
            MagicMock(), "unknown", tmp_path, Path("/work/.venv-codesign/bin/python"), 240
        )


async def test_hierarchy_dispatch_preserves_every_complete_assignment(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    observed = {}

    async def spawn(**kwargs):
        observed[kwargs["member_id"]] = kwargs["task"]
        return SimpleNamespace(success=True, summary="ready", details={})

    monkeypatch.setattr(
        "victor.agent.subagents.orchestrator.SubAgentOrchestrator",
        lambda orchestrator: SimpleNamespace(spawn=AsyncMock(side_effect=spawn)),
    )
    case = await matrix.build_case(
        MagicMock(), "multi_level_hierarchy", tmp_path, Path(sys.executable), 240
    )
    result = await case.team.run()
    assert result.success
    for member in case.team._config.members:
        task = json.loads(observed[member.id])
        assert task["member"]["id"] == member.id
        assert task["member"]["name"] == member.name
        assert task["member"]["assignment"] == member.goal
    assert {item.metadata["hierarchy_level"] for item in result.member_results.values()} == {
        1,
        2,
        3,
    }

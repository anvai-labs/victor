"""Real worktree dispatch and adapter identity regressions."""

from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.core.shared_types import SubAgentRole
from victor.teams.types import TeamConfig, TeamFormation, TeamMember
from victor.teams.unified_coordinator import UnifiedTeamCoordinator


def config(context):
    return TeamConfig(
        name="isolation",
        goal="write",
        formation=TeamFormation.PARALLEL,
        members=[
            TeamMember(
                id=name,
                name=name,
                role=SubAgentRole.EXECUTOR,
                goal="write",
                allowed_tools=["write", "shell"],
            )
            for name in ("a", "b")
        ],
        shared_context=context,
    )


@pytest.mark.asyncio
async def test_dispatch_materializes_distinct_worktrees_and_forwards_identity(
    tmp_path, monkeypatch
):
    repo = tmp_path / "repo"
    repo.mkdir()
    for command in (
        ["init"],
        ["config", "user.email", "test@example.com"],
        ["config", "user.name", "Test"],
        ["commit", "--allow-empty", "-m", "seed"],
    ):
        subprocess.run(["git", *command], cwd=repo, check=True, capture_output=True)
    seen = []

    async def spawn(**kwargs):
        seen.append(kwargs)
        path = Path(kwargs["working_directory"])
        (path / "same.py").write_text(kwargs["member_id"])
        return SimpleNamespace(success=True, summary="done", details={"child_session_id": "wire"})

    mocked = MagicMock()
    mocked.spawn = AsyncMock(side_effect=spawn)
    monkeypatch.setattr(
        "victor.agent.subagents.orchestrator.SubAgentOrchestrator", lambda parent: mocked
    )
    coord = UnifiedTeamCoordinator(MagicMock(), lightweight_mode=True)
    result = await coord.execute_team_config(
        config(
            {
                "parallel_worktree_isolation": True,
                "repo_root": str(repo),
                "worktree_parent": str(tmp_path / "members"),
                "branch_prefix": "feat/isolation",
                "parent_session_id": "root",
                "team_id": "team",
            }
        )
    )
    assert result.success, result.error
    assert len(seen) == 2
    assert len({item["working_directory"] for item in seen}) == 2
    assert not (repo / "same.py").exists()
    for item in seen:
        assert (Path(item["working_directory"]) / "same.py").read_text() == item["member_id"]
        assert item["parent_session_id"] == "root"
        assert item["team_id"] == "team"
        assert result.member_results[item["member_id"]].metadata["child_session_id"] == "wire"
    # Preserved by default: the caller can inspect both deliverables after return.


@pytest.mark.asyncio
@pytest.mark.parametrize("context", [{}, {"dry_run_worktrees": True}])
async def test_isolation_fails_closed_without_materialized_worktrees(context, caplog):
    coord = UnifiedTeamCoordinator(lightweight_mode=True)
    coord.set_formation(TeamFormation.PARALLEL)
    member = SimpleNamespace(
        id="a", role="executor", execute_task=AsyncMock(), receive_message=AsyncMock()
    )
    coord.add_member(member)
    result = await coord.execute_task("work", {"parallel_worktree_isolation": True, **context})
    assert not result["success"]
    member.execute_task.assert_not_awaited()
    assert "could not be materialized" in caplog.text

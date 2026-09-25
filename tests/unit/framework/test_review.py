"""Unit tests for the independent review panel (victor.framework.review).

No LLM, no network: AgentTeam is faked at the seam and member outputs are
canned TeamResult objects built from the real teams types, so the aggregation
and coercion contracts are what's under test.
"""

from __future__ import annotations

import pytest

from victor.framework import review as review_mod
from victor.framework.review import (
    APPROVE,
    REQUEST_CHANGES,
    REVIEW_INCOMPLETE,
    ReviewerSpec,
    _member_goal,
    parse_reviewer_output,
    review_diff,
)
from victor.framework.teams import TeamFormation
from victor.teams.types import MemberResult, TeamResult


def _member_output_json(verdict: str, findings: list[dict] | None = None) -> str:
    import json

    return json.dumps(
        {"verdict": verdict, "summary": "s", "findings": findings or [], "confidence": 0.9}
    )


def _team_result(results: list[tuple[str, str, bool]]) -> TeamResult:
    """(display_name, output, success) triples → TeamResult keyed by auto ids."""
    member_results = {}
    for i, (name, output, success) in enumerate(results):
        member_results[f"rev{i}"] = MemberResult(
            member_id=f"rev{i}",
            success=success,
            output=output,
            metadata={"display_name": name, "member_id": f"rev{i}"},
        )
    return TeamResult(
        success=True,
        final_output="",
        member_results=member_results,
        formation=TeamFormation.PARALLEL,
    )


class FakeAgentTeam:
    """Captures members and replays canned outputs keyed by display name."""

    outputs: dict[str, tuple[str, bool]] = {}
    last_members: list = []

    def __init__(self, result: TeamResult):
        self._result = result

    @classmethod
    async def create(cls, *, members, **kwargs):
        FakeAgentTeam.last_members = members
        results = []
        for i, m in enumerate(members):
            output, ok = FakeAgentTeam.outputs.get(m.name, ("", False))
            results.append((m.name, output, ok))
        return cls(_team_result(results))

    async def run(self):
        return self._result


@pytest.fixture()
def fake_team(monkeypatch):
    FakeAgentTeam.outputs = {}
    monkeypatch.setattr(review_mod, "AgentTeam", FakeAgentTeam)
    return FakeAgentTeam


SPECS = [
    ReviewerSpec(provider="zai", model="glm-5.3", name="zai/glm"),
    ReviewerSpec(provider="inferflux", model="qwen3-coder-30b", name="inferflux/qwen"),
]


def test_parse_reviewer_output_valid_with_prose():
    text = 'blahblah {"verdict": "approve", "summary": "ok", "findings": [], "confidence": 0.7} trailing'
    parsed = parse_reviewer_output(text)
    assert parsed is not None
    assert parsed["verdict"] == APPROVE and parsed["confidence"] == 0.7


def test_parse_reviewer_output_coerces_bad_fields():
    parsed = parse_reviewer_output(
        '{"verdict": "ship it", "findings": [{"severity": "apocalyptic", "line": "x"}], "confidence": 9}'
    )
    assert parsed is None  # unknown verdict is unparseable, never coerced to approve

    parsed = parse_reviewer_output(
        '{"verdict": "request_changes", "findings": [{"severity": "apocalyptic", "line": "x", "summary": "f"}], "confidence": 9}'
    )
    assert parsed["verdict"] == REQUEST_CHANGES
    assert parsed["findings"][0]["severity"] == "minor"  # unknown severity coerced
    assert parsed["findings"][0]["line"] is None
    assert parsed["confidence"] == 1.0  # clamped from 9


def test_parse_reviewer_output_garbage():
    assert parse_reviewer_output("no json at all") is None
    assert parse_reviewer_output("{broken") is None
    assert parse_reviewer_output("[1,2,3]") is None
    assert parse_reviewer_output("") is None


def test_member_goal_truncates_long_diff():
    goal = _member_goal("intent", "x" * (review_mod._MAX_DIFF_CHARS + 10))
    assert "truncated" in goal


async def test_review_diff_any_block_blocks(fake_team):
    fake_team.outputs = {
        "zai/glm": (_member_output_json(APPROVE), True),
        "inferflux/qwen": (
            _member_output_json(
                REQUEST_CHANGES,
                [
                    {
                        "severity": "major",
                        "file": "a.py",
                        "line": 12,
                        "summary": "race on shared state",
                    }
                ],
            ),
            True,
        ),
    }
    verdict = await review_diff("diff", SPECS, orchestrator=object())
    assert verdict.verdict == REQUEST_CHANGES
    assert verdict.blocked_by == ["inferflux/qwen"]
    assert verdict.findings[0].reviewer == "inferflux/qwen"
    assert verdict.findings[0].line == 12
    assert not verdict.approved


async def test_review_diff_all_approve_approves(fake_team):
    fake_team.outputs = {
        "zai/glm": (_member_output_json(APPROVE), True),
        "inferflux/qwen": (_member_output_json(APPROVE), True),
    }
    verdict = await review_diff("diff", SPECS, orchestrator=object())
    assert verdict.verdict == APPROVE and verdict.approved


async def test_review_diff_all_abstain_is_incomplete(fake_team):
    # one explicit abstain, one member failure — nothing may be inferred
    fake_team.outputs = {
        "zai/glm": (_member_output_json("abstain"), True),
        "inferflux/qwen": ("", False),
    }
    verdict = await review_diff("diff", SPECS, orchestrator=object())
    assert verdict.verdict == REVIEW_INCOMPLETE
    assert verdict.reviewer_verdicts["inferflux/qwen"] == "abstain"


async def test_review_diff_unparseable_output_is_abstain(fake_team):
    fake_team.outputs = {
        "zai/glm": ("I think this looks fine overall, LGTM!", True),  # no JSON contract
        "inferflux/qwen": (_member_output_json(APPROVE), True),
    }
    verdict = await review_diff("diff", SPECS, orchestrator=object())
    assert verdict.reviewer_verdicts["zai/glm"] == "abstain"
    assert verdict.verdict == APPROVE  # the one real verdict carries it


async def test_review_diff_runner_failure_with_valid_output_still_counts(fake_team):
    # A cleanup-phase failure can flip MemberResult.success after a full verdict;
    # the verdict text is still the reviewer's and must count.
    fake_team.outputs = {
        "zai/glm": (
            _member_output_json(
                REQUEST_CHANGES,
                [{"severity": "critical", "file": "b.py", "line": 1, "summary": "cmd injection"}],
            ),
            False,
        ),
        "inferflux/qwen": (_member_output_json(APPROVE), True),
    }
    verdict = await review_diff("diff", SPECS, orchestrator=object())
    assert verdict.verdict == REQUEST_CHANGES


async def test_review_diff_members_carry_provider_and_model(fake_team):
    fake_team.outputs = {s.display_name: (_member_output_json(APPROVE), True) for s in SPECS}
    await review_diff("diff", SPECS, orchestrator=object())
    by_name = {m.name: m for m in FakeAgentTeam.last_members}
    assert by_name["zai/glm"].provider == "zai"
    assert by_name["inferflux/qwen"].model == "qwen3-coder-30b"


async def test_review_diff_requires_reviewers():
    with pytest.raises(ValueError):
        await review_diff("diff", [], orchestrator=object())


class _FakeProc:
    def __init__(self, stdout: bytes, returncode: int, stderr: bytes = b""):
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode

    async def communicate(self):
        return self._stdout, self._stderr


class _FakeAgentTeamForPR(FakeAgentTeam):
    pass


async def test_review_pull_request_fetches_diff_and_reviews(monkeypatch):
    seen: dict = {}

    async def fake_review_diff(diff, specs, *, orchestrator, intent, timeout_seconds):
        seen["diff"] = diff
        return "panel-verdict"

    monkeypatch.setattr(review_mod, "review_diff", fake_review_diff)

    async def fake_exec(*cmd, **kwargs):
        assert tuple(cmd[:3]) == ("gh", "pr", "diff")
        assert "-R" not in cmd
        return _FakeProc(b"diff --git a/f b/f\n", 0)

    monkeypatch.setattr(review_mod.asyncio, "create_subprocess_exec", fake_exec)
    verdict = await review_mod.review_pull_request(
        42, SPECS, orchestrator=object(), timeout_seconds=10
    )
    assert verdict == "panel-verdict"
    assert seen["diff"].startswith("diff --git")


async def test_review_pull_request_passes_repo_and_raises_on_gh_failure(monkeypatch):
    async def fake_exec(*cmd, **kwargs):
        assert "-R" in cmd and "org/repo" in cmd
        return _FakeProc(b"", 1, stderr=b"boom")

    monkeypatch.setattr(review_mod.asyncio, "create_subprocess_exec", fake_exec)
    with pytest.raises(RuntimeError, match="boom"):
        await review_mod.review_pull_request(
            42, SPECS, orchestrator=object(), repo="org/repo", timeout_seconds=10
        )


async def test_review_pull_request_empty_diff_is_rejected(monkeypatch):
    async def fake_exec(*cmd, **kwargs):
        return _FakeProc(b"   \n", 0)

    monkeypatch.setattr(review_mod.asyncio, "create_subprocess_exec", fake_exec)
    with pytest.raises(RuntimeError, match="empty diff"):
        await review_mod.review_pull_request(42, SPECS, orchestrator=object(), timeout_seconds=10)


async def test_resolve_member_result_positional_fallback():
    """Coordinators that omit display_name metadata still resolve positionally."""
    member_results = {
        "auto-1": MemberResult(member_id="auto-1", success=True, output="first", metadata={}),
        "auto-2": MemberResult(member_id="auto-2", success=True, output="second", metadata={}),
    }
    result = TeamResult(
        success=True,
        final_output="",
        member_results=member_results,
        formation=TeamFormation.PARALLEL,
    )
    specs = [ReviewerSpec(provider="a", model="m"), ReviewerSpec(provider="b", model="m")]
    out1, ok1 = review_mod._resolve_member_result(result, specs[0])
    out2, ok2 = review_mod._resolve_member_result(result, specs[1])
    assert (out1, ok1) == ("first", True)
    assert (out2, ok2) == ("second", True)


def test_resolve_member_result_no_results_is_failure():
    class _Empty:
        member_results = {}

    out, ok = review_mod._resolve_member_result(_Empty(), SPECS[0])
    assert out is None and ok is False

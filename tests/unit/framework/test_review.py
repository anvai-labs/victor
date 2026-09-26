"""Unit tests for the independent review panel (victor.framework.review).

No LLM, no network: AgentTeam is faked at the seam and member outputs are
canned TeamResult objects built from the real teams types, so the aggregation
and strict validation contracts are what's under test.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from victor.framework import review as review_mod
from victor.framework.review import (
    APPROVE,
    REQUEST_CHANGES,
    REVIEW_INCOMPLETE,
    ReviewerSpec,
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
        team = cls(_team_result(results))
        team.members = [SimpleNamespace(id=f"rev{i}", name=m.name) for i, m in enumerate(members)]
        return team

    async def run(self):
        return self._result


@pytest.fixture()
def fake_team(monkeypatch):
    FakeAgentTeam.outputs = {}
    FakeAgentTeam.last_members = []
    monkeypatch.setattr(review_mod, "AgentTeam", FakeAgentTeam)
    return FakeAgentTeam


SPECS = [
    ReviewerSpec(provider="zai", model="glm-5.3", name="zai/glm"),
    ReviewerSpec(provider="inferflux", model="qwen3-coder-30b", name="inferflux/qwen"),
]


@pytest.mark.parametrize(
    "text",
    [
        "prefix " + _member_output_json(APPROVE),
        _member_output_json(APPROVE) + " trailing",
        "```json\n" + _member_output_json(APPROVE) + "\n```",
        '{"verdict":"abstain","verdict":"approve","summary":"s","findings":[],"confidence":1}',
    ],
)
def test_parse_reviewer_output_requires_only_json(text):
    assert parse_reviewer_output(text) is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("verdict", "APPROVE"),
        ("summary", 3),
        ("findings", {}),
        ("findings", [False]),
        ("confidence", True),
        ("confidence", "1"),
        ("confidence", float("nan")),
        ("confidence", float("inf")),
        ("confidence", 1.1),
        ("confidence", -0.1),
        ("extra", "field"),
    ],
)
def test_parse_reviewer_output_rejects_wrong_fields(field, value):
    data = json.loads(_member_output_json(APPROVE))
    data[field] = value
    assert parse_reviewer_output(json.dumps(data)) is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("severity", "apocalyptic"),
        ("file", ""),
        ("line", True),
        ("line", 0),
        ("line", "2"),
        ("summary", []),
        ("extra", 1),
    ],
)
def test_parse_reviewer_output_rejects_wrong_finding(field, value):
    finding = {"severity": "major", "file": "f.py", "line": 1, "summary": "bug"}
    finding[field] = value
    assert parse_reviewer_output(_member_output_json(REQUEST_CHANGES, [finding])) is None


def test_parse_reviewer_output_complete_contract():
    assert parse_reviewer_output(_member_output_json(APPROVE))["verdict"] == APPROVE
    data = json.loads(_member_output_json(APPROVE))
    del data["confidence"]
    assert parse_reviewer_output(json.dumps(data)) is None


def test_parse_reviewer_output_garbage():
    assert parse_reviewer_output("no json at all") is None
    assert parse_reviewer_output("{broken") is None
    assert parse_reviewer_output("[1,2,3]") is None
    assert parse_reviewer_output("") is None


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
    assert verdict.verdict == REVIEW_INCOMPLETE


@pytest.mark.parametrize(
    "output,success",
    [
        (_member_output_json(APPROVE), False),
        (_member_output_json(REQUEST_CHANGES), False),
        (_member_output_json("abstain"), True),
    ],
)
async def test_review_diff_every_required_member_must_succeed(fake_team, output, success):
    fake_team.outputs = {
        "zai/glm": (output, success),
        "inferflux/qwen": (_member_output_json(APPROVE), True),
    }
    verdict = await review_diff("diff", SPECS, orchestrator=object())
    assert verdict.verdict == REVIEW_INCOMPLETE


async def test_review_diff_truncation_cannot_approve(fake_team):
    fake_team.outputs = {s.display_name: (_member_output_json(APPROVE), True) for s in SPECS}
    verdict = await review_diff(
        "x" * (review_mod._MAX_DIFF_CHARS + 1), SPECS, orchestrator=object()
    )
    assert verdict.verdict == REVIEW_INCOMPLETE


@pytest.mark.parametrize(
    "specs",
    [
        [ReviewerSpec("", "m")],
        [ReviewerSpec("a", " ")],
        [ReviewerSpec("a", "m", " ")],
        [ReviewerSpec(" a", "m")],
        [ReviewerSpec("a", "m"), ReviewerSpec("a", "m")],
        [ReviewerSpec("a", "m", "same"), ReviewerSpec("b", "m", "same")],
    ],
)
async def test_review_diff_rejects_invalid_identity(fake_team, specs):
    with pytest.raises(ValueError):
        await review_diff("diff", specs, orchestrator=object())
    assert fake_team.last_members == []


async def test_review_diff_members_carry_provider_and_model(fake_team):
    fake_team.outputs = {s.display_name: (_member_output_json(APPROVE), True) for s in SPECS}
    await review_diff("diff", SPECS, orchestrator=object())
    by_name = {m.name: m for m in FakeAgentTeam.last_members}
    assert by_name["zai/glm"].provider == "zai"
    assert by_name["inferflux/qwen"].model == "qwen3-coder-30b"
    assert all(
        m.allowed_tools == [] and m.to_team_member().allowed_tools == []
        for m in FakeAgentTeam.last_members
    )


async def test_review_diff_requires_reviewers():
    with pytest.raises(ValueError):
        await review_diff("diff", [], orchestrator=object())


class _FakeProc:
    def __init__(self, stdout: bytes, returncode: int, stderr: bytes = b""):
        self.stdout = asyncio.StreamReader()
        self.stdout.feed_data(stdout)
        self.stdout.feed_eof()
        self.returncode = returncode
        self.killed = False
        self.waited = False

    async def wait(self):
        self.waited = True
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9
        self.stdout.feed_eof()


BASE = "a" * 40
HEAD = "b" * 40


@pytest.mark.parametrize("change_head", [False, True])
async def test_review_pull_request_fetches_pinned_diff(monkeypatch, change_head):
    commands = []

    async def fake_exec(*cmd, **kwargs):
        commands.append(cmd)
        assert kwargs["stderr"] == asyncio.subprocess.DEVNULL
        if cmd[1:3] == ("pr", "view"):
            return _FakeProc(
                json.dumps(
                    {
                        "baseRefOid": BASE,
                        "headRefOid": HEAD if len(commands) == 1 or not change_head else "c" * 40,
                        "url": "https://github.com/org/repo/pull/42",
                        "changedFiles": 1,
                        "additions": 3,
                        "deletions": 0,
                    }
                ).encode(),
                0,
            )
        assert cmd == (
            "gh",
            "api",
            f"repos/org/repo/compare/{BASE}...{HEAD}",
            "-H",
            "Accept: application/vnd.github.diff",
        )
        return _FakeProc(
            b"diff --git a/f b/f\n--- a/f\n+++ b/f\n@@ -0,0 +1,3 @@\n+diff --git fake\n+@@ fake\n+++literal\n",
            0,
        )

    async def fake_review_diff(diff, specs, **kwargs):
        assert diff.startswith("diff --git")
        return review_mod.ReviewVerdict(APPROVE)

    monkeypatch.setattr(review_mod.asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(review_mod, "review_diff", fake_review_diff)
    if change_head:
        with pytest.raises(RuntimeError, match="changed"):
            await review_mod.review_pull_request(42, SPECS, orchestrator=object())
    else:
        verdict = await review_mod.review_pull_request(42, SPECS, orchestrator=object())
        assert (verdict.repository, verdict.base_sha, verdict.head_sha) == ("org/repo", BASE, HEAD)
        assert len(commands) == 3


@pytest.mark.parametrize("mode", ["failure", "empty", "oversize", "timeout", "cancel"])
async def test_gh_fetch_failure_cleanup(monkeypatch, mode):
    proc = _FakeProc(b"x" if mode == "failure" else b"", 1 if mode == "failure" else 0)
    if mode in ("timeout", "cancel"):
        proc.returncode = None

        async def blocked(_size):
            if mode == "cancel":
                raise asyncio.CancelledError()
            await asyncio.Future()

        proc.stdout.read = blocked
    if mode == "oversize":
        proc = _FakeProc(b"x" * 10, 0)

    async def fake_exec(*cmd, **kwargs):
        return proc

    monkeypatch.setattr(review_mod.asyncio, "create_subprocess_exec", fake_exec)
    expected = asyncio.CancelledError if mode == "cancel" else (RuntimeError, asyncio.TimeoutError)
    with pytest.raises(expected):
        await review_mod._gh_output(["gh", "test"], limit=5, timeout=0.01)
    if mode in ("timeout", "cancel"):
        assert proc.killed and proc.waited


@pytest.mark.parametrize("identity", ["missing", "mismatched", "valid"])
def test_resolve_member_result_uses_canonical_member_id(identity):
    mr = MemberResult(
        member_id="rev1" if identity == "valid" else "other",
        success=True,
        output="out",
        metadata={"display_name": "zai/glm"},
    )
    result = _team_result([])
    result.member_results = {"rev1" if identity != "missing" else "other": mr}
    assert review_mod._resolve_member_result(result, "rev1") == (
        ("out", True) if identity == "valid" else (None, False)
    )
    assert mr.metadata == {"display_name": "zai/glm"}


@pytest.mark.parametrize("reason", ["file_limit", "missing_hunk", "binary"])
async def test_review_pull_request_incomplete_diff_never_dispatches(monkeypatch, reason):
    async def fake_exec(*cmd, **kwargs):
        if cmd[1:3] == ("pr", "view"):
            identity = {
                "baseRefOid": BASE,
                "headRefOid": HEAD,
                "url": "https://github.com/org/repo/pull/42",
                "changedFiles": 300 if reason == "file_limit" else 1,
                "additions": 1,
                "deletions": 0,
            }
            return _FakeProc(json.dumps(identity).encode(), 0)
        return _FakeProc(
            (
                b"diff --git a/f b/f\nBinary files differ\n"
                if reason == "binary"
                else b"diff --git a/f b/f\n"
            ),
            0,
        )

    async def forbidden(*args, **kwargs):
        pytest.fail("incomplete server diff dispatched to reviewers")

    monkeypatch.setattr(review_mod.asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(review_mod, "review_diff", forbidden)
    with pytest.raises(RuntimeError):
        await review_mod.review_pull_request(42, SPECS, orchestrator=object())


async def test_review_diff_cancellation_propagates(fake_team, monkeypatch):
    async def cancelled(self):
        raise asyncio.CancelledError()

    monkeypatch.setattr(fake_team, "run", cancelled)
    with pytest.raises(asyncio.CancelledError):
        await review_diff("diff", SPECS, orchestrator=object())


@pytest.mark.parametrize("mode", ["timeout", "cancel"])
async def test_gh_real_child_is_reaped(monkeypatch, mode):
    import sys

    real_exec = asyncio.create_subprocess_exec
    started = asyncio.Event()
    children = []

    async def tracked_exec(*args, **kwargs):
        child = await real_exec(*args, **kwargs)
        children.append(child)
        started.set()
        return child

    monkeypatch.setattr(review_mod.asyncio, "create_subprocess_exec", tracked_exec)
    task = asyncio.create_task(
        review_mod._gh_output(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            limit=32,
            timeout=0.05 if mode == "timeout" else 10,
        )
    )
    try:
        await asyncio.wait_for(started.wait(), 5)
        if mode == "cancel":
            task.cancel()
        with pytest.raises(asyncio.CancelledError if mode == "cancel" else asyncio.TimeoutError):
            await asyncio.wait_for(task, 10)
        assert children[0].returncode is not None
    finally:
        for child in children:
            if child.returncode is None:
                child.kill()
            await child.wait()


@pytest.mark.parametrize(
    "parameter,value",
    [
        ("tool_budget", True),
        ("tool_budget", 0),
        ("tool_budget", -1),
        ("temperature", True),
        ("temperature", -0.1),
        ("temperature", 2.1),
        ("temperature", float("nan")),
        ("temperature", float("inf")),
        ("timeout_seconds", True),
        ("timeout_seconds", 0),
    ],
)
async def test_review_diff_rejects_execution_options_before_create(fake_team, parameter, value):
    with pytest.raises(ValueError):
        await review_diff("diff", SPECS, orchestrator=object(), **{parameter: value})
    assert fake_team.last_members == []


@pytest.mark.parametrize("timeout", [False, 0, -1, float("inf")])
async def test_pr_timeout_invalid_before_network(monkeypatch, timeout):
    async def forbidden(*args, **kwargs):
        pytest.fail("network before timeout validation")

    monkeypatch.setattr(review_mod.asyncio, "create_subprocess_exec", forbidden)
    with pytest.raises(ValueError):
        await review_mod.review_pull_request(
            42, SPECS, orchestrator=object(), timeout_seconds=timeout
        )

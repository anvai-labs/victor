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

"""Independent multi-provider PR review panel.

Each reviewer is a PARALLEL team member with its own provider/model and
independent message history. No tools are granted: an untrusted diff is input
evidence, not authority to run commands or change the workspace.

Verdicts use a strict JSON contract and aggregate fail closed. A successful
reviewer requesting changes blocks approval; otherwise every required reviewer
must finish successfully and approve the complete diff. These are advisory model
opinions, not proof of correctness or authorization to merge.

Layering: framework surface built on ``AgentTeam`` (framework) only — no
runtime internals. Agents and CLIs consume ``review_diff``/``ReviewVerdict``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from urllib.parse import urlparse
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from victor.framework.teams import AgentTeam, TeamFormation, TeamMemberSpec

logger = logging.getLogger(__name__)

APPROVE = "approve"
REQUEST_CHANGES = "request_changes"
ABSTAIN = "abstain"
REVIEW_INCOMPLETE = "review_incomplete"

_VALID_VERDICTS = {APPROVE, REQUEST_CHANGES, ABSTAIN}
_VALID_SEVERITIES = {"critical", "major", "minor", "nit"}

# Oversized diffs are not dispatched: an incomplete review cannot approve.
_MAX_DIFF_CHARS = 24_000
_MAX_INTENT_CHARS = 2_000

_MAX_OUTPUT_CHARS = 64_000

_REVIEWER_BRIEF = """You are an INDEPENDENT adversarial code reviewer. You did not write this change
and you owe its author nothing. Attack the diff like a hostile reviewer whose job
is to find reasons it must NOT merge. Do not soften findings; do not approve
unless every checklist item is resolved or genuinely not applicable.

Checklist:
1. Correctness: logic errors, unhandled cases, wrong assumptions.
2. Security: injection, secrets, authz, unsafe deserialization, SSRF.
3. Concurrency and shared state: races, reentrancy, ordering, cleanup on failure.
4. Layering and API contracts: boundary violations, breaking changes, drift.
5. Tests: does the change pin its claims with positive AND negative tests?
   Would the tests catch a regression of the fixed behavior?
6. Hidden scope: anything in the diff that is not described by its stated intent.

Rules: cite file:line for every finding; severity must be one of
critical|major|minor|nit; if you cannot review responsibly (for example the diff
is truncated or you lack context), verdict must be "abstain" and say why.

Respond with ONLY this JSON — no prose, no markdown fences:
{{"verdict": "approve|request_changes|abstain",
  "summary": "<one short paragraph>",
  "findings": [{{"severity": "critical|major|minor|nit", "file": "<path>",
                 "line": <int or null>, "summary": "<what and why"}}],
  "confidence": <0.0-1.0>}}

State intent of the change under review:
{intent}

Diff:
```diff
{diff}
```"""


@dataclass(frozen=True)
class ReviewerSpec:
    """One panel member: an isolated agent pinned to a provider/model."""

    provider: str
    model: str
    name: str = ""

    @property
    def display_name(self) -> str:
        return self.name or f"{self.provider}/{self.model}"


@dataclass(frozen=True)
class Finding:
    severity: str
    file: str
    line: Optional[int]
    summary: str
    reviewer: str


@dataclass
class ReviewVerdict:
    """Aggregated panel outcome. ``verdict`` is one of APPROVE,
    REQUEST_CHANGES, REVIEW_INCOMPLETE."""

    verdict: str
    findings: list[Finding] = field(default_factory=list)
    reviewer_verdicts: dict[str, str] = field(default_factory=dict)
    reviewer_summaries: dict[str, str] = field(default_factory=dict)
    blocked_by: list[str] = field(default_factory=list)
    repository: Optional[str] = None
    base_sha: Optional[str] = None
    head_sha: Optional[str] = None

    @property
    def approved(self) -> bool:
        return self.verdict == APPROVE


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def parse_reviewer_output(text: str) -> Optional[dict[str, Any]]:
    """Validate one complete JSON verdict; never extract prose or coerce fields."""
    if not isinstance(text, str) or len(text) > _MAX_OUTPUT_CHARS:
        return None
    try:
        data = json.loads(text, object_pairs_hook=_unique_object)
    except (ValueError, RecursionError):
        return None
    if not isinstance(data, dict) or set(data) != {"verdict", "summary", "findings", "confidence"}:
        return None
    if not isinstance(data["verdict"], str) or data["verdict"] not in _VALID_VERDICTS:
        return None
    if not isinstance(data["summary"], str) or not data["summary"].strip():
        return None
    confidence = data["confidence"]
    if (
        type(confidence) not in (int, float)
        or not 0 <= confidence <= 1
        or not math.isfinite(confidence)
    ):
        return None
    if not isinstance(data["findings"], list):
        return None
    for finding in data["findings"]:
        if not isinstance(finding, dict) or set(finding) != {"severity", "file", "line", "summary"}:
            return None
        if not isinstance(finding["severity"], str) or finding["severity"] not in _VALID_SEVERITIES:
            return None
        if any(
            not isinstance(finding[key], str) or not finding[key].strip()
            for key in ("file", "summary")
        ):
            return None
        line = finding["line"]
        if line is not None and (type(line) is not int or line < 1):
            return None
    # An approval containing blocking findings contradicts its own contract.
    if data["verdict"] == APPROVE and any(
        f["severity"] in {"critical", "major"} for f in data["findings"]
    ):
        return None
    return data


def _validate_reviewers(reviewers: Sequence[ReviewerSpec]) -> None:
    if not reviewers:
        raise ValueError("review_diff requires at least one ReviewerSpec")
    names: set[str] = set()
    for spec in reviewers:
        if not isinstance(spec, ReviewerSpec):
            raise ValueError("expected ReviewerSpec")
        if any(
            not isinstance(v, str) or not v.strip() or v != v.strip()
            for v in (spec.provider, spec.model)
        ):
            raise ValueError("reviewer provider and model must be nonempty, trimmed strings")
        if (
            not isinstance(spec.name, str)
            or (spec.name and not spec.name.strip())
            or spec.name != spec.name.strip()
        ):
            raise ValueError("reviewer name must be empty or a trimmed, nonempty string")
        if spec.display_name in names:
            raise ValueError("reviewer identities must be unique")
        names.add(spec.display_name)


def _validate_timeout(timeout_seconds: int) -> None:
    if type(timeout_seconds) is not int or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be a positive integer")


def _member_goal(intent: str, diff: str) -> str:
    return _REVIEWER_BRIEF.format(
        intent=intent or "(not stated — infer from the diff, and lower confidence)",
        diff=diff,
    )


def _resolve_member_result(result: Any, member_id: str) -> tuple[Optional[str], bool]:
    """Resolve only the configured team member ID; metadata cannot claim identity."""
    member_results = getattr(result, "member_results", None)
    if not isinstance(member_results, dict):
        return None, False
    member = member_results.get(member_id)
    if member is None or getattr(member, "member_id", None) != member_id:
        return None, False
    return getattr(member, "output", None), getattr(member, "success", None) is True


async def review_diff(
    diff: str,
    reviewers: Sequence[ReviewerSpec],
    *,
    orchestrator: Any,
    intent: str = "",
    timeout_seconds: int = 1200,
    temperature: float = 0.1,
    tool_budget: int = 8,
) -> ReviewVerdict:
    """Run the reviewer panel over ``diff`` and aggregate a fail-closed verdict.

    Each reviewer runs as a PARALLEL team member with its own
    provider/model and no tools. Reviewer failure or unparseable output counts as an
    abstention, never as an approval.
    """
    _validate_reviewers(reviewers)
    _validate_timeout(timeout_seconds)
    if type(tool_budget) is not int or tool_budget <= 0:
        raise ValueError("tool_budget must be a positive integer")
    if (
        type(temperature) not in (int, float)
        or not 0 <= temperature <= 2
        or not math.isfinite(temperature)
    ):
        raise ValueError("temperature must be finite and between zero and two")
    if not isinstance(intent, str) or len(intent) > _MAX_INTENT_CHARS:
        raise ValueError("intent exceeds review prompt budget")
    if not isinstance(diff, str) or not diff.strip() or len(diff) > _MAX_DIFF_CHARS:
        return ReviewVerdict(
            verdict=REVIEW_INCOMPLETE,
            reviewer_verdicts={s.display_name: ABSTAIN for s in reviewers},
            reviewer_summaries={
                s.display_name: "Complete bounded diff required." for s in reviewers
            },
        )

    goal = _member_goal(intent, diff)
    members = [
        TeamMemberSpec(
            role="reviewer",
            name=r.display_name,
            goal=goal,
            provider=r.provider,
            model=r.model,
            temperature=temperature,
            tool_budget=tool_budget,
            allowed_tools=[],
        )
        for r in reviewers
    ]

    team = await AgentTeam.create(
        orchestrator=orchestrator,
        name="pr-review",
        goal="Independent adversarial review of a proposed change.",
        members=members,
        formation=TeamFormation.PARALLEL,
        shared_context={"capture_member_usage": True},
        timeout_seconds=timeout_seconds,
    )
    member_ids = {member.name: member.id for member in team.members}
    if (
        len(member_ids) != len(reviewers)
        or set(member_ids) != {s.display_name for s in reviewers}
        or len(set(member_ids.values())) != len(reviewers)
    ):
        raise RuntimeError("review team identity mismatch")
    result = await asyncio.wait_for(team.run(), timeout=timeout_seconds)

    verdict = ReviewVerdict(verdict=REVIEW_INCOMPLETE)
    all_approve = True
    for spec in reviewers:
        output, success = _resolve_member_result(result, member_ids[spec.display_name])
        parsed = parse_reviewer_output(output) if output else None
        if not success:
            logger.warning("review panel: reviewer %s failed to produce output", spec.display_name)
            verdict.reviewer_verdicts[spec.display_name] = ABSTAIN
            all_approve = False
            continue
        if parsed is None:
            logger.warning(
                "review panel: reviewer %s returned unparseable output", spec.display_name
            )
            verdict.reviewer_verdicts[spec.display_name] = ABSTAIN
            all_approve = False
            continue

        v = parsed["verdict"]
        verdict.reviewer_verdicts[spec.display_name] = v
        verdict.reviewer_summaries[spec.display_name] = parsed["summary"]
        for f in parsed["findings"]:
            verdict.findings.append(
                Finding(
                    severity=f["severity"],
                    file=f["file"],
                    line=f["line"],
                    summary=f["summary"],
                    reviewer=spec.display_name,
                )
            )
        if v == REQUEST_CHANGES:
            verdict.blocked_by.append(spec.display_name)
        if v != APPROVE:
            all_approve = False

    if verdict.blocked_by:
        verdict.verdict = REQUEST_CHANGES
    elif all_approve and getattr(result, "success", None) is True:
        verdict.verdict = APPROVE
    else:
        verdict.verdict = REVIEW_INCOMPLETE
    return verdict


async def _gh_output(cmd: list[str], *, limit: int, timeout: float = 120) -> str:
    """Read bounded stdout, discard potentially sensitive stderr, reap on failure."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )

    async def read() -> str:
        if proc.stdout is None:
            raise RuntimeError("gh stdout unavailable")
        output = bytearray()
        while True:
            chunk = await proc.stdout.read(min(65536, limit + 1 - len(output)))
            if not chunk:
                break
            output.extend(chunk)
            if len(output) > limit:
                raise RuntimeError("gh output exceeds review limit")
        await proc.wait()
        if proc.returncode != 0:
            raise RuntimeError("gh review fetch failed")
        text = output.decode("utf-8", errors="strict")
        if not text.strip():
            raise RuntimeError("gh returned an empty diff or identity")
        return text

    try:
        return await asyncio.wait_for(read(), timeout=timeout)
    except BaseException as primary:
        if proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            except Exception as cleanup:
                primary.add_note(f"gh signal failed: {type(cleanup).__name__}")
        # asyncio has no public Process.close. Closing its owned transport also
        # releases stdout if an exited child left an inherited pipe open.
        transport = getattr(proc, "_transport", None)
        if transport is not None:
            try:
                transport.close()
            except Exception as cleanup:
                primary.add_note(f"gh transport cleanup failed: {type(cleanup).__name__}")
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except Exception as cleanup:
            primary.add_note(f"gh cleanup failed: {type(cleanup).__name__}")
        raise


async def review_pull_request(
    pr_number: int,
    reviewers: Sequence[ReviewerSpec],
    *,
    orchestrator: Any,
    repo: Optional[str] = None,
    intent: str = "",
    timeout_seconds: int = 1200,
) -> ReviewVerdict:
    """Review an immutable GitHub base/head pair, rejecting a moved PR afterwards.

    The returned identity binds this advisory result, not a later merge operation.
    A caller must compare head_sha again before applying it to a mutable PR.
    """
    _validate_reviewers(reviewers)
    _validate_timeout(timeout_seconds)
    if not isinstance(intent, str) or len(intent) > _MAX_INTENT_CHARS:
        raise ValueError("intent exceeds review prompt budget")
    if type(pr_number) is not int or pr_number <= 0:
        raise ValueError("positive PR number required")
    cmd = [
        "gh",
        "pr",
        "view",
        str(pr_number),
        "--json",
        "baseRefOid,headRefOid,url,changedFiles,additions,deletions",
    ]
    if repo:
        cmd += ["-R", repo]

    async def identity() -> tuple[str, str, str, int, int, int]:
        data = json.loads(await _gh_output(cmd, limit=16_384), object_pairs_hook=_unique_object)
        if not isinstance(data, dict):
            raise RuntimeError("invalid PR identity")
        base, head, url = data.get("baseRefOid"), data.get("headRefOid"), data.get("url")
        if not all(
            isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{40}", sha) for sha in (base, head)
        ) or not isinstance(url, str):
            raise RuntimeError("invalid PR identity")
        parsed = urlparse(url)
        match = re.fullmatch(
            r"/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)/pull/" + str(pr_number), parsed.path
        )
        if (
            parsed.scheme != "https"
            or parsed.netloc != "github.com"
            or parsed.query
            or parsed.fragment
            or match is None
        ):
            raise RuntimeError("unsupported PR repository identity")
        counts = [data.get(key) for key in ("changedFiles", "additions", "deletions")]
        if any(type(value) is not int or value < 0 for value in counts) or not 0 < counts[0] < 300:
            raise RuntimeError("PR exceeds bounded diff completeness contract")
        return match.group(1), base, head, counts[0], counts[1], counts[2]

    pinned = await identity()
    repository, base_sha, head_sha, files, additions, deletions = pinned
    diff = await _gh_output(
        [
            "gh",
            "api",
            f"repos/{repository}/compare/{base_sha}...{head_sha}",
            "-H",
            "Accept: application/vnd.github.diff",
        ],
        limit=_MAX_DIFF_CHARS * 4,
    )
    # GitHub limits displayed/API diffs. Compare exact PR statistics with the
    # immutable patch rather than treating a size-bounded response as complete.
    seen_files = added = deleted = 0
    in_hunk = False
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            seen_files += 1
            in_hunk = False
        elif line.startswith("@@ "):
            in_hunk = True
        elif line.startswith(("Binary files ", "GIT binary patch")):
            raise RuntimeError("binary diff requires separate review")
        elif in_hunk:
            added += line.startswith("+")
            deleted += line.startswith("-")
    if (seen_files, added, deleted) != (files, additions, deletions):
        raise RuntimeError("incomplete PR diff")
    verdict = await review_diff(
        diff,
        reviewers,
        orchestrator=orchestrator,
        intent=intent,
        timeout_seconds=timeout_seconds,
    )
    if await identity() != pinned:
        raise RuntimeError("PR identity changed during review")
    verdict.repository = repository
    verdict.base_sha = base_sha
    verdict.head_sha = head_sha
    return verdict

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

Each reviewer is an ISOLATED team member with its own provider/model and no
shared context with the caller — a hostile second pair of eyes, not a
self-review. Agent speed makes fresh-agent review cheaper than waiting on a
human for every PR; heterogeneity (different providers) keeps the panel from
sharing one model's blind spots.

Verdicts are strict-JSON, coerced, and aggregated FAIL-CLOSED:
- any reviewer says request_changes  → REQUEST_CHANGES
- no blocker and at least one approve → APPROVE
- no reviewer finished (all abstain/fail) → REVIEW_INCOMPLETE

Layering: framework surface built on ``AgentTeam`` (framework) only — no
runtime internals. Agents and CLIs consume ``review_diff``/``ReviewVerdict``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
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

# Diffs above this are truncated for the reviewer prompt; the reviewer is told
# the diff was truncated so it can abstain rather than review blind.
_MAX_DIFF_CHARS = 160_000

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

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

    @property
    def approved(self) -> bool:
        return self.verdict == APPROVE


def parse_reviewer_output(text: str) -> Optional[dict[str, Any]]:
    """Extract and coerce a reviewer's strict-JSON verdict.

    Returns a normalized dict (verdict/summary/findings/confidence) or None when
    the output is unparseable — callers treat None as an abstention.
    """
    match = _JSON_RE.search(text or "")
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None

    verdict = str(data.get("verdict") or "").strip().lower()
    if verdict not in _VALID_VERDICTS:
        return None

    findings: list[dict[str, Any]] = []
    for raw in data.get("findings") or []:
        if not isinstance(raw, dict):
            continue
        severity = str(raw.get("severity") or "minor").strip().lower()
        if severity not in _VALID_SEVERITIES:
            severity = "minor"
        line = raw.get("line")
        findings.append(
            {
                "severity": severity,
                "file": str(raw.get("file") or "").strip(),
                "line": int(line) if isinstance(line, int) else None,
                "summary": str(raw.get("summary") or "").strip(),
            }
        )

    try:
        confidence = float(data.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5

    return {
        "verdict": verdict,
        "summary": str(data.get("summary") or "").strip(),
        "findings": findings,
        "confidence": max(0.0, min(1.0, confidence)),
    }


def _member_goal(intent: str, diff: str) -> str:
    truncated = False
    if len(diff) > _MAX_DIFF_CHARS:
        diff = diff[:_MAX_DIFF_CHARS]
        truncated = True
    body = _REVIEWER_BRIEF.format(
        intent=intent or "(not stated — infer from the diff, and lower confidence)",
        diff=diff + ("\n... (truncated)" if truncated else ""),
    )
    if truncated:
        body += (
            "\n\nNOTE: the diff was truncated. If the unseen part could change "
            'your verdict, respond "abstain".'
        )
    return body


def _resolve_member_result(result: Any, spec: ReviewerSpec) -> tuple[Optional[str], bool]:
    """Return (output_text, member_success) for one spec from a TeamResult.

    Member ids are auto-generated, so match on the identity metadata the
    coordinator copies into ``MemberResult.metadata`` (display_name/member_id),
    falling back to positional order for coordinators that preserve it.
    """
    member_results = getattr(result, "member_results", None)
    if not member_results:
        return None, False

    display = spec.display_name
    for member_id, mr in member_results.items():
        metadata = getattr(mr, "metadata", None) or {}
        if display in (metadata.get("display_name"), metadata.get("member_id"), member_id):
            return getattr(mr, "output", None), bool(getattr(mr, "success", False))

    # Positional fallback: first result not yet claimed by an earlier spec.
    for member_id, mr in member_results.items():
        metadata = getattr(mr, "metadata", None)
        if isinstance(metadata, dict) and not metadata.get("_claimed"):
            metadata["_claimed"] = True
            return getattr(mr, "output", None), bool(getattr(mr, "success", False))
    return None, False


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

    Each reviewer runs as an isolated PARALLEL team member with its own
    provider/model. Reviewer failure or unparseable output counts as an
    abstention, never as an approval.
    """
    if not reviewers:
        raise ValueError("review_diff requires at least one ReviewerSpec")

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
    result = await team.run()

    verdict = ReviewVerdict(verdict=REVIEW_INCOMPLETE)
    any_approve = False
    for spec in reviewers:
        output, success = _resolve_member_result(result, spec)
        parsed = parse_reviewer_output(output) if output else None
        if not success and parsed is None:
            logger.warning("review panel: reviewer %s failed to produce output", spec.display_name)
            verdict.reviewer_verdicts[spec.display_name] = ABSTAIN
            continue
        if parsed is None:
            logger.warning(
                "review panel: reviewer %s returned unparseable output", spec.display_name
            )
            verdict.reviewer_verdicts[spec.display_name] = ABSTAIN
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
        elif v == APPROVE:
            any_approve = True

    if verdict.blocked_by:
        verdict.verdict = REQUEST_CHANGES
    elif any_approve:
        verdict.verdict = APPROVE
    else:
        verdict.verdict = REVIEW_INCOMPLETE
    return verdict


async def review_pull_request(
    pr_number: int,
    reviewers: Sequence[ReviewerSpec],
    *,
    orchestrator: Any,
    repo: Optional[str] = None,
    intent: str = "",
    timeout_seconds: int = 1200,
) -> ReviewVerdict:
    """Fetch a PR diff with the gh CLI and run the reviewer panel over it."""
    import subprocess  # local import: only needed on this path

    cmd = ["gh", "pr", "diff", str(pr_number)]
    if repo:
        cmd += ["-R", repo]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(
            f"gh pr diff {pr_number} failed ({proc.returncode}): "
            f"{stderr.decode(errors='replace').strip()[:300]}"
        )
    diff = stdout.decode(errors="replace")
    if not diff.strip():
        raise RuntimeError(f"gh pr diff {pr_number} returned an empty diff")
    return await review_diff(
        diff,
        reviewers,
        orchestrator=orchestrator,
        intent=intent,
        timeout_seconds=timeout_seconds,
    )

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

"""``victor review`` — independent multi-provider PR review from the CLI.

Runs a panel of isolated reviewer agents (each pinned to its own
provider/model) over a pull-request diff and prints the aggregated verdict.
Exit codes: 0 approved, 1 changes requested, 2 review incomplete — usable as
a CI/automation gate.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import List, Optional

import typer

from victor.framework.review import (
    APPROVE,
    REQUEST_CHANGES,
    ReviewerSpec,
    ReviewVerdict,
    _validate_reviewers,
    _MAX_DIFF_CHARS,
    review_diff,
    review_pull_request,
)

review_app = typer.Typer(add_completion=False, help="Independent multi-provider PR review.")


def _parse_reviewers(raw: str) -> List[ReviewerSpec]:
    if not raw.strip():
        return []
    specs: List[ReviewerSpec] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            raise typer.BadParameter("empty reviewer entry")
        if ":" not in part:
            raise typer.BadParameter(f"reviewer {part!r} must be provider:model (e.g. zai:glm-5.3)")
        provider, model = part.split(":", 1)
        specs.append(ReviewerSpec(provider=provider.strip(), model=model.strip()))
    try:
        _validate_reviewers(specs)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    return specs


def _default_reviewers(provider: Optional[str], model: Optional[str]) -> List[ReviewerSpec]:
    if provider is not None or model is not None:
        if provider is None or model is None:
            raise typer.BadParameter("--provider and --model must be supplied together")
        return _parse_reviewers(f"{provider}:{model}")
    from victor.config.settings import load_settings

    settings = load_settings(fresh=True)
    p = getattr(settings.provider, "default_provider", None)
    m = getattr(settings.provider, "default_model", None)
    if not p or not m:
        raise typer.BadParameter(
            "No default provider/model configured — pass --reviewers provider:model"
        )
    return _parse_reviewers(f"{p}:{m}")


def _read_diff_file(path: Path) -> str:
    """Bound allocation and preserve the exact UTF-8 evidence before Agent.create."""
    try:
        with path.open("rb") as stream:
            raw = stream.read(_MAX_DIFF_CHARS * 4 + 1)
        if len(raw) > _MAX_DIFF_CHARS * 4:
            raise ValueError("oversized diff")
        text = raw.decode("utf-8", errors="strict")
        if not text.strip() or len(text) > _MAX_DIFF_CHARS:
            raise ValueError("empty or oversized diff")
        return text
    except (OSError, ValueError) as exc:
        typer.echo("review incomplete: diff file must be bounded, nonempty UTF-8", err=True)
        raise typer.Exit(2) from exc


def _print_verdict(verdict: ReviewVerdict) -> None:
    typer.echo("\n=== PR review panel ===")
    if verdict.head_sha:
        typer.echo(f"Reviewed {verdict.repository}: {verdict.base_sha}...{verdict.head_sha}")
    for reviewer, v in verdict.reviewer_verdicts.items():
        typer.echo(f"  {reviewer:<40} {v}")
    typer.echo(f"\nVERDICT: {verdict.verdict}")
    if verdict.blocked_by:
        typer.echo(f"blocked by: {', '.join(verdict.blocked_by)}")
    if verdict.findings:
        typer.echo("\nFindings:")
        for f in verdict.findings:
            loc = f"{f.file}" + (f":{f.line}" if f.line else "")
            typer.echo(f"  [{f.severity:>8}] {loc} — {f.summary}  ({f.reviewer})")
    for reviewer, summary in verdict.reviewer_summaries.items():
        if summary:
            typer.echo(f"\n{reviewer}: {summary}")


async def _run_review(
    specs: List[ReviewerSpec],
    *,
    pr_number: int,
    repo: Optional[str],
    diff_text: Optional[str],
    intent: str,
    timeout_seconds: int,
) -> int:
    from victor.framework.agent import Agent

    agent = await Agent.create()
    try:
        orchestrator = agent.get_orchestrator()
        if diff_text is not None:
            verdict = await review_diff(
                diff_text,
                specs,
                orchestrator=orchestrator,
                intent=intent,
                timeout_seconds=timeout_seconds,
            )
        else:
            verdict = await review_pull_request(
                pr_number,
                specs,
                orchestrator=orchestrator,
                repo=repo,
                intent=intent,
                timeout_seconds=timeout_seconds,
            )
    finally:
        close = getattr(agent, "close", None)
        if close is not None:
            await close()

    _print_verdict(verdict)
    if verdict.verdict == APPROVE:
        return 0
    if verdict.verdict == REQUEST_CHANGES:
        return 1
    return 2


@review_app.command("pr")
def review_pr(
    pr_number: int = typer.Argument(..., help="Pull request number."),
    repo: Optional[str] = typer.Option(None, "--repo", "-R", help="OWNER/REPO for gh."),
    reviewers: str = typer.Option(
        "",
        "--reviewers",
        help="Comma-separated provider:model pairs (e.g. zai:glm-5.3,inferflux:qwen3-coder-30b). "
        "Empty = this profile's default provider/model.",
    ),
    intent: str = typer.Option("", "--intent", help="Stated intent of the change."),
    diff_file: Optional[Path] = typer.Option(
        None, "--diff-file", help="Review this diff file instead of fetching a PR."
    ),
    timeout_seconds: int = typer.Option(1200, "--timeout", min=1, help="Panel wall-clock budget."),
    provider: Optional[str] = typer.Option(None, "--provider", help="Fallback panel provider."),
    model: Optional[str] = typer.Option(None, "--model", help="Fallback panel model."),
) -> None:
    """Review a PR with a panel of independent agents (fresh context each)."""
    specs = _parse_reviewers(reviewers) or _default_reviewers(provider, model)
    diff_text = _read_diff_file(diff_file) if diff_file else None

    async def _run() -> int:
        return await _run_review(
            specs,
            pr_number=pr_number,
            repo=repo,
            diff_text=diff_text,
            intent=intent,
            timeout_seconds=timeout_seconds,
        )

    try:
        code = asyncio.run(asyncio.wait_for(_run(), timeout=timeout_seconds + 180))
    except asyncio.TimeoutError:
        typer.echo("review panel exceeded its wall-clock budget", err=True)
        raise typer.Exit(2)
    except (RuntimeError, ValueError, OSError) as exc:
        typer.echo(f"review incomplete: {type(exc).__name__}", err=True)
        raise typer.Exit(2) from exc
    raise typer.Exit(code)


@review_app.command("diff")
def review_diff_file(
    diff_file: Path = typer.Argument(..., help="Path to a unified diff.", exists=True),
    reviewers: str = typer.Option("", "--reviewers"),
    intent: str = typer.Option("", "--intent"),
    timeout_seconds: int = typer.Option(1200, "--timeout", min=1),
    provider: Optional[str] = typer.Option(None, "--provider"),
    model: Optional[str] = typer.Option(None, "--model"),
) -> None:
    """Review a diff file with a panel of independent agents."""
    specs = _parse_reviewers(reviewers) or _default_reviewers(provider, model)
    diff_text = _read_diff_file(diff_file)

    async def _run() -> int:
        return await _run_review(
            specs,
            pr_number=0,
            repo=None,
            diff_text=diff_text,
            intent=intent,
            timeout_seconds=timeout_seconds,
        )

    try:
        code = asyncio.run(asyncio.wait_for(_run(), timeout=timeout_seconds + 180))
    except asyncio.TimeoutError:
        typer.echo("review panel exceeded its wall-clock budget", err=True)
        raise typer.Exit(2)
    except (RuntimeError, ValueError, OSError) as exc:
        typer.echo(f"review incomplete: {type(exc).__name__}", err=True)
        raise typer.Exit(2) from exc
    raise typer.Exit(code)

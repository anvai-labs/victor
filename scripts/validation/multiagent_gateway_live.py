#!/usr/bin/env python3
"""Gateway review, pipeline pause/resume, selection, and usage reconciliation.

The optional mixed mode uses configured InferFlux members and a ZAI reviewer.
The approval signal precedes the reviewer (ZAI-only) or reviser (mixed).
All completed member work uses the real configured Sandhi gateway.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, cast
from unittest.mock import patch
import uuid

WORKTREE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKTREE))

from aiohttp import ClientSession, web
from victor.agent.member_approval_context import MemberApprovalPause
from victor.agent.subagents.base import SubAgent
from victor.framework import Agent
from victor.framework.graph_checkpoint import MemoryCheckpointer
from victor.framework.hitl import ApprovalRequest
from victor.framework.member_event_sink import MemberEventSink, current_member_sink
from victor.framework.teams import AgentTeam, TeamFormation, TeamMemberSpec
from victor.teams.types import MemberResult, TeamMember
from victor.teams.unified_coordinator import StateGraphNodeConfig, UnifiedTeamCoordinator

MEMBER_NAMES = ("writer", "reviewer", "reviser", "selecteda", "selectedb", "fallbacka", "fallbackb")
TASK_CONTRACT_VERSION = 2
NUMERIC_DOMAIN = (
    "Inputs are Python ints or floats that are integer multiples of 0.25 in [-1024, 1024]. "
    "For the additive invariant, x, y, and x+y must all be in that domain. "
    "Booleans, strings, sequences, non-finite values, and overflow are outside the contract."
)


def failure(evidence: dict[str, Any], check: str, **details: Any) -> None:
    """Record a failed check without preventing independent checks from running."""
    evidence["failures"].append({"check": check, **details})


async def collect_acceptance(
    *,
    root: Path,
    python: Path,
    mixed: bool,
    results: list[MemberResult],
    client: ClientSession,
    gateway_url: str,
    admin: str,
    evidence: dict[str, Any],
) -> None:
    """Collect independent acceptance results even after an execution failure.

    Exceptions are recorded by type: upstream errors can include private request headers.
    Raw run artifacts stay in the private output directory, not in committed evidence.
    """
    if mixed:
        try:
            review = json.loads((root / "review.json").read_text())
            evidence["review"] = review
            valid = (
                isinstance(review, dict)
                and set(review) == {"verdict", "findings"}
                and review["verdict"] == "approved"
                and isinstance(review["findings"], list)
                and all(isinstance(item, str) for item in review["findings"])
            )
            if not valid:
                failure(evidence, "review", reason="Review must be a valid approved verdict")
        except (OSError, ValueError) as exc:
            failure(evidence, "review", error_type=type(exc).__name__)

    reconciled = evidence["usage_reconciliation"] = []
    for member in results:
        entry: dict[str, Any] = {"member_id": member.member_id, "passed": False}
        reconciled.append(entry)
        try:
            session = member.metadata["session_id"]
            entry.update(session_id=session, victor=member.metadata["usage"])
            async with client.get(
                gateway_url + "/admin/usage/run/" + session,
                headers={"Authorization": f"Bearer {admin}"},
            ) as response:
                entry["http_status"] = response.status
                if response.status != 200:
                    raise ValueError("Run lookup failed")
                total = (await response.json())["run"]["total"]
            entry["sandhi"] = total
            usage = entry["victor"]
            gateway_input = (
                total["tokens_in"] + total["cache_read_tokens"] + total["cache_creation_tokens"]
            )
            entry["passed"] = (
                usage["input_tokens"] == gateway_input
                and usage["output_tokens"] == total["tokens_out"]
                and usage["total_tokens"] == gateway_input + total["tokens_out"]
            )
        except Exception as exc:
            entry["error_type"] = type(exc).__name__
        if not entry["passed"]:
            failure(evidence, "usage", member_id=member.member_id)

    required = {filename for name in MEMBER_NAMES for filename in (f"{name}.py", f"test_{name}.py")}
    files = sorted(p.name for p in root.glob("*.py") if p.is_file())
    missing = sorted(required - set(files))
    evidence.update(deliverables=files, missing_deliverables=missing)
    if missing or len(list(root.glob("test_*.py"))) != len(MEMBER_NAMES):
        failure(evidence, "deliverables", missing=missing)
    try:
        tests = await asyncio.to_thread(
            subprocess.run,
            [str(python), "-m", "pytest", "-q", str(root)],
            cwd=root,
            text=True,
            capture_output=True,
            timeout=60,
        )
        evidence["pytest"] = {
            "returncode": tests.returncode,
            "stdout": tests.stdout,
            "stderr": tests.stderr,
            "timeout": False,
        }
        if tests.returncode != 0:
            failure(evidence, "pytest", returncode=tests.returncode)
    except (OSError, subprocess.TimeoutExpired) as exc:
        evidence["pytest"] = {
            "returncode": None,
            "error_type": type(exc).__name__,
            "timeout": isinstance(exc, subprocess.TimeoutExpired),
        }
        failure(evidence, "pytest", error_type=type(exc).__name__)


async def validate(args: argparse.Namespace) -> int:
    root = Path(args.output_dir).resolve() / f"gateway-{uuid.uuid4().hex[:10]}"
    root.mkdir(parents=True, mode=0o700)
    settings = json.loads((args.gateway_state / "client.json").read_text())
    admin = (args.gateway_state / "admin-token").read_text().strip()
    python = WORKTREE / ".venv-codesign/bin/python"
    observed: list[dict[str, Any]] = []
    evidence: dict[str, Any] = {
        "provider": "zai",
        "model": "glm-5.3",
        "scope": "ZAI-only; no InferFlux claim",
        "task_contract_version": TASK_CONTRACT_VERSION,
        "numeric_domain": NUMERIC_DOMAIN,
        "failures": [],
        "passed": False,
    }
    results: list[MemberResult] = []
    configured_members: list[TeamMember] = []
    previous_cwd = Path.cwd()
    gateway_env = (
        "SANDHI_GATEWAY_URL",
        "SANDHI_GATEWAY_VIRTUAL_KEY_ZAI",
        "SANDHI_GATEWAY_VIRTUAL_KEY_INFERFLUX",
    )
    previous_env = {key: os.environ.get(key) for key in gateway_env}
    runner: web.AppRunner | None = None
    started = time.monotonic()
    async with ClientSession() as client:

        try:

            async def forward(request: web.Request) -> web.Response:
                body = await request.read()
                if request.path.endswith("/chat/completions"):
                    observed.append(
                        {
                            "session_id": request.headers.get("x-sandhi-session"),
                            "run_id": request.headers.get("x-sandhi-run-id"),
                            "model": json.loads(body).get("model"),
                            "reasoning_effort_present": "reasoning_effort" in json.loads(body),
                        }
                    )
                headers = {
                    k: v
                    for k, v in request.headers.items()
                    if k.lower() not in {"host", "content-length", "transfer-encoding"}
                }
                headers["Accept-Encoding"] = "identity"
                async with client.request(
                    request.method, settings["url"] + request.path_qs, data=body, headers=headers
                ) as upstream:
                    response = web.Response(
                        status=upstream.status,
                        body=await upstream.read(),
                        content_type="application/json",
                    )
                    return response

            app = web.Application()
            app.router.add_route("*", "/{tail:.*}", forward)
            runner = web.AppRunner(app)
            await runner.setup()
            await web.TCPSite(runner, "127.0.0.1", args.proxy_port).start()
            os.environ["SANDHI_GATEWAY_URL"] = f"http://127.0.0.1:{args.proxy_port}"
            os.environ["SANDHI_GATEWAY_VIRTUAL_KEY_ZAI"] = settings["virtual_key"]
            local = None
            if args.mixed:
                local = json.loads((args.gateway_state / "inferflux.json").read_text())
                os.environ["SANDHI_GATEWAY_VIRTUAL_KEY_INFERFLUX"] = local["virtual_key"]
                evidence.update(
                    provider="inferflux+zai",
                    model="qwen3-coder-30b+glm-5.3",
                    scope="Configured InferFlux members plus ZAI reviewer through one Sandhi gateway",
                )
            os.chdir(root)
            agent = await Agent.create(
                provider="zai",
                model="glm-5.3",
                workspace=str(root),
                enable_observability=False,
                session_id=root.name,
            )
            orchestrator = agent.get_orchestrator()
            assert orchestrator.provider.extra_config["gateway"]["url"].endswith(
                str(args.proxy_port)
            )

            def spec(name: str) -> TeamMemberSpec:
                return TeamMemberSpec(
                    role="executor",
                    name=name,
                    provider="inferflux" if args.mixed and name != "reviewer" else "zai",
                    model="qwen3-coder-30b" if args.mixed and name != "reviewer" else "glm-5.3",
                    reasoning_effort="high" if args.mixed else None,
                    goal=f"Assigned member: {name}. Complete these steps in order: "
                    f"1. Write {root}/{name}.py with function {name}(x) returning x * 2. "
                    f"Numeric task contract v{TASK_CONTRACT_VERSION}: {NUMERIC_DOMAIN} "
                    f"2. Write {root}/test_{name}.py importing that function and defining "
                    f"def test_{name}(): assert {name}(4) == 8. "
                    f"3. Run {python} -m pytest {root}/test_{name}.py -q. "
                    "Both files must exist and one test must pass. Use write twice then shell. "
                    "Run shell with readonly=False. Never repeat a successful write; move to the next step. "
                    "Return file references and test result.",
                    allowed_tools=["read", "write", "shell"],
                    tool_budget=12,
                    max_iterations=12,
                )

            reviewer = spec("reviewer")
            if args.mixed:
                reviewer.goal = (
                    f"Review {root}/writer.py and its tests. Think carefully about zero, negative, "
                    "and fractional inputs and the invariant f(x+y)=f(x)+f(y). "
                    f"Numeric task contract v{TASK_CONTRACT_VERSION}: {NUMERIC_DOMAIN} "
                    f"Write {root}/reviewer.py with reviewer(x) calling writer(x). "
                    f"Write {root}/test_reviewer.py with one def test_reviewer() asserting "
                    "all edge cases and the invariant. "
                    f"Write {root}/review.json as exact JSON with verdict ('approved' or 'needs_work') "
                    "and findings (list of strings). "
                    f"Run {python} -m pytest {root}/test_reviewer.py -q. "
                    "Do not modify writer.py. Run shell with readonly=False. Deliver all three files and passing tests."
                )

            team = await AgentTeam.create_review_team(
                orchestrator,
                "Gateway review",
                "Deliver assigned files",
                writer=spec("writer"),
                reviewer=reviewer,
                reviser=spec("reviser"),
                shared_context={
                    "thread_id": root.name,
                    "parent_session_id": root.name,
                    "capture_member_usage": True,
                },
                timeout_seconds=600,
            )
            checkpointer = MemoryCheckpointer()  # type: ignore[no-untyped-call]
            cast(UnifiedTeamCoordinator, team._coordinator).with_checkpointer(checkpointer)
            configured_members.extend(team._config.members)
            gate_id = team._config.members[2 if args.mixed else 1].id
            calls = []
            original = SubAgent._execute_with_retry

            async def gate(member: SubAgent) -> Any:
                calls.append(member.config.member_id)
                if member.config.member_id == gate_id and calls.count(gate_id) == 1:
                    raise MemberApprovalPause(
                        ApprovalRequest(
                            id="live-gate",
                            title="Review checkpoint",
                            description="Injected validation gate",
                        )
                    )
                return await original(member)

            with patch.object(SubAgent, "_execute_with_retry", gate):
                paused = await team.run()
                evidence["pipeline"] = {"pause": paused.to_dict(), "calls": calls}
                results.extend(paused.member_results.values())
                assert paused.status == "awaiting_approval", paused.to_dict()
                assert paused.paused_member_id == gate_id
                team._config.shared_context["approval_decision"] = {
                    "member_id": gate_id,
                    "approved": True,
                }
                resumed = await team.run()
            results.clear()
            evidence["pipeline"] = {
                "pause": paused.to_dict(),
                "calls": calls,
                "result": resumed.to_dict(),
                "injected_approval_signal": True,
            }
            results.extend(resumed.member_results.values())

            assert resumed.success, resumed.final_output
            assert calls.count(team._config.members[0].id) == 1, calls
            if args.mixed:
                assert calls.count(team._config.members[1].id) == 1, calls
            assert calls.count(gate_id) == 2, calls
            assert len(resumed.member_results) == 3

            for label, fail in (("selected", False), ("fallback", True)):
                dynamic = await AgentTeam.create(
                    orchestrator,
                    label,
                    "Deliver assigned files",
                    [spec(label + "a"), spec(label + "b")],
                    formation=TeamFormation.SEQUENTIAL,
                )
                configured_members.extend(dynamic._config.members)
                coord = cast(UnifiedTeamCoordinator, dynamic._coordinator)
                for member in coord._adapt_team_members(dynamic._config.members):
                    coord.add_member(member)

                def select(state: dict[str, Any]) -> TeamFormation:
                    if fail:
                        raise ValueError("Injected strategy failure")
                    return TeamFormation.PARALLEL

                coord.with_state_graph_config(StateGraphNodeConfig(formation_strategy=select))
                sink = MemberEventSink()
                token = current_member_sink.set(sink)
                try:
                    state = await coord(
                        {
                            "task": "Deliver assigned files",
                            "parent_session_id": root.name,
                            "capture_member_usage": True,
                        }
                    )
                finally:
                    current_member_sink.reset(token)
                    await sink.close()
                events = [asdict(e) async for e in sink.drain()]
                out = state["team_output"]
                results.extend(out["member_results"].values())
                evidence[label] = {
                    "success": out["success"],
                    "formation": out["formation"],
                    "member_results": {
                        key: value.to_dict() for key, value in out["member_results"].items()
                    },
                    "events": events,
                }
                assert out["success"], out
                assert out["formation"] == ("sequential" if fail else "parallel")
                warnings = [e for e in events if e["kind"] == "team_formation_warning"]
                assert bool(warnings) == fail
                evidence[label].update(formation=out["formation"], warning_events=warnings)

            ids = [member.metadata["session_id"] for member in results]
            assert len(set(ids)) == len(ids) == len(MEMBER_NAMES), ids
            assert {r["session_id"] for r in observed} == set(ids)
            assert all(r["session_id"] == r["run_id"] for r in observed)
            expected = {
                f"{root.name}-{m.id}": {"provider": m.provider, "model": m.model}
                for m in configured_members
            }
            for request in observed:
                assert request["model"] == expected[request["session_id"]]["model"]
                if expected[request["session_id"]]["provider"] == "inferflux":
                    assert not request["reasoning_effort_present"], request
            evidence["member_routes"] = expected
        except asyncio.CancelledError:
            failure(evidence, "execution", error_type="CancelledError")
            raise
        except Exception as exc:
            failure(evidence, "execution", error_type=type(exc).__name__)
        finally:
            try:
                await collect_acceptance(
                    root=root,
                    python=python,
                    mixed=args.mixed,
                    results=results,
                    client=client,
                    gateway_url=settings["url"],
                    admin=admin,
                    evidence=evidence,
                )
            except Exception as exc:
                failure(evidence, "acceptance_collection", error_type=type(exc).__name__)
            finally:
                evidence.update(
                    passed=not evidence["failures"],
                    requests=observed,
                    elapsed_seconds=round(time.monotonic() - started, 2),
                    worktree_commit=subprocess.check_output(
                        ["git", "-C", str(WORKTREE), "rev-parse", "HEAD"],
                        text=True,
                    ).strip(),
                    working_tree_dirty=bool(
                        subprocess.check_output(
                            ["git", "-C", str(WORKTREE), "status", "--porcelain"],
                            text=True,
                        ).strip()
                    ),
                )
                try:
                    (root / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
                    (root / "requests.json").write_text(json.dumps(observed, indent=2) + "\n")
                finally:
                    os.chdir(previous_cwd)
                    for key, value in previous_env.items():
                        if value is None:
                            os.environ.pop(key, None)
                        else:
                            os.environ[key] = value
                    if runner is not None:
                        await runner.cleanup()
            print(
                json.dumps(
                    {
                        "evidence": str(root / "evidence.json"),
                        "members": len(results),
                        "passed": evidence["passed"],
                        "elapsed_seconds": evidence["elapsed_seconds"],
                    }
                )
            )
    return 0 if evidence["passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gateway-state", type=Path, required=True)
    parser.add_argument(
        "--mixed", action="store_true", help="Use InferFlux members plus ZAI reviewer"
    )
    parser.add_argument("--proxy-port", type=int, default=18082)
    raise SystemExit(asyncio.run(validate(parser.parse_args())))

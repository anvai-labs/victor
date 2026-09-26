#!/usr/bin/env python3
"""Run a reproducible six-formation artifact battery on a configured local model.

Writes structured evidence for successes AND failures. No classifier tuning, no
silent skipped formations, no reuse of prior artifacts. Run once per compared model.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

WORKTREE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKTREE))
from aiohttp import ClientSession, web
from victor.framework import Agent
from victor.framework.member_event_sink import MemberEventSink, current_member_sink
from victor.framework.teams import AgentTeam, TeamFormation, TeamMemberSpec

FORMATIONS = ("sequential", "pipeline", "parallel", "hierarchical", "consensus", "reflection")


async def main(args):
    root = Path(args.output_dir).resolve() / f"battery-{uuid.uuid4().hex[:10]}"
    root.mkdir(parents=True)
    python = WORKTREE / ".venv-codesign/bin/python"
    client_config = json.loads(Path(args.gateway_client).read_text())
    observed = []
    records = []
    harness_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    async with ClientSession() as upstream:

        async def forward(request):
            body = await request.read()
            if request.path.endswith("/chat/completions"):
                observed.append(
                    {
                        "session_id": request.headers.get("x-sandhi-session"),
                        "model": json.loads(body).get("model"),
                    }
                )
            headers = {
                k: v
                for k, v in request.headers.items()
                if k.lower() not in {"host", "content-length", "transfer-encoding"}
            }
            async with upstream.request(
                request.method, client_config["url"] + request.path_qs, headers=headers, data=body
            ) as response:
                return web.Response(
                    status=response.status,
                    body=await response.read(),
                    content_type="application/json",
                )

        app = web.Application()
        app.router.add_route("*", "/{tail:.*}", forward)
        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", args.proxy_port).start()
        os.environ["SANDHI_GATEWAY_URL"] = f"http://127.0.0.1:{args.proxy_port}"
        os.environ["SANDHI_GATEWAY_VIRTUAL_KEY_INFERFLUX"] = client_config["virtual_key"]
        os.chdir(root)
        try:
            for name in FORMATIONS:
                case = root / name
                case.mkdir()
                started = time.monotonic()
                offset = len(observed)
                sink = MemberEventSink()
                token = current_member_sink.set(sink)
                record = {"formation": name, "model": args.model, "passed": False}
                try:
                    agent = await Agent.create(
                        provider="inferflux",
                        model=args.model,
                        workspace=str(case),
                        enable_observability=False,
                        session_id=f"{root.name}-{name}",
                    )

                    # Do not request explicit member provider overrides: this battery
                    # measures one model; each member inherits the configured gateway.
                    def spec(member):
                        return TeamMemberSpec(
                            role="executor",
                            name=member,
                            goal=f"Assigned member {member}. Complete in order: "
                            f"1. Write {case}/{member}.py with def {member}(x): return x * 2. "
                            f"2. Write {case}/test_{member}.py with from {member} import {member} "
                            f"and def test_{member}(): assert {member}(4) == 8. "
                            f"3. Run {python} -m pytest {case}/test_{member}.py -q using shell readonly=False. "
                            "Do not repeat successful writes. Both files must exist and pytest must pass. "
                            'Your final output must be exactly {"status":"ready"}.',
                            allowed_tools=["read", "write", "shell"],
                            tool_budget=12,
                            max_iterations=10,
                        )

                    members = [spec("first"), spec("second")]
                    context = {
                        "parent_session_id": f"{root.name}-{name}",
                        "consensus_max_rounds": 3,
                    }
                    if name == "reflection":
                        members[1].goal = (
                            f"Read {case}/first.py and {case}/test_first.py. "
                            f"Run {python} -m pytest {case}/test_first.py -q using shell readonly=False. "
                            f"Write {case}/review.json with exactly verdict and feedback keys, "
                            "verdict=satisfied only if the function doubles input and its test passes, "
                            "otherwise verdict=needs_work. feedback must be a string. "
                            "Return only that JSON object as your final response."
                        )
                        goal = (
                            members[0].goal
                            + f" The generator performs the writes. The critic reads and tests "
                            f"these files and writes {case}/review.json with exactly "
                            '{"verdict":"satisfied","feedback":"tests passed"} only when satisfied. '
                            "The critic returns that same JSON as its final response."
                        )
                        team = await AgentTeam.create_reflection_team(
                            agent.get_orchestrator(),
                            name,
                            goal,
                            generator=members[0],
                            critic=members[1],
                            verdict_format="json",
                            rounds=3,
                            shared_context=context,
                            timeout_seconds=args.timeout,
                        )
                    else:
                        team = await AgentTeam.from_agent(
                            agent,
                            name,
                            "Complete your assigned deliverable and test.",
                            members,
                            formation=TeamFormation(name),
                            shared_context=context,
                            timeout_seconds=args.timeout,
                        )
                    result = await asyncio.wait_for(team.run(), args.timeout)
                    record["result"] = result.to_dict()
                    required = ["first.py", "test_first.py"]
                    required += (
                        ["review.json"] if name == "reflection" else ["second.py", "test_second.py"]
                    )
                    record["missing_artifacts"] = [f for f in required if not (case / f).is_file()]
                    tests = subprocess.run(
                        [str(python), "-m", "pytest", "-q", str(case)],
                        cwd=case,
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    record["pytest"] = {
                        "exit_code": tests.returncode,
                        "output": tests.stdout + tests.stderr,
                    }
                    contract_ok = True
                    if name == "consensus":
                        contract_ok = result.consensus_achieved is True
                    elif name == "reflection":
                        review = json.loads((case / "review.json").read_text())
                        record["review"] = review
                        contract_ok = (
                            set(review) == {"verdict", "feedback"}
                            and review["verdict"] == "satisfied"
                            and isinstance(review["feedback"], str)
                        )
                    sessions = {item["session_id"] for item in observed[offset:]}
                    record["passed"] = (
                        result.success
                        and contract_ok
                        and all(m.success for m in result.member_results.values())
                        and not record["missing_artifacts"]
                        and tests.returncode == 0
                        and None not in sessions
                        and len(sessions) == 2
                        and all(item["model"] == args.model for item in observed[offset:])
                    )
                except Exception as exc:
                    record["error"] = f"{type(exc).__name__}: {exc}"
                finally:
                    current_member_sink.reset(token)
                    await sink.close()
                    record["events"] = [asdict(event) async for event in sink.drain()]
                    record["requests"] = observed[offset:]
                    record["elapsed_seconds"] = round(time.monotonic() - started, 2)
                    records.append(record)
                    (root / "evidence.json").write_text(
                        json.dumps(
                            {
                                "model": args.model,
                                "provider": "inferflux",
                                "cases": records,
                                "worktree_commit": subprocess.check_output(
                                    ["git", "-C", str(WORKTREE), "rev-parse", "HEAD"], text=True
                                ).strip(),
                                "working_tree_dirty": bool(
                                    subprocess.check_output(
                                        ["git", "-C", str(WORKTREE), "status", "--porcelain"],
                                        text=True,
                                    ).strip()
                                ),
                                "classifier_changes": False,
                                "harness_sha256": harness_sha256,
                                "timeout_seconds": args.timeout,
                            },
                            indent=2,
                        )
                        + "\n"
                    )
                    print(
                        json.dumps(
                            {
                                "formation": name,
                                "passed": record["passed"],
                                "elapsed_seconds": record["elapsed_seconds"],
                            }
                        ),
                        flush=True,
                    )
            print(
                json.dumps(
                    {
                        "evidence": str(root / "evidence.json"),
                        "passed": sum(r["passed"] for r in records),
                        "total": len(records),
                    }
                )
            )
        finally:
            await runner.cleanup()
    return int(any(not record["passed"] for record in records))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--gateway-client", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--proxy-port", type=int, default=18083)
    raise SystemExit(asyncio.run(main(parser.parse_args())))

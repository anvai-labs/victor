#!/usr/bin/env python3
"""Live InferFlux formation validation with observed wire-session isolation.

Run with this worktree's .venv-codesign/bin/python. INFERFLUX_API_KEY supplies
credentials. A local forwarding proxy observes actual session headers without
recording credentials or modifying model traffic. Each run gets a fresh directory.
"""

from __future__ import annotations

import argparse
import asyncio
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
from victor.framework.member_event_sink import (
    MEMBER_THROTTLED,
    MemberEventSink,
    current_member_sink,
)
from victor.framework.teams import AgentTeam, TeamFormation, TeamMemberSpec


async def validate(args):
    output_dir = Path(args.output_dir).resolve() / f"capacity-{uuid.uuid4().hex[:10]}"
    output_dir.mkdir(parents=True)
    python = WORKTREE / ".venv-codesign/bin/python"
    observed = []
    async with ClientSession() as upstream:

        async def forward(request):
            body = await request.read()
            if request.path.endswith("/chat/completions"):
                prompt = json.dumps(json.loads(body).get("messages", []))
                assigned = [
                    name
                    for name in ("double", "square", "increment")
                    if f"Assigned module: {output_dir}/{name}.py" in prompt
                ]
                observed.append(
                    {
                        "member_name": assigned[0] if len(assigned) == 1 else None,
                        "session_id": request.headers.get("x-inferflux-session-id"),
                        "path": request.path,
                        "time": time.time(),
                    }
                )
            headers = {
                key: value
                for key, value in request.headers.items()
                if key.lower() not in {"host", "content-length", "transfer-encoding"}
            }
            headers["Accept-Encoding"] = "identity"
            async with upstream.request(
                request.method,
                args.upstream.rstrip("/") + request.path_qs,
                data=body,
                headers=headers,
            ) as response:
                downstream = web.StreamResponse(
                    status=response.status,
                    headers={
                        key: value
                        for key, value in response.headers.items()
                        if key.lower()
                        not in {"content-length", "transfer-encoding", "content-encoding"}
                    },
                )
                await downstream.prepare(request)
                async for chunk in response.content.iter_any():
                    await downstream.write(chunk)
                await downstream.write_eof()
                return downstream

        app = web.Application()
        app.router.add_route("*", "/{tail:.*}", forward)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", args.proxy_port)
        await site.start()
        os.chdir(output_dir)
        started = time.monotonic()
        try:
            agent = await Agent.create(
                provider="inferflux",
                model="qwen3-coder-30b",
                workspace=str(output_dir),
                enable_observability=False,
                session_id=f"live-{uuid.uuid4().hex}",
            )
            orchestrator = agent.get_orchestrator()
            provider = orchestrator.provider
            provider.base_url = f"http://127.0.0.1:{args.proxy_port}/v1"
            # Explicit declaration read from the rig's running serving config.
            # The current ROCm admin API does not expose this field (handoff G18).
            provider.extra_config["max_parallel_sequences"] = args.capacity
            members = []
            for name, expression, example in [
                ("double", "x * 2", "assert double(4) == 8"),
                ("square", "x * x", "assert square(5) == 25"),
                ("increment", "x + 1", "assert increment(9) == 10"),
            ]:
                members.append(
                    TeamMemberSpec(
                        role="executor",
                        name=name,
                        goal=(
                            f"Assigned module: {output_dir}/{name}.py. "
                            "Complete these three steps in order:\n"
                            f"1. Write {output_dir}/{name}.py with function {name}(x) returning {expression}.\n"
                            f"2. Write {output_dir}/test_{name}.py importing {name} and defining "
                            f"a pytest function named test_{name} whose body is: {example}.\n"
                            f"3. Run {python} -m pytest {output_dir}/test_{name}.py -q.\n"
                            "Both files must exist and pytest must collect and pass one test. "
                            "Use write twice, then shell. Return file references and the test result."
                        ),
                        allowed_tools=["read", "write", "edit", "shell", "ls"],
                        tool_budget=12,
                        max_iterations=12,
                    )
                )
            team = await AgentTeam.from_agent(
                agent,
                "Capacity live validation",
                "Each member delivers its assigned module and passing test.",
                members,
                formation=TeamFormation.PARALLEL,
                shared_context={"capacity_aware_parallelism": True},
                timeout_seconds=420,
            )
            sink = MemberEventSink()
            token = current_member_sink.set(sink)
            try:
                result = await team.run()
            finally:
                current_member_sink.reset(token)
            await sink.close()
            events = [event async for event in sink.drain()]
            expected_ids = {member.id for member in team._config.members}
            session_ids = {entry["session_id"] for entry in observed}
            assert observed and None not in session_ids, "Missing InferFlux session header"
            assert len(session_ids) == len(expected_ids), (session_ids, expected_ids)
            member_sessions = {
                name: sorted(
                    {entry["session_id"] for entry in observed if entry["member_name"] == name}
                )
                for name in ("double", "square", "increment")
            }
            assert all(len(ids) == 1 for ids in member_sessions.values()), member_sessions
            assert len({ids[0] for ids in member_sessions.values()}) == 3, member_sessions
            assert result.success, result.final_output
            for name in ("double", "square", "increment"):
                assert (output_dir / f"{name}.py").is_file(), name
                assert (output_dir / f"test_{name}.py").is_file(), name
            tests = subprocess.run(
                [str(python), "-m", "pytest", "-q", str(output_dir)],
                cwd=output_dir,
                text=True,
                capture_output=True,
            )
            (output_dir / "pytest.log").write_text(tests.stdout + tests.stderr)
            assert tests.returncode == 0, tests.stdout + tests.stderr
            assert any(event.kind == MEMBER_THROTTLED for event in events), "Missing throttle event"
            evidence = {
                "worktree_commit": subprocess.check_output(
                    ["git", "-C", str(WORKTREE), "rev-parse", "HEAD"], text=True
                ).strip(),
                "working_tree_dirty": bool(
                    subprocess.check_output(
                        ["git", "-C", str(WORKTREE), "status", "--porcelain"], text=True
                    ).strip()
                ),
                "elapsed_seconds": round(time.monotonic() - started, 2),
                "provider": "inferflux",
                "model": "qwen3-coder-30b",
                "capacity": args.capacity,
                "member_ids": sorted(expected_ids),
                "session_ids": sorted(session_ids),
                "requests": observed,
                "member_sessions": member_sessions,
                "pytest": tests.stdout.strip(),
                "throttled_member_ids": [
                    event.member_id for event in events if event.kind == MEMBER_THROTTLED
                ],
                "deliverables": sorted(path.name for path in output_dir.glob("*.py")),
            }
            (output_dir / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
            print(json.dumps({"evidence": str(output_dir / "evidence.json"), **evidence}, indent=2))
        finally:
            (output_dir / "observed-requests.json").write_text(
                json.dumps(observed, indent=2) + "\n"
            )
            await runner.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--upstream", default="http://127.0.0.1:8080")
    parser.add_argument("--proxy-port", type=int, default=18081)
    parser.add_argument("--capacity", type=int, default=2)
    args = parser.parse_args()
    asyncio.run(validate(args))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""ZAI-only live review, pipeline pause/resume, selection, and usage reconciliation.

This complements, and does not claim to replace, the R9700 or edge-model matrix.
The approval signal is deliberately injected before the reviewer's first LLM call.
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
from victor.teams.unified_coordinator import StateGraphNodeConfig


async def validate(args):
    root = Path(args.output_dir).resolve() / f"gateway-{uuid.uuid4().hex[:10]}"
    root.mkdir(parents=True)
    settings = json.loads((args.gateway_state / "client.json").read_text())
    admin = (args.gateway_state / "admin-token").read_text().strip()
    python = WORKTREE / ".venv-codesign/bin/python"
    observed = []
    evidence = {"provider": "zai", "model": "glm-5.3", "scope": "ZAI-only; no InferFlux claim"}
    started = time.monotonic()
    async with ClientSession() as client:

        async def forward(request):
            body = await request.read()
            if request.path.endswith("/chat/completions"):
                observed.append(
                    {
                        "session_id": request.headers.get("x-sandhi-session"),
                        "run_id": request.headers.get("x-sandhi-run-id"),
                        "model": json.loads(body).get("model"),
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
        os.chdir(root)
        agent = await Agent.create(
            provider="zai",
            model="glm-5.3",
            workspace=str(root),
            enable_observability=False,
            session_id=root.name,
        )
        orchestrator = agent.get_orchestrator()
        assert orchestrator.provider.extra_config["gateway"]["url"].endswith(str(args.proxy_port))

        def spec(name):
            return TeamMemberSpec(
                role="executor",
                name=name,
                provider="zai",
                model="glm-5.3",
                goal=f"Assigned member: {name}. Complete these steps in order: "
                f"1. Write {root}/{name}.py with function {name}(x) returning x * 2. "
                f"2. Write {root}/test_{name}.py importing that function and defining "
                f"def test_{name}(): assert {name}(4) == 8. "
                f"3. Run {python} -m pytest {root}/test_{name}.py -q. "
                "Both files must exist and one test must pass. Use write twice then shell. "
                "Return file references and test result.",
                allowed_tools=["read", "write", "shell"],
                tool_budget=12,
                max_iterations=12,
            )

        results = []
        try:
            team = await AgentTeam.create_review_team(
                orchestrator,
                "Gateway review",
                "Deliver assigned files",
                writer=spec("writer"),
                reviewer=spec("reviewer"),
                reviser=spec("reviser"),
                shared_context={
                    "thread_id": root.name,
                    "parent_session_id": root.name,
                    "capture_member_usage": True,
                },
                timeout_seconds=600,
            )
            checkpointer = MemoryCheckpointer()
            team._coordinator.with_checkpointer(checkpointer)
            gate_id = team._config.members[1].id
            calls = []
            original = SubAgent._execute_with_retry

            async def gate(member):
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
                assert paused.status == "awaiting_approval", paused.to_dict()
                assert paused.paused_member_id == gate_id
                team._config.shared_context["approval_decision"] = {
                    "member_id": gate_id,
                    "approved": True,
                }
                resumed = await team.run()
            assert resumed.success, resumed.final_output
            assert calls.count(team._config.members[0].id) == 1, calls
            assert calls.count(gate_id) == 2, calls
            assert len(resumed.member_results) == 3
            evidence["pipeline"] = {
                "pause": paused.to_dict(),
                "calls": calls,
                "result": resumed.to_dict(),
                "injected_approval_signal": True,
            }
            results.extend(resumed.member_results.values())

            for label, fail in (("selected", False), ("fallback", True)):
                dynamic = await AgentTeam.create(
                    orchestrator,
                    label,
                    "Deliver assigned files",
                    [spec(label + "a"), spec(label + "b")],
                    formation=TeamFormation.SEQUENTIAL,
                )
                coord = dynamic._coordinator
                for member in coord._adapt_team_members(dynamic._config.members):
                    coord.add_member(member)

                def select(state):
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
                assert out["success"], out
                assert out["formation"] == ("sequential" if fail else "parallel")
                warnings = [e for e in events if e["kind"] == "team_formation_warning"]
                assert bool(warnings) == fail
                evidence[label] = {"formation": out["formation"], "warning_events": warnings}
                results.extend(out["member_results"].values())

            ids = [member.metadata["session_id"] for member in results]
            assert len(set(ids)) == len(ids), ids
            assert {r["session_id"] for r in observed} == set(ids)
            assert all(r["session_id"] == r["run_id"] for r in observed)
            assert all(r["model"] == "glm-5.3" for r in observed)
            reconciled = []
            for member in results:
                session = member.metadata["session_id"]
                async with client.get(
                    settings["url"] + "/admin/usage/run/" + session,
                    headers={"Authorization": f"Bearer {admin}"},
                ) as response:
                    assert response.status == 200
                    tree = (await response.json())["run"]
                usage = member.metadata["usage"]
                total = tree["total"]
                gateway_input = (
                    total["tokens_in"] + total["cache_read_tokens"] + total["cache_creation_tokens"]
                )
                assert usage["input_tokens"] == gateway_input, (usage, total)
                assert usage["output_tokens"] == total["tokens_out"], (usage, total)
                assert usage["total_tokens"] == gateway_input + total["tokens_out"], (usage, total)
                reconciled.append(
                    {
                        "member_id": member.member_id,
                        "session_id": session,
                        "victor": usage,
                        "sandhi": total,
                    }
                )
            tests = subprocess.run(
                [str(python), "-m", "pytest", "-q", str(root)],
                cwd=root,
                text=True,
                capture_output=True,
            )
            assert tests.returncode == 0, tests.stdout + tests.stderr
            assert len(list(root.glob("test_*.py"))) == 7
            evidence.update(
                usage_reconciliation=reconciled,
                pytest=tests.stdout,
                deliverables=sorted(p.name for p in root.glob("*.py")),
                requests=observed,
                elapsed_seconds=round(time.monotonic() - started, 2),
                worktree_commit=subprocess.check_output(
                    ["git", "-C", str(WORKTREE), "rev-parse", "HEAD"], text=True
                ).strip(),
                working_tree_dirty=True,
            )
            (root / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        "evidence": str(root / "evidence.json"),
                        "members": len(ids),
                        "elapsed_seconds": evidence["elapsed_seconds"],
                    }
                )
            )
        finally:
            (root / "requests.json").write_text(json.dumps(observed, indent=2) + "\n")
            await runner.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gateway-state", type=Path, required=True)
    parser.add_argument("--proxy-port", type=int, default=18082)
    asyncio.run(validate(parser.parse_args()))

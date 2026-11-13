"""Public Victor Agent -> real services -> Sandhi -> synthetic Messages acceptance.

Set SANDHI_PEER_TEST_BINARY to an explicitly built candidate. No vendor/network
credentials are read. The process gets a closed synthetic environment and vault.
"""

import asyncio
import json
import os
from pathlib import Path
import socket
import sqlite3
import subprocess
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from victor.providers.openai_provider import OpenAIProvider
from victor.providers.sandhi_transport import resolve_transport_class

pytestmark = pytest.mark.integration
PROTOCOLS = ["chat_completions", "anthropic_messages", "responses", "gemini_generate_content"]


def _post(root, path, body):
    request = urllib.request.Request(
        root + path,
        data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer synthetic-admin", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.load(response)


@pytest.fixture(params=["tokens", "none"])
def agent_gateway(tmp_path, request):
    binary = os.environ.get("SANDHI_PEER_TEST_BINARY")
    if not binary:
        pytest.skip("set SANDHI_PEER_TEST_BINARY to the paired Sandhi candidate")
    records = []
    control = {"missing_usage": False, "paths": [], "auth_mode": request.param}
    for turn in (1, 2):
        for speed in ("slow", "fast"):
            path = tmp_path / f"round-{turn}-{speed}.txt"
            path.write_text(f"round-{turn}-{speed}-content\n")
            control["paths"].append(str(path))
    usage = {
        "input_tokens": 2,
        "output_tokens": 4,
        "cache_read_input_tokens": 6,
        "cache_creation_input_tokens": 2,
    }
    response = {
        "id": "msg_fixture",
        "type": "message",
        "role": "assistant",
        "model": "fixture",
        "content": [{"type": "text", "text": "reply"}],
        "stop_reason": "end_turn",
        "usage": usage,
    }

    class Origin(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            records.append({"body": body, "headers": dict(self.headers), "path": self.path})
            turn = len(records)
            value = {**response, "content": [], "stop_reason": "end_turn"}
            if turn <= 2:
                value["stop_reason"] = "tool_use"
                value["content"] = [
                    {
                        "type": "tool_use",
                        "id": f"round-{turn}-{speed}",
                        "name": "read",
                        "input": {"path": control["paths"][(turn - 1) * 2 + index]},
                    }
                    for index, speed in enumerate(("slow", "fast"))
                ]
            else:
                value["content"] = [
                    {
                        "type": "text",
                        "text": "READY: all four requested reads completed successfully.",
                    }
                ]
            if body.get("stream"):
                start = {
                    **value,
                    "content": [],
                    "stop_reason": None,
                    "usage": {**usage, "output_tokens": 0},
                }
                frames = [{"type": "message_start", "message": start}]
                for index, block in enumerate(value["content"]):
                    if block["type"] == "tool_use":
                        frames.extend(
                            [
                                {
                                    "type": "content_block_start",
                                    "index": index,
                                    "content_block": {**block, "input": {}},
                                },
                                {
                                    "type": "content_block_delta",
                                    "index": index,
                                    "delta": {
                                        "type": "input_json_delta",
                                        "partial_json": json.dumps(block["input"]),
                                    },
                                },
                            ]
                        )
                    else:
                        frames.extend(
                            [
                                {
                                    "type": "content_block_start",
                                    "index": index,
                                    "content_block": {"type": "text", "text": ""},
                                },
                                {
                                    "type": "content_block_delta",
                                    "index": index,
                                    "delta": {"type": "text_delta", "text": block["text"]},
                                },
                            ]
                        )
                    frames.append({"type": "content_block_stop", "index": index})
                frames.extend(
                    [
                        {
                            "type": "message_delta",
                            "delta": {"stop_reason": value["stop_reason"]},
                            "usage": {"output_tokens": 4},
                        },
                        {"type": "message_stop"},
                    ]
                )
                payload = "".join(
                    "data: " + json.dumps(frame) + "\n\n" for frame in frames
                ).encode()
                content_type = "text/event-stream"
            else:
                payload = json.dumps(value).encode()
                content_type = "application/json"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    origin = ThreadingHTTPServer(("127.0.0.1", 0), Origin)
    thread = threading.Thread(target=origin.serve_forever, daemon=True)
    thread.start()
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    root = f"http://127.0.0.1:{port}"
    vault = tmp_path / "vault"
    vault.mkdir(mode=0o700)
    credential = vault / "anthropic__default"
    credential.write_text("synthetic-origin")
    credential.chmod(0o600)
    database = tmp_path / "gateway.db"
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "streaming_deadlines": {
                    "ceiling_ms": 10000,
                    "default": {"setup_ms": 2000, "idle_ms": 2000, "body_ms": 5000},
                }
            }
        )
    )
    env = {
        "PATH": "/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "SANDHI_BIND": f"127.0.0.1:{port}",
        "SANDHI_LOG": "warn",
        "SANDHI_STORE": str(database),
        "SANDHI_ADMIN_TOKEN": "synthetic-admin",
        "SANDHI_VAULT_BACKEND": "credstore",
        "SANDHI_CREDENTIALS_DIR": str(vault),
        "SANDHI_CONFIG": str(config),
    }
    if request.param == "tokens":
        env["SANDHI_AUTH_MODE"] = "tokens"
    else:
        # Exercise the server's actual zero-auth default, while keeping routing
        # server-owned and the admin API separately authenticated.
        env["SANDHI_STARTER_UPSTREAM"] = "route/agent-loop"
    with (tmp_path / "stderr.log").open("wb") as log:
        process = subprocess.Popen(
            [str(Path(binary).resolve())], env=env, stdout=subprocess.DEVNULL, stderr=log
        )
        try:
            import time

            for _ in range(200):
                if process.poll() is not None:
                    raise AssertionError((tmp_path / "stderr.log").read_text())
                try:
                    urllib.request.urlopen(root + "/healthz", timeout=0.1).close()
                    break
                except (OSError, urllib.error.URLError):
                    time.sleep(0.02)
            else:
                raise AssertionError("synthetic gateway did not start")
            source_base = f"http://127.0.0.1:{origin.server_port}"
            _post(root, "/admin/keys/reference", {"provider": "anthropic", "base_url": source_base})
            policy = {
                "default_intent": {"mode": "auto_stable_prefix", "lifetime": "5m"},
                "models": {
                    "fixture": {
                        "profile": "anthropic_breakpoints",
                        "lifetimes": ["5m"],
                        "max_breakpoints": 4,
                    }
                },
            }
            definition = {
                "id": "peer-fixture",
                "provider": "custom-messages",
                "credential_ref": "anthropic:default",
                "endpoint": {"endpoint_family": "anthropic_messages"},
                "base_url": source_base,
                "models": [
                    {
                        "id": "fixture",
                        "max_input_tokens": 100000,
                        "max_output_tokens": 128,
                        "capabilities": {"streaming": True, "prompt_cache_usage": True},
                    }
                ],
                "cache_policy": policy,
            }
            _post(root, "/admin/backends", {"definition": definition})
            _post(
                root,
                "/admin/routes",
                {
                    "definition": {
                        "id": "agent-loop",
                        "models": [
                            {
                                "id": "gpt-4o-mini",
                                "targets": [{"backend": "peer-fixture", "model": "fixture"}],
                            }
                        ],
                    }
                },
            )
            gateway = {"url": root}
            if request.param == "tokens":
                gateway["virtual_key"] = _post(
                    root,
                    "/admin/keys/share",
                    {"upstream": "route/agent-loop", "models": ["gpt-4o-mini"]},
                )["virtual_key"]
            yield gateway, records, control, database
        finally:
            process.terminate()
            try:
                process.wait(6)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            origin.shutdown()
            origin.server_close()
            thread.join()


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("streaming", [False, True])
async def test_public_agent_real_services_parallel_tool_loop_conserves_source_receipts(
    agent_gateway, tmp_path, protocol, monkeypatch, streaming
):
    monkeypatch.chdir(tmp_path)
    from victor.config.settings import Settings, set_project_root

    set_project_root(tmp_path)
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.framework.agent import Agent
    from victor.core.container import reset_container
    from victor.core.database import reset_all_databases

    reset_container()
    reset_all_databases()
    config, records, control, database = agent_gateway
    settings = Settings(
        working_directory=str(tmp_path),
        analytics_enabled=False,
        use_semantic_tool_selection=False,
        preload_embeddings=False,
        enable_planning=False,
        conversation_memory_enabled=False,
        conversation_embeddings_enabled=False,
        tool_cache_enabled=False,
        parallel_tool_execution=True,
        chat_max_iterations=5,
    )
    if control["auth_mode"] == "none":
        from victor.config.settings import ProviderConfig, ProviderGatewayConfig

        for name in ("SANDHI_GATEWAY_VIRTUAL_KEY", "SANDHI_GATEWAY_VIRTUAL_KEY_OPENAI"):
            monkeypatch.delenv(name, raising=False)
        provider_config = ProviderConfig(gateway=ProviderGatewayConfig(**config, protocol=protocol))
        # Inject only settings loading; public Agent creation, configuration
        # resolution, provider factory and real dispatch remain exercised.
        monkeypatch.setattr("victor.config.settings.load_settings", lambda **_: settings)
        monkeypatch.setattr(settings, "load_provider_config", lambda _: provider_config)
        agent = await Agent.create(
            provider="openai",
            model="gpt-4o-mini",
            max_tokens=64,
            tools=["read"],
            enable_observability=False,
            workspace=str(tmp_path),
        )
        orch = agent._orchestrator
    else:
        provider = resolve_transport_class("openai", OpenAIProvider, {})(
            gateway={**config, "protocol": protocol}
        )
        orch = AgentOrchestrator(
            settings=settings,
            provider=provider,
            model="gpt-4o-mini",
            max_tokens=64,
            provider_name="openai",
        )
        agent = Agent.from_orchestrator(orch)
    orch.set_enabled_tools({"read"})
    assert type(orch._chat_service).__name__ == "ChatService"
    assert type(orch._tool_service).__name__ == "ToolService"
    assert orch._tool_pipeline.config.parallel_tool_execution
    read = orch.tools.get("read")
    original = read.execute
    started = []
    completed = []
    signals = {turn: asyncio.Event() for turn in (1, 2)}

    async def monitored(*args, **kwargs):
        path = kwargs["path"]
        assert path in control["paths"], "agent escaped the synthetic fixture scope"
        assert path not in started, "tool side effect replayed"
        started.append(path)
        turn = 1 if "round-1" in path else 2
        if "slow" in path:
            await asyncio.wait_for(signals[turn].wait(), timeout=3)
        result = await original(*args, **kwargs)
        completed.append(path)
        if "fast" in path:
            signals[turn].set()
        return result

    monkeypatch.setattr(read, "execute", monitored)
    try:
        prompt = "Read the four requested files in two rounds, using each pair of read tool calls, then report READY."
        overrides = {"max_iterations": 5, "tool_budget": 8}
        if streaming:
            from victor.framework.events import EventType

            async def consume():
                return [
                    event
                    async for event in agent.stream(prompt, runtime_context_overrides=overrides)
                ]

            events = await asyncio.wait_for(consume(), timeout=30)
            assert "READY" in "".join(
                event.content for event in events if event.type == EventType.CONTENT
            )
            assert events[-1].type == EventType.STREAM_END and events[-1].success
            assert not any(
                event.type in (EventType.ERROR, EventType.TOOL_ERROR) for event in events
            )
        else:
            result = await asyncio.wait_for(
                agent.run(prompt, runtime_context_overrides=overrides), timeout=30
            )
            assert "READY" in result.content
            assert result.success
        report = orch.get_last_task_report()
        assert report["request_count"] == 3
        assert report["api_total_tokens"] == 42
        assert report["cache_read_tokens"] == 18
        (tmp_path / "agent-result-evidence.json").write_text(
            json.dumps(
                {
                    "protocol": protocol,
                    "auth_mode": control["auth_mode"],
                    "streaming": streaming,
                    "real_services": [
                        type(orch._chat_service).__name__,
                        type(orch._tool_service).__name__,
                    ],
                    "request_count": report["request_count"],
                    "api_total_tokens": report["api_total_tokens"],
                    "cache_read_tokens": report["cache_read_tokens"],
                    "tool_start_paths": started,
                    "tool_completion_paths": completed,
                },
                indent=2,
            )
        )
    finally:
        await agent.close()
        read.execute = original
    assert len(records) == 3
    assert started == control["paths"]
    assert completed == [control["paths"][i] for i in (1, 0, 3, 2)]
    # Check identity joins even if the framework restores input order after
    # completion-order execution. The neutral SDK loop separately tests wire
    # tool results supplied in reverse order.
    results = [
        block
        for message in records[-1]["body"]["messages"]
        if message["role"] == "user"
        for block in message["content"]
        if isinstance(block, dict) and block.get("type") == "tool_result"
    ]
    assert len(results) == 4
    assert {block["tool_use_id"] for block in results} == {
        "round-1-slow",
        "round-1-fast",
        "round-2-slow",
        "round-2-fast",
    }
    for block in results:
        assert block["tool_use_id"] + "-content" in json.dumps(block["content"])
    with sqlite3.connect(database) as connection:
        if control["auth_mode"] == "none":
            assert connection.execute(
                "SELECT DISTINCT virtual_key_id,subject_id,group_id FROM usage_events"
            ).fetchall() == [("starter_shared_v1", "builtin_starter", "builtin_starter")]
            assert {
                row[0]
                for row in connection.execute("SELECT DISTINCT scope FROM budget_reservation")
            } == {"starter:shared:v1", "upstream:anthropic"}
        assert connection.execute("SELECT count(*) FROM budget_physical_attempt").fetchone() == (3,)
        assert (
            connection.execute("SELECT charged_tokens FROM budget_settlement_outbox").fetchall()
            == [(14,)] * 3
        )
        assert (
            connection.execute("SELECT settled,actual FROM budget_reservation").fetchall()
            == [(1, 14)] * 6
        )
        source = [
            json.loads(row[0])
            for row in connection.execute("SELECT usage_json FROM budget_terminal_observation")
        ]
        assert all(
            (u["tokens_in"], u["cache_read_tokens"], u["cache_creation_tokens"], u["tokens_out"])
            == (2, 6, 2, 4)
            for u in source
        )

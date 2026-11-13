"""Synthetic real Victor -> Sandhi gateway -> Messages source acceptance.

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

from victor.providers.anthropic_provider import AnthropicProvider
from victor.providers.google_provider import GoogleProvider
from victor.providers.openai_provider import OpenAIProvider
from victor.providers.zai_provider import ZAIProvider
from victor.providers.sandhi_transport import resolve_transport_class
from victor.providers.base import Message, ProviderError

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


@pytest.fixture
def gateway(tmp_path):
    binary = os.environ.get("SANDHI_PEER_TEST_BINARY")
    if not binary:
        pytest.skip("set SANDHI_PEER_TEST_BINARY to the paired Sandhi candidate")
    records = []
    control = {"missing_usage": False}
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
            if body.get("stream"):
                start = {**response, "content": [], "stop_reason": None}
                if control["missing_usage"]:
                    start.pop("usage")
                frames = [
                    {"type": "message_start", "message": start},
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    },
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": "reply"},
                    },
                    {"type": "content_block_stop", "index": 0},
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "end_turn"},
                        **({} if control["missing_usage"] else {"usage": {"output_tokens": 4}}),
                    },
                    {"type": "message_stop"},
                ]
                payload = "".join(
                    "data: " + json.dumps(frame) + "\n\n" for frame in frames
                ).encode()
                content_type = "text/event-stream"
            else:
                value = dict(response)
                if control["missing_usage"]:
                    value.pop("usage")
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
        "SANDHI_AUTH_MODE": "tokens",
        "SANDHI_BIND": f"127.0.0.1:{port}",
        "SANDHI_LOG": "warn",
        "SANDHI_STORE": str(database),
        "SANDHI_ADMIN_TOKEN": "synthetic-admin",
        "SANDHI_VAULT_BACKEND": "credstore",
        "SANDHI_CREDENTIALS_DIR": str(vault),
        "SANDHI_CONFIG": str(config),
        "SANDHI_BUFFERED_ACCOUNTING": "tracked",
    }
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
                        "max_input_tokens": 128,
                        "max_output_tokens": 32,
                        "capabilities": {"streaming": True, "prompt_cache_usage": True},
                    }
                ],
                "cache_policy": policy,
            }
            _post(root, "/admin/backends", {"definition": definition})
            key = _post(
                root,
                "/admin/keys/share",
                {"upstream": "backend/peer-fixture", "models": ["fixture"]},
            )["virtual_key"]
            yield {"url": root, "virtual_key": key}, records, control, database
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


async def test_real_two_hop_client_dialects_defer_cache_once_and_preserve_source_receipts(gateway):
    config, records, _, database = gateway
    messages = [
        Message(role="system", content="Stable policy."),
        Message(role="user", content="hello"),
    ]
    count = 0
    for label, native in [
        ("anthropic", AnthropicProvider),
        ("google", GoogleProvider),
        ("openai", OpenAIProvider),
        ("zai", ZAIProvider),
    ]:
        for protocol in PROTOCOLS:
            provider = resolve_transport_class(label, native, {})(
                gateway={**config, "protocol": protocol}
            )
            try:
                for streaming in (False, True):
                    before = len(records)
                    if streaming:
                        chunks = [
                            chunk
                            async for chunk in provider.stream(
                                messages,
                                model="fixture",
                                max_tokens=16,
                                cache_intent={"mode": "auto_stable_prefix", "lifetime": "5m"},
                            )
                        ]
                        assert chunks[-1].is_final
                        terminal = chunks[-1]
                        assert "".join(chunk.content for chunk in chunks) == "reply"
                    else:
                        result = await provider.chat(
                            messages,
                            model="fixture",
                            max_tokens=16,
                            cache_intent={"mode": "auto_stable_prefix", "lifetime": "5m"},
                        )
                        assert result.content == "reply"
                        terminal = result
                    units = terminal.metadata["sandhi_usage"]["typed_usage"]
                    assert (
                        terminal.metadata["sandhi_usage"]["usage_basis"]
                        == "gateway_client_projection"
                    )
                    assert units["tokens_in"] == (4 if protocol == "gemini_generate_content" else 2)
                    assert units["cache_creation_tokens"] == (
                        0 if protocol == "gemini_generate_content" else 2
                    )
                    assert units["cache_read_tokens"] == 6
                    assert units["tokens_out"] == 4
                    assert len(records) == before + 1, (label, protocol, streaming)
                    body = records[-1]["body"]
                    assert body["system"][0]["cache_control"] == {"type": "ephemeral", "ttl": "5m"}
                    assert json.dumps(body).count("cache_control") == 1
                    assert "sandhi_cache" not in body
                    assert "compiled_plan" not in json.dumps(body)
                    count += 1
                before = len(records)
                with pytest.raises(ProviderError) as refused:
                    await provider.chat(
                        messages,
                        model="fixture",
                        max_tokens=16,
                        cache_intent={"mode": "auto_stable_prefix", "lifetime": "1h"},
                    )
                assert not refused.value.recoverable
                assert len(records) == before
            finally:
                await provider.close()
    assert count == 32
    for _ in range(100):
        with sqlite3.connect(database) as connection:
            settled = connection.execute(
                "SELECT count(*) FROM budget_settlement_outbox"
            ).fetchone()[0]
        if settled == count:
            break
        await asyncio.sleep(0.02)
    assert settled == count
    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            "SELECT c.execution_id,c.context_json,o.usage_json,a.evidence_json,s.charged_tokens FROM budget_execution_context c JOIN budget_terminal_observation o USING(execution_id) JOIN budget_physical_attempt a USING(execution_id) JOIN budget_execution_intent i USING(execution_id) JOIN budget_settlement_outbox s ON s.reservation_id=i.reservation_id"
        ).fetchall()
    assert len(rows) == count
    assert len({row[0] for row in rows}) == count
    for _, context_json, usage_json, attempt_json, charge in rows:
        context, source, attempt = map(json.loads, (context_json, usage_json, attempt_json))
        assert context["provider"] == "custom-messages"
        assert context["backend"]["id"] == "peer-fixture"
        assert source["tokens_in"] == 2
        assert source["cache_read_tokens"] == 6
        assert source["cache_creation_tokens"] == 2
        assert source["tokens_out"] == 4
        assert attempt["attempt_ordinal"] == 1
        assert charge == 14


async def test_real_two_hop_unknown_usage_is_terminal_through_default_resilience(gateway):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock
    from victor.providers.resilience import ResilientProvider

    config, records, control, database = gateway
    control["missing_usage"] = True
    for protocol in PROTOCOLS:
        provider = resolve_transport_class("zai", ZAIProvider, {})(
            gateway={**config, "protocol": protocol}
        )
        fallback = SimpleNamespace(
            name="fallback", chat=AsyncMock(return_value="replayed"), stream=Mock()
        )
        resilient = ResilientProvider(provider, fallback_providers=[fallback])
        try:
            for streaming in (False, True):
                before = len(records)
                chunks = []
                with pytest.raises(ProviderError) as terminal:
                    if streaming:
                        async for chunk in resilient.stream(
                            [
                                Message(role="system", content="Stable policy."),
                                Message(role="user", content="hello"),
                            ],
                            model="fixture",
                            max_tokens=16,
                        ):
                            chunks.append(chunk)
                    else:
                        await resilient.chat(
                            [
                                Message(role="system", content="Stable policy."),
                                Message(role="user", content="hello"),
                            ],
                            model="fixture",
                            max_tokens=16,
                        )
                assert not terminal.value.recoverable
                assert len(records) == before + 1, (protocol, streaming, str(terminal.value))
                assert not any(chunk.is_final for chunk in chunks)
                fallback.chat.assert_not_awaited()
                fallback.stream.assert_not_called()
        finally:
            await provider.close()
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT count(*) FROM budget_execution_intent").fetchone()[0] == 8
        assert (
            connection.execute("SELECT count(*) FROM budget_settlement_outbox").fetchone()[0] == 0
        )

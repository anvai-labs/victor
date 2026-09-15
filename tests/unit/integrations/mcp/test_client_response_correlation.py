"""Real stdio regression tests: transport evidence, not agent narration."""

import asyncio
import json
import subprocess
import sys

import pytest

from victor.integrations.mcp.client import MCPClient
from victor.integrations.mcp.protocol import MCPMessageType
from victor.config.timeouts import McpTimeouts


def child(source):
    client = MCPClient(health_check_interval=0, auto_reconnect=False)
    client.process = subprocess.Popen(
        [sys.executable, "-u", "-c", source],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return client


@pytest.mark.asyncio
async def test_notifications_and_wrong_ids_never_become_tool_results():
    client = child("""import sys,json
for line in sys.stdin:
 r=json.loads(line)
 print(json.dumps({'jsonrpc':'2.0','method':'notifications/progress','params':{}}),flush=True)
 print(json.dumps({'jsonrpc':'2.0','id':'wrong','result':{'value':'wrong'}}),flush=True)
 print(json.dumps({'jsonrpc':'2.0','id':r['id'],'result':r['params']}),flush=True)
""")
    try:
        result = await client._send_request(MCPMessageType.PING, {"value": "correct"})
        assert result["result"] == {"value": "correct"}
    finally:
        await client.cleanup()


@pytest.mark.asyncio
async def test_concurrent_requests_keep_their_own_results():
    client = child("""import sys,json
for line in sys.stdin:
 r=json.loads(line)
 print(json.dumps({'jsonrpc':'2.0','id':r['id'],'result':r['params']}),flush=True)
""")
    try:
        results = await asyncio.gather(
            *(client._send_request(MCPMessageType.PING, {"n": n}) for n in range(16))
        )
        assert [r["result"]["n"] for r in results] == list(range(16))
    finally:
        await client.cleanup()


@pytest.mark.asyncio
async def test_timeout_retires_transport_before_another_request(monkeypatch):
    monkeypatch.setattr(McpTimeouts, "RESPONSE", 0.05)
    client = child("""import sys,json,time
for line in sys.stdin:
 r=json.loads(line); time.sleep(.2)
 print(json.dumps({'jsonrpc':'2.0','id':r['id'],'result':r['params']}),flush=True)
""")
    process = client.process
    try:
        assert await client._send_request(MCPMessageType.PING, {"n": 1}) is None
        assert client.process is None
        assert process.poll() is not None
        assert await client._send_request(MCPMessageType.PING, {"n": 2}) is None
    finally:
        await client.cleanup()


@pytest.mark.asyncio
async def test_lost_tool_response_reports_uncertainty(monkeypatch):
    from unittest.mock import AsyncMock

    client = MCPClient(health_check_interval=0, auto_reconnect=False)
    client.initialized = True
    monkeypatch.setattr(client, "_send_request", AsyncMock(return_value=None))
    result = await client.call_tool("browser_act", operationId="lost-1")
    assert not result.success
    assert "reconcile" in result.error.lower()
    assert "may have completed" in result.error.lower()


@pytest.mark.asyncio
async def test_eof_retires_transport():
    client = child("import sys; sys.stdin.readline()")
    process = client.process
    try:
        assert await client._send_request(MCPMessageType.PING, {}) is None
        assert client.process is None
        assert process.poll() is not None
    finally:
        await client.cleanup()


@pytest.mark.asyncio
async def test_failed_exchange_retires_sandbox_owner():
    from unittest.mock import AsyncMock, MagicMock

    client = child("import sys; sys.stdin.readline(); print('invalid json', flush=True)")
    wrapper = MagicMock()
    wrapper.terminate_all = AsyncMock()
    client._sandboxed_process = wrapper
    try:
        assert await client._send_request(MCPMessageType.PING, {}) is None
        wrapper.terminate_all.assert_awaited_once()
        assert client._sandboxed_process is None
        assert client.process is None
    finally:
        await client.cleanup()

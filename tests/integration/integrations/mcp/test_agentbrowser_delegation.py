"""Opt-in acceptance against a built AgentBrowser checkout, no model provider.

AGENTBROWSER_ROOT=/path/to/built/agentbrowser pytest <this file>
Transport uses the real MCP binary; browser effects are separately qualified by
AgentBrowser's real Chromium/panel fixture. This test needs only FakeEngine.
"""

import asyncio
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import urllib.request

import pytest
from victor.integrations.mcp.client import MCPClient


@pytest.mark.asyncio
async def test_bound_agentbrowser_over_real_stdio():
    root_value = os.environ.get("AGENTBROWSER_ROOT")
    if not root_value:
        pytest.skip("Set AGENTBROWSER_ROOT to a built AgentBrowser checkout")
    root = Path(root_value).resolve()
    node = shutil.which("node")
    assert node
    operator = secrets.token_urlsafe(32)
    key_hash = hashlib.sha256(operator.encode()).hexdigest()
    source = f"""
const {{ buildServer }} = await import({json.dumps((root / 'packages/api/dist/server.js').as_uri())});
const {{ FakeEngine }} = await import({json.dumps((root / 'packages/testkit/dist/index.js').as_uri())});
const app = await buildServer({{ engine: new FakeEngine(), apiKeys: new Map([[{json.dumps(key_hash)}, 'owner']]) }});
await app.listen({{port:0,host:'127.0.0.1'}});
console.log(JSON.stringify({{port:app.server.address().port}}));
process.on('SIGTERM', async () => {{ await app.close(); process.exit(0); }});
"""
    server = subprocess.Popen(
        [node, "--input-type=module", "-e", source],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    client = MCPClient(health_check_interval=0, auto_reconnect=False)
    try:
        ready = await asyncio.wait_for(asyncio.to_thread(server.stdout.readline), 10)
        base = f"http://127.0.0.1:{json.loads(ready)['port']}"

        def request(path, body=None, method="POST"):
            req = urllib.request.Request(
                base + path,
                data=json.dumps(body or {}).encode() if method == "POST" else None,
                method=method,
                headers={
                    "Authorization": f"Bearer {operator}",
                    "Content-Type": "application/json",
                    "X-AgentBrowser-Operation-Id": secrets.token_hex(16),
                },
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                return json.load(response)

        session_id = request("/v1/sessions", {"controlMode": "delegated"})["sessionId"]
        path = f"/v1/sessions/{session_id}"
        page_id = request(path + "/pages")["pageId"]
        review = request(path + "/control/prepare-resume")
        token = request(path + "/control/delegate", {"epoch": review["epoch"]})["token"]
        client.process = subprocess.Popen(
            [node, str(root / "packages/mcp-server/dist/bin.js")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            env={
                **os.environ,
                "AGENTBROWSER_BASE_URL": base,
                "AGENTBROWSER_SESSION_ID": session_id,
                "AGENTBROWSER_API_KEY": token,
            },
        )
        assert await client.initialize()
        tools = await client.refresh_tools()
        names = {tool.name for tool in tools}
        assert "browser_session" in names
        assert not names.intersection({"browser_create", "browser_close", "browser_cookies"})
        attached = await client.call_tool("browser_session")
        assert attached.success
        assert json.loads(attached.result)["pages"][0]["pageId"] == page_id
        action = await client.call_tool(
            "browser_act", pageId=page_id, action="press", key="Tab", operationId="victor-step-1"
        )
        assert action.success, action.error
        record = await client.call_tool("browser_operation", operationId="victor-step-1")
        assert record.success
        assert json.loads(record.result)["status"] == "completed"
        request(path + "/control/takeover")
        assert not (await client.call_tool("browser_session")).success
    finally:
        await client.cleanup()
        server.terminate()
        try:
            await asyncio.to_thread(server.wait, timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
            await asyncio.to_thread(server.wait)
        for stream in (server.stdout, server.stderr):
            if stream:
                stream.close()

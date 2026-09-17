"""Real stdio regression tests: transport evidence, not agent narration."""

import asyncio
import json
import os
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
        start_new_session=os.name == "posix",
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


@pytest.mark.asyncio
async def test_retirement_detaches_before_await_and_refuses_replacement(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    client = child("import sys; sys.stdin.readline()")
    entered, release = asyncio.Event(), asyncio.Event()
    owner = MagicMock()

    async def terminate():
        entered.set()
        await release.wait()

    owner.terminate_all = AsyncMock(side_effect=terminate)
    client._sandboxed_process = owner
    launch = AsyncMock()
    monkeypatch.setattr(client, "_start_process", launch)
    request = asyncio.create_task(client._send_request(MCPMessageType.PING, {}))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        assert client.process is None
        assert not await client.connect(["replacement"])
        launch.assert_not_called()
    finally:
        release.set()
        await request
        await client.cleanup()


@pytest.mark.asyncio
async def test_cancellation_during_retirement_cannot_revive_old_transport(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    monkeypatch.setattr(McpTimeouts, "RESPONSE", 0.05)
    client = child("import sys,time; sys.stdin.readline(); time.sleep(2)")
    process = client.process
    entered, release = asyncio.Event(), asyncio.Event()
    owner = MagicMock()

    async def terminate():
        entered.set()
        await release.wait()

    owner.terminate_all = AsyncMock(side_effect=terminate)
    client._sandboxed_process = owner
    request = asyncio.create_task(client._send_request(MCPMessageType.PING, {}))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        assert client.process is None
        assert await client._send_request(MCPMessageType.PING, {}) is None
    finally:
        release.set()
        await client.cleanup()
        if process.poll() is None:
            process.kill()
        await asyncio.to_thread(process.wait, timeout=3)


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="requires POSIX process groups")
@pytest.mark.parametrize("direction", ["read", "write"])
async def test_inherited_pipes_are_killed_without_blocking_event_loop(
    tmp_path, monkeypatch, direction
):
    import time

    ready, release = tmp_path / "ready", tmp_path / "release"
    pid_file = tmp_path / "descendant.pid"
    descendant = f"""import os,pathlib,signal,time
pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid()))
signal.signal(signal.SIGTERM, signal.SIG_IGN)
pathlib.Path({str(ready)!r}).write_text('ready')
end=time.monotonic()+3
while not pathlib.Path({str(release)!r}).exists() and time.monotonic()<end: time.sleep(.01)
"""
    source = f"""import subprocess,sys,time
subprocess.Popen([sys.executable,'-c',{descendant!r}])
time.sleep(5)
"""
    client = child(source)
    monkeypatch.setattr(McpTimeouts, "RESPONSE", 0.1)
    heartbeat = asyncio.Event()
    try:
        for _ in range(200):
            if ready.exists():
                break
            await asyncio.sleep(0.01)
        assert ready.exists()
        asyncio.get_running_loop().call_later(0.2, heartbeat.set)
        start = time.monotonic()
        assert (
            await client._send_request(
                MCPMessageType.PING, {"data": "x" * 1000000} if direction == "write" else {}
            )
            is None
        )
        await asyncio.wait_for(heartbeat.wait(), 1)
        assert time.monotonic() - start < 1.5
        assert client.process is None
        descendant_pid = int(pid_file.read_text())
        for _ in range(200):
            if not client.get_status()["transport_cleanup_pending"]:
                break
            await asyncio.sleep(0.01)
        assert not client.get_status()["transport_cleanup_pending"]
        try:
            os.kill(descendant_pid, 0)
        except ProcessLookupError:
            pass
        else:
            # Linux can retain a reparented zombie briefly; it owns no pipe or
            # executable resources and is no longer a live descendant.
            proc_stat = f"/proc/{descendant_pid}/stat"
            assert os.path.exists(proc_stat)
            with open(proc_stat) as stat_file:
                assert stat_file.read().split()[2] == "Z"
    finally:
        release.write_text("release")
        await client.cleanup()
        # Let the owned worker observe EOF/broken pipe and finish closing streams.
        for _ in range(200):
            if not client.get_status().get("transport_cleanup_pending", False):
                break
            await asyncio.sleep(0.01)


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="requires POSIX sessions")
async def test_escaped_descendant_is_quarantined_until_its_pipe_closes(tmp_path, monkeypatch):
    from unittest.mock import AsyncMock

    ready, release = tmp_path / "escaped.ready", tmp_path / "escaped.release"
    descendant = f"""import os,pathlib,time
os.setsid()
pathlib.Path({str(ready)!r}).write_text('ready')
end=time.monotonic()+5
while not pathlib.Path({str(release)!r}).exists() and time.monotonic()<end: time.sleep(.01)
"""
    source = f"""import subprocess,sys,time
subprocess.Popen([sys.executable,'-c',{descendant!r}])
time.sleep(5)
"""
    client = child(source)
    monkeypatch.setattr(McpTimeouts, "RESPONSE", 0.1)
    heartbeat = asyncio.Event()
    try:
        for _ in range(200):
            if ready.exists():
                break
            await asyncio.sleep(0.01)
        assert ready.exists()
        asyncio.get_running_loop().call_later(0.2, heartbeat.set)
        assert await client._send_request(MCPMessageType.PING, {}) is None
        await asyncio.wait_for(heartbeat.wait(), 1)
        assert client.get_status()["transport_cleanup_pending"]

        launch = AsyncMock()
        monkeypatch.setattr(client, "_start_process", launch)
        assert not await client.connect(["replacement"])
        launch.assert_not_called()

        release.write_text("release")
        for _ in range(300):
            if not client.get_status()["transport_cleanup_pending"]:
                break
            await asyncio.sleep(0.01)
        assert not client.get_status()["transport_cleanup_pending"]
    finally:
        release.write_text("release")
        await client.cleanup()


@pytest.mark.asyncio
async def test_retirement_at_worker_handoff_closes_every_stream():
    import io
    import threading
    from unittest.mock import MagicMock
    from victor.integrations.mcp.stdio_transport import StdioTransport

    process = MagicMock()
    process.stdin = io.StringIO()
    process.stdout = io.StringIO('{"id":"1","result":{}}\n')
    process.stderr = io.StringIO()
    process.poll.return_value = 0
    transport = StdioTransport(process)
    reached, release, done = threading.Event(), threading.Event(), threading.Event()
    done.set()

    class GatedCompletion:
        def clear(self):
            done.clear()

        def is_set(self):
            return done.is_set()

        def set(self):
            reached.set()
            if not release.wait(3):
                raise RuntimeError("worker handoff gate timed out")
            done.set()

    transport.worker_done = GatedCompletion()
    result = transport.exchange("{}", "1")
    try:
        assert await asyncio.to_thread(reached.wait, 2)
        transport.retire()
        await asyncio.to_thread(transport.wait_and_close)
        assert not transport.settled()
    finally:
        release.set()
    await asyncio.wait_for(result, 2)
    assert all(stream.closed for stream in (process.stdin, process.stdout, process.stderr))
    assert transport.settled()


@pytest.mark.asyncio
async def test_owner_cleanup_revokes_group_before_transport_close(monkeypatch):
    import io
    from unittest.mock import AsyncMock, MagicMock
    from victor.integrations.mcp.sandbox import OwnedProcessGroup
    from victor.integrations.mcp.stdio_transport import StdioTransport

    process = MagicMock()
    process.pid = 12345
    process.returncode = None
    process.stdin = io.StringIO()
    process.stdout = io.StringIO()
    process.stderr = io.StringIO()
    process.wait.return_value = 0
    group = OwnedProcessGroup(leader_pid=12345, pgid=12345)
    owner = MagicMock()

    async def terminate_all():
        process.returncode = 0
        group.invalidate()

    owner.terminate_all = AsyncMock(side_effect=terminate_all)
    transport = StdioTransport(process, owner, process_group=group)
    kill_group = MagicMock()
    monkeypatch.setattr(os, "killpg", kill_group)

    await transport.cleanup()

    kill_group.assert_not_called()
    assert transport.cleanup_done.is_set()


def test_final_group_signal_error_still_marks_cleanup_done(monkeypatch):
    import io
    from unittest.mock import MagicMock
    from victor.integrations.mcp.sandbox import OwnedProcessGroup
    from victor.integrations.mcp.stdio_transport import StdioTransport

    process = MagicMock()
    process.pid = 12345
    process.returncode = None
    process.stdin = io.StringIO()
    process.stdout = io.StringIO()
    process.stderr = io.StringIO()
    process.wait.return_value = 0
    group = OwnedProcessGroup(leader_pid=12345, pgid=12345)
    transport = StdioTransport(process, process_group=group)
    monkeypatch.setattr(
        "victor.integrations.mcp.stdio_transport.wait_for_exit_without_reaping",
        lambda *_: True,
    )
    monkeypatch.setattr(
        "victor.integrations.mcp.stdio_transport.signal_process_tree",
        MagicMock(side_effect=PermissionError("denied")),
    )

    transport.wait_and_close()

    assert transport.cleanup_done.is_set()
    assert not group.active


def test_uncertain_group_exit_force_kills_and_reaps_direct_child(monkeypatch):
    import io
    from unittest.mock import MagicMock
    from victor.integrations.mcp.sandbox import OwnedProcessGroup
    from victor.integrations.mcp.stdio_transport import StdioTransport

    process = MagicMock()
    process.pid = 12345
    process.returncode = None
    process.stdin = io.StringIO()
    process.stdout = io.StringIO()
    process.stderr = io.StringIO()
    process.wait.side_effect = [
        subprocess.TimeoutExpired("wait", McpTimeouts.TERMINATE),
        0,
    ]
    group = OwnedProcessGroup(leader_pid=12345, pgid=12345)
    transport = StdioTransport(process, process_group=group)
    monkeypatch.setattr(
        "victor.integrations.mcp.stdio_transport.wait_for_exit_without_reaping",
        lambda *_: None,
    )

    transport.wait_and_close()

    process.kill.assert_called_once_with()
    assert process.wait.call_count == 2
    assert transport.cleanup_done.is_set()
    assert not group.active

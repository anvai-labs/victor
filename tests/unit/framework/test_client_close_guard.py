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

"""VictorClient close-guard: close() must not tear down while a stream is active."""

from __future__ import annotations

import asyncio
import time

import pytest

from victor.framework import client as client_mod
from victor.framework.client import VictorClient
from victor.framework.session_config import SessionConfig


def _client() -> VictorClient:
    c = VictorClient(SessionConfig.from_cli_flags())
    # No real agent/container so close() exercises only the drain guard.
    c._agent = None
    c._container = None
    return c


async def test_stream_raises_when_closing():
    c = _client()
    c._closing = True
    gen = c.stream("hello")
    with pytest.raises(RuntimeError, match="closing"):
        await gen.__anext__()


async def test_close_sets_closing_flag():
    c = _client()
    await c.close()
    assert c._closing is True


async def test_close_drains_active_stream_before_returning():
    c = _client()
    c._active_streams = 1

    async def _release():
        await asyncio.sleep(0.2)
        c._active_streams = 0

    releaser = asyncio.create_task(_release())
    started = time.monotonic()
    await c.close()
    elapsed = time.monotonic() - started

    assert c._closing is True
    assert elapsed >= 0.19, "close() should wait for the active stream to drain"
    assert c._active_streams == 0
    await releaser


async def test_close_force_proceeds_after_drain_timeout(monkeypatch, caplog):
    monkeypatch.setattr(client_mod, "_CLOSE_DRAIN_TIMEOUT_SECONDS", 0.3)
    c = _client()
    c._active_streams = 1  # never released -> must force after the (short) timeout

    started = time.monotonic()
    with caplog.at_level("WARNING"):
        await c.close()
    elapsed = time.monotonic() - started

    assert 0.3 <= elapsed < 2.0, "close() should force-proceed after the drain timeout"
    assert any("still active after drain timeout" in r.message for r in caplog.records)


async def test_close_is_fast_when_no_active_streams():
    c = _client()
    started = time.monotonic()
    await c.close()
    assert time.monotonic() - started < 0.1

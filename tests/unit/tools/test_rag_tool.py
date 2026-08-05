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

"""Tests for the unified ``rag`` dispatcher (Phase 3 — BaseTool-class delegation).

The rag dispatcher resolves a victor-rag ``BaseTool`` class, instantiates it,
and awaits ``execute(**kwargs)``. Tests mock the resolver to verify routing and
the absent-package graceful path (there is no shell fallback for a knowledge
base).
"""

from unittest.mock import patch

import pytest


def _fake_tool_class(capture: dict, output: str = "ok"):
    """Build a fake BaseTool-like class that records execute kwargs."""

    class _FakeTool:
        async def execute(self, **kwargs):
            capture.update(kwargs)
            return {"success": True, "output": output}

    return _FakeTool


@pytest.mark.asyncio
async def test_rag_search_delegates_and_maps_kwargs():
    from victor.tools.unified.rag_tool import rag_tool

    captured: dict = {}
    with patch(
        "victor.tools.unified.rag_tool.resolve_vertical_callable",
        return_value=(_fake_tool_class(captured, "results"), "victor_rag.tools.search"),
    ):
        result = await rag_tool('rag search "auth flow" --k 5')
    assert captured == {"query": "auth flow", "k": 5}
    assert "results" in result


@pytest.mark.asyncio
async def test_rag_delete_maps_doc_id():
    from victor.tools.unified.rag_tool import rag_tool

    captured: dict = {}
    with patch(
        "victor.tools.unified.rag_tool.resolve_vertical_callable",
        return_value=(
            _fake_tool_class(captured, "deleted"),
            "victor_rag.tools.management",
        ),
    ):
        result = await rag_tool("rag delete doc-42")
    assert captured == {"doc_id": "doc-42"}
    assert "deleted" in result


@pytest.mark.asyncio
async def test_rag_ingest_maps_path_and_type():
    from victor.tools.unified.rag_tool import rag_tool

    captured: dict = {}
    with patch(
        "victor.tools.unified.rag_tool.resolve_vertical_callable",
        return_value=(
            _fake_tool_class(captured, "ingested"),
            "victor_rag.tools.ingest",
        ),
    ):
        await rag_tool("rag ingest --path docs.md --type markdown")
    assert captured["path"] == "docs.md"
    assert captured["doc_type"] == "markdown"


@pytest.mark.asyncio
async def test_rag_absent_returns_graceful_message():
    from victor.tools.unified.rag_tool import rag_tool

    with patch(
        "victor.tools.unified.rag_tool.resolve_vertical_callable",
        return_value=(None, None),
    ):
        result = await rag_tool("rag list")
    assert "### ❌ ERROR" in result
    assert "victor-rag" in result


@pytest.mark.asyncio
async def test_rag_no_subcommand_errors():
    from victor.tools.unified.rag_tool import rag_tool

    result = await rag_tool("rag")
    assert "### ❌ ERROR" in result


def test_rag_registered_name():
    from victor.tools.unified.rag_tool import rag_tool

    assert rag_tool.Tool.name == "rag"

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

"""Structured image artifacts across the MCP seam (VAS-14 artifact channel).

Pins the full chain an agentbrowser screenshot rides: MCP image content blocks
survive ``call_tool`` as bounded ``images``, the adapter attaches them to
``ToolResult.metadata``, ``map_event`` carries them on the TOOL_END action, and
the shared presenter never lets them be invisible on text surfaces.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock
from typing import Any

from victor.integrations.mcp.client import (
    _MAX_IMAGE_B64_CHARS,
    _MAX_RESULT_IMAGES,
    MCPClient,
)
from victor.integrations.mcp.protocol import MCPMessageType
from victor.ui.chat_app.event_mapping import RenderKind, map_event
from victor.ui.rendering.markdown_presenters import image_artifact_placeholder


def _image_block(b64: str = "aGVsbG8=", mime: str = "image/png") -> dict:
    return {"type": "image", "data": b64, "mimeType": mime}


async def test_image_blocks_become_structured_artifacts(monkeypatch):
    client = MCPClient("stdio", command=["fake"])
    client.initialized = True
    response = {
        "result": {
            "content": [
                {"type": "text", "text": "screenshot captured"},
                _image_block(),
            ],
            "isError": False,
        }
    }

    async def fake_send(message_type, params):
        assert message_type is MCPMessageType.CALL_TOOL
        return response

    monkeypatch.setattr(client, "_send_request", fake_send)
    result = await client.call_tool("browser_screenshot")

    assert result.success is True
    assert "[image: image/png]" in str(result.result)
    assert "screenshot captured" in str(result.result)
    assert result.images == [{"mime_type": "image/png", "b64": "aGVsbG8="}]


async def test_image_bounds_drop_overflow_with_a_visible_note(monkeypatch):
    client = MCPClient("stdio", command=["fake"])
    client.initialized = True
    oversized = "x" * (_MAX_IMAGE_B64_CHARS + 1)
    blocks: list[dict[str, Any]] = [{"type": "text", "text": "many"}]
    blocks += [_image_block(b64=f"{i:03d}") for i in range(_MAX_RESULT_IMAGES + 2)]
    blocks.append(_image_block(b64=oversized))
    response = {"result": {"content": blocks}}

    async def fake_send(message_type, params):
        return response

    monkeypatch.setattr(client, "_send_request", fake_send)
    result = await client.call_tool("browser_screenshot")

    assert len(result.images) == _MAX_RESULT_IMAGES
    assert "image(s) dropped" in str(result.result)
    assert str(_MAX_RESULT_IMAGES) in str(result.result)


def test_map_event_carries_images_on_tool_end():
    event = SimpleNamespace(
        event_type="tool_result",
        tool_name="mcp_browser_screenshot",
        content="screenshot captured",
        result=None,
        metadata={
            "tool_result": {
                "success": True,
                "images": [{"mime_type": "image/png", "b64": "aGVsbG8="}],
                "arguments": {},
            }
        },
    )
    action = map_event(event)
    assert action.kind is RenderKind.TOOL_END
    assert action.images == [{"mime_type": "image/png", "b64": "aGVsbG8="}]


def test_placeholder_keeps_artifacts_visible_on_text_surfaces():
    line = image_artifact_placeholder([{"mime_type": "image/png", "b64": "x" * 2048}])
    assert line.startswith("🖼️")
    assert "image/png" in line
    assert image_artifact_placeholder(None) == ""
    assert image_artifact_placeholder([]) == ""


def _handler():
    from victor.agent.streaming.handler import StreamingChatHandler

    return StreamingChatHandler(settings=MagicMock(), message_adder=MagicMock())


class TestPipelineSeam:
    """The channel must survive the REAL executor and chunk generator.

    A prior review caught the images field being severed at the executor
    (metadata dropped in the wrap) while fabricated-shape tests stayed green —
    these drive the actual producers instead of hand-built dicts.
    """

    async def test_executor_preserves_images_from_tool_result_metadata(self):
        from unittest.mock import AsyncMock, MagicMock

        from victor.agent.tool_executor import ToolExecutor
        from victor.tools.base import BaseTool, ToolRegistry, ToolResult

        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "browser_screenshot"
        mock_tool.is_idempotent = True
        mock_tool.execute = AsyncMock(
            return_value=ToolResult(
                success=True,
                output="screenshot captured",
                metadata={"images": [{"mime_type": "image/png", "b64": "aGVsbG8="}]},
            )
        )
        registry = ToolRegistry()
        registry.register(mock_tool)
        executor = ToolExecutor(tool_registry=registry)
        exec_result = await executor.execute("browser_screenshot", {})

        assert exec_result.success is True
        assert exec_result.images == [{"mime_type": "image/png", "b64": "aGVsbG8="}]
        # The text result stays text — images never leak into the LLM-visible output.
        assert "aGVsbG8" not in str(exec_result.result)

    async def test_chunk_generator_carries_images_from_flat_result_dict(self):
        generator = _handler()
        chunk = generator.generate_tool_result_chunk(
            "browser_screenshot",
            {"target": "e1_1"},
            0.12,
            True,
            None,
            result="screenshot captured",
            original_result="screenshot captured",
            images=[{"mime_type": "image/png", "b64": "aGVsbG8="}],
        )
        payload = chunk.metadata["tool_result"]
        assert payload["images"] == [{"mime_type": "image/png", "b64": "aGVsbG8="}]
        assert payload["result"] == "screenshot captured"

    async def test_flat_result_dict_to_chunk_roundtrip(self):
        """The results-dict shape tool_service emits reaches the chunk metadata."""
        generator = _handler()
        result_dict = {
            "name": "browser_screenshot",
            "success": True,
            "elapsed": 0.1,
            "args": {},
            "result": "captured",
            "full_result": "captured",
            "was_pruned": False,
            "tool_call_id": "c1",
            "images": [{"mime_type": "image/png", "b64": "aGVsbG8="}],
        }
        chunk = generator.generate_tool_result_chunk(
            result_dict["name"],
            result_dict["args"],
            result_dict["elapsed"],
            result_dict["success"],
            None,
            follow_up_suggestions=result_dict.get("follow_up_suggestions"),
            was_pruned=result_dict["was_pruned"],
            result=result_dict["result"],
            original_result=result_dict["full_result"],
            tool_call_id=result_dict["tool_call_id"],
            images=result_dict.get("images"),
        )
        assert chunk.metadata["tool_result"]["images"] == [
            {"mime_type": "image/png", "b64": "aGVsbG8="}
        ]


async def test_dataless_image_block_gets_its_own_note_not_the_bound_note(monkeypatch):
    client = MCPClient("stdio", command=["fake"])
    client.initialized = True
    response = {
        "result": {
            "content": [
                {"type": "image", "data": "", "mimeType": "image/png"},
                {"type": "text", "text": "ok"},
            ]
        }
    }

    async def fake_send(message_type, params):
        return response

    monkeypatch.setattr(client, "_send_request", fake_send)
    result = await client.call_tool("browser_screenshot")

    assert result.images == []
    assert "[image block without data]" in str(result.result)
    assert "dropped" not in str(result.result)

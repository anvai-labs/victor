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

"""Tests for concurrent tool rendering and tool_call_id plumbing.

Covers the regression where the live renderer tracked a single in-flight tool
in scalar state, so parallel batches (e.g. shell + web) clobbered each
other's name/start-time and interleaved output.
"""

from rich.console import Console

from victor.ui.rendering.live_renderer import LiveDisplayRenderer


def _renderer() -> tuple[LiveDisplayRenderer, Console]:
    console = Console(record=True, width=120, force_terminal=False)
    renderer = LiveDisplayRenderer(console)
    renderer.start()
    return renderer, console


class TestConcurrentTracking:
    def test_two_tools_tracked_independently(self):
        renderer, _ = _renderer()
        renderer.on_tool_start("shell", {"cmd": "grep x", "readonly": True}, tool_call_id="a1")
        renderer.on_tool_start("web", {"cmd": "search kimi"}, tool_call_id="b2")
        assert set(renderer._active_tools) == {"a1", "b2"}
        renderer.cleanup()

    def test_result_retires_matching_entry_by_id(self):
        renderer, _ = _renderer()
        renderer.on_tool_start("shell", {"cmd": "grep x"}, tool_call_id="a1")
        renderer.on_tool_start("web", {"cmd": "search"}, tool_call_id="b2")
        renderer.on_tool_result(
            name="web", success=True, elapsed=1.0, arguments={}, result="ok", tool_call_id="b2"
        )
        assert set(renderer._active_tools) == {"a1"}
        renderer.cleanup()

    def test_result_falls_back_to_name_match_without_id(self):
        renderer, _ = _renderer()
        renderer.on_tool_start("shell", {"cmd": "grep x"})
        renderer.on_tool_result(name="shell", success=True, elapsed=0.5, arguments={}, result="ok")
        assert renderer._active_tools == {}
        renderer.cleanup()

    def test_same_tool_twice_retires_one_entry_per_result(self):
        renderer, _ = _renderer()
        renderer.on_tool_start("code", {"cmd": 'grep "a"'}, tool_call_id="c1")
        renderer.on_tool_start("code", {"cmd": 'grep "b"'}, tool_call_id="c2")
        renderer.on_tool_result(
            name="code", success=True, elapsed=0.1, arguments={}, result="x", tool_call_id="c2"
        )
        assert set(renderer._active_tools) == {"c1"}
        renderer.on_tool_result(
            name="code", success=True, elapsed=0.2, arguments={}, result="y", tool_call_id="c1"
        )
        assert renderer._active_tools == {}
        renderer.cleanup()

    def test_both_status_lines_render(self):
        renderer, console = _renderer()
        renderer.on_tool_start("shell", {"cmd": "grep x"}, tool_call_id="a1")
        renderer.on_tool_start("web", {"cmd": "search"}, tool_call_id="b2")
        renderer.on_tool_result(
            name="shell", success=True, elapsed=0.05, arguments={}, result="m", tool_call_id="a1"
        )
        renderer.on_tool_result(
            name="web", success=True, elapsed=1.2, arguments={}, result="n", tool_call_id="b2"
        )
        renderer.cleanup()
        out = console.export_text()
        assert "✓ shell" in out
        assert "✓ web" in out

    def test_cleanup_clears_active_tools(self):
        renderer, _ = _renderer()
        renderer.on_tool_start("shell", {"cmd": "x"}, tool_call_id="a1")
        renderer.cleanup()
        assert renderer._active_tools == {}


class TestChunkPlumbing:
    def test_tool_start_chunk_carries_id(self):
        from victor.agent.streaming.handler import StreamingChatHandler

        handler = StreamingChatHandler.__new__(StreamingChatHandler)
        chunk = handler.generate_tool_start_chunk(
            "shell", {"cmd": "ls"}, "running", tool_call_id="call_123"
        )
        assert chunk.metadata["tool_start"]["tool_call_id"] == "call_123"

    def test_tool_result_chunk_carries_id(self):
        from victor.agent.streaming.handler import StreamingChatHandler

        handler = StreamingChatHandler.__new__(StreamingChatHandler)
        chunk = handler.generate_tool_result_chunk(
            "shell", {"cmd": "ls"}, 0.1, True, result="out", tool_call_id="call_123"
        )
        assert chunk.metadata["tool_result"]["tool_call_id"] == "call_123"

    def test_dispatcher_forwards_ids(self):
        from unittest.mock import MagicMock

        from victor.ui.rendering.event_dispatcher import EventDispatcher

        renderer = MagicMock()
        dispatcher = EventDispatcher(renderer, MagicMock(), MagicMock(), MagicMock())
        dispatcher._handle_tool_start_metadata(
            {
                "tool_start": {
                    "name": "shell",
                    "arguments": {"cmd": "ls"},
                    "tool_call_id": "call_9",
                    "batch_index": 2,
                    "batch_total": 3,
                    "execution_mode": "parallel_batch",
                }
            },
            None,
        )
        kwargs = renderer.on_tool_start.call_args.kwargs
        assert kwargs["tool_call_id"] == "call_9"
        assert kwargs["batch_index"] == 2
        assert kwargs["batch_total"] == 3
        assert kwargs["execution_mode"] == "parallel_batch"

        dispatcher._handle_tool_result_metadata(
            {"tool_result": {"name": "shell", "success": True, "tool_call_id": "call_9"}},
            None,
        )
        assert renderer.on_tool_result.call_args.kwargs["tool_call_id"] == "call_9"


class TestDeadRendererRemoved:
    def test_unwired_renderer_cluster_deleted(self):
        import importlib

        import pytest as _pytest

        for module in (
            "victor.ui.rendering.tool_display",
            "victor.ui.rendering.live_manager",
            "victor.ui.rendering.thinking_display",
        ):
            with _pytest.raises(ModuleNotFoundError):
                importlib.import_module(module)

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

"""Lock-in tests for the compact tool status line and real result previews.

Replaces the old `• [DONE] name CATEGORY BADGE ⚡ Hint • completed • 47ms`
row (which said "execute" three ways) and the per-result `▸ Category` group
headers (which grouped nothing, since results arrive in execution order).
"""

from rich.console import Console

from victor.ui.rendering.live_renderer import LiveDisplayRenderer


def _render(**result_kwargs) -> str:
    console = Console(record=True, width=120, force_terminal=False)
    renderer = LiveDisplayRenderer(console)
    renderer.start()
    defaults = {
        "name": "code",
        "success": True,
        "elapsed": 0.047,
        "arguments": {"cmd": 'grep "x" .'},
        "result": "a.py:1: x",
        "original_result": "a.py:1: x",
    }
    defaults.update(result_kwargs)
    renderer.on_tool_result(**defaults)
    renderer.cleanup()
    return console.export_text()


class TestCompactStatusLine:
    def test_success_uses_check_not_done_token(self):
        out = _render()
        assert "✓ code" in out
        assert "[DONE]" not in out
        assert "completed" not in out

    def test_failure_uses_cross_and_error_line(self):
        out = _render(success=False, result=None, original_result=None, error="boom")
        assert "✗ code" in out
        assert "Error: boom" in out

    def test_readonly_invocation_has_no_access_badge(self):
        out = _render()
        line = next(ln for ln in out.splitlines() if "✓ code" in ln)
        for badge in ("MIXED", "READONLY", "WRITE", "EXECUTE", "read+write"):
            assert badge not in line

    def test_no_group_headers_rendered(self):
        out = _render()
        assert "▸" not in out

    def test_real_preview_rendered_not_boilerplate(self):
        out = _render(result="match one\nmatch two", original_result="match one\nmatch two")
        assert "match one" in out
        assert "Tool completed successfully" not in out

    def test_duration_present(self):
        out = _render(elapsed=0.047)
        assert "47ms" in out

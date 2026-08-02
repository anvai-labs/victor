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

"""Tests for incremental (head/tail) streaming markdown render.

Verifies the McGugan-style optimization in LiveDisplayRenderer: the stable HEAD
(complete markdown blocks) is rendered once and cached; only the active TAIL
(in-progress last block) is re-rendered each tick. Gates: byte-identical final
output, no mid-code-block split, head cached (perf), env-flag fallback, and
exception fallback.
"""

from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from victor.ui.rendering import LiveDisplayRenderer
from victor.ui.rendering.markdown import find_safe_split
from victor.ui.theme import victor_theme


class TestFindSafeSplit:
    def test_splits_after_completed_paragraph(self):
        c = "First para.\n\nSecond para streaming"
        assert c[find_safe_split(c) :] == "Second para streaming"

    def test_single_block_has_empty_head(self):
        assert find_safe_split("just one paragraph") == 0

    def test_does_not_split_inside_open_code_fence(self):
        c = "Intro.\n\n```python\nx = 1\nstill cod"
        # Tail must include the whole in-flight code block (no mid-fence split).
        assert c[find_safe_split(c) :].startswith("```python")

    def test_splits_after_closed_fence(self):
        c = "Intro.\n\n```python\nx = 1\n```\n\nNext para"
        assert c[find_safe_split(c) :] == "Next para"

    def test_head_and_tail_partition_content(self):
        c = "# Title\n\nP1.\n\nP2.\n\nP3 streaming"
        s = find_safe_split(c)
        assert c[:s] + c[s:] == c  # clean partition
        # Head holds the completed blocks; tail is the in-progress P3.
        assert c[s:] == "P3 streaming"

    def test_multiple_blank_lines_split_at_last(self):
        c = "A.\n\n\n\nB."
        # The latest safe boundary wins; tail is the in-progress block.
        assert c[find_safe_split(c) :] == "B."


def _renderer_with_mock_live(mock_live_class, mock_render):
    mock_live = MagicMock()
    mock_live_class.return_value = mock_live
    mock_render.side_effect = lambda content: f"R({len(content)})"
    console = Console(theme=victor_theme)
    renderer = LiveDisplayRenderer(console)
    renderer.start()
    return renderer, mock_live


class TestIncrementalRender:
    @patch("victor.ui.rendering.live_renderer.render_markdown_with_hooks")
    @patch("victor.ui.rendering.live_renderer.Live")
    def test_head_is_rendered_once_during_long_tail(self, mock_live_class, mock_render):
        """Perf gate: a completed HEAD is cached, not re-rendered every tick."""
        renderer, _ = _renderer_with_mock_live(mock_live_class, mock_render)

        # Two completed blocks establish a stable HEAD.
        renderer.on_content("AAA.\n\n")
        renderer.on_content("BBB.\n\n")
        head = "AAA.\n\nBBB.\n\n"
        head_render_count_before = sum(1 for c in mock_render.call_args_list if c.args[0] == head)

        # Stream the third (in-progress) block over many ticks.
        for _ in range(20):
            renderer.on_content("CCC")

        head_render_count_after = sum(1 for c in mock_render.call_args_list if c.args[0] == head)
        # HEAD was rendered when block 2 completed, then NOT again during the 20 ticks.
        assert head_render_count_after == head_render_count_before == 1

    @patch("victor.ui.rendering.live_renderer.render_markdown_with_hooks")
    @patch("victor.ui.rendering.live_renderer.Live")
    def test_final_head_equals_full_visible_when_all_blocks_complete(
        self, mock_live_class, mock_render
    ):
        """Byte-identical gate: once the last block completes, HEAD == full content
        and the tail is empty, so the final render equals a full re-render."""
        renderer, _ = _renderer_with_mock_live(mock_live_class, mock_render)
        doc = "P1.\n\nP2.\n\nP3.\n\n"  # all blocks complete (trailing blank)
        for tok in [doc[:7], doc[7:14], doc[14:]]:
            renderer.on_content(tok)

        visible = renderer._content_buffer
        assert renderer._rendered_head_source == visible
        # The last render call's tail argument is empty (head absorbed it all).
        assert mock_render.call_args_list[-1].args[0] == ""

    @patch("victor.ui.rendering.live_renderer.render_markdown_with_hooks")
    @patch("victor.ui.rendering.live_renderer.Live")
    def test_no_mid_code_block_split_in_renderer(self, mock_live_class, mock_render):
        """While a code block is open, the HEAD must not absorb a partial fence."""
        renderer, _ = _renderer_with_mock_live(mock_live_class, mock_render)
        renderer.on_content("Intro.\n\n")
        renderer.on_content("```python\n")
        for _ in range(10):
            renderer.on_content("x = 1\n")

        # The HEAD is the completed "Intro." block only; the open fence stays in tail.
        assert renderer._rendered_head_source == "Intro.\n\n"

    @patch("victor.ui.rendering.live_renderer.render_markdown_with_hooks")
    @patch("victor.ui.rendering.live_renderer.Live")
    def test_flag_off_falls_back_to_full_render(self, mock_live_class, mock_render):
        """VICTOR_INCREMENTAL_RENDER=0 disables the optimization (full re-render)."""
        renderer, _ = _renderer_with_mock_live(mock_live_class, mock_render)
        with patch.object(LiveDisplayRenderer, "_incremental_render_enabled", return_value=False):
            renderer.on_content("A.\n\n")
            renderer.on_content("B.")
        visible = renderer._content_buffer
        # Last render call received the FULL visible slice (not a head/tail split).
        assert mock_render.call_args_list[-1].args[0] == visible

    @patch(
        "victor.ui.rendering.live_renderer.find_safe_split",
        side_effect=RuntimeError("split boom"),
    )
    @patch("victor.ui.rendering.live_renderer.render_markdown_with_hooks")
    @patch("victor.ui.rendering.live_renderer.Live")
    def test_exception_falls_back_to_full_render(self, mock_live_class, mock_render, _mock_split):
        """If the incremental path raises, the stream must not break — fall back
        to a full re-render of the visible slice."""
        mock_live = MagicMock()
        mock_live_class.return_value = mock_live
        mock_render.side_effect = lambda content: f"R({len(content)})"
        console = Console(theme=victor_theme)
        renderer = LiveDisplayRenderer(console)
        renderer.start()

        # find_safe_split raises -> incremental path excepts -> full-render fallback.
        renderer.on_content("A.\n\nB.")  # must not raise
        visible = renderer._content_buffer
        assert renderer._rendered_head is None  # cache reset by the fallback
        assert mock_render.call_args_list[-1].args[0] == visible  # full visible rendered

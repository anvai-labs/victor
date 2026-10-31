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

"""Unit tests for shared rendering helpers in ``victor.ui.rendering.utils``."""

from __future__ import annotations

import pytest
from rich.console import Console

from victor.ui.output_formatter import OutputFormatter
from victor.ui.rendering.live_renderer import LiveDisplayRenderer
from victor.ui.rendering.utils import generate_text_preview


@pytest.mark.parametrize("text", ["", None])
def test_preview_of_empty_text_is_empty(text: object) -> None:
    assert generate_text_preview(text) == ""  # type: ignore[arg-type]


def test_preview_keeps_leading_lines_and_marks_more() -> None:
    text = "\n".join(f"Line {i}" for i in range(10))
    assert generate_text_preview(text, num_lines=3) == "Line 0\nLine 1\nLine 2..."
    assert generate_text_preview("a\nb", num_lines=3) == "a\nb"
    assert generate_text_preview("a\nb\nc", num_lines=3) == "a\nb\nc"


def test_preview_truncates_long_lines() -> None:
    preview = generate_text_preview("x" * 130 + "\nshort", num_lines=3)
    assert preview == "x" * 120 + "...\nshort"
    assert generate_text_preview("y" * 120) == "y" * 120
    assert generate_text_preview("abcdef", max_line_length=3) == "abc..."


def test_renderers_delegate_to_the_shared_preview() -> None:
    text = "\n".join(f"row {i} " + "z" * 200 for i in range(6))
    expected = generate_text_preview(text, 4)
    assert OutputFormatter()._generate_preview(text, num_lines=4) == expected
    assert LiveDisplayRenderer(Console())._generate_preview(text, num_lines=4) == expected

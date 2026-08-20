# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Focused tests for processing.native base and streaming helpers."""

from __future__ import annotations

from types import SimpleNamespace

from victor.processing.native import _base
from victor.processing.native import streaming


def test_get_native_version_handles_string_and_non_string_versions(monkeypatch) -> None:
    monkeypatch.setattr(_base, "_native", SimpleNamespace(__version__="1.2.3"))
    assert _base.get_native_version() == "1.2.3"

    monkeypatch.setattr(_base, "_native", SimpleNamespace(__version__=7))
    assert _base.get_native_version() == "7"

    monkeypatch.setattr(_base, "_native", None)
    assert _base.get_native_version() is None


def test_streaming_helpers_use_python_fallback_when_native_missing(monkeypatch) -> None:
    monkeypatch.setattr(streaming, "_native", None)
    monkeypatch.setattr(streaming, "_NATIVE_AVAILABLE", False)

    content = "alpha<think>secret</think>omega"

    assert streaming.strip_thinking_tokens(content) == "alphasecretomega"
    assert streaming.contains_thinking_tokens(content) is True
    assert streaming.extract_thinking_content(content) == ("alphaomega", "secret")
    assert streaming.detect_circular_phrases("Let me read the file first") is True
    assert streaming.find_circular_patterns("Now let me check this") == [
        (0, 10, "Now let me"),
        (4, 16, "let me check"),
    ]


def test_streaming_helpers_delegate_to_native_module_when_present(monkeypatch) -> None:
    native = SimpleNamespace(
        strip_thinking_tokens=lambda content: f"native-strip:{content}",
        contains_thinking_tokens=lambda content: content == "x",
        find_thinking_tokens=lambda content: [(1, 2, 3)],
        extract_thinking_content=lambda content: ("main", "thinking"),
        detect_circular_phrases=lambda text: True,
        count_circular_patterns=lambda text: 4,
        find_circular_patterns=lambda text: [(0, len(text), text)],
    )
    monkeypatch.setattr(streaming, "_native", native)
    monkeypatch.setattr(streaming, "_NATIVE_AVAILABLE", True)

    assert streaming.strip_thinking_tokens("abc") == "native-strip:abc"
    assert streaming.contains_thinking_tokens("x") is True
    assert streaming.find_thinking_tokens("abc") == [(1, 2, 3)]
    assert streaming.extract_thinking_content("abc") == ("main", "thinking")
    assert streaming.detect_circular_phrases("abc") is True
    assert streaming.count_circular_patterns("abc") == 4
    assert streaming.find_circular_patterns("abc") == [(0, 3, "abc")]

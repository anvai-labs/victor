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

"""Tests for inline completion protocol types and data structures."""

from pathlib import Path

from victor.processing.completion.protocol import (
    CompletionCapabilities,
    CompletionContext,
    CompletionItem,
    CompletionItemKind,
    CompletionItemLabelDetails,
    CompletionList,
    CompletionMetrics,
    CompletionParams,
    CompletionTriggerKind,
    InlineCompletionItem,
    InlineCompletionList,
    InlineCompletionParams,
    InsertTextFormat,
    Position,
    Range,
    TextEdit,
)

# =============================================================================
# ENUM TESTS
# =============================================================================


class TestCompletionItemKind:
    """Tests for CompletionItemKind enum."""

    def test_text_kind(self):
        """Test TEXT kind value."""
        assert CompletionItemKind.TEXT == 1

    def test_method_kind(self):
        """Test METHOD kind value."""
        assert CompletionItemKind.METHOD == 2

    def test_function_kind(self):
        """Test FUNCTION kind value."""
        assert CompletionItemKind.FUNCTION == 3

    def test_class_kind(self):
        """Test CLASS kind value."""
        assert CompletionItemKind.CLASS == 7

    def test_variable_kind(self):
        """Test VARIABLE kind value."""
        assert CompletionItemKind.VARIABLE == 6

    def test_keyword_kind(self):
        """Test KEYWORD kind value."""
        assert CompletionItemKind.KEYWORD == 14

    def test_snippet_kind(self):
        """Test SNIPPET kind value."""
        assert CompletionItemKind.SNIPPET == 15


class TestInsertTextFormat:
    """Tests for InsertTextFormat enum."""

    def test_plain_text(self):
        """Test PLAIN_TEXT value."""
        assert InsertTextFormat.PLAIN_TEXT == 1

    def test_snippet(self):
        """Test SNIPPET value."""
        assert InsertTextFormat.SNIPPET == 2


class TestCompletionTriggerKind:
    """Tests for CompletionTriggerKind enum."""

    def test_invoked(self):
        """Test INVOKED trigger kind."""
        assert CompletionTriggerKind.INVOKED == 1

    def test_trigger_character(self):
        """Test TRIGGER_CHARACTER trigger kind."""
        assert CompletionTriggerKind.TRIGGER_CHARACTER == 2

    def test_trigger_for_incomplete(self):
        """Test TRIGGER_FOR_INCOMPLETE trigger kind."""
        assert CompletionTriggerKind.TRIGGER_FOR_INCOMPLETE == 3


# =============================================================================
# POSITION TESTS
# =============================================================================


class TestPosition:
    """Tests for Position dataclass."""

    def test_creation(self):
        """Test position creation."""
        pos = Position(line=10, character=5)
        assert pos.line == 10
        assert pos.character == 5

    def test_lt_different_lines(self):
        """Test less than comparison on different lines."""
        pos1 = Position(5, 10)
        pos2 = Position(10, 0)
        assert pos1 < pos2
        assert not (pos2 < pos1)

    def test_lt_same_line(self):
        """Test less than comparison on same line."""
        pos1 = Position(5, 5)
        pos2 = Position(5, 10)
        assert pos1 < pos2
        assert not (pos2 < pos1)

    def test_lt_equal_positions(self):
        """Test less than with equal positions."""
        pos1 = Position(5, 5)
        pos2 = Position(5, 5)
        assert not (pos1 < pos2)
        assert not (pos2 < pos1)

    def test_le_equal(self):
        """Test less than or equal with equal positions."""
        pos1 = Position(5, 5)
        pos2 = Position(5, 5)
        assert pos1 <= pos2

    def test_le_less_than(self):
        """Test less than or equal when less."""
        pos1 = Position(3, 5)
        pos2 = Position(5, 5)
        assert pos1 <= pos2


# =============================================================================
# RANGE TESTS
# =============================================================================


class TestRange:
    """Tests for Range dataclass."""

    def test_creation(self):
        """Test range creation."""
        start = Position(0, 0)
        end = Position(0, 10)
        r = Range(start=start, end=end)
        assert r.start == start
        assert r.end == end


# =============================================================================
# TEXT EDIT TESTS
# =============================================================================


class TestTextEdit:
    """Tests for TextEdit dataclass."""

    def test_creation(self):
        """Test text edit creation."""
        r = Range(Position(0, 0), Position(0, 5))
        edit = TextEdit(range=r, new_text="hello")
        assert edit.new_text == "hello"
        assert edit.range.start.character == 0


# =============================================================================
# COMPLETION CONTEXT TESTS
# =============================================================================


class TestCompletionContext:
    """Tests for CompletionContext dataclass."""

    def test_creation_invoked(self):
        """Test context with invoked trigger."""
        ctx = CompletionContext(trigger_kind=CompletionTriggerKind.INVOKED)
        assert ctx.trigger_kind == CompletionTriggerKind.INVOKED
        assert ctx.trigger_character is None

    def test_creation_with_trigger_char(self):
        """Test context with trigger character."""
        ctx = CompletionContext(
            trigger_kind=CompletionTriggerKind.TRIGGER_CHARACTER,
            trigger_character=".",
        )
        assert ctx.trigger_character == "."


# =============================================================================
# COMPLETION PARAMS TESTS
# =============================================================================


class TestCompletionParams:
    """Tests for CompletionParams dataclass."""

    def test_creation_minimal(self):
        """Test minimal params creation."""
        params = CompletionParams(
            file_path=Path("test.py"),
            position=Position(10, 5),
        )
        assert params.file_path == Path("test.py")
        assert params.prefix == ""
        assert params.max_results == 10

    def test_creation_full(self):
        """Test full params creation."""
        ctx = CompletionContext(CompletionTriggerKind.TRIGGER_CHARACTER, ".")
        params = CompletionParams(
            file_path=Path("test.py"),
            position=Position(10, 5),
            context=ctx,
            prefix="self.",
            suffix="",
            file_content="class Test:\n    def method(self):\n        self.",
            language="python",
            max_results=20,
        )
        assert params.context.trigger_character == "."
        assert params.prefix == "self."
        assert params.language == "python"


# =============================================================================
# COMPLETION ITEM LABEL DETAILS TESTS
# =============================================================================


class TestCompletionItemLabelDetails:
    """Tests for CompletionItemLabelDetails dataclass."""

    def test_creation_empty(self):
        """Test empty label details."""
        details = CompletionItemLabelDetails()
        assert details.detail is None
        assert details.description is None

    def test_creation_full(self):
        """Test full label details."""
        details = CompletionItemLabelDetails(
            detail="(self, x: int) -> int",
            description="mymodule.MyClass",
        )
        assert details.detail == "(self, x: int) -> int"
        assert details.description == "mymodule.MyClass"


# =============================================================================
# COMPLETION ITEM TESTS
# =============================================================================


class TestCompletionItem:
    """Tests for CompletionItem dataclass."""

    def test_creation_minimal(self):
        """Test minimal completion item."""
        item = CompletionItem(label="print")
        assert item.label == "print"
        assert item.kind == CompletionItemKind.TEXT
        assert item.deprecated is False
        assert item.confidence == 1.0

    def test_creation_function(self):
        """Test function completion item."""
        item = CompletionItem(
            label="print",
            kind=CompletionItemKind.FUNCTION,
            detail="(value: Any) -> None",
            documentation="Print values to the console.",
            insert_text="print($1)",
            insert_text_format=InsertTextFormat.SNIPPET,
        )
        assert item.kind == CompletionItemKind.FUNCTION
        assert item.insert_text_format == InsertTextFormat.SNIPPET

    def test_creation_with_ai_fields(self):
        """Test completion item with AI fields."""
        item = CompletionItem(
            label="calculate_total",
            kind=CompletionItemKind.METHOD,
            provider="lmstudio",
            confidence=0.95,
            tokens_used=50,
            latency_ms=150.5,
        )
        assert item.provider == "lmstudio"
        assert item.confidence == 0.95
        assert item.tokens_used == 50
        assert item.latency_ms == 150.5

    def test_creation_with_text_edit(self):
        """Test completion item with text edit."""
        edit = TextEdit(
            range=Range(Position(10, 0), Position(10, 4)),
            new_text="print()",
        )
        item = CompletionItem(
            label="print",
            text_edit=edit,
        )
        assert item.text_edit.new_text == "print()"


# =============================================================================
# INLINE COMPLETION ITEM TESTS
# =============================================================================


class TestInlineCompletionItem:
    """Tests for InlineCompletionItem dataclass."""

    def test_creation_minimal(self):
        """Test minimal inline completion item."""
        item = InlineCompletionItem(insert_text="print('hello')")
        assert item.insert_text == "print('hello')"
        assert item.range is None
        assert item.is_complete is True

    def test_creation_full(self):
        """Test full inline completion item."""
        item = InlineCompletionItem(
            insert_text="def calculate(x, y):\n    return x + y",
            range=Range(Position(5, 0), Position(5, 0)),
            filter_text="def calc",
            provider="ollama",
            confidence=0.9,
            is_complete=True,
            tokens_used=100,
            latency_ms=250.0,
        )
        assert item.provider == "ollama"
        assert item.tokens_used == 100


# =============================================================================
# INLINE COMPLETION PARAMS TESTS
# =============================================================================


class TestInlineCompletionParams:
    """Tests for InlineCompletionParams dataclass."""

    def test_creation_minimal(self):
        """Test minimal inline completion params."""
        params = InlineCompletionParams(
            file_path=Path("test.py"),
            position=Position(10, 20),
        )
        assert params.max_tokens == 256
        assert params.temperature == 0.0

    def test_creation_full(self):
        """Test full inline completion params."""
        params = InlineCompletionParams(
            file_path=Path("main.py"),
            position=Position(50, 10),
            prefix="def calculate(self, x, y):\n    ",
            suffix="\n    return result",
            file_content="Full file content here",
            language="python",
            max_tokens=512,
            temperature=0.2,
            stop_sequences=["\n\n", "def "],
        )
        assert params.max_tokens == 512
        assert params.temperature == 0.2
        assert len(params.stop_sequences) == 2


# =============================================================================
# COMPLETION LIST TESTS
# =============================================================================


class TestCompletionList:
    """Tests for CompletionList dataclass."""

    def test_creation_empty(self):
        """Test empty completion list."""
        lst = CompletionList(is_incomplete=False)
        assert lst.is_incomplete is False
        assert lst.items == []

    def test_creation_with_items(self):
        """Test completion list with items."""
        items = [
            CompletionItem("print"),
            CompletionItem("len"),
        ]
        lst = CompletionList(is_incomplete=True, items=items)
        assert lst.is_incomplete is True
        assert len(lst.items) == 2


# =============================================================================
# INLINE COMPLETION LIST TESTS
# =============================================================================


class TestInlineCompletionList:
    """Tests for InlineCompletionList dataclass."""

    def test_creation_empty(self):
        """Test empty inline completion list."""
        lst = InlineCompletionList()
        assert lst.items == []

    def test_creation_with_items(self):
        """Test inline completion list with items."""
        items = [
            InlineCompletionItem("completion 1"),
            InlineCompletionItem("completion 2"),
        ]
        lst = InlineCompletionList(items=items)
        assert len(lst.items) == 2


# =============================================================================
# COMPLETION CAPABILITIES TESTS
# =============================================================================


class TestCompletionCapabilities:
    """Tests for CompletionCapabilities dataclass."""

    def test_default_capabilities(self):
        """Test default capabilities."""
        caps = CompletionCapabilities()
        assert caps.supports_completion is True
        assert caps.supports_inline_completion is False
        assert caps.supports_resolve is False
        assert caps.supports_snippets is False
        assert caps.trigger_characters == []
        assert caps.max_context_lines == 100

    def test_custom_capabilities(self):
        """Test custom capabilities."""
        caps = CompletionCapabilities(
            supports_completion=True,
            supports_inline_completion=True,
            supports_snippets=True,
            supports_multi_line=True,
            supports_streaming=True,
            trigger_characters=[".", "("],
            max_context_lines=200,
            supported_languages=["python", "javascript"],
        )
        assert caps.supports_inline_completion is True
        assert caps.supports_multi_line is True
        assert len(caps.trigger_characters) == 2
        assert "python" in caps.supported_languages


# =============================================================================
# COMPLETION METRICS TESTS
# =============================================================================


class TestCompletionMetrics:
    """Tests for CompletionMetrics dataclass."""

    def test_default_metrics(self):
        """Test default metrics."""
        metrics = CompletionMetrics()
        assert metrics.total_requests == 0
        assert metrics.successful_requests == 0
        assert metrics.cache_hits == 0

    def test_average_latency_no_requests(self):
        """Test average latency with no requests."""
        metrics = CompletionMetrics()
        assert metrics.average_latency_ms == 0.0

    def test_average_latency_with_requests(self):
        """Test average latency with requests."""
        metrics = CompletionMetrics(
            total_requests=10,
            successful_requests=10,
            total_latency_ms=1000.0,
        )
        assert metrics.average_latency_ms == 100.0

    def test_success_rate_no_requests(self):
        """Test success rate with no requests."""
        metrics = CompletionMetrics()
        assert metrics.success_rate == 0.0

    def test_success_rate_with_requests(self):
        """Test success rate with requests."""
        metrics = CompletionMetrics(
            total_requests=100,
            successful_requests=90,
            failed_requests=10,
        )
        assert metrics.success_rate == 0.9

    def test_full_metrics(self):
        """Test full metrics."""
        metrics = CompletionMetrics(
            total_requests=100,
            successful_requests=95,
            failed_requests=5,
            total_latency_ms=5000.0,
            total_tokens_used=10000,
            cache_hits=30,
            cache_misses=70,
        )
        assert metrics.total_tokens_used == 10000
        assert metrics.cache_hits == 30

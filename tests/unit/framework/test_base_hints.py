"""Tests for base task hints (WS-5)."""

from victor.core.vertical_types import TaskTypeHint
from victor.framework.prompts.base_hints import BaseTaskHints, COMMON_HINTS


class TestBaseTaskHints:

    def test_common_hints_exist(self):
        assert "exploration" in COMMON_HINTS
        assert "debugging" in COMMON_HINTS
        assert "refactoring" in COMMON_HINTS

    def test_common_hints_are_task_type_hints(self):
        for key, value in COMMON_HINTS.items():
            assert isinstance(value, TaskTypeHint)
            assert value.task_type == key

    def test_get_hint_returns_common(self):
        hint = BaseTaskHints.get_hint("debugging")
        assert hint is not None
        assert "DEBUG" in hint

    def test_get_hint_returns_none_for_unknown(self):
        hint = BaseTaskHints.get_hint("nonexistent_task_type")
        assert hint is None

    def test_subclass_vertical_hints_take_priority(self):
        class CodingHints(BaseTaskHints):
            @classmethod
            def get_vertical_hints(cls):
                return {"debugging": "[CODING DEBUG] Custom hint"}

        hint = CodingHints.get_hint("debugging")
        assert "CODING DEBUG" in hint

    def test_get_all_hints(self):
        hints = BaseTaskHints.get_all_hints()
        assert len(hints) == len(COMMON_HINTS)
        # Values should be strings, not TaskTypeHint objects
        for v in hints.values():
            assert isinstance(v, str)

    def test_get_task_type_hints(self):
        hints = BaseTaskHints.get_task_type_hints()
        assert len(hints) == len(COMMON_HINTS)
        for v in hints.values():
            assert isinstance(v, TaskTypeHint)

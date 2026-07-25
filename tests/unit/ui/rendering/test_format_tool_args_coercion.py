"""format_tool_args must tolerate non-mapping argument shapes.

Regression: session modality-doc-review-fixes-b4e87728 — a raw JSON string
reached format_tool_args and 'str' object has no attribute 'items' replaced
the real provider error in victor run output.
"""

from victor.ui.rendering.utils import (
    _coerce_arguments_mapping,
    format_tool_args,
    format_tool_args_bash_style,
)


class TestCoerceArgumentsMapping:
    def test_dict_passthrough(self):
        assert _coerce_arguments_mapping({"a": 1}) == {"a": 1}

    def test_json_string_parsed(self):
        assert _coerce_arguments_mapping('{"cmd": "ls"}') == {"cmd": "ls"}

    def test_non_json_string_wrapped(self):
        assert _coerce_arguments_mapping("not json {") == {"args": "not json {"}

    def test_json_string_non_object_wrapped(self):
        assert _coerce_arguments_mapping("[1, 2]") == {"args": "[1, 2]"}

    def test_empty_string(self):
        assert _coerce_arguments_mapping("   ") == {}

    def test_none(self):
        assert _coerce_arguments_mapping(None) == {}

    def test_other_type_wrapped(self):
        assert _coerce_arguments_mapping(42) == {"args": "42"}


class TestFormatToolArgsCoercion:
    def test_string_json_arguments(self):
        out = format_tool_args('{"path": "a.py", "limit": 5}')
        assert "path='a.py'" in out
        assert "limit=5" in out

    def test_string_raw_arguments_do_not_raise(self):
        out = format_tool_args('echo "hi" | head')
        assert "args=" in out

    def test_bash_style_string_arguments(self):
        out = format_tool_args_bash_style('{"cmd": "ls"}')
        assert "--cmd='ls'" in out

    def test_dict_unchanged_behavior(self):
        assert format_tool_args({"path": "x.py"}) == "path='x.py'"

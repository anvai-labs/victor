"""Unit tests for per-tool preview strategy system.

Tests are written in TDD style: each test specifies the expected contract
of the strategy, independent of implementation details.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from victor.ui.rendering.tool_preview import (
    RenderedPreview,
    ToolPreviewRenderer,
    _DiffPreviewStrategy,
    _WritePreviewStrategy,
    _ReadPreviewStrategy,
    _ShellPreviewStrategy,
    _SearchPreviewStrategy,
    _DirectoryPreviewStrategy,
    _GenericPreviewStrategy,
    renderer,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _shell_repr(stdout: str = "", stderr: str = "", rc: int = 0) -> str:
    """Return a Python repr string matching the real shell tool output format."""
    return str({"success": rc == 0, "stdout": stdout, "stderr": stderr, "return_code": rc})


def _ls_repr(names: list[str]) -> str:
    items = [{"path": n, "full_path": f"/repo/{n}"} for n in names]
    return str({"items": items})


# ---------------------------------------------------------------------------
# TestRenderedPreview
# ---------------------------------------------------------------------------


class TestRenderedPreview:
    def test_defaults(self):
        p = RenderedPreview()
        assert p.lines == []
        assert p.header is None
        assert p.total_line_count == 0
        assert p.syntax_hint == "text"

    def test_explicit_values(self):
        p = RenderedPreview(
            lines=["a", "b"], header="hdr", total_line_count=10, syntax_hint="python"
        )
        assert p.lines == ["a", "b"]
        assert p.header == "hdr"
        assert p.total_line_count == 10
        assert p.syntax_hint == "python"


# ---------------------------------------------------------------------------
# TestDiffPreviewStrategy
# ---------------------------------------------------------------------------


class TestDiffPreviewStrategy:
    strategy = _DiffPreviewStrategy()

    def test_diff_with_old_and_new_strings(self):
        args = {"old_string": "x = 1", "new_string": "x = 2"}
        p = self.strategy.render("edit", args, "", max_lines=5)
        assert p.header is not None
        assert "+1" in p.header
        assert "-1" in p.header
        # Lines should include the changed line
        joined = "\n".join(p.lines)
        assert "+x = 2" in joined
        assert "-x = 1" in joined

    def test_diff_multiline(self):
        old = "a = 1\nb = 2\nc = 3"
        new = "a = 10\nb = 2\nc = 30"
        args = {"old_string": old, "new_string": new}
        p = self.strategy.render("edit", args, "", max_lines=10)
        assert p.header is not None
        assert p.total_line_count > 0
        joined = "\n".join(p.lines)
        assert "+a = 10" in joined
        assert "+c = 30" in joined

    def test_diff_identical_strings_shows_zero_change(self):
        args = {"old_string": "same", "new_string": "same"}
        p = self.strategy.render("edit", args, "", max_lines=5)
        assert p.header is not None
        assert "+0" in p.header
        assert "-0" in p.header

    def test_diff_fallback_to_result_when_no_args(self):
        result = str({"success": True, "operations_applied": 3, "message": "Applied 3 ops"})
        p = self.strategy.render("edit", {}, result, max_lines=3)
        # Should parse result and show op count summary
        assert p.lines or p.header  # something is shown
        joined = " ".join(p.lines) + (p.header or "")
        # Either "3" or "operation" should appear
        assert "3" in joined or "operation" in joined.lower()

    def test_diff_max_lines_respected(self):
        old = "\n".join(f"line{i}" for i in range(20))
        new = "\n".join(f"changed{i}" for i in range(20))
        args = {"old_string": old, "new_string": new}
        p = self.strategy.render("edit", args, "", max_lines=3)
        assert len(p.lines) <= 3


# ---------------------------------------------------------------------------
# TestWritePreviewStrategy
# ---------------------------------------------------------------------------


class TestWritePreviewStrategy:
    strategy = _WritePreviewStrategy()

    def test_write_with_content_in_args(self):
        content = "import os\nprint(os.getcwd())\nprint('done')"
        args = {"file_path": "run.py", "content": content}
        p = self.strategy.render("write", args, "", max_lines=2)
        assert p.header is not None
        assert "3" in p.header  # 3 lines
        assert "run.py" in p.header
        assert "import os" in p.lines

    def test_write_line_count_in_header(self):
        content = "\n".join(f"line{i}" for i in range(50))
        args = {"file_path": "big.py", "content": content}
        p = self.strategy.render("write", args, "", max_lines=3)
        assert "50" in p.header

    def test_write_syntax_hint_python(self):
        args = {"file_path": "foo.py", "content": "x = 1"}
        p = self.strategy.render("write", args, "", max_lines=3)
        assert p.syntax_hint == "python"

    def test_write_syntax_hint_typescript(self):
        args = {"file_path": "comp.ts", "content": "const x = 1;"}
        p = self.strategy.render("write", args, "", max_lines=3)
        assert p.syntax_hint == "typescript"

    def test_write_syntax_hint_unknown_extension(self):
        args = {"file_path": "data.xyz", "content": "stuff"}
        p = self.strategy.render("write", args, "", max_lines=3)
        assert p.syntax_hint == "text"

    def test_write_no_content_in_args_uses_result(self):
        result = str({"success": True, "file_path": "output.py"})
        args = {"file_path": "output.py"}
        p = self.strategy.render("write", args, result, max_lines=3)
        # Should at minimum not crash and return something
        assert isinstance(p, RenderedPreview)

    def test_write_max_lines_respected(self):
        content = "\n".join(f"line{i}" for i in range(20))
        args = {"file_path": "f.py", "content": content}
        p = self.strategy.render("write", args, "", max_lines=2)
        assert len(p.lines) <= 2


# ---------------------------------------------------------------------------
# TestReadPreviewStrategy
# ---------------------------------------------------------------------------


class TestReadPreviewStrategy:
    strategy = _ReadPreviewStrategy()

    def test_read_parses_victor_metadata_header(self):
        raw = "[File: victor/ui/commands/chat.py]\n[Lines 1-3 of 1633]\nfoo = 1\nbar = 2\nbaz = 3"
        p = self.strategy.render("read", {"path": "victor/ui/commands/chat.py"}, raw, max_lines=3)
        assert p.header is not None
        assert "File:" in p.header
        assert "Lines" in p.header
        # Content lines should NOT include the metadata lines
        assert "foo = 1" in p.lines

    def test_read_without_metadata(self):
        raw = "line one\nline two\nline three\nline four"
        p = self.strategy.render("read", {}, raw, max_lines=2)
        assert p.header is None
        assert len(p.lines) <= 2
        assert "line one" in p.lines

    def test_read_empty(self):
        p = self.strategy.render("read", {}, "", max_lines=3)
        assert p.lines == []

    def test_read_syntax_hint_from_path_arg(self):
        raw = "[File: foo.py]\nprint('hello')"
        p = self.strategy.render("read", {"path": "foo.py"}, raw, max_lines=3)
        assert p.syntax_hint == "python"

    def test_read_max_lines_respected(self):
        raw = "\n".join(f"line {i}" for i in range(20))
        p = self.strategy.render("read", {}, raw, max_lines=2)
        assert len(p.lines) <= 2

    def test_read_size_metadata_line_goes_to_header(self):
        raw = "[File: a.py]\n[Lines 1-10 of 100]\n[Size: 1,234 bytes]\ncode here"
        p = self.strategy.render("read", {}, raw, max_lines=3)
        assert "code here" in p.lines
        assert "Size" in p.header


# ---------------------------------------------------------------------------
# TestShellPreviewStrategy
# ---------------------------------------------------------------------------


class TestShellPreviewStrategy:
    strategy = _ShellPreviewStrategy()

    def test_shell_success_shows_exit_0_and_stdout(self):
        raw = _shell_repr(stdout="hello\nworld", rc=0)
        p = self.strategy.render("shell", {}, raw, max_lines=3)
        assert p.header == "[exit 0]"
        assert "hello" in p.lines
        assert "world" in p.lines

    def test_shell_nonzero_exit_shows_exit_code(self):
        raw = _shell_repr(stdout="", stderr="command not found", rc=127)
        p = self.strategy.render("shell", {}, raw, max_lines=3)
        assert "127" in p.header
        assert "command not found" in p.lines

    def test_shell_empty_stdout_still_shows_header(self):
        raw = _shell_repr(stdout="", rc=0)
        p = self.strategy.render("shell", {}, raw, max_lines=3)
        assert p.header == "[exit 0]"
        assert p.lines == []

    def test_shell_prefers_stdout_over_stderr_on_success(self):
        raw = _shell_repr(stdout="output", stderr="some warning", rc=0)
        p = self.strategy.render("shell", {}, raw, max_lines=3)
        assert "output" in p.lines
        # stderr not shown when rc=0 and stdout exists

    def test_shell_invalid_input_falls_back(self):
        p = self.strategy.render("shell", {}, "not a dict at all just plain text", max_lines=3)
        assert isinstance(p, RenderedPreview)
        # Should not raise; falls back to generic

    def test_shell_max_lines_respected(self):
        stdout = "\n".join(f"line{i}" for i in range(20))
        raw = _shell_repr(stdout=stdout, rc=0)
        p = self.strategy.render("shell", {}, raw, max_lines=2)
        assert len(p.lines) <= 2

    def test_shell_total_line_count_reflects_full_stdout(self):
        stdout = "\n".join(f"line{i}" for i in range(10))
        raw = _shell_repr(stdout=stdout, rc=0)
        p = self.strategy.render("shell", {}, raw, max_lines=2)
        assert p.total_line_count >= 10


# ---------------------------------------------------------------------------
# TestSearchPreviewStrategy
# ---------------------------------------------------------------------------


class TestSearchPreviewStrategy:
    strategy = _SearchPreviewStrategy()

    def test_grep_plain_text_matches(self):
        raw = "foo.py:1:match one\nfoo.py:2:match two\nbar.py:5:match three"
        p = self.strategy.render("grep", {}, raw, max_lines=2)
        assert p.header is not None
        assert "3" in p.header
        assert len(p.lines) <= 2

    def test_search_empty_result(self):
        p = self.strategy.render("grep", {}, "", max_lines=3)
        assert p.header is not None
        assert "0" in p.header

    def test_glob_structured_result_with_matches_key(self):
        raw = str({"matches": ["a.py", "b.py", "c.py"]})
        p = self.strategy.render("glob", {}, raw, max_lines=2)
        assert "3" in p.header
        assert len(p.lines) <= 2

    def test_search_counts_non_empty_lines_only(self):
        raw = "match1\n\nmatch2\n"  # blank lines should not count
        p = self.strategy.render("grep", {}, raw, max_lines=3)
        assert "2" in p.header

    def test_code_search_structured_results_key(self):
        raw = str({"results": [{"file": "a.py"}, {"file": "b.py"}]})
        p = self.strategy.render("code_search", {}, raw, max_lines=3)
        assert "2" in p.header


# ---------------------------------------------------------------------------
# TestDirectoryPreviewStrategy
# ---------------------------------------------------------------------------


class TestDirectoryPreviewStrategy:
    strategy = _DirectoryPreviewStrategy()

    def test_ls_item_list(self):
        raw = _ls_repr(["a.py", "b.py", "c.py"])
        p = self.strategy.render("ls", {}, raw, max_lines=2)
        assert "3" in p.header
        assert len(p.lines) <= 2
        # Names should appear
        assert any("a.py" in l or "b.py" in l for l in p.lines)

    def test_ls_empty_dir(self):
        raw = str({"items": []})
        p = self.strategy.render("ls", {}, raw, max_lines=3)
        assert "0" in p.header
        assert p.lines == []

    def test_ls_max_lines_respected(self):
        raw = _ls_repr([f"file{i}.py" for i in range(20)])
        p = self.strategy.render("ls", {}, raw, max_lines=3)
        assert len(p.lines) <= 3

    def test_ls_falls_back_for_plain_text(self):
        p = self.strategy.render("ls", {}, "plain text output", max_lines=3)
        assert isinstance(p, RenderedPreview)

    def test_ls_total_line_count_is_item_count(self):
        raw = _ls_repr(["a.py", "b.py", "c.py", "d.py", "e.py"])
        p = self.strategy.render("ls", {}, raw, max_lines=2)
        assert p.total_line_count == 5


# ---------------------------------------------------------------------------
# TestGenericPreviewStrategy
# ---------------------------------------------------------------------------


class TestGenericPreviewStrategy:
    strategy = _GenericPreviewStrategy()

    def test_generic_first_n_lines(self):
        raw = "line1\nline2\nline3\nline4\nline5"
        p = self.strategy.render("unknown", {}, raw, max_lines=3)
        assert len(p.lines) == 3
        assert p.lines[0] == "line1"

    def test_generic_empty(self):
        p = self.strategy.render("unknown", {}, "", max_lines=3)
        assert p.lines == []
        assert p.total_line_count == 0

    def test_generic_long_lines_clamped(self):
        long_line = "x" * 200
        p = self.strategy.render("unknown", {}, long_line, max_lines=1)
        assert len(p.lines) == 1
        assert len(p.lines[0]) < 200  # was truncated
        assert p.lines[0].endswith("…")

    def test_generic_total_line_count(self):
        raw = "\n".join(f"l{i}" for i in range(50))
        p = self.strategy.render("unknown", {}, raw, max_lines=3)
        assert p.total_line_count == 50

    def test_generic_no_header(self):
        p = self.strategy.render("unknown", {}, "some text", max_lines=3)
        assert p.header is None


# ---------------------------------------------------------------------------
# TestToolPreviewRenderer
# ---------------------------------------------------------------------------


class TestToolPreviewRenderer:
    def test_dispatch_edit_to_diff_strategy(self):
        r = ToolPreviewRenderer()
        args = {"old_string": "a = 1", "new_string": "a = 2"}
        p = r.render("edit", args, "", max_lines=5)
        assert p.header is not None
        assert "+" in p.header and "-" in p.header

    def test_dispatch_shell_to_shell_strategy(self):
        r = ToolPreviewRenderer()
        raw = _shell_repr(stdout="hello", rc=0)
        p = r.render("shell", {}, raw, max_lines=3)
        assert p.header == "[exit 0]"

    def test_dispatch_ls_to_directory_strategy(self):
        r = ToolPreviewRenderer()
        raw = _ls_repr(["a.py", "b.py"])
        p = r.render("ls", {}, raw, max_lines=3)
        assert "2" in p.header

    def test_unknown_tool_uses_generic(self):
        r = ToolPreviewRenderer()
        p = r.render("totally_unknown_tool", {}, "line1\nline2", max_lines=3)
        assert p.header is None
        assert "line1" in p.lines

    def test_custom_strategy_registration(self):
        r = ToolPreviewRenderer()
        mock_strategy = MagicMock()
        mock_strategy.render.return_value = RenderedPreview(
            lines=["custom"], header="custom header"
        )
        r.register("my_custom_tool", mock_strategy)
        p = r.render("my_custom_tool", {}, "raw", max_lines=3)
        assert p.header == "custom header"
        assert "custom" in p.lines
        mock_strategy.render.assert_called_once_with("my_custom_tool", {}, "raw", 3)

    def test_strategy_exception_falls_back_to_generic(self):
        r = ToolPreviewRenderer()
        bad_strategy = MagicMock()
        bad_strategy.render.side_effect = RuntimeError("boom")
        r.register("bad_tool", bad_strategy)
        p = r.render("bad_tool", {}, "fallback content\nline2", max_lines=3)
        # Must not raise; falls back to generic
        assert "fallback content" in p.lines

    def test_max_lines_1_returns_at_most_one_line(self):
        r = ToolPreviewRenderer()
        raw = "a\nb\nc\nd\ne"
        p = r.render("unknown", {}, raw, max_lines=1)
        assert len(p.lines) <= 1

    def test_module_level_renderer_singleton_works(self):
        # The module-level `renderer` should be usable directly
        p = renderer.render("ls", {}, _ls_repr(["x.py"]), max_lines=3)
        assert "1" in p.header

    def test_all_registered_tool_names_dispatch_correctly(self):
        """Every registered tool must produce a non-None result (no crashes)."""
        r = ToolPreviewRenderer()
        for tool_name in r._strategies:
            p = r.render(tool_name, {}, "sample output", max_lines=3)
            assert isinstance(
                p, RenderedPreview
            ), f"Strategy for {tool_name!r} returned non-RenderedPreview"

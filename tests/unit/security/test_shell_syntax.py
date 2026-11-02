"""Lexical write boundaries shared by readonly and damage-scoped policies."""

from __future__ import annotations

import pytest

from victor.security.shell_syntax import ShellWrite, scan_file_writes


@pytest.mark.parametrize(
    "command, target",
    [
        ("echo data>out.txt", "out.txt"),
        ("cat <>file", "file"),
        ("echo a>>out.txt", "out.txt"),
        ("echo a&>out.txt", "out.txt"),
        ("echo a&>>out.txt", "out.txt"),
        ("echo a>|out.txt", "out.txt"),
        ('echo a>"file with spaces"', "file with spaces"),
        ("echo a>file\\ with\\ spaces", "file with spaces"),
    ],
)
def test_writes_do_not_require_whitespace(command, target):
    assert [write.target for write in scan_file_writes(command)] == [target]


@pytest.mark.parametrize(
    "command",
    [
        "echo 'text>file'",
        'echo "text>file"',
        "echo text\\>file",
        "cat <file",
        "cat 2>&1",
        "cat 3>&-",
        "cat 2>&3",
    ],
)
def test_literal_text_and_descriptor_operations_do_not_open_files(command):
    assert scan_file_writes(command) == []


def test_dynamic_target_is_preserved_instead_of_resolved_as_a_literal_path():
    assert scan_file_writes('echo a>"$OUT"') == [ShellWrite(">", "$OUT", True)]
    assert scan_file_writes("echo a>'$OUT'") == [ShellWrite(">", "$OUT", False)]
    assert scan_file_writes("echo a>\\$OUT") == [ShellWrite(">", "$OUT", False)]


def test_adjacent_quoted_targets_preserve_both_writes():
    assert scan_file_writes("echo a>'out file'2>error") == [
        ShellWrite(">", "out file2"),
        ShellWrite(">", "error"),
    ]


@pytest.mark.parametrize(
    "command",
    [
        'echo "$(echo x>/etc/hosts)"',
        r"echo `echo \`echo x>/etc/hosts\``",
        "cat <(echo x>/etc/hosts)",
    ],
)
def test_nested_substitution_redirections_are_visible(command):
    assert [write.target for write in scan_file_writes(command)] == ["/etc/hosts"]


def test_substitution_body_quote_context_is_independent_of_its_parent():
    from victor.security.shell_syntax import scan_substitutions

    children, sanitized = scan_substitutions("echo \"$(printf '(%s)' value)\"")
    assert children == ["printf '(%s)' value"]
    assert "printf" not in sanitized
    assert scan_substitutions("echo '$(touch out)' ")[0] == []


def test_excessive_substitution_nesting_is_rejected_without_recursion_failure():
    from victor.security.shell_syntax import MAX_SHELL_NESTING, ShellSyntaxError

    with pytest.raises(ShellSyntaxError, match="nesting"):
        scan_file_writes(
            "echo " + "$(echo " * (MAX_SHELL_NESTING + 1) + "x" + ")" * (MAX_SHELL_NESTING + 1)
        )

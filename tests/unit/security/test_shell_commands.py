"""Inspection permission belongs to a concrete operation, never a CLI family."""

from __future__ import annotations

import shlex

import pytest

from victor.security.shell_commands import inspection_command


@pytest.mark.parametrize(
    "command",
    [
        "gh pr merge 1",
        "gh repo delete repo",
        "gh api repos/a/b --method PATCH",
        "gh api repos/a/b --field x=y",
        "gh api graphql --method GET",
        "docker buildx inspect --bootstrap",
        "helm repo update",
        "kubectl config use-context x",
        "pip wheel pkg",
        "pip config set x y",
        "npm audit fix",
        "npm pkg set name=bad",
        "python3 -c pass",
        "pytest --help",
        "make test",
        "cargo test",
        "go test ./...",
        "ruff check --fix file.py",
        "eslint --fix file.js",
    ],
)
def test_nested_mutation_or_code_execution_requires_permission(command):
    assert inspection_command(shlex.split(command)) is False


@pytest.mark.parametrize(
    "command",
    [
        "gh -R owner/repo pr view 1",
        "gh api repos/a/b --method HEAD",
        "gh api repos/a/b -XGET",
        "docker --context local buildx inspect",
        "helm repo list",
        "kubectl -n ns get pods",
        "pip config list",
        "npm pkg get name",
        "npm version",
        "cargo tree --locked",
        "go version",
        "python3 --version",
        "node --help",
        "ruff check file.py",
        "black --check file.py",
    ],
)
def test_explicit_inspection_operations_remain_available(command):
    assert inspection_command(shlex.split(command)) is True


@pytest.mark.parametrize(
    "command",
    [
        "pip list --log output.log",
        "pip list --log=output.log",
        "gh pr view 1 --web=true",
        "docker compose config -ooutput.yml",
        "docker compose config --out=output.yml",
        "cargo tree",
        "printf '%n' PATH",
        "printf -vPATH '%s' 0",
        "printf -v okay '%n' PATH",
        "read -aPATH",
        "printf -v 'arr[$(echo fixture)]' x",
        "read 'arr[1]'",
        "ffmpeg -i in.mp4 -vf metadata=mode=print:file=output -f image2pipe -",
        "ffprobe -report input.mp4",
        "tesseract input.png stdout -c debug_file=output",
        "uniq input output",
        "date -s now",
        "hostname changed",
        "mktemp",
    ],
)
def test_complete_inspection_grammar_rejects_effectful_argument_shapes(command):
    assert inspection_command(shlex.split(command)) is False


@pytest.mark.parametrize(
    "command",
    [
        "printf '%%n'",
        "printf -v result '%s' x",
        "printf -vresult '%s' x",
        "read -raresult",
        "read -- result",
        "date -u +%F",
        "uniq -c input",
    ],
)
def test_builtin_receivers_and_literal_formats_remain_available(command):
    assert inspection_command(shlex.split(command)) is True


@pytest.mark.parametrize(
    "command",
    [
        'ACTION=-delete; find . -name -- "$ACTION"',
        'FLAGS=--output=out; git diff --src-prefix -- "$FLAGS"',
        "FLAGS=--checkpoint-action=exec=helper; tar -tf -- $FLAGS",
        'FLAGS=--pre=./helper; rg "$FLAGS" pattern file',
        'printf "$FORMAT" value',
    ],
)
def test_opaque_option_values_require_execution_permission(command):
    from victor.tools.bash import _validate_readonly_command

    assert _validate_readonly_command(command)[0] is False


@pytest.mark.parametrize(
    "command",
    [
        'git diff -- "$FILE"',
        'find "./$DIR" -name "*.py"',
        'printf "%s" "$VALUE"',
    ],
)
def test_provable_positional_and_literal_format_dynamic_data(command):
    from victor.tools.bash import _validate_readonly_command

    assert _validate_readonly_command(command)[0] is True


@pytest.mark.parametrize(
    "command, expected",
    [
        ("echo fixture", True),
        ("whoami", True),
        ("id -u", True),
        ("uname -s", True),
        ("cat file", False),
        ("git status", False),
        ("gh pr view 1", False),
        ("echo fixture <file", False),
        ("echo fixture >file", False),
        ('echo "$VALUE"', False),
        ("echo $(uname)", False),
        ("echo first; echo second", False),
        ("VALUE=x echo fixture", False),
        ("whoami --unknown", False),
        ("id --name", False),
    ],
)
def test_only_literal_process_metadata_can_use_output_cache(command, expected):
    from victor.security.shell_commands import cacheable_inspection

    assert cacheable_inspection(command) is expected

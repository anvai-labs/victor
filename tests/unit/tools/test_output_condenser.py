"""Tests for command-aware shell output condensation (rtk-style)."""

from __future__ import annotations

import re

import pytest

from victor.tools.output_condenser import (
    LINE_FILTERS,
    CondensationResult,
    _cap_section,
    _effective_command,
    condense_shell_output,
    tee_raw_output,
)
from victor.tools.subprocess_executor import _truncate_output_by_lines


def _pytest_output(n_pass: int = 200, failures: bool = True) -> str:
    lines = ["============================= test session starts =============================="]
    lines.append("platform darwin -- Python 3.12.1, pytest-8.0.0, pluggy-1.4.0")
    lines.append("rootdir: /repo")
    lines.append(f"collected {n_pass + (2 if failures else 0)} items")
    lines.append("")
    for i in range(n_pass):
        lines.append(f"tests/unit/test_mod_{i}.py::test_case_{i} PASSED")
    if failures:
        lines.append(
            "=================================== FAILURES ==================================="
        )
        lines.append(
            "_______________________________ test_widget_render _______________________________"
        )
        lines.append("    def test_widget_render():")
        lines.append(">       assert render() == 'ok'")
        lines.append("E       AssertionError: assert 'fail' == 'ok'")
        lines.append("tests/unit/test_widget.py:42: AssertionError")
        lines.append(
            "=========================== short test summary info ============================"
        )
        lines.append("FAILED tests/unit/test_widget.py::test_widget_render - AssertionError")
        lines.append(
            f"========================= 1 failed, {n_pass} passed in 12.34s ========================="
        )
    else:
        lines.append(
            f"============================== {n_pass} passed in 8.21s ==============================="
        )
    return "\n".join(lines)


class TestEffectiveCommand:
    def test_plain_command(self):
        assert _effective_command("pytest tests/unit") == "pytest tests/unit"

    def test_last_segment_of_chain(self):
        assert _effective_command("cd /repo && pytest -q") == "pytest -q"

    def test_env_prefix_stripped(self):
        assert _effective_command("CI=1 FOO=bar pytest") == "pytest"

    def test_wrapper_stripped(self):
        assert _effective_command("uv run pytest tests") == "pytest tests"

    def test_piped_command_excluded(self):
        assert _effective_command("pytest | tail -50") is None


class TestPytestCondenser:
    def test_failures_and_summary_kept_pass_noise_dropped(self):
        out = _pytest_output(n_pass=500, failures=True)
        result = condense_shell_output("pytest tests/unit", out, "", 1, tee_enabled=False)
        assert result is not None
        assert result.condenser == "pytest"
        assert "AssertionError: assert 'fail' == 'ok'" in result.stdout
        assert "FAILED tests/unit/test_widget.py::test_widget_render" in result.stdout
        assert "1 failed, 500 passed in 12.34s" in result.stdout
        assert "test_case_250 PASSED" not in result.stdout
        assert result.savings_pct > 80

    def test_all_pass_collapses_to_summary(self):
        out = _pytest_output(n_pass=300, failures=False)
        result = condense_shell_output("python -m pytest", out, "", 0, tee_enabled=False)
        assert result is not None
        assert "300 passed in 8.21s" in result.stdout
        assert "progress omitted" in result.stdout
        assert "test_case_10 PASSED" not in result.stdout

    def test_no_summary_line_fails_open(self):
        # Crash before reporting: no summary line → passthrough.
        out = "\n".join(
            ["Traceback (most recent call last):"] + [f"  frame {i}" for i in range(40)]
        )
        assert condense_shell_output("pytest", out, "", 2, tee_enabled=False) is None

    def test_small_output_passthrough(self):
        assert condense_shell_output("pytest", "3 passed in 0.1s", "", 0, tee_enabled=False) is None


class TestGitStatusCondenser:
    def _status(self, untracked: int = 60) -> str:
        lines = [
            "On branch develop",
            "Your branch is up to date with 'origin/develop'.",
            "",
            "Changes not staged for commit:",
            '  (use "git add <file>..." to update what will be committed)',
            '  (use "git restore <file>..." to discard changes in working directory)',
            "\tmodified:   victor/tools/bash.py",
            "\tmodified:   victor/config/tool_settings.py",
            "",
            "Untracked files:",
            '  (use "git add <file>..." to include in what will be committed)',
        ]
        lines.extend(f"\tscratch/file_{i}.txt" for i in range(untracked))
        lines.append("")
        lines.append('no changes added to commit (use "git add" and/or "git commit -a")')
        return "\n".join(lines)

    def test_groups_and_caps(self):
        result = condense_shell_output("git status", self._status(60), "", 0, tee_enabled=False)
        assert result is not None
        assert result.condenser == "git-status"
        assert "On branch develop" in result.stdout
        assert "modified (2):" in result.stdout
        assert "untracked (60):" in result.stdout
        assert "+40 more" in result.stdout
        assert "scratch/file_59.txt" not in result.stdout

    def test_porcelain_passthrough(self):
        porcelain = "\n".join(f"?? f{i}.txt" for i in range(60))
        assert (
            condense_shell_output("git status --porcelain", porcelain, "", 0, tee_enabled=False)
            is None
        )


class TestLineFilters:
    def test_git_push_progress_stripped(self):
        stderr = "\n".join(
            [
                "Enumerating objects: 500, done.",
                "Counting objects: 100% (500/500), done.",
                "Compressing objects: 100% (300/300), done.",
                "Writing objects: 100% (400/400), 1.2 MiB | 5.0 MiB/s, done.",
                "remote: Resolving deltas: 100% (200/200), completed with 50 local objects.",
            ]
            * 8
            + ["To github.com:org/repo.git", "   abc1234..def5678  develop -> develop"]
        )
        result = condense_shell_output("git push origin develop", "", stderr, 0, tee_enabled=False)
        assert result is not None
        assert result.condenser == "git-network"
        assert "develop -> develop" in result.stderr
        assert "Counting objects" not in result.stderr

    def test_pip_install_noise_stripped(self):
        out = "\n".join(
            [
                "Collecting requests",
                "  Downloading requests-2.32.0-py3-none-any.whl (64 kB)",
                "Requirement already satisfied: idna in ./venv/lib (3.6)",
            ]
            * 15
            + ["Successfully installed requests-2.32.0"]
        )
        result = condense_shell_output("pip install requests", out, "", 0, tee_enabled=False)
        assert result is not None
        assert "Successfully installed requests-2.32.0" in result.stdout
        assert "Collecting requests" not in result.stdout

    def test_cargo_build_keeps_errors(self):
        out = "\n".join(
            [f"   Compiling crate_{i} v0.1.0" for i in range(40)]
            + [
                "error[E0308]: mismatched types",
                "  --> src/main.rs:10:5",
                "error: could not compile `app` due to previous error",
            ]
        )
        result = condense_shell_output("cargo build", out, "", 101, tee_enabled=False)
        assert result is not None
        assert "error[E0308]: mismatched types" in result.stdout
        assert "Compiling crate_5" not in result.stdout

    def test_all_filters_have_valid_patterns(self):
        for spec in LINE_FILTERS:
            assert isinstance(spec.match_command, re.Pattern)
            assert spec.name


class TestTee:
    def test_tee_writes_and_hints(self, tmp_path, monkeypatch):
        monkeypatch.setattr("victor.tools.output_condenser._tee_dir", lambda: tmp_path)
        out = _pytest_output(n_pass=400, failures=True)
        result = condense_shell_output("pytest tests", out, "", 1, tee_enabled=True)
        assert result is not None
        assert result.raw_log_path is not None
        assert "[condensed by victor; full output:" in result.stdout
        raw = (tmp_path / result.raw_log_path.split("/")[-1]).read_text()
        assert "test_case_399 PASSED" in raw  # raw log is lossless

    def test_tee_rotation(self, tmp_path, monkeypatch):
        monkeypatch.setattr("victor.tools.output_condenser._tee_dir", lambda: tmp_path)
        for i in range(25):
            tee_raw_output(f"cmd {i}", "x" * 3000, "")
        assert len(list(tmp_path.glob("*.log"))) <= 20

    def test_small_output_not_teed(self, tmp_path, monkeypatch):
        monkeypatch.setattr("victor.tools.output_condenser._tee_dir", lambda: tmp_path)
        assert tee_raw_output("cmd", "small", "") is None


class TestCapSection:
    def test_keeps_head_and_tail(self):
        lines = [f"line {i}" for i in range(100)]
        capped = _cap_section(lines, 10)
        assert len(capped) == 11  # 10 kept + omission marker
        assert capped[0] == "line 0"
        assert capped[-1] == "line 99"
        assert any("omitted" in line for line in capped)


class TestTailPreservingTruncation:
    def test_head_and_tail_kept(self):
        text = "\n".join(f"line {i}" for i in range(1000))
        truncated, was_truncated, _ = _truncate_output_by_lines(text, 100, stream_name="stdout")
        assert was_truncated
        assert "line 0" in truncated
        assert "line 999" in truncated  # tail survives now
        assert "middle lines omitted" in truncated

    def test_no_truncation_under_limit(self):
        text = "\n".join(f"line {i}" for i in range(50)) + "\n"
        result, was_truncated, _ = _truncate_output_by_lines(text, 100)
        assert not was_truncated
        assert result == text


class TestGoTestCondenser:
    def test_failures_kept_ok_collapsed(self):
        out = "\n".join(
            [f"ok  \tgithub.com/x/pkg{i}\t0.01{i}s" for i in range(30)]
            + [
                "--- FAIL: TestWidget (0.00s)",
                "    widget_test.go:12: expected 1 got 2",
                "FAIL",
                "FAIL\tgithub.com/x/widget\t0.52s",
                "ok  \tgithub.com/x/last\t(cached)",
            ]
        )
        result = condense_shell_output("go test ./...", out, "", 1, tee_enabled=False)
        assert result is not None
        assert result.condenser == "go-test"
        assert "--- FAIL: TestWidget (0.00s)" in result.stdout
        assert "widget_test.go:12: expected 1 got 2" in result.stdout
        assert "ok: 31 packages, 1 cached" in result.stdout
        assert "github.com/x/pkg5" not in result.stdout

    def test_verbose_pass_noise_stripped(self):
        out = "\n".join(
            [
                item
                for i in range(40)
                for item in (f"=== RUN   TestCase{i}", f"--- PASS: TestCase{i} (0.00s)")
            ]
            + ["PASS", "ok  \tgithub.com/x/pkg\t1.20s"]
        )
        result = condense_shell_output("go test -v ./pkg", out, "", 0, tee_enabled=False)
        assert result is not None
        assert "=== RUN" not in result.stdout
        assert "--- PASS" not in result.stdout
        assert "ok: 1 packages" in result.stdout

    def test_unrecognized_fails_open(self):
        out = "\n".join(f"random line {i}" for i in range(50))
        assert condense_shell_output("go test ./...", out, "", 1, tee_enabled=False) is None


class TestNewLineFilters:
    def test_npm_test_jest_pass_lines_stripped(self):
        out = "\n".join(
            [f"PASS src/mod_{i}.test.ts" for i in range(20)]
            + [f"  ✓ renders widget {i} (3 ms)" for i in range(20)]
            + [
                "FAIL src/broken.test.ts",
                "  ● broken › explodes",
                "Tests:       1 failed, 40 passed, 41 total",
            ]
        )
        result = condense_shell_output("npm test", out, "", 1, tee_enabled=False)
        assert result is not None
        assert result.condenser == "npm-test"
        assert "FAIL src/broken.test.ts" in result.stdout
        assert "Tests:       1 failed, 40 passed, 41 total" in result.stdout
        assert "PASS src/mod_3.test.ts" not in result.stdout

    def test_make_directory_noise_stripped(self):
        out = "\n".join(
            [
                "make[1]: Entering directory '/repo/sub'",
                "gcc -c foo.c",
                "make[1]: Leaving directory '/repo/sub'",
            ]
            * 15
        )
        result = condense_shell_output("make all", out, "", 0, tee_enabled=False)
        assert result is not None
        assert "gcc -c foo.c" in result.stdout
        assert "Entering directory" not in result.stdout

    def test_docker_pull_layer_noise_stripped(self):
        out = "\n".join(
            [f"{'a1b2c3d4e5f'[:11]}{i % 10}: Pull complete" for i in range(30)]
            + ["Digest: sha256:deadbeef", "Status: Downloaded newer image for python:3.12"]
        )
        result = condense_shell_output("docker pull python:3.12", out, "", 0, tee_enabled=False)
        assert result is not None
        assert "Status: Downloaded newer image" in result.stdout
        assert "Pull complete" not in result.stdout

    def test_apt_install_noise_stripped(self):
        out = "\n".join(
            ["Reading package lists...", "Building dependency tree..."]
            + [f"Get:{i} http://archive.ubuntu.com jammy/main pkg{i}" for i in range(20)]
            + [f"Unpacking pkg{i} (1.0-1)" for i in range(20)]
            + ["Setting up pkg1 (1.0-1)", "1 upgraded, 20 newly installed, 0 to remove."]
        )
        result = condense_shell_output("apt-get install -y pkgs", out, "", 0, tee_enabled=False)
        assert result is not None
        assert "1 upgraded, 20 newly installed" in result.stdout
        assert "Unpacking pkg5" not in result.stdout


class TestUserFilterOverlay:
    def _write_yaml(self, path, body):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")

    def test_user_filter_applies_and_wins_over_builtin(self, tmp_path, monkeypatch):
        import victor.tools.output_condenser as oc

        yaml_path = tmp_path / ".victor" / "output_filters.yaml"
        self._write_yaml(
            yaml_path,
            "filters:\n"
            "  - name: user-make\n"
            '    match_command: "^make\\\\b"\n'
            '    strip_lines: ["^gcc "]\n',
        )
        monkeypatch.setattr(oc, "_user_line_filters", lambda: oc._load_filters_file(yaml_path))
        out = "\n".join(["gcc -c foo.c", "make[1]: Entering directory '/x'", "ld foo.o"] * 15)
        result = condense_shell_output("make", out, "", 0, tee_enabled=False)
        assert result is not None
        assert result.condenser == "user-make"  # user filter matched before builtin
        assert "gcc -c foo.c" not in result.stdout
        assert "ld foo.o" in result.stdout

    def test_invalid_entries_skipped(self, tmp_path):
        import victor.tools.output_condenser as oc

        yaml_path = tmp_path / "output_filters.yaml"
        self._write_yaml(
            yaml_path,
            "filters:\n"
            "  - name: broken\n"
            '    match_command: "([unclosed"\n'
            "  - name: valid\n"
            '    match_command: "^valid\\\\b"\n',
        )
        specs = oc._load_filters_file(yaml_path)
        assert [s.name for s in specs] == ["valid"]

    def test_missing_file_returns_empty(self, tmp_path):
        import victor.tools.output_condenser as oc

        assert oc._load_filters_file(tmp_path / "nope.yaml") == []

    def test_mtime_cache_reloads_on_change(self, tmp_path):
        import os

        import victor.tools.output_condenser as oc

        yaml_path = tmp_path / "output_filters.yaml"
        self._write_yaml(yaml_path, 'filters:\n  - name: one\n    match_command: "^one"\n')
        assert [s.name for s in oc._load_filters_file(yaml_path)] == ["one"]
        self._write_yaml(yaml_path, 'filters:\n  - name: two\n    match_command: "^two"\n')
        os.utime(yaml_path, (1e9, 1e9))  # force distinct mtime
        assert [s.name for s in oc._load_filters_file(yaml_path)] == ["two"]


class TestSavingsAccounting:
    def test_result_reports_savings(self):
        r = CondensationResult(
            stdout="a", stderr="", condenser="x", original_chars=1000, condensed_chars=100
        )
        assert r.savings_pct == pytest.approx(90.0)

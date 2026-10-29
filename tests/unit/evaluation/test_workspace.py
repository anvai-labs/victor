# Copyright 2025 Vijaykumar Singh
# SPDX-License-Identifier: Apache-2.0
"""Tests for evaluation workspace abstractions."""

from pathlib import Path

from victor.evaluation.workspace import (
    SWEBenchWorkspace,
    SimpleWorkspace,
    WorkspaceProtocol,
    create_workspace,
)


class TestSWEBenchWorkspace:
    """Tests for SWE-bench workspace implementation."""

    def test_repo_dir_returns_correct_subdirectory(self, tmp_path: Path) -> None:
        """Verify repo_dir returns workspace/repo path."""
        ws = SWEBenchWorkspace(_root=tmp_path)
        assert ws.repo_dir == tmp_path / "repo"

    def test_code_dir_is_alias_for_repo_dir(self, tmp_path: Path) -> None:
        """Verify code_dir and repo_dir return the same path."""
        ws = SWEBenchWorkspace(_root=tmp_path)
        assert ws.code_dir == ws.repo_dir

    def test_root_returns_workspace_root(self, tmp_path: Path) -> None:
        """Verify root returns the workspace root directory."""
        ws = SWEBenchWorkspace(_root=tmp_path)
        assert ws.root == tmp_path

    def test_task_info_path(self, tmp_path: Path) -> None:
        """Verify task_info returns correct path."""
        ws = SWEBenchWorkspace(_root=tmp_path)
        assert ws.task_info == tmp_path / "task_info.json"

    def test_test_file_path(self, tmp_path: Path) -> None:
        """Verify test_file returns correct path."""
        ws = SWEBenchWorkspace(_root=tmp_path)
        assert ws.test_file == tmp_path / "test_verification.py"

    def test_exists_returns_false_when_empty(self, tmp_path: Path) -> None:
        """Verify exists() returns False when repo not created."""
        ws = SWEBenchWorkspace(_root=tmp_path)
        assert not ws.exists()

    def test_exists_returns_true_when_setup(self, tmp_path: Path) -> None:
        """Verify exists() returns True when properly set up."""
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir(parents=True)
        ws = SWEBenchWorkspace(_root=tmp_path)
        assert ws.exists()

    def test_implements_workspace_protocol(self, tmp_path: Path) -> None:
        """Verify SWEBenchWorkspace implements WorkspaceProtocol."""
        ws = SWEBenchWorkspace(_root=tmp_path)
        assert isinstance(ws, WorkspaceProtocol)


class TestSimpleWorkspace:
    """Tests for simple workspace implementation."""

    def test_code_dir_equals_root(self, tmp_path: Path) -> None:
        """Verify code_dir returns root for simple workspaces."""
        ws = SimpleWorkspace(_root=tmp_path)
        assert ws.code_dir == tmp_path

    def test_root_returns_workspace_root(self, tmp_path: Path) -> None:
        """Verify root returns the workspace root."""
        ws = SimpleWorkspace(_root=tmp_path)
        assert ws.root == tmp_path

    def test_solution_file_path(self, tmp_path: Path) -> None:
        """Verify solution_file returns correct path."""
        ws = SimpleWorkspace(_root=tmp_path)
        assert ws.solution_file == tmp_path / "solution.py"

    def test_test_file_path(self, tmp_path: Path) -> None:
        """Verify test_file returns correct path."""
        ws = SimpleWorkspace(_root=tmp_path)
        assert ws.test_file == tmp_path / "test.py"

    def test_exists_returns_true_when_root_exists(self, tmp_path: Path) -> None:
        """Verify exists() returns True when root exists."""
        ws = SimpleWorkspace(_root=tmp_path)
        assert ws.exists()

    def test_implements_workspace_protocol(self, tmp_path: Path) -> None:
        """Verify SimpleWorkspace implements WorkspaceProtocol."""
        ws = SimpleWorkspace(_root=tmp_path)
        assert isinstance(ws, WorkspaceProtocol)


class TestCreateWorkspace:
    """Tests for workspace factory function."""

    def test_creates_swebench_workspace(self, tmp_path: Path) -> None:
        """Verify swe_bench type creates SWEBenchWorkspace."""
        ws = create_workspace(tmp_path, "swe_bench")
        assert isinstance(ws, SWEBenchWorkspace)

    def test_creates_swebench_workspace_uppercase(self, tmp_path: Path) -> None:
        """Verify SWE_BENCH type creates SWEBenchWorkspace."""
        ws = create_workspace(tmp_path, "SWE_BENCH")
        assert isinstance(ws, SWEBenchWorkspace)

    def test_creates_simple_workspace_for_unknown(self, tmp_path: Path) -> None:
        """Verify unknown types create SimpleWorkspace."""
        ws = create_workspace(tmp_path, "human_eval")
        assert isinstance(ws, SimpleWorkspace)

    def test_all_workspaces_have_code_dir(self, tmp_path: Path) -> None:
        """Verify all workspace types have code_dir property."""
        for benchmark_type in ["swe_bench", "human_eval", "custom"]:
            ws = create_workspace(tmp_path, benchmark_type)
            assert hasattr(ws, "code_dir")
            assert isinstance(ws.code_dir, Path)

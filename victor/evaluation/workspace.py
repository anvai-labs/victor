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

"""Workspace abstraction for evaluation pipelines.

Provides a protocol-based abstraction for workspace directory structures,
enabling different evaluation benchmarks to define their own layouts while
maintaining a consistent interface for agents.

SOLID Principles:
- SRP: Each workspace class handles a single benchmark's layout
- DIP: Agents depend on WorkspaceProtocol, not concrete implementations
- OCP: New benchmarks can add new workspace classes without modifying existing code
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class WorkspaceProtocol(Protocol):
    """Abstract interface for evaluation workspace directories.

    Different benchmarks may organize their workspaces differently.
    This protocol abstracts those differences so agents can work
    uniformly across benchmarks.

    Example structures:
        SWE-bench: workspace/repo/, workspace/task_info.json
        HumanEval: workspace/solution.py, workspace/test.py
    """

    @property
    def root(self) -> Path:
        """Root workspace directory."""
        ...

    @property
    def code_dir(self) -> Path:
        """Directory containing the code to work on."""
        ...


@dataclass
class SWEBenchWorkspace:
    """SWE-bench workspace implementation.

    Directory structure:
        workspace/
        ├── repo/               # Cloned repository (agent works here)
        ├── test_verification.py  # Test code from dataset
        └── task_info.json     # Task metadata
    """

    _root: Path

    @property
    def root(self) -> Path:
        """Root workspace directory."""
        return self._root

    @property
    def code_dir(self) -> Path:
        """Directory containing the cloned repository code."""
        return self._root / "repo"

    @property
    def repo_dir(self) -> Path:
        """Alias for code_dir - the cloned repository."""
        return self.code_dir

    @property
    def task_info(self) -> Path:
        """Path to task metadata JSON file."""
        return self._root / "task_info.json"

    @property
    def test_file(self) -> Path:
        """Path to test verification script."""
        return self._root / "test_verification.py"

    def exists(self) -> bool:
        """Check if workspace is properly set up."""
        return self._root.exists() and self.code_dir.exists()


@dataclass
class SimpleWorkspace:
    """Simple workspace for benchmarks without repository structure.

    Used for HumanEval, MBPP, and similar code generation benchmarks
    where code is written directly without a pre-existing repository.

    Directory structure:
        workspace/
        ├── solution.py         # Generated solution
        └── test.py            # Test cases
    """

    _root: Path

    @property
    def root(self) -> Path:
        """Root workspace directory."""
        return self._root

    @property
    def code_dir(self) -> Path:
        """Code directory (same as root for simple workspaces)."""
        return self._root

    @property
    def solution_file(self) -> Path:
        """Path to solution file."""
        return self._root / "solution.py"

    @property
    def test_file(self) -> Path:
        """Path to test file."""
        return self._root / "test.py"

    def exists(self) -> bool:
        """Check if workspace is properly set up."""
        return self._root.exists()


def create_workspace(workspace_root: Path, benchmark_type: str) -> WorkspaceProtocol:
    """Factory function to create appropriate workspace for benchmark type.

    Args:
        workspace_root: Root directory for the workspace
        benchmark_type: Type of benchmark (e.g., "swe_bench", "human_eval")

    Returns:
        Appropriate workspace implementation
    """
    if benchmark_type in ("swe_bench", "SWE_BENCH"):
        return SWEBenchWorkspace(_root=workspace_root)
    else:
        return SimpleWorkspace(_root=workspace_root)

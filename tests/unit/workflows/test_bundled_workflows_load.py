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

"""Guard: every bundled YAML workflow file loads (and compiles) strictly.

Directory loaders and workflow providers log and skip a file that fails to
parse, so a broken bundled workflow (for example an unsupported node type)
can ship unnoticed. This guard calls the per-file loader directly, so any
parse failure fails the test for that file.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from victor.core.yaml_utils import safe_load
from victor.workflows.unified_compiler import UnifiedWorkflowCompiler
from victor.workflows.yaml_loader import (
    YAMLWorkflowConfig,
    _parse_node,
    load_workflow_from_file,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

# Bundled workflow locations: vertical packages and the core package.
_GLOBS = (
    "verticals/*/victor_*/workflows/**/*.yaml",
    "verticals/*/victor_*/workflows/**/*.yml",
    "victor/**/workflows/**/*.yaml",
    "victor/**/workflows/**/*.yml",
)

# Shared node libraries: no ``workflows`` key, only a top-level ``nodes`` list
# that workflows pull in with ``$ref``. Each node is parsed on its own instead.
NODE_LIBRARIES = {"victor/workflows/common/hitl_gates.yaml"}

# Lower bound so a moved directory or broken glob cannot make the guard pass
# vacuously (27 files at the time of writing).
MIN_BUNDLED_FILES = 25


def _tracked_files() -> set[str] | None:
    """Git-tracked paths, so local untracked YAML cannot change the file set."""
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z", "--", "verticals", "victor"],
            cwd=REPO_ROOT,
            capture_output=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None  # not a git checkout: fall back to the filesystem view
    return {p for p in out.decode().split("\0") if p}


def _bundled_files() -> list[str]:
    tracked = _tracked_files()
    found: set[str] = set()
    for pattern in _GLOBS:
        for path in REPO_ROOT.glob(pattern):
            rel = path.relative_to(REPO_ROOT).as_posix()
            if "/tests/" in f"/{rel}" or (tracked is not None and rel not in tracked):
                continue
            found.add(rel)
    return sorted(found)


BUNDLED_FILES = _bundled_files()


def test_bundled_workflow_discovery_is_not_vacuous() -> None:
    assert len(BUNDLED_FILES) >= MIN_BUNDLED_FILES, BUNDLED_FILES
    assert "verticals/victor-coding/victor_coding/workflows/team_node_example.yaml" in (
        BUNDLED_FILES
    )
    for library in NODE_LIBRARIES:
        assert library in BUNDLED_FILES, f"stale NODE_LIBRARIES entry: {library}"


@pytest.mark.parametrize("rel_path", BUNDLED_FILES)
def test_bundled_workflow_file_loads(rel_path: str) -> None:
    path = REPO_ROOT / rel_path
    if rel_path in NODE_LIBRARIES:
        data = safe_load(path.read_text())
        assert "workflows" not in data, f"{rel_path} is no longer a node library"
        nodes = data["nodes"]
        assert nodes, f"{rel_path} defines no nodes"
        config = YAMLWorkflowConfig(base_dir=path.parent)
        for node_data in nodes:
            _parse_node(node_data, config)
        return

    loaded = load_workflow_from_file(path)
    workflows = loaded if isinstance(loaded, dict) else {loaded.name: loaded}
    assert workflows, f"{rel_path} defines no workflows"
    compiler = UnifiedWorkflowCompiler(enable_caching=False)
    for name, workflow in workflows.items():
        assert workflow.nodes, f"{rel_path}: workflow '{name}' has no nodes"
        assert compiler.compile_definition(workflow) is not None

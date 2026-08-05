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

"""Import boundary guard tests for stage-transition runtime migration.

Migration Note (2026-05-04):
- Stage-transition batching is an active production runtime seam.
- Canonical ownership belongs in ``victor.agent.services`` rather than
  ``victor.agent.coordinators`` because this is effectful runtime behavior,
  not a state-passed policy seam.
- Legacy coordinator modules remain as compatibility re-export paths only.

These tests prevent regression by ensuring no new internal production code
imports stage-transition runtime helpers from ``victor.agent.coordinators``.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

LEGACY_STAGE_TRANSITION_EXPORTS = {
    "StageTransitionCoordinator",
    "TransitionDecision",
    "TransitionResult",
    "TurnContext",
    "TransitionStrategyProtocol",
    "HeuristicOnlyTransitionStrategy",
    "EdgeModelTransitionStrategy",
    "HybridTransitionStrategy",
    "create_transition_strategy",
}

LEGACY_STAGE_TRANSITION_MODULES = (
    "victor.agent.coordinators.stage_transition_coordinator",
    "victor.agent.coordinators.transition_strategies",
)


def _get_python_files(directory: Path) -> list[Path]:
    """Get all Python files in a directory recursively."""
    files = []
    for root, dirs, filenames in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in ["__pycache__", "test", "tests"]]

        for filename in filenames:
            if filename.endswith(".py"):
                files.append(Path(root) / filename)
    return files


def _check_stage_transition_import(file_path: Path) -> bool:
    """Check if a file imports stage-transition runtime from coordinator paths."""
    try:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(file_path))
    except (SyntaxError, UnicodeDecodeError):
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == "victor.agent.coordinators":
                if any(alias.name in LEGACY_STAGE_TRANSITION_EXPORTS for alias in node.names):
                    return True

            if node.module and any(
                node.module == package or node.module.startswith(f"{package}.")
                for package in LEGACY_STAGE_TRANSITION_MODULES
            ):
                return True

        elif isinstance(node, ast.Import):
            for alias in node.names:
                if any(
                    alias.name == package or alias.name.startswith(f"{package}.")
                    for package in LEGACY_STAGE_TRANSITION_MODULES
                ):
                    return True

    return False


def test_internal_code_does_not_import_stage_transition_runtime_from_coordinators() -> None:
    """Internal production code should use service-owned stage-transition modules."""
    repo_root = Path(__file__).resolve().parents[3]
    victor_dir = repo_root / "victor"
    violations = []

    for py_file in _get_python_files(victor_dir):
        relative_path = py_file.relative_to(repo_root)
        if _check_stage_transition_import(py_file):
            violations.append(str(relative_path))

    assert not violations, (
        f"Found {len(violations)} file(s) importing stage-transition runtime from "
        f"deprecated coordinator paths. Use victor.agent.services.stage_transition_* instead.\n"
        f"Violations: {violations}"
    )


def test_canonical_stage_transition_service_imports_work() -> None:
    """Canonical service-owned stage-transition imports should be available."""
    from victor.agent.services.stage_transition_runtime import (
        StageTransitionCoordinator,
        TransitionDecision,
        TransitionResult,
        TurnContext,
    )
    from victor.agent.services.stage_transition_strategies import (
        EdgeModelTransitionStrategy,
        HeuristicOnlyTransitionStrategy,
        HybridTransitionStrategy,
        TransitionStrategyProtocol,
        create_transition_strategy,
    )

    assert StageTransitionCoordinator is not None
    assert TransitionDecision is not None
    assert TransitionResult is not None
    assert TurnContext is not None
    assert EdgeModelTransitionStrategy is not None
    assert HeuristicOnlyTransitionStrategy is not None
    assert HybridTransitionStrategy is not None
    assert TransitionStrategyProtocol is not None
    assert create_transition_strategy is not None


def test_legacy_stage_transition_modules_reexport_service_runtime() -> None:
    """Legacy coordinator modules should remain compatibility re-export paths."""
    from victor.agent.coordinators.stage_transition_coordinator import (
        StageTransitionCoordinator as legacy_stage_transition_coordinator,
    )
    from victor.agent.coordinators.transition_strategies import (
        HybridTransitionStrategy as legacy_hybrid_transition_strategy,
    )
    from victor.agent.services.stage_transition_runtime import (
        StageTransitionCoordinator as canonical_stage_transition_coordinator,
    )
    from victor.agent.services.stage_transition_strategies import (
        HybridTransitionStrategy as canonical_hybrid_transition_strategy,
    )

    assert legacy_stage_transition_coordinator is canonical_stage_transition_coordinator
    assert legacy_hybrid_transition_strategy is canonical_hybrid_transition_strategy

# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Guardrails for state-passed exploration runtime ownership.

Internal production runtime should prefer the shared
``ExplorationStatePassedCoordinator`` surface for exploration decisions.
The direct ``create_exploration_coordinator()`` helper remains available for
public factories and compatibility helpers, but service/runtime code should not
import it directly.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path("victor")
ALLOWED_IMPORTERS = {
    Path("victor/agent/coordinators/coordinator_factory.py"),
    Path("victor/agent/coordinators/factory_support.py"),
    Path("victor/agent/factory/coordination_builders.py"),
}
LEGACY_EXPLORATION_EXPORTS = frozenset({"ExplorationCoordinator", "create_exploration_coordinator"})


def test_internal_code_does_not_import_direct_exploration_helper_from_coordinator_package() -> None:
    violations: list[str] = []

    for path in sorted(ROOT.rglob("*.py")):
        if path in ALLOWED_IMPORTERS:
            continue

        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))

        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue

            if node.module == "victor.agent.coordinators":
                imported_names = {alias.name for alias in node.names}
                bad_names = sorted(imported_names & LEGACY_EXPLORATION_EXPORTS)
                if bad_names or "*" in imported_names:
                    violations.append(
                        f"{path}:{node.lineno} imports coordinator-owned exploration runtime "
                        f"exports: {', '.join(bad_names or ['*'])}"
                    )

            if node.module == "victor.agent.coordinators.factory_support":
                imported_names = {alias.name for alias in node.names}
                if "create_exploration_coordinator" in imported_names or "*" in imported_names:
                    violations.append(
                        f"{path}:{node.lineno} imports create_exploration_coordinator() "
                        "from victor.agent.coordinators.factory_support"
                    )

    assert not violations, "\n".join(violations)


def test_canonical_state_passed_exploration_imports_work() -> None:
    from victor.agent.coordinators.exploration_state_passed import (
        ExplorationStatePassedCoordinator,
    )

    assert ExplorationStatePassedCoordinator is not None

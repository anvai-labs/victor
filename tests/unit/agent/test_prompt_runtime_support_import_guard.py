# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Guardrails for removed prompt compatibility surfaces."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path("victor")
ALLOWED_FILES: set[Path] = set()


def _find_prompt_runtime_import_violations(path: Path, source: str) -> list[str]:
    tree = ast.parse(source, filename=str(path))
    violations: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == "victor.agent.services.prompt_runtime_support":
                violations.append(f"{path}:{node.lineno} imports PromptRuntimeSupport directly")

            if node.module == "victor.agent.services.system_prompt_runtime" and any(
                alias.name == "SystemPromptCoordinator" for alias in node.names
            ):
                violations.append(f"{path}:{node.lineno} imports SystemPromptCoordinator directly")

            if node.module == "victor.agent.services":
                for alias in node.names:
                    if alias.name == "SystemPromptCoordinator":
                        violations.append(
                            f"{path}:{node.lineno} imports SystemPromptCoordinator from services package"
                        )
                    if alias.name == "PromptRuntimeSupport":
                        violations.append(
                            f"{path}:{node.lineno} imports PromptRuntimeSupport from services package"
                        )

            if node.module in {
                "victor.agent.coordinators.factory_support",
                "victor.agent.factory.coordination_builders",
            }:
                disallowed = {
                    alias.name
                    for alias in node.names
                    if alias.name
                    in {
                        "create_prompt_runtime_support",
                        "create_system_prompt_coordinator",
                    }
                }
                for name in sorted(disallowed):
                    violations.append(f"{path}:{node.lineno} imports {name}")

            if node.module == "victor.agent.coordinators" and any(
                alias.name == "SystemPromptCoordinator" for alias in node.names
            ):
                violations.append(
                    f"{path}:{node.lineno} imports SystemPromptCoordinator from coordinators"
                )

        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "victor.agent.services.prompt_runtime_support":
                    violations.append(
                        f"{path}:{node.lineno} imports prompt_runtime_support module directly"
                    )
                if alias.name == "victor.agent.services.system_prompt_runtime":
                    violations.append(
                        f"{path}:{node.lineno} imports system_prompt_runtime module directly"
                    )

    return violations


def test_guard_catches_services_reexport_imports() -> None:
    violations = _find_prompt_runtime_import_violations(
        Path("victor/agent/example.py"),
        "from victor.agent.services import SystemPromptCoordinator\n",
    )

    assert violations == [
        "victor/agent/example.py:1 imports SystemPromptCoordinator from services package"
    ]


def test_guard_catches_plain_module_imports() -> None:
    violations = _find_prompt_runtime_import_violations(
        Path("victor/agent/example.py"),
        "import victor.agent.services.prompt_runtime_support as prompt_support\n",
    )

    assert violations == [
        "victor/agent/example.py:1 imports prompt_runtime_support module directly"
    ]


def test_guard_catches_removed_coordinator_package_imports() -> None:
    violations = _find_prompt_runtime_import_violations(
        Path("victor/agent/example.py"),
        "from victor.agent.coordinators import SystemPromptCoordinator\n",
    )

    assert violations == [
        "victor/agent/example.py:1 imports SystemPromptCoordinator from coordinators"
    ]


def test_internal_runtime_code_does_not_import_prompt_runtime_support() -> None:
    violations: list[str] = []

    for path in sorted(ROOT.rglob("*.py")):
        if path in ALLOWED_FILES:
            continue

        source = path.read_text(encoding="utf-8")
        violations.extend(_find_prompt_runtime_import_violations(path, source))

    assert not violations, "\n".join(violations)

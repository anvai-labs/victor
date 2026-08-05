# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Guardrails for deleted state compatibility shims."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path("victor")
REMOVED_STATE_MODULES = frozenset(
    {
        "victor.agent.state_coordinator",
        "victor.agent.services.state_compat",
    }
)


def test_deleted_state_shim_files_do_not_exist() -> None:
    assert not Path("victor/agent/state_coordinator.py").exists()
    assert not Path("victor/agent/services/state_compat.py").exists()


def test_internal_code_does_not_import_deleted_state_shim_modules() -> None:
    violations: list[str] = []

    for path in sorted(ROOT.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in REMOVED_STATE_MODULES:
                violations.append(f"{path}:{node.lineno} imports removed module {node.module}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in REMOVED_STATE_MODULES:
                        violations.append(
                            f"{path}:{node.lineno} imports removed module {alias.name}"
                        )

    assert not violations, "\n".join(violations)

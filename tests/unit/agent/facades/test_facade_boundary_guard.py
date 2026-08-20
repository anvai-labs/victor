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

"""AST guards for facade ownership boundaries.

These facades are intentionally limited to constructor wiring, property access,
and compatibility shims/warnings. They must not grow new behavior-owning
methods.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
# Only OrchestrationFacade remains: the seven per-domain facades were removed as
# dead parallel views (zero production readers).
FACADE_SPECS = (("victor/agent/facades/orchestration_facade.py", "OrchestrationFacade"),)
ALLOWED_PRIVATE_HELPERS: dict = {}


def _load_class_node(relative_path: str, class_name: str) -> ast.ClassDef:
    file_path = REPO_ROOT / relative_path
    tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node

    raise AssertionError(f"Could not find class {class_name} in {relative_path}")


def _is_property_accessor(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Name) and decorator.id == "property":
            return True
        if isinstance(decorator, ast.Attribute) and decorator.attr in {
            "setter",
            "deleter",
        }:
            return True
    return False


def test_facades_expose_only_constructor_and_property_accessors() -> None:
    """Prevent facades from growing public behavior-owning methods."""
    violations: list[str] = []

    for relative_path, class_name in FACADE_SPECS:
        class_node = _load_class_node(relative_path, class_name)
        allowed_private_helpers = ALLOWED_PRIVATE_HELPERS.get((relative_path, class_name), set())
        for node in class_node.body:
            if isinstance(node, ast.AsyncFunctionDef):
                violations.append(
                    f"{relative_path}:{node.lineno} {class_name}.{node.name} must not be async"
                )
                continue

            if not isinstance(node, ast.FunctionDef):
                continue

            if node.name == "__init__":
                continue

            if node.name in allowed_private_helpers:
                continue

            if _is_property_accessor(node):
                continue

            violations.append(
                f"{relative_path}:{node.lineno} {class_name}.{node.name} must stay a "
                "property/compatibility accessor, not a behavior method"
            )

    assert not violations, "Facade boundary violations:\n" + "\n".join(violations)

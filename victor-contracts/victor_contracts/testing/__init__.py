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

"""SDK contract test utilities for external verticals.

Drop-in test helpers that validate a vertical class implements
the SDK contract correctly and respects import boundaries.

Usage in your vertical's test suite::

    from victor_contracts.testing import assert_valid_vertical, assert_import_boundaries

    def test_contract():
        from my_vertical.assistant import MyVerticalAssistant
        assert_valid_vertical(MyVerticalAssistant)

    def test_boundaries():
        violations = assert_import_boundaries("my_vertical")
        assert not violations, f"Import violations: {violations}"
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, List, Optional, Tuple, Type

# Default forbidden import prefixes for external verticals
_DEFAULT_FORBIDDEN_PREFIXES = (
    "victor.agent.",
    "victor.core.container",
    "victor.core.bootstrap",
    "victor.evaluation.",
    "victor.storage.",
    "victor.config.settings",
    "victor.config.api_keys",
)

# Allowed import prefixes (safe for external verticals)
_ALLOWED_PREFIXES = (
    "victor.framework.",
    "victor.core.verticals.",
    "victor.core.vertical_types",
    "victor.tools.base",
    "victor.tools.registry",
    "victor.security.",
    "victor_contracts.",
)


def assert_valid_vertical(vertical_cls: Type[Any]) -> None:
    """Validate that a vertical class implements the SDK contract.

    Checks:
    - Required methods exist: get_name, get_description, get_tools, get_system_prompt
    - get_tools returns a non-empty list of strings
    - get_system_prompt returns a non-empty string
    - If _victor_manifest is present, validates manifest fields

    Args:
        vertical_cls: The vertical class to validate.

    Raises:
        AssertionError: If the contract is violated.
    """
    # Check required methods exist
    required_methods = ["get_name", "get_description", "get_tools", "get_system_prompt"]
    for method_name in required_methods:
        method = getattr(vertical_cls, method_name, None)
        assert (
            method is not None
        ), f"Vertical {vertical_cls.__name__} missing required method: {method_name}"
        assert callable(method), f"Vertical {vertical_cls.__name__}.{method_name} is not callable"

    # Validate get_tools returns list of strings
    tools = vertical_cls.get_tools()
    assert isinstance(tools, list), (
        f"Vertical {vertical_cls.__name__}.get_tools() must return list, "
        f"got {type(tools).__name__}"
    )
    assert len(tools) > 0, f"Vertical {vertical_cls.__name__}.get_tools() returned empty list"
    for tool in tools:
        assert isinstance(tool, str), (
            f"Vertical {vertical_cls.__name__}.get_tools() items must be str, "
            f"got {type(tool).__name__}: {tool!r}"
        )

    # Validate get_system_prompt returns non-empty string
    prompt = vertical_cls.get_system_prompt()
    assert isinstance(prompt, str), (
        f"Vertical {vertical_cls.__name__}.get_system_prompt() must return str, "
        f"got {type(prompt).__name__}"
    )
    assert (
        len(prompt.strip()) > 0
    ), f"Vertical {vertical_cls.__name__}.get_system_prompt() returned empty string"

    # Validate manifest if present
    manifest = getattr(vertical_cls, "_victor_manifest", None)
    if manifest is not None:
        assert hasattr(manifest, "name"), "Manifest missing 'name' field"
        assert hasattr(manifest, "version"), "Manifest missing 'version' field"
        assert hasattr(manifest, "provides"), "Manifest missing 'provides' field"
        assert (
            isinstance(manifest.name, str) and manifest.name
        ), "Manifest.name must be a non-empty string"


def assert_import_boundaries(
    package_path: str,
    forbidden_prefixes: Optional[Tuple[str, ...]] = None,
) -> List[str]:
    """Scan a package for forbidden imports from victor internals.

    Walks all .py files in the given package directory and checks
    ``import`` and ``from ... import`` statements against a list of
    forbidden module prefixes.

    Args:
        package_path: Path to the package directory to scan (e.g., "victor_coding").
        forbidden_prefixes: Tuple of module prefixes that are forbidden.
            Defaults to internal victor modules that external verticals
            should not import from.

    Returns:
        List of violation strings in the format "file:line: import statement".
        Empty list means no violations found.
    """
    if forbidden_prefixes is None:
        forbidden_prefixes = _DEFAULT_FORBIDDEN_PREFIXES

    violations: List[str] = []
    package_dir = Path(package_path)

    if not package_dir.is_dir():
        return [f"Package directory not found: {package_path}"]

    for py_file in sorted(package_dir.rglob("*.py")):
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file))
        except (SyntaxError, UnicodeDecodeError):
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if any(alias.name.startswith(p) for p in forbidden_prefixes):
                        violations.append(f"{py_file}:{node.lineno}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module and any(node.module.startswith(p) for p in forbidden_prefixes):
                    violations.append(f"{py_file}:{node.lineno}: from {node.module} import ...")
            elif isinstance(node, ast.Call):
                module_name = _extract_dynamic_import_module(node)
                if module_name and any(module_name.startswith(p) for p in forbidden_prefixes):
                    violations.append(f"{py_file}:{node.lineno}: dynamic import {module_name}")

    return violations


def _extract_dynamic_import_module(node: ast.Call) -> str | None:
    """Return the module path from supported dynamic import calls."""

    func = node.func
    is_dynamic_import = False
    if isinstance(func, ast.Attribute) and func.attr == "import_module":
        is_dynamic_import = True
    elif isinstance(func, ast.Name) and func.id in {"import_module", "__import__"}:
        is_dynamic_import = True

    if not is_dynamic_import or not node.args:
        return None

    first_arg = node.args[0]
    if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
        return first_arg.value
    return None


from victor_contracts.testing.fixtures import MockPluginContext  # noqa: E402

__all__ = [
    "assert_valid_vertical",
    "assert_import_boundaries",
    "MockPluginContext",
]

"""Contract boundary enforcement for external vertical definition layers.

Enforces that vertical DEFINITION files (assistant.py, plugin.py) have
no MODULE-LEVEL imports from victor.framework. Imports inside function
bodies, methods, TYPE_CHECKING blocks, and __getattr__ are allowed
(they're lazy/deferred and don't execute on module import).

Architecture: Plugin (bootstrap) -> Vertical (config) -> Extension (runtime)
See CLAUDE.md "Plugin -> Vertical -> Extension Architecture"
"""

import ast
import importlib
from pathlib import Path
from typing import List, Tuple

import pytest

from victor_contracts.testing.boundaries import DEFINITION_LAYER_FORBIDDEN_PREFIXES

# Sourced from the tiered manifest (co-design review item 22b) rather than a
# locally maintained list, so this stays in sync with the definition-layer
# tier every other consumer of victor_contracts.testing.boundaries enforces.
BANNED_PREFIXES = DEFINITION_LAYER_FORBIDDEN_PREFIXES

DEFINITION_FILES = {"assistant.py", "plugin.py"}

VERTICALS = [
    "victor_invest",
    "victor_coding",
    "victor_research",
    "victor_devops",
    "victor_rag",
    "victor_dataanalysis",
]


def _find_source(pkg: str) -> Path:
    try:
        mod = importlib.import_module(pkg)
        return Path(mod.__path__[0]) if hasattr(mod, "__path__") else None
    except ImportError:
        return None


def _get_module_level_imports(source: str) -> List[Tuple[int, str]]:
    """Extract only MODULE-LEVEL imports (not inside functions/methods/if blocks)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    violations = []
    for node in ast.iter_child_nodes(tree):
        # Only check top-level statements
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith(BANNED_PREFIXES):
                violations.append((node.lineno, node.module))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(BANNED_PREFIXES):
                    violations.append((node.lineno, alias.name))
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            module_name = _extract_dynamic_import_module(node.value)
            if module_name and module_name.startswith(BANNED_PREFIXES):
                violations.append((node.lineno, module_name))
        # Check inside TYPE_CHECKING — these are OK (type-only)
        # Check inside if/try blocks at module level — still module-level
        elif isinstance(node, ast.If):
            # Skip TYPE_CHECKING blocks
            test = node.test
            if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
                continue
            if isinstance(test, ast.Attribute) and getattr(test, "attr", "") == "TYPE_CHECKING":
                continue
            # Other if blocks at module level: check their imports
            for child in ast.walk(node):
                if isinstance(child, ast.ImportFrom) and child.module:
                    if child.module.startswith(BANNED_PREFIXES):
                        violations.append((child.lineno, child.module))
                elif isinstance(child, ast.Call):
                    module_name = _extract_dynamic_import_module(child)
                    if module_name and module_name.startswith(BANNED_PREFIXES):
                        violations.append((child.lineno, module_name))
    return violations


def _extract_dynamic_import_module(node: ast.Call) -> str | None:
    """Return module path for supported dynamic import calls."""

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


def _scan_definition_imports(source_dir: Path) -> List[Tuple[str, int, str]]:
    """Scan definition-layer files for module-level banned imports."""
    results = []
    for py_file in source_dir.rglob("*.py"):
        rel = str(py_file.relative_to(source_dir))
        if Path(rel).name not in DEFINITION_FILES:
            continue
        source = py_file.read_text(encoding="utf-8")
        for lineno, module in _get_module_level_imports(source):
            results.append((rel, lineno, module))
    return results


def test_module_level_import_scanner_flags_dynamic_forbidden_imports() -> None:
    source = "\n".join(
        [
            "import importlib",
            "importlib.import_module('victor.agent.orchestrator')",
            "__import__('victor.core.container')",
            "def lazy():",
            "    importlib.import_module('victor.agent.runtime')",
        ]
    )

    violations = _get_module_level_imports(source)

    assert violations == [
        (2, "victor.agent.orchestrator"),
        (3, "victor.core.container"),
    ]


def test_banned_prefixes_has_not_shrunk() -> None:
    """Regression guard (co-design review item 22b): BANNED_PREFIXES is sourced
    from victor_contracts.testing.boundaries.DEFINITION_LAYER_FORBIDDEN_PREFIXES.
    Pin the exact expected set so a future edit to the shared manifest can't
    silently loosen this guard without a visible, deliberate test change here.
    """
    assert set(BANNED_PREFIXES) == {
        "victor.agent",
        "victor.core",
        "victor.framework",
        "victor.providers",
        "victor.security",
        "victor.storage",
        "victor.tools",
        "victor.workflows",
        "victor.evaluation",
        "victor.observability",
    }


@pytest.mark.parametrize("pkg", VERTICALS)
def test_vertical_definition_layer_sdk_only(pkg: str):
    """Definition files (assistant.py, plugin.py) must not have module-level framework imports.

    Imports inside function bodies, TYPE_CHECKING blocks, and __getattr__
    are allowed (deferred/lazy).
    """
    src = _find_source(pkg)
    if src is None:
        pytest.skip(f"{pkg} not installed")

    violations = _scan_definition_imports(src)
    assert not violations, (
        f"{pkg} definition-layer files have module-level framework imports "
        f"(should use victor_contracts or defer to function bodies):\n"
        + "\n".join(f"  {f}:{line}: {mod}" for f, line, mod in violations)
    )

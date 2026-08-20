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

"""Import boundary tests for external vertical packages.

Validates that external verticals (victor-coding, victor-research, etc.)
import only from the stable public API surface:
  - victor.framework.*
  - victor.tools.*
  - victor.security.*
  - victor.core.verticals.*
  - victor_contracts.*

And do NOT import from internal modules:
  - victor.agent.*  (orchestrator internals)
  - victor.core.container.*  (DI internals)
  - victor.evaluation.*  (evaluation internals)

This test scans actual installed packages to detect import violations.
"""

import ast
import importlib
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pytest

# Allowed import prefixes for external verticals
ALLOWED_PREFIXES = frozenset(
    {
        "victor.framework",
        "victor.tools",
        "victor.security",
        "victor.core.verticals",
        "victor.core.vertical_types",
        "victor.core.tool_dependency_loader",
        "victor.core.tool_types",
        "victor_contracts",
    }
)

# Forbidden import prefixes (internal APIs)
FORBIDDEN_PREFIXES = frozenset(
    {
        "victor.agent.",
        "victor.core.container",
        "victor.core.mode_config",
        "victor.evaluation.",
        "victor.workflows.executor",
        "victor.workflows.definition",
        "victor.processing.",
        "victor.protocols.",
    }
)

# Known violations to track migration progress (baseline).
# All external verticals have been migrated to use victor.framework.extensions,
# victor.framework.processing, and victor.framework.lsp re-export modules.
KNOWN_VIOLATIONS: Dict[str, Set[str]] = {}


def _find_package_root(package_name: str) -> Path | None:
    """Find the root directory of an installed package."""
    try:
        spec = importlib.util.find_spec(package_name)
        if spec and spec.origin:
            return Path(spec.origin).parent
        if spec and spec.submodule_search_locations:
            locations = list(spec.submodule_search_locations)
            if locations:
                return Path(locations[0])
    except (ModuleNotFoundError, ValueError):
        pass
    return None


def _is_inrepo_vertical(package_root: Path) -> bool:
    """True iff the installed package is this monorepo's in-repo vertical copy.

    An external install (e.g. the archived standalone ``victor-coding`` repo) can
    shadow the in-repo ``verticals/<pkg>/`` editable install on a dev machine.
    This guard validates the in-repo vertical that actually ships; when a
    different install is shadowing it we skip rather than flag its out-of-scope
    imports. CI installs the in-repo vertical, so the guard still runs there.
    """
    repo_root = Path(__file__).resolve().parents[4]
    try:
        package_root.resolve().relative_to(repo_root / "verticals")
        return True
    except ValueError:
        return False


def _extract_victor_imports(filepath: Path) -> List[str]:
    """Extract all victor.* imports from a Python file using AST parsing."""
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filepath))
    except (SyntaxError, UnicodeDecodeError):
        return []

    imports: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("victor.") or alias.name.startswith("victor_"):
                    imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module and (
                node.module.startswith("victor.") or node.module.startswith("victor_")
            ):
                imports.append(node.module)
    return imports


def _check_import_against_boundaries(
    imp: str,
) -> Tuple[bool, str]:
    """Check if an import violates boundaries.

    Returns:
        (is_violation, reason)
    """
    for forbidden in FORBIDDEN_PREFIXES:
        if imp.startswith(forbidden):
            return True, f"imports forbidden internal module: {imp}"
    return False, ""


def _scan_package_imports(
    package_root: Path,
) -> Dict[str, List[Tuple[str, str]]]:
    """Scan all .py files in a package for boundary violations.

    Returns:
        Dict of filepath -> list of (import_path, reason) violations
    """
    violations: Dict[str, List[Tuple[str, str]]] = {}

    for py_file in package_root.rglob("*.py"):
        # Skip test files and __pycache__
        rel = py_file.relative_to(package_root)
        if "__pycache__" in str(rel):
            continue

        imports = _extract_victor_imports(py_file)
        file_violations: List[Tuple[str, str]] = []

        for imp in imports:
            is_violation, reason = _check_import_against_boundaries(imp)
            if is_violation:
                file_violations.append((imp, reason))

        if file_violations:
            violations[str(rel)] = file_violations

    return violations


# External vertical packages to check if installed
EXTERNAL_VERTICALS = ["victor_coding", "victor_research", "victor_invest"]


@pytest.mark.parametrize("package_name", EXTERNAL_VERTICALS)
def test_external_vertical_import_boundaries(package_name: str):
    """Verify external verticals only import from stable public API surface."""
    package_root = _find_package_root(package_name)
    if package_root is None:
        pytest.skip(f"{package_name} is not installed")
    if not _is_inrepo_vertical(package_root):
        pytest.skip(
            f"installed {package_name} is not the in-repo vertical "
            f"(at {package_root}) — an external/standalone install is shadowing it; "
            f"CI validates the in-repo vertical."
        )

    violations = _scan_package_imports(package_root)
    known = KNOWN_VIOLATIONS.get(package_name, set())

    # Separate known vs new violations
    new_violations: Dict[str, List[Tuple[str, str]]] = {}
    for filepath, file_violations in violations.items():
        new_in_file = [
            (imp, reason)
            for imp, reason in file_violations
            if not any(imp.startswith(k) for k in known)
        ]
        if new_in_file:
            new_violations[filepath] = new_in_file

    if new_violations:
        lines = [f"\n{package_name} has NEW import boundary violations:\n"]
        for filepath, file_violations in sorted(new_violations.items()):
            for imp, reason in file_violations:
                lines.append(f"  {filepath}: {reason}")
        lines.append("\nFix: import from victor.framework.extensions instead of internal modules.")
        lines.append("Or add to KNOWN_VIOLATIONS baseline if migration is deferred.")
        pytest.fail("\n".join(lines))


@pytest.mark.parametrize("package_name", EXTERNAL_VERTICALS)
def test_external_vertical_known_violations_not_stale(package_name: str):
    """Ensure KNOWN_VIOLATIONS baseline stays current (no stale entries)."""
    package_root = _find_package_root(package_name)
    if package_root is None:
        pytest.skip(f"{package_name} is not installed")
    if not _is_inrepo_vertical(package_root):
        pytest.skip(
            f"installed {package_name} is not the in-repo vertical "
            f"(at {package_root}) — an external/standalone install is shadowing it; "
            f"CI validates the in-repo vertical."
        )

    violations = _scan_package_imports(package_root)
    known = KNOWN_VIOLATIONS.get(package_name, set())
    if not known:
        return  # No known violations to check

    # Collect all actual violation import prefixes
    actual_violation_prefixes: Set[str] = set()
    for file_violations in violations.values():
        for imp, _ in file_violations:
            actual_violation_prefixes.add(imp)

    stale = {k for k in known if not any(a.startswith(k) for a in actual_violation_prefixes)}
    if stale:
        pytest.fail(
            f"{package_name} has stale KNOWN_VIOLATIONS entries (violations fixed!):\n"
            f"  {stale}\n"
            f"Remove these from the baseline in test_external_vertical_import_boundaries.py"
        )


def test_framework_extensions_module_exports():
    """Verify victor.framework.extensions exports all documented symbols."""
    from victor.framework import extensions

    expected_exports = {
        # Safety coordination
        "SafetyCoordinator",
        "SafetyAction",
        "SafetyCategory",
        "SafetyRule",
        # Conversation coordination
        "ConversationCoordinator",
        "ConversationStats",
        "ConversationTurn",
        "TurnType",
        # Workflow execution
        "WorkflowExecutor",
        "WorkflowContext",
        "ComputeNode",
        "NodeResult",
        "ExecutorNodeStatus",
        "register_compute_handler",
        "get_compute_handler",
        # Workflow definition
        "WorkflowBuilder",
        "WorkflowDefinition",
        "workflow",
        "AgentNode",
        "ConditionNode",
        "ParallelNode",
        # Code correction
        "CodeCorrectionMiddleware",
        "CodeCorrectionConfig",
        "CorrectionResult",
        # Code validation
        "CodeValidationResult",
        "Language",
        # Mode configuration
        "ModeConfigRegistry",
        "ModeDefinition",
        "ModeConfig",
        "RegistryBasedModeConfigProvider",
        # Service container
        "ServiceContainer",
        "ServiceLifetime",
        # Agent specs
        "AgentSpec",
        "AgentCapabilities",
        "AgentConstraints",
        "ModelPreference",
        "OutputFormat",
        "DelegationPolicy",
        # Sub-agents
        "SubAgent",
        "SubAgentConfig",
        "SubAgentResult",
        "SubAgentRole",
        "set_role_tool_provider",
        # Middleware
        "MiddlewareChain",
        "MiddlewareAbortError",
        "create_middleware_chain",
        # Vertical context
        "VerticalContext",
        "create_vertical_context",
        "VerticalContextProtocol",
        "MutableVerticalContextProtocol",
        # Handler registry
        "HandlerRegistry",
        "get_handler_registry",
        "register_global_handler",
        "register_vertical_handlers",
        # Provider access
        "ProviderRegistry",
        # Vertical registration and base types
        "register_vertical",
        "VerticalBase",
        "StageDefinition",
        "VerticalConfig",
        "VerticalExtensions",
        # Tool dependency types
        "ToolDependencyConfig",
        "ToolDependency",
        "create_vertical_tool_dependency_provider",
        # Safety pattern types
        "SafetyExtensionProtocol",
        "SafetyPattern",
        # Promoted protocols
        "PromptContributorProtocol",
        "ModeConfigProviderProtocol",
        "ServiceProviderProtocol",
        "ToolDependencyProviderProtocol",
        "WorkflowProviderProtocol",
        "RLConfigProviderProtocol",
        "TeamSpecProviderProtocol",
        "EnrichmentStrategyProtocol",
        "MiddlewareProtocol",
        # Tool dependency infrastructure
        "BaseToolDependencyProvider",
        "YAMLToolDependencyProvider",
        "load_tool_dependency_yaml",
        "ToolDependencyLoader",
        "TieredToolConfig",
        "TaskTypeHint",
    }

    assert set(extensions.__all__) == expected_exports, (
        f"victor.framework.extensions.__all__ mismatch.\n"
        f"Missing: {expected_exports - set(extensions.__all__)}\n"
        f"Extra: {set(extensions.__all__) - expected_exports}"
    )


def test_framework_extensions_lazy_import():
    """Verify that lazy imports in victor.framework.extensions resolve correctly."""
    # Just test that __getattr__ doesn't raise for known names
    from victor.framework.extensions import __all__ as all_names
    from victor.framework import extensions

    for name in all_names:
        assert hasattr(extensions, name), f"extensions.{name} failed to resolve"


def test_framework_extensions_uses_sdk_vertical_contracts():
    """Public extension-layer vertical symbols should resolve to SDK contracts."""
    from victor.framework import extensions
    from victor_contracts import (
        StageDefinition,
        VerticalBase,
        VerticalConfig,
        VerticalExtensions,
    )
    from victor_contracts import register_vertical

    assert extensions.register_vertical is register_vertical
    assert extensions.VerticalBase is VerticalBase
    assert extensions.StageDefinition is StageDefinition
    assert extensions.VerticalConfig is VerticalConfig
    assert extensions.VerticalExtensions is VerticalExtensions


def test_framework_processing_module_exports():
    """Verify victor.framework.processing exports all documented symbols."""
    from victor.framework import processing

    expected_exports = {
        "OperationType",
        "EditOperation",
        "EditTransaction",
        "FileEditor",
        "get_default_text_chunker",
        "InsertTextFormat",
        "CompletionTriggerKind",
        "CompletionContext",
        "CompletionParams",
        "CompletionItemLabelDetails",
        "CompletionItem",
        "InlineCompletionItem",
        "InlineCompletionParams",
        "CompletionList",
        "InlineCompletionList",
        "CompletionCapabilities",
        "CompletionMetrics",
    }

    assert set(processing.__all__) == expected_exports, (
        f"victor.framework.processing.__all__ mismatch.\n"
        f"Missing: {expected_exports - set(processing.__all__)}\n"
        f"Extra: {set(processing.__all__) - expected_exports}"
    )


def test_framework_processing_lazy_import():
    """Verify that lazy imports in victor.framework.processing resolve correctly."""
    from victor.framework.processing import __all__ as all_names
    from victor.framework import processing

    for name in all_names:
        assert hasattr(processing, name), f"processing.{name} failed to resolve"


def test_framework_lsp_module_exports():
    """Verify victor.framework.lsp exports all documented symbols."""
    from victor.framework import lsp

    expected_exports = {
        "DiagnosticSeverity",
        "CompletionItemKind",
        "SymbolKind",
        "DiagnosticTag",
        "Position",
        "Range",
        "Location",
        "LocationLink",
        "DiagnosticRelatedInformation",
        "Diagnostic",
        "CompletionItem",
        "Hover",
        "DocumentSymbol",
        "SymbolInformation",
        "TextEdit",
        "TextDocumentIdentifier",
        "VersionedTextDocumentIdentifier",
        "TextDocumentEdit",
    }

    assert set(lsp.__all__) == expected_exports, (
        f"victor.framework.lsp.__all__ mismatch.\n"
        f"Missing: {expected_exports - set(lsp.__all__)}\n"
        f"Extra: {set(lsp.__all__) - expected_exports}"
    )


def test_framework_lsp_lazy_import():
    """Verify that lazy imports in victor.framework.lsp resolve correctly."""
    from victor.framework.lsp import __all__ as all_names
    from victor.framework import lsp

    for name in all_names:
        assert hasattr(lsp, name), f"lsp.{name} failed to resolve"

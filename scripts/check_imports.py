#!/usr/bin/env python3
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

"""Import validation script to prevent circular dependencies.

This script enforces architectural boundaries by detecting forbidden import patterns:

1. **Tools should not import from agent/**
   - Tools must remain isolated and reusable
   - Prevents tight coupling between tools and orchestrator
   - Exception: tools/base.py can import from core protocols

2. **Providers should not import from agent/**
   - Providers are pure adapters for LLM APIs
   - Should only depend on base.py and config
   - Exception: providers/base.py can import from core protocols

3. **Config should not import from agent/ or tools/**
   - Configuration is foundational and should have minimal dependencies
   - Exception: config/settings.py can import from config/orchestrator_constants.py

Design Rationale:
-----------------
These rules enforce a clean dependency flow:
    config/ ← providers/ ← tools/ ← agent/ ← ui/

This prevents circular dependencies and makes the codebase more maintainable.

Usage:
------
    python scripts/check_imports.py
    python scripts/check_imports.py --strict  # Fail on warnings too

Exit codes:
    0 - No violations
    1 - Import violations found
    2 - Script error (file not found, etc.)
"""

import argparse
import ast
import sys
from pathlib import Path
from typing import List, Set, Tuple


class ImportViolation:
    """Represents an import rule violation."""

    def __init__(
        self,
        file_path: Path,
        line_number: int,
        imported_module: str,
        rule: str,
        severity: str = "error",
    ):
        self.file_path = file_path
        self.line_number = line_number
        self.imported_module = imported_module
        self.rule = rule
        self.severity = severity

    def __str__(self) -> str:
        return (
            f"{self.severity.upper()}: {self.file_path}:{self.line_number} - "
            f"Imports {self.imported_module} ({self.rule})"
        )


class ImportValidator:
    """Validates imports against architectural rules."""

    # Modules that are allowed to violate rules (with justification)
    EXCEPTIONS = {
        "victor/tools/base.py": {
            "allowed_imports": ["victor.core.protocols"],
            "reason": "BaseTool needs protocol definitions",
        },
        "victor/providers/base.py": {
            "allowed_imports": ["victor.core.protocols"],
            "reason": "BaseProvider needs protocol definitions",
        },
        "victor/config/settings.py": {
            "allowed_imports": [
                "victor.config.orchestrator_constants",
                "victor.config.model_capabilities",
            ],
            "reason": "Settings needs to reference centralized constants",
        },
        # TODO: These are existing violations that should be refactored
        # They should be moved to agent/ or use dependency injection
        "victor/tools/semantic_selector.py": {
            "allowed_imports": [
                "victor.agent.tool_sequence_tracker",
                "victor.agent.debug_logger",
                "victor.agent.unified_classifier",
            ],
            "reason": "TECH DEBT: Should use DI or move to agent/",
        },
        "victor/tools/filesystem.py": {
            "allowed_imports": ["victor.agent.change_tracker"],
            "reason": "TECH DEBT: Should use DI or move change_tracker to core/",
        },
        "victor/tools/file_editor_tool.py": {
            "allowed_imports": ["victor.agent.change_tracker"],
            "reason": "TECH DEBT: Should use DI or move change_tracker to core/",
        },
        "victor/tools/patch_tool.py": {
            "allowed_imports": ["victor.agent.change_tracker"],
            "reason": "TECH DEBT: Should use DI or move change_tracker to core/",
        },
    }

    def __init__(self, project_root: Path, strict: bool = False):
        self.project_root = project_root
        self.strict = strict
        self.violations: List[ImportViolation] = []

    def check_file(self, file_path: Path) -> None:
        """Check a Python file for import violations.

        Args:
            file_path: Path to the Python file
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=str(file_path))
        except SyntaxError as e:
            print(f"WARNING: Syntax error in {file_path}: {e}", file=sys.stderr)
            return
        except Exception as e:
            print(f"WARNING: Could not parse {file_path}: {e}", file=sys.stderr)
            return

        # Get relative path for rule matching
        rel_path = file_path.relative_to(self.project_root)
        rel_path_str = str(rel_path)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self._check_import(file_path, rel_path_str, node.lineno, alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    self._check_import(file_path, rel_path_str, node.lineno, node.module)

    def _check_import(
        self, file_path: Path, rel_path: str, line_number: int, module_name: str
    ) -> None:
        """Check a single import statement.

        Args:
            file_path: Absolute path to the file
            rel_path: Relative path from project root
            line_number: Line number of the import
            module_name: Name of the imported module
        """
        # Check if this file has exceptions
        exception = self.EXCEPTIONS.get(rel_path)
        if exception and any(
            module_name.startswith(allowed) for allowed in exception["allowed_imports"]
        ):
            return  # Allowed exception

        # Rule 1: Tools should not import from agent/
        if rel_path.startswith("victor/tools/") and module_name.startswith("victor.agent"):
            # Exception for tools importing from agent/protocols (which is being moved to core/protocols)
            if module_name.startswith("victor.agent.protocols"):
                # Warning: this should be migrated to victor.core.protocols
                violation = ImportViolation(
                    file_path=file_path,
                    line_number=line_number,
                    imported_module=module_name,
                    rule="Tools should not import from agent/ (migrate to core.protocols)",
                    severity="warning",
                )
                self.violations.append(violation)
            else:
                violation = ImportViolation(
                    file_path=file_path,
                    line_number=line_number,
                    imported_module=module_name,
                    rule="Tools must not import from agent/ - breaks isolation",
                    severity="error",
                )
                self.violations.append(violation)

        # Rule 2: Providers should not import from agent/
        if rel_path.startswith("victor/providers/") and module_name.startswith("victor.agent"):
            violation = ImportViolation(
                file_path=file_path,
                line_number=line_number,
                imported_module=module_name,
                rule="Providers must not import from agent/ - breaks isolation",
                severity="error",
            )
            self.violations.append(violation)

        # Rule 3: Config should not import from agent/ or tools/
        if rel_path.startswith("victor/config/"):
            if module_name.startswith("victor.agent"):
                violation = ImportViolation(
                    file_path=file_path,
                    line_number=line_number,
                    imported_module=module_name,
                    rule="Config must not import from agent/ - config is foundational",
                    severity="error",
                )
                self.violations.append(violation)
            elif module_name.startswith("victor.tools"):
                violation = ImportViolation(
                    file_path=file_path,
                    line_number=line_number,
                    imported_module=module_name,
                    rule="Config must not import from tools/ - config is foundational",
                    severity="error",
                )
                self.violations.append(violation)

        # Rule 4: Core (victor/) must not import from external vertical packages.
        # Use entry points or CapabilityRegistry for capability discovery.
        EXTERNAL_VERTICAL_PACKAGES = (
            "victor_coding",
            "victor_devops",
            "victor_research",
            "victor_rag",
            "victor_dataanalysis",
            "victor_invest",
        )
        if rel_path.startswith("victor/") and any(
            module_name.startswith(pkg) for pkg in EXTERNAL_VERTICAL_PACKAGES
        ):
            violation = ImportViolation(
                file_path=file_path,
                line_number=line_number,
                imported_module=module_name,
                rule="Core must not import from external verticals - use entry points or CapabilityRegistry",
                severity="error",
            )
            self.violations.append(violation)

    def validate_directory(self, directory: Path) -> None:
        """Validate all Python files in a directory.

        Args:
            directory: Directory to scan
        """
        for file_path in directory.rglob("*.py"):
            # Skip __pycache__ and other generated files
            if "__pycache__" in str(file_path):
                continue
            self.check_file(file_path)

    def report(self) -> int:
        """Print validation report and return exit code.

        Returns:
            0 if no violations (or only warnings in non-strict mode), 1 otherwise
        """
        if not self.violations:
            print("✓ No import violations found")
            return 0

        # Separate errors and warnings
        errors = [v for v in self.violations if v.severity == "error"]
        warnings = [v for v in self.violations if v.severity == "warning"]

        # Print errors
        if errors:
            print(f"\n❌ Found {len(errors)} import violation(s):\n")
            for violation in errors:
                print(f"  {violation}")

        # Print warnings
        if warnings:
            print(f"\n⚠️  Found {len(warnings)} import warning(s):\n")
            for violation in warnings:
                print(f"  {violation}")

        # Print summary
        print(f"\nSummary: {len(errors)} errors, {len(warnings)} warnings")

        # In strict mode, warnings also cause failure
        if self.strict and warnings:
            print("\n❌ Failing due to warnings (strict mode enabled)")
            return 1

        return 1 if errors else 0


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success, 1 for violations, 2 for errors)
    """
    parser = argparse.ArgumentParser(
        description="Validate imports to prevent circular dependencies"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors",
    )
    parser.add_argument(
        "--directory",
        type=Path,
        help="Directory to validate (default: victor/)",
    )
    args = parser.parse_args()

    # Find project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Validate victor directory
    victor_dir = project_root / "victor"
    if not victor_dir.exists():
        print(f"ERROR: Victor directory not found at {victor_dir}", file=sys.stderr)
        return 2

    # Run validation
    validator = ImportValidator(project_root, strict=args.strict)

    target_dir = args.directory if args.directory else victor_dir
    if not target_dir.exists():
        print(f"ERROR: Directory not found: {target_dir}", file=sys.stderr)
        return 2

    print(f"Validating imports in {target_dir}...")
    validator.validate_directory(target_dir)

    return validator.report()


if __name__ == "__main__":
    sys.exit(main())

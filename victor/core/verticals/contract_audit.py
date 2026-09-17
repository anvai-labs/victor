"""Static contract auditing for extracted vertical repositories."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Iterable

try:
    import tomllib
except ImportError:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib

# NOTE (co-design review item 22a): victor_contracts.testing.boundaries now
# defines a RUNTIME_LAYER_FORBIDDEN_PREFIXES manifest that is a strict
# superset of this list (it additionally forbids victor.config.settings,
# victor.config.api_keys, victor.evaluation, and victor.storage). Migrating
# this auditor onto that manifest is deferred until the ~28 real violations
# it would surface in verticals/victor-coding and verticals/victor-rag
# (direct victor.config.settings / victor.storage.* imports) are fixed in a
# follow-up PR — landing the fuller list here today would flip the
# currently-green Vertical Contract Boundaries CI check red.
FORBIDDEN_RUNTIME_IMPORT_PREFIXES = (
    "victor.framework",
    "victor.core",
    "victor.security",
    "victor.agent",
    "victor.workflows",
    "victor.providers",
)
ALLOWED_RUNTIME_IMPORT_PREFIXES = (
    # Documented extension surface for extracted plugins during the contract-first
    # migration. Keep this narrow so private framework internals remain blocked.
    "victor.framework.extensions",
)
_RUNTIME_IMPORT_REPLACEMENTS = {
    "victor.framework": "victor_contracts.capabilities or a specific victor_contracts runtime adapter",
    "victor.framework.enrichment": "victor_contracts.enrichment_runtime",
    "victor.framework.capabilities": "victor_contracts.capabilities",
    "victor.framework.capability_config_helpers": "victor_contracts.capabilities",
    "victor.framework.capability_loader": "victor_contracts.capabilities",
    "victor.framework.config": "victor_contracts.safety",
    "victor.framework.multi_agent": "victor_contracts.multi_agent",
    "victor.framework.rl.config": "victor_contracts.rl",
    "victor.framework.team_registry": "PluginContext team registration or victor_contracts.team_schema",
    "victor.framework.tool_naming": "victor_contracts.constants",
    "victor.providers": "victor_contracts.provider_runtime",
    "victor.security.safety.pii": "victor_contracts.safety",
    "victor.workflows": "victor_contracts.workflow_runtime",
}
_IGNORED_DIR_NAMES = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "htmlcov",
    "node_modules",
    "site",
    "tests",
    "venv",
}


@dataclass(frozen=True)
class ContractAuditConfig:
    """Optional repo-local configuration for the contract auditor."""

    source_roots: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()


@dataclass(frozen=True)
class AuditIssue:
    """A single contract audit finding."""

    level: str
    code: str
    message: str
    path: str | None = None
    line: int | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "level": self.level,
            "code": self.code,
            "message": self.message,
            "path": self.path,
            "line": self.line,
        }


@dataclass
class VerticalContractAuditReport:
    """Structured report for a single vertical repository audit."""

    root_path: Path
    project_name: str = ""
    plugin_entry_points: list[str] = field(default_factory=list)
    issues: list[AuditIssue] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        """Return the number of error findings."""

        return sum(1 for issue in self.issues if issue.level == "error")

    @property
    def warning_count(self) -> int:
        """Return the number of warning findings."""

        return sum(1 for issue in self.issues if issue.level == "warning")

    @property
    def passed(self) -> bool:
        """Return True when the audit has no errors."""

        return self.error_count == 0

    def add_issue(
        self,
        level: str,
        code: str,
        message: str,
        *,
        path: str | None = None,
        line: int | None = None,
    ) -> None:
        """Append a finding to the report."""

        self.issues.append(
            AuditIssue(level=level, code=code, message=message, path=path, line=line)
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "root_path": str(self.root_path),
            "project_name": self.project_name,
            "passed": self.passed,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "plugin_entry_points": list(self.plugin_entry_points),
            "issues": [issue.to_dict() for issue in self.issues],
        }


class VerticalContractAuditor:
    """Audit extracted vertical repositories against core integration rules."""

    def audit_paths(self, paths: Iterable[str | Path]) -> list[VerticalContractAuditReport]:
        """Audit multiple repository paths."""

        return [self.audit_path(path) for path in paths]

    def audit_path(self, path: str | Path) -> VerticalContractAuditReport:
        """Audit a single extracted vertical repository path."""

        root_path = Path(path).expanduser()
        if root_path.is_file() and root_path.name == "pyproject.toml":
            root_path = root_path.parent

        report = VerticalContractAuditReport(root_path=root_path)

        if not root_path.exists():
            report.add_issue(
                "error",
                "missing_path",
                "Path does not exist.",
            )
            return report

        pyproject_path = root_path / "pyproject.toml"
        if not pyproject_path.exists():
            report.add_issue(
                "error",
                "missing_pyproject",
                "Repository does not contain a pyproject.toml file.",
            )
            return report

        try:
            pyproject_data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        except Exception as exc:
            report.add_issue(
                "error",
                "invalid_pyproject",
                f"Could not parse pyproject.toml: {exc}",
                path="pyproject.toml",
            )
            return report

        project = pyproject_data.get("project", {}) if isinstance(pyproject_data, dict) else {}
        report.project_name = str(project.get("name") or root_path.name)
        audit_config = self._load_audit_config(pyproject_data)

        entry_points = project.get("entry-points", {}) if isinstance(project, dict) else {}
        plugin_entries = entry_points.get("victor.plugins", {})
        if isinstance(plugin_entries, dict):
            report.plugin_entry_points = sorted(str(key) for key in plugin_entries.keys())
        if not report.plugin_entry_points:
            report.add_issue(
                "error",
                "missing_plugin_entry_point",
                'project.entry-points."victor.plugins" is missing or empty.',
                path="pyproject.toml",
            )

        legacy_entries = entry_points.get("victor.verticals", {})
        if isinstance(legacy_entries, dict) and legacy_entries:
            report.add_issue(
                "warning",
                "legacy_vertical_entry_point",
                "Legacy victor.verticals entry points are still declared; migrate to victor.plugins.",
                path="pyproject.toml",
            )

        dependency_names = self._collect_dependency_names(project)
        required_dependency_names = self._collect_dependency_names(
            project,
            include_optional=False,
        )
        if "victor-contracts" not in dependency_names:
            report.add_issue(
                "warning",
                "missing_sdk_dependency",
                "victor-contracts is not declared in project dependencies.",
                path="pyproject.toml",
            )
        if "victor-ai" in required_dependency_names:
            report.add_issue(
                "error",
                "required_core_runtime_dependency",
                "victor-ai is declared as a required dependency; external verticals must "
                "depend on victor-contracts and keep victor-ai optional/runtime-only.",
                path="pyproject.toml",
            )

        for missing_root in self._find_missing_source_roots(root_path, audit_config):
            report.add_issue(
                "warning",
                "missing_contract_audit_source_root",
                f"Configured contract-audit source root '{missing_root}' does not exist.",
                path="pyproject.toml",
            )

        for issue in self._scan_python_sources(root_path, audit_config):
            report.issues.append(issue)

        return report

    def _load_audit_config(self, pyproject_data: dict[str, object]) -> ContractAuditConfig:
        """Load repo-local contract-audit config from pyproject metadata."""

        if not isinstance(pyproject_data, dict):
            return ContractAuditConfig()

        tool_data = pyproject_data.get("tool", {})
        if not isinstance(tool_data, dict):
            return ContractAuditConfig()

        victor_data = tool_data.get("victor", {})
        if not isinstance(victor_data, dict):
            return ContractAuditConfig()

        raw_config = victor_data.get("contract_audit", {})
        if not isinstance(raw_config, dict):
            return ContractAuditConfig()

        raw_source_roots = raw_config.get("source_roots", [])
        raw_exclude = raw_config.get("exclude", [])

        source_roots = tuple(str(item) for item in raw_source_roots if str(item).strip())
        exclude = tuple(str(item) for item in raw_exclude if str(item).strip())
        return ContractAuditConfig(source_roots=source_roots, exclude=exclude)

    def _find_missing_source_roots(
        self,
        root_path: Path,
        config: ContractAuditConfig,
    ) -> list[str]:
        """Return configured source roots that do not exist."""

        missing: list[str] = []
        for raw_root in config.source_roots:
            candidate = (root_path / raw_root).resolve()
            try:
                candidate.relative_to(root_path.resolve())
            except ValueError:
                missing.append(raw_root)
                continue
            if not candidate.exists():
                missing.append(raw_root)
        return missing

    def _collect_dependency_names(
        self,
        project: dict[str, object],
        *,
        include_optional: bool = True,
    ) -> set[str]:
        """Return normalized dependency package names from project metadata."""

        dependencies: list[str] = []
        raw_dependencies = project.get("dependencies", [])
        if isinstance(raw_dependencies, list):
            dependencies.extend(str(dep) for dep in raw_dependencies)

        if include_optional:
            optional_dependencies = project.get("optional-dependencies", {})
            if isinstance(optional_dependencies, dict):
                for group_dependencies in optional_dependencies.values():
                    if isinstance(group_dependencies, list):
                        dependencies.extend(str(dep) for dep in group_dependencies)

        names: set[str] = set()
        for dependency in dependencies:
            normalized = dependency.strip()
            if not normalized:
                continue
            token = normalized.split(";", 1)[0]
            token = token.split("[", 1)[0]
            for separator in ("<", ">", "=", "!", "~", " "):
                token = token.split(separator, 1)[0]
            token = token.strip().lower()
            if token:
                names.add(token)
        return names

    def _scan_python_sources(
        self,
        root_path: Path,
        config: ContractAuditConfig,
    ) -> list[AuditIssue]:
        """Scan repository Python sources for forbidden runtime imports."""

        issues: list[AuditIssue] = []
        seen_paths: set[Path] = set()
        for scan_root in self._iter_scan_roots(root_path, config):
            candidates = [scan_root] if scan_root.is_file() else list(scan_root.rglob("*.py"))
            for source_path in candidates:
                if source_path in seen_paths:
                    continue
                seen_paths.add(source_path)
                if self._should_skip_path(root_path, source_path, config):
                    continue
                relative_path = source_path.relative_to(root_path)
                try:
                    tree = ast.parse(source_path.read_text(encoding="utf-8"))
                except SyntaxError as exc:
                    issues.append(
                        AuditIssue(
                            level="warning",
                            code="invalid_python_source",
                            message=f"Could not parse Python source: {exc.msg}",
                            path=str(relative_path),
                            line=exc.lineno,
                        )
                    )
                    continue

                for node in ast.walk(tree):
                    module_name = self._extract_imported_module(node)
                    if module_name and self._is_forbidden_runtime_import(module_name):
                        replacement = self._replacement_hint(module_name)
                        replacement_suffix = f" Use {replacement} instead." if replacement else ""
                        issues.append(
                            AuditIssue(
                                level="error",
                                code="forbidden_runtime_import",
                                message=(
                                    f"Direct runtime import '{module_name}' leaks core internals "
                                    "into the extracted vertical boundary."
                                    f"{replacement_suffix}"
                                ),
                                path=str(relative_path),
                                line=getattr(node, "lineno", None),
                            )
                        )
                    dynamic_module_name = self._extract_dynamic_import_module(node)
                    if dynamic_module_name and self._is_forbidden_runtime_import(
                        dynamic_module_name
                    ):
                        replacement = self._replacement_hint(dynamic_module_name)
                        replacement_suffix = f" Use {replacement} instead." if replacement else ""
                        issues.append(
                            AuditIssue(
                                level="error",
                                code="forbidden_runtime_dynamic_import",
                                message=(
                                    f"Dynamic runtime import '{dynamic_module_name}' leaks core "
                                    "internals into the extracted vertical boundary."
                                    f"{replacement_suffix}"
                                ),
                                path=str(relative_path),
                                line=getattr(node, "lineno", None),
                            )
                        )
        return issues

    def _iter_scan_roots(self, root_path: Path, config: ContractAuditConfig) -> Iterable[Path]:
        """Yield filesystem roots to scan for Python sources."""

        if not config.source_roots:
            yield root_path
            return

        root_resolved = root_path.resolve()
        for raw_root in config.source_roots:
            candidate = (root_path / raw_root).resolve()
            try:
                candidate.relative_to(root_resolved)
            except ValueError:
                continue
            if not candidate.exists():
                continue
            yield candidate

    def _should_skip_path(
        self,
        root_path: Path,
        source_path: Path,
        config: ContractAuditConfig,
    ) -> bool:
        """Return True for ignored files or directories."""

        relative_path = source_path.relative_to(root_path)
        relative_parts = relative_path.parts
        if any(part in _IGNORED_DIR_NAMES for part in relative_parts[:-1]):
            return True
        if self._matches_exclude_patterns(relative_path, config.exclude):
            return True
        filename = relative_parts[-1]
        return filename.startswith("test_")

    def _matches_exclude_patterns(
        self,
        relative_path: Path,
        patterns: tuple[str, ...],
    ) -> bool:
        """Return True when *relative_path* matches any configured exclude pattern."""

        path_str = relative_path.as_posix()
        for pattern in patterns:
            if fnmatch(path_str, pattern) or relative_path.match(pattern):
                return True
        return False

    def _extract_imported_module(self, node: ast.AST) -> str | None:
        """Return the imported module path for an import node."""

        if isinstance(node, ast.Import):
            for alias in node.names:
                if self._is_forbidden_runtime_import(alias.name):
                    return alias.name
            return None
        if isinstance(node, ast.ImportFrom):
            return node.module
        return None

    def _extract_dynamic_import_module(self, node: ast.AST) -> str | None:
        """Return the module path from supported dynamic import calls."""

        if not isinstance(node, ast.Call):
            return None

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

    def _is_forbidden_runtime_import(self, module_name: str) -> bool:
        """Return True when *module_name* crosses the allowed vertical boundary."""

        if any(
            module_name == prefix or module_name.startswith(f"{prefix}.")
            for prefix in ALLOWED_RUNTIME_IMPORT_PREFIXES
        ):
            return False

        return any(
            module_name == prefix or module_name.startswith(f"{prefix}.")
            for prefix in FORBIDDEN_RUNTIME_IMPORT_PREFIXES
        )

    def _replacement_hint(self, module_name: str) -> str | None:
        """Return the preferred SDK replacement for a forbidden runtime import."""

        for prefix, replacement in sorted(
            _RUNTIME_IMPORT_REPLACEMENTS.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        ):
            if module_name == prefix or module_name.startswith(f"{prefix}."):
                return replacement
        return None

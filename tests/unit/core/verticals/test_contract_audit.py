"""Tests for extracted-vertical contract auditing."""

from pathlib import Path

from victor.core.verticals.contract_audit import VerticalContractAuditor


def _write_pyproject(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_audit_reports_missing_plugin_entry_point(tmp_path: Path) -> None:
    repo = tmp_path / "victor-sample"
    package_dir = repo / "victor_sample"
    package_dir.mkdir(parents=True)
    _write_pyproject(
        repo / "pyproject.toml",
        """
[project]
name = "victor-sample"
version = "0.1.0"
dependencies = ["victor-contracts>=0.1.0"]
""".strip(),
    )
    (package_dir / "__init__.py").write_text("", encoding="utf-8")

    report = VerticalContractAuditor().audit_path(repo)

    assert report.passed is False
    assert any(issue.code == "missing_plugin_entry_point" for issue in report.issues)


def test_audit_reports_forbidden_core_runtime_imports(tmp_path: Path) -> None:
    repo = tmp_path / "victor-leaky"
    package_dir = repo / "victor_leaky"
    package_dir.mkdir(parents=True)
    _write_pyproject(
        repo / "pyproject.toml",
        """
[project]
name = "victor-leaky"
version = "0.1.0"
dependencies = ["victor-contracts>=0.1.0"]

[project.entry-points."victor.plugins"]
coding = "victor_leaky.plugin:get_plugin"
""".strip(),
    )
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "plugin.py").write_text(
        "from victor.framework.agent import Agent\n",
        encoding="utf-8",
    )

    report = VerticalContractAuditor().audit_path(repo)

    assert report.passed is False
    assert any(issue.code == "forbidden_runtime_import" for issue in report.issues)


def test_audit_reports_sdk_replacement_hints(tmp_path: Path) -> None:
    repo = tmp_path / "victor-replaceable"
    package_dir = repo / "victor_replaceable"
    package_dir.mkdir(parents=True)
    _write_pyproject(
        repo / "pyproject.toml",
        """
[project]
name = "victor-replaceable"
version = "0.1.0"
dependencies = ["victor-contracts>=0.1.0"]

[project.entry-points."victor.plugins"]
replaceable = "victor_replaceable.plugin:get_plugin"
""".strip(),
    )
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "plugin.py").write_text(
        "\n".join(
            [
                "from victor.workflows.executor import NodeResult",
                "from victor.security.safety.pii import PIIScanner",
                "from victor.framework import CapabilityLoader",
            ]
        ),
        encoding="utf-8",
    )

    report = VerticalContractAuditor().audit_path(repo)

    assert report.passed is False
    messages = "\n".join(issue.message for issue in report.issues)
    assert "victor_contracts.workflow_runtime" in messages
    assert "victor_contracts.safety" in messages
    assert "victor_contracts.capabilities" in messages


def test_audit_allows_documented_framework_extension_imports(tmp_path: Path) -> None:
    repo = tmp_path / "victor-extension-user"
    package_dir = repo / "victor_extension_user"
    package_dir.mkdir(parents=True)
    _write_pyproject(
        repo / "pyproject.toml",
        """
[project]
name = "victor-extension-user"
version = "0.1.0"
dependencies = ["victor-contracts>=0.1.0"]

[project.entry-points."victor.plugins"]
extension_user = "victor_extension_user.plugin:get_plugin"
""".strip(),
    )
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "plugin.py").write_text(
        "\n".join(
            [
                "from victor.framework.extensions import SafetyExtensionProtocol",
                "__import__('victor.framework.extensions')",
            ]
        ),
        encoding="utf-8",
    )

    report = VerticalContractAuditor().audit_path(repo)

    assert report.passed is True
    assert report.error_count == 0


def test_audit_reports_forbidden_dynamic_core_runtime_imports(tmp_path: Path) -> None:
    repo = tmp_path / "victor-dynamic-leaky"
    package_dir = repo / "victor_dynamic_leaky"
    package_dir.mkdir(parents=True)
    _write_pyproject(
        repo / "pyproject.toml",
        """
[project]
name = "victor-dynamic-leaky"
version = "0.1.0"
dependencies = ["victor-contracts>=0.1.0"]

[project.entry-points."victor.plugins"]
dynamic = "victor_dynamic_leaky.plugin:get_plugin"
""".strip(),
    )
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "plugin.py").write_text(
        "\n".join(
            [
                "import importlib",
                "importlib.import_module('victor.agent.orchestrator')",
                "__import__('victor.core.container')",
            ]
        ),
        encoding="utf-8",
    )

    report = VerticalContractAuditor().audit_path(repo)

    assert report.passed is False
    dynamic_issues = [
        issue for issue in report.issues if issue.code == "forbidden_runtime_dynamic_import"
    ]
    assert len(dynamic_issues) == 2


def test_audit_rejects_required_victor_ai_dependency(tmp_path: Path) -> None:
    repo = tmp_path / "victor-core-dependent"
    package_dir = repo / "victor_core_dependent"
    package_dir.mkdir(parents=True)
    _write_pyproject(
        repo / "pyproject.toml",
        """
[project]
name = "victor-core-dependent"
version = "0.1.0"
dependencies = ["victor-contracts>=0.1.0", "victor-ai>=0.7.0"]

[project.entry-points."victor.plugins"]
dependent = "victor_core_dependent.plugin:get_plugin"
""".strip(),
    )
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "plugin.py").write_text(
        "from victor_contracts.core.plugins import VictorPlugin\n",
        encoding="utf-8",
    )

    report = VerticalContractAuditor().audit_path(repo)

    assert report.passed is False
    assert any(issue.code == "required_core_runtime_dependency" for issue in report.issues)


def test_audit_allows_optional_victor_ai_runtime_dependency(tmp_path: Path) -> None:
    repo = tmp_path / "victor-optional-runtime"
    package_dir = repo / "victor_optional_runtime"
    package_dir.mkdir(parents=True)
    _write_pyproject(
        repo / "pyproject.toml",
        """
[project]
name = "victor-optional-runtime"
version = "0.1.0"
dependencies = ["victor-contracts>=0.1.0"]

[project.optional-dependencies]
runtime = ["victor-ai>=0.7.0"]

[project.entry-points."victor.plugins"]
optional = "victor_optional_runtime.plugin:get_plugin"
""".strip(),
    )
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "plugin.py").write_text(
        "from victor_contracts.core.plugins import VictorPlugin\n",
        encoding="utf-8",
    )

    report = VerticalContractAuditor().audit_path(repo)

    assert report.passed is True
    assert not any(issue.code == "required_core_runtime_dependency" for issue in report.issues)


def test_audit_passes_sdk_pure_vertical_repo(tmp_path: Path) -> None:
    repo = tmp_path / "victor-clean"
    package_dir = repo / "victor_clean"
    package_dir.mkdir(parents=True)
    _write_pyproject(
        repo / "pyproject.toml",
        """
[project]
name = "victor-clean"
version = "0.1.0"
dependencies = ["victor-contracts>=0.1.0"]

[project.entry-points."victor.plugins"]
clean = "victor_clean.plugin:get_plugin"
""".strip(),
    )
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "plugin.py").write_text(
        "from victor_contracts.core.plugins import VictorPlugin\n",
        encoding="utf-8",
    )

    report = VerticalContractAuditor().audit_path(repo)

    assert report.passed is True
    assert report.error_count == 0


def test_audit_respects_configured_source_roots_and_excludes(tmp_path: Path) -> None:
    repo = tmp_path / "victor-scoped"
    package_dir = repo / "victor_scoped"
    legacy_dir = repo / "src" / "investigator"
    package_dir.mkdir(parents=True)
    legacy_dir.mkdir(parents=True)
    _write_pyproject(
        repo / "pyproject.toml",
        """
[project]
name = "victor-scoped"
version = "0.1.0"
dependencies = ["victor-contracts>=0.1.0"]

[project.entry-points."victor.plugins"]
scoped = "victor_scoped.plugin:get_plugin"

[tool.victor.contract_audit]
source_roots = ["victor_scoped"]
exclude = ["victor_scoped/ignored.py"]
""".strip(),
    )
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "plugin.py").write_text(
        "from victor_contracts.core.plugins import VictorPlugin\n",
        encoding="utf-8",
    )
    (package_dir / "ignored.py").write_text(
        "from victor.framework.agent import Agent\n",
        encoding="utf-8",
    )
    (legacy_dir / "app.py").write_text(
        "from victor.framework.agent import Agent\n",
        encoding="utf-8",
    )

    report = VerticalContractAuditor().audit_path(repo)

    assert report.passed is True
    assert report.error_count == 0


def test_audit_warns_on_missing_configured_source_root(tmp_path: Path) -> None:
    repo = tmp_path / "victor-missing-root"
    package_dir = repo / "victor_missing_root"
    package_dir.mkdir(parents=True)
    _write_pyproject(
        repo / "pyproject.toml",
        """
[project]
name = "victor-missing-root"
version = "0.1.0"
dependencies = ["victor-contracts>=0.1.0"]

[project.entry-points."victor.plugins"]
missing = "victor_missing_root.plugin:get_plugin"

[tool.victor.contract_audit]
source_roots = ["victor_missing_root", "nonexistent"]
""".strip(),
    )
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "plugin.py").write_text(
        "from victor_contracts.core.plugins import VictorPlugin\n",
        encoding="utf-8",
    )

    report = VerticalContractAuditor().audit_path(repo)

    assert report.passed is True
    assert report.warning_count == 1
    assert any(issue.code == "missing_contract_audit_source_root" for issue in report.issues)


def test_runtime_import_replacements_target_non_deprecated_contract_modules() -> None:
    """Audit remediation pointers must never recommend a deprecated bridge.

    The six bridges in ``victor_contracts._deprecation`` warn since SDK 0.9.0
    and are removed no earlier than 0.10.0 (see CONTRACT_STABILITY.md), so the
    contract auditor must steer extracted verticals elsewhere.
    """
    import importlib.util
    import re

    from victor_contracts._deprecation import DEPRECATED_BRIDGE_REPLACEMENTS

    from victor.core.verticals.contract_audit import _RUNTIME_IMPORT_REPLACEMENTS

    deprecated = set(DEPRECATED_BRIDGE_REPLACEMENTS)
    assert deprecated, "deprecated bridge registry should not be empty"

    for forbidden_prefix, replacement in _RUNTIME_IMPORT_REPLACEMENTS.items():
        targets = re.findall(r"victor_contracts\.(\w+)", replacement)
        assert targets, f"replacement for {forbidden_prefix!r} names no victor_contracts module"
        for target in targets:
            assert target not in deprecated, (
                f"replacement for {forbidden_prefix!r} points at deprecated "
                f"victor_contracts.{target}"
            )
            assert importlib.util.find_spec(f"victor_contracts.{target}") is not None, (
                f"replacement for {forbidden_prefix!r} points at nonexistent "
                f"victor_contracts.{target}"
            )

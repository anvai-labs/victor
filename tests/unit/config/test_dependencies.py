"""Dependency checks must see pulled source pins, not stale editable metadata."""

from importlib import metadata
import shlex

from victor.config import dependencies as deps
from victor.config.validation import ConfigValidationResult, ConfigValidator


def source_tree(tmp_path, requirements):
    root = tmp_path / "victor checkout"
    module = root / "victor/config/dependencies.py"
    module.parent.mkdir(parents=True)
    module.touch()
    (root / "pyproject.toml").write_text(
        '[project]\nname = "victor-ai"\ndependencies = ' + repr(requirements) + "\n"
    )
    return module


def test_source_pin_overrides_stale_editable_metadata(tmp_path, monkeypatch):
    module = source_tree(tmp_path, ["sandhi-gateway==0.13.0"])
    monkeypatch.setattr(deps, "__file__", str(module))
    monkeypatch.setattr(deps.metadata, "requires", lambda _: ["sandhi-gateway==0.1.5"])
    monkeypatch.setattr(deps.metadata, "version", lambda _: "0.11.0")
    issues = deps.check_dependencies()
    assert len(issues) == 1
    assert "sandhi-gateway==0.13.0" in issues[0].message
    assert "0.11.0" in issues[0].message
    command = shlex.split(issues[0].suggestion.removeprefix("Run: "))
    assert command[:4] == ["uv", "pip", "install", "--python"]
    assert command[-2:] == ["-e", str(module.parents[2])]


def test_wheel_uses_metadata_and_ignores_inactive_extras(tmp_path, monkeypatch):
    monkeypatch.setattr(deps, "__file__", str(tmp_path / "victor/config/dependencies.py"))
    monkeypatch.setattr(
        deps.metadata,
        "requires",
        lambda _: [
            "sandhi-gateway==0.13.0",
            'numpy>=1.24; extra == "embeddings"',
            'win-only==1; sys_platform == "win32"',
        ],
    )
    monkeypatch.setattr(deps.metadata, "version", lambda _: "0.13.0")
    assert deps.check_dependencies() == []


def test_missing_dependency_warns_and_uv_tool_repair_preserves_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(deps, "__file__", str(tmp_path / "victor/config/dependencies.py"))
    monkeypatch.setattr(deps.metadata, "requires", lambda _: ["sandhi-gateway==0.13.0"])

    def missing(_):
        raise metadata.PackageNotFoundError

    monkeypatch.setattr(deps.metadata, "version", missing)
    monkeypatch.setattr(deps.sys, "prefix", str(tmp_path))
    (tmp_path / "uv-receipt.toml").touch()
    issue = deps.check_dependencies()[0]
    assert "not installed" in issue.message
    assert issue.suggestion == "Run: uv tool upgrade victor-ai"


def test_matching_source_requirements_do_not_warn(tmp_path, monkeypatch):
    monkeypatch.setattr(deps, "__file__", str(source_tree(tmp_path, ["sandhi-gateway==0.13.0"])))
    monkeypatch.setattr(deps.metadata, "version", lambda _: "0.13.0")
    assert deps.check_dependencies() == []


def test_wrong_project_name_does_not_override_wheel_metadata(tmp_path, monkeypatch):
    module = source_tree(tmp_path, ["sandhi-gateway==0.13.0"])
    (module.parents[2] / "pyproject.toml").write_text('[project]\nname = "other"\n')
    monkeypatch.setattr(deps, "__file__", str(module))
    monkeypatch.setattr(deps.metadata, "requires", lambda _: ["sandhi-gateway==0.10.1"])
    monkeypatch.setattr(deps.metadata, "version", lambda _: "0.10.1")
    assert deps.check_dependencies() == []


def test_config_validation_keeps_dependency_warning_actionable(monkeypatch):
    issue = deps.DependencyIssue("sandhi-gateway==0.13.0 required; installed 0.11.0", "Run: repair")
    monkeypatch.setattr(deps, "check_dependencies", lambda: [issue])
    result = ConfigValidationResult()
    ConfigValidator()._validate_dependencies(None, result)
    assert result.is_valid()
    assert result.warnings[0].category == "dependencies"
    assert result.warnings[0].suggestion == "Run: repair"


def test_doctor_reports_dependency_pin_mismatch(monkeypatch):
    from victor.ui.commands.doctor import DoctorChecks, Severity

    monkeypatch.setattr(
        deps,
        "check_dependencies",
        lambda: [
            deps.DependencyIssue("sandhi-gateway==0.13.0 required; installed 0.11.0", "Run: repair")
        ],
    )
    doctor = DoctorChecks()
    doctor.check_dependencies()
    assert any(
        c.severity == Severity.WARNING and c.suggestion == "Run: repair" for c in doctor.checks
    )


def test_onboarding_shows_environment_repair_without_installing(monkeypatch):
    from io import StringIO
    from rich.console import Console
    from victor.ui.commands.onboarding import OnboardingWizard

    monkeypatch.setattr(
        deps,
        "check_dependencies",
        lambda: [
            deps.DependencyIssue(
                "sandhi-gateway==0.13.0 required; installed 0.11.0",
                "Run: uv tool upgrade victor-ai",
            )
        ],
    )
    output = StringIO()
    OnboardingWizard(console=Console(file=output, width=120))._show_environment_issues()
    assert "Environment needs attention" in output.getvalue()
    assert "uv tool upgrade victor-ai" in output.getvalue()

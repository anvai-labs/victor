"""Security gates must distinguish completed audits from missing coverage."""

from copy import deepcopy
from datetime import date
import importlib.util
import json
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[3] / "scripts/ci/security_report_check.py"
SPEC = importlib.util.spec_from_file_location("security_report_check", PATH)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def python_report(vulnerable=False):
    return {
        "dependencies": [
            {
                "name": "diskcache",
                "version": "5.6.3",
                "vulns": (
                    [{"id": "PYSEC-2026-2447", "aliases": ["CVE-2025-69872"]}] if vulnerable else []
                ),
            }
        ],
        "fixes": [],
    }


def exception():
    return {
        "package": "diskcache",
        "versions": ["5.6.3"],
        "advisories": ["CVE-2025-69872"],
        "scopes": ["python"],
        "owner": "test-owner",
        "reason": "Test fixture only",
        "expires": "2026-11-01",
    }


def trivy_report(severity="HIGH", scope="filesystem"):
    target = {
        "Class": "lang-pkgs",
        "Type": "python-pkg" if scope == "container" else "pip",
        "Target": "requirements.txt",
        "Packages": [{"Name": "diskcache", "Version": "5.6.3"}],
        "Vulnerabilities": [
            {
                "VulnerabilityID": "CVE-2025-69872",
                "PkgName": "diskcache",
                "InstalledVersion": "5.6.3",
                "Severity": severity,
            }
        ],
    }
    results = [target]
    if scope == "container":
        results.append(
            {
                "Class": "os-pkgs",
                "Type": "debian",
                "Target": "debian",
                "Packages": [{"Name": "libc6", "Version": "2.41"}],
            }
        )
    return {
        "SchemaVersion": 2,
        "ArtifactType": "container_image" if scope == "container" else "filesystem",
        "Results": results,
    }


def test_nested_advisory_cannot_be_misread_as_clean():
    report = python_report(True)
    assert report.get("vulnerabilities", []) == []  # Demonstrate the previous bug.
    assert gate.check(report, [], "python", 1)


def test_clean_complete_audit_passes():
    assert not gate.check(python_report(), [], "python", 0)


@pytest.mark.parametrize("status", [0, 2, 127, -9])
def test_vulnerable_report_with_wrong_or_failed_exit_blocks(status):
    with pytest.raises(ValueError):
        gate.check(python_report(True), [], "python", status)


def test_error_status_cannot_reuse_clean_report():
    with pytest.raises(ValueError, match="mismatch"):
        gate.check(python_report(), [], "python", 1)


@pytest.mark.parametrize(
    "report",
    [
        {},
        [],
        {"vulnerabilities": []},
        {"dependencies": []},
        {"dependencies": [{"name": "private", "skip_reason": "Not on PyPI"}]},
        {"dependencies": [{"name": "x", "version": "1"}]},
        {"dependencies": [{"name": "x", "version": "1", "vulns": None}]},
    ],
)
def test_incomplete_python_coverage_blocks(report):
    with pytest.raises(ValueError):
        gate.check(report, [], "python", 0)


def test_alias_exception_is_scoped_to_package_version_and_surface():
    entry = exception()
    report = python_report(True)
    assert not gate.check(report, [entry], "python", 1)
    changed = deepcopy(report)
    changed["dependencies"][0]["name"] = "different-package"
    assert gate.check(changed, [entry], "python", 1)
    changed["dependencies"][0]["name"] = "diskcache"
    changed["dependencies"][0]["version"] = "5.6.2"
    assert gate.check(changed, [entry], "python", 1)
    assert gate.check(trivy_report(scope="container"), [entry], "container", 0)


@pytest.mark.parametrize(
    "field,value",
    [
        ("expires", "2025-01-01"),
        ("owner", ""),
        ("reason", ""),
        ("versions", ["*"]),
        ("scopes", ["all"]),
        ("advisories", []),
    ],
)
def test_invalid_or_expired_exceptions_block(tmp_path, field, value):
    entry = exception()
    entry[field] = value
    path = tmp_path / "exceptions.json"
    path.write_text(json.dumps({"version": 1, "exceptions": [entry]}))
    with pytest.raises(ValueError):
        gate.read_exceptions(path, date(2026, 9, 12))


def test_exception_expiry_is_checked_again_at_each_invocation(tmp_path):
    path = tmp_path / "exceptions.json"
    path.write_text(json.dumps({"version": 1, "exceptions": [exception()]}))
    assert gate.read_exceptions(path, date(2026, 10, 1))
    with pytest.raises(ValueError, match="Expired"):
        gate.read_exceptions(path, date(2026, 11, 2))


@pytest.mark.parametrize("severity", ["HIGH", "CRITICAL", "UNKNOWN"])
def test_container_blocks_high_critical_and_unclassified_findings(severity):
    assert gate.check(trivy_report(severity, "container"), [], "container", 0)


def test_lower_severity_remains_in_report_without_failing_higher_floor():
    report = trivy_report("MEDIUM")
    assert not gate.check(report, [], "filesystem", 0, "HIGH")
    assert gate.check(report, [], "filesystem", 0, "MEDIUM")


@pytest.mark.parametrize("artifact_type", ["filesystem", "repository"])
def test_checkout_and_worktree_scans_enforce_the_same_policy(artifact_type):
    report = trivy_report("LOW")
    report["ArtifactType"] = artifact_type
    assert not gate.check(report, [], "filesystem", 0, "CRITICAL")
    report["Results"][0]["Vulnerabilities"][0]["Severity"] = "CRITICAL"
    assert gate.check(report, [], "filesystem", 0, "CRITICAL")
    with pytest.raises(ValueError, match="artifact type"):
        gate.check(report, [], "container", 0)
    del report["Results"][0]["Packages"]
    with pytest.raises(ValueError, match="scanned packages"):
        gate.check(report, [], "filesystem", 0)


@pytest.mark.parametrize("artifact_type", ["unknown", "container_image", "", None])
def test_wrong_filesystem_artifact_type_is_rejected(artifact_type):
    report = trivy_report("LOW")
    report["ArtifactType"] = artifact_type
    with pytest.raises(ValueError, match="artifact type"):
        gate.check(report, [], "filesystem", 0)


@pytest.mark.parametrize(
    "mutation", ["schema", "type", "empty", "no-os", "no-packages", "severity"]
)
def test_incomplete_trivy_coverage_blocks(mutation):
    report = trivy_report(scope="container")
    if mutation == "schema":
        report["SchemaVersion"] = 3
    elif mutation == "type":
        report["ArtifactType"] = "filesystem"
    elif mutation == "empty":
        report["Results"] = []
    elif mutation == "no-os":
        report["Results"] = report["Results"][:1]
    elif mutation == "no-packages":
        del report["Results"][0]["Packages"]
    else:
        report["Results"][0]["Vulnerabilities"][0]["Severity"] = "unexpected"
    with pytest.raises(ValueError):
        gate.check(report, [], "container", 0)


def test_trivy_failure_cannot_use_an_existing_report():
    with pytest.raises(ValueError):
        gate.check(trivy_report(), [], "filesystem", 1)


def test_omitted_manifest_cannot_be_hidden_by_another_clean_target():
    report = trivy_report("LOW")
    with pytest.raises(ValueError, match="Cargo.lock"):
        gate.check(report, [], "filesystem", 0, targets={"requirements.txt", "rust/Cargo.lock"})
    assert not gate.check(report, [], "filesystem", 0, targets={"requirements.txt"})


def test_lockfile_discovery_covers_nested_packages_and_ignores_install_artifacts(tmp_path):
    for name in [
        "requirements.txt",
        "rust/Cargo.lock",
        "verticals/new/native/Cargo.lock",
        "extension/package-lock.json",
        "node_modules/nested/package-lock.json",
        ".worktrees/old/Cargo.lock",
    ]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("malformed contents must still require coverage")
    assert gate.expected_targets(tmp_path) == {
        "requirements.txt",
        "rust/Cargo.lock",
        "verticals/new/native/Cargo.lock",
        "extension/package-lock.json",
    }


@pytest.mark.parametrize(
    "package", [None, {}, {"Name": "diskcache"}, {"Name": "diskcache", "Version": ""}]
)
def test_malformed_package_inventory_is_not_clean(package):
    report = trivy_report("LOW")
    report["Results"][0]["Packages"] = [package]
    with pytest.raises(ValueError):
        gate.check(report, [], "filesystem", 0)


def test_container_language_scan_must_include_python():
    report = trivy_report("LOW", "container")
    report["Results"][0]["Type"] = "npm"
    with pytest.raises(ValueError, match="Python"):
        gate.check(report, [], "container", 0)


def test_cargo_synthetic_root_requires_a_complete_named_member_inventory():
    report = trivy_report("LOW")
    target = report["Results"][0]
    target["Type"] = "cargo"
    target["Packages"][0]["ID"] = "diskcache@5.6.3"
    root = {
        "ID": "workspace-hash",
        "AnalyzedBy": "cargo",
        "Relationship": "root",
        "DependsOn": ["diskcache@5.6.3"],
    }
    target["Packages"].append(root)
    assert not gate.check(report, [], "filesystem", 0)
    root["DependsOn"] = ["missing@1.0"]
    with pytest.raises(ValueError, match="Missing Cargo"):
        gate.check(report, [], "filesystem", 0)


@pytest.mark.parametrize("content", [None, "not json", '{"dependencies":[]}'])
def test_cli_missing_or_invalid_report_is_failure(tmp_path, content):
    report = tmp_path / "report.json"
    if content is not None:
        report.write_text(content)
    exceptions = tmp_path / "exceptions.json"
    exceptions.write_text('{"version":1,"exceptions":[]}')
    assert (
        gate.main(
            [
                str(report),
                "--scope",
                "python",
                "--scanner-status",
                "0",
                "--exceptions",
                str(exceptions),
            ]
        )
        == 2
    )

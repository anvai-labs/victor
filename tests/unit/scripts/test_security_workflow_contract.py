"""Validate publication prerequisites locally, without a release rehearsal."""

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]


def workflow(name):
    return yaml.safe_load((ROOT / ".github/workflows" / name).read_text())


def ancestors(jobs, name):
    needs = jobs[name].get("needs", [])
    if isinstance(needs, str):
        needs = [needs]
    return set(needs).union(*(ancestors(jobs, need) for need in needs))


def assert_release_gate(jobs):
    assert not jobs["security-scan"].get("continue-on-error")
    assert "if" not in jobs["security-scan"]
    for name in (
        "create-release",
        "publish-pypi",
        "publish-native-pypi",
        "publish-docker",
        "publish-testpypi",
    ):
        assert "security-scan" in ancestors(jobs, name)
        # Explicit status overrides can permit publication after failed needs.
        assert not any(
            word in str(jobs[name].get("if", "")) for word in ("always(", "failure(", "cancelled(")
        )
    assert {"build-python", "build-native", "build-binary"} <= ancestors(jobs, "generate-checksums")


def test_release_security_is_a_transitive_publication_prerequisite():
    assert_release_gate(workflow("release.yml")["jobs"])


@pytest.mark.parametrize("bypass", ["advisory", "missing-need", "always", "early-checksum"])
def test_release_guard_rejects_old_and_future_bypasses(bypass):
    jobs = deepcopy(workflow("release.yml")["jobs"])
    if bypass == "advisory":
        jobs["security-scan"]["continue-on-error"] = True
    elif bypass == "missing-need":
        jobs["create-release"]["needs"].remove("security-scan")
    elif bypass == "always":
        jobs["publish-docker"]["if"] = "always()"
    else:
        jobs["generate-checksums"]["needs"].remove("build-binary")
    with pytest.raises(AssertionError):
        assert_release_gate(jobs)


def test_scan_config_preserves_library_and_unfixed_evidence():
    config = yaml.safe_load((ROOT / ".github/security/trivy.yaml").read_text())
    assert set(config["pkg"]["types"]) == {"os", "library"}
    assert config["pkg"]["include-dev-deps"] is True
    assert config["scan"]["scanners"] == ["vuln"]
    assert set(config["severity"]) == {"UNKNOWN", "LOW", "MEDIUM", "HIGH", "CRITICAL"}
    assert config["list-all-pkgs"] is True
    assert config["vulnerability"]["ignore-unfixed"] is False
    assert not config["scan"].get("skip-dirs")
    assert not config["scan"].get("skip-files")
    assert not [
        line
        for line in (ROOT / config["ignorefile"]).read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_each_job_scans_once_and_converts_the_same_report():
    action = yaml.safe_load((ROOT / ".github/actions/security-scan/action.yml").read_text())
    steps = action["runs"]["steps"]
    scans = [s for s in steps if s.get("uses", "").startswith("aquasecurity/trivy-action@")]
    assert len(scans) == 1
    assert scans[0]["with"]["format"] == "json"
    assert scans[0]["with"]["exit-code"] == "0"
    assert not scans[0].get("continue-on-error")
    render = next(s for s in steps if s["name"] == "Render the same report for review")
    assert "trivy convert --format table security-report.json" in render["run"]
    assert "trivy convert --format sarif" in render["run"]
    gate = next(s for s in steps if s["name"] == "Enforce the security decision")
    assert gate["if"] == "always()"
    assert "scan_status=2" in gate["run"]
    assert "--scanner-status" in gate["run"]
    assert not gate.get("continue-on-error")


def test_both_python_pipelines_use_the_same_checker():
    for name in ("security.yml", "ci-fast.yml"):
        steps = workflow(name)["jobs"]["dependency-audit"]["steps"]
        assert sum(s.get("uses") == "./.github/actions/python-audit" for s in steps) == 1
        assert all('report.get("vulnerabilities"' not in s.get("run", "") for s in steps)

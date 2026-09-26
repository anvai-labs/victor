"""Validate publication prerequisites locally, without a release rehearsal."""

from copy import deepcopy
import hashlib
import importlib.util
import json
import sys
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
    assert {
        "build-python",
        "build-native",
        "build-binary",
        "build-vscode",
        "generate-sbom",
    } <= ancestors(jobs, "generate-checksums")


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


@pytest.fixture
def release_contract():
    spec = importlib.util.spec_from_file_location(
        "release_contract", ROOT / "scripts/ci/release_contract.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "event,ref,test_only,version,expected",
    [
        ("push", "refs/tags/v0.10.0", False, "0.10.0", (True, True, False)),
        ("push", "refs/tags/v0.10.0rc1", False, "0.10.0rc1", (False, True, True)),
        ("workflow_dispatch", "refs/tags/v0.10.0", True, "0.10.0", (False, False, True)),
        ("workflow_dispatch", "refs/heads/develop", True, "0.10.0", (False, False, True)),
    ],
)
def test_release_destinations_are_exclusive(
    release_contract, event, ref, test_only, version, expected
):
    plan = release_contract.release_plan(event, ref, test_only, version)
    assert (
        tuple(plan[k] == "true" for k in ("production", "github_release", "testpypi")) == expected
    )
    assert plan["version"] == version


@pytest.mark.parametrize(
    "event,ref,test_only,version",
    [
        ("workflow_dispatch", "refs/heads/main", False, "0.10.0"),
        ("push", "refs/heads/main", False, "0.10.0"),
        ("push", "refs/tags/v0.10.1", False, "0.10.0"),
        ("pull_request", "refs/tags/v0.10.0", False, "0.10.0"),
        ("workflow_dispatch", "refs/heads/main", "false", "0.10.0"),
        ("workflow_dispatch", "refs/heads/main", True, "refs/heads/main"),
    ],
)
def test_invalid_publication_context_has_no_destinations(
    release_contract, event, ref, test_only, version
):
    with pytest.raises(ValueError):
        release_contract.release_plan(event, ref, test_only, version)


def test_publication_jobs_consume_the_release_plan():
    jobs = workflow("release.yml")["jobs"]
    for name in ("publish-pypi", "publish-native-pypi", "publish-docker"):
        assert "release-plan" in jobs[name]["needs"]
        assert jobs[name]["if"] == "needs.release-plan.outputs.production == 'true'"
    assert jobs["create-release"]["if"] == "needs.release-plan.outputs.github_release == 'true'"
    assert jobs["publish-testpypi"]["if"] == "needs.release-plan.outputs.testpypi == 'true'"
    assert "create-release" not in ancestors(jobs, "publish-testpypi")
    assert "publish-native-pypi" in jobs["publish-pypi"]["needs"]
    extension = jobs["build-vscode"]["steps"]
    assert (
        next(s["run"] for s in extension if s.get("name") == "Run tests") == "xvfb-run -a npm test"
    )
    assert "npx --no-install vsce" in next(
        s["run"] for s in extension if s.get("name") == "Package extension (VSIX)"
    )
    for job in jobs.values():
        for step in job.get("steps", []):
            if step.get("uses", "").startswith("actions/upload-artifact@"):
                assert step["with"]["if-no-files-found"] == "error"
    assert all("|| true" not in s.get("run", "") for s in jobs["build-vscode"]["steps"])
    create = next(s for s in jobs["create-release"]["steps"] if s.get("name") == "Create Release")
    assert "artifacts/vscode-extension/*" in create["with"]["files"]
    assert "artifacts/sbom/*" in create["with"]["files"]


def test_checksums_verify_flat_downloads_including_vsix_and_sbom(release_contract, tmp_path):
    artifacts = tmp_path / "artifacts"
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    for directory, filename in [
        ("python-package", "victor.whl"),
        ("binary-macos", "victor.tar.gz"),
        ("native-wheels-macos", "victor-native.whl"),
        ("vscode-extension", "victor.vsix"),
        ("sbom", "sbom.json"),
    ]:
        path = artifacts / directory / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(filename.encode())
        (downloads / filename).write_bytes(path.read_bytes())
    manifest = downloads / "checksums.txt"
    release_contract.write_checksums(artifacts, manifest)
    lines = manifest.read_text().splitlines()
    assert len(lines) == 5
    for line in lines:
        digest, filename = line.split("  ")
        assert "/" not in filename
        assert hashlib.sha256((downloads / filename).read_bytes()).hexdigest() == digest
    (artifacts / "vscode-extension" / "victor.vsix").unlink()
    with pytest.raises(ValueError, match="missing a required"):
        release_contract.write_checksums(artifacts, manifest)
    (artifacts / "vscode-extension" / "victor.vsix").write_bytes(b"extension")
    # Two artifact directories cannot silently overwrite the same release asset.
    (artifacts / "sbom" / "victor.whl").write_bytes(b"different")
    with pytest.raises(ValueError, match="duplicate"):
        release_contract.write_checksums(artifacts, manifest)


def test_ruff_uses_the_project_ci_pin(release_contract, tmp_path):
    project = tmp_path / "pyproject.toml"
    project.write_text('[project.optional-dependencies]\nci = ["ruff==0.16.7"]\n')
    assert release_contract.ruff_requirement(project) == "ruff==0.16.7"
    project.write_text('[project.optional-dependencies]\nci = ["ruff>=0.16.7"]\n')
    with pytest.raises(ValueError):
        release_contract.ruff_requirement(project)
    for name in ("ci-fast.yml", "release.yml"):
        steps = [s for job in workflow(name)["jobs"].values() for s in job.get("steps", [])]
        installers = [
            s["run"] for s in steps if "pip install" in s.get("run", "") and "ruff" in s["run"]
        ]
        assert installers and all("release_contract.py ruff" in s for s in installers)


def test_native_version_check_rejects_partial_or_unconsumable_bumps(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "version_sync", ROOT / "scripts/check_version_sync.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    paths = {
        "rust/pyproject.toml": '[project]\nversion = "0.8.2"\n',
        "rust/crates/python-bindings/Cargo.toml": '[package]\nversion = "0.8.2"\n',
        "rust/Cargo.lock": '[[package]]\nname = "victor_native"\nversion = "0.8.2"\n',
        "pyproject.toml": '[project.optional-dependencies]\nnative = ["victor-native>=0.8.2"]\n',
    }
    for name, text in paths.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    assert module.native_version_errors(tmp_path) == []
    for name, text in paths.items():
        path = tmp_path / name
        path.write_text(text.replace("0.8.2", "0.8.1"))
        assert module.native_version_errors(tmp_path), name
        path.write_text(text)


@pytest.mark.parametrize("test_only", ["true", "false"])
def test_dispatch_webhook_flag_reaches_publication_outputs(
    release_contract, tmp_path, monkeypatch, test_only
):
    (tmp_path / "VERSION").write_text("0.10.0\n")
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "0.10.0"\n')
    event = tmp_path / "event.json"
    event.write_text(json.dumps({"inputs": {"publish_testpypi": test_only}}))
    output = tmp_path / "output"
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event))
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/develop")
    monkeypatch.setattr(sys, "argv", ["release_contract.py", "plan", "--root", str(tmp_path)])
    if test_only == "true":
        release_contract.main()
        values = dict(line.split("=", 1) for line in output.read_text().splitlines())
        assert values["testpypi"] == "true"
        assert values["production"] == values["github_release"] == "false"
    else:
        with pytest.raises(ValueError, match="matching"):
            release_contract.main()
        assert not output.exists()

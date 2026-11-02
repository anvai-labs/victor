"""Release artifacts must come from the complete installed-wheel matrix."""

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
TARGETS = {
    "ubuntu-24.04": "x86_64-unknown-linux-gnu",
    "ubuntu-24.04-arm": "aarch64-unknown-linux-gnu",
    "macos-15": "aarch64-apple-darwin",
    "macos-15-intel": "x86_64-apple-darwin",
    "windows-2025": "x86_64-pc-windows-msvc",
}


def load(name):
    return yaml.safe_load((ROOT / ".github/workflows" / name).read_text())


def assert_contract(qualification, release):
    assert (
        release["jobs"]["build-native"]["uses"]
        == "./.github/workflows/native-ccg-qualification.yml"
    )
    assert release["jobs"]["build-native"]["needs"] == "test"
    job = qualification["jobs"]["wheel"]
    matrix = job["strategy"]["matrix"]
    assert set(matrix["os"]) == set(TARGETS)
    assert set(matrix["python"]) == {"3.12", "3.13"}
    assert {row["os"]: row["target"] for row in matrix["include"]} == TARGETS
    assert job["env"]["RUSTFLAGS"] == "-C target-cpu=generic"
    assert not job.get("continue-on-error")
    probe = next(
        step for step in job["steps"] if "check_native_ccg_wheel.py" in step.get("run", "")
    )
    assert (
        probe["run"] == "python scripts/ci/check_native_ccg_wheel.py > native-ccg-wheel-check.json"
    )
    assert "if" not in probe and not probe.get("continue-on-error")
    build = next(step for step in job["steps"] if "PyO3/maturin-action@" in step.get("uses", ""))
    assert build["with"]["target"] == "${{ matrix.target }}"
    assert "--locked" in build["with"]["args"] and "--release" in build["with"]["args"]
    assert build["with"]["manylinux"] == "auto"
    artifacts = [
        step["with"] for step in job["steps"] if "actions/upload-artifact@" in step.get("uses", "")
    ]
    wheels = next(
        artifact for artifact in artifacts if artifact["name"].startswith("native-wheels-")
    )
    assert "matrix.target" in wheels["name"] and "matrix.python" in wheels["name"]
    assert wheels["path"] == "rust/dist/*.whl"
    assert wheels["if-no-files-found"] == "error"
    assert qualification["concurrency"]["group"].startswith("native-ccg-qualification-")


def test_release_reuses_every_qualified_platform_interpreter_wheel():
    assert_contract(load("native-ccg-qualification.yml"), load("release.yml"))


@pytest.mark.parametrize(
    "bypass",
    ["interpreter", "platform", "advisory", "skipped-probe", "native-cpu", "release-drift"],
)
def test_packaging_guard_rejects_incomplete_or_unqualified_wheels(bypass):
    qualification, release = deepcopy(load("native-ccg-qualification.yml")), deepcopy(
        load("release.yml")
    )
    job = qualification["jobs"]["wheel"]
    if bypass == "interpreter":
        job["strategy"]["matrix"]["python"].remove("3.13")
    elif bypass == "platform":
        job["strategy"]["matrix"]["os"].remove("macos-15-intel")
    elif bypass == "advisory":
        job["continue-on-error"] = True
    elif bypass == "skipped-probe":
        next(step for step in job["steps"] if "check_native_ccg_wheel.py" in step.get("run", ""))[
            "if"
        ] = "false"
    elif bypass == "native-cpu":
        job["env"]["RUSTFLAGS"] = "-C target-cpu=native"
    else:
        release["jobs"]["build-native"]["uses"] = "./.github/workflows/old-native.yml"
    with pytest.raises(AssertionError):
        assert_contract(qualification, release)

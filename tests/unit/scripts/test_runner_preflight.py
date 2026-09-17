"""Tests for the self-hosted runner compatibility preflight."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "runner_preflight",
    Path(__file__).resolve().parents[3] / "scripts/ci/runner_preflight.py",
)
assert spec is not None and spec.loader is not None
runner_preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner_preflight)

compatibility_errors = runner_preflight.compatibility_errors
parse_version = runner_preflight.parse_version
read_os_release = runner_preflight.read_os_release
version_at_least = runner_preflight.version_at_least


def test_numeric_versions_compare_without_lexical_errors():
    assert parse_version("24.04") == (24, 4)
    assert version_at_least("2.39", "2.38")
    assert version_at_least("24.4", "24.04.0")
    assert not version_at_least("2.9", "2.38")


def test_os_release_parser_handles_quotes_and_comments(tmp_path: Path):
    os_release = tmp_path / "os-release"
    os_release.write_text(
        "# comment\nID=\"ubuntu\"\nVERSION_ID='24.04'\nNAME=Ubuntu\n",
        encoding="utf-8",
    )

    assert read_os_release(os_release) == {
        "ID": "ubuntu",
        "VERSION_ID": "24.04",
        "NAME": "Ubuntu",
    }


def test_compatibility_reports_all_host_mismatches(monkeypatch):
    monkeypatch.setattr(runner_preflight.shutil, "which", lambda _command: None)

    errors = compatibility_errors(
        os_release={"ID": "ubuntu", "VERSION_ID": "22.04"},
        libc_name="glibc",
        glibc_version="2.35",
        required_os="ubuntu",
        minimum_os_version="24.04",
        minimum_glibc="2.38",
        required_commands=["docker"],
    )

    assert errors == [
        "OS version '22.04' is below '24.04'",
        "glibc '2.35' is below '2.38'",
        "required command 'docker' is unavailable",
    ]


def test_compatibility_accepts_a_prepared_noble_host(monkeypatch):
    monkeypatch.setattr(runner_preflight.shutil, "which", lambda command: f"/usr/bin/{command}")

    assert not compatibility_errors(
        os_release={"ID": "ubuntu", "VERSION_ID": "24.04"},
        libc_name="glibc",
        glibc_version="2.39",
        required_os="ubuntu",
        minimum_os_version="24.04",
        minimum_glibc="2.38",
        required_commands=["docker"],
    )


def test_glibc_requirement_rejects_a_different_libc():
    errors = compatibility_errors(
        os_release={"ID": "alpine", "VERSION_ID": "3.22"},
        libc_name="musl",
        glibc_version="2.39",
        required_os=None,
        minimum_os_version=None,
        minimum_glibc="2.38",
        required_commands=[],
    )

    assert errors == ["libc 'musl' does not satisfy the glibc requirement"]

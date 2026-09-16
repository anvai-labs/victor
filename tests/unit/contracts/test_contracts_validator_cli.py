"""Tests for the contracts validation CLI."""

from __future__ import annotations

from types import SimpleNamespace

from victor_contracts.cli import contracts_main, main
from victor_contracts.core.plugins import VictorPlugin
from victor_contracts.validation import validate_vertical_package
from victor_contracts.verticals.manifest import ExtensionManifest
from victor_contracts.verticals.protocols.base import VerticalBase


class _FakeEntryPoint:
    def __init__(self, name: str, obj):
        self.group = "victor.plugins"
        self.name = name
        self._obj = obj

    def load(self):
        return self._obj


class _ValidVertical(VerticalBase):
    name = "valid_vertical"
    description = "Valid vertical"
    version = "1.2.3"
    _victor_manifest = ExtensionManifest(
        name="valid_vertical",
        version="1.2.3",
        api_version=1,
        min_framework_version=">=0.1.0",
    )

    @classmethod
    def get_name(cls) -> str:
        return cls.name

    @classmethod
    def get_description(cls) -> str:
        return cls.description

    @classmethod
    def get_tools(cls):
        return ["read"]

    @classmethod
    def get_system_prompt(cls) -> str:
        return "valid prompt"


class _TooNewVertical(VerticalBase):
    name = "too_new_vertical"
    description = "Too new for installed core"
    version = "1.2.3"
    _victor_manifest = ExtensionManifest(
        name="too_new_vertical",
        version="1.2.3",
        api_version=1,
        min_framework_version=">=99.0.0",
    )

    @classmethod
    def get_name(cls) -> str:
        return cls.name

    @classmethod
    def get_description(cls) -> str:
        return cls.description

    @classmethod
    def get_tools(cls):
        return ["read"]

    @classmethod
    def get_system_prompt(cls) -> str:
        return "too new prompt"


class _PluginWrappingValidVertical(VictorPlugin):
    @property
    def name(self) -> str:
        return "valid-plugin"

    def register(self, context) -> None:
        context.register_vertical(_ValidVertical)

    def get_cli_app(self):
        return None

    def on_activate(self) -> None:
        return None

    def on_deactivate(self) -> None:
        return None

    async def on_activate_async(self) -> None:
        return None

    async def on_deactivate_async(self) -> None:
        return None

    def health_check(self) -> dict[str, object]:
        return {"healthy": True}


def _patch_distribution(monkeypatch, entry_points):
    fake_dist = SimpleNamespace(entry_points=entry_points, metadata={"Name": "victor-fake"})
    monkeypatch.setattr(
        "victor_contracts.validation.distribution",
        lambda package_name: fake_dist,
    )
    monkeypatch.setattr(
        "victor_contracts.validation.installed_version",
        lambda name: "0.6.1",
    )


def test_validate_vertical_package_accepts_valid_sdk_vertical(monkeypatch):
    """A package with a valid registered vertical should pass validation."""

    _patch_distribution(monkeypatch, [_FakeEntryPoint("valid", _ValidVertical)])

    report = validate_vertical_package("victor-fake")

    assert report.ok is True
    assert report.package_name == "victor-fake"
    assert report.verticals == ["valid_vertical"]
    assert report.issues == []


def test_validate_vertical_package_reports_framework_version_skew(monkeypatch):
    """Validator should fail packages whose manifests require a newer core."""

    _patch_distribution(monkeypatch, [_FakeEntryPoint("too_new", _TooNewVertical)])

    report = validate_vertical_package("victor-fake")

    assert report.ok is False
    assert any(issue.code == "framework_version_incompatible" for issue in report.issues)


def test_validate_vertical_package_accepts_valid_plugin_entry_point(monkeypatch):
    """Validator should share plugin parsing semantics with SDK discovery."""

    _patch_distribution(monkeypatch, [_FakeEntryPoint("valid", _PluginWrappingValidVertical)])

    report = validate_vertical_package("victor-fake")

    assert report.ok is True
    assert report.verticals == ["valid_vertical"]
    assert report.issues == []


def test_cli_returns_nonzero_for_invalid_package(monkeypatch, capsys):
    """The CLI should exit nonzero and print findings for invalid packages."""

    _patch_distribution(monkeypatch, [_FakeEntryPoint("too_new", _TooNewVertical)])

    exit_code = main(["check", "victor-fake"])
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "framework_version_incompatible" in output


def test_contracts_cli_uses_semantic_program_name(capsys):
    """The new CLI entry point should present itself as victor-contracts."""

    exit_code = contracts_main([])
    output = capsys.readouterr().out

    assert exit_code == 2
    assert "usage: victor-contracts" in output


def test_contract_checks_survive_optimized_python():
    """The public package validator must reject invalid definitions with -O."""
    import json
    from pathlib import Path
    import subprocess
    import sys

    source = Path(__file__).resolve()
    program = """
import json, runpy, sys
from types import SimpleNamespace
sys.path.insert(0, sys.argv[2])
ns = runpy.run_path(sys.argv[1])
valid = ns['_ValidVertical']
class Invalid(valid):
    @classmethod
    def get_tools(cls):
        return []
import victor_contracts.validation as validation
validation.installed_version = lambda name: '0.6.1'
results = []
for vertical in (valid, Invalid):
    entry = ns['_FakeEntryPoint']('candidate', vertical)
    validation.distribution = lambda name: SimpleNamespace(entry_points=[entry], metadata={'Name':'candidate'})
    report = validation.validate_vertical_package('candidate')
    results.append({'ok':report.ok, 'issues':[issue.code for issue in report.issues]})
print(json.dumps(results))
"""
    sdk = source.parents[3] / "victor-contracts"
    for flags in ([], ["-O"]):
        result = subprocess.run(
            [sys.executable, *flags, "-c", program, str(source), str(sdk)],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        valid, invalid = json.loads(result.stdout)
        assert valid["ok"] is True
        assert invalid["ok"] is False
        assert "protocol_drift" in invalid["issues"]

"""Integrity checks for the contract-only external vertical example."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_DIR = REPO_ROOT / "examples" / "external_vertical"
EXAMPLE_SRC_DIR = EXAMPLE_DIR / "src"
README_PATH = EXAMPLE_DIR / "README.md"
PYPROJECT_PATH = EXAMPLE_DIR / "pyproject.toml"
INIT_PATH = EXAMPLE_SRC_DIR / "victor_security" / "__init__.py"
ASSISTANT_PATH = EXAMPLE_SRC_DIR / "victor_security" / "assistant.py"
TOOLS_PATH = EXAMPLE_SRC_DIR / "victor_security" / "tools.py"
MODE_CONFIG_PATH = EXAMPLE_SRC_DIR / "victor_security" / "mode_config.py"
ROOT_VERTICAL_TOML_PATH = EXAMPLE_DIR / "victor-vertical.toml"
SRC_VERTICAL_TOML_PATH = EXAMPLE_SRC_DIR / "victor_security" / "victor-vertical.toml"


def _example_project_data() -> dict[str, object]:
    """Return parsed example package metadata."""

    return tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))


def test_external_vertical_example_metadata_matches_security_assistant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Package metadata, entry points, and the example assistant should stay aligned."""

    project_data = _example_project_data()
    project = project_data["project"]
    entry_points = project["entry-points"]["victor.plugins"]
    mode_config_entry_points = project["entry-points"]["victor.mode_configs"]

    monkeypatch.syspath_prepend(str(EXAMPLE_SRC_DIR))
    from victor_security import SecurityAssistant, plugin

    definition = SecurityAssistant.get_definition()

    assert project["name"] == "victor-security"
    assert project["version"] == SecurityAssistant.version
    assert "victor-contracts>=0.7.0" in project["dependencies"]
    assert "victor-ai>=0.3.0" in project["optional-dependencies"]["runtime"]
    assert entry_points["security"] == "victor_security:plugin"
    assert mode_config_entry_points["security"] == (
        "victor_security.mode_config:SecurityModeConfigProvider"
    )
    assert plugin.name == "security"

    assert SecurityAssistant.get_name() == "security"
    assert definition.name == "security"
    assert definition.workflow_metadata.initial_stage == "reconnaissance"
    assert definition.workflow_metadata.workflow_spec == {
        "stage_order": ["reconnaissance", "analysis", "reporting"]
    }
    assert definition.team_metadata.default_team == "security_review_team"


def test_external_vertical_readme_documents_current_install_and_entry_point_flow() -> None:
    """README examples should stay aligned with the package metadata contract."""

    readme = README_PATH.read_text(encoding="utf-8")

    required_snippets = [
        "pip install -e .",
        'pip install -e ".[runtime]"',
        'pip install -e ".[test]"',
        'security = "victor_security:plugin"',
        'security = "victor_security.mode_config:SecurityModeConfigProvider"',
        "get_definition()",
        "`victor-contracts`",
        "`victor_contracts`",
        "`victor-ai`",
        "SecurityAssistant",
        "SecretPatternScanTool",
        "SecurityModeConfigProvider",
        "secret_pattern_scan",
        "context.register_tool(SecretPatternScanTool())",
        "victor-contracts check victor-security",
        "Contract-only package dependency for authoring",
        "`victor.plugins`",
        "`victor.mode_configs`",
        "VerticalLoader",
        "## Running The Example's Tests",
        "pytest",
    ]

    missing = sorted(snippet for snippet in required_snippets if snippet not in readme)
    assert not missing, f"External vertical README is missing required snippets: {missing}"


def test_external_vertical_example_uses_contract_import_namespace() -> None:
    """The example source should prefer victor_contracts over victor_sdk."""

    init_source = INIT_PATH.read_text(encoding="utf-8")
    assistant_source = ASSISTANT_PATH.read_text(encoding="utf-8")
    tools_source = TOOLS_PATH.read_text(encoding="utf-8")
    mode_config_source = MODE_CONFIG_PATH.read_text(encoding="utf-8")

    assert "from victor_contracts import PluginContext, VictorPlugin" in init_source
    assert "from victor_contracts import (" in assistant_source
    assert "from victor_contracts.verticals.mode_config import (" in mode_config_source
    assert "victor_sdk" not in init_source
    assert "victor_sdk" not in assistant_source
    assert "victor_sdk" not in tools_source
    assert "victor_sdk" not in mode_config_source
    # The custom tool must stay authoring-pure: no runtime or SDK imports.
    assert "import victor" not in tools_source
    assert "from victor" not in tools_source


def test_external_vertical_plugin_registers_custom_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SecurityPlugin.register() should register the vertical and the custom tool."""

    monkeypatch.syspath_prepend(str(EXAMPLE_SRC_DIR))
    from victor_security import SecurityAssistant, SecurityPlugin

    registered_tools: list[object] = []
    registered_verticals: list[object] = []

    class _CapturingContext:
        def register_tool(self, tool_instance: object) -> None:
            registered_tools.append(tool_instance)

        def register_vertical(self, vertical_class: object) -> None:
            registered_verticals.append(vertical_class)

    SecurityPlugin().register(_CapturingContext())

    assert registered_verticals == [SecurityAssistant]
    assert [getattr(tool, "name", None) for tool in registered_tools] == ["secret_pattern_scan"]


def test_external_vertical_toml_copies_are_byte_identical() -> None:
    """The root victor-vertical.toml must mirror the canonical src copy exactly."""

    src_bytes = SRC_VERTICAL_TOML_PATH.read_bytes()
    root_bytes = ROOT_VERTICAL_TOML_PATH.read_bytes()

    assert src_bytes == root_bytes, (
        "examples/external_vertical/victor-vertical.toml has drifted from the "
        "canonical src/victor_security/victor-vertical.toml copy"
    )


def test_external_vertical_toml_validates_against_package_schema() -> None:
    """The canonical victor-vertical.toml must parse via VerticalPackageMetadata."""

    from victor.core.verticals.package_schema import VerticalPackageMetadata

    metadata = VerticalPackageMetadata.from_toml(SRC_VERTICAL_TOML_PATH)
    project = _example_project_data()["project"]

    assert metadata.name == "security"
    assert metadata.version == project["version"]
    assert metadata.class_spec.module == "victor_security.assistant"
    assert metadata.class_spec.class_name == "SecurityAssistant"
    assert "secret_pattern_scan" in metadata.class_spec.provides_tools

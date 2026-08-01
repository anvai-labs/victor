from pathlib import Path
from unittest.mock import Mock

import tomllib

from victor_contracts import VictorPlugin, VerticalBase

from victor_dataanalysis.assistant import DataAnalysisAssistant
from victor_dataanalysis.plugin import DataAnalysisPlugin, plugin

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _entry_points() -> dict:
    pyproject = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return pyproject["project"]["entry-points"]


def test_pyproject_registers_plugin_instance_entry_point() -> None:
    entry_points = _entry_points()

    assert entry_points["victor.plugins"]["dataanalysis"] == "victor_dataanalysis.plugin:plugin"


def test_pyproject_registers_runtime_extension_entry_points() -> None:
    entry_points = _entry_points()

    assert entry_points["victor.tool_dependencies"]["dataanalysis"] == (
        "victor_dataanalysis.tool_dependencies:get_provider"
    )
    assert entry_points["victor.safety_rules"]["dataanalysis"] == (
        "victor_dataanalysis.safety:create_all_dataanalysis_safety_rules"
    )
    assert entry_points["victor.framework.teams.providers"]["dataanalysis"] == (
        "victor_dataanalysis.teams:DataAnalysisTeamSpecProvider"
    )


def test_pyproject_registers_contract_extension_entry_points() -> None:
    entry_points = _entry_points()

    assert "victor.sdk.protocols" not in entry_points
    assert entry_points["victor.extension.protocols"] == {
        "dataanalysis-tools": "victor_dataanalysis.protocols:DataAnalysisToolProvider",
        "dataanalysis-safety": "victor_dataanalysis.protocols:DataAnalysisSafetyProvider",
        "dataanalysis-prompts": "victor_dataanalysis.protocols:DataAnalysisPromptProvider",
        "dataanalysis-workflows": "victor_dataanalysis.protocols:DataAnalysisWorkflowProvider",
    }


def test_pyproject_keeps_contracts_in_base_dependencies_without_victor_runtime() -> None:
    project = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]

    assert any(dependency.startswith("victor-contracts") for dependency in project["dependencies"])
    assert all("victor-ai" not in dependency for dependency in project["dependencies"])


def test_plugin_implements_protocol_and_registers_vertical() -> None:
    context = Mock()

    assert isinstance(plugin, VictorPlugin)
    assert isinstance(plugin, DataAnalysisPlugin)
    assert plugin.name == "dataanalysis"

    plugin.register(context)

    context.register_vertical.assert_called_once_with(DataAnalysisAssistant)


def test_plugin_health_check_reports_vertical() -> None:
    health = plugin.health_check()

    assert health["healthy"] is True
    assert health["vertical"] == "dataanalysis"
    assert health["vertical_class"] == "DataAnalysisAssistant"


def test_assistant_inherits_contract_vertical_base() -> None:
    assert issubclass(DataAnalysisAssistant, VerticalBase)

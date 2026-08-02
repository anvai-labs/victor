# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the scaffold command."""

import pytest
import tempfile
from pathlib import Path
from typer.testing import CliRunner

from victor.ui.commands.scaffold import (
    scaffold_app,
    validate_vertical_name,
    to_class_name,
    to_title,
    get_template_dir,
)


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_validate_vertical_name_valid(self):
        """Test valid vertical names."""
        assert validate_vertical_name("security") == "security"
        assert validate_vertical_name("data_analysis") == "data_analysis"
        assert validate_vertical_name("ml") == "ml"
        assert validate_vertical_name("Security") == "security"  # Lowercased

    def test_validate_vertical_name_invalid(self):
        """Test invalid vertical names."""
        import typer

        with pytest.raises(typer.BadParameter):
            validate_vertical_name("")

        with pytest.raises(typer.BadParameter):
            validate_vertical_name("123abc")  # Starts with number

        with pytest.raises(typer.BadParameter):
            validate_vertical_name("my-vertical")  # Contains hyphen

        with pytest.raises(typer.BadParameter):
            validate_vertical_name("victor")  # Reserved name

    def test_to_class_name(self):
        """Test class name conversion."""
        assert to_class_name("security") == "Security"
        assert to_class_name("data_analysis") == "DataAnalysis"
        assert to_class_name("ml") == "Ml"

    def test_to_title(self):
        """Test title conversion."""
        assert to_title("security") == "Security"
        assert to_title("data_analysis") == "Data Analysis"
        assert to_title("ml") == "Ml"

    def test_get_template_dir(self):
        """Test template directory exists."""
        template_dir = get_template_dir()
        assert template_dir.exists()
        assert (template_dir / "__init__.py.j2").exists()
        assert (template_dir / "assistant.py.j2").exists()
        assert (template_dir / "safety.py.j2").exists()
        assert (template_dir / "prompts.py.j2").exists()
        assert (template_dir / "mode_config.py.j2").exists()
        assert (template_dir / "service_provider.py.j2").exists()


class TestScaffoldCommand:
    """Tests for the scaffold command."""

    @pytest.fixture
    def runner(self):
        """Create a CLI runner."""
        return CliRunner()

    def test_dry_run(self, runner):
        """Test dry run mode shows what would be created."""
        result = runner.invoke(
            scaffold_app,
            ["create", "test_vertical", "--description", "Test vertical", "--dry-run"],
        )
        assert result.exit_code == 0
        assert "Dry run mode" in result.output
        assert "Would create" in result.output

    def test_reserved_name(self, runner):
        """Test reserved names are rejected."""
        result = runner.invoke(
            scaffold_app,
            ["create", "victor", "--dry-run"],
        )
        assert result.exit_code == 1
        assert "reserved name" in result.output

    def test_invalid_name(self, runner):
        """Test invalid names are rejected."""
        result = runner.invoke(
            scaffold_app,
            ["create", "my-vertical", "--dry-run"],
        )
        assert result.exit_code == 1
        assert "Invalid vertical name" in result.output

    def test_service_provider_flag(self, runner):
        """Test service provider flag adds service_provider.py."""
        result = runner.invoke(
            scaffold_app,
            ["create", "test_vertical", "--service-provider", "--dry-run"],
        )
        assert result.exit_code == 0
        # With service provider, should show 6 files
        assert "service_provider.py" in result.output


class TestTemplateRendering:
    """Tests for template rendering."""

    def test_init_template(self):
        """Test __init__.py template renders correctly."""
        from jinja2 import Environment, FileSystemLoader

        template_dir = get_template_dir()
        env = Environment(loader=FileSystemLoader(str(template_dir)))

        context = {
            "name": "security",
            "name_class": "Security",
            "name_title": "Security",
            "name_upper": "SECURITY",
            "description": "Security analysis assistant",
        }

        template = env.get_template("__init__.py.j2")
        content = template.render(**context)

        assert "Copyright 2025" in content
        assert "SecurityAssistant" in content
        assert "contract-first definition layer" in content
        assert "victor_contracts" in content
        assert "victor" "_sdk" not in content
        assert "SecuritySafetyExtension" not in content
        assert "SecurityPromptContributor" not in content
        assert "SecurityModeConfigProvider" not in content

    def test_assistant_template(self):
        """Test assistant.py template renders correctly."""
        from jinja2 import Environment, FileSystemLoader

        template_dir = get_template_dir()
        env = Environment(loader=FileSystemLoader(str(template_dir)))

        context = {
            "name": "security",
            "name_class": "Security",
            "name_title": "Security",
            "name_upper": "SECURITY",
            "description": "Security analysis assistant",
        }

        template = env.get_template("assistant.py.j2")
        content = template.render(**context)

        assert "class SecurityAssistant(VerticalBase):" in content
        assert 'name = "security"' in content
        assert "from victor_contracts import (" in content
        assert "victor" "_sdk" not in content
        assert "CapabilityRequirement" in content
        assert "ToolRequirement" in content
        assert "get_definition()" not in content  # scaffolded via VerticalBase

    def test_prompts_template_uses_contract_import_namespace(self):
        """Test prompts.py template imports contract symbols from victor_contracts."""
        from jinja2 import Environment, FileSystemLoader

        template_dir = get_template_dir()
        env = Environment(loader=FileSystemLoader(str(template_dir)))

        context = {
            "name": "security",
            "name_class": "Security",
            "name_title": "Security",
            "name_upper": "SECURITY",
            "description": "Security analysis assistant",
        }

        template = env.get_template("prompts.py.j2")
        content = template.render(**context)

        assert "from victor_contracts import ToolNames" in content
        assert "victor" "_sdk" not in content

    def test_plugin_init_help_uses_contract_language(self):
        """Plugin scaffold help should match the generated contract-first template."""
        from victor.ui.commands.plugin import init_plugin

        help_text = init_plugin.__doc__

        assert help_text is not None
        assert "contract-first definition layer" in help_text
        assert "victor_contracts" in help_text
        assert "SDK" "-first" not in help_text
        assert "victor-contracts" not in help_text

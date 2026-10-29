"""Tests for victor skill CLI commands.

Covers:
- victor skill list
- victor skill list --category coding
- victor skill info <name>
- victor skill search <query>
- victor skill create <name>
"""

from __future__ import annotations

import os
import tempfile
from unittest.mock import patch, MagicMock

import yaml
from typer.testing import CliRunner

from victor_contracts.skills import SkillDefinition

runner = CliRunner()


def _make_skill(name: str = "debug", category: str = "coding", **kwargs):
    return SkillDefinition(
        name=name,
        description=kwargs.get("description", f"Skill: {name}"),
        category=category,
        prompt_fragment=kwargs.get("prompt_fragment", f"Prompt for {name}."),
        required_tools=kwargs.get("required_tools", ["read_file"]),
        optional_tools=kwargs.get("optional_tools", []),
        tags=kwargs.get("tags", frozenset()),
    )


def _get_app():
    """Import the skills CLI app."""
    from victor.ui.commands.skills import skills_app

    return skills_app


class TestSkillList:
    """victor skill list command."""

    def test_list_empty(self):
        app = _get_app()
        with patch(
            "victor.ui.commands.skills._build_registry",
            return_value=MagicMock(list_all=MagicMock(return_value=[])),
        ):
            result = runner.invoke(app, ["list"])
            assert result.exit_code == 0
            assert "No skills" in result.output

    def test_list_shows_skills(self):
        app = _get_app()
        skills = [
            _make_skill("debug", category="coding", description="Debug a test"),
            _make_skill("deploy", category="devops", description="Deploy service"),
        ]
        mock_registry = MagicMock(list_all=MagicMock(return_value=skills))
        with patch(
            "victor.ui.commands.skills._build_registry",
            return_value=mock_registry,
        ):
            result = runner.invoke(app, ["list"])
            assert result.exit_code == 0
            assert "debug" in result.output
            assert "deploy" in result.output

    def test_list_filter_by_category(self):
        app = _get_app()
        skills = [
            _make_skill("debug", category="coding"),
        ]
        mock_registry = MagicMock(
            search=MagicMock(return_value=skills),
        )
        with patch(
            "victor.ui.commands.skills._build_registry",
            return_value=mock_registry,
        ):
            result = runner.invoke(app, ["list", "--category", "coding"])
            assert result.exit_code == 0
            assert "debug" in result.output
            mock_registry.search.assert_called_once_with(category="coding")


class TestSkillInfo:
    """victor skill info <name> command."""

    def test_info_found(self):
        app = _get_app()
        skill = _make_skill(
            "debug",
            description="Debug a failing test",
            required_tools=["read_file", "grep"],
            optional_tools=["code_search"],
            tags=frozenset({"testing", "debugging"}),
        )
        mock_registry = MagicMock(get_optional=MagicMock(return_value=skill))
        with patch(
            "victor.ui.commands.skills._build_registry",
            return_value=mock_registry,
        ):
            result = runner.invoke(app, ["info", "debug"])
            assert result.exit_code == 0
            assert "debug" in result.output
            assert "Debug a failing test" in result.output
            assert "read_file" in result.output

    def test_info_not_found(self):
        app = _get_app()
        mock_registry = MagicMock(get_optional=MagicMock(return_value=None))
        with patch(
            "victor.ui.commands.skills._build_registry",
            return_value=mock_registry,
        ):
            result = runner.invoke(app, ["info", "nonexistent"])
            assert result.exit_code == 0
            assert "not found" in result.output.lower()


class TestSkillSearch:
    """victor skill search <query> command."""

    def test_search_finds_results(self):
        app = _get_app()
        skills = [_make_skill("debug_test", description="Debug a failing test")]
        mock_registry = MagicMock(search=MagicMock(return_value=skills))
        with patch(
            "victor.ui.commands.skills._build_registry",
            return_value=mock_registry,
        ):
            result = runner.invoke(app, ["search", "debug"])
            assert result.exit_code == 0
            assert "debug_test" in result.output
            mock_registry.search.assert_called_once_with(query="debug")

    def test_search_no_results(self):
        app = _get_app()
        mock_registry = MagicMock(search=MagicMock(return_value=[]))
        with patch(
            "victor.ui.commands.skills._build_registry",
            return_value=mock_registry,
        ):
            result = runner.invoke(app, ["search", "nonexistent"])
            assert result.exit_code == 0
            assert "No skills" in result.output


class TestSkillCreate:
    """victor skill create command."""

    def test_create_writes_yaml(self):
        app = _get_app()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "victor.ui.commands.skills._get_skills_dir",
                return_value=tmpdir,
            ):
                result = runner.invoke(
                    app,
                    [
                        "create",
                        "my_skill",
                        "--description",
                        "My custom skill",
                        "--category",
                        "devops",
                        "--prompt",
                        "Do the devops thing.",
                        "--tools",
                        "read,grep,shell",
                    ],
                )
                assert result.exit_code == 0
                assert "Created" in result.output

                # Verify file was written
                path = os.path.join(tmpdir, "my_skill.yaml")
                assert os.path.exists(path)
                with open(path) as f:
                    data = yaml.safe_load(f)
                assert data["name"] == "my_skill"
                assert data["description"] == "My custom skill"
                assert data["category"] == "devops"
                assert data["required_tools"] == ["read", "grep", "shell"]

    def test_create_default_category(self):
        app = _get_app()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "victor.ui.commands.skills._get_skills_dir",
                return_value=tmpdir,
            ):
                result = runner.invoke(
                    app,
                    [
                        "create",
                        "simple",
                        "--description",
                        "Simple skill",
                        "--prompt",
                        "Do it.",
                        "--tools",
                        "read",
                    ],
                )
                assert result.exit_code == 0
                path = os.path.join(tmpdir, "simple.yaml")
                with open(path) as f:
                    data = yaml.safe_load(f)
                assert data["category"] == "custom"


class TestSkillRemove:
    """victor skill remove command."""

    def test_remove_existing(self):
        app = _get_app()
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a skill file first
            path = os.path.join(tmpdir, "old_skill.yaml")
            with open(path, "w") as f:
                yaml.dump({"name": "old_skill"}, f)

            with patch(
                "victor.ui.commands.skills._get_skills_dir",
                return_value=tmpdir,
            ):
                result = runner.invoke(app, ["remove", "old_skill"])
                assert result.exit_code == 0
                assert "Removed" in result.output
                assert not os.path.exists(path)

    def test_remove_nonexistent(self):
        app = _get_app()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "victor.ui.commands.skills._get_skills_dir",
                return_value=tmpdir,
            ):
                result = runner.invoke(app, ["remove", "nonexistent"])
                assert result.exit_code == 0
                assert "not found" in result.output.lower()

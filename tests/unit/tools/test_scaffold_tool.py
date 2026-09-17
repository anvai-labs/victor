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

"""Tests for scaffold_tool module."""

import pytest
import tempfile
from pathlib import Path

from victor.tools.scaffold_tool import scaffold, TEMPLATES


class TestScaffold:
    """Tests for scaffold function."""

    @pytest.mark.asyncio
    async def test_scaffold_list_templates(self):
        """Test listing available templates."""
        result = await scaffold(operation="list")
        assert result["success"] is True
        assert "templates" in result
        assert len(result["templates"]) > 0

    @pytest.mark.asyncio
    async def test_scaffold_invalid_operation(self):
        """Test invalid operation."""
        result = await scaffold(operation="invalid_op")
        assert result["success"] is False
        assert "Unknown operation" in result["error"]

    @pytest.mark.asyncio
    async def test_scaffold_create_requires_template(self):
        """Test create requires template."""
        result = await scaffold(operation="create")
        assert result["success"] is False
        # Should mention template or name

    @pytest.mark.asyncio
    async def test_scaffold_create_invalid_template(self):
        """Test create with invalid template."""
        result = await scaffold(operation="create", template="nonexistent_template")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_scaffold_create_fastapi(self):
        """Test creating FastAPI project."""
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            # Change to temp directory for project creation
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                result = await scaffold(operation="create", template="fastapi", name="test_project")
                assert result["success"] is True
                # Check project directory was created
                project_dir = Path(tmpdir) / "test_project"
                assert project_dir.exists()
            finally:
                os.chdir(old_cwd)

    @pytest.mark.asyncio
    async def test_scaffold_create_requires_name(self):
        """Test creating project requires name."""
        result = await scaffold(operation="create", template="fastapi")
        assert result["success"] is False
        assert "name" in result["error"].lower()


class TestTemplates:
    """Tests for TEMPLATES constant."""

    def test_templates_has_fastapi(self):
        """Test TEMPLATES includes FastAPI."""
        assert "fastapi" in TEMPLATES
        assert "name" in TEMPLATES["fastapi"]
        assert "files" in TEMPLATES["fastapi"]

    def test_templates_has_flask(self):
        """Test TEMPLATES includes Flask."""
        assert "flask" in TEMPLATES

    def test_templates_has_python_cli(self):
        """Test TEMPLATES includes python-cli."""
        assert "python-cli" in TEMPLATES

    def test_templates_have_required_fields(self):
        """Test all templates have required fields."""
        for name, template in TEMPLATES.items():
            assert "name" in template, f"{name} missing 'name'"
            assert "description" in template, f"{name} missing 'description'"
            assert "files" in template, f"{name} missing 'files'"

    def test_templates_has_python_feature(self):
        """Test TEMPLATES includes python_feature."""
        assert "python_feature" in TEMPLATES
        assert "name" in TEMPLATES["python_feature"]
        assert "files" in TEMPLATES["python_feature"]


class TestVariableInterpolation:
    """Tests for variable interpolation in scaffold tool."""

    @pytest.mark.asyncio
    async def test_from_template_with_variables(self):
        """Test from-template operation with variable interpolation."""
        import os
        import shutil

        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                variables = {
                    "feature_name": "User Authentication",
                    "feature_filename": "user_authentication.py",
                    "test_filename": "test_user_authentication.py",
                    "feature_module": "user_authentication",
                }

                result = await scaffold(
                    operation="from-template",
                    template="python_feature",
                    variables=variables,
                )

                assert result["success"] is True
                assert result["operation"] == "from-template"
                assert result["count"] == 2

                # Check files were created
                feature_file = Path("features/user_authentication.py")
                test_file = Path("tests/test_user_authentication.py")

                assert feature_file.exists()
                assert test_file.exists()

                # Check content interpolation
                feature_content = feature_file.read_text()
                assert "User Authentication" in feature_content
                assert "Hello from user_authentication!" in feature_content

                test_content = test_file.read_text()
                assert "test_user_authentication" in test_content

            finally:
                os.chdir(old_cwd)

    @pytest.mark.asyncio
    async def test_from_template_missing_variables(self):
        """Test from-template with missing variables."""
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                # Missing required variables
                result = await scaffold(
                    operation="from-template",
                    template="python_feature",
                    variables={"feature_name": "Test"},
                )

                assert result["success"] is False
                assert "Missing variable" in result["error"]

            finally:
                os.chdir(old_cwd)

    @pytest.mark.asyncio
    async def test_from_template_requires_template(self):
        """Test from-template requires template parameter."""
        result = await scaffold(
            operation="from-template",
            variables={"feature_name": "Test"},
        )

        assert result["success"] is False
        assert "template" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_from_template_requires_variables(self):
        """Test from-template requires variables parameter."""
        result = await scaffold(
            operation="from-template",
            template="python_feature",
        )

        assert result["success"] is False
        assert "variables" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_from_template_invalid_template(self):
        """Test from-template with invalid template."""
        result = await scaffold(
            operation="from-template",
            template="nonexistent_template",
            variables={"feature_name": "Test"},
        )

        assert result["success"] is False
        assert "Unknown template" in result["error"]

    @pytest.mark.asyncio
    async def test_create_with_variable_interpolation(self):
        """Test create operation with variable interpolation."""
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                variables = {
                    "feature_name": "Data Processor",
                    "feature_filename": "data_processor.py",
                    "test_filename": "test_data_processor.py",
                    "feature_module": "data_processor",
                }

                result = await scaffold(
                    operation="create",
                    template="python_feature",
                    name="my_feature",
                    variables=variables,
                )

                assert result["success"] is True
                assert "my_feature" in result["project_dir"]

                # Check files were created in the project directory
                feature_file = Path("my_feature/features/data_processor.py")
                test_file = Path("my_feature/tests/test_data_processor.py")

                assert feature_file.exists()
                assert test_file.exists()

                # Check content interpolation
                feature_content = feature_file.read_text()
                assert "Data Processor" in feature_content

            finally:
                os.chdir(old_cwd)

    @pytest.mark.asyncio
    async def test_variable_interpolation_in_paths(self):
        """Test that variables are interpolated in file paths."""
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                variables = {
                    "feature_name": "API Handler",
                    "feature_filename": "api_handler.py",
                    "test_filename": "test_api_handler.py",
                    "feature_module": "api_handler",
                }

                result = await scaffold(
                    operation="from-template",
                    template="python_feature",
                    variables=variables,
                )

                assert result["success"] is True
                assert "features/api_handler.py" in result["files_created"]
                assert "tests/test_api_handler.py" in result["files_created"]

            finally:
                os.chdir(old_cwd)

    @pytest.mark.asyncio
    async def test_variable_interpolation_in_content(self):
        """Test that variables are interpolated in file content."""
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                variables = {
                    "feature_name": "OAuth2 Provider",
                    "feature_filename": "oauth2_provider.py",
                    "test_filename": "test_oauth2_provider.py",
                    "feature_module": "oauth2_provider",
                }

                result = await scaffold(
                    operation="from-template",
                    template="python_feature",
                    variables=variables,
                )

                assert result["success"] is True

                # Check feature file content
                feature_file = Path("features/oauth2_provider.py")
                content = feature_file.read_text()

                assert "OAuth2 Provider" in content
                assert "# TODO: Implement OAuth2 Provider" in content
                assert "def main():" in content
                assert 'print("Hello from oauth2_provider!")' in content

                # Check test file content
                test_file = Path("tests/test_oauth2_provider.py")
                test_content = test_file.read_text()

                assert "Tests for OAuth2 Provider" in test_content
                assert "test_oauth2_provider" in test_content

            finally:
                os.chdir(old_cwd)


class TestScaffoldInitGitTimeout:
    """Co-design review item 15: init-git previously had NO timeout at all
    on any of its three git invocations — a hook or a locked filesystem
    could hang the tool call forever. Now routed through run_managed_process
    with a bounded timeout."""

    @pytest.mark.asyncio
    async def test_init_git_reports_timeout_without_hanging(self, monkeypatch):
        async def _fake_run_managed_process(*, argv, timeout):
            return (b"", b"", -1, True, False)  # timed_out=True

        monkeypatch.setattr(
            "victor.tools.subprocess_executor.run_managed_process",
            _fake_run_managed_process,
        )

        result = await scaffold(operation="init-git")

        assert result["success"] is False
        assert "timed out" in result["error"].lower()
        assert "git init" in result["error"]

    @pytest.mark.asyncio
    async def test_init_git_reports_failure_with_stderr(self, monkeypatch):
        calls = []

        async def _fake_run_managed_process(*, argv, timeout):
            calls.append(argv)
            if argv == ["git", "init"]:
                return (b"", b"fatal: not a git repository", 128, False, False)
            return (b"", b"", 0, False, False)

        monkeypatch.setattr(
            "victor.tools.subprocess_executor.run_managed_process",
            _fake_run_managed_process,
        )

        result = await scaffold(operation="init-git")

        assert result["success"] is False
        assert "fatal: not a git repository" in result["error"]
        # Only the first (failing) step ran — add/commit never dispatched.
        assert calls == [["git", "init"]]

    @pytest.mark.asyncio
    async def test_init_git_runs_all_three_steps_on_success(self, monkeypatch):
        calls = []

        async def _fake_run_managed_process(*, argv, timeout):
            calls.append(argv)
            return (b"", b"", 0, False, False)

        monkeypatch.setattr(
            "victor.tools.subprocess_executor.run_managed_process",
            _fake_run_managed_process,
        )

        result = await scaffold(operation="init-git")

        assert result["success"] is True
        assert calls == [
            ["git", "init"],
            ["git", "add", "."],
            ["git", "commit", "-m", "Initial commit"],
        ]

    @pytest.mark.asyncio
    async def test_init_git_nothing_to_commit_is_not_a_failure(self, monkeypatch):
        """git commit failing with 'nothing to commit' (e.g. init-git run on
        an already-committed or still-empty directory) must not be reported
        as a tool failure — the repo is still correctly initialized."""
        calls = []

        async def _fake_run_managed_process(*, argv, timeout):
            calls.append(argv)
            if argv == ["git", "commit", "-m", "Initial commit"]:
                return (b"", b"nothing to commit, working tree clean", 1, False, False)
            return (b"", b"", 0, False, False)

        monkeypatch.setattr(
            "victor.tools.subprocess_executor.run_managed_process",
            _fake_run_managed_process,
        )

        result = await scaffold(operation="init-git")

        assert result["success"] is True
        assert calls == [
            ["git", "init"],
            ["git", "add", "."],
            ["git", "commit", "-m", "Initial commit"],
        ]

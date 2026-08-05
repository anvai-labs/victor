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

"""Tests for environment setup utilities."""

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

import pytest

from victor.evaluation.env_setup import (
    SetupStrategy,
    SetupResult,
    EnvironmentConfig,
    EnvironmentSetup,
    validate_environment,
    get_python_requirements,
    get_node_dependencies,
    quick_setup,
)
from victor.evaluation.test_runners import Language


class TestSetupStrategy:
    """Tests for SetupStrategy enum."""

    def test_strategy_values(self):
        """Test all setup strategy values."""
        assert SetupStrategy.VIRTUALENV.value == "virtualenv"
        assert SetupStrategy.CONDA.value == "conda"
        assert SetupStrategy.SYSTEM.value == "system"
        assert SetupStrategy.DOCKER.value == "docker"
        assert SetupStrategy.NPM.value == "npm"
        assert SetupStrategy.YARN.value == "yarn"
        assert SetupStrategy.CARGO.value == "cargo"
        assert SetupStrategy.GO_MOD.value == "go_mod"


class TestSetupResult:
    """Tests for SetupResult dataclass."""

    def test_default_values(self):
        """Test default result values."""
        result = SetupResult(success=True)
        assert result.success is True
        assert result.language == Language.UNKNOWN
        assert result.strategy == SetupStrategy.SYSTEM
        assert result.python_version == ""
        assert result.error_message == ""
        assert result.env_vars == {}
        assert result.installed_packages == []
        assert result.setup_duration_seconds == 0.0

    def test_success_result(self):
        """Test successful result."""
        result = SetupResult(
            success=True,
            language=Language.PYTHON,
            strategy=SetupStrategy.VIRTUALENV,
            python_version="Python 3.11.5",
            install_output="Successfully installed packages",
            setup_duration_seconds=5.5,
        )
        assert result.success is True
        assert result.language == Language.PYTHON
        assert "3.11" in result.python_version

    def test_failure_result(self):
        """Test failure result."""
        result = SetupResult(
            success=False,
            language=Language.PYTHON,
            error_message="pip install failed",
        )
        assert result.success is False
        assert result.error_message == "pip install failed"


class TestEnvironmentConfig:
    """Tests for EnvironmentConfig dataclass."""

    def test_default_values(self):
        """Test default config values."""
        config = EnvironmentConfig()
        assert config.use_virtualenv is True
        assert config.python_version == ""
        assert config.install_dev_deps is True
        assert config.timeout_seconds == 600
        assert config.cache_dir is None
        assert config.docker_image == "python:3.11"
        assert config.verbose is False

    def test_custom_config(self):
        """Test custom config values."""
        config = EnvironmentConfig(
            use_virtualenv=False,
            python_version="3.10",
            timeout_seconds=300,
            verbose=True,
        )
        assert config.use_virtualenv is False
        assert config.python_version == "3.10"
        assert config.timeout_seconds == 300
        assert config.verbose is True


class TestEnvironmentSetup:
    """Tests for EnvironmentSetup class."""

    def test_init_default(self):
        """Test default initialization."""
        setup = EnvironmentSetup()
        assert setup.config is not None
        assert setup.config.use_virtualenv is True

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = EnvironmentConfig(timeout_seconds=120)
        setup = EnvironmentSetup(config)
        assert setup.config.timeout_seconds == 120

    @pytest.mark.asyncio
    async def test_setup_unsupported_language(self):
        """Test setup for unsupported language."""
        setup = EnvironmentSetup()

        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            # Create an empty directory (unknown language)
            result = await setup.setup_environment(project_dir)
            # Result depends on detect_language, but should complete
            assert isinstance(result, SetupResult)

    @pytest.mark.asyncio
    async def test_setup_python_no_files(self):
        """Test Python setup with no config files."""
        setup = EnvironmentSetup()
        config = EnvironmentConfig(timeout_seconds=30)

        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            # Create a .py file to be detected as Python
            (project_dir / "main.py").write_text("print('hello')")

            # Mock detect_language to ensure Python is detected
            with patch("victor.evaluation.env_setup.detect_language") as mock_detect:
                mock_detect.return_value = Language.PYTHON

                # Mock the subprocess to avoid actual installation
                with patch(
                    "victor.evaluation.env_setup.asyncio.create_subprocess_shell"
                ) as mock_proc:
                    mock_process = AsyncMock()
                    mock_process.returncode = 0
                    mock_process.communicate = AsyncMock(return_value=(b"OK", b""))
                    mock_proc.return_value = mock_process

                    result = await setup.setup_environment(project_dir, config)
                    # Should complete (may succeed or fail depending on system)
                    assert isinstance(result, SetupResult)
                    assert result.language == Language.PYTHON

    @pytest.mark.asyncio
    async def test_setup_python_with_requirements(self):
        """Test Python setup with requirements.txt."""
        setup = EnvironmentSetup()
        config = EnvironmentConfig(timeout_seconds=30)

        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            (project_dir / "main.py").write_text("print('hello')")
            (project_dir / "requirements.txt").write_text("pytest>=7.0.0")

            # Mock the subprocess to avoid actual installation
            with patch("asyncio.create_subprocess_shell") as mock_proc:
                mock_process = AsyncMock()
                mock_process.returncode = 0
                mock_process.communicate = AsyncMock(return_value=(b"OK", b""))
                mock_proc.return_value = mock_process

                result = await setup.setup_environment(project_dir, config)
                assert result.language == Language.PYTHON

    @pytest.mark.asyncio
    async def test_setup_python_timeout(self):
        """Test Python setup with timeout."""
        setup = EnvironmentSetup()
        config = EnvironmentConfig(timeout_seconds=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            (project_dir / "main.py").write_text("print('hello')")
            (project_dir / "requirements.txt").write_text("pytest")

            # Mock subprocess that times out
            with patch("asyncio.create_subprocess_shell") as mock_proc:
                mock_process = AsyncMock()
                mock_process.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
                mock_proc.return_value = mock_process

                result = await setup.setup_environment(project_dir, config)
                assert result.success is False
                assert "timed out" in result.error_message

    @pytest.mark.asyncio
    async def test_setup_node_with_package_json(self):
        """Test Node.js setup with package.json."""
        setup = EnvironmentSetup()

        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            package = {"name": "test", "dependencies": {"lodash": "^4.0.0"}}
            (project_dir / "package.json").write_text(json.dumps(package))

            # Mock npm/yarn availability and subprocess
            with patch("shutil.which") as mock_which:
                mock_which.side_effect = lambda cmd: ("/usr/bin/npm" if cmd == "npm" else None)

                with patch("asyncio.create_subprocess_shell") as mock_proc:
                    mock_process = AsyncMock()
                    mock_process.returncode = 0
                    mock_process.communicate = AsyncMock(return_value=(b"OK", b""))
                    mock_proc.return_value = mock_process

                    with patch("subprocess.run") as mock_run:
                        mock_run.return_value = MagicMock(stdout="v18.0.0")

                        result = await setup.setup_environment(project_dir)
                        assert result.language in (
                            Language.JAVASCRIPT,
                            Language.TYPESCRIPT,
                        )

    @pytest.mark.asyncio
    async def test_setup_go_project(self):
        """Test Go project setup."""
        setup = EnvironmentSetup()

        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            (project_dir / "go.mod").write_text("module test\n\ngo 1.21")
            (project_dir / "main.go").write_text("package main\n\nfunc main() {}")

            with patch("shutil.which") as mock_which:
                mock_which.return_value = "/usr/local/go/bin/go"

                with patch("asyncio.create_subprocess_shell") as mock_proc:
                    mock_process = AsyncMock()
                    mock_process.returncode = 0
                    mock_process.communicate = AsyncMock(return_value=(b"OK", b""))
                    mock_proc.return_value = mock_process

                    with patch("subprocess.run") as mock_run:
                        mock_run.return_value = MagicMock(stdout="go version go1.21")

                        result = await setup.setup_environment(project_dir)
                        assert result.language == Language.GO
                        assert result.strategy == SetupStrategy.GO_MOD

    @pytest.mark.asyncio
    async def test_setup_rust_project(self):
        """Test Rust project setup."""
        setup = EnvironmentSetup()

        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            (project_dir / "Cargo.toml").write_text('[package]\nname = "test"')
            (project_dir / "src").mkdir()
            (project_dir / "src" / "main.rs").write_text("fn main() {}")

            with patch("shutil.which") as mock_which:
                mock_which.return_value = "/usr/local/bin/cargo"

                with patch("asyncio.create_subprocess_shell") as mock_proc:
                    mock_process = AsyncMock()
                    mock_process.returncode = 0
                    mock_process.communicate = AsyncMock(return_value=(b"OK", b""))
                    mock_proc.return_value = mock_process

                    with patch("subprocess.run") as mock_run:
                        mock_run.return_value = MagicMock(stdout="rustc 1.70.0")

                        result = await setup.setup_environment(project_dir)
                        assert result.language == Language.RUST
                        assert result.strategy == SetupStrategy.CARGO


class TestValidateEnvironment:
    """Tests for validate_environment function."""

    def test_validate_python(self):
        """Test validating Python environment."""
        with patch("shutil.which") as mock_which:
            mock_which.side_effect = lambda cmd: (
                f"/usr/bin/{cmd}" if cmd in ["python", "pip", "pytest"] else None
            )

            result = validate_environment(Language.PYTHON)
            assert result["python"] is True
            assert result["pip"] is True
            assert result["pytest"] is True

    def test_validate_javascript(self):
        """Test validating JavaScript environment."""
        with patch("shutil.which") as mock_which:
            mock_which.side_effect = lambda cmd: (
                f"/usr/bin/{cmd}" if cmd in ["node", "npm"] else None
            )

            result = validate_environment(Language.JAVASCRIPT)
            assert result["node"] is True
            assert result["npm"] is True

    def test_validate_go(self):
        """Test validating Go environment."""
        with patch("shutil.which") as mock_which:
            mock_which.side_effect = lambda cmd: ("/usr/local/go/bin/go" if cmd == "go" else None)

            result = validate_environment(Language.GO)
            assert result["go"] is True

    def test_validate_rust(self):
        """Test validating Rust environment."""
        with patch("shutil.which") as mock_which:
            mock_which.side_effect = lambda cmd: (
                f"/usr/bin/{cmd}" if cmd in ["cargo", "rustc"] else None
            )

            result = validate_environment(Language.RUST)
            assert result["cargo"] is True
            assert result["rustc"] is True

    def test_validate_java(self):
        """Test validating Java environment."""
        with patch("shutil.which") as mock_which:
            mock_which.side_effect = lambda cmd: (
                f"/usr/bin/{cmd}" if cmd in ["java", "javac"] else None
            )

            result = validate_environment(Language.JAVA)
            assert result["java"] is True
            assert result["javac"] is True

    def test_validate_missing_tools(self):
        """Test validating environment with missing tools."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = None

            result = validate_environment(Language.PYTHON)
            assert result["python"] is False
            assert result["pip"] is False
            assert result["pytest"] is False

    def test_validate_unknown_language(self):
        """Test validating unknown language."""
        result = validate_environment(Language.UNKNOWN)
        assert result == {}


class TestGetPythonRequirements:
    """Tests for get_python_requirements function."""

    def test_requirements_txt(self):
        """Test parsing requirements.txt."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            (project_dir / "requirements.txt").write_text(
                "pytest>=7.0.0\n" "requests==2.28.0\n" "# comment\n" "-r other.txt\n" "flask\n"
            )

            reqs = get_python_requirements(project_dir)
            assert "pytest>=7.0.0" in reqs
            assert "requests==2.28.0" in reqs
            assert "flask" in reqs
            assert "# comment" not in reqs

    def test_pyproject_toml(self):
        """Test parsing pyproject.toml dependencies."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            (project_dir / "pyproject.toml").write_text(
                "[project]\n"
                'name = "test"\n'
                "dependencies = [\n"
                '    "requests>=2.0",\n'
                '    "click",\n'
                "]\n"
            )

            reqs = get_python_requirements(project_dir)
            assert "requests>=2.0" in reqs
            assert "click" in reqs

    def test_no_requirements(self):
        """Test project with no requirements files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            reqs = get_python_requirements(project_dir)
            assert reqs == []


class TestGetNodeDependencies:
    """Tests for get_node_dependencies function."""

    def test_package_json(self):
        """Test parsing package.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            package = {
                "name": "test",
                "dependencies": {"lodash": "^4.0.0"},
                "devDependencies": {"jest": "^29.0.0"},
            }
            (project_dir / "package.json").write_text(json.dumps(package))

            deps = get_node_dependencies(project_dir)
            assert deps["lodash"] == "^4.0.0"
            assert deps["jest"] == "^29.0.0"

    def test_no_package_json(self):
        """Test project with no package.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            deps = get_node_dependencies(project_dir)
            assert deps == {}

    def test_invalid_package_json(self):
        """Test invalid package.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            (project_dir / "package.json").write_text("invalid json {")

            deps = get_node_dependencies(project_dir)
            assert deps == {}


class TestQuickSetup:
    """Tests for quick_setup function."""

    @pytest.mark.asyncio
    async def test_quick_setup_success(self):
        """Test successful quick setup."""
        with patch.object(EnvironmentSetup, "setup_environment") as mock_setup:
            mock_setup.return_value = SetupResult(success=True)

            with tempfile.TemporaryDirectory() as tmpdir:
                result = await quick_setup(Path(tmpdir))
                assert result is True

    @pytest.mark.asyncio
    async def test_quick_setup_failure(self):
        """Test failed quick setup."""
        with patch.object(EnvironmentSetup, "setup_environment") as mock_setup:
            mock_setup.return_value = SetupResult(success=False, error_message="Setup failed")

            with tempfile.TemporaryDirectory() as tmpdir:
                result = await quick_setup(Path(tmpdir))
                assert result is False

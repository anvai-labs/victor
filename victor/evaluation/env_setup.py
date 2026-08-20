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

"""Environment setup utilities for agentic benchmarks.

This module provides functionality to set up project environments for
benchmark evaluation, including dependency installation, virtual environment
management, and environment validation.

Example usage:
    from victor.evaluation.env_setup import (
        EnvironmentSetup,
        SetupResult,
        detect_language,
    )

    setup = EnvironmentSetup()
    result = await setup.setup_environment(project_dir)

    if result.success:
        print(f"Environment ready: {result.python_version}")
    else:
        print(f"Setup failed: {result.error_message}")
"""

import asyncio
import json
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from victor.evaluation.test_runners import Language, detect_language

logger = logging.getLogger(__name__)


class SetupStrategy(Enum):
    """Environment setup strategy."""

    VIRTUALENV = "virtualenv"  # Use Python venv
    CONDA = "conda"  # Use conda environment
    SYSTEM = "system"  # Use system Python
    DOCKER = "docker"  # Use Docker container
    NPM = "npm"  # Use npm for Node.js
    YARN = "yarn"  # Use yarn for Node.js
    CARGO = "cargo"  # Use cargo for Rust
    GO_MOD = "go_mod"  # Use go modules


@dataclass
class SetupResult:
    """Result of environment setup."""

    success: bool
    language: Language = Language.UNKNOWN
    strategy: SetupStrategy = SetupStrategy.SYSTEM
    python_version: str = ""
    node_version: str = ""
    go_version: str = ""
    rust_version: str = ""
    java_version: str = ""
    install_output: str = ""
    error_message: str = ""
    env_vars: dict[str, str] = field(default_factory=dict)
    installed_packages: list[str] = field(default_factory=list)
    setup_duration_seconds: float = 0.0


@dataclass
class EnvironmentConfig:
    """Configuration for environment setup."""

    use_virtualenv: bool = True
    python_version: str = ""  # Empty = use system default
    node_version: str = ""
    install_dev_deps: bool = True
    timeout_seconds: int = 600
    cache_dir: Optional[Path] = None
    docker_image: str = "python:3.11"
    verbose: bool = False


class EnvironmentSetup:
    """Handles environment setup for benchmark projects."""

    def __init__(self, config: Optional[EnvironmentConfig] = None):
        self.config = config or EnvironmentConfig()

    async def setup_environment(
        self,
        project_dir: Path,
        config: Optional[EnvironmentConfig] = None,
    ) -> SetupResult:
        """Set up the environment for a project.

        Args:
            project_dir: Path to project directory
            config: Optional configuration override

        Returns:
            SetupResult with setup status and details
        """
        cfg = config or self.config
        language = detect_language(project_dir)

        start_time = asyncio.get_event_loop().time()

        try:
            if language == Language.PYTHON:
                result = await self._setup_python(project_dir, cfg)
            elif language in (Language.JAVASCRIPT, Language.TYPESCRIPT):
                result = await self._setup_node(project_dir, cfg)
            elif language == Language.GO:
                result = await self._setup_go(project_dir, cfg)
            elif language == Language.RUST:
                result = await self._setup_rust(project_dir, cfg)
            elif language == Language.JAVA:
                result = await self._setup_java(project_dir, cfg)
            else:
                result = SetupResult(
                    success=False,
                    language=language,
                    error_message=f"Unsupported language: {language.value}",
                )

            result.setup_duration_seconds = asyncio.get_event_loop().time() - start_time
            return result

        except Exception as e:
            logger.exception(f"Environment setup failed: {e}")
            return SetupResult(
                success=False,
                language=language,
                error_message=str(e),
                setup_duration_seconds=asyncio.get_event_loop().time() - start_time,
            )

    async def _setup_python(
        self,
        project_dir: Path,
        config: EnvironmentConfig,
    ) -> SetupResult:
        """Set up Python environment."""
        result = SetupResult(
            success=False,
            language=Language.PYTHON,
        )

        # Get Python version
        try:
            version_output = subprocess.run(
                [sys.executable, "--version"],
                capture_output=True,
                text=True,
            )
            result.python_version = version_output.stdout.strip()
        except Exception:
            result.python_version = f"Python {sys.version_info.major}.{sys.version_info.minor}"

        # Determine install command
        install_cmds = []

        # Check for pyproject.toml (modern Python)
        if (project_dir / "pyproject.toml").exists():
            install_cmds.append(f"{sys.executable} -m pip install -e .")
            if config.install_dev_deps:
                install_cmds.append(
                    f"{sys.executable} -m pip install -e '.[dev,test]' 2>/dev/null || true"
                )

        # Check for setup.py
        elif (project_dir / "setup.py").exists():
            install_cmds.append(f"{sys.executable} -m pip install -e .")
            if config.install_dev_deps:
                install_cmds.append(
                    f"{sys.executable} -m pip install -e '.[dev]' 2>/dev/null || true"
                )

        # Check for requirements.txt
        if (project_dir / "requirements.txt").exists():
            install_cmds.append(f"{sys.executable} -m pip install -r requirements.txt")

        # Check for requirements-dev.txt
        if config.install_dev_deps:
            for dev_req in [
                "requirements-dev.txt",
                "requirements_dev.txt",
                "dev-requirements.txt",
            ]:
                if (project_dir / dev_req).exists():
                    install_cmds.append(f"{sys.executable} -m pip install -r {dev_req}")
                    break

        # Also install pytest if not present
        install_cmds.append(
            f"{sys.executable} -m pip install pytest pytest-json-report 2>/dev/null || true"
        )

        if not install_cmds:
            result.success = True
            result.install_output = "No installation required"
            return result

        # Run installation
        combined_cmd = " && ".join(install_cmds)
        result.strategy = SetupStrategy.SYSTEM

        try:
            process = await asyncio.create_subprocess_shell(
                combined_cmd,
                cwd=str(project_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=config.timeout_seconds,
            )

            result.install_output = stdout.decode("utf-8", errors="replace")
            if stderr:
                result.install_output += "\n" + stderr.decode("utf-8", errors="replace")

            result.success = process.returncode == 0
            if not result.success:
                result.error_message = f"Installation failed with code {process.returncode}"

        except asyncio.TimeoutError:
            result.error_message = f"Installation timed out after {config.timeout_seconds}s"
        except Exception as e:
            result.error_message = str(e)

        return result

    async def _setup_node(
        self,
        project_dir: Path,
        config: EnvironmentConfig,
    ) -> SetupResult:
        """Set up Node.js environment."""
        result = SetupResult(
            success=False,
            language=Language.JAVASCRIPT,
        )

        # Check if npm or yarn is available
        npm_path = shutil.which("npm")
        yarn_path = shutil.which("yarn")

        if not npm_path and not yarn_path:
            result.error_message = "Neither npm nor yarn found"
            return result

        # Get Node version
        try:
            version_output = subprocess.run(
                ["node", "--version"],
                capture_output=True,
                text=True,
            )
            result.node_version = version_output.stdout.strip()
        except Exception:
            pass

        # Determine package manager preference
        use_yarn = False
        if (project_dir / "yarn.lock").exists():
            use_yarn = True
        elif (project_dir / "package-lock.json").exists():
            use_yarn = False
        elif yarn_path:
            use_yarn = True

        if use_yarn and yarn_path:
            install_cmd = "yarn install"
            result.strategy = SetupStrategy.YARN
        else:
            install_cmd = "npm install"
            result.strategy = SetupStrategy.NPM

        # Run installation
        try:
            process = await asyncio.create_subprocess_shell(
                install_cmd,
                cwd=str(project_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=config.timeout_seconds,
            )

            result.install_output = stdout.decode("utf-8", errors="replace")
            if stderr:
                result.install_output += "\n" + stderr.decode("utf-8", errors="replace")

            result.success = process.returncode == 0
            if not result.success:
                result.error_message = f"Installation failed with code {process.returncode}"

        except asyncio.TimeoutError:
            result.error_message = f"Installation timed out after {config.timeout_seconds}s"
        except Exception as e:
            result.error_message = str(e)

        return result

    async def _setup_go(
        self,
        project_dir: Path,
        config: EnvironmentConfig,
    ) -> SetupResult:
        """Set up Go environment."""
        result = SetupResult(
            success=False,
            language=Language.GO,
            strategy=SetupStrategy.GO_MOD,
        )

        go_path = shutil.which("go")
        if not go_path:
            result.error_message = "Go not found"
            return result

        # Get Go version
        try:
            version_output = subprocess.run(
                ["go", "version"],
                capture_output=True,
                text=True,
            )
            result.go_version = version_output.stdout.strip()
        except Exception:
            pass

        # Download dependencies
        try:
            process = await asyncio.create_subprocess_shell(
                "go mod download && go build ./...",
                cwd=str(project_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=config.timeout_seconds,
            )

            result.install_output = stdout.decode("utf-8", errors="replace")
            if stderr:
                result.install_output += "\n" + stderr.decode("utf-8", errors="replace")

            result.success = process.returncode == 0
            if not result.success:
                result.error_message = f"Go build failed with code {process.returncode}"

        except asyncio.TimeoutError:
            result.error_message = f"Go setup timed out after {config.timeout_seconds}s"
        except Exception as e:
            result.error_message = str(e)

        return result

    async def _setup_rust(
        self,
        project_dir: Path,
        config: EnvironmentConfig,
    ) -> SetupResult:
        """Set up Rust environment."""
        result = SetupResult(
            success=False,
            language=Language.RUST,
            strategy=SetupStrategy.CARGO,
        )

        cargo_path = shutil.which("cargo")
        if not cargo_path:
            result.error_message = "Cargo not found"
            return result

        # Get Rust version
        try:
            version_output = subprocess.run(
                ["rustc", "--version"],
                capture_output=True,
                text=True,
            )
            result.rust_version = version_output.stdout.strip()
        except Exception:
            pass

        # Build project
        try:
            process = await asyncio.create_subprocess_shell(
                "cargo build",
                cwd=str(project_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=config.timeout_seconds,
            )

            result.install_output = stdout.decode("utf-8", errors="replace")
            if stderr:
                result.install_output += "\n" + stderr.decode("utf-8", errors="replace")

            result.success = process.returncode == 0
            if not result.success:
                result.error_message = f"Cargo build failed with code {process.returncode}"

        except asyncio.TimeoutError:
            result.error_message = f"Rust build timed out after {config.timeout_seconds}s"
        except Exception as e:
            result.error_message = str(e)

        return result

    async def _setup_java(
        self,
        project_dir: Path,
        config: EnvironmentConfig,
    ) -> SetupResult:
        """Set up Java environment."""
        result = SetupResult(
            success=False,
            language=Language.JAVA,
        )

        # Check for Maven or Gradle
        maven_path = shutil.which("mvn")
        gradle_path = shutil.which("gradle")

        use_maven = (project_dir / "pom.xml").exists()
        use_gradle = (project_dir / "build.gradle").exists() or (
            project_dir / "build.gradle.kts"
        ).exists()

        if not use_maven and not use_gradle:
            result.error_message = "No pom.xml or build.gradle found"
            return result

        # Get Java version
        try:
            version_output = subprocess.run(
                ["java", "-version"],
                capture_output=True,
                text=True,
            )
            result.java_version = version_output.stderr.strip().split("\n")[0]
        except Exception:
            pass

        # Build command
        if use_maven:
            if not maven_path:
                result.error_message = "Maven not found"
                return result
            build_cmd = "mvn compile -DskipTests -B"
        else:
            if not gradle_path and not (project_dir / "gradlew").exists():
                result.error_message = "Gradle not found"
                return result
            wrapper = project_dir / "gradlew"
            gradle_cmd = str(wrapper) if wrapper.exists() else "gradle"
            build_cmd = f"{gradle_cmd} compileJava compileTestJava -x test"

        # Run build
        try:
            process = await asyncio.create_subprocess_shell(
                build_cmd,
                cwd=str(project_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=config.timeout_seconds,
            )

            result.install_output = stdout.decode("utf-8", errors="replace")
            if stderr:
                result.install_output += "\n" + stderr.decode("utf-8", errors="replace")

            result.success = process.returncode == 0
            if not result.success:
                result.error_message = f"Build failed with code {process.returncode}"

        except asyncio.TimeoutError:
            result.error_message = f"Java build timed out after {config.timeout_seconds}s"
        except Exception as e:
            result.error_message = str(e)

        return result


def validate_environment(language: Language) -> dict[str, bool]:
    """Check if required tools are available for a language.

    Args:
        language: Programming language to check

    Returns:
        Dictionary of tool names to availability status
    """
    checks: dict[Language, dict[str, str]] = {
        Language.PYTHON: {
            "python": "python",
            "pip": "pip",
            "pytest": "pytest",
        },
        Language.JAVASCRIPT: {
            "node": "node",
            "npm": "npm",
        },
        Language.TYPESCRIPT: {
            "node": "node",
            "npm": "npm",
            "tsc": "tsc",
        },
        Language.GO: {
            "go": "go",
        },
        Language.RUST: {
            "cargo": "cargo",
            "rustc": "rustc",
        },
        Language.JAVA: {
            "java": "java",
            "javac": "javac",
        },
    }

    tools = checks.get(language, {})
    return {name: shutil.which(cmd) is not None for name, cmd in tools.items()}


def get_python_requirements(project_dir: Path) -> list[str]:
    """Extract Python requirements from a project.

    Args:
        project_dir: Path to project directory

    Returns:
        List of requirement specifiers
    """
    requirements = []

    # Check requirements.txt
    req_file = project_dir / "requirements.txt"
    if req_file.exists():
        for line in req_file.read_text().strip().split("\n"):
            line = line.strip()
            if line and not line.startswith("#") and not line.startswith("-"):
                requirements.append(line)

    # Check pyproject.toml
    pyproject = project_dir / "pyproject.toml"
    if pyproject.exists():
        try:
            # Simple TOML parsing for dependencies
            content = pyproject.read_text()
            # Look for dependencies section
            if "dependencies" in content:
                import re

                deps_match = re.search(
                    r"dependencies\s*=\s*\[(.*?)\]",
                    content,
                    re.DOTALL,
                )
                if deps_match:
                    deps_str = deps_match.group(1)
                    for dep in re.findall(r'"([^"]+)"', deps_str):
                        requirements.append(dep)
        except Exception:
            pass

    return requirements


def get_node_dependencies(project_dir: Path) -> dict[str, str]:
    """Extract Node.js dependencies from package.json.

    Args:
        project_dir: Path to project directory

    Returns:
        Dictionary of package names to versions
    """
    package_json = project_dir / "package.json"
    if not package_json.exists():
        return {}

    try:
        with open(package_json) as f:
            pkg = json.load(f)

        deps = {}
        deps.update(pkg.get("dependencies", {}))
        deps.update(pkg.get("devDependencies", {}))
        return deps
    except (json.JSONDecodeError, KeyError):
        return {}


async def quick_setup(project_dir: Path) -> bool:
    """Quick setup helper that returns success/failure.

    Args:
        project_dir: Path to project directory

    Returns:
        True if setup succeeded
    """
    setup = EnvironmentSetup()
    result = await setup.setup_environment(project_dir)
    if not result.success:
        logger.warning(f"Environment setup failed: {result.error_message}")
    return result.success

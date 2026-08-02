# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Framework-level test runner detection.

Detects the appropriate test runner for a project based on its structure.
Supports pytest, django, and unittest. Used by:
- Benchmark harness (SWE-bench test validation)
- Any workflow that needs to run project tests

Usage:
    from victor.context.test_runner import detect_test_runner

    config = detect_test_runner(Path("/path/to/project"))
    print(config.runner_type)  # "django"
    print(config.command)      # ["python", "-m", "django", "test", ...]
    print(config.env)          # {"DJANGO_SETTINGS_MODULE": "..."}
"""

import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TestRunnerConfig:
    """Configuration for running a project's test suite."""

    command: List[str]
    env: Dict[str, str] = field(default_factory=dict)
    runner_type: str = "pytest"  # "pytest", "django", "unittest", "custom"


def detect_test_runner(
    project_root: Path,
    test_files: Optional[List[str]] = None,
) -> TestRunnerConfig:
    """Detect the appropriate test runner for a project.

    Detection order:
    1. Django (manage.py or django dependency)
    2. pytest (pytest.ini, conftest.py, pyproject.toml [tool.pytest])
    3. unittest (fallback)

    Args:
        project_root: Path to project root
        test_files: Optional specific test files to run

    Returns:
        TestRunnerConfig with command, env, and runner_type
    """
    python = sys.executable

    # 1. Check for Django
    django_settings = _detect_django(project_root)
    if django_settings:
        # Django source repo uses tests/runtests.py
        runtests = project_root / "tests" / "runtests.py"
        if runtests.exists():
            cmd = [python, str(runtests), "--verbosity=2"]
            if test_files:
                labels = _files_to_django_labels(test_files)
                cmd.extend(labels)
            env = {"DJANGO_SETTINGS_MODULE": django_settings}
            logger.info(
                "Detected Django source repo (runtests.py, settings=%s)",
                django_settings,
            )
        else:
            cmd = [python, "-m", "django", "test", "--verbosity=2"]
            if test_files:
                labels = _files_to_django_labels(test_files)
                cmd.extend(labels)
            env = {"DJANGO_SETTINGS_MODULE": django_settings}
            logger.info("Detected Django project (settings=%s)", django_settings)
        return TestRunnerConfig(command=cmd, env=env, runner_type="django")

    # 2. Check for pytest (Python)
    if _detect_pytest(project_root):
        cmd = [python, "-m", "pytest", "-xvs"]
        if test_files:
            cmd.extend(test_files)
        logger.info("Detected pytest project")
        return TestRunnerConfig(command=cmd, runner_type="pytest")

    # 3. Check for non-Python ecosystems
    non_python = _detect_non_python_runner(project_root, test_files)
    if non_python:
        return non_python

    # 4. Default to pytest (most common for Python)
    cmd = [python, "-m", "pytest", "-xvs"]
    if test_files:
        cmd.extend(test_files)
    return TestRunnerConfig(command=cmd, runner_type="pytest")


def _detect_django(project_root: Path) -> Optional[str]:
    """Detect Django project and return settings module.

    Returns:
        DJANGO_SETTINGS_MODULE string or None
    """
    manage_py = project_root / "manage.py"
    if manage_py.exists():
        # Parse manage.py for settings module
        try:
            content = manage_py.read_text(encoding="utf-8", errors="ignore")
            # Match: os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myapp.settings')
            match = re.search(
                r"DJANGO_SETTINGS_MODULE['\"],\s*['\"]([^'\"]+)['\"]",
                content,
            )
            if match:
                return match.group(1)
        except Exception:
            pass
        # Fallback: search for settings.py
        return _find_django_settings(project_root)

    # Check if this IS django itself (source checkout — no manage.py)
    # Django source has django/ package + tests/runtests.py
    runtests = project_root / "tests" / "runtests.py"
    django_pkg = project_root / "django" / "__init__.py"
    if runtests.exists() and django_pkg.exists():
        logger.info("Detected Django source repository (tests/runtests.py)")
        return "test_sqlite"  # Django's default test settings

    # Check if django is a dependency
    for config_file in ["setup.py", "setup.cfg", "pyproject.toml"]:
        cfg = project_root / config_file
        if cfg.exists():
            try:
                text = cfg.read_text(encoding="utf-8", errors="ignore")
                if "django" in text.lower() and "install_requires" in text.lower():
                    return _find_django_settings(project_root)
            except Exception:
                pass

    return None


def _find_django_settings(project_root: Path) -> Optional[str]:
    """Search for Django settings module in common locations."""
    # Check common patterns
    for pattern in [
        "*/settings.py",
        "*/settings/*.py",
        "settings.py",
    ]:
        matches = list(project_root.glob(pattern))
        for match in matches:
            if match.name == "__init__.py":
                continue
            # Convert path to module: myproject/settings.py → myproject.settings
            try:
                rel = match.relative_to(project_root)
                module = str(rel).replace("/", ".").replace("\\", ".")
                if module.endswith(".py"):
                    module = module[:-3]
                # Skip test settings
                if "test" not in module.lower():
                    return module
            except ValueError:
                continue
    return None


def _detect_pytest(project_root: Path) -> bool:
    """Check if project uses pytest."""
    indicators = [
        project_root / "pytest.ini",
        project_root / "conftest.py",
    ]
    for indicator in indicators:
        if indicator.exists():
            return True

    # Check pyproject.toml for [tool.pytest]
    pyproject = project_root / "pyproject.toml"
    if pyproject.exists():
        try:
            content = pyproject.read_text(encoding="utf-8", errors="ignore")
            if "[tool.pytest" in content:
                return True
        except Exception:
            pass

    return False


def _detect_non_python_runner(
    project_root: Path, test_files: Optional[List[str]] = None
) -> Optional[TestRunnerConfig]:
    """Detect test runners for non-Python ecosystems."""
    # Node.js / JavaScript / TypeScript
    package_json = project_root / "package.json"
    if package_json.exists():
        cmd = ["npm", "test"]
        if test_files:
            cmd = ["npx", "jest"] + test_files
        logger.info("Detected Node.js project")
        return TestRunnerConfig(command=cmd, runner_type="npm")

    # Rust
    cargo_toml = project_root / "Cargo.toml"
    if cargo_toml.exists():
        cmd = ["cargo", "test"]
        if test_files:
            # Rust test files use module paths
            cmd.extend(["--", "--test-threads=1"])
        logger.info("Detected Rust project")
        return TestRunnerConfig(command=cmd, runner_type="cargo")

    # Go
    go_mod = project_root / "go.mod"
    if go_mod.exists():
        cmd = ["go", "test", "./..."]
        if test_files:
            cmd = ["go", "test", "-v", "-run", ".*"]
        logger.info("Detected Go project")
        return TestRunnerConfig(command=cmd, runner_type="go")

    # Java / Gradle
    if (project_root / "build.gradle").exists() or (project_root / "build.gradle.kts").exists():
        cmd = ["./gradlew", "test"]
        logger.info("Detected Gradle project")
        return TestRunnerConfig(command=cmd, runner_type="gradle")

    # Java / Maven
    if (project_root / "pom.xml").exists():
        cmd = ["mvn", "test"]
        logger.info("Detected Maven project")
        return TestRunnerConfig(command=cmd, runner_type="maven")

    return None


def _files_to_django_labels(test_files: List[str]) -> List[str]:
    """Convert file paths to Django test labels.

    Django expects dotted module paths, not file paths:
    - tests/test_utils/tests.py → test_utils.tests
    - tests/auth_tests/test_views.py → auth_tests.test_views
    """
    labels = []
    for f in test_files:
        # Strip tests/ prefix if present
        f = re.sub(r"^tests/", "", f)
        # Convert to dotted path
        label = f.replace("/", ".").replace("\\", ".")
        if label.endswith(".py"):
            label = label[:-3]
        labels.append(label)
    return labels

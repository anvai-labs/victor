# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for framework test runner detection."""

from pathlib import Path

import pytest

from victor.context.test_runner import (
    TestRunnerConfig,
    detect_test_runner,
    _detect_django,
    _files_to_django_labels,
)


class TestDetectTestRunner:
    """Tests for detect_test_runner."""

    def test_detect_django_project(self, tmp_path):
        """Django project with manage.py detected correctly."""
        (tmp_path / "manage.py").write_text(
            "import os\n" "os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myapp.settings')\n"
        )
        config = detect_test_runner(tmp_path)
        assert config.runner_type == "django"
        assert "django" in " ".join(config.command)
        assert config.env["DJANGO_SETTINGS_MODULE"] == "myapp.settings"

    def test_detect_pytest_project(self, tmp_path):
        """Pytest project with conftest.py detected."""
        (tmp_path / "conftest.py").write_text("# conftest\n")
        config = detect_test_runner(tmp_path)
        assert config.runner_type == "pytest"
        assert "pytest" in " ".join(config.command)

    def test_detect_pytest_from_pyproject(self, tmp_path):
        """Pytest detected from pyproject.toml [tool.pytest] section."""
        (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\nminversion = '6.0'\n")
        config = detect_test_runner(tmp_path)
        assert config.runner_type == "pytest"

    def test_default_fallback_is_pytest(self, tmp_path):
        """No config files → default to pytest."""
        config = detect_test_runner(tmp_path)
        assert config.runner_type == "pytest"

    def test_test_files_passed_to_command(self, tmp_path):
        """Test files are appended to command."""
        (tmp_path / "conftest.py").write_text("")
        config = detect_test_runner(tmp_path, test_files=["tests/test_foo.py"])
        assert "tests/test_foo.py" in config.command

    def test_django_test_files_converted_to_labels(self, tmp_path):
        """Django test files converted to dotted module paths."""
        (tmp_path / "manage.py").write_text(
            "os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'app.settings')\n"
        )
        config = detect_test_runner(tmp_path, test_files=["tests/test_utils/tests.py"])
        assert config.runner_type == "django"
        # Should have dotted label, not file path
        assert "test_utils.tests" in config.command


class TestDetectDjango:
    """Tests for Django settings detection."""

    def test_parse_manage_py_settings(self, tmp_path):
        """Extract DJANGO_SETTINGS_MODULE from manage.py."""
        (tmp_path / "manage.py").write_text(
            "#!/usr/bin/env python\n"
            "import os\n"
            "os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django.test_utils.tests')\n"
        )
        result = _detect_django(tmp_path)
        assert result == "django.test_utils.tests"

    def test_no_manage_py(self, tmp_path):
        """No manage.py → not django."""
        result = _detect_django(tmp_path)
        assert result is None

    def test_fallback_to_settings_py(self, tmp_path):
        """manage.py without settings → find settings.py."""
        (tmp_path / "manage.py").write_text("# no settings here\n")
        proj = tmp_path / "myproject"
        proj.mkdir()
        (proj / "settings.py").write_text("# django settings\n")
        result = _detect_django(tmp_path)
        assert result == "myproject.settings"


class TestFilesToDjangoLabels:
    """Tests for file path to Django label conversion."""

    def test_basic_conversion(self):
        labels = _files_to_django_labels(["tests/test_utils/tests.py"])
        assert labels == ["test_utils.tests"]

    def test_strips_tests_prefix(self):
        labels = _files_to_django_labels(["tests/auth/test_views.py"])
        assert labels == ["auth.test_views"]

    def test_no_tests_prefix(self):
        labels = _files_to_django_labels(["myapp/test_models.py"])
        assert labels == ["myapp.test_models"]

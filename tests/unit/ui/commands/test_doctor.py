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

"""Tests for victor doctor command."""

import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
from unittest.mock import MagicMock, patch

from victor.ui.commands.doctor import (
    Severity,
    DiagnosticCheck,
    DoctorChecks,
    run_doctor,
)


class TestSeverity:
    """Tests for Severity enum."""

    def test_severity_values(self):
        """Severity has correct values."""
        assert Severity.ERROR.value == "error"
        assert Severity.WARNING.value == "warning"
        assert Severity.INFO.value == "info"
        assert Severity.SUCCESS.value == "success"


class TestDiagnosticCheck:
    """Tests for DiagnosticCheck dataclass."""

    def test_basic_check(self):
        """Basic check creation."""
        check = DiagnosticCheck(
            name="Test Check",
            severity=Severity.SUCCESS,
            message="Check passed",
        )
        assert check.name == "Test Check"
        assert check.severity == Severity.SUCCESS
        assert check.message == "Check passed"
        # Note: passed is set in add_check, not in the constructor
        # The dataclass defaults to False, but add_check sets it based on severity

    def test_check_with_suggestion(self):
        """Check with suggestion."""
        check = DiagnosticCheck(
            name="Test Check",
            severity=Severity.ERROR,
            message="Check failed",
            suggestion="Fix it",
        )
        assert check.suggestion == "Fix it"
        assert not check.passed


class TestDoctorChecks:
    """Tests for DoctorChecks class."""

    def test_init(self):
        """DoctorChecks initialization."""
        doctor = DoctorChecks(verbose=False)
        assert not doctor.verbose
        assert doctor.checks == []

    def test_init_verbose(self):
        """DoctorChecks with verbose mode."""
        doctor = DoctorChecks(verbose=True)
        assert doctor.verbose

    def test_add_check(self):
        """Adding a check works."""
        doctor = DoctorChecks()
        doctor.add_check(
            name="Test",
            severity=Severity.SUCCESS,
            message="Passed",
        )
        assert len(doctor.checks) == 1
        assert doctor.checks[0].name == "Test"

    def test_check_python_version_success(self):
        """Python version check passes for 3.10+."""
        doctor = DoctorChecks()
        doctor.check_python_version()

        if sys.version_info >= (3, 10):
            # Should have a success check
            has_success = any(
                check.name == "Python Version" and check.severity == Severity.SUCCESS
                for check in doctor.checks
            )
            assert has_success

    def test_check_dependencies(self):
        """Check dependencies finds required packages."""
        doctor = DoctorChecks()
        doctor.check_dependencies()

        # Should have checks for dependencies
        has_checks = any(
            "Dependency:" in check.name or "Optional:" in check.name for check in doctor.checks
        )
        assert has_checks

    def test_run_all_checks(self):
        """Running all checks executes all check methods."""
        doctor = DoctorChecks()
        checks = doctor.run_all_checks()

        # Should have multiple checks
        assert len(checks) > 0

        # Should have at least: Python Version, Dependencies, API Keys
        check_names = [check.name for check in checks]
        assert "Python Version" in check_names
        assert any("Dependency:" in name or "Optional:" in name for name in check_names)

    def test_print_results_success(self, capsys):
        """Printing results shows summary."""
        doctor = DoctorChecks()
        doctor.add_check(
            name="Test",
            severity=Severity.SUCCESS,
            message="Passed",
        )

        exit_code = doctor.print_results()
        captured = capsys.readouterr()

        # Should have summary
        assert "Summary:" in captured.out or "summary" in captured.out.lower()
        assert exit_code == 0

    def test_check_config_directory_uses_global_victor_dir(self, tmp_path):
        """Config directory checks should resolve through centralized Victor paths."""
        doctor = DoctorChecks()
        global_dir = tmp_path / ".victor"
        global_dir.mkdir(exist_ok=True)

        with patch(
            "victor.ui.commands.doctor.get_project_paths",
            return_value=SimpleNamespace(global_victor_dir=global_dir),
        ):
            doctor.check_config_directory()

        assert any(
            check.name == "Configuration Directory" and str(global_dir) in check.message
            for check in doctor.checks
        )

    def test_fix_issues_creates_canonical_config_dir(self, tmp_path):
        """Auto-fix should create the centralized global Victor config directory."""
        doctor = DoctorChecks()
        global_dir = tmp_path / "missing-victor-config"

        with patch(
            "victor.ui.commands.doctor.get_project_paths",
            return_value=SimpleNamespace(global_victor_dir=global_dir),
        ):
            doctor.check_config_directory()
            fixes = doctor.fix_issues()

        assert global_dir.exists()
        assert (global_dir / "config.yaml").exists()
        assert any(str(global_dir) in fix for fix in fixes)


class TestRunDoctor:
    """Tests for run_doctor entry point."""

    def test_run_doctor_success(self):
        """Running doctor returns success code."""
        exit_code = run_doctor(verbose=False, fix=False)
        # Should be 0 or 1 depending on system state
        assert isinstance(exit_code, int)
        assert exit_code in [0, 1]

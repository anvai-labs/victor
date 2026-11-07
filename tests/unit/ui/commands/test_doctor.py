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
from types import SimpleNamespace
import pytest
from unittest.mock import patch

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

    @pytest.mark.parametrize("minor, supported", [(10, False), (11, False), (12, True), (13, True)])
    def test_check_python_version(self, monkeypatch, minor, supported):
        """Diagnostics reject retired versions and accept the tested range."""
        monkeypatch.setattr(sys, "version_info", (3, minor, 0))
        doctor = DoctorChecks()
        doctor.check_python_version()
        check = doctor.checks[0]
        assert check.severity == (Severity.SUCCESS if supported else Severity.ERROR)
        if not supported:
            assert "requires 3.12+" in check.message

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


class TestNativeExtensionsCheck:
    """check_native_extensions: native wheel drift surfacing.

    Regression context (2026-11-06): a venv silently ran victor_native 0.15.0
    under victor-ai 0.26.x and pinned a chat session at 100% CPU inside a
    native dedup livelock. These tests pin the doctor behavior that makes such
    drift loud: stale = ERROR (exit-1 class), patch drift = WARNING.
    """

    def _run(self, monkeypatch, *, available=True, status=None, numpy=False):
        from victor.processing.native import _base as native_base

        monkeypatch.setattr(native_base, "is_native_available", lambda: available)
        monkeypatch.setattr(native_base, "get_native_version_status", lambda: dict(status or {}))
        monkeypatch.setattr(native_base, "numpy_accelerator_available", lambda: numpy)
        doctor = DoctorChecks()
        doctor.check_native_extensions()
        return doctor.checks

    def test_stale_wheel_is_error_with_remediation(self, monkeypatch):
        checks = self._run(
            monkeypatch,
            status={
                "state": "stale",
                "native_version": "0.15.0",
                "package_version": "0.26.2",
            },
        )
        check = next(c for c in checks if c.name == "Native Extensions")
        assert check.severity == Severity.ERROR
        assert "0.15.0" in check.message and "0.26.2" in check.message
        assert check.suggestion and "pip install" in check.suggestion

    def test_patch_drift_is_warning(self, monkeypatch):
        checks = self._run(
            monkeypatch,
            status={
                "state": "patch-drift",
                "native_version": "0.26.1",
                "package_version": "0.26.2",
            },
        )
        check = next(c for c in checks if c.name == "Native Extensions")
        assert check.severity == Severity.WARNING

    def test_matched_versions_are_success(self, monkeypatch):
        checks = self._run(
            monkeypatch,
            status={
                "state": "ok",
                "native_version": "0.26.2",
                "package_version": "0.26.2",
            },
        )
        check = next(c for c in checks if c.name == "Native Extensions")
        assert check.severity == Severity.SUCCESS
        assert "0.26.2" in check.message

    def test_unknown_state_still_reports_loaded_version(self, monkeypatch):
        checks = self._run(
            monkeypatch,
            status={"state": "unknown", "native_version": "0.26.2", "package_version": None},
        )
        check = next(c for c in checks if c.name == "Native Extensions")
        assert check.severity == Severity.SUCCESS

    def test_newer_and_skipped_states_are_info(self, monkeypatch):
        for state in ("newer", "skipped"):
            checks = self._run(
                monkeypatch,
                status={
                    "state": state,
                    "native_version": "9.9.9",
                    "package_version": "0.26.2",
                },
            )
            check = next(c for c in checks if c.name == "Native Extensions")
            assert check.severity == Severity.INFO, state

    def test_native_missing_is_warning_with_install_hint(self, monkeypatch):
        checks = self._run(monkeypatch, available=False)
        check = next(c for c in checks if c.name == "Native Extensions")
        assert check.severity == Severity.WARNING
        assert "victor-ai[native]" in (check.suggestion or "")

    def test_import_failure_is_warning_not_crash(self, monkeypatch):
        import sys

        import victor.processing.native as native_pkg

        # ``from victor.processing.native import _base`` returns the attribute
        # bound on the package when present, so the shim must remove both the
        # sys.modules entry and the package attribute.
        monkeypatch.setitem(sys.modules, "victor.processing.native._base", None)
        monkeypatch.delattr(native_pkg, "_base", raising=False)
        doctor = DoctorChecks()
        doctor.check_native_extensions()  # must not raise
        assert len(doctor.checks) == 1
        assert doctor.checks[0].severity == Severity.WARNING

    def test_numpy_dispatch_note_emitted_both_ways(self, monkeypatch):
        for available, expected in ((True, Severity.SUCCESS), (False, Severity.INFO)):
            checks = self._run(
                monkeypatch,
                status={"state": "ok", "native_version": "0.26.2", "package_version": "0.26.2"},
                numpy=available,
            )
            check = next(c for c in checks if c.name == "NumPy Accelerator")
            assert check.severity == expected

    def test_run_all_checks_includes_native_extensions(self):
        doctor = DoctorChecks()
        names = [c.name for c in doctor.run_all_checks()]
        assert "Native Extensions" in names


class TestGraphIndexingCheck:
    """check_graph_indexing reports the optional graph install without network I/O."""

    @staticmethod
    def _by_name(doctor):
        return {check.name: check for check in doctor.checks}

    @staticmethod
    def _imports(monkeypatch, broken=None, versions=None):
        """Route doctor imports: ``broken`` maps a module to the exception it raises."""
        import victor.ui.commands.doctor as doctor_module

        real = doctor_module.import_module
        broken = broken or {}

        def fake(name):
            if name in broken:
                raise broken[name]
            return real(name)

        monkeypatch.setattr(doctor_module, "import_module", fake)
        if versions is not None:
            monkeypatch.setattr("importlib.metadata.version", lambda name: versions[name])

    @staticmethod
    def _forbid_downloads(monkeypatch):
        """The doctor must never reach code that makes the language pack fetch grammars."""
        import victor.core.graph_rag.dependencies as deps

        def forbidden(*args, **kwargs):
            raise AssertionError("doctor reached a downloading code path")

        monkeypatch.setattr(deps, "require_graph_dependencies", forbidden)
        try:
            import tree_sitter_language_pack as pack

            monkeypatch.setattr(pack, "get_language", forbidden)
            monkeypatch.setattr(pack, "download", forbidden)
            monkeypatch.setattr(pack, "download_all", forbidden)
        except ImportError:
            pass
        try:
            import victor_codegraph

            monkeypatch.setattr(victor_codegraph, "parse", forbidden)
        except ImportError:
            pass

    def test_the_check_never_reaches_a_downloading_code_path(self, monkeypatch):
        self._forbid_downloads(monkeypatch)
        doctor = DoctorChecks()
        doctor.check_graph_indexing()  # must not raise
        assert "Graph Indexing" in self._by_name(doctor)

    def test_absent_graph_install_is_informational(self, monkeypatch):
        self._imports(monkeypatch, broken={"victor_codegraph": ImportError("absent")})
        doctor = DoctorChecks()
        doctor.check_graph_indexing()
        checks = self._by_name(doctor)
        assert checks["Graph Indexing"].severity == Severity.INFO
        assert "Graph Analysis Plugin" not in checks

    def test_a_crashing_codegraph_import_warns_instead_of_crashing(self, monkeypatch):
        self._imports(monkeypatch, broken={"victor_codegraph": RuntimeError("abi skew")})
        doctor = DoctorChecks()
        doctor.check_graph_indexing()
        check = self._by_name(doctor)["Graph Indexing"]
        assert check.severity == Severity.WARNING and "abi skew" in check.message

    def test_a_missing_grammar_wheel_is_named(self, monkeypatch):
        pytest.importorskip("victor_codegraph")
        self._forbid_downloads(monkeypatch)
        self._imports(monkeypatch, broken={"tree_sitter_ruby": ImportError("absent")})
        doctor = DoctorChecks()
        doctor.check_graph_indexing()
        check = self._by_name(doctor)["Graph Indexing"]
        assert check.severity == Severity.WARNING
        assert "ruby" in check.message and "victor-ai[graph]" in check.suggestion

    @pytest.mark.parametrize(
        ("error", "fragment"),
        [(ImportError("absent"), "still reporting success"), (RuntimeError("skew"), "skew")],
    )
    def test_an_unusable_analysis_plugin_warns_and_never_crashes(
        self, monkeypatch, error, fragment
    ):
        pytest.importorskip("victor_codegraph")
        self._forbid_downloads(monkeypatch)
        self._imports(monkeypatch, broken={"victor_coding": error})
        doctor = DoctorChecks()
        doctor.check_graph_indexing()
        check = self._by_name(doctor)["Graph Analysis Plugin"]
        assert check.severity == Severity.WARNING and fragment in check.message

    @pytest.mark.parametrize(
        ("installed", "severity"),
        [("0.8.0", Severity.WARNING), ("0.14.0", Severity.SUCCESS), ("0.16.0", Severity.SUCCESS)],
    )
    def test_the_analysis_plugin_must_meet_the_extra_floor(self, monkeypatch, installed, severity):
        pytest.importorskip("victor_codegraph")
        pytest.importorskip("victor_coding")
        self._forbid_downloads(monkeypatch)
        self._imports(monkeypatch, versions={"victor-coding": installed})
        doctor = DoctorChecks()
        doctor.check_graph_indexing()
        assert self._by_name(doctor)["Graph Analysis Plugin"].severity == severity

    def test_uncached_grammars_are_listed_with_a_prefetch_command(self, monkeypatch):
        import types

        import victor.ui.commands.doctor as doctor_module

        pytest.importorskip("victor_codegraph")
        pack = types.SimpleNamespace(
            downloaded_languages=lambda: ["java", "python"], cache_dir=lambda: "/cache/tslp"
        )
        real = doctor_module.import_module
        monkeypatch.setattr(
            doctor_module,
            "import_module",
            lambda name: pack if name == "tree_sitter_language_pack" else real(name),
        )
        doctor = DoctorChecks()
        doctor.check_graph_indexing()
        check = self._by_name(doctor)["Grammar Downloads"]
        assert check.severity == Severity.INFO and "/cache/tslp" in check.message
        assert "not yet cached:" in check.message and "ruby" in check.message
        assert "java," not in check.message.split("not yet cached:")[1]
        assert "download_all()" in check.suggestion

    def test_run_all_checks_includes_graph_indexing(self, monkeypatch):
        called = []
        monkeypatch.setattr(DoctorChecks, "check_graph_indexing", lambda self: called.append(True))
        for name in [
            n
            for n in dir(DoctorChecks)
            if n.startswith("check_") and n not in ("check_graph_indexing", "check_pip_package")
        ]:
            monkeypatch.setattr(DoctorChecks, name, lambda self: None)
        DoctorChecks().run_all_checks()
        assert called == [True]

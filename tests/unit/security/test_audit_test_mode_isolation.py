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
# See the License for the specific language and governing permissions and
# limitations under the License.

"""Tests for audit log test-mode isolation and benign-path suppression.

These tests address audit-noise findings from the audit/ directory review:
- 99.6% of HOME_MANIPULATION events came from test/temp HOME directories.
- Security events were misclassified as severity=info / outcome=success.
- Audit logs were written to the project ./audit/ during test runs,
  polluting the project database (violating test-isolation).

The fix mirrors the existing TEST_MODE telemetry-redirect pattern in
``victor/core/bootstrap.py`` and suppresses only *benign* (test/temp) HOME
paths while the process is under pytest. Real manipulation is still
detected and now logged at the correct severity.
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Test-mode detection
# ---------------------------------------------------------------------------
class TestTestModeDetection:
    """Verify the shared test-mode sentinel used across audit + secure_paths."""

    @pytest.mark.parametrize("env_var", ["TEST_MODE", "PYTEST_XDIST_WORKER", "PYTEST_CURRENT_TEST"])
    def test_each_sentinel_env_var_triggers_test_mode(self, monkeypatch, env_var):
        from victor.config.secure_paths import is_test_mode

        for v in ["TEST_MODE", "PYTEST_XDIST_WORKER", "PYTEST_CURRENT_TEST"]:
            monkeypatch.delenv(v, raising=False)
        assert is_test_mode() is False
        monkeypatch.setenv(env_var, "1")
        assert is_test_mode() is True


class TestBenignTestHomePathDetection:
    """Verify the classifier that recognizes legitimate test/temp HOME values."""

    @pytest.mark.parametrize(
        "home",
        [
            "/tmp/victor_test_home_abc123",
            "/var/folders/xx/yy/T/victor_test_home_zzz",
            "/tmp/fake_home",
            "/private/tmp/victor_test_home_42",
        ],
    )
    def test_recognizes_benign_test_paths(self, home):
        from victor.config.secure_paths import is_benign_test_home_path

        assert is_benign_test_home_path(Path(home)) is True

    @pytest.mark.parametrize(
        "home",
        [
            "/Users/attacker/.hidden",  # not a temp/test dir
            "/home/shared/.victor",
            "/Users/vijaysingh",  # normal home
        ],
    )
    def test_rejects_non_test_paths(self, home):
        from victor.config.secure_paths import is_benign_test_home_path

        assert is_benign_test_home_path(Path(home)) is False


# ---------------------------------------------------------------------------
# get_secure_home: benign suppression + real detection preserved
# ---------------------------------------------------------------------------
class TestSecureHomeManipulationSuppression:
    """SEC-001: suppress benign test noise but never weaken real detection."""

    def test_detection_still_fires_but_audit_redirected_under_test_mode(
        self, monkeypatch, tmp_path
    ):
        """SEC-001 contract: detection must fire for any env/passwd mismatch.

        We do NOT suppress benign detections at the detector (that would weaken a
        security control). Instead the *noise* is solved at the audit-logging
        layer: FileAuditLogger redirects events to <tmp>/victor_test_audit under
        test mode, so the project audit database is never polluted. This test
        pins that the detector still reports the event and still returns the
        trusted passwd entry.
        """
        from victor.config import secure_paths

        test_home = tmp_path / "victor_test_home_unit"
        test_home.mkdir()
        passwd_home = tmp_path / "real_home"
        passwd_home.mkdir()

        monkeypatch.setenv("HOME", str(test_home))
        monkeypatch.setenv("TEST_MODE", "1")

        logged = []
        with (
            patch.object(
                secure_paths,
                "_log_security_event",
                side_effect=lambda *a, **k: logged.append(a),
            ),
            patch.object(secure_paths, "get_real_home_from_passwd", return_value=passwd_home),
        ):
            result = secure_paths.get_secure_home()

        # Detection still fires (security contract preserved)
        assert len(logged) == 1
        assert logged[0][0] == secure_paths.SECURITY_EVENT_HOME_MANIPULATION
        # passwd entry returned (more trusted than a mismatched env HOME)
        assert result == passwd_home

    def test_real_manipulation_still_detected_and_logged(self, monkeypatch, tmp_path):
        """Outside test mode, a HOME that differs from passwd is still an alert."""
        from victor.config import secure_paths

        attacker_home = tmp_path / "attacker_spoof"
        attacker_home.mkdir()
        passwd_home = tmp_path / "real_home"
        passwd_home.mkdir()

        monkeypatch.setenv("HOME", str(attacker_home))
        for v in ["TEST_MODE", "PYTEST_XDIST_WORKER", "PYTEST_CURRENT_TEST"]:
            monkeypatch.delenv(v, raising=False)

        logged = []
        with (
            patch.object(
                secure_paths,
                "_log_security_event",
                side_effect=lambda *a, **k: logged.append(a),
            ),
            patch.object(secure_paths, "get_real_home_from_passwd", return_value=passwd_home),
        ):
            result = secure_paths.get_secure_home()

        # Detection fires for a real spoof (non-temp path, non-test mode)
        assert len(logged) == 1
        assert logged[0][0] == secure_paths.SECURITY_EVENT_HOME_MANIPULATION
        # passwd entry returned (more trusted)
        assert result == passwd_home


class TestSecurityEventSeverity:
    """A security detection must be WARNING, never INFO."""

    def test_log_security_event_uses_warning_severity(self):
        from victor.config import secure_paths
        from victor.security.audit.protocol import Severity

        captured = {}

        class _StubAudit:
            def log_event(self, **kwargs):
                captured.update(kwargs)

        with patch.object(secure_paths, "_get_audit_manager", return_value=_StubAudit()):
            secure_paths._log_security_event(
                secure_paths.SECURITY_EVENT_HOME_MANIPULATION, {"env_home": "/x"}
            )

        assert captured.get("severity") == Severity.WARNING


# ---------------------------------------------------------------------------
# Audit logger test-mode redirection (mirrors bootstrap telemetry redirect)
# ---------------------------------------------------------------------------
class TestAuditLoggerTestModeRedirection:
    """Audit logs must NOT land in the project ./audit/ during tests."""

    def test_audit_logger_redirects_to_test_dir_under_test_mode(self, monkeypatch, tmp_path):
        from victor.security.audit.logger import FileAuditLogger

        monkeypatch.setenv("TEST_MODE", "1")
        # Simulate a "project" root we must not pollute
        project_root = tmp_path / "project"
        project_root.mkdir()

        logger = FileAuditLogger(root_path=project_root)

        # The audit directory inside the project must NOT be created/used
        project_audit = project_root / "audit"
        assert not str(logger._audit_dir).startswith(str(project_audit))
        # And it must be under the system temp test-audit area
        assert "victor_test_audit" in str(logger._audit_dir)

    def test_audit_logger_uses_project_dir_in_production(self, monkeypatch, tmp_path):
        from victor.security.audit.logger import FileAuditLogger

        for v in ["TEST_MODE", "PYTEST_XDIST_WORKER", "PYTEST_CURRENT_TEST"]:
            monkeypatch.delenv(v, raising=False)
        project_root = tmp_path / "project"
        project_root.mkdir()

        logger = FileAuditLogger(root_path=project_root)

        assert logger._audit_dir == project_root / "audit"

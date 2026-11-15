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

"""Tests for TemporalContextAnalyzer module."""

import pytest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from victor.tools.verification import (
    ClaimIssue,
    TemporalContextAnalyzer,
    TemporalNature,
)


@pytest.fixture
def project_with_git(tmp_path: Path) -> Path:
    """Create a temporary project with git repository."""
    import subprocess

    # Initialize git repo
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=tmp_path,
        capture_output=True,
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, capture_output=True)

    # Create a file
    test_file = tmp_path / "src" / "lib.rs"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text("// TODO: Refactor this temporary code\npub fn test() {}\n")

    # Commit the file
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=tmp_path,
        capture_output=True,
    )

    return tmp_path


class TestTemporalContextAnalyzer:
    """Tests for TemporalContextAnalyzer class."""

    def test_estimate_temporal_nature_without_git(self, tmp_path: Path):
        """Test temporal analysis without git repository."""
        analyzer = TemporalContextAnalyzer(project_root=tmp_path)

        nature = analyzer.estimate_temporal_nature("src/lib.rs")

        assert nature == TemporalNature.UNKNOWN

    @pytest.mark.skipif(
        True,
        reason="Requires git repository",  # Skip in CI where git might not be available
    )
    def test_estimate_temporal_nature_with_todo(self, project_with_git: Path):
        """Test temporal analysis with TODO marker."""
        analyzer = TemporalContextAnalyzer(project_root=project_with_git)

        nature = analyzer.estimate_temporal_nature("src/lib.rs")

        # File has TODO, should lean towards temporary
        assert nature in {TemporalNature.TEMPORARY, TemporalNature.UNKNOWN}

    def test_check_for_removal_plan(self, project_with_git: Path):
        """Test checking for removal plan."""
        analyzer = TemporalContextAnalyzer(project_root=project_with_git)

        # File has TODO but not explicit removal plan
        has_removal = analyzer.check_for_removal_plan("src/lib.rs")

        assert isinstance(has_removal, bool)

    def test_get_file_age_days(self, project_with_git: Path):
        """Test getting file age."""
        analyzer = TemporalContextAnalyzer(project_root=project_with_git)

        age = analyzer.get_file_age_days("src/lib.rs")

        assert age >= 0  # Should be non-negative

    def test_is_recently_modified(self, project_with_git: Path):
        """Test checking if file is recently modified."""
        analyzer = TemporalContextAnalyzer(project_root=project_with_git)

        # Just created, should be recent
        is_recent = analyzer.is_recently_modified("src/lib.rs", days=30)

        assert isinstance(is_recent, bool)

    def test_analyze_issue_temporal_context(self, project_with_git: Path):
        """Test full temporal context analysis."""
        analyzer = TemporalContextAnalyzer(project_root=project_with_git)

        issue = ClaimIssue(
            issue_type="test_issue",
            description="Test issue",
            file_path="src/lib.rs",
        )

        context = analyzer.analyze_issue_temporal_context(issue)

        assert "temporal_nature" in context
        assert "confidence" in context
        assert "reason" in context
        assert 0.0 <= context["confidence"] <= 1.0

    def test_batch_analyze_files(self, project_with_git: Path):
        """Test batch analysis of multiple files."""
        analyzer = TemporalContextAnalyzer(project_root=project_with_git)

        files = ["src/lib.rs", "nonexistent.rs"]
        results = analyzer.batch_analyze_files(files)

        assert "src/lib.rs" in results
        assert "nonexistent.rs" in results


class TestTemporalGitPolicyBoundary:
    """Temporal git evidence follows the admitted launch policy."""

    @staticmethod
    def _install_policy(monkeypatch, policy: str, backend_available: bool) -> None:
        from victor.config.sandbox_settings import SandboxSettings
        from victor.tools.sandbox import backends

        settings = SimpleNamespace(
            sandbox=SandboxSettings(sandbox_enabled=True, sandbox_policy=policy)
        )
        monkeypatch.setattr("victor.config.settings.get_settings", lambda: settings)
        monkeypatch.setattr(backends.platform, "system", lambda: "Linux")
        monkeypatch.setattr(
            backends.shutil,
            "which",
            lambda name: "/usr/bin/bwrap" if backend_available and name == "bwrap" else None,
        )

    @staticmethod
    def _trap_launch(monkeypatch, launched: list[list[str]]) -> None:
        def _run(argv: list[str], **_kwargs):
            launched.append([str(value) for value in argv])
            return MagicMock(stdout="123\n", returncode=0)

        monkeypatch.setattr("subprocess.run", _run)

    @staticmethod
    def _analyzer(project_with_git: Path, **kwargs):
        from victor.tools.verification.temporal_analyzer import GitHistoryAnalyzer

        return GitHistoryAnalyzer(project_with_git, **kwargs)

    def test_required_available_wraps_temporal_git(self, monkeypatch, project_with_git):
        self._install_policy(monkeypatch, "required", backend_available=True)
        launched: list[list[str]] = []
        self._trap_launch(monkeypatch, launched)

        output = self._analyzer(project_with_git)._run_git(["status"])

        assert output == "123"
        assert launched and launched[0][0] == "bwrap"
        assert "git" in launched[0]

    def test_required_unavailable_skips_temporal_git(self, monkeypatch, project_with_git):
        self._install_policy(monkeypatch, "required", backend_available=False)
        launched: list[list[str]] = []
        self._trap_launch(monkeypatch, launched)

        history = self._analyzer(project_with_git).get_file_history("src/lib.rs")

        assert launched == []
        assert history.modification_count == 0

    @pytest.mark.parametrize("policy", ["disabled", "preferred"])
    def test_non_required_policy_keeps_temporal_git_raw(
        self, monkeypatch, project_with_git, policy
    ):
        self._install_policy(monkeypatch, policy, backend_available=True)
        launched: list[list[str]] = []
        self._trap_launch(monkeypatch, launched)

        self._analyzer(project_with_git)._run_git(["status"])

        assert launched and launched[0][0] == "git"

    def test_unreadable_policy_skips_temporal_git(self, monkeypatch, project_with_git):
        def _unavailable():
            raise RuntimeError("settings unavailable")

        monkeypatch.setattr("victor.config.settings.get_settings", _unavailable)
        launched: list[list[str]] = []
        self._trap_launch(monkeypatch, launched)

        assert self._analyzer(project_with_git)._run_git(["status"]) == ""
        assert launched == []

    def test_admitted_snapshot_overrides_reloaded_required_settings(
        self, monkeypatch, project_with_git
    ):
        from victor.config.settings import get_settings
        from victor.runtime.policy_snapshot import RunPolicySnapshot

        self._install_policy(monkeypatch, "disabled", backend_available=True)
        admitted = RunPolicySnapshot.from_settings(get_settings())
        self._install_policy(monkeypatch, "required", backend_available=False)
        launched: list[list[str]] = []
        self._trap_launch(monkeypatch, launched)

        output = self._analyzer(project_with_git, run_policy_snapshot=admitted)._run_git(["status"])

        assert output == "123"
        assert launched and launched[0][0] == "git"

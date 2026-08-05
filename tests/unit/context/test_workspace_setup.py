# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for workspace setup utilities."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from victor.context.workspace_setup import (
    detect_project_name,
    ensure_project_importable,
)


class TestDetectProjectName:
    """Tests for project name detection."""

    def test_detect_from_pyproject(self, tmp_path):
        """Detect name from pyproject.toml."""
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "my-project"\n')
        result = detect_project_name(tmp_path)
        assert result == "my_project"

    def test_detect_from_directory_name(self, tmp_path):
        """Fallback to directory name."""
        result = detect_project_name(tmp_path)
        # tmp_path name varies, just check it returns something
        assert result is not None

    def test_detect_handles_missing_files(self, tmp_path):
        """Works even with no config files."""
        result = detect_project_name(tmp_path)
        assert isinstance(result, str)


class TestEnsureProjectImportable:
    """Tests for ensure_project_importable."""

    @pytest.mark.asyncio
    async def test_already_importable_noop(self, tmp_path):
        """If project imports fine, no installation needed."""
        # 'os' is always importable
        result = await ensure_project_importable("os", tmp_path)
        assert result is True

    @pytest.mark.asyncio
    async def test_no_build_system_skips(self, tmp_path):
        """No setup.py/pyproject.toml → skip install, return False."""
        result = await ensure_project_importable("nonexistent_pkg_xyz", tmp_path)
        assert result is False

    @pytest.mark.asyncio
    async def test_install_with_setup_py(self, tmp_path):
        """Project with setup.py triggers pip install."""
        (tmp_path / "setup.py").write_text("from setuptools import setup\nsetup()")

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.returncode = 0
            mock_proc.communicate = AsyncMock(return_value=(b"", b""))
            mock_exec.return_value = mock_proc

            result = await ensure_project_importable("nonexistent_pkg_xyz", tmp_path)

        # pip install was called
        mock_exec.assert_called_once()
        call_args = mock_exec.call_args[0]
        assert "-m" in call_args
        assert "pip" in call_args
        assert "install" in call_args

    @pytest.mark.asyncio
    async def test_install_failure_returns_false(self, tmp_path):
        """Failed pip install returns False."""
        (tmp_path / "setup.py").write_text("from setuptools import setup\nsetup()")

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.returncode = 1
            mock_proc.communicate = AsyncMock(return_value=(b"", b"error: bad setup"))
            mock_exec.return_value = mock_proc

            result = await ensure_project_importable("nonexistent_pkg_xyz", tmp_path)
            assert result is False


class TestEnsureProjectImportableFastPath:
    """The fast-path must require a real pip-installed distribution, not a bare
    source-namespace import. A bare __import__ succeeds for a project's source
    dir on sys.path even when its C-extensions aren't built (e.g. astropy),
    which then yields "0 tests collected". Only a pip-installed distribution
    means extensions were actually built.
    """

    @pytest.mark.asyncio
    async def test_pip_installed_package_skips_install(self):
        """A pip-installed package (django) returns True without installing."""
        import importlib.metadata
        from pathlib import Path
        from unittest.mock import AsyncMock, patch

        from victor.context.workspace_setup import ensure_project_importable

        # Pick a name that IS pip-installed in the test env (pytest itself).
        installed_name = "pytest"
        with patch(
            "victor.context.workspace_setup.asyncio.create_subprocess_exec",
            AsyncMock(),
        ) as mock_exec:
            result = await ensure_project_importable(
                installed_name, Path("/nonexistent"), install_deps=True
            )
        assert result is True
        assert not mock_exec.called  # no install attempted — already installed

    @pytest.mark.asyncio
    async def test_uninstalled_package_triggers_install(self):
        """A package NOT pip-installed must attempt pip install -e ."""
        from pathlib import Path
        from unittest.mock import AsyncMock, MagicMock, patch

        from victor.context.workspace_setup import ensure_project_importable

        nonexistent = "definitely_not_installed_xyz_12345"
        fake_root = Path("/tmp/fake_project_xyz")
        with (
            patch.object(Path, "exists", return_value=True),
            patch(
                "victor.context.workspace_setup.asyncio.create_subprocess_exec",
                AsyncMock(),
            ) as mock_exec,
            patch("victor.context.workspace_setup.asyncio.wait_for", AsyncMock()),
        ):
            mock_proc = MagicMock()
            mock_proc.returncode = 1  # install fails
            with patch(
                "victor.context.workspace_setup.asyncio.wait_for",
                AsyncMock(return_value=(b"", b"err")),
            ):
                mock_exec.return_value = mock_proc
                result = await ensure_project_importable(nonexistent, fake_root, install_deps=True)
        # Not installed (PackageNotFoundError) → must attempt install
        assert mock_exec.called, "pip install should have been attempted"


@pytest.mark.asyncio
async def test_cleans_up_partial_install_on_failure(tmp_path):
    """A failed editable install must best-effort uninstall the partial/shadow
    so it doesn't poison later tasks' imports (SWE-bench venv pollution)."""
    (tmp_path / "setup.py").write_text("from setuptools import setup\nsetup()")

    with patch("victor.context.workspace_setup.asyncio.create_subprocess_exec") as mock_exec:
        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"", b"build failed"))
        mock_exec.return_value = mock_proc

        result = await ensure_project_importable("nonexistent_pkg_xyz", tmp_path)

    assert result is False
    # install attempt + cleanup uninstall == 2 subprocess calls
    assert mock_exec.call_count == 2
    cleanup_args = mock_exec.call_args_list[1][0]
    assert "uninstall" in cleanup_args
    assert "nonexistent_pkg_xyz" in cleanup_args

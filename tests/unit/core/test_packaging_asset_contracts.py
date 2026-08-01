"""Packaging contract tests for vertical asset availability."""

from __future__ import annotations

from pathlib import Path
import sys

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def test_pyproject_enables_setuptools_package_data_for_vertical_assets() -> None:
    """Wheel packaging must include vertical YAML/TOML/SVG assets."""
    pyproject = _repo_root() / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    setuptools_cfg = data["tool"]["setuptools"]
    assert setuptools_cfg["include-package-data"] is True

    package_data = setuptools_cfg["package-data"]["victor"]
    required_patterns = {
        "verticals/contrib/*/*.yaml",
        "verticals/contrib/*/*.toml",
        "verticals/contrib/*/*.svg",
        "verticals/contrib/*/workflows/*.yaml",
        "verticals/contrib/*/workflows/*.svg",
        # The TUI stylesheet: App.CSS_PATH loads it at startup. A wheel that
        # drops it crashes the TUI with StylesheetError (regression guard).
        "ui/tui/*.tcss",
    }
    missing = sorted(required_patterns - set(package_data))
    assert not missing, f"Missing required package-data patterns: {missing}"


def test_tui_stylesheet_asset_is_present_on_disk() -> None:
    """The TUI stylesheet declared in package-data must actually exist."""
    stylesheet = _repo_root() / "victor" / "ui" / "tui" / "styles.tcss"
    assert stylesheet.is_file(), f"Missing TUI stylesheet: {stylesheet}"


def test_manifest_includes_vertical_contrib_assets() -> None:
    """Source distributions should include vertical runtime assets."""
    manifest = (_repo_root() / "MANIFEST.in").read_text(encoding="utf-8")
    assert "recursive-include victor/verticals/contrib *.yaml *.yml *.toml *.svg" in manifest


def test_manifest_includes_tui_stylesheet() -> None:
    """Source distributions must include the TUI stylesheet."""
    manifest = (_repo_root() / "MANIFEST.in").read_text(encoding="utf-8")
    assert "recursive-include victor/ui/tui *.tcss" in manifest

"""First-party verticals older than the supported minimum are refused before import."""

from __future__ import annotations

import pytest

from victor.core.verticals.compatibility_gate import (
    FIRST_PARTY_MIN_VERTICAL_VERSIONS,
    VerticalCompatibilityError,
    VerticalCompatibilityGate,
)


def _gate() -> VerticalCompatibilityGate:
    return VerticalCompatibilityGate(framework_version_provider=lambda: "0.23.0")


@pytest.mark.parametrize("distribution", sorted(FIRST_PARTY_MIN_VERTICAL_VERSIONS))
def test_old_first_party_vertical_is_refused_with_upgrade_hint(distribution) -> None:
    report = _gate().assess_distribution(distribution, "0.13.0")
    assert not report.compatible
    assert f"pip install -U '{distribution}>=0.14.0'" in report.errors[0]
    with pytest.raises(VerticalCompatibilityError):
        report.raise_if_incompatible()


@pytest.mark.parametrize("version", ["0.14.0", "0.14.1", "1.0.0"])
def test_supported_first_party_versions_pass(version) -> None:
    assert _gate().assess_distribution("victor-coding", version).compatible


def test_distribution_names_are_canonicalized() -> None:
    assert not _gate().assess_distribution("victor_coding", "0.13.0").compatible
    assert not _gate().assess_distribution("Victor.Coding", "0.13.0").compatible


def test_third_party_and_unparseable_versions_pass() -> None:
    assert _gate().assess_distribution("victor-invest", "0.1.0").compatible
    assert _gate().assess_distribution("victor-coding", "not-a-version").compatible


class _Dist:
    def __init__(self, version: str, direct_url: str | None) -> None:
        self.version = version
        self._direct_url = direct_url

    def read_text(self, name: str):
        return self._direct_url if name == "direct_url.json" else None


def _preflight(monkeypatch, dist: _Dist) -> None:
    import importlib.metadata as metadata

    from victor.core.verticals.vertical_loader import VerticalLoader

    monkeypatch.setattr(
        metadata, "packages_distributions", lambda: {"victor_coding": ["victor-coding"]}
    )
    monkeypatch.setattr(metadata, "distribution", lambda name: dist)
    VerticalLoader.__new__(VerticalLoader)._preflight_distribution_version(
        "victor_coding.assistant"
    )


def test_loader_preflight_refuses_old_installed_vertical(monkeypatch) -> None:
    with pytest.raises(VerticalCompatibilityError, match="victor-coding 0.13.0"):
        _preflight(monkeypatch, _Dist("0.13.0", None))


def test_loader_preflight_accepts_current_vertical(monkeypatch) -> None:
    _preflight(monkeypatch, _Dist("0.14.0", None))


def test_loader_preflight_skips_editable_checkouts(monkeypatch) -> None:
    editable = '{"url": "file:///src/victor-coding", "dir_info": {"editable": true}}'
    _preflight(monkeypatch, _Dist("0.8.0", editable))


def test_loader_preflight_ignores_modules_without_distribution(monkeypatch) -> None:
    import importlib.metadata as metadata

    from victor.core.verticals.vertical_loader import VerticalLoader

    monkeypatch.setattr(metadata, "packages_distributions", lambda: {})
    VerticalLoader.__new__(VerticalLoader)._preflight_distribution_version("local_vertical.mod")

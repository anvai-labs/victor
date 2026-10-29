"""Tests for provider entry-point registry loading."""

from __future__ import annotations

import warnings


from victor.framework.providers.protocol import (
    ProviderRegistry,
    _WARNED_LEGACY_PROVIDER_GROUPS,
)


def test_provider_registry_prefers_canonical_team_provider_group(monkeypatch) -> None:
    """Canonical team-spec providers should win over legacy duplicates."""

    class _TeamProvider:
        def get_team_specs(self):
            return {"alpha": {"name": "Alpha"}}

    class _EntryPoint:
        def __init__(self, name: str) -> None:
            self.name = name

        def load(self):
            return _TeamProvider

    def _mock_entry_points(*, group: str):
        if group == "victor.team_spec_providers":
            return [_EntryPoint("coding")]
        if group == "victor.framework.teams.providers":
            return [_EntryPoint("coding")]
        return []

    monkeypatch.setattr("importlib.metadata.entry_points", _mock_entry_points)
    _WARNED_LEGACY_PROVIDER_GROUPS.clear()

    registry = ProviderRegistry()

    # DefaultTeamSpecProvider + one canonical external provider
    assert len(registry._team_providers) == 2
    assert registry.get_all_team_specs() == {"coding": {"alpha": {"name": "Alpha"}}}


def test_provider_registry_loads_canonical_workflow_provider_group(monkeypatch) -> None:
    """Canonical workflow providers should be discovered without legacy groups."""

    class _WorkflowProvider:
        def get_workflows(self):
            return {"ship_it": {"name": "ship_it"}}

    class _EntryPoint:
        def __init__(self, name: str) -> None:
            self.name = name

        def load(self):
            return _WorkflowProvider

    def _mock_entry_points(*, group: str):
        if group == "victor.workflow_providers":
            return [_EntryPoint("devops")]
        return []

    monkeypatch.setattr("importlib.metadata.entry_points", _mock_entry_points)
    _WARNED_LEGACY_PROVIDER_GROUPS.clear()

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        registry = ProviderRegistry()

    assert registry.get_all_workflows() == {"devops": {"ship_it": {"name": "ship_it"}}}
    assert not recorded


def test_provider_registry_warns_once_per_legacy_group(monkeypatch) -> None:
    """Repeated registry construction should not spam legacy-group warnings."""

    class _TeamProvider:
        def get_team_specs(self):
            return {"alpha": {"name": "Alpha"}}

    class _EntryPoint:
        def __init__(self, name: str) -> None:
            self.name = name

        def load(self):
            return _TeamProvider

    def _mock_entry_points(*, group: str):
        if group == "victor.team_spec_providers":
            return []
        if group == "victor.framework.teams.providers":
            return [_EntryPoint("coding")]
        return []

    monkeypatch.setattr("importlib.metadata.entry_points", _mock_entry_points)
    _WARNED_LEGACY_PROVIDER_GROUPS.clear()

    ProviderRegistry()
    # Second construction should not log again
    ProviderRegistry()

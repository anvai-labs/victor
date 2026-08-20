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

"""Tests for the cross-vendor review-team preset (AgentTeam.create_review_team)."""

import types

import pytest

from victor.framework.teams import AgentTeam, TeamFormation, TeamMemberSpec


@pytest.fixture(autouse=True)
def _stub_coordinator(monkeypatch):
    """Avoid building a real coordinator; the preset only shapes config."""
    monkeypatch.setattr(
        "victor.framework.teams.create_coordinator",
        lambda orchestrator, **kw: types.SimpleNamespace(),
    )


async def test_review_team_is_pipeline_with_per_member_providers():
    team = await AgentTeam.create_review_team(
        orchestrator=types.SimpleNamespace(),
        name="Review",
        goal="Ship a correct change",
        writer=TeamMemberSpec(role="executor", goal="write code", provider="openai"),
        reviewer=TeamMemberSpec(role="reviewer", goal="review code", provider="anthropic"),
    )
    config = team._config
    assert config.formation == TeamFormation.PIPELINE
    assert len(config.members) == 2
    assert config.members[0].provider == "openai"
    assert config.members[1].provider == "anthropic"


async def test_review_team_includes_optional_reviser():
    team = await AgentTeam.create_review_team(
        orchestrator=types.SimpleNamespace(),
        name="Review",
        goal="goal",
        writer=TeamMemberSpec(role="executor", goal="write", provider="openai"),
        reviewer=TeamMemberSpec(role="reviewer", goal="review", provider="anthropic"),
        reviser=TeamMemberSpec(role="executor", goal="apply feedback", provider="openai"),
    )
    assert len(team._config.members) == 3


async def test_review_team_warns_on_same_vendor(caplog):
    with caplog.at_level("WARNING"):
        await AgentTeam.create_review_team(
            orchestrator=types.SimpleNamespace(),
            name="SameVendor",
            goal="goal",
            writer=TeamMemberSpec(role="executor", goal="write", provider="openai"),
            reviewer=TeamMemberSpec(role="reviewer", goal="review", provider="openai"),
        )
    assert any("share provider" in r.message for r in caplog.records)


async def test_review_team_no_warning_on_cross_vendor(caplog):
    with caplog.at_level("WARNING"):
        await AgentTeam.create_review_team(
            orchestrator=types.SimpleNamespace(),
            name="CrossVendor",
            goal="goal",
            writer=TeamMemberSpec(role="executor", goal="write", provider="openai"),
            reviewer=TeamMemberSpec(role="reviewer", goal="review", provider="anthropic"),
        )
    assert not any("share provider" in r.message for r in caplog.records)

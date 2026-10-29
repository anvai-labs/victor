# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Types promoted to the SDK are defined there once; core and framework re-export them.

Each of these used to be a separate copy in core/framework (identical apart
from cosmetics), so an object built from one import path failed isinstance
checks against the other and the copies could drift.
"""

import importlib

import pytest

PAIRS = [
    ("victor.core.mode_config", "ModeLevel", "victor_contracts.verticals.mode_config", "ModeLevel"),
    (
        "victor.core.mode_config",
        "ModeConfig",
        "victor_contracts.verticals.mode_config",
        "ModeConfig",
    ),
    (
        "victor.core.mode_config",
        "ModeDefinition",
        "victor_contracts.verticals.mode_config",
        "ModeDefinition",
    ),
    (
        "victor.core.mode_config",
        "VerticalModeConfig",
        "victor_contracts.verticals.mode_config",
        "VerticalModeConfig",
    ),
    (
        "victor.framework.vertical_protocols",
        "CCGBuilderProtocol",
        "victor_contracts.verticals.protocols.storage",
        "CCGBuilderProtocol",
    ),
    ("victor.framework.teams", "TeamMemberSpec", "victor_contracts.team_schema", "TeamMemberSpec"),
    ("victor.framework.teams", "ROLE_MAPPING", "victor_contracts.team_schema", "ROLE_MAPPING"),
]


@pytest.mark.parametrize("core_mod, core_name, sdk_mod, sdk_name", PAIRS)
def test_core_name_is_the_sdk_object(core_mod, core_name, sdk_mod, sdk_name):
    core = getattr(importlib.import_module(core_mod), core_name)
    sdk = getattr(importlib.import_module(sdk_mod), sdk_name)
    assert core is sdk


def test_objects_interoperate_across_import_paths():
    from victor_contracts.verticals.mode_config import ModeConfig as SdkModeConfig

    from victor.core.mode_config import DEFAULT_MODES, ModeConfig

    config = ModeConfig(tool_budget=5, max_iterations=3)
    assert isinstance(config, SdkModeConfig)
    assert all(isinstance(mode.to_mode_config(), SdkModeConfig) for mode in DEFAULT_MODES.values())


def test_ccg_registry_key_is_shared():
    """Registration and lookup both use the framework import path; it must be the SDK key."""
    from victor_contracts.verticals.protocols.storage import CCGBuilderProtocol as Sdk

    from victor.framework.vertical_protocols import CCGBuilderProtocol

    assert {CCGBuilderProtocol: "x"}[Sdk] == "x"


def test_team_member_spec_carries_execution_overrides_to_runtime_member():
    """The framework copy had these fields and the SDK copy did not: vertical team
    definitions (which import the SDK) could not declare heterogeneous teams."""
    from victor_contracts.team_schema import TeamMemberSpec

    spec = TeamMemberSpec(
        role="reviewer",
        goal="Review",
        provider="anthropic",
        model="m",
        temperature=0.0,
        reasoning_effort="high",
        formation_role="critic",
    )
    member = spec.to_team_member()
    assert (member.provider, member.model, member.temperature) == ("anthropic", "m", 0.0)
    assert (member.reasoning_effort, member.formation_role) == ("high", "critic")


def test_team_member_spec_rejects_unknown_formation_role():
    from victor_contracts.team_schema import TeamMemberSpec

    with pytest.raises(ValueError):
        TeamMemberSpec(role="critic", goal="x", formation_role="not-a-role").to_team_member()


def test_supervisor_roles_map_to_planner_from_the_framework_path():
    """The framework copy lacked the supervisor aliases and fell back to EXECUTOR."""
    from victor.core.shared_types import SubAgentRole
    from victor.framework.teams import TeamMemberSpec

    assert TeamMemberSpec(role="supervisor", goal="x").to_team_member().role is SubAgentRole.PLANNER


def test_team_spec_round_trip_keeps_execution_overrides():
    from victor_contracts.team_schema import TeamFormation, TeamMemberSpec, TeamSpec

    spec = TeamSpec(
        name="t",
        description="d",
        vertical="v",
        formation=TeamFormation.SEQUENTIAL,
        members=[
            TeamMemberSpec(
                role="reviewer",
                goal="g",
                provider="p",
                model="m",
                temperature=0.2,
                reasoning_effort="high",
                formation_role="critic",
            ),
            TeamMemberSpec(role="writer", goal="g"),
        ],
    )
    reviewer, writer = TeamSpec.from_dict(spec.to_dict()).members
    assert (reviewer.provider, reviewer.model, reviewer.temperature) == ("p", "m", 0.2)
    assert (reviewer.reasoning_effort, reviewer.formation_role) == ("high", "critic")
    assert (writer.provider, writer.formation_role) == (None, None)

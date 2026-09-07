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

"""Definition-surface tests - runnable with only victor-contracts installed."""

from __future__ import annotations

from victor_security import SecurityAssistant

EXPECTED_STAGE_ORDER = ["reconnaissance", "analysis", "reporting"]


def test_identity_and_versions_are_consistent() -> None:
    from victor_contracts.verticals.registration import get_vertical_manifest

    import victor_security

    assert SecurityAssistant.get_name() == "security"
    assert SecurityAssistant.name == "security"
    assert SecurityAssistant.version == victor_security.__version__

    manifest = get_vertical_manifest(SecurityAssistant)
    assert manifest is not None, "@register_vertical manifest metadata is required"
    assert manifest.name == "security"
    assert manifest.version == SecurityAssistant.version


def test_definition_exposes_tools_and_capabilities() -> None:
    definition = SecurityAssistant.get_definition()

    assert definition.name == "security"

    tool_names = [req.tool_name for req in definition.tool_requirements]
    assert tool_names == SecurityAssistant.get_tools()
    required = [req.tool_name for req in definition.tool_requirements if req.required]
    assert set(required) == {"read", "ls", "code_search"}

    capability_ids = [req.capability_id for req in definition.capability_requirements]
    assert capability_ids == ["file_ops", "git", "web_access"]
    optional_ids = {req.capability_id for req in definition.capability_requirements if req.optional}
    assert optional_ids == {"git", "web_access"}


def test_definition_workflow_metadata_matches_stages() -> None:
    definition = SecurityAssistant.get_definition()
    stages = SecurityAssistant.get_stages()

    assert list(stages) == EXPECTED_STAGE_ORDER
    assert definition.workflow_metadata.initial_stage == "reconnaissance"
    assert definition.workflow_metadata.workflow_spec == {"stage_order": EXPECTED_STAGE_ORDER}


def test_task_type_hints_cover_declared_workflows() -> None:
    hints = SecurityAssistant.get_task_type_hints()

    assert set(hints) == {"vulnerability_scan", "dependency_audit", "incident_review"}
    for task_type, hint in hints.items():
        assert isinstance(hint["tool_budget"], int), task_type
        assert hint["tool_budget"] > 0, task_type
        assert hint["priority_tools"], task_type


def test_team_metadata_declares_default_review_team() -> None:
    definition = SecurityAssistant.get_definition()

    assert definition.team_metadata.default_team == "security_review_team"
    team_ids = [team.team_id for team in definition.team_metadata.teams]
    assert team_ids == ["security_review_team"]

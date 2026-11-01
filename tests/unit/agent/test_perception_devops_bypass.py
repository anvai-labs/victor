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

"""Unit tests for devops/workflow command bypass in perception clarification."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from victor.agent.perception_integration import PerceptionIntegration


@pytest.fixture
def perception() -> PerceptionIntegration:
    """Create a PerceptionIntegration with mocked dependencies."""
    return PerceptionIntegration()


@pytest.fixture
def task_analysis() -> SimpleNamespace:
    """Create a task analysis that requires confirmation (AMBIGUOUS intent)."""
    return SimpleNamespace(
        is_action_task=True,
        requires_confirmation=True,
    )


class TestDevopsWorkflowBypass:
    """Devops/workflow commands should not trigger clarification."""

    @pytest.mark.parametrize(
        "message",
        [
            "commit push pr and drive ci green",
            "commit and push",
            "push to develop",
            "create a pr",
            "open pr",
            "submit pr",
            "merge pr",
            "run ci",
            "drive ci green",
            "check ci status",
            "deploy to production",
            "release v1.0",
            "publish package",
            "ship it",
            "git commit",
            "git push origin main",
            "git merge feature-branch",
            "lint and format",
            "check tests",
            "run tests",
        ],
    )
    def test_devops_commands_skip_clarification(
        self, perception: PerceptionIntegration, task_analysis: SimpleNamespace, message: str
    ) -> None:
        """Devops commands bypass the confirmation gate even with AMBIGUOUS intent."""
        result = perception._assess_clarification_need(
            query=message,
            task_analysis=task_analysis,
            confidence=0.3,  # Low confidence would normally trigger clarification
            context=None,
            conversation_history=None,
        )
        assert result["needs_clarification"] is False
        assert result["clarification_reason"] is None

    @pytest.mark.parametrize(
        "message",
        [
            "fix the bug",
            "update the handler",
            "refactor this",
            "check the handler",
            "check this",
            "format it",
            "merge these",
            "format the handler",
            "release the lock",
            "publish the event",
            "tag the element",
            "merge the functions",
        ],
    )
    def test_non_devops_actions_still_require_clarification(
        self, perception: PerceptionIntegration, task_analysis: SimpleNamespace, message: str
    ) -> None:
        """Non-devops action commands with AMBIGUOUS intent and no target still trigger clarification."""
        result = perception._assess_clarification_need(
            query=message,
            task_analysis=task_analysis,
            confidence=0.3,
            context=None,
            conversation_history=None,
        )
        assert result["needs_clarification"] is True

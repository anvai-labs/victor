# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from victor.agent.services.coordination_advisor_runtime import (
    CoordinationAdvisorRuntime,
)


def test_suggest_for_task_uses_runtime_subject_helper():
    runtime = CoordinationAdvisorRuntime()
    runtime_subject = SimpleNamespace(name="runtime")
    suggestion = MagicMock()

    with patch(
        "victor.framework.coordination_runtime.build_runtime_coordination_suggestion",
        return_value=suggestion,
    ) as build_suggestion:
        result = runtime.suggest_for_task(
            runtime_subject=runtime_subject,
            task_type="feature",
            complexity="high",
            mode="build",
        )

    assert result is suggestion
    build_suggestion.assert_called_once_with(
        runtime_subject=runtime_subject,
        task_type="feature",
        complexity="high",
        mode="build",
    )


def test_suggest_for_task_uses_explicit_coordination_advisor():
    runtime = CoordinationAdvisorRuntime()
    advisor = MagicMock()
    suggestion = MagicMock()
    advisor.suggest_for_task.return_value = suggestion

    result = runtime.suggest_for_task(
        coordination_advisor=advisor,
        task_type="feature",
        complexity="high",
        mode="build",
    )

    assert result is suggestion
    advisor.suggest_for_task.assert_called_once_with(
        task_type="feature",
        complexity="high",
        mode="build",
    )


def test_serialize_suggestion_delegates_to_framework_serializer():
    runtime = CoordinationAdvisorRuntime()
    suggestion = MagicMock()

    with patch(
        "victor.framework.coordination_runtime.serialize_coordination_suggestion",
        return_value={"action": "suggest"},
    ) as serializer:
        payload = runtime.serialize_suggestion(suggestion, vertical="coding")

    assert payload == {"action": "suggest"}
    serializer.assert_called_once_with(
        suggestion,
        vertical="coding",
        available_teams=None,
        available_workflows=None,
        default_workflow=None,
    )

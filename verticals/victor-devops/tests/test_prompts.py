# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for the DevOps prompt contributor."""

from victor_contracts.verticals import PromptContributorProtocol, TaskTypeHint

from victor_devops.prompts import DEVOPS_TASK_TYPE_HINTS, DevOpsPromptContributor


class TestTaskTypeHints:
    def test_classifier_aligned_hints_present(self):
        # Keys that must align with TaskTypeClassifier task types
        assert "infrastructure" in DEVOPS_TASK_TYPE_HINTS
        assert "ci_cd" in DEVOPS_TASK_TYPE_HINTS
        assert "general" in DEVOPS_TASK_TYPE_HINTS

    def test_hint_shapes(self):
        for task_type, hint in DEVOPS_TASK_TYPE_HINTS.items():
            assert isinstance(hint, TaskTypeHint)
            assert hint.task_type == task_type
            assert hint.hint.strip(), f"{task_type} hint is empty"
            assert hint.tool_budget > 0
            assert hint.priority_tools, f"{task_type} has no priority tools"

    def test_precision_tasks_use_low_temperature(self):
        """Config-generation tasks need deterministic output."""
        for task_type in ("dockerfile", "kubernetes", "terraform"):
            assert DEVOPS_TASK_TYPE_HINTS[task_type].temperature_override <= 0.3


class TestPromptContributor:
    def test_implements_protocol(self):
        contributor = DevOpsPromptContributor()

        assert isinstance(contributor, PromptContributorProtocol)

    def test_get_task_type_hints_returns_copy(self):
        contributor = DevOpsPromptContributor()

        hints = contributor.get_task_type_hints()
        assert hints == DEVOPS_TASK_TYPE_HINTS
        hints.pop("infrastructure")
        assert "infrastructure" in DEVOPS_TASK_TYPE_HINTS, "mutation leaked into module dict"

    def test_system_prompt_section_has_security_checklist(self):
        section = DevOpsPromptContributor().get_system_prompt_section()

        assert "Security Checklist" in section
        assert "No hardcoded secrets" in section

    def test_grounding_rules_forbid_invention(self):
        rules = DevOpsPromptContributor().get_grounding_rules()

        assert "GROUNDING" in rules
        assert "tool output" in rules

    def test_priority_is_medium(self):
        assert DevOpsPromptContributor().get_priority() == 5

    def test_context_hints_for_known_task_type(self):
        contributor = DevOpsPromptContributor()

        hint = contributor.get_context_hints("terraform")
        assert hint == DEVOPS_TASK_TYPE_HINTS["terraform"].hint

    def test_context_hints_for_unknown_task_type(self):
        contributor = DevOpsPromptContributor()

        assert contributor.get_context_hints("cooking") is None
        assert contributor.get_context_hints(None) is None

# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for the enhanced DevOps conversation manager and context tracking."""

from victor_contracts import TurnType

from victor_devops.conversation_enhanced import (
    DevOpsContext,
    EnhancedDevOpsConversationManager,
)


class TestDevOpsContext:
    def test_add_deployment_records_fields(self):
        ctx = DevOpsContext()

        ctx.add_deployment("api", "staging", "success", version="v1.2.3")

        assert ctx.deployments == [
            {
                "service": "api",
                "environment": "staging",
                "status": "success",
                "version": "v1.2.3",
            }
        ]

    def test_add_infrastructure_change(self):
        ctx = DevOpsContext()

        ctx.add_infrastructure_change("create", "aws_s3_bucket.assets", "terraform")

        assert ctx.infrastructure_changes[0]["type"] == "create"
        assert ctx.infrastructure_changes[0]["tool"] == "terraform"

    def test_add_container_operation_optional_resource(self):
        ctx = DevOpsContext()

        ctx.add_container_operation("restart", "docker")

        assert ctx.container_operations[0]["resource_id"] is None

    def test_add_pipeline_run(self):
        ctx = DevOpsContext()

        ctx.add_pipeline_run("ci-main", "passed", duration=120.5)

        assert ctx.pipeline_runs[0] == {
            "pipeline": "ci-main",
            "status": "passed",
            "duration": 120.5,
        }

    def test_to_dict_round_trip(self):
        ctx = DevOpsContext()
        ctx.add_deployment("api", "prod", "success")

        data = ctx.to_dict()

        assert set(data) == {
            "deployments",
            "infrastructure_changes",
            "container_operations",
            "pipeline_runs",
            "system_changes",
            "alerts",
        }
        assert len(data["deployments"]) == 1


class TestEnhancedConversationManager:
    def test_add_message_returns_turn_id(self):
        manager = EnhancedDevOpsConversationManager()

        turn_id = manager.add_message("user", "deploy to staging", TurnType.USER)

        assert turn_id
        history = manager.get_history()
        assert any(m["content"] == "deploy to staging" for m in history)

    def test_track_deployment_flows_into_context(self):
        manager = EnhancedDevOpsConversationManager()

        manager.track_deployment("api", "staging", "success", "v2.0.0")

        ctx = manager.get_devops_context()
        assert ctx.deployments[0]["service"] == "api"

    def test_devops_summary_reflects_tracked_work(self):
        manager = EnhancedDevOpsConversationManager()
        manager.track_deployment("api", "staging", "success")
        manager.track_pipeline_run("ci", "passed")

        summary = manager.get_devops_summary()

        assert summary  # non-empty summary of DevOps work

    def test_clear_history_resets_conversation(self):
        manager = EnhancedDevOpsConversationManager()
        manager.add_message("user", "hello", TurnType.USER)

        manager.clear_history()

        assert manager.get_history() == []

    def test_needs_summarization_threshold(self):
        manager = EnhancedDevOpsConversationManager(max_history_turns=10, summarization_threshold=2)

        assert manager.needs_summarization() is False
        manager.add_message("user", "one", TurnType.USER)
        manager.add_message("assistant", "two", TurnType.ASSISTANT)
        manager.add_message("user", "three", TurnType.USER)

        assert manager.needs_summarization() is True

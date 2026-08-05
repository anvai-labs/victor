# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Behavioral tests for DevOps workflow escape hatches (conditions + transforms)."""

from victor_devops.escape_hatches import (
    CONDITIONS,
    TRANSFORMS,
    container_build_status,
    deployment_ready,
    generate_deployment_summary,
    health_check_status,
    infrastructure_drift,
    merge_deployment_results,
    pipeline_stage_gate,
    rollback_needed,
    security_scan_verdict,
)


class TestDeploymentReady:
    def test_invalid_config_fails(self):
        assert deployment_ready({"config_valid": False}) == "failed"

    def test_missing_dependencies_blocks(self):
        ctx = {"config_valid": True, "dependencies_met": False}
        assert deployment_ready(ctx) == "blocked"

    def test_production_requires_approval(self):
        ctx = {
            "config_valid": True,
            "dependencies_met": True,
            "environment": "production",
            "approval_status": "pending",
        }
        assert deployment_ready(ctx) == "blocked"

    def test_approved_production_is_ready(self):
        ctx = {
            "config_valid": True,
            "dependencies_met": True,
            "environment": "production",
            "approval_status": "approved",
        }
        assert deployment_ready(ctx) == "ready"

    def test_rejected_approval_fails(self):
        ctx = {
            "config_valid": True,
            "dependencies_met": True,
            "approval_status": "rejected",
        }
        assert deployment_ready(ctx) == "failed"


class TestHealthCheckStatus:
    def test_no_results_is_unhealthy(self):
        assert health_check_status({}) == "unhealthy"

    def test_all_healthy(self):
        ctx = {"health_results": {"api": {"status": "healthy"}, "db": {"status": "healthy"}}}
        assert health_check_status(ctx) == "healthy"

    def test_degraded_between_thresholds(self):
        results = {f"ep{i}": {"status": "healthy"} for i in range(9)}
        results["ep9"] = {"status": "down"}
        assert health_check_status({"health_results": results}) == "degraded"

    def test_below_minimum_is_unhealthy(self):
        results = {"a": {"status": "healthy"}, "b": {"status": "down"}, "c": {"status": "down"}}
        assert health_check_status({"health_results": results}) == "unhealthy"


class TestRollbackNeeded:
    def test_failed_deploy_rolls_back(self):
        assert rollback_needed({"deploy_result": {"success": False}}) == "rollback"

    def test_high_error_rate_rolls_back(self):
        ctx = {"deploy_result": {"success": True}, "health_status": "healthy", "error_rate": 0.2}
        assert rollback_needed(ctx) == "rollback"

    def test_degraded_health_monitors(self):
        ctx = {"deploy_result": {"success": True}, "health_status": "degraded", "error_rate": 0.0}
        assert rollback_needed(ctx) == "monitor"

    def test_stable_deployment(self):
        ctx = {"deploy_result": {"success": True}, "health_status": "healthy", "error_rate": 0.0}
        assert rollback_needed(ctx) == "stable"


class TestContainerBuildStatus:
    def test_failed_build(self):
        assert container_build_status({"build_result": {"success": False}}) == "failed"

    def test_oversized_image_warns(self):
        ctx = {"build_result": {"success": True}, "image_size": 3000, "max_size": 2000}
        assert container_build_status(ctx) == "warning"

    def test_successful_build(self):
        ctx = {"build_result": {"success": True}, "image_size": 150}
        assert container_build_status(ctx) == "success"


class TestInfrastructureDrift:
    def test_no_changes_no_drift(self):
        assert infrastructure_drift({"plan_changes": {}}) == "no_drift"

    def test_destroy_is_destructive(self):
        assert infrastructure_drift({"plan_changes": {"destroy": 1}}) == "destructive"

    def test_many_changes_major_drift(self):
        assert infrastructure_drift({"plan_changes": {"create": 8, "update": 5}}) == "major_drift"

    def test_few_changes_minor_drift(self):
        assert infrastructure_drift({"plan_changes": {"update": 2}}) == "minor_drift"


class TestSecurityScanVerdict:
    def test_critical_findings_fail(self):
        assert security_scan_verdict({"scan_results": {"critical": 1}}) == "fail"

    def test_high_findings_fail_at_default_threshold(self):
        assert security_scan_verdict({"scan_results": {"high": 2}}) == "fail"

    def test_medium_findings_warn_at_medium_threshold(self):
        ctx = {"scan_results": {"medium": 3}, "severity_threshold": "medium"}
        assert security_scan_verdict(ctx) == "warn"

    def test_clean_scan_passes(self):
        assert security_scan_verdict({"scan_results": {}}) == "pass"


class TestPipelineStageGate:
    def test_failing_tests_abort(self):
        ctx = {"stage_results": {"tests_passed": False}, "allow_failures": False}
        assert pipeline_stage_gate(ctx) == "abort"

    def test_low_coverage_aborts(self):
        ctx = {"stage_results": {"tests_passed": True, "coverage": 0.5}}
        assert pipeline_stage_gate(ctx) == "abort"

    def test_passing_gate_proceeds(self):
        ctx = {"stage_results": {"tests_passed": True, "coverage": 0.9}}
        assert pipeline_stage_gate(ctx) == "proceed"


class TestTransforms:
    def test_merge_deployment_results_all_success(self):
        ctx = {
            "monitoring_result": {"success": True},
            "notification_result": {"success": True},
            "docs_result": {"success": True},
        }
        merged = merge_deployment_results(ctx)

        assert merged["all_tasks_success"] is True
        assert merged["monitoring_updated"] is True

    def test_merge_deployment_results_docs_optional(self):
        ctx = {
            "monitoring_result": {"success": True},
            "notification_result": {"success": True},
        }
        merged = merge_deployment_results(ctx)

        assert merged["all_tasks_success"] is True
        assert merged["docs_updated"] is False

    def test_generate_deployment_summary_defaults(self):
        summary = generate_deployment_summary({})

        assert summary["environment"] == "unknown"
        assert summary["rollback_performed"] is False

    def test_generate_deployment_summary_passthrough(self):
        summary = generate_deployment_summary(
            {"target_env": "staging", "deploy_version": "v1.2.3", "duration": 42}
        )

        assert summary["environment"] == "staging"
        assert summary["version"] == "v1.2.3"
        assert summary["duration_seconds"] == 42


class TestRegistries:
    def test_conditions_registry_complete(self):
        assert set(CONDITIONS) == {
            "deployment_ready",
            "health_check_status",
            "rollback_needed",
            "container_build_status",
            "infrastructure_drift",
            "security_scan_verdict",
            "pipeline_stage_gate",
        }
        assert all(callable(fn) for fn in CONDITIONS.values())

    def test_transforms_registry_complete(self):
        assert set(TRANSFORMS) == {
            "merge_deployment_results",
            "generate_deployment_summary",
        }
        assert all(callable(fn) for fn in TRANSFORMS.values())

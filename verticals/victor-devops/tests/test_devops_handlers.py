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

"""Unit tests for DevOps vertical handlers."""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

import pytest


class MockToolResult:
    """Mock tool result for testing."""

    def __init__(self, success: bool = True, output: Any = None, error: str = None):
        self.success = success
        self.output = output
        self.error = error


class MockComputeNode:
    """Mock compute node for testing."""

    def __init__(
        self,
        node_id: str = "test_node",
        input_mapping: Dict[str, Any] = None,
        output_key: str = None,
    ):
        self.id = node_id
        self.input_mapping = input_mapping or {}
        self.output_key = output_key


class MockWorkflowContext:
    """Mock workflow context for testing."""

    def __init__(self, data: Dict[str, Any] = None):
        self._data = data or {}

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value


class TestContainerOpsHandler:
    """Tests for ContainerOpsHandler."""

    @pytest.fixture
    def handler(self):
        from victor_devops.handlers import ContainerOpsHandler

        return ContainerOpsHandler()

    @pytest.fixture
    def mock_registry(self):
        registry = MagicMock()
        registry.execute = AsyncMock(
            return_value=MockToolResult(
                success=True,
                output="Successfully built image",
            )
        )
        return registry

    @pytest.mark.asyncio
    async def test_build_operation(self, handler, mock_registry):
        """Test container build operation."""
        node = MockComputeNode(
            input_mapping={
                "operation": "build",
                "dockerfile": "Dockerfile",
                "tag": "myapp:v1",
            },
            output_key="build_result",
        )
        context = MockWorkflowContext()

        result = await handler(node, context, mock_registry)

        assert result.status.value == "completed"
        assert result.output["success"] is True
        assert result.output["operation"] == "build"
        mock_registry.execute.assert_called_once()
        call_args = mock_registry.execute.call_args
        assert "docker build" in call_args.kwargs["command"]
        assert "-t myapp:v1" in call_args.kwargs["command"]

    @pytest.mark.asyncio
    async def test_push_operation(self, handler, mock_registry):
        """Test container push operation."""
        node = MockComputeNode(
            input_mapping={
                "operation": "push",
                "tag": "registry.example.com/myapp:v1",
            },
        )
        context = MockWorkflowContext()

        result = await handler(node, context, mock_registry)

        assert result.status.value == "completed"
        call_args = mock_registry.execute.call_args
        assert "docker push" in call_args.kwargs["command"]

    @pytest.mark.asyncio
    async def test_pull_operation(self, handler, mock_registry):
        """Test container pull operation."""
        node = MockComputeNode(
            input_mapping={
                "operation": "pull",
                "image": "nginx:latest",
            },
        )
        context = MockWorkflowContext()

        result = await handler(node, context, mock_registry)

        assert result.status.value == "completed"
        call_args = mock_registry.execute.call_args
        assert "docker pull nginx:latest" in call_args.kwargs["command"]

    @pytest.mark.asyncio
    async def test_run_operation(self, handler, mock_registry):
        """Test container run operation."""
        node = MockComputeNode(
            input_mapping={
                "operation": "run",
                "image": "myapp:v1",
            },
        )
        context = MockWorkflowContext()

        result = await handler(node, context, mock_registry)

        assert result.status.value == "completed"
        call_args = mock_registry.execute.call_args
        assert "docker run -d myapp:v1" in call_args.kwargs["command"]

    @pytest.mark.asyncio
    async def test_stop_operation(self, handler, mock_registry):
        """Test container stop operation."""
        node = MockComputeNode(
            input_mapping={
                "operation": "stop",
                "container_id": "abc123",
            },
        )
        context = MockWorkflowContext()

        result = await handler(node, context, mock_registry)

        assert result.status.value == "completed"
        call_args = mock_registry.execute.call_args
        assert "docker stop abc123" in call_args.kwargs["command"]

    @pytest.mark.asyncio
    async def test_unknown_operation(self, handler, mock_registry):
        """Test handling of unknown operation."""
        node = MockComputeNode(
            input_mapping={
                "operation": "invalid_op",
            },
        )
        context = MockWorkflowContext()

        result = await handler(node, context, mock_registry)

        assert result.status.value == "failed"
        assert "Unknown operation" in result.error

    @pytest.mark.asyncio
    async def test_operation_failure(self, handler):
        """Test handling of operation failure."""
        mock_registry = MagicMock()
        mock_registry.execute = AsyncMock(
            return_value=MockToolResult(success=False, error="Build failed")
        )

        node = MockComputeNode(
            input_mapping={
                "operation": "build",
                "dockerfile": "Dockerfile",
            },
        )
        context = MockWorkflowContext()

        result = await handler(node, context, mock_registry)

        assert result.status.value == "failed"

    @pytest.mark.asyncio
    async def test_podman_runtime(self, mock_registry):
        """Test using podman runtime instead of docker."""
        from victor_devops.handlers import ContainerOpsHandler

        handler = ContainerOpsHandler(runtime="podman")
        node = MockComputeNode(
            input_mapping={
                "operation": "build",
                "dockerfile": "Dockerfile",
                "tag": "myapp:v1",
            },
        )
        context = MockWorkflowContext()

        await handler(node, context, mock_registry)

        call_args = mock_registry.execute.call_args
        assert "podman build" in call_args.kwargs["command"]


class TestTerraformHandler:
    """Tests for TerraformHandler."""

    @pytest.fixture
    def handler(self):
        from victor_devops.handlers import TerraformHandler

        return TerraformHandler()

    @pytest.fixture
    def mock_registry(self):
        registry = MagicMock()
        registry.execute = AsyncMock(
            return_value=MockToolResult(
                success=True,
                output="Terraform operation completed",
            )
        )
        return registry

    @pytest.mark.asyncio
    async def test_init_operation(self, handler, mock_registry):
        """Test terraform init operation."""
        node = MockComputeNode(
            input_mapping={
                "operation": "init",
            },
            output_key="tf_result",
        )
        context = MockWorkflowContext()

        result = await handler(node, context, mock_registry)

        assert result.status.value == "completed"
        call_args = mock_registry.execute.call_args
        assert "terraform init" in call_args.kwargs["command"]

    @pytest.mark.asyncio
    async def test_plan_operation(self, handler, mock_registry):
        """Test terraform plan operation."""
        node = MockComputeNode(
            input_mapping={
                "operation": "plan",
            },
        )
        context = MockWorkflowContext()

        result = await handler(node, context, mock_registry)

        assert result.status.value == "completed"
        call_args = mock_registry.execute.call_args
        assert "terraform plan -out=tfplan" in call_args.kwargs["command"]

    @pytest.mark.asyncio
    async def test_apply_with_auto_approve(self, handler, mock_registry):
        """Test terraform apply with auto-approve."""
        node = MockComputeNode(
            input_mapping={
                "operation": "apply",
                "auto_approve": True,
            },
        )
        context = MockWorkflowContext()

        result = await handler(node, context, mock_registry)

        assert result.status.value == "completed"
        call_args = mock_registry.execute.call_args
        assert "terraform apply -auto-approve" in call_args.kwargs["command"]

    @pytest.mark.asyncio
    async def test_apply_from_plan(self, handler, mock_registry):
        """Test terraform apply from plan file."""
        node = MockComputeNode(
            input_mapping={
                "operation": "apply",
                "auto_approve": False,
            },
        )
        context = MockWorkflowContext()

        result = await handler(node, context, mock_registry)

        assert result.status.value == "completed"
        call_args = mock_registry.execute.call_args
        assert "terraform apply tfplan" in call_args.kwargs["command"]

    @pytest.mark.asyncio
    async def test_destroy_operation(self, handler, mock_registry):
        """Test terraform destroy operation."""
        node = MockComputeNode(
            input_mapping={
                "operation": "destroy",
                "auto_approve": True,
            },
        )
        context = MockWorkflowContext()

        result = await handler(node, context, mock_registry)

        assert result.status.value == "completed"
        call_args = mock_registry.execute.call_args
        assert "terraform destroy -auto-approve" in call_args.kwargs["command"]

    @pytest.mark.asyncio
    async def test_workspace_selection(self, handler, mock_registry):
        """Test terraform workspace selection."""
        node = MockComputeNode(
            input_mapping={
                "operation": "plan",
                "workspace": "production",
            },
        )
        context = MockWorkflowContext()

        result = await handler(node, context, mock_registry)

        assert result.status.value == "completed"
        # Should have called execute twice: workspace select + plan
        assert mock_registry.execute.call_count == 2
        first_call = mock_registry.execute.call_args_list[0]
        assert "workspace select production" in first_call.kwargs["command"]

    @pytest.mark.asyncio
    async def test_unknown_operation(self, handler, mock_registry):
        """Test handling of unknown operation."""
        node = MockComputeNode(
            input_mapping={
                "operation": "invalid_op",
            },
        )
        context = MockWorkflowContext()

        result = await handler(node, context, mock_registry)

        assert result.status.value == "failed"
        assert "Unknown operation" in result.error

    @pytest.mark.asyncio
    async def test_opentofu_binary(self, mock_registry):
        """Test using OpenTofu instead of Terraform."""
        from victor_devops.handlers import TerraformHandler

        handler = TerraformHandler(binary="tofu")
        node = MockComputeNode(
            input_mapping={
                "operation": "init",
            },
        )
        context = MockWorkflowContext()

        await handler(node, context, mock_registry)

        call_args = mock_registry.execute.call_args
        assert "tofu init" in call_args.kwargs["command"]


class TestHandlerRegistration:
    """Tests for handler registration."""

    def test_handlers_dict_exists(self):
        """Test HANDLERS dict is defined."""
        from victor_devops.handlers import HANDLERS

        assert "container_ops" in HANDLERS
        assert "terraform_apply" in HANDLERS

    def test_register_handlers(self):
        """Test handler registration function."""
        from victor_devops.handlers import register_handlers

        # Should not raise
        register_handlers()


class TestEscapeHatches:
    """Tests for DevOps escape hatch conditions."""

    def test_deployment_ready_all_valid(self):
        """Test deployment ready when all conditions met."""
        from victor_devops.escape_hatches import deployment_ready

        ctx = {
            "config_valid": True,
            "dependencies_met": True,
            "approval_status": "approved",
            "environment": "staging",
        }
        assert deployment_ready(ctx) == "ready"

    def test_deployment_ready_invalid_config(self):
        """Test deployment blocked when config invalid."""
        from victor_devops.escape_hatches import deployment_ready

        ctx = {
            "config_valid": False,
            "dependencies_met": True,
        }
        assert deployment_ready(ctx) == "failed"

    def test_deployment_ready_missing_dependencies(self):
        """Test deployment blocked when dependencies not met."""
        from victor_devops.escape_hatches import deployment_ready

        ctx = {
            "config_valid": True,
            "dependencies_met": False,
        }
        assert deployment_ready(ctx) == "blocked"

    def test_deployment_ready_production_needs_approval(self):
        """Test production deployment requires approval."""
        from victor_devops.escape_hatches import deployment_ready

        ctx = {
            "config_valid": True,
            "dependencies_met": True,
            "approval_status": "pending",
            "environment": "production",
        }
        assert deployment_ready(ctx) == "blocked"

    def test_health_check_status_healthy(self):
        """Test health check with all healthy endpoints."""
        from victor_devops.escape_hatches import health_check_status

        ctx = {
            "health_results": {
                "api": {"status": "healthy"},
                "db": {"status": "healthy"},
            },
            "min_healthy_pct": 0.8,
        }
        assert health_check_status(ctx) == "healthy"

    def test_health_check_status_degraded(self):
        """Test health check with some unhealthy endpoints."""
        from victor_devops.escape_hatches import health_check_status

        ctx = {
            "health_results": {
                "api": {"status": "healthy"},
                "db": {"status": "healthy"},
                "cache": {"status": "unhealthy"},
            },
            "min_healthy_pct": 0.6,
        }
        assert health_check_status(ctx) == "degraded"

    def test_rollback_needed_on_failure(self):
        """Test rollback needed when deployment fails."""
        from victor_devops.escape_hatches import rollback_needed

        ctx = {
            "deploy_result": {"success": False},
        }
        assert rollback_needed(ctx) == "rollback"

    def test_rollback_needed_high_error_rate(self):
        """Test rollback needed on high error rate."""
        from victor_devops.escape_hatches import rollback_needed

        ctx = {
            "deploy_result": {"success": True},
            "health_status": "healthy",
            "error_rate": 0.15,
        }
        assert rollback_needed(ctx) == "rollback"

    def test_container_build_status_success(self):
        """Test container build status success."""
        from victor_devops.escape_hatches import container_build_status

        ctx = {
            "build_result": {"success": True},
            "image_size": 500,
            "max_size": 2000,
        }
        assert container_build_status(ctx) == "success"

    def test_container_build_status_warning_on_size(self):
        """Test container build warning on large image."""
        from victor_devops.escape_hatches import container_build_status

        ctx = {
            "build_result": {"success": True},
            "image_size": 3000,
            "max_size": 2000,
        }
        assert container_build_status(ctx) == "warning"

    def test_security_scan_verdict_pass(self):
        """Test security scan verdict pass."""
        from victor_devops.escape_hatches import security_scan_verdict

        ctx = {
            "scan_results": {
                "critical": 0,
                "high": 0,
                "medium": 2,
            },
            "severity_threshold": "high",
        }
        assert security_scan_verdict(ctx) == "pass"

    def test_security_scan_verdict_fail_critical(self):
        """Test security scan verdict fail on critical."""
        from victor_devops.escape_hatches import security_scan_verdict

        ctx = {
            "scan_results": {
                "critical": 1,
                "high": 0,
                "medium": 0,
            },
        }
        assert security_scan_verdict(ctx) == "fail"


class TestMLOpsHandler:
    """Tests for MLOpsHandler."""

    @pytest.fixture
    def handler(self):
        from victor_devops.handlers import MLOpsHandler

        return MLOpsHandler()

    @pytest.fixture
    def mock_registry(self):
        return MagicMock()

    @pytest.mark.asyncio
    async def test_default_operation_is_register(self, handler, mock_registry):
        """Test default operation is register."""
        node = MockComputeNode(
            input_mapping={
                "model_name": "my_model",
                "model_path": "/path/to/model",
            },
        )
        context = MockWorkflowContext()

        result = await handler(node, context, mock_registry)

        # Will fail because mlflow is not installed
        assert result.status.value == "failed"
        assert "MLflow" in result.error or "mlflow" in result.error.lower()

    @pytest.mark.asyncio
    async def test_register_operation(self, handler, mock_registry):
        """Test register operation."""
        node = MockComputeNode(
            input_mapping={
                "operation": "register",
                "model_name": "test_model",
                "model_path": "/models/test.pkl",
                "metrics": {"accuracy": 0.95},
            },
            output_key="mlops_result",
        )
        context = MockWorkflowContext()

        result = await handler(node, context, mock_registry)

        assert result.status.value == "failed"
        assert context.get("mlops_result") is not None

    @pytest.mark.asyncio
    async def test_log_experiment_operation(self, handler, mock_registry):
        """Test log_experiment operation."""
        node = MockComputeNode(
            input_mapping={
                "operation": "log_experiment",
                "experiment_name": "my_experiment",
                "metrics": {"loss": 0.1, "accuracy": 0.95},
                "params": {"learning_rate": 0.001},
            },
        )
        context = MockWorkflowContext()

        result = await handler(node, context, mock_registry)

        assert result.status.value == "failed"

    @pytest.mark.asyncio
    async def test_serve_operation(self, handler, mock_registry):
        """Test serve operation."""
        node = MockComputeNode(
            input_mapping={
                "operation": "serve",
                "model_name": "production_model",
                "version": "1",
                "port": 5001,
            },
        )
        context = MockWorkflowContext()

        result = await handler(node, context, mock_registry)

        assert result.status.value == "failed"

    @pytest.mark.asyncio
    async def test_compare_operation(self, handler, mock_registry):
        """Test compare operation."""
        node = MockComputeNode(
            input_mapping={
                "operation": "compare",
                "model_name": "my_model",
            },
        )
        context = MockWorkflowContext()

        result = await handler(node, context, mock_registry)

        assert result.status.value == "failed"

    @pytest.mark.asyncio
    async def test_promote_operation(self, handler, mock_registry):
        """Test promote operation."""
        node = MockComputeNode(
            input_mapping={
                "operation": "promote",
                "model_name": "my_model",
                "version": "3",
                "stage": "Production",
            },
        )
        context = MockWorkflowContext()

        result = await handler(node, context, mock_registry)

        assert result.status.value == "failed"

    @pytest.mark.asyncio
    async def test_list_models_operation(self, handler, mock_registry):
        """Test list_models operation."""
        node = MockComputeNode(
            input_mapping={
                "operation": "list_models",
            },
        )
        context = MockWorkflowContext()

        result = await handler(node, context, mock_registry)

        assert result.status.value == "failed"

    @pytest.mark.asyncio
    async def test_unknown_operation(self, handler, mock_registry):
        """Test unknown operation returns error."""
        node = MockComputeNode(
            input_mapping={
                "operation": "invalid_operation",
            },
        )
        context = MockWorkflowContext()

        result = await handler(node, context, mock_registry)

        assert result.status.value == "failed"

    @pytest.mark.asyncio
    async def test_context_output_key(self, handler, mock_registry):
        """Test output key is set from node."""
        node = MockComputeNode(
            input_mapping={
                "operation": "register",
                "model_name": "test",
            },
            output_key="mlflow_result",
        )
        context = MockWorkflowContext()

        await handler(node, context, mock_registry)

        assert context.get("mlflow_result") is not None

    @pytest.mark.asyncio
    async def test_custom_tracking_uri(self, mock_registry):
        """Test custom tracking URI."""
        from victor_devops.handlers import MLOpsHandler

        handler = MLOpsHandler(tracking_uri="http://mlflow.example.com")
        node = MockComputeNode(
            input_mapping={
                "operation": "register",
                "model_name": "test",
            },
        )
        context = MockWorkflowContext()

        result = await handler(node, context, mock_registry)

        # Should fail with mlflow not installed, not tracking URI error
        assert result.status.value == "failed"


class TestMLOpsInRegistry:
    """Tests for MLOps handler in HANDLERS dict."""

    def test_mlops_handler_in_registry(self):
        """Test mlops handler is registered."""
        from victor_devops.handlers import HANDLERS

        assert "mlops" in HANDLERS

    def test_mlops_exported(self):
        """Test MLOpsHandler class is in __all__."""
        from victor_devops import handlers

        assert "MLOpsHandler" in handlers.__all__

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

"""DevOps vertical compute handlers.

Domain-specific handlers for DevOps workflows:
- container_ops: Docker/Podman container operations
- terraform_apply: Infrastructure as Code execution
- mlops: MLOps pipeline operations (MLflow, model registry, deployment)

Usage:
    from victor_devops import handlers
    handlers.register_handlers()

    # In YAML workflow:
    - id: build_image
      type: compute
      handler: container_ops
      inputs:
        operation: build
        dockerfile: Dockerfile
        tag: myapp:latest
      output: build_result

    # MLOps workflow:
    - id: register_model
      type: compute
      handler: mlops
      inputs:
        operation: register
        model_name: my_model
        model_path: $ctx.model_path
        metrics: $ctx.metrics
      output: registry_result
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional

from victor_contracts.workflows import (
    ComputeHandlerRegistrar,
    ExecutorNodeStatus,
    NodeResult,
    register_compute_handlers,
)

if TYPE_CHECKING:
    from victor_contracts.verticals.protocols.tools import (
        ToolRegistryProtocol as ToolRegistry,
    )
    from victor_contracts.workflows import ComputeNodeProtocol as ComputeNode
    from victor_contracts.workflows import WorkflowContextProtocol as WorkflowContext

logger = logging.getLogger(__name__)


@dataclass
class ContainerOpsHandler:
    """Docker/Podman container operations.

    Manages container lifecycle (build, run, stop, etc.).

    Example YAML:
        - id: build_image
          type: compute
          handler: container_ops
          inputs:
            operation: build
            dockerfile: Dockerfile
            tag: myapp:latest
          output: build_result
    """

    runtime: str = "docker"

    async def __call__(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> "NodeResult":
        start_time = time.time()

        operation = node.input_mapping.get("operation", "build")
        dockerfile = node.input_mapping.get("dockerfile", "Dockerfile")
        tag = node.input_mapping.get("tag", "latest")
        image = node.input_mapping.get("image", "")

        if operation == "build":
            cmd = f"{self.runtime} build -f {dockerfile} -t {tag} ."
        elif operation == "push":
            cmd = f"{self.runtime} push {tag}"
        elif operation == "pull":
            cmd = f"{self.runtime} pull {image}"
        elif operation == "run":
            cmd = f"{self.runtime} run -d {image}"
        elif operation == "stop":
            container_id = node.input_mapping.get("container_id", "")
            cmd = f"{self.runtime} stop {container_id}"
        else:
            return NodeResult(
                node_id=node.id,
                status=ExecutorNodeStatus.FAILED,
                error=f"Unknown operation: {operation}",
                duration_seconds=time.time() - start_time,
            )

        try:
            result = await tool_registry.execute("shell", command=cmd)

            output = {
                "operation": operation,
                "success": result.success,
                "output": result.output,
            }

            output_key = node.output_key or node.id
            context.set(output_key, output)

            return NodeResult(
                node_id=node.id,
                status=(
                    ExecutorNodeStatus.COMPLETED if result.success else ExecutorNodeStatus.FAILED
                ),
                output=output,
                duration_seconds=time.time() - start_time,
                tool_calls_used=1,
            )
        except Exception as e:
            return NodeResult(
                node_id=node.id,
                status=ExecutorNodeStatus.FAILED,
                error=str(e),
                duration_seconds=time.time() - start_time,
            )


@dataclass
class TerraformHandler:
    """Terraform/OpenTofu IaC operations.

    Manages infrastructure provisioning.

    Example YAML:
        - id: apply_infra
          type: compute
          handler: terraform_apply
          inputs:
            operation: apply
            workspace: production
            auto_approve: true
          output: terraform_result
    """

    binary: str = "terraform"

    async def __call__(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> "NodeResult":
        start_time = time.time()

        operation = node.input_mapping.get("operation", "plan")
        workspace = node.input_mapping.get("workspace")
        auto_approve = node.input_mapping.get("auto_approve", False)
        tool_calls = 0

        if workspace:
            await tool_registry.execute(
                "shell", command=f"{self.binary} workspace select {workspace}"
            )
            tool_calls += 1

        if operation == "init":
            cmd = f"{self.binary} init"
        elif operation == "plan":
            cmd = f"{self.binary} plan -out=tfplan"
        elif operation == "apply":
            cmd = f"{self.binary} apply"
            if auto_approve:
                cmd += " -auto-approve"
            else:
                cmd += " tfplan"
        elif operation == "destroy":
            cmd = f"{self.binary} destroy"
            if auto_approve:
                cmd += " -auto-approve"
        else:
            return NodeResult(
                node_id=node.id,
                status=ExecutorNodeStatus.FAILED,
                error=f"Unknown operation: {operation}",
                duration_seconds=time.time() - start_time,
            )

        try:
            result = await tool_registry.execute("shell", command=cmd)
            tool_calls += 1

            output = {
                "operation": operation,
                "workspace": workspace,
                "success": result.success,
                "output": result.output,
            }

            output_key = node.output_key or node.id
            context.set(output_key, output)

            return NodeResult(
                node_id=node.id,
                status=(
                    ExecutorNodeStatus.COMPLETED if result.success else ExecutorNodeStatus.FAILED
                ),
                output=output,
                duration_seconds=time.time() - start_time,
                tool_calls_used=tool_calls,
            )
        except Exception as e:
            return NodeResult(
                node_id=node.id,
                status=ExecutorNodeStatus.FAILED,
                error=str(e),
                duration_seconds=time.time() - start_time,
            )


@dataclass
class MLOpsHandler:
    """MLOps pipeline operations handler.

    Supports model lifecycle management using MLflow:
    - Model registration and versioning
    - Experiment tracking
    - Model deployment (local serving)
    - Model comparison and promotion

    Example YAML:
        - id: register_model
          type: compute
          handler: mlops
          inputs:
            operation: register
            model_name: my_classifier
            model_path: models/classifier.pkl
            metrics:
              accuracy: 0.95
              f1_score: 0.93
          output: registry_result

    Supported operations:
        - register: Register a model with metrics
        - log_experiment: Log experiment run
        - serve: Start model serving endpoint
        - compare: Compare model versions
        - promote: Promote model to production stage
        - list_models: List registered models

    Output varies by operation:
        - register: {model_uri, version, run_id}
        - log_experiment: {run_id, experiment_id}
        - serve: {endpoint_url, status}
        - compare: {comparison_table, best_model}
        - promote: {model_name, stage, version}
    """

    tracking_uri: str = "mlruns"

    async def __call__(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> "NodeResult":
        start_time = time.time()

        operation = node.input_mapping.get("operation", "register")
        model_name = node.input_mapping.get("model_name")
        model_path = node.input_mapping.get("model_path")
        metrics = node.input_mapping.get("metrics", {})
        params = node.input_mapping.get("params", {})
        experiment_name = node.input_mapping.get("experiment_name", "default")
        stage = node.input_mapping.get("stage", "Staging")
        version = node.input_mapping.get("version")
        port = node.input_mapping.get("port", 5001)

        try:
            result = await self._run_mlops(
                operation=operation,
                model_name=model_name,
                model_path=model_path,
                metrics=metrics,
                params=params,
                experiment_name=experiment_name,
                stage=stage,
                version=version,
                port=port,
            )

            output_key = node.output_key or node.id
            context.set(output_key, result)

            return NodeResult(
                node_id=node.id,
                status=(
                    ExecutorNodeStatus.COMPLETED
                    if result.get("success")
                    else ExecutorNodeStatus.FAILED
                ),
                output=result,
                duration_seconds=time.time() - start_time,
                error=result.get("error"),
            )

        except Exception as e:
            logger.exception(f"MLOps operation failed: {e}")
            return NodeResult(
                node_id=node.id,
                status=ExecutorNodeStatus.FAILED,
                error=str(e),
                duration_seconds=time.time() - start_time,
            )

    async def _run_mlops(
        self,
        operation: str,
        model_name: str,
        model_path: str,
        metrics: Dict[str, Any],
        params: Dict[str, Any],
        experiment_name: str,
        stage: str,
        version: str,
        port: int,
    ) -> Dict[str, Any]:
        """Run MLOps operation asynchronously."""
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._run_mlops_sync,
            operation,
            model_name,
            model_path,
            metrics,
            params,
            experiment_name,
            stage,
            version,
            port,
        )

    def _run_mlops_sync(
        self,
        operation: str,
        model_name: str,
        model_path: str,
        metrics: Dict[str, Any],
        params: Dict[str, Any],
        experiment_name: str,
        stage: str,
        version: str,
        port: int,
    ) -> Dict[str, Any]:
        """Synchronous MLOps execution."""
        try:
            import mlflow
            from mlflow.tracking import MlflowClient

            mlflow.set_tracking_uri(self.tracking_uri)
            client = MlflowClient()

            if operation == "register":
                return self._register_model(mlflow, client, model_name, model_path, metrics, params)
            elif operation == "log_experiment":
                return self._log_experiment(mlflow, experiment_name, metrics, params)
            elif operation == "serve":
                return self._serve_model(model_name, version, port)
            elif operation == "compare":
                return self._compare_models(client, model_name)
            elif operation == "promote":
                return self._promote_model(client, model_name, version, stage)
            elif operation == "list_models":
                return self._list_models(client)
            else:
                return {
                    "success": False,
                    "error": (
                        f"Unknown operation: {operation}. Use: register, "
                        "log_experiment, serve, compare, promote, list_models"
                    ),
                }

        except ImportError as e:
            return {
                "success": False,
                "error": f"MLflow not installed. Install with: pip install mlflow. Error: {e}",
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"MLOps operation failed: {e}",
            }

    def _register_model(
        self,
        mlflow,
        client,
        model_name: str,
        model_path: str,
        metrics: Dict,
        params: Dict,
    ) -> Dict[str, Any]:
        """Register a model with MLflow."""
        with mlflow.start_run() as run:
            # Log parameters
            for key, value in params.items():
                mlflow.log_param(key, value)

            # Log metrics
            for key, value in metrics.items():
                mlflow.log_metric(key, value)

            # Log model
            if model_path:
                mlflow.log_artifact(model_path)

            # Register model
            model_uri = f"runs:/{run.info.run_id}/model"
            result = mlflow.register_model(model_uri, model_name)

            return {
                "success": True,
                "operation": "register",
                "model_name": model_name,
                "model_uri": model_uri,
                "version": result.version,
                "run_id": run.info.run_id,
            }

    def _log_experiment(
        self, mlflow, experiment_name: str, metrics: Dict, params: Dict
    ) -> Dict[str, Any]:
        """Log an experiment run."""
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run() as run:
            for key, value in params.items():
                mlflow.log_param(key, value)
            for key, value in metrics.items():
                mlflow.log_metric(key, value)

            return {
                "success": True,
                "operation": "log_experiment",
                "experiment_name": experiment_name,
                "run_id": run.info.run_id,
                "experiment_id": run.info.experiment_id,
            }

    def _serve_model(self, model_name: str, version: str, port: int) -> Dict[str, Any]:
        """Start model serving (returns command, doesn't actually start)."""
        model_uri = f"models:/{model_name}/{version or 'latest'}"
        serve_cmd = f"mlflow models serve -m {model_uri} -p {port} --no-conda"

        return {
            "success": True,
            "operation": "serve",
            "model_name": model_name,
            "version": version or "latest",
            "port": port,
            "command": serve_cmd,
            "endpoint_url": f"http://localhost:{port}/invocations",
            "note": (
                "Run the command to start serving. Use "
                "tool_registry.execute('shell', command=...) to start."
            ),
        }

    def _compare_models(self, client, model_name: str) -> Dict[str, Any]:
        """Compare versions of a registered model."""
        try:
            versions = client.search_model_versions(f"name='{model_name}'")
            comparison = []

            for v in versions:
                run = client.get_run(v.run_id)
                comparison.append(
                    {
                        "version": v.version,
                        "stage": v.current_stage,
                        "run_id": v.run_id,
                        "metrics": run.data.metrics,
                        "status": v.status,
                    }
                )

            # Find best version by first metric
            best_version = None
            if comparison and comparison[0].get("metrics"):
                first_metric = list(comparison[0]["metrics"].keys())[0]
                best_version = max(comparison, key=lambda x: x["metrics"].get(first_metric, 0))

            return {
                "success": True,
                "operation": "compare",
                "model_name": model_name,
                "versions": comparison,
                "best_version": best_version,
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Model comparison failed: {e}",
            }

    def _promote_model(self, client, model_name: str, version: str, stage: str) -> Dict[str, Any]:
        """Promote a model version to a stage."""
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
        )

        return {
            "success": True,
            "operation": "promote",
            "model_name": model_name,
            "version": version,
            "stage": stage,
        }

    def _list_models(self, client) -> Dict[str, Any]:
        """List all registered models."""
        models = client.search_registered_models()

        return {
            "success": True,
            "operation": "list_models",
            "models": [
                {
                    "name": m.name,
                    "latest_versions": [
                        {"version": v.version, "stage": v.current_stage} for v in m.latest_versions
                    ],
                }
                for m in models
            ],
            "count": len(models),
        }


HANDLERS = {
    "container_ops": ContainerOpsHandler(),
    "terraform_apply": TerraformHandler(),
    "mlops": MLOpsHandler(),
}


def register_handlers(
    registrar: Optional[ComputeHandlerRegistrar] = None,
):
    """Register DevOps handlers through an explicit host-side registrar."""
    registered = register_compute_handlers(registrar, HANDLERS)
    if registrar is not None:
        for name in HANDLERS:
            logger.debug(f"Registered DevOps handler: {name}")
    return registered


__all__ = [
    "ContainerOpsHandler",
    "TerraformHandler",
    "MLOpsHandler",
    "HANDLERS",
    "register_handlers",
]

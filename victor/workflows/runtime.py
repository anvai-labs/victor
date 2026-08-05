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

"""Workflow Runtime - Serverless-like Workflow Execution.

Provides a complete runtime that manages the full lifecycle:
1. Start services (PostgreSQL, Redis, etc.)
2. Prepare deployment target (Docker, K8s, etc.)
3. Execute workflow
4. Cleanup and stop services

This gives a "serverless" experience where services spin up on-demand
and shut down after workflow completion.

Example:
    from victor.workflows import WorkflowRuntime, RuntimeConfig
    from victor.workflows.deployment import DeploymentTarget, DockerConfig
    from victor.workflows.services import ServiceConfig

    # Configure runtime
    config = RuntimeConfig(
        services=[
            ServiceConfig(name="db", service_type="postgres", config={"host": "localhost"}),
            ServiceConfig(name="cache", service_type="redis"),
        ],
        deployment=DeploymentConfig(
            target=DeploymentTarget.DOCKER,
            docker=DockerConfig(image="victor-worker:latest"),
        ),
    )

    # Create runtime
    runtime = WorkflowRuntime(config)

    # Execute workflow (services start, execute, services stop)
    result = await runtime.execute(workflow, {"input": "data"})

    # Or use as context manager
    async with runtime as rt:
        result1 = await rt.execute(workflow1, {"input": "data1"})
        result2 = await rt.execute(workflow2, {"input": "data2"})
    # Services automatically cleaned up
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    List,
    Optional,
)

from victor.workflows.deployment import (
    DeploymentConfig,
    DeploymentHandler,
    DeploymentTarget,
    get_deployment_handler,
)
from victor.workflows.service_lifecycle import (
    ServiceConfig,
    ServiceManager,
)
from victor.workflows.unified_executor import (
    ExecutorConfig,
    ExecutorResult,
    StateGraphExecutor,
)

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.tools.registry import ToolRegistry
    from victor.workflows.definition import WorkflowDefinition

logger = logging.getLogger(__name__)

# Lazy-loaded presentation adapter for icon rendering
_presentation = None


def _get_icon(name: str) -> str:
    """Get icon from presentation adapter."""
    global _presentation
    if _presentation is None:
        from victor.agent.presentation import create_presentation_adapter

        _presentation = create_presentation_adapter()
    return _presentation.icon(name, with_color=False)


@dataclass
class HITLServerConfig:
    """Configuration for the HITL (Human-in-the-Loop) server.

    When workflows contain HITL nodes, a web server is started
    to provide a UI for approvals.

    Attributes:
        enabled: Enable HITL server (auto-detect if None)
        host: Host to bind HITL server to
        port: Port for HITL server
        require_auth: Require authentication for HITL API
        auth_token: Bearer token for authentication
        keep_running_after_complete: Keep server running after workflow completes
    """

    enabled: Optional[bool] = None  # None = auto-detect based on workflow
    host: str = "0.0.0.0"
    port: int = 8080
    require_auth: bool = False
    auth_token: Optional[str] = None
    keep_running_after_complete: bool = True  # For audit review


@dataclass
class RuntimeConfig:
    """Configuration for the Workflow Runtime.

    Attributes:
        services: List of services to start/stop with workflow
        deployment: Deployment target configuration
        executor: Executor configuration
        hitl: HITL server configuration
        start_services_on_init: Start services when runtime initializes
        stop_services_on_complete: Stop services after each workflow
        health_check_interval: Interval for service health checks (seconds)
        max_health_check_failures: Max consecutive health failures before abort
    """

    services: List[ServiceConfig] = field(default_factory=list)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)
    executor: ExecutorConfig = field(default_factory=ExecutorConfig)
    hitl: HITLServerConfig = field(default_factory=HITLServerConfig)
    start_services_on_init: bool = False
    stop_services_on_complete: bool = True
    health_check_interval: float = 30.0
    max_health_check_failures: int = 3


@dataclass
class RuntimeResult:
    """Result from workflow runtime execution.

    Includes execution result plus runtime metadata.
    """

    execution: ExecutorResult
    services_started: List[str]
    services_healthy: Dict[str, bool]
    deployment_target: DeploymentTarget
    total_duration_seconds: float
    hitl_server_url: Optional[str] = None  # URL of HITL server if started

    @property
    def success(self) -> bool:
        """Whether execution succeeded."""
        return self.execution.success

    @property
    def state(self) -> Dict[str, Any]:
        """Final workflow state."""
        return self.execution.state

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from final state."""
        return self.execution.get(key, default)


class WorkflowRuntime:
    """Serverless-like Workflow Runtime.

    Manages the complete lifecycle of workflow execution:
    - Service initialization and cleanup
    - Deployment target preparation
    - Workflow execution
    - Health monitoring

    Can be used as a context manager for automatic cleanup.

    Example:
        # One-shot execution (services start/stop per workflow)
        runtime = WorkflowRuntime(config)
        result = await runtime.execute(workflow, {"input": "data"})

        # Persistent runtime (services stay up across workflows)
        config = RuntimeConfig(
            start_services_on_init=True,
            stop_services_on_complete=False,
        )
        runtime = WorkflowRuntime(config)
        await runtime.start()
        result1 = await runtime.execute(workflow1, {})
        result2 = await runtime.execute(workflow2, {})
        await runtime.stop()

        # Context manager (auto cleanup)
        async with WorkflowRuntime(config) as rt:
            result = await rt.execute(workflow, {})
    """

    def __init__(
        self,
        config: Optional[RuntimeConfig] = None,
        orchestrator: Optional["AgentOrchestrator"] = None,
        tool_registry: Optional["ToolRegistry"] = None,
    ):
        """Initialize the workflow runtime.

        Args:
            config: Runtime configuration
            orchestrator: Agent orchestrator for agent nodes
            tool_registry: Tool registry for compute nodes
        """
        self.config = config or RuntimeConfig()
        self.orchestrator = orchestrator
        self.tool_registry = tool_registry

        self._service_manager = ServiceManager()
        self._deployment_handler: Optional[DeploymentHandler] = None
        self._executor: Optional[StateGraphExecutor] = None
        self._services_running = False
        self._service_handles: Dict[str, Any] = {}

        # HITL server management
        self._hitl_server_task: Optional[asyncio.Task] = None
        self._hitl_server_running = False
        self._hitl_store: Optional[Any] = None
        self._hitl_url: Optional[str] = None

    async def start(self) -> None:
        """Start the runtime (initialize services and deployment).

        Call this manually if not using context manager and want
        persistent services across multiple workflow executions.
        """
        if self._services_running:
            return

        logger.info("Starting workflow runtime...")

        # Initialize services
        if self.config.services:
            logger.info(f"Starting {len(self.config.services)} services...")
            self._service_handles = await self._service_manager.initialize_services(
                self.config.services
            )
            logger.info("Services started successfully")

        # Initialize deployment handler
        self._deployment_handler = get_deployment_handler(self.config.deployment)

        # Initialize executor
        self._executor = StateGraphExecutor(
            orchestrator=self.orchestrator,
            tool_registry=self.tool_registry,
            config=self.config.executor,
        )

        self._services_running = True
        logger.info("Workflow runtime started")

    async def stop(self) -> None:
        """Stop the runtime (cleanup services and deployment)."""
        if not self._services_running:
            return

        logger.info("Stopping workflow runtime...")

        # Cleanup services
        await self._service_manager.cleanup_services()
        self._service_handles = {}

        # Cleanup deployment
        if self._deployment_handler:
            await self._deployment_handler.cleanup()

        self._services_running = False
        logger.info("Workflow runtime stopped")

    async def execute(
        self,
        workflow: "WorkflowDefinition",
        initial_context: Optional[Dict[str, Any]] = None,
        *,
        thread_id: Optional[str] = None,
    ) -> RuntimeResult:
        """Execute a workflow with full lifecycle management.

        If start_services_on_init is False (default), this method will:
        1. Start services (including HITL server if needed)
        2. Execute workflow
        3. Stop services (if stop_services_on_complete is True)

        HITL nodes in the workflow will automatically use the HITL store
        and server started by this runtime.

        Args:
            workflow: The workflow to execute
            initial_context: Initial context/state data
            thread_id: Thread ID for checkpointing

        Returns:
            RuntimeResult with execution outcome and metadata
        """
        start_time = time.time()
        services_started: List[str] = []
        hitl_started = False

        try:
            # Start services if not already running
            if not self._services_running:
                await self.start()
                services_started = list(self._service_handles.keys())

            # Auto-start HITL server if workflow has HITL nodes
            should_start_hitl = self.config.hitl.enabled
            if should_start_hitl is None:
                # Auto-detect
                should_start_hitl = self._has_hitl_nodes(workflow)

            if should_start_hitl and not self._hitl_server_running:
                await self._start_hitl_server()
                hitl_started = True

            # Add service handles and HITL info to context
            context = dict(initial_context or {})
            if self._service_handles:
                context["_services"] = self._service_handles

            # Add HITL store and URL to context for HITL nodes
            if self._hitl_store:
                context["_hitl_store"] = self._hitl_store
                context["_hitl_url"] = self._hitl_url
                context["_hitl_api_base"] = (
                    f"http://{self.config.hitl.host}:{self.config.hitl.port}/hitl"
                    if self.config.hitl.host != "0.0.0.0"
                    else f"http://localhost:{self.config.hitl.port}/hitl"
                )

            # Health check services
            services_healthy = await self._service_manager.health_check_all()

            # Prepare deployment
            if self._deployment_handler:
                await self._deployment_handler.prepare(workflow, self.config.deployment)

            # Execute workflow
            if self._executor is None:
                self._executor = StateGraphExecutor(
                    orchestrator=self.orchestrator,
                    tool_registry=self.tool_registry,
                    config=self.config.executor,
                )

            result = await self._executor.execute(
                workflow,
                context,
                thread_id=thread_id,
            )

            return RuntimeResult(
                execution=result,
                services_started=services_started,
                services_healthy=services_healthy,
                deployment_target=self.config.deployment.target,
                total_duration_seconds=time.time() - start_time,
                hitl_server_url=self._hitl_url,
            )

        finally:
            # Stop services if configured
            if self.config.stop_services_on_complete and self._services_running:
                await self.stop()

            # Stop HITL server unless configured to keep running
            if hitl_started and not self.config.hitl.keep_running_after_complete:
                await self._stop_hitl_server()

    async def stream(
        self,
        workflow: "WorkflowDefinition",
        initial_context: Optional[Dict[str, Any]] = None,
        *,
        thread_id: Optional[str] = None,
    ) -> AsyncIterator[tuple]:
        """Stream workflow execution with lifecycle management.

        Args:
            workflow: The workflow to execute
            initial_context: Initial context/state data
            thread_id: Thread ID for checkpointing

        Yields:
            Tuple of (node_id, current_state) after each node
        """
        try:
            # Start services if not already running
            if not self._services_running:
                await self.start()

            # Add service handles to context
            context = dict(initial_context or {})
            if self._service_handles:
                context["_services"] = self._service_handles

            # Prepare deployment
            if self._deployment_handler:
                await self._deployment_handler.prepare(workflow, self.config.deployment)

            # Stream execution
            if self._executor is None:
                self._executor = StateGraphExecutor(
                    orchestrator=self.orchestrator,
                    tool_registry=self.tool_registry,
                    config=self.config.executor,
                )

            async for node_id, state in self._executor.stream(
                workflow,
                context,
                thread_id=thread_id,
            ):
                yield (node_id, state)

        finally:
            # Stop services if configured
            if self.config.stop_services_on_complete and self._services_running:
                await self.stop()

    # =========================================================================
    # HITL Server Management
    # =========================================================================

    def _has_hitl_nodes(self, workflow: "WorkflowDefinition") -> bool:
        """Check if workflow contains HITL nodes.

        Args:
            workflow: The workflow definition

        Returns:
            True if workflow has HITL nodes
        """
        from victor.workflows.definition import WorkflowNodeType

        # workflow.nodes is a Dict[str, WorkflowNode]
        nodes = workflow.nodes.values() if isinstance(workflow.nodes, dict) else workflow.nodes
        for node in nodes:
            # Check node type
            if hasattr(node, "node_type"):
                if node.node_type == WorkflowNodeType.HITL:
                    return True
            # Check if it's a HITLNode instance
            if node.__class__.__name__ == "HITLNode":
                return True
        return False

    async def _start_hitl_server(self) -> None:
        """Start the HITL server in the background."""
        if self._hitl_server_running:
            return

        try:
            from victor.workflows.hitl_api import (
                SQLiteHITLStore,
                create_hitl_app,
            )

            # Create SQLite store for persistence
            self._hitl_store = SQLiteHITLStore()

            # Create the HITL FastAPI app
            app = create_hitl_app(
                store=self._hitl_store,
                require_auth=self.config.hitl.require_auth,
                auth_token=self.config.hitl.auth_token,
            )

            # Start uvicorn server in background
            import uvicorn

            config = uvicorn.Config(
                app,
                host=self.config.hitl.host,
                port=self.config.hitl.port,
                log_level="warning",  # Reduce noise
            )
            server = uvicorn.Server(config)

            # Run server in background task
            self._hitl_server_task = asyncio.create_task(server.serve())
            self._hitl_server_running = True

            # Build URL
            host = self.config.hitl.host
            if host == "0.0.0.0":
                host = "localhost"
            self._hitl_url = f"http://{host}:{self.config.hitl.port}/hitl/ui"

            # Give server time to start
            await asyncio.sleep(0.5)

            logger.info(f"HITL server started at {self._hitl_url}")
            print(f"\n{_get_icon('clipboard')} HITL Approval UI: {self._hitl_url}")
            print(f"   API Docs: http://{host}:{self.config.hitl.port}/docs\n")

        except ImportError as e:
            logger.warning(f"HITL server not available: {e}")
        except Exception as e:
            logger.error(f"Failed to start HITL server: {e}")

    async def _stop_hitl_server(self) -> None:
        """Stop the HITL server."""
        if not self._hitl_server_running:
            return

        if self._hitl_server_task:
            self._hitl_server_task.cancel()
            try:
                await self._hitl_server_task
            except asyncio.CancelledError:
                pass

        self._hitl_server_running = False
        self._hitl_server_task = None
        logger.info("HITL server stopped")

    def get_hitl_store(self) -> Optional[Any]:
        """Get the HITL store for this runtime.

        Use this to access the HITL store for creating requests
        or querying history.

        Returns:
            HITL store or None if HITL not enabled
        """
        return self._hitl_store

    def get_hitl_url(self) -> Optional[str]:
        """Get the HITL UI URL.

        Returns:
            HITL UI URL or None if not running
        """
        return self._hitl_url

    def get_service(self, name: str) -> Optional[Any]:
        """Get a service handle by name.

        Args:
            name: Service name

        Returns:
            Service handle or None
        """
        return self._service_manager.get_service(name)

    async def health_check(self) -> Dict[str, bool]:
        """Check health of all services.

        Returns:
            Dictionary of service name -> health status
        """
        return await self._service_manager.health_check_all()

    async def __aenter__(self) -> "WorkflowRuntime":
        """Enter context manager - start runtime."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager - stop runtime."""
        await self.stop()


# Convenience function
async def run_workflow(
    workflow: "WorkflowDefinition",
    initial_context: Optional[Dict[str, Any]] = None,
    *,
    services: Optional[List[ServiceConfig]] = None,
    deployment: Optional[DeploymentConfig] = None,
    thread_id: Optional[str] = None,
) -> RuntimeResult:
    """Execute a workflow with optional services and deployment.

    Convenience function for one-shot workflow execution.

    Args:
        workflow: The workflow to execute
        initial_context: Initial context/state data
        services: Services to start for this workflow
        deployment: Deployment target configuration
        thread_id: Thread ID for checkpointing

    Returns:
        RuntimeResult with execution outcome
    """
    config = RuntimeConfig(
        services=services or [],
        deployment=deployment or DeploymentConfig(),
    )
    runtime = WorkflowRuntime(config)
    return await runtime.execute(workflow, initial_context, thread_id=thread_id)


__all__ = [
    "RuntimeConfig",
    "RuntimeResult",
    "WorkflowRuntime",
    "run_workflow",
]

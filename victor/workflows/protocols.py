"""Workflow protocol definitions for graph-based workflow execution.

This module defines the core protocols (interfaces) for the workflow system,
following SOLID principles and enabling LangGraph-like workflow capabilities.

Protocols defined:
- IWorkflowNode: Interface for workflow nodes (steps)
- IWorkflowEdge: Interface for edges connecting nodes
- IWorkflowGraph: Interface for the workflow graph structure
- ICheckpointStore: Interface for state persistence
- IWorkflowExecutor: Interface for workflow execution
- NodeRunner: Protocol for node execution strategies (ISP + DIP compliant)
- IWorkflowCompiler: Interface for workflow compilation (DIP compliant)
- IWorkflowLoader: Interface for loading workflow definitions (DIP compliant)
- IWorkflowValidator: Interface for validating workflow definitions (DIP compliant)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    Type,
    runtime_checkable,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from victor.workflows.streaming import WorkflowStreamChunk


class ProtocolNodeStatus(Enum):
    """Status of a workflow protocol node execution.

    Renamed from NodeStatus to be semantically distinct:
    - ProtocolNodeStatus (here): Workflow protocol node status
    - FrameworkNodeStatus (victor.framework.graph): Framework graph node status
    - ExecutorNodeStatus (victor.workflows.executor): Executor node status

    Attributes:
        PENDING: Node has not started execution.
        RUNNING: Node is currently executing.
        COMPLETED: Node finished successfully.
        FAILED: Node execution failed.
        SKIPPED: Node was skipped (e.g., due to conditional edge).
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class RetryPolicy:
    """Configuration for node retry behavior.

    Attributes:
        max_retries: Maximum number of retry attempts.
        delay_seconds: Base delay between retries.
        exponential_backoff: Whether to use exponential backoff.
        retry_on_exceptions: Tuple of exception types to retry on.
    """

    max_retries: int = 3
    delay_seconds: float = 1.0
    exponential_backoff: bool = True
    retry_on_exceptions: Tuple[Type[Exception], ...] = (Exception,)


@dataclass
class NodeResult:
    """Result of a node execution.

    Attributes:
        status: The execution status.
        output: The output data from the node.
        error: The exception if the node failed.
        metadata: Additional metadata about the execution.
    """

    status: ProtocolNodeStatus
    output: Any = None
    error: Optional[Exception] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class IWorkflowNode(Protocol):
    """Protocol for workflow nodes.

    A workflow node represents a single step in a workflow graph.
    Nodes are executed by the workflow executor and can produce
    outputs that affect workflow state.

    Example:
        class MyNode:
            @property
            def id(self) -> str:
                return "my_node"

            @property
            def name(self) -> str:
                return "My Custom Node"

            @property
            def retry_policy(self) -> RetryPolicy:
                return RetryPolicy(max_retries=3)

            async def execute(
                self,
                state: Dict[str, Any],
                context: Optional[Dict[str, Any]] = None
            ) -> NodeResult:
                # Process state and return result
                return NodeResult(
                    status=ProtocolNodeStatus.COMPLETED,
                    output={"processed": True}
                )
    """

    @property
    def id(self) -> str:
        """Unique identifier for the node."""
        ...

    @property
    def name(self) -> str:
        """Human-readable name for the node."""
        ...

    @property
    def retry_policy(self) -> RetryPolicy:
        """Retry policy for this node."""
        ...

    async def execute(
        self, state: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> NodeResult:
        """Execute the node with the given state.

        Args:
            state: Current workflow state.
            context: Optional execution context.

        Returns:
            NodeResult with status and output.
        """
        ...


@runtime_checkable
class IWorkflowEdge(Protocol):
    """Protocol for workflow edges.

    An edge connects two nodes and can include conditional logic
    to determine whether traversal should occur based on state.

    Example:
        class MyEdge:
            @property
            def source_id(self) -> str:
                return "node_a"

            @property
            def target_id(self) -> str:
                return "node_b"

            def should_traverse(self, state: Dict[str, Any]) -> bool:
                return state.get("continue", True)
    """

    @property
    def source_id(self) -> str:
        """ID of the source node."""
        ...

    @property
    def target_id(self) -> str:
        """ID of the target node."""
        ...

    def should_traverse(self, state: Dict[str, Any]) -> bool:
        """Determine if this edge should be traversed.

        Args:
            state: Current workflow state.

        Returns:
            True if the edge should be traversed.
        """
        ...


@runtime_checkable
class IWorkflowGraph(Protocol):
    """Protocol for workflow graphs.

    A workflow graph is a directed graph of nodes connected by edges.
    It defines the structure and flow of a workflow.

    Example:
        graph = MyGraph()
        graph.add_node(start_node)
        graph.add_node(process_node)
        graph.add_edge(simple_edge)
        errors = graph.validate()
    """

    def add_node(self, node: IWorkflowNode) -> "IWorkflowGraph":
        """Add a node to the graph.

        Args:
            node: The node to add.

        Returns:
            Self for fluent interface.
        """
        ...

    def add_edge(self, edge: IWorkflowEdge) -> "IWorkflowGraph":
        """Add an edge to the graph.

        Args:
            edge: The edge to add.

        Returns:
            Self for fluent interface.
        """
        ...

    def get_node(self, node_id: str) -> Optional[IWorkflowNode]:
        """Get a node by its ID.

        Args:
            node_id: The node's unique identifier.

        Returns:
            The node if found, None otherwise.
        """
        ...

    def get_entry_node(self) -> Optional[IWorkflowNode]:
        """Get the entry node of the graph.

        Returns:
            The entry node if set, None otherwise.
        """
        ...

    def get_next_nodes(self, node_id: str, state: Dict[str, Any]) -> List[IWorkflowNode]:
        """Get the next nodes to execute after the given node.

        Args:
            node_id: The current node's ID.
            state: Current workflow state (for conditional edges).

        Returns:
            List of nodes to execute next.
        """
        ...

    def validate(self) -> List[str]:
        """Validate the graph structure.

        Returns:
            List of validation error messages (empty if valid).
        """
        ...


@runtime_checkable
class ICheckpointStore(Protocol):
    """Protocol for checkpoint storage.

    A checkpoint store persists workflow state for resumption,
    debugging, and fault tolerance.

    Example:
        store = MyCheckpointStore()
        await store.save("wf_123", "cp_1", {"step": 3, "data": [1,2,3]})
        state = await store.load("wf_123", "cp_1")
    """

    async def save(
        self,
        workflow_id: str,
        checkpoint_id: str,
        state: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save a checkpoint.

        Args:
            workflow_id: Unique workflow instance identifier.
            checkpoint_id: Unique checkpoint identifier.
            state: The state to save.
            metadata: Optional metadata about the checkpoint.
        """
        ...

    async def load(self, workflow_id: str, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Load a checkpoint.

        Args:
            workflow_id: Unique workflow instance identifier.
            checkpoint_id: Unique checkpoint identifier.

        Returns:
            The saved state if found, None otherwise.
        """
        ...

    async def list_checkpoints(self, workflow_id: str) -> List[str]:
        """List all checkpoints for a workflow.

        Args:
            workflow_id: Unique workflow instance identifier.

        Returns:
            List of checkpoint IDs.
        """
        ...

    async def delete(self, workflow_id: str, checkpoint_id: str) -> bool:
        """Delete a checkpoint.

        Args:
            workflow_id: Unique workflow instance identifier.
            checkpoint_id: Unique checkpoint identifier.

        Returns:
            True if deleted, False if not found.
        """
        ...


@runtime_checkable
class IWorkflowExecutor(Protocol):
    """Protocol for workflow execution.

    A workflow executor runs a workflow graph, managing state
    transitions, error handling, and checkpointing.

    Example:
        executor = MyExecutor()
        result = await executor.execute(graph, {"input": "data"})
    """

    async def execute(
        self,
        graph: IWorkflowGraph,
        initial_state: Dict[str, Any],
        checkpoint_store: Optional[ICheckpointStore] = None,
    ) -> Dict[str, Any]:
        """Execute a workflow graph.

        Args:
            graph: The workflow graph to execute.
            initial_state: Initial workflow state.
            checkpoint_store: Optional store for checkpointing.

        Returns:
            Final workflow state.
        """
        ...

    async def resume(
        self,
        graph: IWorkflowGraph,
        checkpoint_store: ICheckpointStore,
        workflow_id: str,
        checkpoint_id: str,
    ) -> Dict[str, Any]:
        """Resume a workflow from a checkpoint.

        Args:
            graph: The workflow graph to execute.
            checkpoint_store: Store containing the checkpoint.
            workflow_id: Workflow instance identifier.
            checkpoint_id: Checkpoint to resume from.

        Returns:
            Final workflow state.
        """
        ...

    def cancel(self) -> None:
        """Cancel the currently executing workflow."""
        ...


@runtime_checkable
class IWorkflowRuntimeExecutor(Protocol):
    """Protocol for workflow-definition runtime executors.

    This protocol models the higher-level executor shape used by the legacy
    workflow runtime and surrounding utility modules: execute a
    ``WorkflowDefinition``-style object from an initial context dictionary with
    implementation-specific keyword arguments.
    """

    async def execute(
        self,
        workflow: Any,
        initial_context: Dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        """Execute a workflow definition from an initial context."""
        ...


@runtime_checkable
class IWorkflowNodeRuntimeExecutor(Protocol):
    """Protocol for executors that can run individual workflow nodes."""

    async def execute_node(
        self,
        node: Any,
        context: Dict[str, Any],
    ) -> Any:
        """Execute a single workflow node against the provided context."""
        ...


@runtime_checkable
class IStreamingWorkflowExecutor(Protocol):
    """Protocol for streaming workflow execution.

    A streaming workflow executor provides real-time visibility into
    workflow execution through async iteration and observer patterns.

    Example:
        executor = MyStreamingExecutor()

        # Async iteration
        async for chunk in executor.astream(graph, initial_state):
            print(f"{chunk.event_type}: {chunk.content}")

        # Observer pattern
        def on_chunk(chunk):
            print(f"Received: {chunk.event_type}")

        unsubscribe = executor.subscribe(on_chunk)
        # ... later
        unsubscribe()

        # Cancellation
        executor.cancel_workflow("wf_123")
    """

    async def astream(
        self,
        graph: IWorkflowGraph,
        initial_state: Dict[str, Any],
        checkpoint_store: Optional[ICheckpointStore] = None,
    ) -> AsyncIterator["WorkflowStreamChunk"]:
        """Stream workflow execution events.

        Args:
            graph: The workflow graph to execute.
            initial_state: Initial workflow state.
            checkpoint_store: Optional store for checkpointing.

        Yields:
            WorkflowStreamChunk events as the workflow executes.

        Example:
            async for chunk in executor.astream(graph, {"input": "data"}):
                if chunk.event_type == WorkflowEventType.AGENT_CONTENT:
                    print(chunk.content, end="")
        """
        # Abstract async generator - yield needed for mypy to recognize as generator
        if False:
            from victor.workflows.streaming import (
                WorkflowStreamChunk,
                WorkflowEventType,
            )

            yield WorkflowStreamChunk(
                event_type=WorkflowEventType.WORKFLOW_START,
                workflow_id="",
            )
        raise NotImplementedError

    def subscribe(
        self,
        callback: Callable[["WorkflowStreamChunk"], None],
    ) -> Callable[[], None]:
        """Subscribe to workflow events using observer pattern.

        Args:
            callback: Function to call for each workflow event.

        Returns:
            Unsubscribe function to stop receiving events.

        Example:
            def handle_chunk(chunk):
                if chunk.event_type == WorkflowEventType.NODE_COMPLETE:
                    print(f"Node {chunk.node_id} completed")

            unsubscribe = executor.subscribe(handle_chunk)
            # ... workflow executes ...
            unsubscribe()  # Stop receiving events
        """
        ...

    def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a running workflow.

        Args:
            workflow_id: ID of the workflow to cancel.

        Returns:
            True if the workflow was cancelled, False if not found.

        Example:
            success = executor.cancel_workflow("wf_123")
            if success:
                print("Workflow cancelled")
        """
        ...


# =============================================================================
# NodeRunner Protocol (ISP + DIP Compliant)
# =============================================================================


@dataclass
class NodeRunnerResult:
    """Result of a node runner execution.

    This is a unified result type used by all NodeRunner implementations,
    decoupling the execution result from specific executor implementations.

    Attributes:
        node_id: ID of the executed node.
        success: Whether execution succeeded.
        output: Output data from the node execution.
        error: Error message if execution failed.
        duration_seconds: Execution time in seconds.
        metadata: Additional metadata (tool calls, retries, etc.).
    """

    node_id: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    duration_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> ProtocolNodeStatus:
        """Get status as ProtocolNodeStatus enum."""
        return ProtocolNodeStatus.COMPLETED if self.success else ProtocolNodeStatus.FAILED


@runtime_checkable
class NodeRunner(Protocol):
    """Protocol for node execution strategies.

    This protocol enables ISP (Interface Segregation) and DIP (Dependency
    Inversion) compliance by:

    1. ISP: Each node type has a focused runner instead of one executor
       handling all node types.
    2. DIP: Executors depend on this protocol, not concrete implementations.

    The protocol uses ExecutionContext (from victor.workflows.context) as
    the unified state type, enabling all runners to work with a consistent
    state model.

    Example:
        class MyCustomRunner:
            async def execute(
                self,
                node_id: str,
                node_config: Dict[str, Any],
                context: ExecutionContext,
            ) -> Tuple[ExecutionContext, NodeRunnerResult]:
                # Execute node logic
                context["data"]["result"] = "processed"
                return context, NodeRunnerResult(
                    node_id=node_id,
                    success=True,
                    output="processed",
                )

            def supports_node_type(self, node_type: str) -> bool:
                return node_type == "custom"

    Usage in executor:
        runners: Dict[str, NodeRunner] = {
            "agent": AgentNodeRunner(...),
            "compute": ComputeNodeRunner(...),
            "transform": TransformNodeRunner(...),
            "hitl": HITLNodeRunner(...),
        }

        runner = runners.get(node.type)
        if runner:
            context, result = await runner.execute(node.id, node.config, context)
    """

    async def execute(
        self,
        node_id: str,
        node_config: Dict[str, Any],
        context: Dict[str, Any],  # ExecutionContext TypedDict
    ) -> Tuple[Dict[str, Any], "NodeRunnerResult"]:
        """Execute a node and return updated context with result.

        Args:
            node_id: Unique identifier of the node to execute.
            node_config: Node configuration (role, goal, tools, etc.).
            context: Current execution context (ExecutionContext TypedDict).

        Returns:
            Tuple of (updated_context, result) where:
            - updated_context: The context with any modifications made
            - result: NodeRunnerResult with execution outcome

        Raises:
            Should not raise - errors should be captured in NodeRunnerResult.
        """
        ...

    def supports_node_type(self, node_type: str) -> bool:
        """Check if this runner supports the given node type.

        Args:
            node_type: Type of node (e.g., "agent", "compute", "transform").

        Returns:
            True if this runner can execute the node type.
        """
        ...


# =============================================================================
# Workflow Compiler Protocols (DIP Compliance)
# =============================================================================


@runtime_checkable
class IWorkflowCompiler(Protocol):
    """Protocol for workflow compilers.

    A workflow compiler transforms workflow definitions into executable graphs.
    This protocol enables Dependency Inversion by allowing the compiler to
    depend on loader and validator abstractions rather than concrete implementations.

    Example:
        class MyCompiler:
            def __init__(
                self,
                loader: IWorkflowLoader,
                validator: IWorkflowValidator
            ):
                self._loader = loader
                self._validator = validator

            def compile(self, workflow_def: Dict[str, Any]) -> CompiledGraph:
                # Load and validate using injected dependencies
                return CompiledGraph(...)
    """

    def compile(self, workflow_def: Dict[str, Any]) -> Any:
        """Compile a workflow definition into an executable graph.

        Args:
            workflow_def: Dictionary containing workflow definition
                           (nodes, edges, configuration, etc.)

        Returns:
            CompiledGraph instance ready for execution

        Raises:
            ValueError: If workflow definition is invalid
            TypeError: If workflow definition has wrong structure
        """
        ...


@runtime_checkable
class IWorkflowLoader(Protocol):
    """Protocol for workflow definition loaders.

    A workflow loader loads workflow definitions from various sources
    (files, strings, URLs, databases, etc.). This protocol enables
    Dependency Inversion by allowing compilers to work with any loader
    implementation.

    Example:
        class FileLoader:
            def load(self, source: str) -> Dict[str, Any]:
                with open(source) as f:
                    return yaml.safe_load(f)

        class DatabaseLoader:
            def __init__(self, db_connection):
                self._db = db_connection

            def load(self, source: str) -> Dict[str, Any]:
                # source is workflow ID
                return self._db.get_workflow(source)
    """

    def load(self, source: str) -> Dict[str, Any]:
        """Load a workflow definition from a source.

        Args:
            source: Source identifier (file path, URL, workflow ID, etc.)

        Returns:
            Dictionary containing workflow definition

        Raises:
            FileNotFoundError: If source doesn't exist
            ValueError: If source is malformed
            TypeError: If source has wrong type
        """
        ...


@runtime_checkable
class IWorkflowValidator(Protocol):
    """Protocol for workflow definition validators.

    A workflow validator checks workflow definitions for correctness
    before compilation. This protocol enables Dependency Inversion by
    allowing compilers to work with any validation strategy.

    Example:
        class StrictValidator:
            def validate(self, workflow_def: Dict[str, Any]) -> ValidationResult:
                # Check all required fields
                errors = []
                if "name" not in workflow_def:
                    errors.append("Missing required field: name")
                if "nodes" not in workflow_def:
                    errors.append("Missing required field: nodes")
                return ValidationResult(
                    is_valid=len(errors) == 0,
                    errors=errors
                )

        @dataclass
        class ValidationResult:
            is_valid: bool
            errors: List[str]
            warnings: List[str] = field(default_factory=list)
    """

    def validate(self, workflow_def: Dict[str, Any]) -> Any:
        """Validate a workflow definition.

        Args:
            workflow_def: Dictionary containing workflow definition to validate

        Returns:
            ValidationResult or similar with:
            - is_valid: bool indicating if definition is valid
            - errors: List of error messages (empty if valid)
            - warnings: Optional list of warnings

        Raises:
            Should not raise - capture all validation issues in result
        """
        ...

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

"""Human-in-the-Loop (HITL) workflow nodes.

Provides LangGraph-style interrupt points for human approval, review,
and intervention during workflow execution.

Node Types:
- APPROVAL: Binary approve/reject decision
- REVIEW: Review and optionally modify intermediate results
- CHOICE: Select from predefined options
- INPUT: Provide freeform text input
- CONFIRMATION: Confirm to proceed (with default behavior on timeout)

Example:
    workflow = (
        WorkflowBuilder("code_change")
        .add_agent("analyze", "researcher", "Find code to change")
        .add_hitl_approval(
            "approve_changes",
            prompt="Proceed with the following changes?",
            context_keys=["files_to_modify", "change_description"],
            timeout=300.0,
        )
        .add_agent("implement", "executor", "Make the changes")
        .build()
    )
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


from typing import Any, Callable, Dict, List, Optional, Protocol

from victor.workflows.definition import WorkflowNode, WorkflowNodeType


def _utc_now() -> datetime:
    """Return current UTC time (timezone-aware)."""
    return datetime.now(timezone.utc)


logger = logging.getLogger(__name__)


class HITLNodeType(str, Enum):
    """Types of human-in-the-loop interactions."""

    APPROVAL = "approval"  # Binary approve/reject
    REVIEW = "review"  # Review and modify
    CHOICE = "choice"  # Select from options
    INPUT = "input"  # Freeform text input
    CONFIRMATION = "confirmation"  # Simple confirmation


class HITLFallback(str, Enum):
    """Fallback behavior when HITL times out."""

    ABORT = "abort"  # Stop workflow execution
    CONTINUE = "continue"  # Continue with default value
    SKIP = "skip"  # Skip this node entirely
    RETRY = "retry"  # Retry the request


class HITLMode(str, Enum):
    """HITL interaction mode - determines transport mechanism."""

    # Local/Interactive modes (require terminal/TTY)
    CLI = "cli"  # Interactive prompt-toolkit CLI prompt

    # Remote/Async modes (serverless-compatible)
    API = "api"  # REST API callback (pause workflow, resume via API)
    WEBHOOK = "webhook"  # POST to webhook, poll for response
    POLLING = "polling"  # Poll endpoint for response

    # Enterprise messaging integrations
    EMAIL = "email"  # Send approval email with links
    SMS = "sms"  # SMS/text message via Twilio/SNS
    SLACK = "slack"  # Slack interactive message
    TEAMS = "teams"  # Microsoft Teams approval card
    DISCORD = "discord"  # Discord bot interaction

    # DevOps/SCM integrations (for PR/MR approvals)
    GITHUB_PR = "github_pr"  # GitHub PR review/approval
    GITHUB_CHECK = "github_check"  # GitHub Check Run (CI gates)
    GITHUB_DEPLOYMENT = "github_deployment"  # GitHub Deployment protection
    GITLAB_MR = "gitlab_mr"  # GitLab Merge Request approval
    GITLAB_PIPELINE = "gitlab_pipeline"  # GitLab pipeline gate
    BITBUCKET_PR = "bitbucket_pr"  # Bitbucket PR approval
    AZURE_DEVOPS_PR = "azure_devops_pr"  # Azure DevOps PR approval

    # Issue/Project management integrations
    JIRA = "jira"  # Jira issue transition approval
    LINEAR = "linear"  # Linear issue approval
    ASANA = "asana"  # Asana task approval
    SERVICENOW = "servicenow"  # ServiceNow change request

    # Incident/On-call integrations
    PAGERDUTY = "pagerduty"  # PagerDuty incident approval
    OPSGENIE = "opsgenie"  # OpsGenie alert approval
    VICTOROPS = "victorops"  # VictorOps/Splunk On-Call

    # Infrastructure/GitOps integrations
    ARGOCD = "argocd"  # ArgoCD sync wave gate
    FLUX = "flux"  # FluxCD gate
    TERRAFORM_CLOUD = "terraform_cloud"  # Terraform Cloud run approval
    SPACELIFT = "spacelift"  # Spacelift stack approval

    # Auto modes (for testing/CI)
    AUTO_APPROVE = "auto_approve"  # Auto-approve without human
    AUTO_REJECT = "auto_reject"  # Auto-reject without human

    # Inline callback (programmatic)
    INLINE = "inline"  # Call a Python function directly

    # Custom hook (user-defined transport)
    CUSTOM_HOOK = "custom_hook"  # User-provided hook function


class HITLCategory(str, Enum):
    """Category of HITL mode for grouping and documentation."""

    LOCAL = "local"  # Requires terminal/TTY
    ASYNC_API = "async_api"  # REST/webhook based
    MESSAGING = "messaging"  # Chat/email based
    SCM = "scm"  # Source control management
    PROJECT = "project"  # Issue/project tracking
    INCIDENT = "incident"  # Incident management
    INFRASTRUCTURE = "infrastructure"  # GitOps/IaC
    AUTO = "auto"  # Automated (no human)
    CUSTOM = "custom"  # User-defined


# Map modes to categories
HITL_MODE_CATEGORIES: Dict[HITLMode, HITLCategory] = {
    HITLMode.CLI: HITLCategory.LOCAL,
    HITLMode.API: HITLCategory.ASYNC_API,
    HITLMode.WEBHOOK: HITLCategory.ASYNC_API,
    HITLMode.POLLING: HITLCategory.ASYNC_API,
    HITLMode.EMAIL: HITLCategory.MESSAGING,
    HITLMode.SMS: HITLCategory.MESSAGING,
    HITLMode.SLACK: HITLCategory.MESSAGING,
    HITLMode.TEAMS: HITLCategory.MESSAGING,
    HITLMode.DISCORD: HITLCategory.MESSAGING,
    HITLMode.GITHUB_PR: HITLCategory.SCM,
    HITLMode.GITHUB_CHECK: HITLCategory.SCM,
    HITLMode.GITHUB_DEPLOYMENT: HITLCategory.SCM,
    HITLMode.GITLAB_MR: HITLCategory.SCM,
    HITLMode.GITLAB_PIPELINE: HITLCategory.SCM,
    HITLMode.BITBUCKET_PR: HITLCategory.SCM,
    HITLMode.AZURE_DEVOPS_PR: HITLCategory.SCM,
    HITLMode.JIRA: HITLCategory.PROJECT,
    HITLMode.LINEAR: HITLCategory.PROJECT,
    HITLMode.ASANA: HITLCategory.PROJECT,
    HITLMode.SERVICENOW: HITLCategory.PROJECT,
    HITLMode.PAGERDUTY: HITLCategory.INCIDENT,
    HITLMode.OPSGENIE: HITLCategory.INCIDENT,
    HITLMode.VICTOROPS: HITLCategory.INCIDENT,
    HITLMode.ARGOCD: HITLCategory.INFRASTRUCTURE,
    HITLMode.FLUX: HITLCategory.INFRASTRUCTURE,
    HITLMode.TERRAFORM_CLOUD: HITLCategory.INFRASTRUCTURE,
    HITLMode.SPACELIFT: HITLCategory.INFRASTRUCTURE,
    HITLMode.AUTO_APPROVE: HITLCategory.AUTO,
    HITLMode.AUTO_REJECT: HITLCategory.AUTO,
    HITLMode.INLINE: HITLCategory.CUSTOM,
    HITLMode.CUSTOM_HOOK: HITLCategory.CUSTOM,
}


# Mapping of deployment targets to supported HITL modes
# This enables validation that workflows with HITL nodes are compatible
# with their deployment target
DEPLOYMENT_HITL_COMPATIBILITY = {
    # Local targets - all modes work
    "inline": [
        HITLMode.CLI,
        HITLMode.INLINE,
        HITLMode.AUTO_APPROVE,
        HITLMode.AUTO_REJECT,
    ],
    "subprocess": [
        HITLMode.CLI,
        HITLMode.INLINE,
        HITLMode.AUTO_APPROVE,
        HITLMode.AUTO_REJECT,
    ],
    "local": [
        HITLMode.CLI,
        HITLMode.INLINE,
        HITLMode.AUTO_APPROVE,
        HITLMode.AUTO_REJECT,
    ],
    # Docker - no TTY by default, need API/webhook
    "docker": [
        HITLMode.API,
        HITLMode.WEBHOOK,
        HITLMode.EMAIL,
        HITLMode.SLACK,
        HITLMode.AUTO_APPROVE,
        HITLMode.AUTO_REJECT,
    ],
    "docker_compose": [
        HITLMode.API,
        HITLMode.WEBHOOK,
        HITLMode.EMAIL,
        HITLMode.SLACK,
        HITLMode.AUTO_APPROVE,
        HITLMode.AUTO_REJECT,
    ],
    # Kubernetes - async/API-based
    "kubernetes": [
        HITLMode.API,
        HITLMode.WEBHOOK,
        HITLMode.EMAIL,
        HITLMode.SLACK,
        HITLMode.TEAMS,
        HITLMode.AUTO_APPROVE,
        HITLMode.AUTO_REJECT,
    ],
    "aks": [
        HITLMode.API,
        HITLMode.WEBHOOK,
        HITLMode.EMAIL,
        HITLMode.SLACK,
        HITLMode.TEAMS,
        HITLMode.AUTO_APPROVE,
        HITLMode.AUTO_REJECT,
    ],
    "eks": [
        HITLMode.API,
        HITLMode.WEBHOOK,
        HITLMode.EMAIL,
        HITLMode.SLACK,
        HITLMode.AUTO_APPROVE,
        HITLMode.AUTO_REJECT,
    ],
    "gke": [
        HITLMode.API,
        HITLMode.WEBHOOK,
        HITLMode.EMAIL,
        HITLMode.SLACK,
        HITLMode.AUTO_APPROVE,
        HITLMode.AUTO_REJECT,
    ],
    # Serverless - must be async (can't hold connection)
    "aws_lambda": [
        HITLMode.API,
        HITLMode.WEBHOOK,
        HITLMode.EMAIL,
        HITLMode.SLACK,
        HITLMode.AUTO_APPROVE,
        HITLMode.AUTO_REJECT,
    ],
    "cloud_run": [
        HITLMode.API,
        HITLMode.WEBHOOK,
        HITLMode.EMAIL,
        HITLMode.SLACK,
        HITLMode.AUTO_APPROVE,
        HITLMode.AUTO_REJECT,
    ],
    "azure_functions": [
        HITLMode.API,
        HITLMode.WEBHOOK,
        HITLMode.EMAIL,
        HITLMode.TEAMS,
        HITLMode.AUTO_APPROVE,
        HITLMode.AUTO_REJECT,
    ],
    # Container services
    "ecs": [
        HITLMode.API,
        HITLMode.WEBHOOK,
        HITLMode.EMAIL,
        HITLMode.SLACK,
        HITLMode.AUTO_APPROVE,
        HITLMode.AUTO_REJECT,
    ],
    "fargate": [
        HITLMode.API,
        HITLMode.WEBHOOK,
        HITLMode.EMAIL,
        HITLMode.SLACK,
        HITLMode.AUTO_APPROVE,
        HITLMode.AUTO_REJECT,
    ],
    "azure_container": [
        HITLMode.API,
        HITLMode.WEBHOOK,
        HITLMode.EMAIL,
        HITLMode.TEAMS,
        HITLMode.AUTO_APPROVE,
        HITLMode.AUTO_REJECT,
    ],
    # Batch/Job services
    "aws_batch": [
        HITLMode.WEBHOOK,
        HITLMode.EMAIL,
        HITLMode.SLACK,
        HITLMode.AUTO_APPROVE,
        HITLMode.AUTO_REJECT,
    ],
    "azure_batch": [
        HITLMode.WEBHOOK,
        HITLMode.EMAIL,
        HITLMode.TEAMS,
        HITLMode.AUTO_APPROVE,
        HITLMode.AUTO_REJECT,
    ],
    "cloud_batch": [
        HITLMode.WEBHOOK,
        HITLMode.EMAIL,
        HITLMode.SLACK,
        HITLMode.AUTO_APPROVE,
        HITLMode.AUTO_REJECT,
    ],
    # Workflow orchestrators - typically have their own approval mechanisms
    "airflow": [
        HITLMode.WEBHOOK,
        HITLMode.EMAIL,
        HITLMode.SLACK,
        HITLMode.AUTO_APPROVE,
        HITLMode.AUTO_REJECT,
    ],
    "temporal": [
        HITLMode.API,
        HITLMode.WEBHOOK,
        HITLMode.EMAIL,
        HITLMode.SLACK,
        HITLMode.AUTO_APPROVE,
        HITLMode.AUTO_REJECT,
    ],
    "step_functions": [
        HITLMode.WEBHOOK,
        HITLMode.EMAIL,
        HITLMode.AUTO_APPROVE,
        HITLMode.AUTO_REJECT,
    ],
    # Data platforms
    "databricks": [
        HITLMode.WEBHOOK,
        HITLMode.EMAIL,
        HITLMode.SLACK,
        HITLMode.AUTO_APPROVE,
        HITLMode.AUTO_REJECT,
    ],
    "spark": [HITLMode.AUTO_APPROVE, HITLMode.AUTO_REJECT],  # Batch-only typically
    "ray": [
        HITLMode.API,
        HITLMode.WEBHOOK,
        HITLMode.AUTO_APPROVE,
        HITLMode.AUTO_REJECT,
    ],
    "dask": [
        HITLMode.API,
        HITLMode.WEBHOOK,
        HITLMode.AUTO_APPROVE,
        HITLMode.AUTO_REJECT,
    ],
    # Task queues
    "celery": [
        HITLMode.API,
        HITLMode.WEBHOOK,
        HITLMode.EMAIL,
        HITLMode.SLACK,
        HITLMode.AUTO_APPROVE,
        HITLMode.AUTO_REJECT,
    ],
    # Serverless platforms
    "modal": [
        HITLMode.API,
        HITLMode.WEBHOOK,
        HITLMode.AUTO_APPROVE,
        HITLMode.AUTO_REJECT,
    ],
}


def get_supported_hitl_modes(deployment_target: str) -> list:
    """Get supported HITL modes for a deployment target.

    Args:
        deployment_target: The deployment target name

    Returns:
        List of supported HITLMode values
    """
    return DEPLOYMENT_HITL_COMPATIBILITY.get(
        deployment_target,
        [HITLMode.AUTO_APPROVE, HITLMode.AUTO_REJECT],  # Default fallback
    )


def validate_hitl_deployment_compatibility(
    hitl_mode: HITLMode,
    deployment_target: str,
) -> tuple[bool, str]:
    """Validate that HITL mode is compatible with deployment target.

    Args:
        hitl_mode: The HITL interaction mode
        deployment_target: The deployment target

    Returns:
        Tuple of (is_compatible, error_message)
    """
    supported = get_supported_hitl_modes(deployment_target)

    if hitl_mode in supported:
        return True, ""

    return False, (
        f"HITL mode '{hitl_mode.value}' not supported for target '{deployment_target}'. "
        f"Supported: {[m.value for m in supported]}"
    )


class HITLStatus(str, Enum):
    """Status of a HITL request."""

    PENDING = "pending"  # Waiting for human response
    APPROVED = "approved"  # Approved to continue
    REJECTED = "rejected"  # Rejected/aborted
    MODIFIED = "modified"  # Modified by human
    TIMEOUT = "timeout"  # Timed out
    SKIPPED = "skipped"  # Skipped by fallback


@dataclass
class HITLRequest:
    """A request for human intervention.

    Attributes:
        request_id: Unique identifier for this request
        node_id: ID of the HITL node
        hitl_type: Type of interaction requested
        prompt: Message to display to human
        context: Contextual data for display
        choices: Available choices (for CHOICE type)
        default_value: Default value if timeout
        timeout: Timeout in seconds
        fallback: Behavior on timeout
        created_at: When request was created
    """

    request_id: str
    node_id: str
    hitl_type: HITLNodeType
    prompt: str
    context: Dict[str, Any] = field(default_factory=dict)
    choices: Optional[List[str]] = None
    default_value: Optional[Any] = None
    timeout: float = 300.0
    fallback: HITLFallback = HITLFallback.ABORT
    created_at: datetime = field(default_factory=_utc_now)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for transmission/display."""
        return {
            "request_id": self.request_id,
            "node_id": self.node_id,
            "hitl_type": self.hitl_type.value,
            "prompt": self.prompt,
            "context": self.context,
            "choices": self.choices,
            "default_value": self.default_value,
            "timeout": self.timeout,
            "fallback": self.fallback.value,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class HITLResponse:
    """Response from human for HITL request.

    Attributes:
        request_id: ID of the request being responded to
        status: Result status
        approved: Whether action was approved (for APPROVAL/CONFIRMATION)
        value: Selected/input value (for CHOICE/INPUT/REVIEW)
        modifications: Modified data (for REVIEW)
        reason: Optional explanation for decision
        responder: Identity of the human who responded (e.g. "Ada <ada@x.com>"),
            populated by the delegated/SSO approval flow; None for anonymous or
            programmatic responses
        responded_at: When response was provided
    """

    request_id: str
    status: HITLStatus
    approved: bool = True
    value: Optional[Any] = None
    modifications: Optional[Dict[str, Any]] = None
    reason: Optional[str] = None
    responder: Optional[str] = None
    responded_at: datetime = field(default_factory=_utc_now)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize response."""
        return {
            "request_id": self.request_id,
            "status": self.status.value,
            "approved": self.approved,
            "value": self.value,
            "modifications": self.modifications,
            "reason": self.reason,
            "responder": self.responder,
            "responded_at": self.responded_at.isoformat(),
        }


class HITLHandler(Protocol):
    """Protocol for handling HITL interactions.

    Implementations provide UI-specific handling for HITL requests
    (CLI, web API, etc.).
    """

    async def request_human_input(self, request: HITLRequest) -> HITLResponse:
        """Request input from human.

        Args:
            request: The HITL request

        Returns:
            HITLResponse with human's decision
        """
        ...


@dataclass
class HITLNode(WorkflowNode):
    """Workflow node that pauses for human intervention.

    Attributes:
        hitl_type: Type of interaction
        prompt: Message to display
        context_keys: Keys from workflow context to include in display
        choices: Available choices (for CHOICE type)
        default_value: Default value if timeout
        timeout: Timeout in seconds (default: 5 minutes)
        fallback: Behavior on timeout
        validator: Optional function to validate input
    """

    hitl_type: HITLNodeType = HITLNodeType.APPROVAL
    prompt: str = "Proceed?"
    context_keys: List[str] = field(default_factory=list)
    choices: Optional[List[str]] = None
    default_value: Optional[Any] = None
    timeout: float = 300.0
    fallback: HITLFallback = HITLFallback.ABORT
    validator: Optional[Callable[[Any], bool]] = None

    @property
    def node_type(self) -> WorkflowNodeType:
        """HITL nodes have their own node type."""
        return WorkflowNodeType.HITL

    def to_dict(self) -> Dict[str, Any]:
        """Serialize node."""
        d = super().to_dict()
        d.update(
            {
                "hitl_type": self.hitl_type.value,
                "prompt": self.prompt,
                "context_keys": self.context_keys,
                "choices": self.choices,
                "default_value": self.default_value,
                "timeout": self.timeout,
                "fallback": self.fallback.value,
                # Note: validator function is not serialized
            }
        )
        # Override type to indicate HITL
        d["type"] = "hitl"
        d["_hitl_node"] = True
        return d

    def create_request(
        self,
        workflow_context: Dict[str, Any],
    ) -> HITLRequest:
        """Create a HITL request from workflow context.

        Args:
            workflow_context: Current workflow execution context

        Returns:
            HITLRequest ready for handler
        """
        # Extract relevant context
        context = {}
        for key in self.context_keys:
            if key in workflow_context:
                context[key] = workflow_context[key]

        return HITLRequest(
            request_id=f"hitl_{uuid.uuid4().hex[:12]}",
            node_id=self.id,
            hitl_type=self.hitl_type,
            prompt=self.prompt,
            context=context,
            choices=self.choices,
            default_value=self.default_value,
            timeout=self.timeout,
            fallback=self.fallback,
        )

    def validate_response(self, response: HITLResponse) -> bool:
        """Validate a HITL response.

        Args:
            response: The response to validate

        Returns:
            True if valid, False otherwise
        """
        if self.validator and response.value is not None:
            try:
                return self.validator(response.value)
            except Exception as e:
                logger.warning(f"HITL validation failed: {e}")
                return False

        return True


class HITLExecutor:
    """Executes HITL nodes during workflow execution.

    Handles the async communication between workflow executor and
    human interface (CLI, API, or external integrations via transports).

    Supports two modes:
    1. Handler mode: Direct handler (CLI, inline) for local/interactive use
    2. Transport mode: Uses transport adapters for remote integrations
       (Slack, Email, GitHub PR, etc.)
    """

    def __init__(
        self,
        handler: Optional[HITLHandler] = None,
        mode: Optional[HITLMode] = None,
        transport_config: Optional[Any] = None,
    ):
        """Initialize executor with handler or transport.

        Args:
            handler: Handler for human interactions (for local modes)
            mode: HITL mode (determines transport if no handler)
            transport_config: Configuration for transport (if using remote mode)
        """
        self.handler = handler
        self.mode = mode or HITLMode.CLI
        self.transport_config = transport_config
        self._transport = None

        self._pending_requests: Dict[str, asyncio.Event] = {}
        self._responses: Dict[str, HITLResponse] = {}
        self._external_refs: Dict[str, str] = {}  # request_id -> external_ref

    def _get_transport(self):
        """Get or create the transport adapter."""
        if self._transport is None and self.mode not in [
            HITLMode.CLI,
            HITLMode.INLINE,
            HITLMode.AUTO_APPROVE,
            HITLMode.AUTO_REJECT,
        ]:
            from victor.workflows.hitl_transports import get_transport

            self._transport = get_transport(self.mode, self.transport_config)
        return self._transport

    def _should_use_transport(self) -> bool:
        """Check if we should use transport vs handler."""
        return (
            self.mode
            not in [
                HITLMode.CLI,
                HITLMode.INLINE,
            ]
            and self._get_transport() is not None
        )

    async def execute_hitl_node(
        self,
        node: HITLNode,
        context: Dict[str, Any],
        workflow_id: Optional[str] = None,
    ) -> HITLResponse:
        """Execute a HITL node, waiting for human response.

        Automatically chooses between:
        - Handler mode: Direct CLI interaction
        - Transport mode: Send via external system (Slack, Email, GitHub, etc.)
        - Auto mode: Automatic approve/reject for testing

        Args:
            node: The HITL node to execute
            context: Current workflow context
            workflow_id: Optional workflow identifier for transport context

        Returns:
            HITLResponse with result

        Raises:
            asyncio.TimeoutError: If timeout and fallback is ABORT
        """
        request = node.create_request(context)
        workflow_id = workflow_id or context.get("_workflow_id", "workflow")

        logger.info(f"HITL request {request.request_id}: {node.hitl_type.value} - {node.prompt}")

        # Handle auto modes first
        if self.mode == HITLMode.AUTO_APPROVE:
            logger.info(f"Auto-approving {request.request_id}")
            return HITLResponse(
                request_id=request.request_id,
                status=HITLStatus.APPROVED,
                approved=True,
                value=node.default_value,
                reason="Auto-approved (testing mode)",
            )
        elif self.mode == HITLMode.AUTO_REJECT:
            logger.info(f"Auto-rejecting {request.request_id}")
            return HITLResponse(
                request_id=request.request_id,
                status=HITLStatus.REJECTED,
                approved=False,
                reason="Auto-rejected (testing mode)",
            )

        try:
            if self._should_use_transport():
                # Transport mode: send to external system and poll
                response = await self._execute_via_transport(request, node, workflow_id)
            elif self.handler:
                # Handler mode: direct interaction
                response = await asyncio.wait_for(
                    self.handler.request_human_input(request),
                    timeout=node.timeout,
                )
            else:
                raise ValueError("No handler or transport configured for HITL")

            # Validate response
            if not node.validate_response(response):
                logger.warning(f"HITL response validation failed for {request.request_id}")
                response = HITLResponse(
                    request_id=request.request_id,
                    status=HITLStatus.REJECTED,
                    approved=False,
                    reason="Response validation failed",
                )

            return response

        except asyncio.TimeoutError:
            logger.warning(f"HITL request {request.request_id} timed out after {node.timeout}s")
            return self._handle_timeout(request, node)

    async def _execute_via_transport(
        self,
        request: HITLRequest,
        node: HITLNode,
        workflow_id: str,
    ) -> HITLResponse:
        """Execute HITL via transport adapter.

        Sends notification to external system and polls for response.

        Args:
            request: The HITL request
            node: The HITL node
            workflow_id: Workflow identifier

        Returns:
            HITLResponse from external system
        """
        transport = self._get_transport()

        # Send to external system
        external_ref = await transport.send(request, workflow_id)
        self._external_refs[request.request_id] = external_ref

        logger.info(f"HITL request {request.request_id} sent via {self.mode.value}: {external_ref}")

        # Wait for response with polling
        response = await transport.wait_for_response(
            request.request_id,
            external_ref,
            timeout=node.timeout,
        )

        if response:
            return response

        # No response within timeout
        raise asyncio.TimeoutError(f"No response within {node.timeout}s")

    def _handle_timeout(
        self,
        request: HITLRequest,
        node: HITLNode,
    ) -> HITLResponse:
        """Handle timeout based on fallback strategy.

        Args:
            request: The timed-out request
            node: The HITL node

        Returns:
            HITLResponse based on fallback behavior
        """
        if node.fallback == HITLFallback.CONTINUE:
            return HITLResponse(
                request_id=request.request_id,
                status=HITLStatus.TIMEOUT,
                approved=True,
                value=node.default_value,
                reason="Timed out, continuing with default",
            )

        elif node.fallback == HITLFallback.SKIP:
            return HITLResponse(
                request_id=request.request_id,
                status=HITLStatus.SKIPPED,
                approved=True,
                reason="Timed out, skipping",
            )

        else:  # ABORT or RETRY
            return HITLResponse(
                request_id=request.request_id,
                status=HITLStatus.TIMEOUT,
                approved=False,
                reason=f"Timed out after {node.timeout}s",
            )


class DefaultHITLHandler:
    """Default CLI handler for HITL requests.

    Uses simple input() for approval/choice prompts.
    Suitable for basic CLI usage.
    """

    async def request_human_input(self, request: HITLRequest) -> HITLResponse:
        """Handle HITL request via CLI.

        Args:
            request: The HITL request

        Returns:
            HITLResponse from user input
        """
        print(f"\n{'='*60}")
        print(f"Human Input Required: {request.hitl_type.value.upper()}")
        print(f"{'='*60}")
        print(f"\n{request.prompt}\n")

        # Show context if available
        if request.context:
            print("Context:")
            for key, value in request.context.items():
                if isinstance(value, str) and len(value) > 200:
                    value = value[:200] + "..."
                print(f"  {key}: {value}")
            print()

        if request.hitl_type == HITLNodeType.APPROVAL:
            return await self._handle_approval(request)
        elif request.hitl_type == HITLNodeType.CHOICE:
            return await self._handle_choice(request)
        elif request.hitl_type == HITLNodeType.INPUT:
            return await self._handle_input(request)
        elif request.hitl_type == HITLNodeType.CONFIRMATION:
            return await self._handle_confirmation(request)
        elif request.hitl_type == HITLNodeType.REVIEW:
            return await self._handle_review(request)
        else:
            return HITLResponse(
                request_id=request.request_id,
                status=HITLStatus.REJECTED,
                approved=False,
                reason=f"Unknown HITL type: {request.hitl_type}",
            )

    async def _handle_approval(self, request: HITLRequest) -> HITLResponse:
        """Handle approval request."""
        while True:
            response = await asyncio.to_thread(input, "Approve? [y/n]: ")
            response = response.strip().lower()

            if response in ("y", "yes"):
                return HITLResponse(
                    request_id=request.request_id,
                    status=HITLStatus.APPROVED,
                    approved=True,
                )
            elif response in ("n", "no"):
                reason = await asyncio.to_thread(input, "Reason (optional): ")
                return HITLResponse(
                    request_id=request.request_id,
                    status=HITLStatus.REJECTED,
                    approved=False,
                    reason=reason.strip() or None,
                )
            else:
                print("Please enter 'y' or 'n'")

    async def _handle_choice(self, request: HITLRequest) -> HITLResponse:
        """Handle choice selection."""
        if not request.choices:
            return HITLResponse(
                request_id=request.request_id,
                status=HITLStatus.REJECTED,
                approved=False,
                reason="No choices provided",
            )

        print("Options:")
        for i, choice in enumerate(request.choices, 1):
            print(f"  [{i}] {choice}")

        while True:
            response = await asyncio.to_thread(input, "Select option number: ")
            try:
                idx = int(response.strip()) - 1
                if 0 <= idx < len(request.choices):
                    return HITLResponse(
                        request_id=request.request_id,
                        status=HITLStatus.APPROVED,
                        approved=True,
                        value=request.choices[idx],
                    )
                else:
                    print(f"Please enter a number between 1 and {len(request.choices)}")
            except ValueError:
                print("Please enter a valid number")

    async def _handle_input(self, request: HITLRequest) -> HITLResponse:
        """Handle freeform input."""
        response = await asyncio.to_thread(input, "Your input: ")
        return HITLResponse(
            request_id=request.request_id,
            status=HITLStatus.APPROVED,
            approved=True,
            value=response.strip(),
        )

    async def _handle_confirmation(self, request: HITLRequest) -> HITLResponse:
        """Handle simple confirmation."""
        response = await asyncio.to_thread(input, "Press Enter to continue or 'q' to abort: ")
        if response.strip().lower() == "q":
            return HITLResponse(
                request_id=request.request_id,
                status=HITLStatus.REJECTED,
                approved=False,
                reason="User aborted",
            )
        return HITLResponse(
            request_id=request.request_id,
            status=HITLStatus.APPROVED,
            approved=True,
        )

    async def _handle_review(self, request: HITLRequest) -> HITLResponse:
        """Handle review with modification."""
        print("Review the context above.")
        response = await asyncio.to_thread(input, "Approve as-is [a], modify [m], or reject [r]: ")
        response = response.strip().lower()

        if response == "a":
            return HITLResponse(
                request_id=request.request_id,
                status=HITLStatus.APPROVED,
                approved=True,
            )
        elif response == "r":
            reason = await asyncio.to_thread(input, "Reason: ")
            return HITLResponse(
                request_id=request.request_id,
                status=HITLStatus.REJECTED,
                approved=False,
                reason=reason.strip(),
            )
        else:
            # Simple modification - just get text input
            print("Enter modifications (one per line, empty line to finish):")
            modifications = {}
            while True:
                line = await asyncio.to_thread(input, "")
                if not line:
                    break
                if "=" in line:
                    key, value = line.split("=", 1)
                    modifications[key.strip()] = value.strip()

            return HITLResponse(
                request_id=request.request_id,
                status=HITLStatus.MODIFIED,
                approved=True,
                modifications=modifications,
            )


# Type alias for cleaner imports
HITLCallback = Callable[[HITLRequest], HITLResponse]


__all__ = [
    # Node types and enums
    "HITLNodeType",
    "HITLFallback",
    "HITLMode",
    "HITLStatus",
    # Request/Response
    "HITLRequest",
    "HITLResponse",
    # Protocols
    "HITLHandler",
    # Node and executor
    "HITLNode",
    "HITLExecutor",
    # Handlers
    "DefaultHITLHandler",
    "HITLCallback",
    # Deployment compatibility
    "DEPLOYMENT_HITL_COMPATIBILITY",
    "get_supported_hitl_modes",
    "validate_hitl_deployment_compatibility",
]

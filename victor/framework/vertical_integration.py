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

"""Reusable pipeline for vertical extension application.

This module provides a unified pipeline for applying vertical extensions
to orchestrators, ensuring both CLI (FrameworkShim) and SDK (Agent.create())
paths apply identical vertical configurations.

Design Philosophy:
- Single implementation for all vertical integration
- Protocol-based access (no private attribute writes)
- Type-safe configuration through MutableVerticalContextProtocol
- SOLID-compliant extension points
- Step handlers for Single Responsibility (Phase 3.1)

Architecture (Refactored with Step Handlers):
    VerticalIntegrationPipeline (Facade)
    │
    └── StepHandlerRegistry
        ├── ToolStepHandler (order=10) - Apply tools filter
        ├── PromptStepHandler (order=20) - Apply system prompt
        ├── ConfigStepHandler (order=40) - Apply stages
        ├── ExtensionsStepHandler (order=45)
        │   ├── MiddlewareStepHandler - Apply middleware
        │   ├── SafetyStepHandler - Apply safety patterns
        │   ├── PromptStepHandler - Apply prompt contributors
        │   └── ConfigStepHandler - Apply mode config & tool deps
        ├── FrameworkStepHandler (order=60)
        │   ├── Workflows - Register workflow definitions
        │   ├── RL Config - Configure RL learners
        │   └── Team Specs - Register team specifications
        └── ContextStepHandler (order=100) - Attach context

Architecture Notes: Pipeline vs Adapter
---------------------------------------

This module contains two distinct components with different responsibilities:

1. **VerticalIntegrationPipeline** (this module):
   - Purpose: Orchestrate the COMPLETE vertical application process
   - Phase: Setup/initialization (one-time during agent creation)
   - Scope: All vertical aspects (tools, prompts, config, extensions, workflows)
   - Usage: Called by FrameworkShim, Agent.create(), SDK initialization
   - Pattern: Facade + Template Method with step handlers
   - Example:
       pipeline = VerticalIntegrationPipeline()
       result = pipeline.apply(orchestrator, vertical)
2. **VerticalIntegrationAdapter** (victor/agent/vertical_integration_adapter.py):
   - Purpose: Provide runtime implementation for middleware/safety application
   - Phase: Runtime (during request processing)
   - Scope: Only middleware chain and safety pattern application
   - Usage: Called by orchestrator's apply_vertical_middleware/safety_patterns methods
   - Pattern: Adapter (wraps orchestrator, provides single implementation)
   - Example:
        adapter = VerticalIntegrationAdapter(orchestrator)
        adapter.apply_middleware(middleware_list)
        adapter.apply_safety_patterns(patterns)

Key Differences:
┌────────────────────────┬──────────────────────────────┬─────────────────────────────────┐
│ Aspect                 │ VerticalIntegrationPipeline   │ VerticalIntegrationAdapter       │
├────────────────────────┼──────────────────────────────┼─────────────────────────────────┤
│ When to use            │ Agent creation/setup         │ Runtime request processing      │
│ What it handles        │ Entire vertical config       │ Middleware & safety only        │
│ Called by              │ FrameworkShim, SDK           │ AgentOrchestrator methods       │
│ Step handlers          │ Yes (orchestration)          │ No (direct implementation)      │
│ Returns                │ IntegrationResult            │ None (void)                     │
│ Caching                │ Yes (config cache)           │ No (stateful)                   │
│ Location               │ victor.framework.vertical_integration │ victor.agent.vertical_integration_adapter │
└────────────────────────┴──────────────────────────────┴─────────────────────────────────┘

When to use which:
- Use Pipeline when: Creating an agent, applying a vertical, SDK initialization
- Use Adapter when: Implementing orchestrator methods, applying middleware/safety at runtime

The Pipeline delegates to the Adapter:
- The Pipeline's MiddlewareStepHandler calls orchestrator.apply_vertical_middleware()
- The orchestrator method delegates to VerticalIntegrationAdapter.apply_middleware()
- This separation allows: (a) Pipeline to orchestrate the full setup, (b) Adapter to
  provide runtime implementation without requiring Pipeline imports

Usage:
    from victor.framework.vertical_integration import (
        VerticalIntegrationPipeline,
        IntegrationResult,
    )

    # Create pipeline
    pipeline = VerticalIntegrationPipeline()

    # Apply vertical
    result = pipeline.apply(orchestrator, vertical)

    # Check result
    if result.success:
        print(f"Applied {result.vertical_name}")
        print(f"Tools: {result.tools_applied}")
    else:
        print(f"Errors: {result.errors}")

    # Custom step handlers (Phase 3.1)
    from victor.framework.step_handlers import StepHandlerRegistry

    registry = StepHandlerRegistry.default()
    registry.add_handler(MyCustomStepHandler())
    pipeline = VerticalIntegrationPipeline(step_registry=registry)
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import logging
from victor.core.json_utils import json_dumps, json_loads
import threading
import time
import warnings
import weakref
from dataclasses import dataclass, field
from enum import Enum

from victor.framework.integration_registry import IntegrationPlanRegistry
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Set,
    Tuple,
    Type,
    Union,
    runtime_checkable,
)

from victor.core.context import bind_active_vertical
from victor.core.events.emit_helper import emit_event_sync
from victor.core.verticals.base import VerticalRegistry
from victor.core.verticals.adapters import ensure_runtime_vertical
from victor.core.verticals.manifest_contract import (
    get_or_create_vertical_manifest,
    get_vertical_runtime_metadata,
)
from victor.core.verticals.namespace_executor import get_namespace_executor_pool

if TYPE_CHECKING:
    from victor.core.shared_types import MutableVerticalContextProtocol
    from victor.framework.step_handlers import StepHandlerRegistry
    from victor.core.verticals.base import VerticalBase
    from victor.framework.vertical_cache_policy import VerticalIntegrationCachePolicy

from victor.framework.capability_runtime import (
    check_capability as _check_capability,
    invoke_capability as _invoke_capability,
)
from victor.framework.framework_integration_registry_service import (
    resolve_framework_integration_registry_service,
)
from victor.framework.strict_mode import ensure_protocol_fallback_allowed
from victor.framework.vertical_cache_policy import (
    InMemoryLRUVerticalIntegrationCachePolicy,
)

logger = logging.getLogger(__name__)

_INTEGRATION_PLAN_VERSION = 1
_INTEGRATION_PLAN_SKIP_REASON = "integration_plan_noop"
_LEGACY_EXTENSION_REGISTRY_GUIDANCE = (
    "Legacy extension-registry APIs in victor.framework.vertical_integration are deprecated and "
    "inactive in the default pipeline. Use StepHandlerRegistry + ExtensionsStepHandler."
)


def _warn_legacy_extension_registry_api(api_name: str) -> None:
    """Emit deprecation warning for inactive legacy extension-registry APIs."""
    message = (
        f"{api_name} is deprecated. {_LEGACY_EXTENSION_REGISTRY_GUIDANCE} "
        "See victor.framework.step_handlers.ExtensionsStepHandler.extension_registry. "
        "This will be removed in v1.0."
    )
    logger.warning(message)


# =============================================================================
# Extension Handler Registry (OCP Compliance)
# =============================================================================


@dataclass
class ExtensionHandlerInfo:
    """Information about an extension handler.

    Attributes:
        name: Name of the extension type
        attr_name: Attribute name on VerticalExtensions object
        handler: Handler function to call
        order: Execution order (lower = earlier)
    """

    name: str
    attr_name: str
    handler: Callable[
        [
            "VerticalIntegrationPipeline",
            Any,
            Any,
            MutableVerticalContextProtocol,
            "IntegrationResult",
        ],
        None,
    ]
    order: int = 100


class ExtensionHandlerRegistry:
    """Registry for extension handlers (OCP compliance).

    .. deprecated::
        This registry is kept for architectural reference but is NOT actively used.
        The active implementation lives in `victor.framework.step_handlers.ExtensionsStepHandler`
        which uses its own `ExtensionHandlerRegistry` instance-based approach.

        This legacy registry represents the original OCP-compliant design for the
        `VerticalIntegrationPipeline`. The codebase has evolved to use step handlers
        with SOLID principles, and the extension handling is now managed by
        `ExtensionsStepHandler` in `step_handlers.py`.

        Keeping this registry serves as documentation of the architectural evolution
        from OCP to SOLID principles. It may be repurposed in future iterations for
        higher-level extension types that operate outside the step handler system.

    This registry allows new extension types to be added without modifying
    the _apply_extensions method. Each extension type registers a handler
    that will be called when that extension is present.

    Example:
        registry = ExtensionHandlerRegistry.default()

        # Register a new extension type
        registry.register(
            name="chain_factory",
            attr_name="chain_factory_provider",
            handler=lambda pipeline, orch, ext, ctx, res: pipeline._apply_chain_factory(orch, ext, ctx, res),
            order=70,
        )

    Note:
        For new extension types, use `victor.framework.step_handlers.ExtensionsStepHandler`
        and its `ExtensionHandlerRegistry` instead.
    """

    def __init__(self) -> None:
        self._handlers: Dict[str, ExtensionHandlerInfo] = {}

    def register(
        self,
        name: str,
        attr_name: str,
        handler: Callable,
        order: int = 100,
    ) -> None:
        """Register an extension handler.

        Args:
            name: Name of the extension type
            attr_name: Attribute name on VerticalExtensions object
            handler: Handler function (pipeline, orch, ext_value, context, result) -> None
            order: Execution order (lower = earlier)
        """
        self._handlers[name] = ExtensionHandlerInfo(
            name=name,
            attr_name=attr_name,
            handler=handler,
            order=order,
        )
        logger.debug(f"Registered extension handler: {name} (order={order})")

    def unregister(self, name: str) -> bool:
        """Unregister an extension handler.

        Args:
            name: Name of the extension type

        Returns:
            True if handler was removed, False if not found
        """
        if name in self._handlers:
            del self._handlers[name]
            return True
        return False

    def get_ordered_handlers(self) -> List[ExtensionHandlerInfo]:
        """Get handlers sorted by execution order.

        Returns:
            List of handler info sorted by order
        """
        return sorted(self._handlers.values(), key=lambda h: h.order)

    def get_handler(self, name: str) -> Optional[ExtensionHandlerInfo]:
        """Get a specific handler by name.

        Args:
            name: Name of the extension type

        Returns:
            Handler info or None if not found
        """
        return self._handlers.get(name)

    @classmethod
    def default(cls) -> "ExtensionHandlerRegistry":
        """Create registry with default extension handlers.

        Returns:
            Registry with built-in extension handlers
        """
        registry = cls()

        # Register default handlers in order
        registry.register(
            name="middleware",
            attr_name="middleware",
            handler=lambda p, o, e, c, r: p._apply_middleware(o, e, c, r),
            order=10,
        )
        registry.register(
            name="safety",
            attr_name="safety_extensions",
            handler=lambda p, o, e, c, r: p._apply_safety(o, e, c, r),
            order=20,
        )
        registry.register(
            name="prompts",
            attr_name="prompt_contributors",
            handler=lambda p, o, e, c, r: p._apply_prompts(o, e, c, r),
            order=30,
        )
        registry.register(
            name="mode_config",
            attr_name="mode_config_provider",
            handler=lambda p, o, e, c, r: p._apply_mode_config(o, e, c, r),
            order=40,
        )
        registry.register(
            name="tool_deps",
            attr_name="tool_dependency_provider",
            handler=lambda p, o, e, c, r: p._apply_tool_deps(o, e, c, r),
            order=50,
        )

        return registry


# Module-level default registry (lazy initialization)
_default_extension_registry: Optional[ExtensionHandlerRegistry] = None


def _get_legacy_extension_registry_no_warn() -> ExtensionHandlerRegistry:
    """Get legacy extension registry without emitting deprecation warnings."""
    global _default_extension_registry
    if _default_extension_registry is None:
        _default_extension_registry = ExtensionHandlerRegistry.default()
    return _default_extension_registry


def get_extension_handler_registry() -> ExtensionHandlerRegistry:
    """Get the default extension handler registry.

    Deprecated:
        This registry is not used by the active step-handler pipeline.

    Returns:
        The default extension handler registry
    """
    _warn_legacy_extension_registry_api("get_extension_handler_registry()")
    return _get_legacy_extension_registry_no_warn()


def register_extension_handler(
    name: str,
    attr_name: str,
    handler: Callable,
    order: int = 100,
) -> None:
    """Register a new extension handler (OCP extension point).

    Deprecated:
        Handlers registered here are inactive for the default integration path.

    This is a compatibility adapter only. The active integration path uses
    `victor.framework.step_handlers.ExtensionsStepHandler` and its registry.
    Handlers registered here affect only legacy/custom pipelines that explicitly
    consume this module-level registry.

    Args:
        name: Name of the extension type
        attr_name: Attribute name on VerticalExtensions object
        handler: Handler function (pipeline, orch, ext_value, context, result) -> None
        order: Execution order (lower = earlier)

    Example:
        def apply_chain_factory(pipeline, orch, chain_factory, ctx, result):
            chains = chain_factory.create_chains()
            # Apply chains...

        register_extension_handler(
            name="chain_factory",
            attr_name="chain_factory_provider",
            handler=apply_chain_factory,
            order=60,
        )
    """
    _warn_legacy_extension_registry_api("register_extension_handler()")
    registry = _get_legacy_extension_registry_no_warn()
    registry.register(name, attr_name, handler, order)


# =============================================================================
# Integration Result
# =============================================================================


@dataclass
class ExtensionLoadErrorInfo:
    """Information about an extension loading error.

    Captures details about a failed extension load for reporting.

    Attributes:
        extension_type: Type of extension that failed
        vertical_name: Name of the vertical
        error_message: The error message
        is_required: Whether this extension was required
        original_exception_type: Type name of the original exception
    """

    extension_type: str
    vertical_name: str
    error_message: str
    is_required: bool = False
    original_exception_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "extension_type": self.extension_type,
            "vertical_name": self.vertical_name,
            "error_message": self.error_message,
            "is_required": self.is_required,
            "original_exception_type": self.original_exception_type,
        }


# =============================================================================
# Enums for Vertical Integration
# =============================================================================


class ValidationStatus(str, Enum):
    """Validation status for vertical capabilities.

    Indicates the validation result:
    - SUCCESS: All capabilities validated successfully
    - PARTIAL: Some capabilities validated, others failed
    - FAILED: All capabilities failed validation
    """

    SUCCESS = "success"  # All capabilities validated
    PARTIAL = "partial"  # Some capabilities validated
    FAILED = "failed"  # All capabilities failed


# Backward compatibility
_ValidationStatusLiteral = Literal["success", "partial", "failed"]


@dataclass
class IntegrationResult:
    """Result from applying a vertical to an orchestrator.

    Attributes:
        success: Whether integration succeeded
        vertical_name: Name of the applied vertical
        context: The created vertical context.
        tools_applied: Set of tools enabled
        middleware_count: Number of middleware applied
        safety_patterns_count: Number of safety patterns applied
        prompt_hints_count: Number of prompt hints applied
        mode_configs_count: Number of mode configs applied
        workflows_count: Number of workflows registered
        rl_learners_count: Number of RL learners configured
        team_specs_count: Number of team specifications registered
        step_status: Per-step execution status for detailed auditing
        errors: List of errors encountered
        warnings: List of warnings
        info: List of informational messages
        extension_errors: List of ExtensionLoadErrorInfo for extension loading failures
        validation_status: Overall validation status - "success", "partial", or "failed"
    """

    success: bool = True
    vertical_name: Optional[str] = None
    context: Optional[MutableVerticalContextProtocol] = None
    tools_applied: Set[str] = field(default_factory=set)
    middleware_count: int = 0
    safety_patterns_count: int = 0
    prompt_hints_count: int = 0
    mode_configs_count: int = 0
    workflows_count: int = 0
    rl_learners_count: int = 0
    team_specs_count: int = 0
    step_status: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: List[str] = field(default_factory=list)
    extension_errors: List[ExtensionLoadErrorInfo] = field(default_factory=list)
    validation_status: ValidationStatus = "success"

    def add_error(self, error: str) -> None:
        """Add an error."""
        self.errors.append(error)
        self.success = False
        self._update_validation_status()

    def add_warning(self, warning: str) -> None:
        """Add a warning."""
        self.warnings.append(warning)

    def add_info(self, message: str) -> None:
        """Add an informational message."""
        self.info.append(message)

    def add_extension_error(
        self,
        extension_type: str,
        vertical_name: str,
        error_message: str,
        is_required: bool = False,
        original_exception: Optional[Exception] = None,
    ) -> None:
        """Add an extension loading error.

        Args:
            extension_type: Type of extension that failed
            vertical_name: Name of the vertical
            error_message: The error message
            is_required: Whether this extension was required
            original_exception: The original exception (optional)
        """
        error_info = ExtensionLoadErrorInfo(
            extension_type=extension_type,
            vertical_name=vertical_name,
            error_message=error_message,
            is_required=is_required,
            original_exception_type=(
                type(original_exception).__name__ if original_exception else None
            ),
        )
        self.extension_errors.append(error_info)

        # If required extension failed, mark as failed
        if is_required:
            self.success = False
            self.validation_status = "failed"
        else:
            # Non-required failures result in partial status
            self._update_validation_status()

    def _update_validation_status(self) -> None:
        """Update validation status based on current errors."""
        if not self.success:
            # Check if any required extension failed
            required_failures = [e for e in self.extension_errors if e.is_required]
            if required_failures or self.errors:
                self.validation_status = "failed"
            elif self.extension_errors or self.warnings:
                self.validation_status = "partial"
        elif self.extension_errors or self.warnings:
            self.validation_status = "partial"
        else:
            self.validation_status = "success"

    def has_extension_errors(self) -> bool:
        """Check if there are any extension loading errors.

        Returns:
            True if there are extension errors
        """
        return len(self.extension_errors) > 0

    def get_required_extension_failures(self) -> List[ExtensionLoadErrorInfo]:
        """Get list of required extension failures.

        Returns:
            List of ExtensionLoadErrorInfo for required extension failures
        """
        return [e for e in self.extension_errors if e.is_required]

    def record_step_status(
        self,
        step_name: str,
        status: str,
        details: Optional[Dict[str, Any]] = None,
        duration_ms: Optional[float] = None,
    ) -> None:
        """Record the execution status of a step handler.

        Tracks per-step status for detailed auditing and debugging.
        Each step is recorded with its status, optional details, and duration.

        Args:
            step_name: Unique identifier for the step handler
            status: Status string - "success", "error", "warning", or "skipped"
            details: Optional dictionary with additional details (counts, messages, etc.)
            duration_ms: Optional execution duration in milliseconds

        Example:
            result.record_step_status(
                "tools",
                "success",
                details={"tools_count": 15, "canonical_count": 12},
                duration_ms=5.3,
            )
        """
        step_record: Dict[str, Any] = {"status": status}
        if details is not None:
            step_record["details"] = details
        if duration_ms is not None:
            step_record["duration_ms"] = duration_ms
        self.step_status[step_name] = step_record

    def get_step_status(self, step_name: str) -> Optional[Dict[str, Any]]:
        """Get the status record for a specific step.

        Args:
            step_name: Name of the step to look up

        Returns:
            Status record dict or None if step not recorded
        """
        return self.step_status.get(step_name)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization (SRP: data -> dict).

        Returns:
            Dict representation suitable for JSON serialization
        """
        return {
            "success": self.success,
            "validation_status": self.validation_status,
            "vertical_name": self.vertical_name,
            "tools_applied": list(self.tools_applied),
            "middleware_count": self.middleware_count,
            "safety_patterns_count": self.safety_patterns_count,
            "prompt_hints_count": self.prompt_hints_count,
            "mode_configs_count": self.mode_configs_count,
            "workflows_count": self.workflows_count,
            "rl_learners_count": self.rl_learners_count,
            "team_specs_count": self.team_specs_count,
            "step_status": self.step_status,
            "errors": self.errors,
            "warnings": self.warnings,
            "info": self.info,
            "extension_errors": [e.to_dict() for e in self.extension_errors],
        }

    def persist(self, base_path: Optional[Path] = None) -> Optional[Path]:
        """Persist integration result to JSONL file for auditing.

        .. deprecated:: 0.7.0
            Use :class:`victor.observability.IntegrationAuditService` directly.
            ``IntegrationResult.persist()`` will be removed in v1.0.

        Args:
            base_path: Base path for audit logs. Defaults to ~/.victor/logs/integration/

        Returns:
            Path to the JSONL file, or None if persistence failed
        """
        import warnings

        warnings.warn(
            "IntegrationResult.persist() is deprecated and will be removed in v1.0. "
            "Use victor.observability.IntegrationAuditService().record(result) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from victor.observability.integration_audit import IntegrationAuditService

        return IntegrationAuditService().record(self, base_path=base_path)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IntegrationResult":
        """Create IntegrationResult from dictionary (SRP: deserialization).

        Args:
            data: Dictionary representation (from to_dict)

        Returns:
            IntegrationResult instance
        """
        # Deserialize extension errors
        extension_errors = []
        for error_dict in data.get("extension_errors", []):
            extension_errors.append(
                ExtensionLoadErrorInfo(
                    extension_type=error_dict.get("extension_type", "unknown"),
                    vertical_name=error_dict.get("vertical_name", "unknown"),
                    error_message=error_dict.get("error_message", ""),
                    is_required=error_dict.get("is_required", False),
                    original_exception_type=error_dict.get("original_exception_type"),
                )
            )

        return cls(
            success=data.get("success", True),
            validation_status=data.get("validation_status", "success"),
            vertical_name=data.get("vertical_name"),
            tools_applied=set(data.get("tools_applied", [])),
            middleware_count=data.get("middleware_count", 0),
            safety_patterns_count=data.get("safety_patterns_count", 0),
            prompt_hints_count=data.get("prompt_hints_count", 0),
            mode_configs_count=data.get("mode_configs_count", 0),
            workflows_count=data.get("workflows_count", 0),
            rl_learners_count=data.get("rl_learners_count", 0),
            team_specs_count=data.get("team_specs_count", 0),
            step_status=data.get("step_status", {}),
            errors=data.get("errors", []),
            warnings=data.get("warnings", []),
            info=data.get("info", []),
            extension_errors=extension_errors,
        )


# =============================================================================
# Integration Protocol
# =============================================================================


@runtime_checkable
class OrchestratorVerticalProtocol(Protocol):
    """Protocol for orchestrators that support vertical integration.

    This defines the methods an orchestrator must implement to
    properly receive vertical configuration. Using a protocol
    ensures SOLID compliance (no private attribute access).
    """

    def set_vertical_context(self, context: MutableVerticalContextProtocol) -> None:
        """Set the vertical context."""
        ...

    def set_enabled_tools(self, tools: Set[str]) -> None:
        """Set enabled tools from vertical."""
        ...

    def apply_vertical_middleware(self, middleware: List[Any]) -> None:
        """Apply middleware from vertical."""
        ...

    def apply_vertical_safety_patterns(self, patterns: List[Any]) -> None:
        """Apply safety patterns from vertical."""
        ...


# =============================================================================
# Integration Pipeline
# =============================================================================


class VerticalIntegrationPipeline:
    """Reusable pipeline for vertical extension application.

    This pipeline encapsulates all vertical integration logic that was
    previously spread across FrameworkShim._apply_vertical() and
    related methods. It provides:

    1. **Unified Integration**: Same logic for CLI and SDK paths
    2. **Protocol Compliance**: Uses proper methods, not private attrs
    3. **Error Handling**: Graceful degradation with detailed errors
    4. **Extensibility**: Hook points for custom integration steps
    5. **Step Handlers**: Focused classes for each integration concern (Phase 3.1)

    The pipeline now uses a registry of step handlers that can be customized.
    Each step handler implements a single concern (Single Responsibility Principle).
    New steps can be added without modifying existing ones (Open/Closed Principle).

    Architecture Notes: Pipeline vs Adapter
    ---------------------------------------
    This class (VerticalIntegrationPipeline) is the SETUP-PHASE orchestrator.
    It coordinates the complete vertical application process through step handlers.
    For runtime middleware/safety application, see VerticalIntegrationAdapter.

    - Pipeline: Use during agent creation, applies entire vertical config
    - Adapter: Use at runtime, applies middleware/safety to active requests

    Example:
        pipeline = VerticalIntegrationPipeline()

        # Apply with full vertical class
        result = pipeline.apply(orchestrator, vertical)

        # Apply with vertical name (from registry)
        result = pipeline.apply(orchestrator, "coding")

        # Apply with custom configuration
        result = pipeline.apply(
            orchestrator,
            "coding",
            config_overrides={"tool_budget": 30},
        )

        # Use custom step handlers (Phase 3.1)
        from victor.framework.step_handlers import StepHandlerRegistry

        registry = StepHandlerRegistry.default()
        registry.add_handler(MyCustomStepHandler())
        pipeline = VerticalIntegrationPipeline(step_registry=registry)
    """

    def __init__(
        self,
        strict_mode: bool = False,
        pre_hooks: Optional[List[Callable[[Any, Type], None]]] = None,
        post_hooks: Optional[List[Callable[[Any, IntegrationResult], None]]] = None,
        step_registry: Optional["StepHandlerRegistry"] = None,
        use_step_handlers: bool = True,
        extension_registry: Optional[ExtensionHandlerRegistry] = None,
        enable_cache: bool = True,
        cache_ttl: int = 3600,
        max_cache_entries: int = 256,
        cache_policy: Optional["VerticalIntegrationCachePolicy"] = None,
        parallel_enabled: bool = True,
        context_factory: Optional[Callable[..., "MutableVerticalContextProtocol"]] = None,
    ):
        """Initialize the pipeline.

        Args:
            strict_mode: If True, fail on any integration error
            pre_hooks: Callables to run before integration
            post_hooks: Callables to run after integration
            step_registry: Custom step handler registry (uses default if None)
            use_step_handlers: If True, use step handlers; if False, use legacy methods
            extension_registry: Legacy extension registry adapter. Ignored in active
                step-handler path; retained for backward-compatible signatures.
            enable_cache: If True, enable configuration caching (default: True)
            cache_ttl: Cache time-to-live in seconds (default: 3600 = 1 hour)
            max_cache_entries: Maximum number of cache entries to retain.
                Set <= 0 for unbounded cache.
            cache_policy: Optional custom cache policy implementation.
            parallel_enabled: If True, enable parallel execution (Phase 2.2, default: False)
            context_factory: Optional factory for the mutable vertical context protocol.
        """
        self._strict_mode = strict_mode
        self._pre_hooks = pre_hooks or []
        self._post_hooks = post_hooks or []
        self._use_step_handlers = use_step_handlers
        self._enable_cache = enable_cache
        self._cache_ttl = cache_ttl
        self._max_cache_entries = max_cache_entries
        self._parallel_enabled = parallel_enabled
        self._cache_policy = cache_policy or InMemoryLRUVerticalIntegrationCachePolicy()
        self._context_factory = context_factory

        # Initialize cache (eagerly, to avoid attribute errors in tests)
        self._cache: Dict[str, str] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self._cache_lock = threading.RLock()

        # Integration plan cache (P3): handler-level fingerprints used for
        # no-op/delta application on cache hits.
        self._integration_plan_cache: Dict[str, Dict[str, Any]] = {}
        self._integration_plan_timestamps: Dict[str, float] = {}
        self._integration_plan_lock = threading.RLock()
        self._applied_plan_by_orchestrator: "weakref.WeakKeyDictionary[Any, Dict[str, Any]]" = (
            weakref.WeakKeyDictionary()
        )
        # Persistent plan registry (survives GC, replaces WeakKeyDictionary for lookups)
        self._plan_registry = IntegrationPlanRegistry.get_instance()
        self._integration_plan_metrics: Dict[str, int] = {
            "compiled": 0,
            "reused": 0,
            "skip_decisions": 0,
            "skipped_handlers": 0,
            "full_replays": 0,
        }

        # Legacy pipeline-level extension registry is intentionally inactive.
        # Active extension integration lives in step_handlers.ExtensionsStepHandler.
        if extension_registry is not None:
            _warn_legacy_extension_registry_api(
                "VerticalIntegrationPipeline(extension_registry=...)"
            )
            logger.debug("Ignoring provided legacy extension_registry in active step-handler path.")

        # Initialize step handler registry
        if step_registry is not None:
            self._step_registry = step_registry
        elif use_step_handlers:
            try:
                from victor.framework.step_handlers import StepHandlerRegistry

                self._step_registry = StepHandlerRegistry.default()
            except ImportError:
                logger.debug("Step handlers not available, using legacy methods")
                self._step_registry = None
                self._use_step_handlers = False
        else:
            self._step_registry = None

    @property
    def step_registry(self) -> Optional["StepHandlerRegistry"]:
        """Get the step handler registry.

        Returns:
            The step handler registry or None if not using step handlers
        """
        return self._step_registry

    def apply(
        self,
        orchestrator: Any,
        vertical: Union[Type["VerticalBase"], str],
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> IntegrationResult:
        """Apply a vertical to an orchestrator.

        This is the main entry point for vertical integration. It:
        1. Resolves the vertical (if string name)
        2. Creates a mutable vertical context
        3. Applies all vertical extensions via step handlers (or legacy methods)
        4. Attaches context to orchestrator

        Args:
            orchestrator: AgentOrchestrator instance
            vertical: Vertical class or name string
            config_overrides: Optional configuration overrides

        Returns:
            IntegrationResult with details of what was applied
        """
        result = IntegrationResult()

        # Resolve vertical from name if needed
        vertical_class = self._resolve_vertical(vertical)
        if vertical_class is None:
            result.add_error(f"Vertical not found: {vertical}")
            return result

        manifest = get_or_create_vertical_manifest(vertical_class)
        runtime_metadata = get_vertical_runtime_metadata(vertical_class)

        with bind_active_vertical(
            runtime_metadata["vertical_name"],
            manifest_version=runtime_metadata["vertical_manifest_version"],
            namespace=runtime_metadata["vertical_plugin_namespace"],
        ):
            result.vertical_name = vertical_class.name

            cache_key: Optional[str] = None
            cache_hit = False

            # Check cache for pre-computed integration metadata (Phase 1: Caching)
            # NOTE: cached IntegrationResult does not include vertical context; we must
            # always replay step handlers to apply side effects on the target orchestrator.
            if self._enable_cache:
                cache_key = self._generate_cache_key(vertical_class)
                if cache_key:
                    cached_result = self._load_from_cache(cache_key)
                    if cached_result:
                        cache_hit = True
                        logger.debug(
                            f"Cache HIT for vertical '{vertical_class.name}' "
                            f"(key: {cache_key[:16]}...) - replaying integration"
                        )
                    else:
                        logger.debug(
                            f"Cache MISS for vertical '{vertical_class.name}' "
                            f"(key: {cache_key[:16]}...)"
                        )

            # Run pre-hooks
            for hook in self._pre_hooks:
                try:
                    hook(orchestrator, vertical_class)
                except Exception as e:
                    result.add_warning(f"Pre-hook error: {e}")

            # Create context
            context = self._create_context(orchestrator, vertical_class, result)
            if context is None:
                return result
            result.context = context

            # Apply integration steps (Phase 1: Remove legacy path)
            # Step handlers are now mandatory for SOLID compliance
            if self._step_registry is None:
                raise RuntimeError(
                    "StepHandlerRegistry required for vertical integration. "
                    "Ensure step handlers are initialized. "
                    "Use create_integration_pipeline() factory for proper setup."
                )

            self._validate_step_handler_dependency_contract(result)

            cached_plan: Optional[Dict[str, Any]] = None
            skip_handlers: Set[str] = set()
            if cache_hit and cache_key:
                cached_plan = self._load_plan_from_cache(cache_key)
                if cached_plan:
                    try:
                        context.set_capability_config(
                            "framework.internal.registration_version",
                            cached_plan.get("signature"),
                        )
                    except Exception:
                        pass
                skip_handlers = self._compute_skip_handlers(
                    orchestrator,
                    cache_key=cache_key,
                    cached_plan=cached_plan,
                )
                if skip_handlers:
                    result.add_info(
                        f"Skipped {len(skip_handlers)} side-effect handler(s) via integration-plan delta"
                    )

            # Use step handlers (Phase 3.1 - SOLID compliant single responsibility)
            self._apply_with_step_handlers(
                orchestrator,
                vertical_class,
                context,
                result,
                skip_handlers=skip_handlers,
            )

            # Run post-hooks
            for hook in self._post_hooks:
                try:
                    hook(orchestrator, result)
                except Exception as e:
                    result.add_warning(f"Post-hook error: {e}")

            logger.debug(
                f"Vertical integration complete: {result.vertical_name}, "
                f"tools={len(result.tools_applied)}, "
                f"middleware={result.middleware_count}, "
                f"safety_patterns={result.safety_patterns_count}"
            )

            # Emit vertical_applied event for observability
            self._emit_vertical_applied_event(orchestrator, result, cache_hit=cache_hit)

            # Note: result.persist() available for opt-in audit logging
            # Not called automatically to avoid duplication with EventBus

            # Save to cache (Phase 1: Caching)
            if self._enable_cache and result.success:
                if cache_key is None:
                    cache_key = self._generate_cache_key(vertical_class)
                if cache_key:
                    self._save_to_cache(cache_key, result)
                    plan = self._build_integration_plan(cache_key, result, base_plan=cached_plan)
                    self._save_plan_to_cache(cache_key, plan)
                    self._set_applied_plan_for_orchestrator(orchestrator, plan)
                    if result.context is not None:
                        try:
                            result.context.set_capability_config(
                                "framework.internal.integration_plan",
                                plan,
                            )
                        except Exception:
                            pass
                    if cache_hit:
                        result.add_info("Integration replayed from cache metadata")

            if manifest is not None:
                result.add_info(
                    "vertical_metadata="
                    f"{manifest.name}@{manifest.version}[{manifest.plugin_namespace}]"
                )

            return result

    def _generate_cache_key(self, vertical: Type["VerticalBase"]) -> Optional[str]:
        """Generate stable cache key from vertical signature.

        The key is based on:
        1. Vertical name
        2. Source code hash (for inline changes)
        3. File modification time (for file changes)

        This ensures cache invalidation when vertical code changes.

        Args:
            vertical: Vertical class

        Returns:
            Cache key string, or None if generation fails
        """
        try:
            # Try to get source file via inspect.getfile()
            # This works for normally imported modules
            try:
                source_file = Path(inspect.getfile(vertical))
            except (TypeError, OSError):
                # For dynamically loaded modules, try module.__file__
                if hasattr(vertical, "__module__"):
                    import sys

                    module_name = vertical.__module__
                    if module_name in sys.modules:
                        module = sys.modules[module_name]
                        if hasattr(module, "__file__") and module.__file__:
                            source_file = Path(module.__file__)
                        else:
                            # Fallback to class-based key
                            return self._generate_class_based_key(vertical)
                    else:
                        return self._generate_class_based_key(vertical)
                else:
                    return self._generate_class_based_key(vertical)

            source_hash = self._hash_source_file(source_file)

            # Combine into key
            key_parts = [
                f"vertical={vertical.name}",
                f"source={source_hash}",
            ]

            key_string = "|".join(key_parts)
            full_hash = hashlib.sha256(key_string.encode()).hexdigest()

            logger.debug("Cache key tier=v1 (source file) for vertical=%s", vertical.name)
            return f"v1_{vertical.name}_{full_hash[:16]}"

        except Exception as e:
            logger.warning(f"Failed to generate cache key: {e}")
            return None

    def _generate_class_based_key(self, vertical: Type["VerticalBase"]) -> Optional[str]:
        """Generate cache key from class properties when source file unavailable.

        This is a fallback for dynamically loaded classes or built-in classes.
        It attempts to include file hash if the module has a __file__ attribute.

        Args:
            vertical: Vertical class

        Returns:
            Cache key string
        """
        try:
            # First, try to get file hash from module
            if hasattr(vertical, "__module__"):
                import sys

                module_name = vertical.__module__

                # Try to get module from sys.modules first
                module = sys.modules.get(module_name)

                # If not in sys.modules, try to get module via inspect
                if not module:
                    try:
                        import inspect

                        # Get module from class using inspect
                        module = inspect.getmodule(vertical)
                    except Exception:
                        pass

                # If we have a module with __file__, use it for hashing
                if module and hasattr(module, "__file__") and module.__file__:
                    try:
                        source_file = Path(module.__file__)
                        if source_file.exists():
                            file_hash = self._hash_source_file(source_file)
                            # Use file hash as primary key component
                            key_parts = [
                                f"vertical={vertical.name}",
                                f"module={module_name}",
                                f"file={file_hash}",
                            ]
                            key_string = "|".join(key_parts)
                            full_hash = hashlib.sha256(key_string.encode()).hexdigest()
                            logger.debug(
                                "Cache key tier=v2 (module file fallback) for vertical=%s",
                                vertical.name,
                            )
                            return f"v2_{vertical.name}_{full_hash[:16]}"
                    except Exception:
                        pass  # Fall back to id-based key

            # Fallback: use class id (won't detect file changes, but stable)
            key_parts = [
                f"vertical={vertical.name}",
                f"id={id(vertical)}",
                f"module={vertical.__module__}",
            ]
            key_string = "|".join(key_parts)
            full_hash = hashlib.sha256(key_string.encode()).hexdigest()
            logger.debug(
                "Cache key tier=v3 (class id fallback) for vertical=%s — "
                "cache invalidation on code change is not available",
                vertical.name,
            )
            return f"v3_{vertical.name}_{full_hash[:16]}"

        except Exception:
            # Ultimate fallback
            logger.debug(
                "Cache key tier=v4 (id-only fallback) for vertical=%s — "
                "cache key is unstable across reloads",
                vertical.name,
            )
            return f"v4_{vertical.name}_{id(vertical)}"

    def _hash_source_file(self, source_file: Path) -> str:
        """Hash source file content and metadata.

        Args:
            source_file: Path to Python source file

        Returns:
            Hex digest hash combining content and mtime
        """
        try:
            # File content hash (first 16 bytes for performance)
            with open(source_file, "rb") as f:
                content_hash = hashlib.sha256(f.read(16384)).hexdigest()[:16]

            # Modification time (for file changes)
            mtime = source_file.stat().st_mtime_ns
            mtime_hash = hashlib.sha256(str(mtime).encode()).hexdigest()[:8]

            return f"{content_hash}_{mtime_hash}"

        except Exception as e:
            logger.warning(f"Failed to hash source file {source_file}: {e}")
            return f"unknown_{hash(source_file)}"

    def _load_from_cache(self, cache_key: str) -> Optional["IntegrationResult"]:
        """Load integration result from in-memory cache.

        Args:
            cache_key: Cache key

        Returns:
            Cached IntegrationResult or None
        """
        # Simple in-memory cache using JSON serialization
        # Note: Uses to_dict()/from_dict() to avoid pickle issues with
        # unpicklable middleware objects. The context is NOT cached.
        with self._cache_lock:
            cached_data = self._cache_policy.load(
                cache_key,
                cache=self._cache,
                timestamps=self._cache_timestamps,
                cache_ttl=self._cache_ttl,
            )
            if cached_data is None:
                return None

            try:
                result = IntegrationResult.from_dict(json_loads(cached_data))

                # Log metrics
                logger.debug(f"Loaded cached integration: {cache_key}")
                return result

            except (ValueError, TypeError, KeyError) as e:
                logger.warning(f"Failed to load from cache: {e}")
                # Clear corrupted cache entry
                self._cache.pop(cache_key, None)
                self._cache_timestamps.pop(cache_key, None)
                return None

    def _save_to_cache(self, cache_key: str, result: "IntegrationResult") -> None:
        """Save integration result to in-memory cache.

        Uses JSON serialization via to_dict() to avoid pickle issues with
        unpicklable middleware objects. Note: The vertical context is NOT
        cached as it contains unpicklable references.

        Args:
            cache_key: Cache key
            result: Integration result to cache
        """
        try:
            # Serialize result using JSON (avoids pickle issues)
            data = json_dumps(result.to_dict())

            with self._cache_lock:
                self._cache_policy.save(
                    cache_key,
                    data,
                    cache=self._cache,
                    timestamps=self._cache_timestamps,
                    max_entries=self._max_cache_entries,
                )

            logger.debug(f"Cached integration result: {cache_key}")

        except (TypeError, ValueError) as e:
            logger.warning(f"Failed to save to cache: {e}")

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics for observability and diagnostics."""
        with self._cache_lock:
            return self._cache_policy.get_stats(
                cache=self._cache,
                max_entries=self._max_cache_entries,
            )

    def clear_cache(self) -> None:
        """Clear integration cache and reset cache policy metrics."""
        with self._cache_lock:
            self._cache_policy.clear(
                cache=self._cache,
                timestamps=self._cache_timestamps,
            )
        with self._integration_plan_lock:
            self._integration_plan_cache.clear()
            self._integration_plan_timestamps.clear()
            for key in list(self._integration_plan_metrics.keys()):
                self._integration_plan_metrics[key] = 0
        self._applied_plan_by_orchestrator = weakref.WeakKeyDictionary()
        self._plan_registry.clear()

    def _hash_plan_payload(self, payload: Any) -> str:
        """Create a deterministic fingerprint for plan payloads."""
        try:
            raw = json_dumps(payload, sort_keys=True, default=str)
        except TypeError:
            raw = repr(payload)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def _build_integration_plan(
        self,
        cache_key: str,
        result: IntegrationResult,
        *,
        base_plan: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build handler-level integration plan from step execution status."""
        base_fingerprints: Dict[str, str] = {}
        if base_plan:
            base_fingerprints = dict(base_plan.get("handler_fingerprints", {}))

        fingerprints: Dict[str, str] = {}
        for handler_name, status in result.step_status.items():
            if (
                status.get("status") == "skipped"
                and status.get("details", {}).get("reason") == _INTEGRATION_PLAN_SKIP_REASON
                and handler_name in base_fingerprints
            ):
                fingerprints[handler_name] = base_fingerprints[handler_name]
                continue
            fingerprints[handler_name] = self._hash_plan_payload(status)

        signature = self._hash_plan_payload(
            {
                "cache_key": cache_key,
                "handler_fingerprints": fingerprints,
            }
        )
        state_snapshot: Dict[str, Any] = {}
        if result.context is not None:
            state_snapshot["vertical_name"] = result.context.name
            state_snapshot["enabled_tools_hash"] = self._hash_plan_payload(
                sorted(result.context.enabled_tools)
            )

        return {
            "version": _INTEGRATION_PLAN_VERSION,
            "cache_key": cache_key,
            "vertical_name": result.vertical_name,
            "signature": signature,
            "handler_fingerprints": fingerprints,
            "state_snapshot": state_snapshot,
            "created_at": time.time(),
        }

    def _save_plan_to_cache(self, cache_key: str, plan: Dict[str, Any]) -> None:
        """Save integration plan with TTL and bounded size semantics."""
        with self._integration_plan_lock:
            self._integration_plan_cache[cache_key] = copy.deepcopy(plan)
            self._integration_plan_timestamps[cache_key] = time.monotonic()
            self._integration_plan_metrics["compiled"] += 1

            if self._max_cache_entries > 0:
                while len(self._integration_plan_cache) > self._max_cache_entries:
                    oldest_key = next(iter(self._integration_plan_cache))
                    self._integration_plan_cache.pop(oldest_key, None)
                    self._integration_plan_timestamps.pop(oldest_key, None)

    def _load_plan_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Load integration plan from in-memory cache."""
        with self._integration_plan_lock:
            plan = self._integration_plan_cache.get(cache_key)
            if plan is None:
                return None

            ts = self._integration_plan_timestamps.get(cache_key)
            if ts is not None and self._cache_ttl > 0:
                age_seconds = time.monotonic() - ts
                if age_seconds > self._cache_ttl:
                    self._integration_plan_cache.pop(cache_key, None)
                    self._integration_plan_timestamps.pop(cache_key, None)
                    return None

            # LRU touch
            touched = self._integration_plan_cache.pop(cache_key)
            self._integration_plan_cache[cache_key] = touched
            self._integration_plan_metrics["reused"] += 1
            return copy.deepcopy(touched)

    def get_integration_plan_stats(self) -> Dict[str, int]:
        """Get integration-plan cache and skip metrics."""
        with self._integration_plan_lock:
            stats = dict(self._integration_plan_metrics)
            stats["size"] = len(self._integration_plan_cache)
            stats["max_entries"] = self._max_cache_entries
        return stats

    def _get_capability_snapshot_value(self, orchestrator: Any, capability: str) -> Any:
        """Read capability value through registry protocol when available."""
        has_capability = getattr(orchestrator, "has_capability", None)
        get_capability_value = getattr(orchestrator, "get_capability_value", None)
        if not callable(has_capability) or not callable(get_capability_value):
            return None

        try:
            if not has_capability(capability):
                return None
        except Exception:
            return None

        try:
            return get_capability_value(capability)
        except Exception:
            return None

    def _resolve_vertical_context(self, orchestrator: Any) -> Optional[Any]:
        """Resolve vertical context via protocol-first reads."""
        context = self._get_capability_snapshot_value(orchestrator, "vertical_context")
        if context is not None:
            return context

        ensure_protocol_fallback_allowed(
            operation="vertical context snapshot",
            fallback_target="get_vertical_context",
        )
        getter = getattr(orchestrator, "get_vertical_context", None)
        if callable(getter):
            try:
                return getter()
            except Exception:
                return None
        return None

    def _resolve_enabled_tools_snapshot(self, orchestrator: Any) -> Optional[Set[str]]:
        """Resolve enabled tools via protocol-first reads."""
        value = self._get_capability_snapshot_value(orchestrator, "enabled_tools")
        if value is not None:
            try:
                return set(value)
            except Exception:
                return None

        ensure_protocol_fallback_allowed(
            operation="enabled tools snapshot",
            fallback_target="get_enabled_tools",
        )
        getter = getattr(orchestrator, "get_enabled_tools", None)
        if callable(getter):
            try:
                value = getter()
            except Exception:
                return None
            if value is not None:
                try:
                    return set(value)
                except Exception:
                    return None
        return None

    def _get_applied_plan_for_orchestrator(self, orchestrator: Any) -> Optional[Dict[str, Any]]:
        """Get previously applied plan metadata for this orchestrator."""
        # Check persistent registry first (survives GC)
        registry_plan = self._plan_registry.get_plan(orchestrator)
        if registry_plan is not None:
            return registry_plan

        # Fallback to WeakKeyDictionary (legacy, kept for backward compat)
        try:
            cached = self._applied_plan_by_orchestrator.get(orchestrator)
        except TypeError:
            cached = None
        if cached is not None:
            return copy.deepcopy(cached)

        context = self._resolve_vertical_context(orchestrator)

        if context is None:
            return None

        get_config = getattr(context, "get_capability_config", None)
        if not callable(get_config):
            return None
        try:
            plan = get_config("framework.internal.integration_plan", None)
        except Exception:
            return None

        if isinstance(plan, dict):
            return copy.deepcopy(plan)
        return None

    def _set_applied_plan_for_orchestrator(self, orchestrator: Any, plan: Dict[str, Any]) -> None:
        """Store applied plan metadata for orchestrator delta/no-op checks."""
        # Write to persistent registry (primary)
        self._plan_registry.set_plan(orchestrator, plan)
        # Also write to WeakKeyDictionary (legacy, kept for backward compat)
        try:
            self._applied_plan_by_orchestrator[orchestrator] = copy.deepcopy(plan)
        except TypeError:
            # Non-weakref-able orchestrator; skip side cache.
            pass

    def _orchestrator_state_snapshot(self, orchestrator: Any) -> Dict[str, Any]:
        """Collect lightweight orchestrator state used for no-op safety checks."""
        snapshot: Dict[str, Any] = {}

        context = self._resolve_vertical_context(orchestrator)

        if context is not None:
            snapshot["vertical_name"] = getattr(context, "name", None)

        enabled_tools = self._resolve_enabled_tools_snapshot(orchestrator)
        if enabled_tools is not None:
            snapshot["enabled_tools_hash"] = self._hash_plan_payload(sorted(enabled_tools))

        return snapshot

    def _plan_state_matches(self, orchestrator: Any, plan: Dict[str, Any]) -> bool:
        """Return True when orchestrator state is compatible with applied-plan snapshot."""
        expected = plan.get("state_snapshot", {})
        if not expected:
            return True

        actual = self._orchestrator_state_snapshot(orchestrator)

        for key in ("vertical_name", "enabled_tools_hash"):
            expected_value = expected.get(key)
            if expected_value is None:
                continue
            actual_value = actual.get(key)
            if actual_value != expected_value:
                return False
        return True

    def _compute_skip_handlers(
        self,
        orchestrator: Any,
        *,
        cache_key: Optional[str],
        cached_plan: Optional[Dict[str, Any]],
    ) -> Set[str]:
        """Compute side-effect handlers that can be skipped as no-op delta."""
        if cache_key is None or cached_plan is None or self._step_registry is None:
            with self._integration_plan_lock:
                self._integration_plan_metrics["full_replays"] += 1
            return set()

        applied_plan = self._get_applied_plan_for_orchestrator(orchestrator)
        if not applied_plan:
            with self._integration_plan_lock:
                self._integration_plan_metrics["full_replays"] += 1
            return set()

        if applied_plan.get("cache_key") != cache_key:
            with self._integration_plan_lock:
                self._integration_plan_metrics["full_replays"] += 1
            return set()

        if not self._plan_state_matches(orchestrator, applied_plan):
            with self._integration_plan_lock:
                self._integration_plan_metrics["full_replays"] += 1
            return set()

        cached_fingerprints = cached_plan.get("handler_fingerprints", {})
        applied_fingerprints = applied_plan.get("handler_fingerprints", {})

        skip: Set[str] = set()
        for handler in self._step_registry.get_ordered_handlers():
            name = str(getattr(handler, "name", handler.__class__.__name__))
            has_side_effects = bool(getattr(handler, "side_effects", True))
            if not has_side_effects:
                continue
            expected = cached_fingerprints.get(name)
            current = applied_fingerprints.get(name)
            if expected and current and expected == current:
                skip.add(name)

        with self._integration_plan_lock:
            self._integration_plan_metrics["skip_decisions"] += 1
            self._integration_plan_metrics["skipped_handlers"] += len(skip)
        return skip

    def _record_skipped_handler(self, result: IntegrationResult, handler_name: str) -> None:
        """Record skipped handler status for integration-plan no-op delta."""
        result.record_step_status(
            handler_name,
            "skipped",
            details={"reason": _INTEGRATION_PLAN_SKIP_REASON},
        )

    def _build_vertical_applied_payload(
        self,
        orchestrator: Any,
        result: IntegrationResult,
        *,
        cache_hit: bool,
    ) -> Dict[str, Any]:
        """Build observability payload for vertical.applied events."""
        cache_stats = self.get_cache_stats() if self._enable_cache else {}
        integration_plan_stats = self.get_integration_plan_stats() if self._enable_cache else {}
        extension_loader_metrics: Dict[str, Any] = {}
        observability_delivery_stats: Dict[str, Any] = {}
        framework_registry_metrics: Dict[str, Any] = {}
        try:
            from victor.core.verticals.extension_loader import VerticalExtensionLoader

            extension_loader_metrics = VerticalExtensionLoader.get_extension_loader_metrics()
        except Exception:
            extension_loader_metrics = {}
        try:
            from victor.core.events import get_observability_bus

            bus = get_observability_bus()
            getter = getattr(bus, "get_delivery_pressure_stats", None) if bus else None
            if callable(getter):
                observability_delivery_stats = getter() or {}
        except Exception:
            observability_delivery_stats = {}
        try:
            registry_service = resolve_framework_integration_registry_service(orchestrator)
            snapshot_metrics = getattr(registry_service, "snapshot_metrics", None)
            if callable(snapshot_metrics):
                framework_registry_metrics = snapshot_metrics() or {}
        except Exception:
            framework_registry_metrics = {}

        return {
            "vertical": result.vertical_name,
            **(
                get_vertical_runtime_metadata(vertical_cls)
                if (
                    result.vertical_name
                    and (vertical_cls := VerticalRegistry.get(result.vertical_name)) is not None
                )
                else {
                    "vertical_manifest_version": "",
                    "vertical_plugin_namespace": "",
                }
            ),
            "tools_count": len(result.tools_applied),
            "middleware_count": result.middleware_count,
            "safety_patterns_count": result.safety_patterns_count,
            "prompt_hints_count": result.prompt_hints_count,
            "workflows_count": result.workflows_count,
            "rl_learners_count": result.rl_learners_count,
            "team_specs_count": result.team_specs_count,
            "success": result.success,
            "error_count": len(result.errors),
            "warning_count": len(result.warnings),
            "cache_enabled": self._enable_cache,
            "cache_hit": cache_hit,
            "cache_stats": cache_stats,
            "integration_plan_stats": integration_plan_stats,
            "extension_loader_metrics": extension_loader_metrics,
            "observability_delivery_stats": observability_delivery_stats,
            "framework_registry_metrics": framework_registry_metrics,
            "category": "vertical",  # Preserve for observability
        }

    def _emit_vertical_applied_event(
        self,
        orchestrator: Any,
        result: IntegrationResult,
        *,
        cache_hit: bool,
    ) -> None:
        """Emit vertical.applied event from sync execution contexts."""
        try:
            from victor.core.events import get_observability_bus

            bus = get_observability_bus()
            if bus:
                payload = self._build_vertical_applied_payload(
                    orchestrator,
                    result,
                    cache_hit=cache_hit,
                )
                emit_event_sync(
                    bus,
                    "vertical.applied",
                    payload,
                    source="VerticalIntegrationPipeline",
                    use_background_loop=True,
                )
        except Exception as e:
            logger.debug(f"Failed to emit vertical_applied event: {e}")

    async def _emit_vertical_applied_event_async(
        self,
        orchestrator: Any,
        result: IntegrationResult,
        *,
        cache_hit: bool,
    ) -> None:
        """Emit vertical.applied event from async execution contexts."""
        try:
            from victor.core.events import get_observability_bus

            bus = get_observability_bus()
            if bus:
                payload = self._build_vertical_applied_payload(
                    orchestrator,
                    result,
                    cache_hit=cache_hit,
                )
                await bus.emit(
                    topic="vertical.applied",
                    data=payload,
                    source="VerticalIntegrationPipeline",
                )
        except Exception as e:
            logger.debug(f"Failed to emit vertical_applied event (async): {e}")

    async def apply_async(
        self,
        orchestrator: Any,
        vertical: Union[Type["VerticalBase"], str],
    ) -> IntegrationResult:
        """Apply vertical integration asynchronously.

        Uses parallel step execution when parallel_enabled=True (default via
        vertical_service). Parallel-safe handlers run concurrently within
        dependency levels; others run sequentially.

        Args:
            orchestrator: Orchestrator instance
            vertical: Vertical class or name string

        Returns:
            IntegrationResult with status and metadata

        Example:
            pipeline = VerticalIntegrationPipeline()
            result = await pipeline.apply_async(orchestrator, "coding")
        """

        # Resolve vertical
        vertical_cls = self._resolve_vertical(vertical)
        if vertical_cls is None:
            result = IntegrationResult(vertical_name=str(vertical))
            result.add_error(f"Vertical not found: {vertical}")
            return result

        cache_key: Optional[str] = None
        cache_hit = False

        # Check cache first (Phase 1)
        # NOTE: like sync apply(), async apply must replay handlers to apply
        # side effects on the target orchestrator.
        if self._enable_cache:
            cache_key = self._generate_cache_key(vertical_cls)
            if cache_key:
                cached_result = self._load_from_cache(cache_key)
                if cached_result:
                    cache_hit = True
                    logger.debug(
                        f"Cache HIT for vertical '{vertical_cls.name}' "
                        f"(key: {cache_key[:16]}...) - replaying integration"
                    )
                else:
                    logger.debug(
                        f"Cache MISS for vertical '{vertical_cls.name}' "
                        f"(key: {cache_key[:16]}...)"
                    )

        # Create context and result
        result = IntegrationResult(vertical_name=vertical_cls.name)
        context = self._create_context(orchestrator, vertical_cls, result)
        if context is None:
            result.add_error("Failed to create vertical context")
            return result
        result.context = context

        # Execute step handlers (Phase 2.1-2.2)
        if self._step_registry is None:
            result.add_error("StepHandlerRegistry required for vertical integration")
            return result

        self._validate_step_handler_dependency_contract(result)

        cached_plan: Optional[Dict[str, Any]] = None
        skip_handlers: Set[str] = set()
        if cache_hit and cache_key:
            cached_plan = self._load_plan_from_cache(cache_key)
            skip_handlers = self._compute_skip_handlers(
                orchestrator,
                cache_key=cache_key,
                cached_plan=cached_plan,
            )
            if skip_handlers:
                result.add_info(
                    f"Skipped {len(skip_handlers)} side-effect handler(s) via integration-plan delta"
                )

        # Choose execution strategy based on parallel_enabled flag
        if self._parallel_enabled:
            # Parallel execution (Phase 2.2)
            await self._apply_with_step_handlers_parallel(
                orchestrator,
                vertical_cls,
                context,
                result,
                skip_handlers=skip_handlers,
            )
        else:
            # Sequential execution (Phase 2.1)
            for handler in self._step_registry.get_ordered_handlers():
                handler_name = str(getattr(handler, "name", handler.__class__.__name__))
                if handler_name in skip_handlers:
                    self._record_skipped_handler(result, handler_name)
                    continue
                try:
                    # Check if handler has async apply method
                    if hasattr(handler, "apply_async"):
                        await handler.apply_async(
                            orchestrator,
                            vertical_cls,
                            context,
                            result,
                            strict_mode=self._strict_mode,
                        )
                    else:
                        # Fallback to sync apply
                        handler.apply(
                            orchestrator,
                            vertical_cls,
                            context,
                            result,
                            strict_mode=self._strict_mode,
                        )
                    if self._should_stop_after_handler(handler_name, result):
                        return
                except Exception as e:
                    if self._strict_mode:
                        result.add_error(f"Step handler '{handler.name}' failed: {e}")
                    else:
                        result.add_warning(f"Step handler '{handler.name}' error: {e}")
                    logger.debug(
                        f"Step handler '{handler.name}' failed: {e}",
                        exc_info=True,
                    )
                    if self._should_stop_after_handler(handler_name, result):
                        return

        # Emit vertical_applied event for observability (async parity with sync apply)
        await self._emit_vertical_applied_event_async(orchestrator, result, cache_hit=cache_hit)

        # Cache result (Phase 1)
        if self._enable_cache and result.success:
            if cache_key is None:
                cache_key = self._generate_cache_key(vertical_cls)
            if cache_key:
                self._save_to_cache(cache_key, result)
                plan = self._build_integration_plan(cache_key, result, base_plan=cached_plan)
                self._save_plan_to_cache(cache_key, plan)
                self._set_applied_plan_for_orchestrator(orchestrator, plan)
                if result.context is not None:
                    try:
                        result.context.set_capability_config(
                            "framework.internal.integration_plan",
                            plan,
                        )
                    except Exception:
                        pass
                if cache_hit:
                    result.add_info("Async integration replayed from cache metadata")

        return result

    def _classify_handlers(self, handlers: List[Any]) -> Tuple[List[Any], List[Any]]:
        """Classify handlers into parallel-safe and sequential groups.

        Classification is metadata-driven to avoid class-name coupling.
        A handler is eligible for parallel execution only when:
        - `parallel_safe` metadata is True, and
        - `side_effects` metadata is False, and
        - it has no declared dependencies on handlers present in this run.

        Args:
            handlers: Ordered list of step handlers.

        Returns:
            Tuple of (parallel_safe_handlers, sequential_handlers)
        """
        independent: List[Any] = []
        dependent: List[Any] = []
        ordered_handlers = sorted(handlers, key=self._handler_sort_key)
        known_handler_names = {
            getattr(handler, "name", None)
            for handler in ordered_handlers
            if getattr(handler, "name", None) is not None
        }

        for handler in ordered_handlers:
            parallel_safe = self._is_parallel_eligible(handler)
            declared_dependencies = tuple(getattr(handler, "depends_on", ()) or ())
            active_dependencies = [d for d in declared_dependencies if d in known_handler_names]

            if parallel_safe and not active_dependencies:
                independent.append(handler)
            else:
                dependent.append(handler)

        return independent, dependent

    def _handler_sort_key(self, handler: Any) -> Tuple[int, str]:
        """Return deterministic sort key for handlers.

        Sorting is stable across runs even with mixed handler implementations.
        """
        order = getattr(handler, "order", 1000)
        try:
            order_value = int(order)
        except (TypeError, ValueError):
            order_value = 1000
        name = str(getattr(handler, "name", handler.__class__.__name__))
        return (order_value, name)

    def _is_parallel_eligible(self, handler: Any) -> bool:
        """Return True if handler is safe for concurrent execution.

        Safety invariant:
        - Handler must explicitly opt in (`parallel_safe=True`)
        - Handler must be side-effect free (`side_effects=False`)
        """
        parallel_safe = bool(getattr(handler, "parallel_safe", False))
        has_side_effects = bool(getattr(handler, "side_effects", True))
        return parallel_safe and not has_side_effects

    def _collect_dependency_contract_violations(self, handlers: List[Any]) -> List[str]:
        """Collect step-handler dependency contract violations.

        Validates that:
        - each handler has a unique name in the active registry
        - each declared dependency points to an active handler name
        """
        if not handlers:
            return []

        known_names: Set[str] = set()
        duplicate_names: Set[str] = set()
        violations: List[str] = []

        for index, handler in enumerate(handlers):
            handler_name = str(getattr(handler, "name", f"handler_{index}"))
            if handler_name in known_names:
                duplicate_names.add(handler_name)
            known_names.add(handler_name)

        if duplicate_names:
            duplicates = ", ".join(sorted(duplicate_names))
            violations.append(f"duplicate handler name(s): {duplicates}")

        for index, handler in enumerate(handlers):
            handler_name = str(getattr(handler, "name", f"handler_{index}"))
            declared_dependencies = tuple(getattr(handler, "depends_on", ()) or ())
            missing_dependencies = sorted(
                {str(dep) for dep in declared_dependencies if str(dep) not in known_names}
            )
            if missing_dependencies:
                missing_names = ", ".join(missing_dependencies)
                violations.append(
                    f"handler '{handler_name}' depends on missing handler(s): {missing_names}"
                )

        return violations

    def _validate_step_handler_dependency_contract(self, result: IntegrationResult) -> bool:
        """Validate active step-handler dependency metadata.

        Returns:
            True when contract is valid, False when violations are recorded.
        """
        if self._step_registry is None:
            return True

        handlers = self._step_registry.get_ordered_handlers()
        violations = self._collect_dependency_contract_violations(handlers)
        if not violations:
            return True

        status = "error" if self._strict_mode else "warning"
        for violation in violations:
            message = f"Step-handler dependency contract violation: {violation}"
            if self._strict_mode:
                result.add_error(message)
            else:
                result.add_warning(message)

        result.record_step_status(
            "dependency_contract",
            status,
            details={
                "strict_mode": self._strict_mode,
                "violations": violations,
            },
        )
        logger.warning(
            "Step-handler dependency contract violations detected (%s): %s",
            status,
            "; ".join(violations),
        )
        return False

    def _build_execution_levels(self, handlers: List[Any]) -> List[List[Any]]:
        """Build deterministic execution levels from metadata dependencies.

        Uses Kahn topological layering over `depends_on`. Each returned level
        contains handlers whose active dependencies are satisfied and can be
        considered for concurrent execution by metadata policy.

        If a cycle is detected, the method falls back to deterministic
        order-based sequential levels.
        """
        if not handlers:
            return []

        ordered_handlers = sorted(handlers, key=self._handler_sort_key)
        name_to_handler: Dict[str, Any] = {}
        duplicate_counts: Dict[str, int] = {}
        ordered_names: List[str] = []

        for index, handler in enumerate(ordered_handlers):
            raw_name = str(getattr(handler, "name", f"handler_{index}"))
            if raw_name in name_to_handler:
                duplicate_counts[raw_name] = duplicate_counts.get(raw_name, 0) + 1
                unique_name = f"{raw_name}#{duplicate_counts[raw_name]}"
                logger.warning(
                    "Duplicate step-handler name '%s' detected; using '%s' for dependency graph",
                    raw_name,
                    unique_name,
                )
            else:
                duplicate_counts[raw_name] = 0
                unique_name = raw_name

            name_to_handler[unique_name] = handler
            ordered_names.append(unique_name)

        indegree: Dict[str, int] = dict.fromkeys(ordered_names, 0)
        dependents: Dict[str, Set[str]] = {name: set() for name in ordered_names}

        for name in ordered_names:
            handler = name_to_handler[name]
            declared_dependencies = tuple(getattr(handler, "depends_on", ()) or ())
            active_dependencies = [dep for dep in declared_dependencies if dep in name_to_handler]

            for dependency in active_dependencies:
                indegree[name] += 1
                dependents[dependency].add(name)

        completed: Set[str] = set()
        levels: List[List[Any]] = []
        ready = [name for name in ordered_names if indegree[name] == 0]

        while ready:
            ready_sorted = sorted(
                ready, key=lambda item: self._handler_sort_key(name_to_handler[item])
            )
            levels.append([name_to_handler[name] for name in ready_sorted])
            completed.update(ready_sorted)

            for done in ready_sorted:
                for dependent in dependents.get(done, ()):
                    indegree[dependent] -= 1

            ready = [
                name for name in ordered_names if name not in completed and indegree[name] == 0
            ]

        if len(completed) != len(ordered_names):
            unresolved = [name for name in ordered_names if name not in completed]
            logger.warning(
                "Detected cyclical step-handler dependencies (%s); "
                "falling back to deterministic sequential order",
                unresolved,
            )
            return [[handler] for handler in ordered_handlers]

        return levels

    def _build_context(self, **kwargs: Any) -> "MutableVerticalContextProtocol":
        # DIP bridge: this is the only intentional framework→agent dependency.
        # Eliminate by passing context_factory= at VerticalIntegrationPipeline construction.
        if self._context_factory is not None:
            return self._context_factory(**kwargs)
        from victor.agent.vertical_context import create_vertical_context

        return create_vertical_context(**kwargs)

    async def _apply_with_step_handlers_parallel(
        self,
        orchestrator: Any,
        vertical: Type["VerticalBase"],
        context: MutableVerticalContextProtocol,
        result: IntegrationResult,
        *,
        skip_handlers: Optional[Set[str]] = None,
    ) -> None:
        """Apply step handlers with parallel execution (Phase 2.2).

        Builds metadata-driven dependency levels and executes each level with:
        - sequential run for handlers that are not parallel-eligible
        - concurrent run for handlers that are parallel-eligible

        Args:
            orchestrator: Orchestrator instance
            vertical: Vertical class
            context: Vertical context
            result: Integration result
        """
        import asyncio

        if self._step_registry is None:
            return

        handlers = self._step_registry.get_ordered_handlers()
        levels = self._build_execution_levels(handlers)
        skip_names = skip_handlers or set()

        for level_index, level_handlers in enumerate(levels):
            sequential_handlers: List[Any] = []
            parallel_handlers: List[Any] = []

            for handler in level_handlers:
                handler_name = str(getattr(handler, "name", handler.__class__.__name__))
                if handler_name in skip_names:
                    self._record_skipped_handler(result, handler_name)
                    continue
                if self._is_parallel_eligible(handler):
                    parallel_handlers.append(handler)
                else:
                    sequential_handlers.append(handler)

            if sequential_handlers:
                logger.debug(
                    "Executing level %s sequential handlers: %s",
                    level_index,
                    [getattr(h, "name", h.__class__.__name__) for h in sequential_handlers],
                )
                for handler in sequential_handlers:
                    await self._run_handler_async(handler, orchestrator, vertical, context, result)
                    handler_name = str(getattr(handler, "name", handler.__class__.__name__))
                    if self._should_stop_after_handler(handler_name, result):
                        return

            if parallel_handlers:
                logger.debug(
                    "Executing level %s parallel handlers: %s",
                    level_index,
                    [getattr(h, "name", h.__class__.__name__) for h in parallel_handlers],
                )
                tasks = [
                    self._run_handler_async(h, orchestrator, vertical, context, result)
                    for h in parallel_handlers
                ]
                results_or_exc = await asyncio.gather(*tasks, return_exceptions=True)
                for idx, maybe_exc in enumerate(results_or_exc):
                    if isinstance(maybe_exc, Exception):
                        failed_handler = parallel_handlers[idx]
                        message = self._format_handler_failure(
                            vertical,
                            failed_handler.name,
                            maybe_exc,
                        )
                        if self._strict_mode:
                            result.add_error(message)
                        else:
                            result.add_warning(message)
                    handler_name = str(
                        getattr(
                            parallel_handlers[idx],
                            "name",
                            parallel_handlers[idx].__class__.__name__,
                        )
                    )
                    if self._should_stop_after_handler(handler_name, result):
                        return

    async def _run_handler_async(
        self,
        handler: Any,
        orchestrator: Any,
        vertical: Type["VerticalBase"],
        context: MutableVerticalContextProtocol,
        result: IntegrationResult,
    ) -> None:
        """Run a single handler with async support (Phase 2.2).

        Args:
            handler: Step handler instance
            orchestrator: Orchestrator instance
            vertical: Vertical class
            context: Vertical context
            result: Integration result

        Raises:
            Exception: If handler fails and strict_mode is enabled
        """
        runtime_metadata = get_vertical_runtime_metadata(vertical)
        namespace = runtime_metadata["vertical_plugin_namespace"] or "default"
        try:
            # Check if handler has async apply method
            if hasattr(handler, "apply_async"):
                await handler.apply_async(
                    orchestrator,
                    vertical,
                    context,
                    result,
                    strict_mode=self._strict_mode,
                )
            else:
                # Run sync handler in thread pool to avoid blocking
                import asyncio

                loop = asyncio.get_running_loop()
                executor = get_namespace_executor_pool().get_executor(namespace)
                await loop.run_in_executor(
                    executor,
                    lambda: handler.apply(
                        orchestrator,
                        vertical,
                        context,
                        result,
                        strict_mode=self._strict_mode,
                    ),
                )
        except Exception as e:
            if self._strict_mode:
                raise
            result.add_warning(self._format_handler_failure(vertical, handler.name, e))
            logger.debug(
                "Handler '%s' failed for vertical '%s' in namespace '%s': %s",
                handler.name,
                runtime_metadata["vertical_name"],
                namespace,
                e,
                exc_info=True,
            )

    def _apply_with_step_handlers(
        self,
        orchestrator: Any,
        vertical: Type["VerticalBase"],
        context: MutableVerticalContextProtocol,
        result: IntegrationResult,
        *,
        skip_handlers: Optional[Set[str]] = None,
    ) -> None:
        """Apply vertical using step handlers.

        This is the new Phase 3.1 implementation using focused step handler
        classes for each concern.

        Args:
            orchestrator: Orchestrator instance
            vertical: Vertical class
            context: Vertical context
            result: Integration result
        """
        if self._step_registry is None:
            return

        skip_names = skip_handlers or set()

        # Execute all step handlers in order
        for handler in self._step_registry.get_ordered_handlers():
            handler_name = str(getattr(handler, "name", handler.__class__.__name__))
            if handler_name in skip_names:
                self._record_skipped_handler(result, handler_name)
                continue
            try:
                handler.apply(
                    orchestrator,
                    vertical,
                    context,
                    result,
                    strict_mode=self._strict_mode,
                )
                if self._should_stop_after_handler(handler_name, result):
                    return
            except Exception as e:
                message = self._format_handler_failure(vertical, handler.name, e)
                if self._strict_mode:
                    result.add_error(message)
                else:
                    result.add_warning(message)
                logger.debug(
                    message,
                    exc_info=True,
                )
                if self._should_stop_after_handler(handler_name, result):
                    return

    @staticmethod
    def _should_stop_after_handler(handler_name: str, result: IntegrationResult) -> bool:
        """Return True when a handler failure must abort remaining integration."""

        return handler_name == "compatibility" and not result.success

    def _format_handler_failure(
        self,
        vertical: Type["VerticalBase"],
        handler_name: str,
        error: Exception,
    ) -> str:
        """Format a namespace-aware handler failure message."""

        runtime_metadata = get_vertical_runtime_metadata(vertical)
        namespace = runtime_metadata["vertical_plugin_namespace"] or "default"
        return (
            f"Step handler '{handler_name}' failed for vertical "
            f"'{runtime_metadata['vertical_name']}' [namespace={namespace}]: {error}"
        )

    # Legacy method removed in Phase 1 refactoring
    # Step handlers now provide SOLID-compliant single responsibility implementation
    # See: victor/framework/step_handlers.py

    def _resolve_vertical(
        self, vertical: Union[Type["VerticalBase"], str]
    ) -> Optional[Type["VerticalBase"]]:
        """Resolve vertical from name or return class.

        Args:
            vertical: Vertical class or name string

        Returns:
            Vertical class or None if not found
        """
        if isinstance(vertical, str):
            from victor.core.verticals.base import VerticalRegistry

            resolved = VerticalRegistry.get(vertical)
            if resolved is None:
                # Try case-insensitive match
                for name in VerticalRegistry.list_names():
                    if name.lower() == vertical.lower():
                        resolved = VerticalRegistry.get(name)
                        break
            return ensure_runtime_vertical(resolved) if resolved is not None else None
        return ensure_runtime_vertical(vertical)

    def _create_context(
        self,
        orchestrator: Any,
        vertical: Type["VerticalBase"],
        result: IntegrationResult,
    ) -> Optional[MutableVerticalContextProtocol]:
        """Create the mutable vertical context for this vertical.

        Args:
            vertical: Vertical class
            result: Result to update

        Returns:
            Mutable vertical context or None on error.
        """
        try:
            from victor.framework.sdk_capability_registry import (
                resolve_capability_requirements,
            )

            config = vertical.get_config()
            definition = vertical.get_definition()
            context = self._build_context(name=vertical.name, config=config)
            capability_resolutions = resolve_capability_requirements(
                definition.capability_requirements,
                orchestrator=orchestrator,
                available_tools=definition.get_tool_names(),
            )
            if capability_resolutions:
                context.apply_capability_configs(
                    {
                        "sdk_capability_resolutions": [
                            resolution.to_dict() for resolution in capability_resolutions
                        ],
                    }
                )
                for resolution in capability_resolutions:
                    if resolution.available:
                        continue
                    prefix = "Optional" if resolution.optional else "Required"
                    result.add_warning(
                        f"{prefix} SDK capability '{resolution.capability_id}' is not currently satisfied: "
                        f"{resolution.reason or 'runtime binding unresolved'}"
                    )
            return context
        except Exception as e:
            result.add_error(f"Failed to create context: {e}")
            return None

    # Legacy tool methods removed in Phase 1 refactoring
    # Replaced by ToolStepHandler in victor/framework/step_handlers.py
    # Tiered tools handled by TieredConfigStepHandler

    # Legacy system prompt method removed in Phase 1 refactoring
    # Replaced by PromptStepHandler in victor/framework/step_handlers.py

    # Legacy stages method removed in Phase 1 refactoring
    # Replaced by ConfigStepHandler in victor/framework/step_handlers.py

    # Legacy extension methods removed in Phase 1 refactoring
    # Replaced by ExtensionsStepHandler in victor/framework/step_handlers.py
    # which coordinates:
    #   - MiddlewareStepHandler for middleware
    #   - SafetyStepHandler for safety extensions
    #   - PromptStepHandler for prompt contributors
    #   - ConfigStepHandler for mode config and tool deps

    # Legacy attach_context removed in Phase 1 refactoring
    # Replaced by ContextStepHandler in victor/framework/step_handlers.py

    # Legacy framework integration methods removed in Phase 1 refactoring
    # Replaced by FrameworkStepHandler in victor/framework/step_handlers.py
    # which handles workflows, RL config, and team specs


# =============================================================================
# Factory Functions
# =============================================================================


def create_integration_pipeline(
    strict: bool = False,
    use_step_handlers: bool = True,
    enable_cache: bool = True,
    cache_ttl: int = 3600,
    max_cache_entries: int = 256,
    cache_policy: Optional["VerticalIntegrationCachePolicy"] = None,
    enable_parallel: bool = True,
) -> VerticalIntegrationPipeline:
    """Create a vertical integration pipeline with feature flags (Phase 2).

    Args:
        strict: Whether to use strict mode
        use_step_handlers: If True, use step handlers (Phase 3.1)
        enable_cache: Enable configuration caching (Phase 1, default: True)
        cache_ttl: Cache time-to-live in seconds (default: 3600)
        max_cache_entries: Maximum cache entries before LRU eviction (default: 256)
        cache_policy: Optional custom cache policy implementation.
        enable_parallel: Enable parallel execution (Phase 2.2, default: False)

    Returns:
        Configured VerticalIntegrationPipeline

    Example:
        # Basic pipeline
        pipeline = create_integration_pipeline()

        # With parallel execution (Phase 2.2)
        pipeline = create_integration_pipeline(enable_parallel=True)

        result = await pipeline.apply_async(orchestrator, "coding")
    """
    return VerticalIntegrationPipeline(
        strict_mode=strict,
        use_step_handlers=use_step_handlers,
        enable_cache=enable_cache,
        cache_ttl=cache_ttl,
        max_cache_entries=max_cache_entries,
        cache_policy=cache_policy,
        parallel_enabled=enable_parallel,
    )


def create_integration_pipeline_with_handlers(
    strict: bool = False,
    custom_handlers: Optional[List[Any]] = None,
    enable_cache: bool = True,
    cache_ttl: int = 3600,
    max_cache_entries: int = 256,
    cache_policy: Optional["VerticalIntegrationCachePolicy"] = None,
    enable_parallel: bool = True,
) -> VerticalIntegrationPipeline:
    """Create a pipeline with custom step handlers.

    This factory function allows adding custom step handlers to the
    default set for specialized integration needs.

    Args:
        strict: Whether to use strict mode
        custom_handlers: Optional list of additional step handlers
        enable_cache: Enable configuration caching (default: True)
        cache_ttl: Cache time-to-live in seconds (default: 3600)
        max_cache_entries: Maximum cache entries before LRU eviction (default: 256)
        cache_policy: Optional custom cache policy implementation.
        enable_parallel: Enable parallel execution (Phase 2.2, default: False)

    Returns:
        Configured VerticalIntegrationPipeline with custom handlers

    Example:
        from victor.framework.step_handlers import BaseStepHandler

        class MyCustomHandler(BaseStepHandler):
            @property
            def name(self) -> str:
                return "my_custom"

            @property
            def order(self) -> int:
                return 55  # Between extensions and framework

            def _do_apply(self, orchestrator, vertical, context, result):
                # Custom integration logic
                pass

        pipeline = create_integration_pipeline_with_handlers(
            custom_handlers=[MyCustomHandler()],
            enable_parallel=True,
        )
    """
    try:
        from victor.framework.step_handlers import StepHandlerRegistry

        registry = StepHandlerRegistry.default()

        if custom_handlers:
            for handler in custom_handlers:
                registry.add_handler(handler)

        return VerticalIntegrationPipeline(
            strict_mode=strict,
            step_registry=registry,
            use_step_handlers=True,
            enable_cache=enable_cache,
            cache_ttl=cache_ttl,
            max_cache_entries=max_cache_entries,
            cache_policy=cache_policy,
            parallel_enabled=enable_parallel,
        )
    except ImportError:
        # Fallback to legacy if step handlers not available
        return VerticalIntegrationPipeline(
            strict_mode=strict,
            use_step_handlers=False,
            enable_cache=enable_cache,
            cache_ttl=cache_ttl,
            max_cache_entries=max_cache_entries,
            cache_policy=cache_policy,
        )


def apply_vertical(
    orchestrator: Any,
    vertical: Union[Type["VerticalBase"], str],
) -> IntegrationResult:
    """Apply a vertical to an orchestrator.

    Convenience function for one-shot vertical application.

    Args:
        orchestrator: Orchestrator instance
        vertical: Vertical class or name

    Returns:
        IntegrationResult
    """
    from victor.framework.vertical_service import apply_vertical_configuration

    return apply_vertical_configuration(orchestrator, vertical, source="framework")


__all__ = [
    # Core classes
    "IntegrationResult",
    "ExtensionLoadErrorInfo",
    "ValidationStatus",
    "VerticalIntegrationPipeline",
    "OrchestratorVerticalProtocol",
    # Extension handler registry (OCP compliance)
    "ExtensionHandlerInfo",
    "ExtensionHandlerRegistry",
    "get_extension_handler_registry",
    "register_extension_handler",
    # Factory functions
    "create_integration_pipeline",
    "create_integration_pipeline_with_handlers",
    "apply_vertical",
]

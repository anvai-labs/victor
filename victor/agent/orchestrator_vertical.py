"""Vertical protocol/storage for the orchestrator facade.

Moved verbatim from ``victor/agent/orchestrator.py`` (facade decomposition):
vertical-context sync, workspace switching, middleware/safety-pattern/team
storage, and coordination suggestions. Checkpoint methods stay in the
class body (service-delegation floor).
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from victor.agent.runtime.context import require_idle_turn
from victor.agent.vertical_context import VerticalContext

logger = logging.getLogger(__name__)


class OrchestratorVerticalMixin:
    """Moved verbatim from orchestrator.py; see module docstring."""

    def get_vertical_context(self) -> VerticalContext:
        """Get vertical context via explicit protocol getter."""
        return self._vertical_context

    @property
    def lsp(self) -> Optional[Any]:
        """Get the LSP capability for code intelligence.

        Returns:
            LSPCapability instance or None if not configured.
        """
        return getattr(self, "_lsp", None)

    def set_lsp(self, lsp_capability: Any) -> None:
        """Set the LSP capability (LSPServiceProtocol/LSPPoolProtocol).

        When a capability is provided, the LSP diagnostics middleware
        (FEP-0019) auto-activates on the tool middleware chain.
        """
        self._lsp = lsp_capability
        if lsp_capability is not None and getattr(self, "_middleware_chain", None):
            from victor.framework.lsp_middleware import register_lsp_on_chain

            register_lsp_on_chain(self, self._middleware_chain)
        logger.debug("LSP capability registered with orchestrator")

    def get_team_suggestions(
        self,
        task_type: str,
        complexity: str,
    ) -> Any:
        """Deprecated compatibility alias for ``get_coordination_suggestion()``.

        Returns coordination suggestions for a task while preserving the older
        team-centric method name used by legacy callers.
        """
        import warnings

        warnings.warn(
            "AgentOrchestrator.get_team_suggestions(...) is deprecated. "
            "Use AgentOrchestrator.get_coordination_suggestion(...) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_coordination_suggestion(task_type, complexity)

    def get_coordination_suggestion(
        self,
        task_type: str,
        complexity: str,
        *,
        mode: Optional[str] = None,
    ) -> Any:
        """Get canonical coordination suggestions for a task.

        Prefers the service-owned coordination runtime, which delegates to the
        shared framework recommendation engine.

        Args:
            task_type: Classified task type (e.g., "feature", "bugfix", "refactor")
            complexity: Complexity level (e.g., "low", "medium", "high", "extreme")
            mode: Optional explicit mode override. Defaults to runtime mode.

        Returns:
            CoordinationSuggestion with team and workflow recommendations
        """
        coordination_runtime = getattr(self, "_coordination_advisor_runtime", None)
        if coordination_runtime is None:
            factory = getattr(self, "_factory", None)
            if factory is not None and hasattr(factory, "create_coordination_advisor_runtime"):
                coordination_runtime = factory.create_coordination_advisor_runtime()
            else:
                from victor.agent.services.coordination_advisor_runtime import (
                    CoordinationAdvisorRuntime,
                )

                coordination_runtime = CoordinationAdvisorRuntime()

            self._coordination_advisor_runtime = coordination_runtime

        mode_controller = getattr(self, "mode_controller", None)
        current_mode = getattr(mode_controller, "current_mode", None)
        resolved_mode = mode or getattr(current_mode, "value", None) or "build"

        return coordination_runtime.suggest_for_task(
            runtime_subject=self,
            task_type=task_type,
            complexity=complexity,
            mode=resolved_mode,
        )

    def set_vertical_context(self, context: VerticalContext) -> None:
        """Set the vertical context (OrchestratorVerticalProtocol).

        This replaces direct assignment to _vertical_context and provides
        a proper API for framework integration.

        Args:
            context: VerticalContext to set
        """
        self._vertical_context = context

        # Sync coordination advisor with new vertical context (if already initialized)
        if getattr(self, "_coordination_advisor", None) is not None:
            self._coordination_advisor.set_vertical_context(context)
            logger.debug(
                "Coordination advisor synced with vertical context: %s",
                context.vertical_name,
            )

        # Sync tool selector with vertical context for vertical-specific tool selection (DIP)
        if hasattr(self, "tool_selector") and self.tool_selector is not None:
            self.tool_selector.set_vertical_context(context)
            if context.has_tool_selection_strategy:
                logger.debug(
                    f"Tool selector synced with vertical tool selection strategy: "
                    f"{context.vertical_name}"
                )

        # Sync middleware chain with vertical context for vertical-aware middleware (DIP)
        if hasattr(self, "_middleware_chain") and self._middleware_chain is not None:
            self._middleware_chain.set_vertical_context(context)
            logger.debug(f"Middleware chain synced with vertical context: {context.vertical_name}")

        logger.debug(f"Vertical context set: {context.vertical_name}")

    def set_tiered_tool_config(self, config: Any) -> None:
        """Set tiered tool configuration (Phase 1: Gap fix).

        Applies tiered tool config from vertical to:
        1. VerticalContext for storage
        2. ToolAccessController.VerticalLayer for access filtering

        Args:
            config: TieredToolConfig from the active vertical
        """
        # Store in vertical context
        if self._vertical_context is not None:
            self._vertical_context.apply_tiered_config(config)

        # Apply to tool access controller
        if self._tool_access_controller is not None:
            self._tool_access_controller.set_tiered_config(config)
            logger.debug("Tiered config applied to ToolAccessController")

        logger.debug("Tiered tool config set")

    def set_workspace(self, workspace_dir: Path) -> None:
        """Set the workspace directory for task execution.

        Updates the global project root and orchestrator's project context
        to work in the specified directory. This is essential for benchmark
        evaluations where each task operates in a different workspace.

        Args:
            workspace_dir: Path to the workspace directory

        Raises:
            RuntimeError: If a turn owns the session; no workspace state is changed.
        """
        require_idle_turn(self._session_accessor.stream_turn_lock)
        from victor.config.settings import set_project_root
        from victor.context.project_context import ProjectContext

        # Update global project root
        set_project_root(workspace_dir)
        logger.debug(f"Project root set to: {workspace_dir}")

        # Create new project context for this workspace
        self.project_context = ProjectContext(root_path=str(workspace_dir))
        self.project_context.load()

        # Set RL repo context for per-repo isolation
        try:
            from victor.agent.services.rl_runtime import get_rl_coordinator

            coordinator = get_rl_coordinator()
            coordinator.set_repo_context(workspace_dir.name)
            # Push the session's actual provider/model into GEPA so prompt
            # optimization uses the right provider (e.g., zai/glm-5.2 from
            # -p zai-coding), not the global settings default (ollama).
            coordinator.set_session_provider(
                getattr(self, "provider_name", None) or getattr(self.provider, "name", ""),
                self.model,
                base_url=getattr(self.provider, "base_url", "") or "",
            )
        except Exception as exc:
            logger.debug("RL coordinator repo context unavailable: %s", exc)

        # Rebuild system prompt with new workspace context (replaces old init.md)
        # Workspace switch is a session reset — unfreeze and re-sample GEPA sections
        if getattr(self, "_prompt_pipeline", None):
            self._prompt_pipeline.unfreeze()
        self._system_prompt_frozen = False

        self._system_prompt = self.build_system_prompt()

        if self._kv_optimization_enabled:
            self._system_prompt_frozen = True
        if self._cache_optimization_enabled:
            self._session_tools = None  # Re-lock tools for new workspace

        # Clear conversation history to remove old workspace context
        # This ensures the model doesn't see init.md from the previous project
        if hasattr(self, "reset_conversation"):
            self.reset_conversation()

    def _apply_vertical_tools(self, tools: Set[str]) -> None:
        """Apply enabled tools to vertical context and access controller.

        Internal helper called by set_enabled_tools. Separated to avoid
        duplication and maintain single method for protocol compliance.

        Args:
            tools: Set of tool names to enable
        """
        self._vertical_context.apply_enabled_tools(tools)
        logger.debug(f"Applied {len(tools)} tools to vertical context")

    def apply_vertical_middleware(self, middleware_list: List[Any]) -> None:
        """Apply middleware from vertical extensions.

        Called by FrameworkShim after orchestrator creation to inject
        vertical-specific middleware. This enables the middleware chain
        pattern for tool execution.

        Delegates to VerticalIntegrationAdapter for single-source implementation.

        Args:
            middleware_list: List of MiddlewareProtocol implementations.
        """
        self._vertical_integration_adapter.apply_middleware(middleware_list)

    def apply_vertical_safety_patterns(self, patterns: List[Any]) -> None:
        """Apply safety patterns from vertical extensions.

        Called by FrameworkShim to inject vertical-specific danger patterns
        into the safety checker.

        Delegates to VerticalIntegrationAdapter for single-source implementation.

        Args:
            patterns: List of SafetyPattern objects.
        """
        self._vertical_integration_adapter.apply_safety_patterns(patterns)

    def get_middleware_chain(self) -> Optional[Any]:
        """Get the middleware chain for tool execution.

        Returns:
            MiddlewareChain instance or None if not initialized.
        """
        return getattr(self, "_middleware_chain", None)

    def set_middleware_chain(self, chain: Any) -> None:
        """Store middleware chain via public runtime port."""
        self._middleware_chain = chain
        # If LSP was set before the chain existed, register its middleware now.
        if getattr(self, "_lsp", None) is not None:
            from victor.framework.lsp_middleware import register_lsp_on_chain

            register_lsp_on_chain(self, chain)

    def _set_vertical_middleware_storage(self, middleware: List[Any]) -> None:
        """Internal: Set vertical middleware storage.

        DIP Compliance: Provides controlled setter instead of direct
        private attribute access. Called by VerticalIntegrationAdapter.

        Args:
            middleware: List of middleware instances
        """
        self.set_middleware(middleware)

    def _set_middleware_chain_storage(self, chain: Any) -> None:
        """Internal: Set middleware chain storage.

        DIP Compliance: Provides controlled setter instead of direct
        private attribute access. Called by VerticalIntegrationAdapter.

        Args:
            chain: MiddlewareChain instance
        """
        self.set_middleware_chain(chain)

    def _set_safety_patterns_storage(self, patterns: List[Any]) -> None:
        """Internal: Set safety patterns storage.

        DIP Compliance: Provides controlled setter instead of direct
        private attribute access. Called by VerticalIntegrationAdapter.

        Args:
            patterns: List of safety pattern instances
        """
        self.set_safety_patterns(patterns)

    def set_middleware(self, middleware: List[Any]) -> None:
        """Store middleware configuration.

        Implements VerticalStorageProtocol.set_middleware().
        Provides a clean public interface for setting vertical middleware,
        replacing direct private attribute access.

        Args:
            middleware: List of MiddlewareProtocol implementations
        """
        self._vertical_middleware = middleware

    def get_middleware(self) -> List[Any]:
        """Retrieve middleware configuration.

        Implements VerticalStorageProtocol.get_middleware().
        Returns the list of middleware instances configured by vertical integration.

        Returns:
            List of middleware instances, or empty list if not set
        """
        return getattr(self, "_vertical_middleware", [])

    def set_safety_patterns(self, patterns: List[Any]) -> None:
        """Store safety patterns.

        Implements VerticalStorageProtocol.set_safety_patterns().
        Provides a clean public interface for setting vertical safety patterns,
        replacing direct private attribute access.

        Args:
            patterns: List of SafetyPattern instances from vertical extensions
        """
        self._vertical_safety_patterns = patterns

    def get_safety_patterns(self) -> List[Any]:
        """Retrieve safety patterns.

        Implements VerticalStorageProtocol.get_safety_patterns().
        Returns the list of safety patterns configured by vertical integration.

        Returns:
            List of safety pattern instances, or empty list if not set
        """
        return getattr(self, "_vertical_safety_patterns", [])

    def set_team_specs(self, specs: Dict[str, Any]) -> None:
        """Store team specifications.

        Implements VerticalStorageProtocol.set_team_specs().
        Provides a clean public interface for setting team specs,
        replacing direct private attribute access.

        Args:
            specs: Dictionary mapping team names to TeamSpec instances
        """
        self._team_specs = specs

    def get_team_specs(self) -> Dict[str, Any]:
        """Retrieve team specifications.

        Implements VerticalStorageProtocol.get_team_specs().
        Returns the dictionary of team specs configured by vertical integration.

        Returns:
            Dictionary of team specs, or empty dict if not set
        """
        return getattr(self, "_team_specs", {})

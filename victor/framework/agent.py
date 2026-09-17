"""Agent - Simplified entry point for Victor agents.

This is the main class users interact with. It wraps AgentOrchestrator
and provides a clean, simple API for common use cases.

Example:
    # Simple use case
    agent = await Agent.create(provider="anthropic")
    result = await agent.run("Write a hello world function")
    print(result.content)

    # Streaming
    async for event in agent.stream("Refactor this"):
        if event.type == EventType.CONTENT:
            print(event.content, end="")

    # Escape hatch to full power
    orchestrator = agent.get_orchestrator()
"""

from __future__ import annotations

import inspect
import logging
import sys
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    Union,
)

logger = logging.getLogger(__name__)

from victor.agent.config import (
    FrameworkCompatibleAgentConfig,
    UnifiedAgentConfig,
    normalize_agent_config,
)
from victor.core.errors import AgentError, CancellationError, ProviderError
from victor.framework.events import AgentExecutionEvent, EventType
from victor.framework.message_execution import execute_message, stream_message_events
from victor.framework.state import State, StateObserver
from victor.framework.task import TaskResult
from victor.framework.tools import ToolSet, ToolsInput
from victor.runtime.context import resolve_execution_context


def _build_verifier(mode: str, orchestrator: Any = None):
    """Construct a Verifier from a verify_mode string (FEP-0018/0019).

    Returns None for unknown/unsupported modes.
    """
    mode = (mode or "none").strip().lower()
    if mode == "none":
        return None
    try:
        from victor.framework.verifiers import LintVerifier, LocalTestVerifier, LSPVerifier

        if mode == "pytest" or mode == "test":
            return LocalTestVerifier()
        if mode == "lint" or mode == "ruff":
            return LintVerifier()
        if mode == "lsp":
            lsp_cap = getattr(orchestrator, "lsp", None) if orchestrator else None
            return LSPVerifier(lsp_capability=lsp_cap)
    except ImportError:
        pass
    return None


if TYPE_CHECKING:
    from victor.core.protocols import OrchestratorProtocol as AgentOrchestrator
    from victor.teams import TeamFormation
    from victor.framework.agent_components import AgentSession
    from victor.framework.teams import AgentTeam, TeamMemberSpec
    from victor.observability.integration import ObservabilityIntegration
    from victor.core.events import ObservabilityBus
    from victor.core.verticals.base import VerticalBase, VerticalConfig
    from victor.framework.session_config import SessionConfig


class Agent:
    """Simplified interface for creating and using Victor agents.

    The Agent class provides a "golden path" API that covers 90% of use cases
    with minimal configuration. For advanced use cases, access the underlying
    AgentOrchestrator via get_orchestrator().

    Attributes:
        state: Observable agent state (stage, tool usage, files)

    Example - Simple usage:
        agent = await Agent.create(provider="anthropic", model="claude-sonnet-4-20250514")
        result = await agent.run("Write a function to parse JSON")
        print(result.content)

    Example - With tools:
        agent = await Agent.create(
            provider="anthropic",
            tools=["filesystem", "git"]
        )
        result = await agent.run("Create a new feature branch and add a README")

    Example - Streaming with events:
        async for event in agent.stream("Refactor this file"):
            if event.type == EventType.THINKING:
                print(f"Thinking: {event.content}")
            elif event.type == EventType.TOOL_CALL:
                print(f"Tool: {event.tool_name}({event.arguments})")
            elif event.type == EventType.CONTENT:
                print(event.content, end="")

    Example - State observation:
        agent.on_state_change(lambda old, new: print(f"{old} -> {new}"))

    Example - Escape hatch to full power:
        orchestrator = agent.get_orchestrator()
        # Access all internal components
    """

    def __init__(
        self,
        orchestrator: "AgentOrchestrator",
        provider: str = "anthropic",
        model: Optional[str] = None,
        vertical: Optional[Type["VerticalBase"]] = None,
        vertical_config: Optional["VerticalConfig"] = None,
    ) -> None:
        """Initialize Agent with orchestrator. Use Agent.create() instead.

        Args:
            orchestrator: AgentOrchestrator instance
            provider: Provider name for reference
            model: Model name for reference
            vertical: Optional vertical class used to create this agent
            vertical_config: Optional vertical configuration applied

        Raises:
            ValueError: If orchestrator is not a valid AgentOrchestrator instance
        """
        # Validate that orchestrator is a proper AgentOrchestrator instance
        # This prevents misuse where someone tries to call Agent(...) directly
        orchestrator_type_name = type(orchestrator).__name__
        if orchestrator_type_name != "AgentOrchestrator":
            raise ValueError(
                f"Agent.__init__() requires an AgentOrchestrator instance, "
                f"but got {type(orchestrator).__module__}.{orchestrator_type_name}. "
                f"Use Agent.create() instead of calling Agent() directly.\n\n"
                f"Correct usage:\n"
                f"  agent = await Agent.create(provider='anthropic', model='claude-3-5-sonnet-20241022')\n"
                f"  agent = await Agent.create()  # Uses default provider/model\n"
                f"  agent = Agent.from_orchestrator(orchestrator)  # If you have an orchestrator\n\n"
                f"See victor/framework/agent.py for more details."
            )

        self._orchestrator = orchestrator
        self._provider = provider
        self._model = model
        self._vertical = vertical
        self._vertical_config = vertical_config
        self._context = resolve_execution_context(orchestrator)
        self._state = State(orchestrator)
        self._state_observers: List[StateObserver] = []
        # LSP capability (language intelligence)
        self._lsp: Optional[Any] = None

    @classmethod
    async def create(
        cls,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        *,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: ToolsInput = None,
        thinking: bool = False,
        airgapped: bool = False,
        profile: Optional[str] = None,
        workspace: Optional[str] = None,
        config: Optional[FrameworkCompatibleAgentConfig] = None,
        vertical: Optional[Type["VerticalBase"]] = None,
        enable_observability: bool = True,
        session_id: Optional[str] = None,
        session_config: Optional["SessionConfig"] = None,
    ) -> "Agent":
        """Create a new Agent instance.

        This is the primary way to create an Agent. For most use cases,
        you only need to specify the provider.

        Args:
            provider: Optional LLM provider name (anthropic, openai, ollama, google, etc.).
                If omitted, uses the active profile/default settings.
            model: Model identifier. If None, uses provider/profile default.
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            tools: Tool configuration - ToolSet, list of category names, or None
            thinking: Enable extended thinking mode (Claude only)
            airgapped: Disable network-dependent tools
            profile: Profile name from ~/.victor/profiles.yaml
            workspace: Working directory for file operations
            config: Advanced configuration. Accepts AgentConfig (deprecated) or
                UnifiedAgentConfig (preferred). Overrides individual options.
            vertical: Optional vertical class or name (e.g., 'coding', 'research').
                When provided, the vertical's configuration (tools, system_prompt,
                stages) is automatically applied.
            enable_observability: Auto-initialize ObservabilityIntegration for
                unified event handling. Defaults to True.
            session_id: Optional session ID for event correlation.
            session_config: Optional SessionConfig with CLI/runtime overrides.
                This is the PREFERRED way to pass CLI flags - immutable and traceable.

        Returns:
            Configured Agent instance

        Raises:
            ProviderError: If provider initialization fails
            AgentError: If configuration is invalid

        Example:
            # Simple
            agent = await Agent.create()

            # With provider
            agent = await Agent.create(provider="openai", model="gpt-4-turbo")

            # With tools
            agent = await Agent.create(tools=ToolSet.coding())

            # With SessionConfig (preferred for CLI/runtime overrides)
            from victor.framework.session_config import SessionConfig
            config = SessionConfig.from_cli_flags(tool_budget=50, enable_smart_routing=True)
            agent = await Agent.create(session_config=config)

            # With vertical (domain-specific assistant)
            agent = await Agent.create(vertical="coding")
        """
        from victor.config.settings import load_settings
        from victor.framework.agent_factory import AgentFactory, InitializationError
        from victor.framework.session_config import SessionConfig

        # Load settings (or use config overrides). fresh=True: mutated below
        # (workspace/config/session overlays) — must not disturb the shared
        # process-wide snapshot.
        settings = load_settings(fresh=True)
        if workspace:
            settings.working_directory = workspace

        config = normalize_agent_config(config)

        # Overlay config settings after normalizing legacy AgentConfig inputs.
        if config is not None:
            for key, value in config.to_settings_dict().items():
                if hasattr(settings, key):
                    setattr(settings, key, value)

        # Apply SessionConfig overrides (preferred over direct settings mutation)
        profile_overrides: Dict[str, Any] = {}
        if session_config is not None:
            session_config.apply_to_settings(settings)
            # FEP-0002: apply RL-driven budget calibration after SessionConfig
            # overrides land. An explicit ``--tool-budget`` override (now
            # applied by apply_to_settings above) wins absolutely — the seam
            # short-circuits to identity-return when explicit_override is set,
            # so calibration can never re-tighten a user-set budget. Default
            # OFF via ``tool_budget_calibration_enabled`` -> zero behavior
            # change for sessions that don't opt in.
            from victor.framework.rl.budget_calibration_seam import (
                apply_budget_calibration,
            )

            explicit_override = session_config.tool_budget is not None
            calibrated_tools = apply_budget_calibration(
                settings, explicit_override=explicit_override
            )
            if calibrated_tools is not settings.tools:
                object.__setattr__(settings, "tools", calibrated_tools)
            profile = session_config.agent_profile or profile
            provider_override = getattr(session_config, "provider_override", None)
            if provider_override is not None:
                provider = provider_override.provider or provider
                model = provider_override.model or model
                profile_overrides = provider_override.to_profile_overrides()

            # Configure the session-scoped shell safety policy (FEP-0013). A
            # ``legacy``/unset profile is a no-op (existing inline allowlist);
            # a non-legacy profile installs a damage-scoped policy the shell
            # tool consults via get_shell_safety_policy().
            try:
                from victor.security.shell_safety_policy import configure_from_session

                configure_from_session(session_config, settings)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Shell safety policy configuration skipped: %s", exc)

        # Resolve a string vertical name into the actual VerticalBase class
        # before any attribute access. The public docstring/example accepts
        # vertical="coding"; without this step the get_config() call below
        # crashed with 'str' object has no attribute 'get_config'. The factory
        # also runs its own resolution, but Agent.create needs the class
        # itself for the backward-compat return value below — keep both
        # paths fed the same resolved object.
        if isinstance(vertical, str):
            try:
                from victor.core.verticals.base import VerticalRegistry

                resolved = VerticalRegistry.get(vertical)
                if resolved is not None:
                    vertical = resolved
                else:
                    logger.debug("Vertical '%s' not in registry; ignoring", vertical)
                    vertical = None
            except Exception as exc:
                logger.debug("Vertical resolution for '%s' failed: %s", vertical, exc)
                vertical = None

        # Extract vertical config for backward compat return value
        vertical_config: Optional["VerticalConfig"] = None
        if vertical is not None and hasattr(vertical, "get_config"):
            vertical_config = vertical.get_config()

        try:
            # Unified creation via AgentFactory — same path as CLI/API
            factory = AgentFactory(
                settings=settings,
                profile=profile or "default",
                provider=provider,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                vertical=vertical,
                thinking=thinking,
                session_id=session_id,
                enable_observability=enable_observability,
                profile_overrides=profile_overrides,
            )
            orchestrator = await factory.create()

            # FEP-0018: wire the verifier from session_config to the
            # turn_executor → AgenticLoop. Framework-owned path; the UI
            # never reaches into the orchestrator directly.
            _verify_mode = getattr(session_config, "verify_mode", "none")
            if _verify_mode and _verify_mode != "none":
                _te = getattr(orchestrator, "turn_executor", None)
                if _te is not None:
                    _verifier = _build_verifier(_verify_mode, orchestrator)
                    if _verifier is not None:
                        _te._verifier = _verifier
                        _te._max_verify_retries = 2

            # FEP-0019: propagate the LSP diagnostics feedback mode to the
            # orchestrator so the auto-activated middleware honors it.
            _lsp_feedback = getattr(session_config, "lsp_feedback", "errors")
            if _lsp_feedback:
                from victor.framework.lsp_middleware import set_lsp_feedback_mode

                set_lsp_feedback_mode(orchestrator, _lsp_feedback)

            # FEP-0019 Phase 3: enable proactive LSP context injection. The flag
            # is read off the turn_executor by AgenticLoop.__init__ — same path
            # as the verifier above; no loop-construction-site changes needed.
            if getattr(session_config, "lsp_perception", False):
                _te = getattr(orchestrator, "turn_executor", None)
                if _te is not None:
                    _te._lsp_context_enabled = True

            resolved_provider = (
                provider
                or getattr(orchestrator, "provider_name", None)
                or getattr(getattr(orchestrator, "provider", None), "name", None)
                or "unknown"
            )
            resolved_model = model or getattr(orchestrator, "model", None)
            return cls(
                orchestrator,
                provider=resolved_provider,
                model=resolved_model,
                vertical=vertical,
                vertical_config=vertical_config,
            )
        except InitializationError as e:
            if e.stage == "credentials":
                raise ProviderError(str(e), provider=provider) from e
            raise AgentError(str(e)) from e
        except Exception as e:
            if "provider" in str(e).lower() or "api" in str(e).lower():
                raise ProviderError(str(e), provider=provider) from e
            raise AgentError(str(e)) from e

    @classmethod
    def from_orchestrator(cls, orchestrator: "AgentOrchestrator") -> "Agent":
        """Create an Agent wrapper around an existing AgentOrchestrator.

        This is the escape hatch for users who have already configured
        an AgentOrchestrator and want to use the simplified API.

        Args:
            orchestrator: Existing AgentOrchestrator instance

        Returns:
            Agent wrapping the orchestrator
        """
        provider = getattr(orchestrator.provider, "name", "unknown")
        model = getattr(orchestrator, "model", None)
        return cls(orchestrator, provider=provider, model=model)

    @classmethod
    async def create_team(
        cls,
        name: str,
        goal: str,
        members: List["TeamMemberSpec"],
        *,
        formation: "TeamFormation" = None,
        provider: str = "anthropic",
        model: Optional[str] = None,
        total_tool_budget: int = 100,
        max_iterations: int = 50,
        timeout_seconds: int = 600,
        shared_context: Optional[Dict[str, Any]] = None,
        config: Optional[FrameworkCompatibleAgentConfig] = None,
    ) -> "AgentTeam":
        """Create a multi-agent team.

        This creates a team of agents that coordinate to achieve a shared goal.
        Teams support different formation patterns for different workflows.

        Args:
            name: Human-readable team name
            goal: Overall team objective
            members: List of TeamMemberSpec defining team composition
            formation: How agents coordinate (SEQUENTIAL, PARALLEL, HIERARCHICAL, PIPELINE)
            provider: LLM provider for all team members
            model: Model identifier for all team members
            total_tool_budget: Total tool calls across all members
            max_iterations: Maximum total iterations
            timeout_seconds: Maximum execution time
            shared_context: Initial context shared with all members
            config: Advanced configuration for the orchestrator

        Returns:
            AgentTeam ready for execution

        Example - Sequential research team:
            from victor.framework.teams import TeamMemberSpec, TeamFormation

            team = await Agent.create_team(
                name="Code Analysis",
                goal="Analyze the authentication module",
                members=[
                    TeamMemberSpec(role="researcher", goal="Find auth code"),
                    TeamMemberSpec(role="analyzer", goal="Analyze patterns"),
                    TeamMemberSpec(role="reviewer", goal="Summarize findings"),
                ],
                formation=TeamFormation.SEQUENTIAL,
            )
            result = await team.run()

        Example - Pipeline for feature implementation:
            team = await Agent.create_team(
                name="Feature Implementation",
                goal="Implement user authentication",
                members=[
                    TeamMemberSpec(role="researcher", goal="Find auth patterns"),
                    TeamMemberSpec(role="planner", goal="Design implementation"),
                    TeamMemberSpec(role="executor", goal="Write the code"),
                    TeamMemberSpec(role="reviewer", goal="Review and fix"),
                ],
                formation=TeamFormation.PIPELINE,
            )

            async for event in team.stream():
                print(f"{event.type}: {event.message}")

        Example - Hierarchical with supervisor:
            from victor.framework.teams import TeamAgentCategory

            team = await Agent.create_team(
                name="Project Team",
                goal="Build a REST API",
                members=[
                    TeamMemberSpec(
                        role="planner",
                        goal="Coordinate team",
                        agent_category=TeamAgentCategory.SUPERVISOR,
                    ),
                    TeamMemberSpec(role="researcher", goal="Research API patterns"),
                    TeamMemberSpec(role="executor", goal="Implement endpoints"),
                    TeamMemberSpec(role="reviewer", goal="Test and review"),
                ],
                formation=TeamFormation.HIERARCHICAL,
            )
        """
        from victor.teams import TeamFormation as TF
        from victor.framework.teams import AgentTeam

        # Default to SEQUENTIAL if not specified
        if formation is None:
            formation = TF.SEQUENTIAL

        # Create an agent to get an orchestrator
        agent = await cls.create(
            provider=provider,
            model=model,
            config=config,
        )

        # Create team using the agent's orchestrator
        return await AgentTeam.create(
            orchestrator=agent.get_orchestrator(),
            name=name,
            goal=goal,
            members=members,
            formation=formation,
            total_tool_budget=total_tool_budget,
            max_iterations=max_iterations,
            timeout_seconds=timeout_seconds,
            shared_context=shared_context,
        )

    # =========================================================================
    # Primary Methods
    # =========================================================================

    @property
    def execution_context(self) -> Any:
        """Return the explicit runtime execution context when available."""
        runtime_context = getattr(self, "_context", None)
        if runtime_context is not None:
            return runtime_context

        runtime_context = resolve_execution_context(self._orchestrator)
        if runtime_context is not None:
            self._context = runtime_context
        return runtime_context

    def _notify_state_observers_if_changed(self, previous_stage: Any) -> Any:
        """Notify observers when the observable stage changes."""
        new_stage = self._state.stage
        if new_stage == previous_stage:
            return previous_stage

        old_state = State(self._orchestrator)
        old_state._orchestrator = self._orchestrator
        for observer in self._state_observers:
            try:
                observer(old_state, self._state)
            except Exception:
                logger.warning("State observer error", exc_info=True)
        return new_stage

    async def run(
        self,
        prompt: str,
        *,
        context: Optional[Dict[str, Any]] = None,
        runtime_context_overrides: Optional[Dict[str, Any]] = None,
    ) -> TaskResult:
        """Run a task and return the complete result.

        This is the simplest way to use the agent - provide a prompt and
        get back the complete response including any tool results.

        Args:
            prompt: What the agent should do
            context: Optional context dict (files, variables, etc.)
            runtime_context_overrides: Optional scoped runtime hints for this turn, such
                as named prompt overlays.

        Returns:
            TaskResult with content, tool_calls, and metadata

        Example:
            result = await agent.run("Explain the authentication flow")
            print(result.content)

            # With context
            result = await agent.run(
                "Fix the bug in this code",
                context={"file": "auth.py", "error": "IndexError"}
            )
        """
        execution_kwargs: Dict[str, Any] = {
            "orchestrator": self._orchestrator,
            "execution_context": self.execution_context,
            "user_message": prompt,
            "context": context,
        }
        if runtime_context_overrides is not None:
            execution_kwargs["runtime_context_overrides"] = runtime_context_overrides

        return await execute_message(**execution_kwargs)

    async def stream(
        self,
        prompt: str,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[AgentExecutionEvent]:
        """Stream events as the agent processes a task.

        This provides real-time visibility into the agent's reasoning
        and actions. Events include thinking, tool calls, content, and errors.

        Args:
            prompt: What the agent should do
            context: Optional context dict

        Yields:
            AgentExecutionEvent objects representing agent actions

        Example:
            async for event in agent.stream("Analyze this codebase"):
                if event.type == EventType.THINKING:
                    print(f"Thinking: {event.content[:50]}...")
                elif event.type == EventType.TOOL_CALL:
                    print(f"Calling {event.tool_name}")
                elif event.type == EventType.TOOL_RESULT:
                    print(f"Result: {event.result[:100]}...")
                elif event.type == EventType.CONTENT:
                    print(event.content, end="", flush=True)
        """
        # Track state for observers
        old_stage = self._state.stage

        async for event in stream_message_events(
            orchestrator=self._orchestrator,
            execution_context=self.execution_context,
            user_message=prompt,
            context=context,
        ):
            old_stage = self._notify_state_observers_if_changed(old_stage)
            yield event

    def chat(self, prompt: str) -> "ChatSession":
        """Start an interactive chat session.

        Returns a ChatSession that maintains conversation context
        across multiple turns.

        Args:
            prompt: Initial message

        Returns:
            ChatSession for multi-turn conversation

        Example:
            session = agent.chat("Let's refactor the auth module")
            response = await session.send("First, show me the current code")
            response = await session.send("Now extract the validation logic")
        """
        return ChatSession(self, prompt)

    def create_session(
        self,
        initial_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> "AgentSession":
        """Create the canonical multi-turn session implementation.

        Args:
            initial_prompt: Optional initial prompt for the first turn.
            **kwargs: Additional ``AgentSession`` constructor options.

        Returns:
            AgentSession with shared runtime/state behavior.
        """
        from victor.framework.agent_components import AgentSession

        return AgentSession(self, initial_prompt, **kwargs)

    async def run_oneshot(
        self,
        prompt: str,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> TaskResult:
        """Execute a single-turn task without maintaining conversation state.

        This is a convenience method that wraps run() for one-shot tasks
        where you don't need to maintain conversation context across calls.

        Args:
            prompt: What the agent should do
            context: Optional context dict (files, variables, etc.)

        Returns:
            TaskResult with content, tool_calls, and metadata

        Example:
            # Single API call to get a response
            result = await agent.run_oneshot("Explain quantum computing")
            print(result.content)

            # With context
            result = await agent.run_oneshot(
                "What's wrong with this code?",
                context={"file": "auth.py", "error": "NullReferenceException"}
            )
        """
        return await self.run(prompt, context=context)

    async def run_interactive(
        self,
        initial_prompt: str,
    ) -> "ChatSession":
        """Start an interactive multi-turn conversation session.

        This is a convenience method that creates a ChatSession with an
        initial prompt, allowing you to send multiple messages while
        maintaining conversation context.

        Args:
            initial_prompt: The first message to start the conversation

        Returns:
            ChatSession for multi-turn conversation

        Example:
            session = await agent.run_interactive("Help me refactor this code")

            # Continue the conversation
            response1 = await session.send("What should we extract first?")
            response2 = await session.send("Now apply that change")

            # Or stream responses
            async for event in session.stream("Show me the full diff"):
                if event.type == EventType.CONTENT:
                    print(event.content, end="")
        """
        return self.chat(initial_prompt)

    # =========================================================================
    # State and Observation
    # =========================================================================

    @property
    def state(self) -> State:
        """Get current agent state.

        Returns:
            State object with stage, tool_calls_used, files_observed, etc.
        """
        return self._state

    @property
    def vertical(self) -> Optional[Type["VerticalBase"]]:
        """Get the vertical class used to create this agent.

        Returns:
            Vertical class or None if not created from a vertical.
        """
        return self._vertical

    @property
    def vertical_config(self) -> Optional["VerticalConfig"]:
        """Get the vertical configuration applied to this agent.

        Returns:
            VerticalConfig or None if not created from a vertical.
        """
        return self._vertical_config

    @property
    def vertical_name(self) -> Optional[str]:
        """Get the name of the vertical used to create this agent.

        Returns:
            Vertical name string or None.
        """
        if self._vertical:
            return self._vertical.name
        return None

    def on_state_change(
        self,
        callback: StateObserver,
    ) -> Callable[[], None]:
        """Register a callback for state changes.

        Args:
            callback: Function called with (old_state, new_state)

        Returns:
            Unsubscribe function

        Example:
            def log_state(old, new):
                print(f"State: {old.stage} -> {new.stage}")

            unsubscribe = agent.on_state_change(log_state)
            # Later: unsubscribe()
        """
        self._state_observers.append(callback)

        def unsubscribe() -> None:
            """Unsubscribe the previously registered callback from event notifications.

            This function removes the callback from the event notification system.
            After calling unsubscribe(), the callback will no longer receive events.

            Example:
                >>> def my_handler(event):
                ...     print(f"Event: {event.type}")
                >>>
                >>> unsubscribe = agent.subscribe_to_events("TOOL", my_handler)
                >>> # Later, stop receiving events:
                >>> unsubscribe()

            Note:
                If the callback is not registered, this function does nothing (no error raised).
                Safe to call multiple times.
            """
            if callback in self._state_observers:
                self._state_observers.remove(callback)

        return unsubscribe

    # =========================================================================
    # Configuration
    # =========================================================================

    async def switch_model(
        self,
        provider: str,
        model: str,
    ) -> None:
        """Switch to a different model.

        Args:
            provider: New provider name
            model: New model identifier
        """
        if hasattr(self._orchestrator, "provider_manager"):
            await self._orchestrator.provider_manager.switch_provider(provider, model)
        self._provider = provider
        self._model = model

    def set_tools(self, tools: Union[ToolSet, List[str]]) -> None:
        """Update available tools.

        Args:
            tools: ToolSet or list of category names
        """
        from victor.framework._internal import configure_tools

        configure_tools(self._orchestrator, tools)

    # =========================================================================
    # Escape Hatch
    # =========================================================================

    def get_orchestrator(self) -> "AgentOrchestrator":
        """Get the underlying AgentOrchestrator for advanced usage.

        This provides access to all internal components when the
        simplified API is insufficient.

        Returns:
            AgentOrchestrator instance

        Example:
            orchestrator = agent.get_orchestrator()

            # Access decomposed components
            controller = orchestrator.conversation_controller
            pipeline = orchestrator.tool_pipeline

            # Access internal state
            metrics = orchestrator.streaming_controller.get_session_history()
        """
        return self._orchestrator

    # =========================================================================
    # Runtime Configuration (CLI/Runtime Override Support)
    # =========================================================================

    def set_tool_budget(self, budget: int, *, user_override: bool = False) -> None:
        """Set the maximum number of tool calls allowed.

        Args:
            budget: Maximum tool calls allowed
            user_override: Whether this is a user-specified override (takes precedence)

        Example:
            agent = await Agent.create()
            agent.set_tool_budget(50)
        """
        if hasattr(self._orchestrator, "unified_tracker"):
            self._orchestrator.unified_tracker.set_tool_budget(budget, user_override=user_override)

    def set_max_iterations(self, max_iterations: int, *, user_override: bool = False) -> None:
        """Set the maximum number of agentic loop iterations.

        Args:
            max_iterations: Maximum iterations allowed
            user_override: Whether this is a user-specified override (takes precedence)

        Example:
            agent = await Agent.create()
            agent.set_max_iterations(20)
        """
        if hasattr(self._orchestrator, "unified_tracker"):
            self._orchestrator.unified_tracker.set_max_iterations(
                max_iterations, user_override=user_override
            )

    def supports_streaming(self) -> bool:
        """Check if the current provider supports streaming responses.

        Returns:
            True if streaming is supported, False otherwise

        Example:
            agent = await Agent.create()
            if agent.supports_streaming():
                async for event in agent.stream("Hello"):
                    print(event.content)
        """
        if hasattr(self._orchestrator, "provider"):
            return getattr(self._orchestrator.provider, "supports_streaming", lambda: True)()
        return True

    def start_embedding_preload(self) -> None:
        """Warm embedding-dependent runtime state when supported.

        This is primarily used by chat surfaces to front-load semantic search
        initialization without exposing orchestrator internals directly.
        """
        if hasattr(self._orchestrator, "start_embedding_preload"):
            self._orchestrator.start_embedding_preload()

    def get_session_metrics(self) -> Dict[str, Any]:
        """Return session-level runtime metrics when available."""
        if hasattr(self._orchestrator, "get_session_metrics"):
            metrics = self._orchestrator.get_session_metrics()
            if isinstance(metrics, dict):
                return metrics
        return {}

    # =========================================================================
    # Observability
    # =========================================================================

    @property
    def event_bus(self) -> Optional["ObservabilityBus"]:
        """Get the ObservabilityBus for subscribing to agent events.

        The ObservabilityBus provides access to all agent events including:
        - Tool execution (start/end)
        - State transitions
        - Model requests/responses
        - Errors

        Returns:
            ObservabilityBus instance, or None if observability is disabled

        Example:
            def on_tool_event(event):
                print(f"Tool: {event.topic} - {event.data}")

            # Subscribe to all tool events
            agent.event_bus.backend.subscribe("tool.*", on_tool_event)
        """
        observability = getattr(self._orchestrator, "observability", None)
        if observability:
            return observability.event_bus
        return None

    @property
    def observability(self) -> Optional["ObservabilityIntegration"]:
        """Get the ObservabilityIntegration for advanced event handling.

        Returns:
            ObservabilityIntegration instance, or None if disabled
        """
        return getattr(self._orchestrator, "observability", None)

    @property
    def provider_name(self) -> Optional[str]:
        """Active provider name (e.g. 'zai', 'ollama') for the current session."""
        orc = self._orchestrator
        return getattr(orc, "provider_name", None) or getattr(
            getattr(orc, "provider", None), "name", None
        )

    @property
    def model(self) -> Optional[str]:
        """Active model name (e.g. 'glm-5.1') for the current session."""
        return getattr(self._orchestrator, "model", None)

    @property
    def provider_base_url(self) -> Optional[str]:
        """Active provider base_url; non-empty when a non-default endpoint is in use."""
        provider = getattr(self._orchestrator, "provider", None)
        return getattr(provider, "base_url", None)

    @property
    def lsp(self) -> Optional[Any]:
        """Get the LSP capability for code intelligence.

        Returns:
            LSPCapability instance or None
        """
        return self._lsp

    def set_lsp(self, lsp_capability: Any) -> None:
        """Set the LSP capability for language intelligence.

        Enables features like hover information, go-to-definition,
        completions, and diagnostics for code operations.

        Args:
            lsp_capability: LSPCapability instance

        Example:
            from victor.framework.capabilities import LSPCapability

            agent.set_lsp(LSPCapability())
        """
        self._lsp = lsp_capability

    def subscribe_to_events(
        self,
        category: str,
        handler: Callable[[Any], None],
    ) -> Optional[Callable[[], None]]:
        """Subscribe to events of a specific category.

        Convenience method for subscribing to EventBus events without
        directly importing observability types.

        Args:
            category: Event category, wildcard alias, or topic pattern.
                Examples: "TOOL", "security_scan", "ALL", "tool.*"
            handler: Callback function receiving VictorEvent

        Returns:
            Unsubscribe function, or None if observability is disabled

        Example:
            def log_tools(event):
                print(f"Tool called: {event.name}")

            unsubscribe = agent.subscribe_to_events("TOOL", log_tools)
            # Later: unsubscribe()
        """
        event_bus = self.event_bus
        if not event_bus:
            return None

        from victor.observability.event_registry import (
            resolve_subscription_topic_pattern,
        )

        topic_pattern = resolve_subscription_topic_pattern(category)

        # Subscribe using canonical event system
        return event_bus.backend.subscribe(topic_pattern, handler)

    # =========================================================================
    # Workflow and Team Execution
    # =========================================================================

    async def run_workflow(
        self,
        workflow_name: str,
        context: Optional[Dict[str, Any]] = None,
        *,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run a workflow by name from the vertical's workflow provider.

        This executes a multi-step workflow defined by the vertical,
        coordinating multiple agents through a DAG of operations.

        Args:
            workflow_name: Name of the workflow to run (e.g., "feature_implementation")
            context: Initial context data for the workflow
            timeout: Overall timeout in seconds (None = no limit)

        Returns:
            Dict with workflow result including outputs, success status, and metrics

        Raises:
            AgentError: If no vertical is configured or workflow not found

        Example:
            # Run a feature implementation workflow
            result = await agent.run_workflow(
                "feature_implementation",
                context={"feature": "Add user authentication"}
            )
            print(result["success"])
            print(result["outputs"])

            # Run an EDA workflow for data analysis
            result = await agent.run_workflow(
                "eda_workflow",
                context={"data_file": "sales.csv"}
            )
        """
        # Check if vertical is configured
        if not self._vertical:
            raise AgentError("No vertical configured. Create agent with vertical= parameter.")

        # Get workflow provider from vertical
        workflow_provider = self._vertical.get_workflow_provider()
        if not workflow_provider:
            raise AgentError(f"Vertical '{self._vertical.name}' does not provide workflows.")

        # Use canonical API: run_compiled_workflow (uses UnifiedWorkflowCompiler internally)
        result = await workflow_provider.run_compiled_workflow(
            workflow_name, context or {}, timeout=timeout
        )

        return result

    async def run_team(
        self,
        team_name: str,
        goal: str,
        *,
        context: Optional[Dict[str, Any]] = None,
        timeout_seconds: int = 600,
    ) -> Dict[str, Any]:
        """Run a pre-configured team from the vertical's team specs.

        This creates and executes a multi-agent team defined by the vertical,
        using the team's formation pattern and member specifications.

        Args:
            team_name: Name of the team spec (e.g., "feature_team", "bug_fix_team")
            goal: Specific goal for this team execution
            context: Initial shared context for team members
            timeout_seconds: Maximum execution time

        Returns:
            Dict with team result including final output and member contributions

        Raises:
            AgentError: If no vertical is configured or team not found

        Example:
            # Run a feature implementation team
            result = await agent.run_team(
                "feature_team",
                goal="Implement user authentication with JWT",
                context={"target_dir": "src/auth/"}
            )

            # Run a bug fix team
            result = await agent.run_team(
                "bug_fix_team",
                goal="Fix the login timeout issue",
                context={"error_log": "TimeoutError at auth.py:45"}
            )
        """
        from victor.framework.team_runtime import (
            resolve_vertical_team_catalog,
            run_named_team,
        )

        # Check if vertical is configured
        if not self._vertical:
            raise AgentError("No vertical configured. Create agent with vertical= parameter.")

        team_catalog = resolve_vertical_team_catalog(self._vertical)
        if not team_catalog.supported:
            raise AgentError(f"Vertical '{self._vertical.name}' does not support teams.")
        if not team_catalog.provider_available:
            raise AgentError(f"Vertical '{self._vertical.name}' does not provide team specs.")
        if not team_catalog.has_team_specs:
            raise AgentError(f"Vertical '{self._vertical.name}' has no team specs defined.")

        team_spec = team_catalog.get(team_name)
        if not team_spec:
            available = team_catalog.list_names()
            raise AgentError(f"Team '{team_name}' not found. " f"Available: {', '.join(available)}")

        team_execution = await run_named_team(
            self._orchestrator,
            team_name=team_name,
            goal=goal,
            context=context,
            timeout_seconds=timeout_seconds,
        )
        if team_execution is None:
            raise AgentError(f"Team '{team_name}' could not be resolved for execution.")
        resolved_team, result = team_execution

        payload: Dict[str, Any] = {}
        to_dict = getattr(result, "to_dict", None)
        if callable(to_dict):
            serialized = to_dict()
            if isinstance(serialized, dict):
                payload = dict(serialized)
        if not payload:
            payload = {
                "success": bool(getattr(result, "success", True)),
                "final_output": (
                    getattr(result, "final_output", None)
                    if getattr(result, "final_output", None) is not None
                    else str(result)
                ),
            }
        payload.update(
            {
                "team_name": resolved_team.team_name,
                "team_display_name": resolved_team.display_name,
                "goal": goal,
                "recommendation_source": resolved_team.recommendation_source,
            }
        )
        return payload

    def get_available_workflows(self) -> List[str]:
        """Get list of available workflow names from the vertical.

        Returns:
            List of workflow names, or empty list if no vertical/workflows

        Example:
            workflows = agent.get_available_workflows()
            print(workflows)  # ['feature_implementation', 'bug_fix', 'code_review']
        """
        from victor.framework.team_runtime import resolve_vertical_workflow_catalog

        workflow_catalog = resolve_vertical_workflow_catalog(self._vertical)
        if not workflow_catalog.supported or not workflow_catalog.provider_available:
            return []
        return workflow_catalog.list_names()

    def get_available_teams(self) -> List[str]:
        """Get list of available team names from the vertical.

        Returns:
            List of team names, or empty list if no vertical/teams

        Example:
            teams = agent.get_available_teams()
            print(teams)  # ['feature_team', 'bug_fix_team', 'review_team']
        """
        from victor.framework.team_runtime import resolve_vertical_team_catalog

        team_catalog = resolve_vertical_team_catalog(self._vertical)
        if not team_catalog.supported or not team_catalog.provider_available:
            return []
        return team_catalog.list_names()

    def get_coordination_suggestion(
        self,
        task_type: str,
        complexity: str,
        *,
        mode: Optional[str] = None,
    ) -> Any:
        """Get shared framework coordination recommendations for a task.

        Args:
            task_type: Classified task type
            complexity: Complexity level string
            mode: Optional mode override. Defaults to the runtime's current mode.

        Returns:
            CoordinationSuggestion with team and workflow recommendations.
        """
        from victor.framework.coordination_runtime import (
            get_runtime_coordination_suggestion,
        )

        return get_runtime_coordination_suggestion(
            runtime_subject=self._orchestrator,
            task_type=task_type,
            complexity=complexity,
            mode=mode,
        )

    async def get_coordination_transitions(
        self,
        task_type: str,
        complexity: Optional[str] = None,
        *,
        mode: Optional[str] = None,
    ) -> Any:
        """Get state-passed coordination transitions for a task.

        This is the framework-facing state-passed companion to
        ``get_coordination_suggestion()``. It returns the raw coordinator result
        so callers can inspect transitions, confidence, and metadata without
        mutating orchestrator state directly.
        """
        from victor.agent.coordinators.state_context import create_snapshot

        orchestration_facade = getattr(self._orchestrator, "orchestration_facade", None)
        if orchestration_facade is None:
            orchestration_facade = getattr(self._orchestrator, "_orchestration_facade", None)

        coordination_state_passed = (
            getattr(orchestration_facade, "coordination_state_passed", None)
            if orchestration_facade is not None
            else None
        )
        if coordination_state_passed is None:
            raise AgentError("Coordination state-passed surface is not available")

        mode_controller = getattr(self._orchestrator, "mode_controller", None)
        current_mode = getattr(mode_controller, "current_mode", None)
        resolved_mode = mode or getattr(current_mode, "value", None) or "build"

        snapshot = create_snapshot(self._orchestrator)
        return await coordination_state_passed.suggest(
            snapshot,
            task_type=task_type,
            complexity=complexity,
            mode=resolved_mode,
        )

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def reset(self) -> None:
        """Reset conversation history and state."""
        self._orchestrator.reset_conversation()
        self._state.reset()

    async def warm_up(self) -> None:
        """Prime the KV cache for faster first responses.

        For local providers (Ollama, LMStudio) with KV prefix caching, the first
        API call is always cold. This sends a minimal 1-token request to prime
        the KV cache with the system prompt, making subsequent calls faster.

        No-op for cloud providers or when KV optimization is disabled.
        """
        if hasattr(self._orchestrator, "warm_up_kv_cache"):
            await self._orchestrator.warm_up_kv_cache()

    async def graceful_shutdown(self) -> Dict[str, bool]:
        """Perform graceful shutdown of all agent components.

        Delegates to orchestrator's graceful_shutdown method if available.
        Flushes analytics, stops health monitoring, and cleans up resources.
        Call this before application exit for a clean shutdown.

        Returns:
            Dictionary with shutdown status for each component.
            Returns empty dict if orchestrator doesn't support graceful_shutdown.
        """
        if self._orchestrator is None:
            return {}

        if hasattr(self._orchestrator, "graceful_shutdown"):
            return await self._orchestrator.graceful_shutdown()

        # Fallback to regular close if graceful_shutdown not available
        await self.close()
        return {}

    async def shutdown(self) -> None:
        """Shutdown the agent and clean up resources.

        This is an alias for close() for compatibility with code that
        expects a shutdown method.
        """
        await self.close()

    async def close(self) -> None:
        """Clean up resources."""
        # Clean up orchestrator
        cleanup = None
        if hasattr(self._orchestrator, "close"):
            cleanup = self._orchestrator.close
        elif hasattr(self._orchestrator, "shutdown"):
            cleanup = self._orchestrator.shutdown

        if cleanup is not None:
            result = cleanup()
            if inspect.isawaitable(result):
                await result
        self._orchestrator = None  # Prevent __del__ warning after proper close

    async def __aenter__(self) -> "Agent":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    def __del__(self) -> None:
        orchestrator = getattr(self, "_orchestrator", None)
        if orchestrator is None:
            return

        # Interpreter shutdown can clear import machinery before __del__ runs.
        # Skip the warning in that state instead of printing a misleading
        # exception that obscures the real command result.
        if getattr(sys, "meta_path", None) is None:
            return

        try:
            warnings.warn(
                "Agent was not closed. Use 'async with' or call "
                "'await agent.close()' to release resources.",
                ResourceWarning,
                stacklevel=2,
            )
        except Exception:
            logger.debug("Skipping Agent.__del__ warning during interpreter shutdown")

    def __repr__(self) -> str:
        return f"Agent(provider={self._provider}, model={self._model}, state={self._state})"


class ChatSession:
    """Multi-turn chat session with maintained context.

    Use this for conversations that span multiple exchanges.
    The session maintains conversation history automatically.

    Note: ChatSession is a simplified interface that delegates to AgentSession
    for full functionality. For advanced features like lifecycle hooks, metrics,
    and pause/resume support, use AgentSession directly.

    Example:
        session = agent.chat("Let's analyze the codebase")
        r1 = await session.send("What's the main entry point?")
        r2 = await session.send("Now explain that function")
        print(session.history)

    Advanced usage (via AgentSession):
        from victor.framework.agent_components import AgentSession, SessionLifecycleHooks

        hooks = SessionLifecycleHooks(
            on_start=lambda s: print(f"Session started"),
            on_close=lambda s, m: print(f"Completed {m.total_turns} turns"),
        )
        session = AgentSession(agent, "Let's analyze", hooks=hooks)
    """

    def __init__(self, agent: Agent, initial_prompt: str) -> None:
        """Initialize chat session.

        Args:
            agent: Agent instance
            initial_prompt: First message in the conversation
        """
        self._delegate = agent.create_session(initial_prompt)

    async def send(self, message: str) -> TaskResult:
        """Send a message in the conversation.

        Args:
            message: User message

        Returns:
            TaskResult with response
        """
        return await self._delegate.send(message)

    async def stream(self, message: str) -> AsyncIterator[AgentExecutionEvent]:
        """Stream a response in the conversation.

        Args:
            message: User message

        Yields:
            AgentExecutionEvent objects
        """
        async for event in self._delegate.stream(message):
            yield event

    @property
    def turn_count(self) -> int:
        """Get number of turns in the conversation."""
        return self._delegate.turn_count

    @property
    def history(self) -> List[Dict[str, str]]:
        """Get conversation history.

        Returns:
            List of message dictionaries with role and content
        """
        return self._delegate.history

    async def __aenter__(self) -> "ChatSession":
        return self

    async def __aexit__(self, *args: Any) -> None:
        # ChatSession is lightweight; Agent.close() handles heavy resources
        pass

    # Expose the underlying AgentSession for advanced usage
    def get_session(self) -> "AgentSession":
        """Get the underlying AgentSession for advanced features.

        Use this to access additional capabilities like:
        - Lifecycle hooks
        - Metrics tracking
        - Pause/resume/close
        - Session state management

        Returns:
            The underlying AgentSession instance
        """
        from victor.framework.agent_components import AgentSession

        return self._delegate

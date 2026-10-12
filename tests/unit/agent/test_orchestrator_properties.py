"""Tests for orchestrator_properties module.

Verifies that install_properties() correctly installs property
definitions onto the AgentOrchestrator class.
"""

import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock


class TestOrchestratorPropertyInstallation:
    """Verify that install_properties populated AgentOrchestrator with the expected properties."""

    def test_install_properties_adds_simple_accessors(self):
        """Simple properties should be installed as real descriptors."""
        from victor.agent.orchestrator import AgentOrchestrator

        # These properties should exist on the class (installed by install_properties)
        simple_props = [
            "conversation_controller",
            "tool_pipeline",
            "streaming_controller",
            "streaming_handler",
            "task_analyzer",
            "provider_manager",
            "context_compactor",
            "tool_output_formatter",
            "usage_analytics",
            "sequence_tracker",
            "recovery_coordinator",
            "chunk_generator",
            "tool_planner",
            "task_coordinator",
            "skill_matcher",
            "code_correction_middleware",
            "checkpoint_manager",
            "vertical_context",
        ]
        for prop_name in simple_props:
            assert hasattr(
                AgentOrchestrator, prop_name
            ), f"Property '{prop_name}' not found on AgentOrchestrator"
            assert isinstance(
                getattr(AgentOrchestrator, prop_name), property
            ), f"'{prop_name}' should be a property descriptor"

    def test_install_properties_adds_lazy_coordinators(self):
        """Lazy coordinator properties should also be installed."""
        from victor.agent.orchestrator import AgentOrchestrator

        lazy_props = [
            "protocol_adapter",
            "turn_executor",
            "runtime_intelligence_integration",
            "subagent_orchestrator",
            "orchestration_facade",
            "coordination_advisor",
            "coordination",
        ]
        for prop_name in lazy_props:
            assert hasattr(
                AgentOrchestrator, prop_name
            ), f"Lazy property '{prop_name}' not found on AgentOrchestrator"
            assert isinstance(
                getattr(AgentOrchestrator, prop_name), property
            ), f"'{prop_name}' should be a property descriptor"

    def test_specialized_chat_coordinators_are_not_installed_on_orchestrator(self):
        """Deprecated specialized chat shims now live on the orchestration facade only."""
        from victor.agent.orchestrator import AgentOrchestrator

        removed_props = [
            "sync_chat_coordinator",
            "streaming_chat_coordinator",
            "unified_chat_coordinator",
            "intelligent_integration",
        ]

        for prop_name in removed_props:
            assert hasattr(AgentOrchestrator, prop_name) is False

    def test_properties_module_has_install_function(self):
        """The module should export install_properties."""
        from victor.agent.orchestrator_properties import install_properties

        assert callable(install_properties)

    def test_properties_module_has_registry(self):
        """The module should have a property registry."""
        from victor.agent.orchestrator_properties import _PROPERTY_REGISTRY

        assert isinstance(_PROPERTY_REGISTRY, dict)
        assert len(_PROPERTY_REGISTRY) >= 20  # At least 20 properties

    def test_skill_matcher_property_returns_shared_matcher(self):
        """skill_matcher should expose the shared framework matcher surface."""
        from victor.agent.orchestrator import AgentOrchestrator

        orchestrator = object.__new__(AgentOrchestrator)
        matcher = MagicMock(name="skill_matcher")
        orchestrator._skill_matcher = matcher

        assert orchestrator.skill_matcher is matcher

    def test_coordination_advisor_property_returns_lazy_framework_surface(self):
        """coordination_advisor should expose the normalized lazy coordination surface."""
        from victor.agent.orchestrator import AgentOrchestrator

        orchestrator = object.__new__(AgentOrchestrator)
        advisor = MagicMock(name="coordination_advisor")
        orchestrator._coordination_advisor = advisor

        assert orchestrator.coordination_advisor is advisor
        assert orchestrator.coordination is advisor

    def test_orchestration_facade_property_resolves_lazy_surface(self):
        """orchestration_facade should resolve lazy runtime proxies to concrete facades."""
        from victor.agent.orchestrator import AgentOrchestrator

        orchestrator = object.__new__(AgentOrchestrator)
        facade = MagicMock(name="orchestration_facade")
        orchestrator._orchestration_facade = SimpleNamespace(get_instance=lambda: facade)

        assert orchestrator.orchestration_facade is facade
        assert orchestrator._orchestration_facade is facade

    def test_removed_compatibility_aliases_are_not_installed_on_orchestrator(self):
        """Deprecated chat/session/tool coordinator aliases now live off-orchestrator."""
        from victor.agent.orchestrator import AgentOrchestrator

        removed_aliases = [
            "_tool_coordinator",
            "_chat_coordinator",
            "_session_coordinator",
            "_mode_workflow_team_coordinator",
        ]

        for prop_name in removed_aliases:
            assert hasattr(AgentOrchestrator, prop_name) is False

    @pytest.mark.parametrize(
        ("property_name", "backing_attr", "facade_attr", "facade_value_attr"),
        [
            (
                "conversation_controller",
                "_conversation_controller",
                "_chat_facade",
                "conversation_controller",
            ),
            ("tool_pipeline", "_tool_pipeline", "_tool_facade", "tool_pipeline"),
            (
                "streaming_handler",
                "_streaming_handler",
                "_orchestration_facade",
                "streaming_handler",
            ),
            (
                "provider_manager",
                "_provider_manager",
                "_provider_facade",
                "provider_manager",
            ),
            (
                "vertical_context",
                "_vertical_context",
                "_orchestration_facade",
                "vertical_context",
            ),
        ],
    )
    def test_simple_properties_prefer_direct_canonical_attributes(
        self,
        property_name,
        backing_attr,
        facade_attr,
        facade_value_attr,
    ):
        """Compatibility facades must not become the behavior owner for simple properties."""
        from victor.agent.orchestrator import AgentOrchestrator

        orchestrator = object.__new__(AgentOrchestrator)
        direct_value = MagicMock(name=f"{property_name}_direct")
        facade_value = MagicMock(name=f"{property_name}_facade")
        setattr(orchestrator, backing_attr, direct_value)
        setattr(
            orchestrator,
            facade_attr,
            SimpleNamespace(**{facade_value_attr: facade_value}),
        )

        assert getattr(orchestrator, property_name) is direct_value

    def test_session_ledger_property_prefers_direct_canonical_attribute(self):
        """session_ledger should read from the canonical backing attribute, not the facade copy."""
        from victor.agent.orchestrator import AgentOrchestrator

        orchestrator = object.__new__(AgentOrchestrator)
        direct_value = MagicMock(name="ledger_direct")
        facade_value = MagicMock(name="ledger_facade")
        orchestrator._session_ledger = direct_value
        orchestrator._session_facade = SimpleNamespace(session_ledger=facade_value)

        assert orchestrator.session_ledger is direct_value

    def test_session_ledger_setter_updates_canonical_backing_attribute(self):
        """session_ledger setter should update the direct canonical slot."""
        from victor.agent.orchestrator import AgentOrchestrator

        orchestrator = object.__new__(AgentOrchestrator)
        orchestrator._session_ledger = MagicMock(name="old_ledger")
        orchestrator._session_facade = SimpleNamespace(
            session_ledger=MagicMock(name="facade_ledger")
        )
        new_value = MagicMock(name="new_ledger")

        orchestrator.session_ledger = new_value

        assert orchestrator._session_ledger is new_value


@pytest.mark.asyncio
async def test_shared_runtime_identity_survives_recovery_and_reaches_both_consumers():
    """Exercise facade wiring, not copies of its resolution policy in test mocks."""
    from unittest.mock import AsyncMock
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.agent.runtime.context import RuntimeIdentity

    orchestrator = object.__new__(AgentOrchestrator)
    orchestrator._runtime_identity = RuntimeIdentity(orchestrator)
    orchestrator.active_session_id = "old-active"
    orchestrator._memory_session_id = "old-memory"
    orchestrator._turn_executor = None
    orchestrator._context_lifecycle_service = SimpleNamespace(
        after_agent_turn=AsyncMock(return_value={})
    )
    orchestrator.get_messages = lambda: []
    orchestrator._session_service = SimpleNamespace(recover_session=MagicMock(return_value=True))
    tools = orchestrator._get_tool_execution_runtime()
    turn = orchestrator.turn_executor
    tools._save_conversation_checkpoint = AsyncMock(return_value=None)
    tools._create_filesystem_checkpoint = AsyncMock(return_value=None)
    tools._record_execution_checkpoint = MagicMock()
    before = orchestrator._runtime_identity.resolve()
    assert before.session_id == "old-active"
    assert orchestrator._get_session_runtime().recover_session("recovered")
    current = orchestrator._runtime_identity.resolve()
    assert current is tools._runtime_context() is turn._runtime_context()
    assert current.session_id == "recovered"
    assert before.session_id == "old-active"
    checkpoint = await tools._maybe_create_execution_checkpoint(
        orchestrator.protocol_adapter, [{"id": "write", "name": "write", "arguments": {}}]
    )
    assert checkpoint.session_id == current.session_id
    await turn._check_lifecycle_context_compaction("hello")
    assert orchestrator._context_lifecycle_service.after_agent_turn.await_args.args[0] is current
    # A later explicit resume must be visible to already-created consumers too.
    orchestrator.active_session_id = "loaded"
    assert tools._runtime_context().session_id == turn._runtime_context().session_id == "loaded"

    child = current.derive_child(
        agent_id="member-agent", display_name="Member", role="researcher", member_id="m"
    )
    orchestrator._runtime_identity.bind_member(child)
    with pytest.raises(ValueError, match="member session"):
        orchestrator._get_session_runtime().recover_session("another-member")
    orchestrator._session_service.recover_session.assert_called_once_with("recovered")
    assert orchestrator.active_session_id == child.session_id
    assert tools._runtime_context() is turn._runtime_context() is child

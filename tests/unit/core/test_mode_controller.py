"""AgentModeController: operational-mode state consulted by tools and the runtime."""

from __future__ import annotations

from unittest import mock

import pytest

from victor.core import mode_controller as mc
from victor.core.mode_controller import MODE_CONFIGS, AgentMode, AgentModeController


@pytest.fixture
def controller():
    return AgentModeController()


class TestSwitching:
    def test_switch_records_history_notifies_and_emits(self, controller):
        seen = []
        controller.register_callback(lambda old, new: seen.append((old, new)))
        hooks = mock.Mock()
        with mock.patch("victor.core.rl.hooks.get_rl_hooks", return_value=hooks):
            assert controller.switch_mode(AgentMode.PLAN) is True
        assert controller.current_mode is AgentMode.PLAN
        assert seen == [(AgentMode.BUILD, AgentMode.PLAN)]
        event = hooks.emit.call_args.args[0]
        assert (event.mode_from, event.mode_to) == ("build", "plan")
        assert event.metadata == {"history_length": 2}

    def test_switch_to_current_mode_is_a_noop(self, controller):
        callback = mock.Mock()
        controller.register_callback(callback)
        assert controller.switch_mode(AgentMode.BUILD) is True
        callback.assert_not_called()

    def test_failing_callback_does_not_block_the_switch(self, controller):
        controller.register_callback(mock.Mock(side_effect=RuntimeError("boom")))
        assert controller.switch_mode(AgentMode.EXPLORE) is True
        assert controller.current_mode is AgentMode.EXPLORE

    def test_missing_or_failing_rl_hooks_are_tolerated(self, controller):
        with mock.patch("victor.core.rl.hooks.get_rl_hooks", return_value=None):
            assert controller.switch_mode(AgentMode.REVIEW)
        with mock.patch("victor.core.rl.hooks.get_rl_hooks", side_effect=RuntimeError("x")):
            assert controller.switch_mode(AgentMode.PLAN)

    def test_previous_mode_walks_history(self, controller):
        assert controller.previous_mode() is None
        controller.switch_mode(AgentMode.PLAN)
        controller.switch_mode(AgentMode.REVIEW)
        assert controller.previous_mode() is AgentMode.PLAN
        assert controller.current_mode is AgentMode.PLAN


class TestToolPolicy:
    def test_disallowed_beats_allow_all(self, controller):
        config = mock.Mock(disallowed_tools={"write"}, allow_all_tools=True, allowed_tools=set())
        with mock.patch.object(AgentModeController, "config", config):
            assert controller.is_tool_allowed("write") is False
            assert controller.is_tool_allowed("read") is True

    def test_allow_list_applies_without_allow_all(self, controller):
        config = mock.Mock(disallowed_tools=set(), allow_all_tools=False, allowed_tools={"read"})
        with mock.patch.object(AgentModeController, "config", config):
            assert controller.is_tool_allowed("read") is True
            assert controller.is_tool_allowed("shell") is False

    def test_priority_defaults_to_neutral(self, controller):
        config = mock.Mock(tool_priorities={"read": 1.5})
        with mock.patch.object(AgentModeController, "config", config):
            assert controller.get_tool_priority("read") == 1.5
            assert controller.get_tool_priority("other") == 1.0


class TestIntrospection:
    def test_status_and_prompt_reflect_current_config(self, controller):
        controller.switch_mode(AgentMode.PLAN)
        config = MODE_CONFIGS[AgentMode.PLAN]
        assert controller.get_status() == {
            "mode": "plan",
            "name": config.name,
            "description": config.description,
            "write_confirmation_required": config.require_write_confirmation,
            "verbose_planning": config.verbose_planning,
        }
        assert controller.get_system_prompt_addition() == config.system_prompt_addition

    def test_mode_list_marks_exactly_the_current_mode(self, controller):
        modes = controller.get_mode_list()
        assert {m["mode"] for m in modes} == {m.value for m in AgentMode}
        assert [m["mode"] for m in modes if m["current"]] == ["build"]


class TestAccessor:
    def test_container_registration_wins(self):
        registered = AgentModeController(AgentMode.REVIEW)
        container = mock.Mock(is_registered=mock.Mock(return_value=True))
        container.get.return_value = registered
        with mock.patch("victor.core.container.get_container", return_value=container):
            assert mc.get_mode_controller() is registered

    def test_legacy_singleton_fallback_and_set_reset(self):
        container = mock.Mock(is_registered=mock.Mock(return_value=False))
        mc.reset_mode_controller()
        try:
            with mock.patch("victor.core.container.get_container", return_value=container):
                first = mc.get_mode_controller()
                assert mc.get_mode_controller() is first
                custom = AgentModeController(AgentMode.EXPLORE)
                mc.set_mode_controller(custom)
                assert mc.get_mode_controller() is custom
        finally:
            mc.reset_mode_controller()


class TestExtendedAgentModeSingleDefinition:
    """RL learning and adaptive control share one five-mode state machine."""

    def test_both_layers_alias_the_core_definition(self):
        from victor.agent.adaptive_mode_controller import AdaptiveAgentMode, AdaptiveModeController
        from victor.core.mode_controller import EXTENDED_MODE_TRANSITIONS, ExtendedAgentMode
        from victor.framework.rl.learners.mode_transition import ModeTransitionLearner, RLAgentMode

        assert RLAgentMode is ExtendedAgentMode
        assert AdaptiveAgentMode is ExtendedAgentMode
        assert ModeTransitionLearner.VALID_TRANSITIONS is EXTENDED_MODE_TRANSITIONS
        assert AdaptiveModeController.VALID_TRANSITIONS is EXTENDED_MODE_TRANSITIONS

    def test_state_machine_is_unchanged(self):
        from victor.core.mode_controller import EXTENDED_MODE_TRANSITIONS, ExtendedAgentMode as M

        assert [m.value for m in M] == ["explore", "plan", "build", "review", "complete"]
        assert EXTENDED_MODE_TRANSITIONS == {
            M.EXPLORE: [M.PLAN, M.BUILD, M.COMPLETE],
            M.PLAN: [M.BUILD, M.EXPLORE, M.COMPLETE],
            M.BUILD: [M.REVIEW, M.EXPLORE, M.COMPLETE],
            M.REVIEW: [M.BUILD, M.COMPLETE],
            M.COMPLETE: [],
        }

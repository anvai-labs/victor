"""TDD tests for AgenticLoopConfig typed dataclass — Wave B.

Verifies: typed dataclass replaces Dict[str,Any] config; defaults match legacy
behavior; from_dict() backward-compat factory; AgenticLoop accepts both forms.
"""

from __future__ import annotations

import pytest


class TestAgenticLoopConfigDataclass:
    def test_config_class_importable(self):
        from victor.framework.agentic_loop import AgenticLoopConfig

        cfg = AgenticLoopConfig()
        assert cfg is not None

    def test_config_defaults_match_legacy_dict_defaults(self):
        """All defaults must equal the hardcoded fallbacks in the original .get() calls."""
        from victor.framework.agentic_loop import AgenticLoopConfig

        cfg = AgenticLoopConfig()
        assert cfg.disable_enhanced_completion is False
        assert cfg.enable_requirement_validation is True
        assert cfg.enable_completion_scoring is True
        assert cfg.enable_context_keywords is True
        assert cfg.enable_calibrated_completion is None
        assert cfg.enable_planning_gate is True
        assert cfg.enable_paradigm_router is True
        assert cfg.enable_topology_routing is True

    def test_config_from_dict_roundtrips_all_eight_keys(self):
        from victor.framework.agentic_loop import AgenticLoopConfig

        d = {
            "disable_enhanced_completion": True,
            "enable_requirement_validation": False,
            "enable_completion_scoring": False,
            "enable_context_keywords": False,
            "enable_calibrated_completion": True,
            "enable_planning_gate": False,
            "enable_paradigm_router": False,
            "enable_topology_routing": False,
        }
        cfg = AgenticLoopConfig.from_dict(d)
        assert cfg.disable_enhanced_completion is True
        assert cfg.enable_requirement_validation is False
        assert cfg.enable_completion_scoring is False
        assert cfg.enable_context_keywords is False
        assert cfg.enable_calibrated_completion is True
        assert cfg.enable_planning_gate is False
        assert cfg.enable_paradigm_router is False
        assert cfg.enable_topology_routing is False

    def test_config_from_dict_ignores_unknown_keys(self):
        from victor.framework.agentic_loop import AgenticLoopConfig

        cfg = AgenticLoopConfig.from_dict({"unknown_key": "value", "enable_planning_gate": False})
        assert cfg.enable_planning_gate is False
        # No KeyError from the unknown key
        assert cfg.disable_enhanced_completion is False  # default preserved

    def test_config_from_empty_dict_uses_defaults(self):
        from victor.framework.agentic_loop import AgenticLoopConfig

        cfg = AgenticLoopConfig.from_dict({})
        assert cfg.enable_planning_gate is True
        assert cfg.enable_paradigm_router is True

    @pytest.mark.parametrize("raw", ["standard", "stategraph", "AUTO", None, "unknown"])
    def test_executor_config_validates_at_construction(self, raw):
        from victor.framework.agentic_loop import AgenticLoopConfig
        from victor.config.agentic_loop_settings import AgenticLoopExecutor

        if raw is None or raw == "unknown":
            with pytest.raises(ValueError, match="executor"):
                AgenticLoopConfig.from_dict({"executor": raw})
        else:
            config = AgenticLoopConfig.from_dict({"executor": raw})
            assert config.executor is AgenticLoopExecutor(raw.lower())

    def test_config_fields_are_typed(self):
        from victor.framework.agentic_loop import AgenticLoopConfig
        import dataclasses

        fields = {f.name: f for f in dataclasses.fields(AgenticLoopConfig)}
        assert "disable_enhanced_completion" in fields
        assert "enable_topology_routing" in fields
        assert len(fields) >= 8


class TestAgenticLoopAcceptsTypedConfig:
    def test_agentic_loop_init_accepts_typed_config_object(self):
        """AgenticLoop.__init__ must accept AgenticLoopConfig as the config argument."""
        from victor.framework.agentic_loop import AgenticLoop, AgenticLoopConfig

        cfg = AgenticLoopConfig(disable_enhanced_completion=True)
        loop = AgenticLoop(
            orchestrator=None,
            turn_executor=None,
            config=cfg,
        )
        # The loop should have stored the config (or derived its attributes from it)
        assert loop.enhanced_completion_evaluator is None  # disabled via config

    def test_agentic_loop_accepts_dict_config_backward_compat(self):
        """AgenticLoop.__init__ must still accept a plain dict (backward compat)."""
        from victor.framework.agentic_loop import AgenticLoop

        loop = AgenticLoop(
            orchestrator=None,
            turn_executor=None,
            config={"disable_enhanced_completion": True},
        )
        assert loop.enhanced_completion_evaluator is None

    def test_agentic_loop_none_config_uses_defaults(self):
        """config=None must use all defaults (same as before)."""
        from victor.framework.agentic_loop import AgenticLoop

        loop = AgenticLoop(orchestrator=None, turn_executor=None, config=None)
        # With defaults, enhanced completion evaluator is enabled
        assert loop.enhanced_completion_evaluator is not None

    def test_config_typed_object_enables_disable_flag(self):
        from victor.framework.agentic_loop import AgenticLoop, AgenticLoopConfig

        cfg = AgenticLoopConfig(enable_planning_gate=False)
        loop = AgenticLoop(orchestrator=None, turn_executor=None, config=cfg)
        # planning_gate.enabled should reflect the config
        assert loop.planning_gate.enabled is False

    def test_config_stored_as_agenticloopconfig_type(self):
        """After construction, loop.config must be an AgenticLoopConfig instance."""
        from victor.framework.agentic_loop import AgenticLoop, AgenticLoopConfig

        loop = AgenticLoop(orchestrator=None, turn_executor=None, config=None)
        assert isinstance(loop.config, AgenticLoopConfig), (
            "AgenticLoop.config must be an AgenticLoopConfig instance after construction, "
            "not a raw dict. Convert dict inputs via AgenticLoopConfig.from_dict()."
        )


def test_executor_addition_preserves_legacy_positional_config():
    from victor.framework.agentic_loop import AgenticLoopConfig
    from victor.config.agentic_loop_settings import AgenticLoopExecutor

    config = AgenticLoopConfig(True)
    assert config.disable_enhanced_completion is True
    assert config.executor is AgenticLoopExecutor.AUTO

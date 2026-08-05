# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Regression guard: SkillRegistry is populated in the modern agent path.

Before this, ``SkillRegistry.from_all_sources()`` was only ever called by the
deprecated ``FrameworkShim`` and the skill CLI. The modern ``AgentFactory`` →
``SkillMatcher`` path passed the *orchestrator* as the matcher's registry, but
the orchestrator has no ``list_all()`` → ``SkillMatcher.initialize`` raised
``AttributeError``, was swallowed by a bare ``except``, and auto-skill-selection
was silently broken.

These tests pin: (1) a shared ``get_skill_registry()`` accessor, (2)
``FrameworkStepHandler.apply_skills`` populating it from the integrated vertical,
and (3) ``AgentFactory._initialize_skill_matcher`` passing the populated registry
to the matcher. Dropping any of these fails CI.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, List
from unittest.mock import MagicMock

import pytest

from victor.core.verticals.base import VerticalBase
from victor_contracts.skills import SkillDefinition


def _skill(name: str) -> SkillDefinition:
    return SkillDefinition(
        name=name,
        description=f"Skill: {name}",
        category="coding",
        prompt_fragment=f"Prompt for {name}.",
        required_tools=["read_file"],
        tags=frozenset(),
    )


class VerticalWithSkills(VerticalBase):
    """Vertical that declares skills via get_skills()."""

    name = "test_skills_vertical"

    @classmethod
    def get_tools(cls) -> List[str]:
        return ["read"]

    @classmethod
    def get_system_prompt(cls) -> str:
        return "Test"

    @classmethod
    def get_skills(cls) -> List[SkillDefinition]:
        return [_skill("alpha_skill"), _skill("beta_skill")]


class _PlainVertical(VerticalBase):
    name = "plain_vertical"

    @classmethod
    def get_tools(cls):
        return ["read"]

    @classmethod
    def get_system_prompt(cls):
        return "Test"


def _ctx_and_result():
    context = MagicMock()
    result = MagicMock()
    result.add_info = MagicMock()
    result.add_warning = MagicMock()
    return context, result


@pytest.fixture(autouse=True)
def _reset_skill_registry():
    """Isolate tests from the module-level skill registry singleton."""
    from victor.framework.skills import reset_skill_registry

    reset_skill_registry()
    yield
    reset_skill_registry()


class TestSkillRegistryAccessor:
    """get_skill_registry() is a shared accessor; reset clears it."""

    def test_singleton_returns_same_instance(self):
        from victor.framework.skills import get_skill_registry

        assert get_skill_registry() is get_skill_registry()

    def test_reset_clears_the_singleton(self):
        from victor.framework.skills import get_skill_registry, reset_skill_registry

        registry = get_skill_registry()
        registry.register(_skill("temp_skill"))
        assert any(s.name == "temp_skill" for s in registry.list_all())

        reset_skill_registry()
        assert get_skill_registry().list_all() == []
        assert get_skill_registry() is not registry


class TestApplySkillsWiring:
    """FrameworkStepHandler.apply_skills loads a vertical's skills into the registry."""

    def test_apply_skills_registers_vertical_skills(self):
        from victor.framework.skills import get_skill_registry
        from victor.framework.step_handlers import FrameworkStepHandler

        handler = FrameworkStepHandler()
        context, result = _ctx_and_result()
        orch = SimpleNamespace()

        handler.apply_skills(orch, VerticalWithSkills, context, result)

        names = {s.name for s in get_skill_registry().list_all()}
        assert {"alpha_skill", "beta_skill"} <= names

    def test_apply_skills_noop_for_vertical_without_skills(self):
        from victor.framework.skills import get_skill_registry
        from victor.framework.step_handlers import FrameworkStepHandler

        handler = FrameworkStepHandler()
        context, result = _ctx_and_result()

        handler.apply_skills(SimpleNamespace(), _PlainVertical, context, result)

        assert get_skill_registry().list_all() == []

    def test_do_apply_invokes_apply_skills(self):
        """The full _do_apply pipeline must populate skills (regression guard)."""
        from victor.framework.skills import get_skill_registry
        from victor.framework.step_handlers import FrameworkStepHandler

        handler = FrameworkStepHandler()
        context, result = _ctx_and_result()

        handler._do_apply(SimpleNamespace(), VerticalWithSkills, context, result)

        names = {s.name for s in get_skill_registry().list_all()}
        assert "alpha_skill" in names


class TestInitializeSkillMatcherUsesRegistry:
    """AgentFactory._initialize_skill_matcher passes the populated registry."""

    @pytest.mark.asyncio
    async def test_matcher_initialized_with_populated_registry(self, monkeypatch):
        from victor.framework import skill_matcher as matcher_mod
        from victor.framework.agent_factory import AgentFactory
        from victor.framework.skills import get_skill_registry
        from victor.framework.step_handlers import FrameworkStepHandler

        # Populate the singleton via the step handler (as the integration pipeline does).
        handler = FrameworkStepHandler()
        handler.apply_skills(SimpleNamespace(), VerticalWithSkills, *_ctx_and_result())
        assert any(s.name == "alpha_skill" for s in get_skill_registry().list_all())

        # Patch SkillMatcher so initialize() records the registry it received
        # (avoids the real embedding cost).
        captured: dict = {}

        class _RecordingMatcher:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

            async def initialize(self, registry: Any) -> None:
                captured["registry"] = registry

        monkeypatch.setattr(matcher_mod, "SkillMatcher", _RecordingMatcher)

        factory = AgentFactory.__new__(AgentFactory)
        factory._orchestrator = SimpleNamespace()

        await factory._initialize_skill_matcher()

        assert "registry" in captured, "matcher.initialize was not called"
        # The matcher received a registry that exposes list_all() AND contains
        # the vertical's skills (i.e. NOT the bare orchestrator, which has none).
        registry = captured["registry"]
        assert hasattr(registry, "list_all")
        names = {s.name for s in registry.list_all()}
        assert "alpha_skill" in names
        # And the matcher was stored on the orchestrator.
        assert getattr(factory._orchestrator, "_skill_matcher", None) is not None

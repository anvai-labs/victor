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

"""Tests for victor.framework.defaults package."""

import re

import pytest

from dataclasses import dataclass, field
from enum import Enum
from typing import List

from victor.framework.defaults import (
    ANALYSIS_STAGE_TOOLS,
    DEFAULT_ACTIVE_LEARNERS,
    DEFAULT_BUDGET_SCALING,
    DEFAULT_COMPLEXITY_MAP,
    DEFAULT_MODES,
    DEFAULT_PATIENCE_MAP,
    DEFAULT_TASK_BUDGETS,
    EXECUTION_STAGE_TOOLS,
    MODIFICATION_STAGE_TOOLS,
    BaseRLConfig,
    DefaultSafetyExtension,
    ModeConfigRegistry,
    ModeDefinition,
    PersonaHelpers,
    RegistryBasedModeConfigProvider,
    StageDefinition,
    create_complexity_map,
    get_default_stages,
    get_stage_tools_for_category,
    scale_budget,
)
from victor.framework.defaults.safety import DefaultSafetyExtension as DirectSafety
from victor.framework.defaults.stages import get_default_stages as direct_get_stages
from victor.framework.stage_manager import get_default_stages as sm_get_stages
from victor_contracts import StageDefinition as SdkStageDefinition

# =============================================================================
# Default Stages
# =============================================================================


class TestGetDefaultStages:
    """Tests for get_default_stages()."""

    def test_returns_seven_stages(self):
        stages = get_default_stages()
        assert len(stages) == 7

    def test_stage_names(self):
        stages = get_default_stages()
        expected = {
            "initial",
            "planning",
            "reading",
            "analysis",
            "execution",
            "verification",
            "completion",
        }
        assert set(stages.keys()) == expected

    def test_stages_are_stage_definitions(self):
        stages = get_default_stages()
        for defn in stages.values():
            assert isinstance(defn, StageDefinition)
            assert isinstance(defn, SdkStageDefinition)

    def test_stage_order_matches_runtime_default_sequence(self):
        stages = get_default_stages()
        assert list(stages.keys()) == list(sm_get_stages().keys())

    def test_all_stages_have_keywords(self):
        stages = get_default_stages()
        for name, defn in stages.items():
            assert len(defn.keywords) > 0, f"Stage '{name}' has no keywords"

    def test_all_stages_have_tools(self):
        stages = get_default_stages()
        for name, defn in stages.items():
            assert len(defn.tools) > 0, f"Stage '{name}' has no tools"

    def test_initial_stage_is_order_zero(self):
        stages = get_default_stages()
        assert list(stages.keys())[0] == "initial"

    def test_completion_is_last(self):
        stages = get_default_stages()
        assert list(stages.keys())[-1] == "completion"

    def test_returns_fresh_dict_each_call(self):
        stages1 = get_default_stages()
        stages2 = get_default_stages()
        assert stages1 is not stages2

    def test_reexport_matches_stage_manager(self):
        """Ensure defaults normalize the runtime stage-manager defaults into SDK contracts."""
        default_stages = direct_get_stages()
        runtime_stages = sm_get_stages()

        assert list(default_stages.keys()) == list(runtime_stages.keys())
        for name, stage in default_stages.items():
            runtime_stage = runtime_stages[name]
            assert stage.name == runtime_stage.name
            assert stage.description == runtime_stage.description
            assert stage.keywords == runtime_stage.keywords
            assert stage.tools == runtime_stage.tools


# =============================================================================
# Default Safety Extension
# =============================================================================


class TestDefaultSafetyExtension:
    """Tests for DefaultSafetyExtension."""

    @pytest.fixture()
    def ext(self):
        return DefaultSafetyExtension()

    def test_get_bash_patterns_returns_list(self, ext):
        patterns = ext.get_bash_patterns()
        assert isinstance(patterns, list)
        assert len(patterns) > 0

    def test_bash_patterns_cover_git(self, ext):
        categories = {p.category for p in ext.get_bash_patterns()}
        assert "git" in categories

    def test_bash_patterns_cover_filesystem(self, ext):
        categories = {p.category for p in ext.get_bash_patterns()}
        assert "filesystem" in categories

    def test_bash_patterns_have_critical_entries(self, ext):
        risk_levels = {p.risk_level for p in ext.get_bash_patterns()}
        assert "CRITICAL" in risk_levels

    def test_get_file_patterns_returns_list(self, ext):
        patterns = ext.get_file_patterns()
        assert isinstance(patterns, list)
        assert len(patterns) > 0

    def test_file_patterns_cover_credentials(self, ext):
        categories = {p.category for p in ext.get_file_patterns()}
        assert "credentials" in categories

    def test_file_patterns_cover_system(self, ext):
        categories = {p.category for p in ext.get_file_patterns()}
        assert "system" in categories

    def test_get_tool_restrictions(self, ext):
        restrictions = ext.get_tool_restrictions()
        assert "write" in restrictions
        assert "shell" in restrictions

    def test_category_is_default(self, ext):
        assert ext.get_category() == "default"

    def test_force_push_pattern_matches(self, ext):
        pattern = next(p for p in ext.get_bash_patterns() if "force" in p.description.lower())
        assert re.search(pattern.pattern, "git push origin main --force")

    def test_hard_reset_pattern_matches(self, ext):
        pattern = next(p for p in ext.get_bash_patterns() if "hard reset" in p.description.lower())
        assert re.search(pattern.pattern, "git reset --hard HEAD~3")

    def test_rm_rf_root_pattern_matches(self, ext):
        pattern = next(
            p
            for p in ext.get_bash_patterns()
            if "root" in p.description.lower() and "delete" in p.description.lower()
        )
        assert re.search(pattern.pattern, "rm -rf /")

    def test_env_file_pattern_matches(self, ext):
        pattern = next(p for p in ext.get_file_patterns() if "environment" in p.description.lower())
        assert re.search(pattern.pattern, "config/.env")

    def test_key_file_pattern_matches(self, ext):
        pattern = next(p for p in ext.get_file_patterns() if "certificate" in p.description.lower())
        assert re.search(pattern.pattern, "server.key")
        assert re.search(pattern.pattern, "cert.pem")

    def test_implements_safety_protocol(self, ext):
        from victor.core.verticals.protocols.safety_provider import (
            SafetyExtensionProtocol,
        )

        assert isinstance(ext, SafetyExtensionProtocol)

    def test_reexport_matches_direct(self):
        assert DirectSafety is DefaultSafetyExtension


# =============================================================================
# Subclass inheritance
# =============================================================================


class TestSafetySubclass:
    """Verify that subclassing DefaultSafetyExtension works correctly."""

    def test_subclass_extends_bash_patterns(self):
        from victor.security.safety.types import SafetyPattern

        class CustomSafety(DefaultSafetyExtension):
            def get_bash_patterns(self):
                patterns = super().get_bash_patterns()
                patterns.append(
                    SafetyPattern(
                        pattern=r"custom_danger",
                        description="Custom danger",
                        risk_level="HIGH",
                        category="custom",
                    )
                )
                return patterns

        ext = CustomSafety()
        patterns = ext.get_bash_patterns()
        custom = [p for p in patterns if p.category == "custom"]
        assert len(custom) == 1
        # base patterns still present
        git_patterns = [p for p in patterns if p.category == "git"]
        assert len(git_patterns) > 0


# =============================================================================
# One-stop imports
# =============================================================================


class TestDefaultsReexports:
    """Verify all re-exports are importable from victor.framework.defaults."""

    def test_get_default_stages(self):
        assert callable(get_default_stages)

    def test_default_safety_extension(self):
        assert DefaultSafetyExtension is not None

    def test_stage_definition(self):
        assert StageDefinition is SdkStageDefinition

    def test_base_rl_config(self):
        assert BaseRLConfig is not None

    def test_default_active_learners(self):
        assert isinstance(DEFAULT_ACTIVE_LEARNERS, list)

    def test_default_patience_map(self):
        assert isinstance(DEFAULT_PATIENCE_MAP, dict)

    def test_mode_config_registry(self):
        assert ModeConfigRegistry is not None

    def test_mode_definition(self):
        assert ModeDefinition is not None

    def test_registry_based_mode_config_provider(self):
        assert RegistryBasedModeConfigProvider is not None

    def test_default_modes(self):
        assert isinstance(DEFAULT_MODES, dict)

    def test_default_task_budgets(self):
        assert isinstance(DEFAULT_TASK_BUDGETS, dict)

    def test_default_complexity_map(self):
        assert isinstance(DEFAULT_COMPLEXITY_MAP, dict)

    def test_create_complexity_map(self):
        assert callable(create_complexity_map)

    def test_persona_helpers(self):
        assert PersonaHelpers is not None

    def test_modification_stage_tools(self):
        assert isinstance(MODIFICATION_STAGE_TOOLS, dict)

    def test_analysis_stage_tools(self):
        assert isinstance(ANALYSIS_STAGE_TOOLS, dict)

    def test_execution_stage_tools(self):
        assert isinstance(EXECUTION_STAGE_TOOLS, dict)

    def test_get_stage_tools_for_category(self):
        assert callable(get_stage_tools_for_category)

    def test_default_budget_scaling(self):
        assert isinstance(DEFAULT_BUDGET_SCALING, dict)

    def test_scale_budget(self):
        assert callable(scale_budget)


# =============================================================================
# Mode Configs
# =============================================================================


class TestDefaultComplexityMap:
    """Tests for DEFAULT_COMPLEXITY_MAP and create_complexity_map."""

    def test_has_all_five_levels(self):
        expected = {"trivial", "simple", "moderate", "complex", "highly_complex"}
        assert set(DEFAULT_COMPLEXITY_MAP.keys()) == expected

    def test_values_are_valid_mode_names(self):
        for mode in DEFAULT_COMPLEXITY_MAP.values():
            assert mode in DEFAULT_MODES, f"Mode '{mode}' not in DEFAULT_MODES"

    def test_create_no_overrides_returns_defaults(self):
        result = create_complexity_map()
        assert result == DEFAULT_COMPLEXITY_MAP
        assert result is not DEFAULT_COMPLEXITY_MAP  # fresh copy

    def test_create_single_override(self):
        result = create_complexity_map(complex="thorough")
        assert result["complex"] == "thorough"
        # other keys unchanged
        assert result["trivial"] == "quick"
        assert result["simple"] == "quick"
        assert result["moderate"] == "standard"
        assert result["highly_complex"] == "extended"

    def test_create_multiple_overrides(self):
        result = create_complexity_map(complex="thorough", highly_complex="architect")
        assert result["complex"] == "thorough"
        assert result["highly_complex"] == "architect"

    def test_create_does_not_mutate_defaults(self):
        original = dict(DEFAULT_COMPLEXITY_MAP)
        create_complexity_map(trivial="skip")
        assert DEFAULT_COMPLEXITY_MAP == original


# =============================================================================
# Persona Helpers
# =============================================================================


class _CommStyle(Enum):
    COLLABORATIVE = "collaborative"
    DIRECTIVE = "directive"


class _DecStyle(Enum):
    PRAGMATIC = "pragmatic"
    ANALYTICAL = "analytical"


@dataclass
class _MockTraits:
    communication_style: _CommStyle = _CommStyle.COLLABORATIVE
    decision_style: _DecStyle = _DecStyle.PRAGMATIC

    def to_prompt_hints(self) -> str:
        return f"Communicates {self.communication_style.value}."


@dataclass
class _MockPersona:
    name: str
    role: str
    expertise: list = field(default_factory=list)
    secondary_expertise: list = field(default_factory=list)
    traits: _MockTraits = field(default_factory=_MockTraits)

    def get_expertise_list(self) -> list:
        return list(self.expertise) + list(self.secondary_expertise)

    def generate_backstory(self) -> str:
        return f"A {self.role} specialising in {', '.join(self.expertise)}."


_MOCK_PERSONAS = {
    "architect": _MockPersona(
        name="Architect",
        role="planner",
        expertise=["design", "patterns"],
        secondary_expertise=["testing"],
    ),
    "reviewer": _MockPersona(
        name="Reviewer",
        role="reviewer",
        expertise=["quality"],
        traits=_MockTraits(
            communication_style=_CommStyle.DIRECTIVE,
            decision_style=_DecStyle.ANALYTICAL,
        ),
    ),
    "executor": _MockPersona(
        name="Executor",
        role="executor",
        expertise=["coding"],
        secondary_expertise=["debugging"],
    ),
}


class TestPersonaHelpers:
    """Tests for PersonaHelpers."""

    @pytest.fixture()
    def helpers(self):
        return PersonaHelpers(_MOCK_PERSONAS)

    def test_get_persona_found(self, helpers):
        p = helpers.get_persona("architect")
        assert p is not None
        assert p.name == "Architect"

    def test_get_persona_not_found(self, helpers):
        assert helpers.get_persona("unknown") is None

    def test_list_personas(self, helpers):
        names = helpers.list_personas()
        assert set(names) == {"architect", "reviewer", "executor"}

    def test_get_personas_for_role(self, helpers):
        planners = helpers.get_personas_for_role("planner")
        assert len(planners) == 1
        assert planners[0].name == "Architect"

    def test_get_personas_for_role_empty(self, helpers):
        assert helpers.get_personas_for_role("researcher") == []

    def test_get_persona_by_expertise_primary(self, helpers):
        results = helpers.get_persona_by_expertise("design")
        assert len(results) == 1
        assert results[0].name == "Architect"

    def test_get_persona_by_expertise_secondary(self, helpers):
        results = helpers.get_persona_by_expertise("testing")
        assert len(results) == 1
        assert results[0].name == "Architect"

    def test_get_persona_by_expertise_both(self, helpers):
        results = helpers.get_persona_by_expertise("debugging")
        assert len(results) == 1
        assert results[0].name == "Executor"

    def test_get_persona_by_expertise_not_found(self, helpers):
        assert helpers.get_persona_by_expertise("nonexistent") == []

    def test_apply_persona_to_spec_unknown(self, helpers):
        """Unknown persona returns spec unchanged."""

        class Spec:
            expertise = None
            backstory = None
            personality = None

        spec = Spec()
        result = helpers.apply_persona_to_spec(spec, "unknown")
        assert result is spec
        assert spec.expertise is None

    def test_apply_persona_to_spec_sets_expertise(self, helpers):
        class Spec:
            expertise = None
            backstory = None
            personality = None

        spec = Spec()
        helpers.apply_persona_to_spec(spec, "architect")
        assert "design" in spec.expertise
        assert "patterns" in spec.expertise
        assert "testing" in spec.expertise

    def test_apply_persona_to_spec_merges_expertise(self, helpers):
        class Spec:
            expertise = ["existing"]
            backstory = None
            personality = None

        spec = Spec()
        helpers.apply_persona_to_spec(spec, "architect")
        assert "existing" in spec.expertise
        assert "design" in spec.expertise

    def test_apply_persona_to_spec_sets_backstory(self, helpers):
        class Spec:
            expertise = None
            backstory = None
            personality = None

        spec = Spec()
        helpers.apply_persona_to_spec(spec, "architect")
        assert spec.backstory is not None
        assert "planner" in spec.backstory

    def test_apply_persona_to_spec_appends_trait_hints(self, helpers):
        class Spec:
            expertise = None
            backstory = "Existing backstory."
            personality = None

        spec = Spec()
        helpers.apply_persona_to_spec(spec, "architect")
        assert "Existing backstory." in spec.backstory
        assert "Communicates" in spec.backstory

    def test_apply_persona_to_spec_sets_personality(self, helpers):
        class Spec:
            expertise = None
            backstory = None
            personality = None

        spec = Spec()
        helpers.apply_persona_to_spec(spec, "reviewer")
        assert "directive" in spec.personality
        assert "analytical" in spec.personality


# =============================================================================
# Task Hints
# =============================================================================

_REQUIRED_STAGES = {"initial", "reading", "executing", "verifying"}


class TestStageToolMappings:
    """Tests for stage-tool mapping constants."""

    def test_modification_has_all_stages(self):
        assert set(MODIFICATION_STAGE_TOOLS.keys()) == _REQUIRED_STAGES

    def test_analysis_has_all_stages(self):
        assert set(ANALYSIS_STAGE_TOOLS.keys()) == _REQUIRED_STAGES

    def test_execution_has_all_stages(self):
        assert set(EXECUTION_STAGE_TOOLS.keys()) == _REQUIRED_STAGES

    def test_modification_executing_has_edit(self):
        assert "edit" in MODIFICATION_STAGE_TOOLS["executing"]

    def test_analysis_executing_is_read_only(self):
        tools = ANALYSIS_STAGE_TOOLS["executing"]
        assert "edit" not in tools
        assert "read" in tools

    def test_execution_executing_has_bash(self):
        assert "shell" in EXECUTION_STAGE_TOOLS["executing"]


class TestGetStageToolsForCategory:
    """Tests for get_stage_tools_for_category()."""

    def test_modification(self):
        result = get_stage_tools_for_category("modification")
        assert result == MODIFICATION_STAGE_TOOLS

    def test_analysis(self):
        result = get_stage_tools_for_category("analysis")
        assert result == ANALYSIS_STAGE_TOOLS

    def test_execution(self):
        result = get_stage_tools_for_category("execution")
        assert result == EXECUTION_STAGE_TOOLS

    def test_unknown_falls_back_to_analysis(self):
        result = get_stage_tools_for_category("unknown")
        assert result == ANALYSIS_STAGE_TOOLS

    def test_returns_copy(self):
        result = get_stage_tools_for_category("modification")
        result["initial"].append("extra_tool")
        # original unchanged
        assert "extra_tool" not in MODIFICATION_STAGE_TOOLS["initial"]


class TestScaleBudget:
    """Tests for scale_budget()."""

    def test_trivial(self):
        assert scale_budget(10, "trivial") == 5

    def test_simple(self):
        assert scale_budget(10, "simple") == 8  # ceil(7.5)

    def test_moderate(self):
        assert scale_budget(10, "moderate") == 10

    def test_complex(self):
        assert scale_budget(10, "complex") == 15

    def test_highly_complex(self):
        assert scale_budget(10, "highly_complex") == 25

    def test_unknown_uses_one_multiplier(self):
        assert scale_budget(10, "unknown") == 10

    def test_minimum_is_one(self):
        assert scale_budget(0, "trivial") == 1

    def test_budget_scaling_has_all_levels(self):
        expected = {"trivial", "simple", "moderate", "complex", "highly_complex"}
        assert set(DEFAULT_BUDGET_SCALING.keys()) == expected

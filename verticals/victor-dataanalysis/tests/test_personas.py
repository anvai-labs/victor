# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for data analysis team personas."""

from victor_contracts import TeamMemberSpec

from victor_dataanalysis.teams.personas import (
    DATA_ANALYSIS_PERSONAS,
    DataAnalysisPersona,
    DataAnalysisPersonaTraits,
    DecisionStyle,
    apply_persona_to_spec,
    get_persona,
    get_personas_for_role,
    list_personas,
)


class TestPersonaCatalog:
    def test_expected_personas_defined(self):
        names = list_personas()

        for expected in (
            "data_engineer",
            "statistician",
            "ml_engineer",
            "visualization_specialist",
            "data_quality_analyst",
            "business_analyst",
        ):
            assert expected in names

    def test_persona_structure(self):
        for name, persona in DATA_ANALYSIS_PERSONAS.items():
            assert isinstance(persona, DataAnalysisPersona)
            assert persona.name, f"{name} has no display name"
            assert persona.role
            assert persona.expertise, f"{name} has no expertise"
            assert isinstance(persona.traits, DataAnalysisPersonaTraits)

    def test_get_persona_lookup(self):
        assert get_persona("statistician") is DATA_ANALYSIS_PERSONAS["statistician"]
        assert get_persona("nonexistent") is None

    def test_get_personas_for_role_filters(self):
        statisticians = get_personas_for_role("statistician")

        assert statisticians, "no statistician personas found"
        assert all(p.role == "statistician" for p in statisticians)

    def test_trait_scales_are_normalized(self):
        for name, persona in DATA_ANALYSIS_PERSONAS.items():
            traits = persona.traits
            for attr in ("quantitative_focus", "risk_tolerance", "visualization_preference"):
                value = getattr(traits, attr)
                assert 0.0 <= value <= 1.0, f"{name}.{attr}={value} outside [0,1]"


class TestPromptHints:
    def test_traits_generate_prompt_hints(self):
        hints = DataAnalysisPersonaTraits().to_prompt_hints()

        assert hints
        assert "Balance rigor with practical considerations." in hints  # PRAGMATIC default

    def test_rigorous_style_demands_methodology(self):
        traits = DataAnalysisPersonaTraits(decision_style=DecisionStyle.RIGOROUS)

        assert "statistical standards" in traits.to_prompt_hints()

    def test_low_risk_tolerance_warns_about_assumptions(self):
        traits = DataAnalysisPersonaTraits(risk_tolerance=0.1)

        assert "risky assumptions" in traits.to_prompt_hints()


class TestApplyPersonaToSpec:
    def test_apply_fills_empty_spec_fields(self):
        spec = TeamMemberSpec(role="executor", goal="profile the dataset")

        result = apply_persona_to_spec(spec, "data_engineer")

        assert result is spec  # modified in place
        assert spec.expertise
        assert spec.backstory
        assert spec.personality

    def test_apply_merges_existing_expertise(self):
        spec = TeamMemberSpec(
            role="executor",
            goal="analyze",
            expertise=["custom_skill"],
        )

        apply_persona_to_spec(spec, "statistician")

        assert "custom_skill" in spec.expertise
        assert len(spec.expertise) > 1

    def test_apply_unknown_persona_is_noop(self):
        spec = TeamMemberSpec(role="executor", goal="analyze")

        result = apply_persona_to_spec(spec, "not_a_persona")

        assert result is spec
        assert not spec.backstory

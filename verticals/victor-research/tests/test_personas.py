# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for research team personas."""

from victor_contracts.team_schema import TeamMemberSpec

from victor_research.teams.personas import (
    RESEARCH_PERSONAS,
    ResearchPersona,
    ResearchPersonaTraits,
    apply_persona_to_spec,
    get_persona,
    get_personas_for_role,
    list_personas,
)


class TestPersonaCatalog:
    def test_expected_personas_defined(self):
        names = list_personas()

        for expected in (
            "web_researcher",
            "academic_researcher",
            "fact_checker",
            "synthesis_specialist",
            "competitive_analyst",
            "citation_specialist",
        ):
            assert expected in names

    def test_persona_structure(self):
        for name, persona in RESEARCH_PERSONAS.items():
            assert isinstance(persona, ResearchPersona)
            assert persona.name, f"{name} has no display name"
            assert persona.role
            assert persona.expertise, f"{name} has no expertise"
            assert isinstance(persona.traits, ResearchPersonaTraits)

    def test_get_persona_lookup(self):
        assert get_persona("fact_checker") is RESEARCH_PERSONAS["fact_checker"]
        assert get_persona("nonexistent") is None

    def test_get_personas_for_role_filters(self):
        researchers = get_personas_for_role("researcher")

        assert len(researchers) >= 2  # web_researcher + academic_researcher
        assert all(p.role == "researcher" for p in researchers)

    def test_trait_scales_are_normalized(self):
        for name, persona in RESEARCH_PERSONAS.items():
            traits = persona.traits
            for attr in (
                "source_rigor",
                "breadth_preference",
                "citation_detail",
                "skepticism",
                "collaboration_preference",
                "verbosity",
            ):
                value = getattr(traits, attr)
                assert 0.0 <= value <= 1.0, f"{name}.{attr}={value} outside [0,1]"


class TestPromptHints:
    def test_traits_generate_prompt_hints(self):
        hints = ResearchPersonaTraits().to_prompt_hints()

        assert hints
        assert "evidence" in hints.lower()  # EVIDENCE_BASED default decision style

    def test_backstory_generation(self):
        backstory = RESEARCH_PERSONAS["web_researcher"].generate_backstory()

        assert backstory


class TestApplyPersonaToSpec:
    def test_apply_fills_empty_spec_fields(self):
        spec = TeamMemberSpec(role="researcher", goal="find AI trend data")

        result = apply_persona_to_spec(spec, "web_researcher")

        assert result is spec  # modified in place
        assert spec.expertise
        assert spec.backstory
        assert spec.personality

    def test_apply_merges_existing_expertise(self):
        spec = TeamMemberSpec(
            role="researcher",
            goal="research",
            expertise=["custom_skill"],
        )

        apply_persona_to_spec(spec, "academic_researcher")

        assert "custom_skill" in spec.expertise
        assert len(spec.expertise) > 1

    def test_apply_unknown_persona_is_noop(self):
        spec = TeamMemberSpec(role="researcher", goal="research")

        result = apply_persona_to_spec(spec, "not_a_persona")

        assert result is spec
        assert not spec.backstory

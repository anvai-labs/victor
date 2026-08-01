# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for DevOps team personas."""

from victor_contracts.team_schema import TeamMemberSpec

from victor_devops.teams.personas import (
    DEVOPS_PERSONAS,
    DevOpsPersona,
    DevOpsPersonaTraits,
    apply_persona_to_spec,
    get_persona,
    get_personas_for_role,
    list_personas,
)


class TestPersonaCatalog:
    def test_expected_personas_defined(self):
        names = list_personas()

        for expected in (
            "infrastructure_architect",
            "ci_cd_engineer",
            "security_specialist",
            "monitoring_expert",
            "container_specialist",
            "configuration_manager",
        ):
            assert expected in names

    def test_persona_structure(self):
        for name, persona in DEVOPS_PERSONAS.items():
            assert isinstance(persona, DevOpsPersona)
            assert persona.name, f"{name} has no display name"
            assert persona.role
            assert persona.expertise, f"{name} has no expertise"
            assert isinstance(persona.traits, DevOpsPersonaTraits)

    def test_get_persona_lookup(self):
        assert get_persona("security_specialist") is DEVOPS_PERSONAS["security_specialist"]
        assert get_persona("nonexistent") is None

    def test_get_personas_for_role_filters(self):
        architects = get_personas_for_role("architect")

        assert architects, "no architect personas found"
        assert all(p.role == "architect" for p in architects)

    def test_trait_scales_are_normalized(self):
        for name, persona in DEVOPS_PERSONAS.items():
            traits = persona.traits
            for attr in ("automation_focus", "risk_tolerance", "collaboration_preference"):
                value = getattr(traits, attr)
                assert 0.0 <= value <= 1.0, f"{name}.{attr}={value} outside [0,1]"


class TestPromptHints:
    def test_traits_generate_prompt_hints(self):
        hints = DevOpsPersonaTraits().to_prompt_hints()

        assert hints
        assert "automat" in hints.lower()  # AUTOMATION_FIRST default style

    def test_low_risk_tolerance_warns_about_rollback(self):
        traits = DevOpsPersonaTraits(risk_tolerance=0.1)

        assert "rollback" in traits.to_prompt_hints().lower()


class TestApplyPersonaToSpec:
    def test_apply_fills_empty_spec_fields(self):
        spec = TeamMemberSpec(role="executor", goal="deploy the service")

        result = apply_persona_to_spec(spec, "infrastructure_architect")

        assert result is spec  # modified in place
        assert spec.expertise
        assert spec.backstory
        assert spec.personality

    def test_apply_merges_existing_expertise(self):
        spec = TeamMemberSpec(
            role="executor",
            goal="deploy",
            expertise=["custom_skill"],
        )

        apply_persona_to_spec(spec, "ci_cd_engineer")

        assert "custom_skill" in spec.expertise
        assert len(spec.expertise) > 1

    def test_apply_unknown_persona_is_noop(self):
        spec = TeamMemberSpec(role="executor", goal="deploy")

        result = apply_persona_to_spec(spec, "not_a_persona")

        assert result is spec
        assert not spec.backstory

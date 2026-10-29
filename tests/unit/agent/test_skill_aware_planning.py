"""Tests for skill-aware task planning.

Covers:
- enrich_plan_with_skills maps step types to matching skills
- Steps with matching skills get skill prompt fragments attached
- Steps without matching skills are unchanged
- Plan enrichment preserves original structure
"""

from __future__ import annotations

from unittest.mock import MagicMock


from victor_contracts.skills import SkillDefinition


def _make_skill(name: str, **kwargs):
    return SkillDefinition(
        name=name,
        description=kwargs.get("description", f"Skill: {name}"),
        category=kwargs.get("category", "coding"),
        prompt_fragment=kwargs.get("prompt_fragment", f"Steps for {name}."),
        required_tools=kwargs.get("required_tools", ["read_file"]),
        tags=kwargs.get("tags", frozenset()),
        phase=kwargs.get("phase", "action"),
    )


class TestEnrichPlanWithSkills:
    """enrich_plan_with_skills maps plan steps to skills."""

    def test_step_matches_skill_by_type(self):
        from victor.framework.skill_planner import enrich_plan_with_skills

        skills = {
            "debug_test_failure": _make_skill(
                "debug_test_failure",
                tags=frozenset({"debug", "test", "bugfix"}),
            ),
            "code_review": _make_skill(
                "code_review",
                tags=frozenset({"review", "quality"}),
            ),
        }

        # Plan steps as [id, type, description, tools]
        steps = [
            [1, "bugfix", "Fix the failing test", "read,edit"],
            [2, "review", "Review the changes", "read,grep"],
        ]

        enriched = enrich_plan_with_skills(steps, skills)
        assert enriched[0]["matched_skill"] == "debug_test_failure"
        assert enriched[1]["matched_skill"] == "code_review"

    def test_step_no_match_is_none(self):
        from victor.framework.skill_planner import enrich_plan_with_skills

        skills = {
            "debug": _make_skill("debug", tags=frozenset({"debug"})),
        }

        steps = [
            [1, "deploy", "Deploy to production", "shell"],
        ]

        enriched = enrich_plan_with_skills(steps, skills)
        assert enriched[0]["matched_skill"] is None

    def test_preserves_step_data(self):
        from victor.framework.skill_planner import enrich_plan_with_skills

        skills = {}
        steps = [
            [1, "analyze", "Analyze the code", "read,grep"],
        ]

        enriched = enrich_plan_with_skills(steps, skills)
        assert enriched[0]["id"] == 1
        assert enriched[0]["type"] == "analyze"
        assert enriched[0]["description"] == "Analyze the code"
        assert enriched[0]["tools"] == "read,grep"

    def test_empty_steps(self):
        from victor.framework.skill_planner import enrich_plan_with_skills

        assert enrich_plan_with_skills([], {}) == []

    def test_description_keyword_match(self):
        """Match skills via step description when type doesn't match."""
        from victor.framework.skill_planner import enrich_plan_with_skills

        skills = {
            "security_audit": _make_skill(
                "security_audit",
                tags=frozenset({"security", "audit", "vulnerabilities"}),
            ),
        }

        steps = [
            [1, "analyze", "Audit for security vulnerabilities", "read,grep"],
        ]

        enriched = enrich_plan_with_skills(steps, skills)
        assert enriched[0]["matched_skill"] == "security_audit"


class TestGenerateSkillAwarePlanPrompt:
    """Plan generation prompt includes available skills."""

    def test_prompt_includes_skills(self):
        from victor.framework.skill_planner import build_skill_aware_plan_prompt

        skills = {
            "debug": _make_skill("debug", description="Debug tests"),
            "refactor": _make_skill("refactor", description="Refactor code"),
        }

        prompt = build_skill_aware_plan_prompt("Fix the auth bug and clean up the code", skills)

        assert "debug" in prompt
        assert "refactor" in prompt
        assert "Fix the auth bug" in prompt

    def test_prompt_includes_selected_skill_sequence(self):
        from victor.framework.skill_planner import build_skill_aware_plan_prompt

        skills = {
            "debug": _make_skill("debug", description="Debug tests"),
            "refactor": _make_skill("refactor", description="Refactor code"),
        }

        prompt = build_skill_aware_plan_prompt(
            "Fix the auth bug and clean up the code",
            skills,
            selected_skills=["debug", "refactor"],
            decomposition_confidence=0.72,
        )

        assert "Suggested ordered skill sequence" in prompt
        assert "debug -> refactor" in prompt
        assert "confidence=0.72" in prompt


class TestSkillDecomposition:
    """Shared framework decomposition helper should reuse matcher ordering."""

    def test_build_skill_decomposition_uses_matcher_sequence(self):
        from victor.framework.skill_planner import (
            SkillDecomposition,
            build_skill_decomposition,
        )

        matcher = MagicMock()
        matcher.initialized = True
        matcher.match_multiple_sync.return_value = [
            (_make_skill("debug", phase="diagnostic"), 0.81),
            (_make_skill("refactor", phase="action"), 0.63),
        ]

        result = build_skill_decomposition("fix the failing auth tests", matcher)

        assert result == SkillDecomposition(
            skills=["debug", "refactor"],
            confidence=0.72,
            rationale="Ordered by framework skill matcher phase sequence",
        )

    def test_build_skill_decomposition_returns_none_when_matcher_unavailable(self):
        from victor.framework.skill_planner import build_skill_decomposition

        matcher = MagicMock()
        matcher.initialized = False

        assert build_skill_decomposition("anything", matcher) is None

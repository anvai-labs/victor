"""Tests for coding vertical skills."""

from victor_coding.skills import (
    CODING_SKILLS,
    code_review,
    debug_test_failure,
    implement_feature,
    text_to_sql,
)
from victor_contracts.skills import SkillDefinition


class TestCodingSkillDefinitions:
    """Verify all coding skills are well-formed."""

    def test_seven_skills_defined(self):
        assert len(CODING_SKILLS) == 7

    def test_all_are_skill_definitions(self):
        for skill in CODING_SKILLS:
            assert isinstance(skill, SkillDefinition)

    def test_unique_names(self):
        names = [s.name for s in CODING_SKILLS]
        assert len(names) == len(set(names))

    def test_all_categories_valid(self):
        for skill in CODING_SKILLS:
            assert skill.category in ("coding", "data", "browser")

    def test_text_to_sql_is_data_category(self):
        assert text_to_sql.category == "data"

    def test_all_have_prompt_fragments(self):
        for skill in CODING_SKILLS:
            assert len(skill.prompt_fragment) > 50

    def test_all_have_required_tools(self):
        for skill in CODING_SKILLS:
            assert len(skill.required_tools) >= 2

    def test_debug_test_failure_tools(self):
        assert "read" in debug_test_failure.required_tools
        assert "shell" in debug_test_failure.required_tools
        assert "edit" in debug_test_failure.required_tools

    def test_code_review_tools(self):
        assert "read" in code_review.required_tools
        assert "grep" in code_review.required_tools

    def test_implement_feature_tools(self):
        assert "write" in implement_feature.required_tools
        assert "edit" in implement_feature.required_tools

    def test_serialization_round_trip(self):
        for skill in CODING_SKILLS:
            d = skill.to_dict()
            restored = SkillDefinition.from_dict(d)
            assert restored.name == skill.name
            assert restored.category == skill.category
            assert set(restored.required_tools) == set(skill.required_tools)


class TestCodingAssistantGetSkills:
    """Verify CodingAssistant.get_skills() returns skills."""

    def test_get_skills_returns_list(self):
        from victor_coding.assistant import CodingAssistant

        skills = CodingAssistant.get_skills()
        assert isinstance(skills, list)
        assert len(skills) == 7

    def test_get_skills_returns_skill_definitions(self):
        from victor_coding.assistant import CodingAssistant

        skills = CodingAssistant.get_skills()
        for skill in skills:
            assert isinstance(skill, SkillDefinition)

    def test_text_to_sql_tools(self):
        assert "database" in text_to_sql.required_tools
        assert "read" in text_to_sql.required_tools
        assert "shell" in text_to_sql.required_tools

    def test_text_to_sql_tags_cover_data_backends(self):
        tags = text_to_sql.tags
        for needed in ("sql", "duckdb", "parquet", "iceberg", "database"):
            assert needed in tags

    def test_text_to_sql_fragment_is_derived_grounding(self):
        fragment = text_to_sql.prompt_fragment
        # Core methodology: derive, never hand-write; exact literals; execute to validate
        assert "Never hand-write" in fragment
        assert "SELECT DISTINCT" in fragment
        assert "read_parquet" in fragment
        assert "VALIDATE BY EXECUTION" in fragment
        # The column-label trap must be called out
        assert "human label" in fragment

    def test_agentbrowser_skill(self):
        from victor_coding.skills import agentbrowser

        assert agentbrowser.category == "browser"
        assert "shell" in agentbrowser.required_tools
        assert "read" in agentbrowser.required_tools
        fragment = agentbrowser.prompt_fragment
        assert "OBSERVE NARROWLY" in fragment
        assert "STALE_TARGET" in fragment
        assert "UNTRUSTED" in fragment
        for needed in ("browser", "automation", "navigate"):
            assert needed in agentbrowser.tags

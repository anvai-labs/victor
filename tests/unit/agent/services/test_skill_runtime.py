from types import SimpleNamespace
from unittest.mock import MagicMock

from victor.agent.services.orchestrator_protocol_adapter import (
    OrchestratorProtocolAdapter,
)
from victor.agent.services.skill_runtime import SkillRuntime


def test_skill_runtime_applies_single_match_and_records_analytics():
    skill = SimpleNamespace(name="debug")
    matcher = MagicMock()
    matcher._initialized = True
    matcher.match_multiple_sync.return_value = [(skill, 0.912)]
    analytics = MagicMock()
    host = SimpleNamespace(
        clear_active_skills=MagicMock(),
        inject_skill=MagicMock(),
        inject_skills=MagicMock(),
        _skill_matcher=matcher,
        _skill_auto_disabled=False,
        _manual_skill_active=False,
        _skill_analytics=analytics,
        _last_skill_match_info=None,
    )
    runtime = SkillRuntime(OrchestratorProtocolAdapter(host))
    runtime.clear_active_skills = MagicMock()
    runtime.inject_skill = MagicMock()
    runtime.inject_skills = MagicMock()

    runtime.apply_skill_for_turn("hello")

    runtime.clear_active_skills.assert_called_once_with()
    runtime.inject_skill.assert_called_once_with(skill)
    assert host._last_skill_match_info == {
        "auto_skill": "debug",
        "auto_skill_score": 0.91,
    }
    analytics.record_selection.assert_called_once_with("debug", 0.912)
    runtime.inject_skills.assert_not_called()


def test_skill_runtime_applies_multiple_matches_and_records_analytics():
    matches = [
        (SimpleNamespace(name="debug"), 0.9),
        (SimpleNamespace(name="refactor"), 0.876),
    ]
    matcher = MagicMock()
    matcher._initialized = True
    matcher.match_multiple_sync.return_value = matches
    analytics = MagicMock()
    host = SimpleNamespace(
        clear_active_skills=MagicMock(),
        inject_skill=MagicMock(),
        inject_skills=MagicMock(),
        _skill_matcher=matcher,
        _skill_auto_disabled=False,
        _manual_skill_active=False,
        _skill_analytics=analytics,
        _last_skill_match_info=None,
    )
    runtime = SkillRuntime(OrchestratorProtocolAdapter(host))
    runtime.clear_active_skills = MagicMock()
    runtime.inject_skill = MagicMock()
    runtime.inject_skills = MagicMock()

    runtime.apply_skill_for_turn("hello")

    runtime.clear_active_skills.assert_called_once_with()
    runtime.inject_skills.assert_called_once_with(matches)
    assert host._last_skill_match_info == {
        "auto_skills": [
            {"name": "debug", "score": 0.9},
            {"name": "refactor", "score": 0.88},
        ],
    }
    analytics.record_multi_selection.assert_called_once_with([("debug", 0.9), ("refactor", 0.876)])
    runtime.inject_skill.assert_not_called()


def test_skill_runtime_clears_last_match_info_on_miss():
    matcher = MagicMock()
    matcher._initialized = True
    matcher.match_multiple_sync.return_value = []
    analytics = MagicMock()
    host = SimpleNamespace(
        clear_active_skills=MagicMock(),
        inject_skill=MagicMock(),
        inject_skills=MagicMock(),
        _skill_matcher=matcher,
        _skill_auto_disabled=False,
        _manual_skill_active=False,
        _skill_analytics=analytics,
        _last_skill_match_info={"auto_skill": "stale"},
    )
    runtime = SkillRuntime(OrchestratorProtocolAdapter(host))
    runtime.clear_active_skills = MagicMock()

    runtime.apply_skill_for_turn("hello")

    runtime.clear_active_skills.assert_called_once_with()
    assert host._last_skill_match_info is None
    analytics.record_miss.assert_called_once_with()


def test_clear_active_skills_restores_base_prompt_and_syncs_conversation():
    prompt_runtime = MagicMock()
    host = SimpleNamespace(
        _active_skill_prompt="ACTIVE SKILL: debug",
        _base_system_prompt="Base prompt.",
        _system_prompt="ACTIVE SKILL: debug\nBase prompt.",
        _kv_optimization_enabled=False,
        _prompt_builder_runtime=prompt_runtime,
    )
    runtime = SkillRuntime(OrchestratorProtocolAdapter(host))

    runtime.clear_active_skills()

    assert host._active_skill_prompt == ""
    assert host._system_prompt == "Base prompt."
    prompt_runtime.sync_conversation_system_prompt.assert_called_once_with()


def test_get_skill_user_prefix_returns_active_prompt():
    host = SimpleNamespace(_active_skill_prompt="ACTIVE SKILL: debug")
    runtime = SkillRuntime(OrchestratorProtocolAdapter(host))

    assert runtime.get_skill_user_prefix() == "ACTIVE SKILL: debug"


def test_inject_skill_updates_system_prompt_when_kv_disabled():
    prompt_runtime = MagicMock()
    skill = SimpleNamespace(
        name="debug",
        description="Skill: debug",
        prompt_fragment="Prompt for debug.",
    )
    host = SimpleNamespace(
        _active_skill_prompt="",
        _base_system_prompt="Base prompt.",
        _system_prompt="Base prompt.",
        _kv_optimization_enabled=False,
        _prompt_builder_runtime=prompt_runtime,
    )
    runtime = SkillRuntime(OrchestratorProtocolAdapter(host))

    runtime.inject_skill(skill)

    assert "ACTIVE SKILL: debug" in host._active_skill_prompt
    assert host._system_prompt.startswith("ACTIVE SKILL: debug")
    assert host._system_prompt.endswith("Base prompt.")
    prompt_runtime.sync_conversation_system_prompt.assert_called_once_with()


def test_inject_skills_caps_at_three_and_preserves_phase_order():
    prompt_runtime = MagicMock()
    host = SimpleNamespace(
        _active_skill_prompt="",
        _base_system_prompt="Base prompt.",
        _system_prompt="Base prompt.",
        _kv_optimization_enabled=False,
        _prompt_builder_runtime=prompt_runtime,
    )
    runtime = SkillRuntime(OrchestratorProtocolAdapter(host))
    skills = [
        (SimpleNamespace(name="debug", description="d", prompt_fragment="pd"), 0.9),
        (SimpleNamespace(name="refactor", description="r", prompt_fragment="pr"), 0.8),
        (SimpleNamespace(name="review", description="v", prompt_fragment="pv"), 0.7),
        (SimpleNamespace(name="extra", description="x", prompt_fragment="px"), 0.6),
    ]

    runtime.inject_skills(skills)

    assert "ACTIVE SKILLS (3)" in host._system_prompt
    assert "debug" in host._system_prompt
    assert "refactor" in host._system_prompt
    assert "review" in host._system_prompt
    assert "extra" not in host._system_prompt
    prompt_runtime.sync_conversation_system_prompt.assert_called_once_with()


def _make_host(matcher):
    return SimpleNamespace(
        clear_active_skills=MagicMock(),
        inject_skill=MagicMock(),
        inject_skills=MagicMock(),
        _skill_matcher=matcher,
        _skill_auto_disabled=False,
        _manual_skill_active=False,
        _skill_analytics=None,
        _last_skill_match_info=None,
    )


def test_bare_continuation_skips_skill_auto_selection():
    """A bare "continue" names no new task — the matcher must not run.

    Regression: 3 irrelevant skills were auto-injected on a continuation turn
    (session codingagent-363cca81).
    """
    matcher = MagicMock()
    matcher._initialized = True
    runtime = SkillRuntime(OrchestratorProtocolAdapter(_make_host(matcher)))
    runtime.clear_active_skills = MagicMock()

    for message in ("continue", "  Continue  ", "go on", "keep going"):
        runtime.apply_skill_for_turn(message)

    matcher.match_multiple_sync.assert_not_called()


def test_continuation_with_payload_still_matches_skills():
    """A continuation that names new work still gets skill matching on it."""
    matcher = MagicMock()
    matcher._initialized = True
    matcher.match_multiple_sync.return_value = []
    runtime = SkillRuntime(OrchestratorProtocolAdapter(_make_host(matcher)))
    runtime.clear_active_skills = MagicMock()

    runtime.apply_skill_for_turn("continue, and also migrate the database schema")

    matcher.match_multiple_sync.assert_called_once()

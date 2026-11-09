"""FEP-0043: generic stage-scoped guidance override on OperationalModeConfig.

Victor stays a generic framework: any consumer registers substring ->
replacement overrides for its own stage key; MODE_CONFIGS is never
mutated globally.
"""

from victor.core.mode_controller import MODE_CONFIGS, AgentMode, OperationalModeConfig


def test_default_build_guidance_contains_stop_line():
    build = MODE_CONFIGS[AgentMode.BUILD]
    assert "STOP excessive reading" in build.system_prompt_addition


def test_stage_override_replaces_guidance_line_for_instance_only():
    build = MODE_CONFIGS[AgentMode.BUILD]
    original = build.system_prompt_addition
    override = OperationalModeConfig(
        name=build.name,
        description=build.description,
        allowed_tools=set(build.allowed_tools),
        disallowed_tools=set(build.disallowed_tools),
        allow_all_tools=build.allow_all_tools,
        system_prompt_addition=original,
        stage_prompt_overrides={
            "STOP excessive reading": "Explore until the caller's declared criteria are met",
        },
    )
    assert "STOP excessive reading" not in override.system_prompt_addition
    assert "Explore until the caller's declared criteria are met" in override.system_prompt_addition
    # global MODE_CONFIGS untouched
    assert MODE_CONFIGS[AgentMode.BUILD].system_prompt_addition == original


def test_override_without_matching_pattern_leaves_guidance_unchanged():
    build = MODE_CONFIGS[AgentMode.BUILD]
    override = OperationalModeConfig(
        name=build.name,
        description=build.description,
        system_prompt_addition=build.system_prompt_addition,
        stage_prompt_overrides={"no such pattern": "replacement"},
    )
    assert override.system_prompt_addition == build.system_prompt_addition

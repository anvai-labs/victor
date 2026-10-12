from victor.agent.runtime.context import AgentRuntimeContext


def test_agent_runtime_context_identity_metadata():
    context = AgentRuntimeContext(
        agent_id="agent_researcher_1",
        display_name="Researcher",
        role="researcher",
        session_id="session_child",
        parent_session_id="session_root",
        team_id="team_1",
        plan_id="plan_1",
        plan_step_id="1",
        member_id="step_1_researcher",
    )

    assert context.identity_metadata() == {
        "agent_id": "agent_researcher_1",
        "display_name": "Researcher",
        "role": "researcher",
        "session_id": "session_child",
        "parent_session_id": "session_root",
        "team_id": "team_1",
        "plan_id": "plan_1",
        "plan_step_id": "1",
        "member_id": "step_1_researcher",
    }


def test_agent_runtime_context_child_session_derivation():
    root = AgentRuntimeContext(
        agent_id="root_agent",
        display_name="Root Agent",
        role="manager",
        session_id="session_root",
    )

    child = root.derive_child(
        agent_id="team_1_step_1",
        display_name="Step 1 Researcher",
        role="researcher",
        member_id="step_1_researcher",
        team_id="team_1",
        plan_id="plan_1",
        plan_step_id="1",
    )

    assert child.parent_session_id == "session_root"
    assert child.session_id == "session_root:team_1:step_1_researcher"
    assert child.agent_id == "team_1_step_1"


def test_runtime_identity_tracks_sessions_and_pins_member_context():
    from types import SimpleNamespace
    from victor.agent.runtime.context import RuntimeIdentity

    state = SimpleNamespace(active_session_id=None, _memory_session_id=None)
    owner = RuntimeIdentity(state)
    initial = owner.resolve()
    assert initial.session_id != RuntimeIdentity(state).resolve().session_id
    assert owner.resolve() is initial
    assert initial.role == "supervisor"
    state.active_session_id, state._memory_session_id = "resumed", "old-memory"
    root = owner.resolve()
    assert root.session_id == "resumed"
    assert owner.resolve() is root
    child = root.derive_child(
        agent_id="agent_1", display_name="Member", role="researcher", member_id="m1"
    )
    owner.bind_member(child)
    assert state.active_session_id == child.session_id
    assert owner.resolve() is child


def test_runtime_identity_rejects_invalid_session_without_losing_previous_context():
    import pytest
    from types import SimpleNamespace
    from victor.agent.runtime.context import RuntimeIdentity

    state = SimpleNamespace(active_session_id="valid", _memory_session_id=None)
    owner = RuntimeIdentity(state)
    previous = owner.resolve()
    state.active_session_id = ""
    with pytest.raises(ValueError, match="session"):
        owner.resolve()
    state.active_session_id = "valid"
    assert owner.resolve() is previous

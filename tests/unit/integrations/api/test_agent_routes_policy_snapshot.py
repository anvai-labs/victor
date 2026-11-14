"""Background-agent API admission inherits an immutable policy snapshot."""

from types import SimpleNamespace

import pytest

from fastapi.testclient import TestClient

from victor.agent.background_agent import BackgroundAgentManager
from victor.integrations.api import fastapi_server


def _create_server(monkeypatch, tmp_path):
    from victor.config import settings as settings_module

    # This test intentionally mutates sandbox policy. Use a fresh uncached
    # object and patch the loader so the process-wide read-only settings
    # snapshot cannot leak "required" into later unrelated shell tests.
    settings = settings_module.load_settings(fresh=True)
    settings.sandbox.sandbox_policy = "required"
    monkeypatch.setattr(settings_module, "load_settings", lambda fresh=False: settings)
    monkeypatch.setattr(
        fastapi_server,
        "load_fastapi_router_registrations",
        lambda *, workspace_root: [],
    )
    return fastapi_server.VictorFastAPIServer(
        workspace_root=str(tmp_path),
        enable_graphql=False,
        allow_unauthenticated=True,
    )


@pytest.mark.asyncio
async def test_agents_start_binds_server_policy_to_detached_background_run(monkeypatch, tmp_path):
    from victor.runtime.policy_snapshot import current_run_policy_snapshot
    import victor.agent.background_agent as background_agent

    captured = {}

    async def stream_chat(task):
        captured["policy"] = current_run_policy_snapshot()
        yield {"type": "content", "content": "background done"}

    manager = BackgroundAgentManager(
        orchestrator=SimpleNamespace(set_mode=lambda mode: None),
        chat_service=SimpleNamespace(stream_chat=stream_chat),
        max_concurrent=1,
    )
    monkeypatch.setattr(background_agent, "get_agent_manager", lambda: manager)

    server = _create_server(monkeypatch, tmp_path)
    client = TestClient(server.app)

    response = client.post("/agents/start", json={"task": "background task"})

    assert response.status_code == 200, response.text
    agent_id = response.json()["agent_id"]
    agent = manager.get_agent(agent_id)
    assert agent is not None
    await agent._task
    assert captured["policy"] is not None
    assert captured["policy"].effective_policy == "required"

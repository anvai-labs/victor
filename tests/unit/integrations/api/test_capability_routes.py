"""Tests for capability-related FastAPI routes."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from victor.integrations.api import fastapi_server


def _create_server(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        fastapi_server,
        "load_fastapi_router_registrations",
        lambda *, workspace_root: [],
    )
    server = fastapi_server.VictorFastAPIServer(
        workspace_root=str(tmp_path),
        enable_graphql=False,
        allow_unauthenticated=True,
    )
    server._orchestrator = SimpleNamespace()
    return server


@pytest.mark.asyncio
async def test_capabilities_recommend_endpoint_uses_shared_discovery_surface(
    monkeypatch, tmp_path: Path
) -> None:
    server = _create_server(monkeypatch, tmp_path)
    payload = {
        "task_type": "feature",
        "complexity": "high",
        "mode": "build",
        "vertical": "coding",
        "count": 1,
        "recommendations": [{"vertical": "coding", "action": "auto_spawn"}],
    }
    discovery = SimpleNamespace(recommend_for_task=MagicMock(return_value=payload))

    with patch(
        "victor.ui.commands.capabilities.get_capability_discovery",
        return_value=discovery,
    ):
        transport = ASGITransport(app=server.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/capabilities/recommend",
                params={
                    "task_type": "feature",
                    "complexity": "high",
                    "mode": "build",
                    "vertical": "coding",
                },
            )

    assert response.status_code == 200
    assert response.json() == payload
    discovery.recommend_for_task.assert_called_once_with(
        task_type="feature",
        complexity="high",
        mode="build",
        vertical="coding",
    )

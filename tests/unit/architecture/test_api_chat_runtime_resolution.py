# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Architecture guardrails for API chat runtime resolution."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def _read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text()


def _method_source(relative_path: str, class_name: str, method_name: str) -> str:
    source = _read(relative_path)
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if (
                    isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and child.name == method_name
                ):
                    segment = ast.get_source_segment(source, child)
                    if segment is None:
                        raise AssertionError(
                            f"Could not recover source for {class_name}.{method_name} in {relative_path}"
                        )
                    return segment
    raise AssertionError(f"Method {class_name}.{method_name} not found in {relative_path}")


def test_api_chat_surfaces_use_canonical_runtime_resolution() -> None:
    """API entrypoints should resolve chat through canonical runtime owners."""
    fastapi_chat_source = _read("victor/integrations/api/routes/chat_routes.py")
    fastapi_agent_source = _read("victor/integrations/api/routes/agent_routes.py")
    fastapi_ws_source = _method_source(
        "victor/integrations/api/fastapi_server.py",
        "VictorFastAPIServer",
        "_handle_ws_message",
    )

    assert "await server._get_victor_client()" in fastapi_chat_source
    assert (
        "from victor.framework.message_execution import resolve_chat_runtime"
        not in fastapi_chat_source
    )
    assert "resolve_chat_runtime" not in fastapi_chat_source
    assert "orchestrator._chat_service" not in fastapi_chat_source
    assert "from victor.runtime.chat_runtime import resolve_chat_service, resolve_chat_runtime" in (
        fastapi_agent_source
    )
    assert "from victor.framework.message_execution import resolve_chat_service" not in (
        fastapi_agent_source
    )
    assert "resolve_chat_service" in fastapi_agent_source
    assert "orchestrator._chat_service" not in fastapi_agent_source
    assert "await self._get_victor_client()" in fastapi_ws_source
    assert "orchestrator._chat_service" not in fastapi_ws_source


def test_fastapi_server_uses_framework_client_factory_for_conversation_helpers() -> None:
    """FastAPI server should not instantiate framework clients directly."""
    fastapi_server_source = _read("victor/integrations/api/fastapi_server.py")

    assert "from victor.framework.client import VictorClient" not in fastapi_server_source
    assert "VictorClient(" not in fastapi_server_source
    assert "FrameworkSessionRunner" in fastapi_server_source
    assert "create_victor_client" in fastapi_server_source

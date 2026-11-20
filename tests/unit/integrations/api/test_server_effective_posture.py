# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Server effective-posture surface + dormancy cleanup (Codex finding 22).

The aiohttp-era middleware stack had public re-exports but no production
FastAPI mount, and ``server_api_key`` / ``server_session_secret`` /
``server_max_sessions`` / ``server_session_ttl_seconds`` were declared (one
even auto-generated a secret!) with NO serving-path consumer — operators
could configure convincing controls the running server never enforced.

Under test: the deprecated export shims warn, the secret is no longer
fabricated, ``server_api_key`` is wired at its actual owner (the auth gate),
``server_max_message_bytes`` is enforced on the WebSocket receive path, and
``effective_posture()`` reports the enforced state of EVERY advertised
control (logged at startup, attached to /status).
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient
from pydantic import SecretStr

from victor.config.groups.server_config import ServerSettings
from victor.integrations.api import fastapi_server


def _create_server(monkeypatch, tmp_path: Path, **server_kwargs):
    monkeypatch.setattr(
        fastapi_server,
        "load_fastapi_router_registrations",
        lambda *, workspace_root: [],
    )
    server_kwargs.setdefault("allow_unauthenticated", True)
    return fastapi_server.VictorFastAPIServer(
        workspace_root=str(tmp_path),
        enable_graphql=False,
        **server_kwargs,
    )


# =============================================================================
# Deprecated middleware export shims (audit-first rule)
# =============================================================================


def test_deprecated_middleware_shim_warns_but_still_works():
    """External compatibility audit (Codex finding 22) found no remaining
    Victor-core, vscode-victor or docs consumers; the shims stay importable
    but every access emits a DeprecationWarning."""
    import importlib
    import warnings

    importlib.import_module("victor.integrations.api").__dict__.pop("APIMiddlewareStack", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from victor.integrations.api import APIMiddlewareStack

        stack = APIMiddlewareStack()

    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert hasattr(stack, "add_authentication")  # still functional
    assert hasattr(stack, "build")


def test_non_deprecated_lazy_exports_do_not_warn():
    """Negative: the deprecation shim is scoped to the middleware names."""
    import importlib
    import warnings

    importlib.import_module("victor.integrations.api").__dict__.pop("EventBroadcaster", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from victor.integrations.api import EventBroadcaster  # noqa: F401

    assert not any(issubclass(w.category, DeprecationWarning) for w in caught)


# =============================================================================
# The session secret is enforced by the web UI serving path — visibly
# =============================================================================


def test_server_session_secret_defaults_to_marked_autogeneration():
    """The web UI server (web/server/main.py) signs HMAC session tokens with
    this secret and fail-fasts when unset. The per-process default is
    restored for dev/test, but it is MARKED: an operator can always tell
    that the running secret was not supplied by configuration."""
    settings = ServerSettings()
    assert settings.server_session_secret is not None
    assert settings.server_session_secret_auto_generated is True


def test_explicit_server_session_secret_is_not_marked_generated():
    """Config compatibility: explicitly configured secrets keep loading and
    are NOT flagged as auto-generated (the marker drives the production
    warning and the posture's secret_source)."""
    settings = ServerSettings(server_session_secret=SecretStr("explicit"))
    assert settings.server_session_secret.get_secret_value() == "explicit"
    assert settings.server_session_secret_auto_generated is False


# =============================================================================
# server_api_key wired at its actual owner (the auth gate)
# =============================================================================


def test_server_api_key_setting_joins_the_credential_map(monkeypatch, tmp_path):
    """A configured ``server.server_api_key`` becomes a serving credential —
    validated-then-ignored was the failure mode under review (Codex finding
    22). Here the setting is injected through the settings loader the server
    actually reads at startup."""
    from victor.config import settings as settings_module

    real_load = settings_module.load_settings
    real_settings = real_load()
    monkeypatch.setattr(real_settings.server, "server_api_key", SecretStr("sk-from-settings"))
    monkeypatch.setattr(settings_module, "load_settings", lambda fresh=False: real_settings)

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

    assert "sk-from-settings" in server.api_keys
    assert server.api_keys["sk-from-settings"] == "server"
    posture = server.effective_posture()
    assert "settings.server.server_api_key" in posture["auth"]["sources"]
    assert posture["auth"]["enforced"] is True


def test_parameter_keys_take_precedence_over_the_settings_key(monkeypatch, tmp_path):
    """A constructor key and the settings key coexist; neither silently
    replaces the other."""
    from victor.config import settings as settings_module

    real_settings = settings_module.load_settings()
    monkeypatch.setattr(real_settings.server, "server_api_key", SecretStr("sk-from-settings"))
    monkeypatch.setattr(settings_module, "load_settings", lambda fresh=False: real_settings)

    monkeypatch.setattr(
        fastapi_server,
        "load_fastapi_router_registrations",
        lambda *, workspace_root: [],
    )
    server = fastapi_server.VictorFastAPIServer(
        workspace_root=str(tmp_path),
        enable_graphql=False,
        allow_unauthenticated=True,
        api_keys={"sk-param": "client-param"},
    )

    assert server.api_keys == {"sk-param": "client-param", "sk-from-settings": "server"}
    assert set(server.effective_posture()["auth"]["sources"]) == {
        "parameter",
        "settings.server.server_api_key",
    }


def test_effective_posture_reports_auth_from_parameter_keys(monkeypatch, tmp_path):
    server = _create_server(
        monkeypatch,
        tmp_path,
        api_keys={"sk-a": "client-a"},
        single_operator_mode=True,
    )
    posture = server.effective_posture()

    assert posture["auth"]["enforced"] is True
    assert posture["auth"]["credentials"] == 1
    assert posture["auth"]["single_operator_mode"] is True
    assert posture["auth"]["multi_principal_refusal_active"] is False


def test_effective_posture_covers_every_advertised_control(monkeypatch, tmp_path):
    """The startup surface reports each supported setting: enforced bounds
    AND the declared-but-not-enforced session trio, with disposition notes."""
    server = _create_server(monkeypatch, tmp_path)
    posture = server.effective_posture()

    # Enforced controls.
    assert posture["websocket"]["delivery_queue_size"] > 0
    assert posture["websocket"]["send_deadline_seconds"] > 0
    assert "server_max_message_bytes" in posture["websocket"]["enforced_settings"]

    # Session-token controls: enforced by the web UI serving path, with the
    # secret's provenance visible (Codex finding 22 correction).
    session_token = posture["session_token"]
    assert session_token["enforced_by"].startswith("web UI serving path")
    assert session_token["secret_configured"] is True
    assert (
        "auto-generated" in session_token["secret_source"]
        or "explicit" in session_token["secret_source"]
    )
    assert session_token["ttl_seconds"] and session_token["max_sessions"]

    # Declared-but-not-enforced: only the unmounted rate limiter.
    dead = posture["declared_but_not_enforced"]
    assert set(dead) == {"rate_limit_rpm"}
    assert all(dead.values())  # every entry carries a disposition note

    # The posture must be JSON-serializable for the startup log.
    json.dumps(posture, default=str)


def test_effective_posture_reports_explicit_secret_as_explicit(monkeypatch, tmp_path):
    """When the operator configured the secret, the posture says so."""
    from victor.config import settings as settings_module

    real_settings = settings_module.load_settings()
    monkeypatch.setattr(real_settings.server, "server_session_secret", SecretStr("operator-secret"))
    monkeypatch.setattr(real_settings.server, "server_session_secret_auto_generated", False)
    monkeypatch.setattr(settings_module, "load_settings", lambda fresh=False: real_settings)

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

    source = server.effective_posture()["session_token"]["secret_source"]
    assert source == "explicit operator configuration"


def test_effective_posture_flags_configured_but_unmounted_rate_limit(monkeypatch, tmp_path):
    server = _create_server(monkeypatch, tmp_path, rate_limit_rpm=120)
    note = server.effective_posture()["declared_but_not_enforced"]["rate_limit_rpm"]
    assert "120" in note
    assert "no rate limiter" in note


# =============================================================================
# server_max_message_bytes enforced at the WebSocket receive path
# =============================================================================


def test_oversized_ws_frame_is_refused_not_processed(monkeypatch, tmp_path):
    """Positive wiring: the advertised inbound cap is enforced on /ws."""
    server = _create_server(
        monkeypatch,
        tmp_path,
        api_keys={"sk-a": "client-a"},
    )
    server._ws_max_message_bytes = 64

    with TestClient(server.app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "auth", "api_key": "sk-a"})
            assert ws.receive_json()["type"] == "auth_success"
            ws.send_json({"type": "chat", "messages": [{"role": "user", "content": "x" * 500}]})
            response = ws.receive_json()

    assert response["type"] == "error"
    assert "server_max_message_bytes" in response["message"]


def test_oversized_ws_events_frame_is_refused(monkeypatch, tmp_path):
    """The /ws/events transport enforces the same inbound cap."""
    server = _create_server(monkeypatch, tmp_path)
    server._ws_max_message_bytes = 32

    with TestClient(server.app) as client:
        with client.websocket_connect("/ws/events") as ws:
            ws.send_json({"type": "subscribe", "categories": ["c" * 500]})
            response = ws.receive_json()

    assert response["type"] == "error"
    assert "server_max_message_bytes" in response["message"]


def test_normal_ws_frames_still_processed_under_cap(monkeypatch, tmp_path):
    """Negative guard: the cap must not reject legitimate small frames."""
    server = _create_server(monkeypatch, tmp_path)
    server._ws_max_message_bytes = 32768

    with TestClient(server.app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "ping"})
            response = ws.receive_json()

    assert response["type"] == "pong"


# =============================================================================
# /status carries the posture
# =============================================================================


def test_status_response_includes_effective_posture(monkeypatch, tmp_path):
    server = _create_server(monkeypatch, tmp_path)

    with TestClient(server.app) as client:
        response = client.get("/status")

    assert response.status_code == 200
    body = response.json()
    assert body["effective_posture"] is not None
    assert "auth" in body["effective_posture"]
    assert "declared_but_not_enforced" in body["effective_posture"]

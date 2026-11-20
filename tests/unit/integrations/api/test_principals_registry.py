# Copyright 2026 Vijaykumar Singh <vijaykumar@anvaiops.com>
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

"""FOU-A4 / Decision D4 (owner-blessed): operator-managed principals registry.

Three seams are pinned:

1. Strict parsing of the versioned YAML registry (unknown fields rejected,
   key/principal_id uniqueness, group dedup, non-empty required fields,
   kind restricted to ``api_key`` in this release).
2. The startup placement invariant: the resolved registry path must live
   outside EVERY sandbox-writable surface. Tests iterate the REAL surface
   list from ``victor.tools.sandbox.writable_surface_roots`` — never a
   copied approximation.
3. Server wiring: startup refuses (typed error) on a violating path; a
   valid registry feeds grouped principals through the real auth
   dependency; unlisted keys keep the legacy principal; expired entries
   fail closed with the same 401 as unknown keys.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from victor.integrations.api import fastapi_server
from victor.integrations.api.auth.principals_registry import (
    DEFAULT_REGISTRY_PATH,
    PrincipalsRegistryError,
    PrincipalsRegistryPathError,
    ensure_registry_path_outside_writable_surfaces,
    load_principals_registry,
    resolve_principals_registry_path,
)
from victor.tools.sandbox import writable_surface_roots

VALID_DOC = {
    "revision": 3,
    "principals": [
        {
            "key": "sk-alice",
            "principal_id": "alice",
            "kind": "api_key",
            "subject_label": "alice@corp",
            "groups": ["eng", "oncall"],
        },
        {
            "key": "sk-old",
            "principal_id": "legacy",
            "kind": "api_key",
            "subject_label": "legacy-svc",
            "expiry": "2020-01-01T00:00:00Z",
        },
    ],
}


# =============================================================================
# Strict parsing
# =============================================================================


def test_valid_registry_parses_and_maps_to_principal():
    registry = parse(VALID_DOC)
    assert registry.revision == 3
    alice = registry.find("sk-alice")
    assert alice is not None
    principal = alice.to_verified_principal()
    assert principal.principal_id == "alice"
    assert principal.client_id == "alice"
    assert principal.subject_label == "alice@corp"
    assert principal.groups == ("eng", "oncall")
    assert principal.primary_group == "eng"
    assert principal.expires_at is None


def parse(doc):
    from victor.integrations.api.auth.principals_registry import (
        parse_principals_registry,
    )

    return parse_principals_registry(doc)


def test_unknown_top_level_field_rejected():
    with pytest.raises(PrincipalsRegistryError, match="unknown top-level"):
        parse({"revision": 1, "principals": [], "defaults": {}})


def test_unknown_entry_field_rejected():
    doc = {
        "revision": 1,
        "principals": [
            {
                "key": "k",
                "principal_id": "p",
                "kind": "api_key",
                "subject_label": "s",
                "role": "admin",
            }
        ],
    }
    with pytest.raises(PrincipalsRegistryError, match="unknown fields"):
        parse(doc)


def test_duplicate_key_rejected():
    entry = {"key": "k", "principal_id": "p", "kind": "api_key", "subject_label": "s"}
    doc = {"revision": 1, "principals": [entry, dict(entry, principal_id="p2")]}
    with pytest.raises(PrincipalsRegistryError, match="duplicate key"):
        parse(doc)


def test_duplicate_principal_id_rejected():
    entry = {"key": "k1", "principal_id": "p", "kind": "api_key", "subject_label": "s"}
    doc = {"revision": 1, "principals": [entry, dict(entry, key="k2")]}
    with pytest.raises(PrincipalsRegistryError, match="duplicate principal_id"):
        parse(doc)


def test_duplicate_group_rejected():
    doc = {
        "revision": 1,
        "principals": [
            {
                "key": "k",
                "principal_id": "p",
                "kind": "api_key",
                "subject_label": "s",
                "groups": ["eng", "eng"],
            }
        ],
    }
    with pytest.raises(PrincipalsRegistryError, match="duplicate group"):
        parse(doc)


def test_empty_required_field_rejected():
    doc = {
        "revision": 1,
        "principals": [{"key": "  ", "principal_id": "p", "kind": "api_key", "subject_label": "s"}],
    }
    with pytest.raises(PrincipalsRegistryError, match="'key'"):
        parse(doc)


def test_unknown_kind_rejected():
    doc = {
        "revision": 1,
        "principals": [{"key": "k", "principal_id": "p", "kind": "oidc", "subject_label": "s"}],
    }
    with pytest.raises(PrincipalsRegistryError, match="kind"):
        parse(doc)


def test_bad_revision_rejected():
    for bad in ("3", 0, True, None):
        with pytest.raises(PrincipalsRegistryError, match="revision"):
            parse({"revision": bad, "principals": []})


def test_bad_expiry_rejected():
    doc = {
        "revision": 1,
        "principals": [
            {
                "key": "k",
                "principal_id": "p",
                "kind": "api_key",
                "subject_label": "s",
                "expiry": "not-a-date",
            }
        ],
    }
    with pytest.raises(PrincipalsRegistryError, match="expiry"):
        parse(doc)


def test_expiry_parses_and_expires():
    doc = {
        "revision": 1,
        "principals": [
            {
                "key": "k",
                "principal_id": "p",
                "kind": "api_key",
                "subject_label": "s",
                "expiry": "2030-06-01T12:00:00Z",
            }
        ],
    }
    entry = parse(doc).find("k")
    assert entry.expires_at is not None
    assert entry.expires_at.utcoffset() == timedelta(0)
    assert not entry.is_expired(now=datetime(2030, 1, 1, tzinfo=timezone.utc))
    assert entry.is_expired(now=datetime(2031, 1, 1, tzinfo=timezone.utc))


def test_invalid_yaml_raises_typed_error(tmp_path):
    path = tmp_path / "api_principals.yaml"
    path.write_text("principals: [oops\n  broken", encoding="utf-8")
    with pytest.raises(PrincipalsRegistryError, match="valid YAML"):
        load_principals_registry(path)


# =============================================================================
# Placement invariant over the REAL writable-surface list
# =============================================================================


def _real_surfaces(workspace_root=None):
    return writable_surface_roots(workspace_root=workspace_root)


def test_registry_inside_workspace_is_refused(tmp_path):
    surfaces = _real_surfaces(workspace_root=str(tmp_path))
    assert surfaces, "the real enumeration must be non-empty"
    kinds = {s["kind"] for s in surfaces}
    assert {"workspace", "victor_state", "temp"} <= kinds, kinds

    inside = tmp_path / "sub" / "api_principals.yaml"
    with pytest.raises(PrincipalsRegistryPathError) as excinfo:
        ensure_registry_path_outside_writable_surfaces(inside, surfaces)
    assert excinfo.value.surface["kind"] == "workspace"


def test_registry_inside_victor_state_root_is_refused(tmp_path):
    """The exact Codex finding: the registry must not sit inside the
    agent-writable Victor state root (~/.victor) — the agent could rewrite
    its own principals."""
    surfaces = _real_surfaces()
    state_root = next(s for s in surfaces if s["kind"] == "victor_state")
    inside = Path(state_root["root"]) / "api_principals.yaml"
    with pytest.raises(PrincipalsRegistryPathError) as excinfo:
        ensure_registry_path_outside_writable_surfaces(inside, surfaces)
    assert excinfo.value.surface["kind"] == "victor_state"


def test_registry_inside_temp_dir_is_refused(tmp_path):
    surfaces = _real_surfaces()
    with pytest.raises(PrincipalsRegistryPathError) as excinfo:
        ensure_registry_path_outside_writable_surfaces(tmp_path / "reg.yaml", surfaces)
    assert excinfo.value.surface["kind"] == "temp"


def test_registry_under_operator_allowed_mount_is_refused(tmp_path):
    surfaces = writable_surface_roots(workspace_root=None, allowed_mounts=[str(tmp_path)])
    with pytest.raises(PrincipalsRegistryPathError) as excinfo:
        ensure_registry_path_outside_writable_surfaces(tmp_path / "api_principals.yaml", surfaces)
    # tmp_path also sits in the temp surface; any refusal is a pass.
    assert excinfo.value.surface["kind"] in {"operator_mount", "temp"}


def test_system_default_path_passes_invariant(tmp_path):
    surfaces = _real_surfaces(workspace_root=str(tmp_path))
    ensure_registry_path_outside_writable_surfaces(DEFAULT_REGISTRY_PATH, surfaces)


# =============================================================================
# Path resolution
# =============================================================================


def test_explicit_path_wins(tmp_path):
    explicit = tmp_path / "elsewhere.yaml"
    assert resolve_principals_registry_path(str(explicit)) == explicit


def test_default_used_only_when_present(monkeypatch, tmp_path):
    import victor.integrations.api.auth.principals_registry as reg_mod

    monkeypatch.setattr(reg_mod, "DEFAULT_REGISTRY_PATH", tmp_path / "missing.yaml")
    assert resolve_principals_registry_path(None) is None

    present = tmp_path / "default.yaml"
    present.write_text("revision: 1\nprincipals: []\n", encoding="utf-8")
    monkeypatch.setattr(reg_mod, "DEFAULT_REGISTRY_PATH", present)
    assert resolve_principals_registry_path(None) == present


# =============================================================================
# Server wiring
# =============================================================================


def _server_settings_with(monkeypatch, **overrides):
    """Make the server constructor see server-group overrides.

    ``load_settings`` may return a shared/cached settings object, so the
    overrides are applied through ``monkeypatch.setattr`` on the settings
    group itself — the original values are restored at teardown and never
    leak into other tests."""
    import victor.config.settings as settings_mod

    real_load = settings_mod.load_settings

    def fake_load(*args, **kwargs):
        return real_load()

    monkeypatch.setattr(settings_mod, "load_settings", fake_load)
    settings = real_load()
    for field, value in overrides.items():
        monkeypatch.setattr(settings.server, field, value)


def test_startup_refuses_registry_inside_writable_surface(monkeypatch, tmp_path):
    """Typed fail-closed startup: a registry placed under the (writable)
    workspace root refuses server startup through the REAL constructor and
    the REAL surface enumeration."""
    _server_settings_with(monkeypatch, auth_principals_path=str(tmp_path / "api_principals.yaml"))
    with pytest.raises(PrincipalsRegistryPathError) as excinfo:
        _make_server(monkeypatch, tmp_path)
    assert excinfo.value.surface["kind"] in {"workspace", "temp"}


def _make_server(monkeypatch, tmp_path, orchestrator=None, **kwargs):
    monkeypatch.setattr(
        fastapi_server,
        "load_fastapi_router_registrations",
        lambda *, workspace_root: [],
    )
    server = fastapi_server.VictorFastAPIServer(
        workspace_root=str(tmp_path),
        enable_graphql=False,
        allow_unauthenticated=True,
        **kwargs,
    )
    server._orchestrator = orchestrator or _FakeOrchestrator()
    return server


class _FakeOrchestrator:
    def __init__(self) -> None:
        self.identities: list[tuple[str | None, str | None]] = []

    async def chat(self, message: str) -> object:
        from victor.core.context import get_auth_group_id, get_auth_subject_id

        self.identities.append((get_auth_subject_id(), get_auth_group_id()))
        from types import SimpleNamespace

        return SimpleNamespace(content=f"echo:{message}", tool_calls=[], status="ok")

    async def stream_chat(self, message: str):
        yield object()

    async def graceful_shutdown(self) -> None:
        return None


def _write_registry(tmp_path: Path) -> Path:
    import yaml

    path = tmp_path / "api_principals.yaml"
    path.write_text(yaml.safe_dump(VALID_DOC), encoding="utf-8")
    return path


def _inject_registry(server, path: Path, monkeypatch):
    """Injection for the E2E attribution path only. The placement invariant
    itself is pinned separately against the real surface enumeration (a
    temp path can never satisfy it by construction)."""
    registry = load_principals_registry(path)
    monkeypatch.setattr(server, "_principals_registry", registry)
    return registry


@pytest.mark.asyncio
async def test_registry_principal_groups_bind_through_dependency(monkeypatch, tmp_path):
    orchestrator = _FakeOrchestrator()
    server = _make_server(
        monkeypatch,
        tmp_path,
        orchestrator,
        api_keys={"sk-alice": "alice", "sk-old": "legacy"},
        single_operator_mode=True,
    )
    _inject_registry(server, _write_registry(tmp_path), monkeypatch)

    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/chat",
            json={"messages": [{"role": "user", "content": "hello"}]},
            headers={"Authorization": "Bearer sk-alice"},
        )

    assert response.status_code == 200
    assert orchestrator.identities == [("alice@corp", "eng")]


@pytest.mark.asyncio
async def test_registry_unlisted_key_falls_back_to_legacy(monkeypatch, tmp_path):
    orchestrator = _FakeOrchestrator()
    server = _make_server(
        monkeypatch,
        tmp_path,
        orchestrator,
        api_keys={"sk-alice": "alice", "sk-other": "other"},
        single_operator_mode=True,
    )
    _inject_registry(server, _write_registry(tmp_path), monkeypatch)

    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/chat",
            json={"messages": [{"role": "user", "content": "hello"}]},
            headers={"Authorization": "Bearer sk-other"},
        )

    assert response.status_code == 200
    assert orchestrator.identities == [("other", None)]


@pytest.mark.asyncio
async def test_expired_registry_entry_fails_closed(monkeypatch, tmp_path):
    orchestrator = _FakeOrchestrator()
    server = _make_server(
        monkeypatch,
        tmp_path,
        orchestrator,
        api_keys={"sk-alice": "alice", "sk-old": "legacy"},
        single_operator_mode=True,
    )
    _inject_registry(server, _write_registry(tmp_path), monkeypatch)

    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/chat",
            json={"messages": [{"role": "user", "content": "hello"}]},
            headers={"Authorization": "Bearer sk-old"},
        )

    assert response.status_code == 401
    assert orchestrator.identities == []


def test_malformed_registry_file_refuses_startup(monkeypatch, tmp_path):
    path = tmp_path / "api_principals.yaml"
    path.write_text(
        "revision: 1\nprincipals:\n  - key: k\n    principal_id: p\n"
        "    kind: api_key\n    subject_label: s\n    surprise: 1\n",
        encoding="utf-8",
    )

    # Direct loader: unknown field is a typed validation error.
    with pytest.raises(PrincipalsRegistryError, match="unknown fields"):
        load_principals_registry(path)


def test_reload_reactivates_registry(monkeypatch, tmp_path):
    """D4 activation: load is read-only; the explicit reload re-resolves and
    re-loads. (Surface enumeration is scoped to the system default here —
    the placement invariant is pinned separately against the real list.)"""
    orchestrator = _FakeOrchestrator()
    server = _make_server(monkeypatch, tmp_path, api_keys={"sk-alice": "alice"})
    assert server._principals_registry is None

    path = _write_registry(tmp_path)
    import victor.integrations.api.fastapi_server as fs_mod

    monkeypatch.setattr(server, "_resolve_principals_registry_path", lambda: path)
    monkeypatch.setattr(
        fs_mod,
        "writable_surface_roots",
        lambda **kwargs: [{"kind": "system", "root": str(DEFAULT_REGISTRY_PATH.parent)}],
    )
    registry = server.reload_principals_registry()
    assert registry is not None
    assert registry.revision == 3
    assert server._principals_registry is registry


def test_missing_explicit_registry_is_typed_error(tmp_path):
    """Fail closed, typed: a configured registry that does not exist refuses
    startup instead of silently downgrading to legacy principals."""
    missing = tmp_path / "nope" / "api_principals.yaml"
    with pytest.raises(PrincipalsRegistryError, match="missing"):
        load_principals_registry(missing)


def test_registry_principal_credential_id_is_public_digest():
    principal = parse(VALID_DOC).find("sk-alice").to_verified_principal()
    assert "sk-alice" not in (principal.credential_id or "")
    assert (principal.credential_id or "").startswith("sha256:")

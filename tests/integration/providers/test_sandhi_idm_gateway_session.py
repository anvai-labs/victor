"""Opt-in real IDM -> private broker file -> Victor Agent -> Sandhi acceptance.

Set IDM_GATEWAY_FIXTURE to the paired IDM checkout's
``tests/integration_gateway_introspection.py`` and SANDHI_OIDC_TEST_BINARY to the
independently built Sandhi candidate. Docker uses only cached Ubuntu 24.04 and a
private hostname mapping; neither global DNS nor TLS verification is changed.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import secrets
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

pytestmark = pytest.mark.integration
RESOURCE = "https://sandhi.local/resource"


def _publish(path: Path, envelope: dict[str, Any]) -> None:
    """Act as the fixture's broker publisher; the production reader stays unchanged."""
    temporary = path.with_name(".access-" + uuid.uuid4().hex)
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w") as output:
            json.dump(envelope, output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _http(root: str, path: str, token: str, body: Any = None) -> tuple[int, Any]:
    request = urllib.request.Request(
        root + path,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        # Never echo credentials or an upstream body into a pytest traceback.
        return error.code, None


def _fixture_module(path: Path) -> Any:
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("paired_idm_gateway_fixture", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def _sandhi(binary: Path, identity: Any, root: Path):
    """Run one owned local container with exact hostname and CA verification."""
    root.mkdir(mode=0o700)
    vault = root / "vault"
    vault.mkdir(mode=0o700)
    upstream_key = secrets.token_urlsafe(32)
    key = vault / "anthropic__default"
    key.write_text(upstream_key)
    key.chmod(0o600)
    config = {
        "issuer": identity.issuer,
        "client_id": identity.consumer_client_id,
        "access_token_audience": RESOURCE,
        "client_secret_env": "SANDHI_DEMO_IDM_SECRET",
        "redirect_url": "https://sandhi.example/auth/callback",
        "ca_file": "/demo/ca.pem",
        "subjects": {
            identity.subject: {
                "role": "admin",
                "grants": {"demo": {"upstream": "backend/demo", "models": ["fixture"]}},
            }
        },
    }
    config_file = root / "oidc.json"
    config_file.write_text(json.dumps(config))
    config_file.chmod(0o600)
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        port = reserved.getsockname()[1]
    name = "sandhi-idm-demo-" + uuid.uuid4().hex
    docker_config = root / "docker-config"
    docker_config.mkdir(mode=0o700)
    (docker_config / "config.json").write_text("{}")
    docker = ["docker", "--host=unix:///var/run/docker.sock", "--config", str(docker_config)]
    environment = {
        "PATH": "/usr/bin:/bin",
        "SANDHI_DEMO_IDM_SECRET": identity.consumer_secret,
    }
    argv = docker + [
        "run",
        "--rm",
        "--name",
        name,
        "--pull=never",
        "--network=host",
        "--add-host=id.localhost:127.0.0.1",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--tmpfs=/tmp:rw,nosuid,noexec,size=67108864",
        "--pids-limit=128",
        "--cpus=2",
        "--mount",
        f"type=bind,src={binary},dst=/usr/local/bin/sandhi-proxy,readonly",
        "--mount",
        f"type=bind,src={identity.ca_file},dst=/demo/ca.pem,readonly",
        "--mount",
        f"type=bind,src={root},dst=/state",
        "--env=SANDHI_DEMO_IDM_SECRET",
        "--env=SANDHI_AUTH_MODE=oidc",
        f"--env=SANDHI_BIND=127.0.0.1:{port}",
        "--env=SANDHI_LOG=warn",
        "--env=SANDHI_OIDC_CONFIG=/state/oidc.json",
        "--env=SANDHI_STORE=/state/usage.db",
        "--env=SANDHI_VAULT_BACKEND=credstore",
        "--env=SANDHI_CREDENTIALS_DIR=/state/vault",
        "ubuntu:24.04",
        "/usr/local/bin/sandhi-proxy",
    ]
    endpoint = f"http://127.0.0.1:{port}"
    log_path = root / "service.log"
    log_path.touch(mode=0o600)
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            argv, env=environment, stdin=subprocess.DEVNULL, stdout=log, stderr=log
        )
        try:
            deadline = time.monotonic() + 25
            while time.monotonic() < deadline:
                assert process.poll() is None, "owned Sandhi exited; private diagnostics retained"
                try:
                    urllib.request.urlopen(endpoint + "/healthz", timeout=0.2).close()
                    break
                except OSError:
                    time.sleep(0.1)
            else:
                raise AssertionError("owned Sandhi readiness deadline")
            yield endpoint, root / "usage.db", upstream_key
        finally:
            subprocess.run(
                docker + ["rm", "--force", name],
                env={"PATH": "/usr/bin:/bin"},
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=15,
                check=False,
            )
            process.wait(timeout=15)


@pytest.mark.asyncio
async def test_actual_resource_session_agent_usage_rotation_and_live_denials(tmp_path, monkeypatch):
    helper = os.environ.get("IDM_GATEWAY_FIXTURE")
    candidate = os.environ.get("SANDHI_OIDC_TEST_BINARY")
    if not helper or not candidate:
        pytest.skip("set paired IDM fixture and Sandhi OIDC candidate explicitly")
    binary = Path(candidate).resolve(strict=True)
    module = _fixture_module(Path(helper).resolve(strict=True))
    monkeypatch.chdir(tmp_path)
    home = tmp_path / "home"
    home.mkdir(mode=0o700, exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    for variable in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "SANDHI_GATEWAY_VIRTUAL_KEY"):
        monkeypatch.delenv(variable, raising=False)

    from victor.config.settings import Settings, set_project_root
    from victor.core.container import reset_container
    from victor.core.database import reset_all_databases
    from victor.framework.agent import Agent
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.providers.base import Message, ProviderError
    from victor.providers.sandhi_transport import SandhiOpenAIProvider

    set_project_root(tmp_path)
    reset_container()
    reset_all_databases()
    token_file = tmp_path / "access.json"
    import sandhi_gateway as binding

    module_root = Path(binding.__file__).parent
    native_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in module_root.glob("*.so")
    }
    evidence: dict[str, Any] = {
        "scope": "local candidates only; no deployment or published-wheel release claim",
        "idm_binary_sha256": hashlib.sha256(
            Path(os.environ["IDM_TEST_BINARY"]).read_bytes()
        ).hexdigest(),
        "idm_fixture_sha256": hashlib.sha256(Path(helper).read_bytes()).hexdigest(),
        "sdk_native_modules_sha256": native_hashes,
        "published_sdk_0171_qualification": "pending publication and clean wheel installation",
    }
    issued: list[dict[str, Any]] = []
    admitted_hashes: list[str] = []
    records: list[dict[str, Any]] = []
    with module.local_gateway_identity(tmp_path / "identity") as identity:
        assert identity.resource_uri == RESOURCE
        assert identity.consumer_client_id != identity.source_client_id
        bootstrap = identity.issue_resource()
        with _sandhi(binary, identity, tmp_path / "sandhi") as (endpoint, database, upstream_key):
            target = tmp_path / "input.txt"
            target.write_text("fixture-content\n")

            class Origin(BaseHTTPRequestHandler):
                def log_message(self, *_args):
                    pass

                def do_POST(self):  # noqa: N802
                    if (
                        self.headers.get("Authorization") is not None
                        or self.headers.get("x-api-key") != upstream_key
                    ):
                        raise AssertionError("gateway sent unexpected upstream credentials")
                    body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                    records.append(body)
                    # Hash-only observation of the actual provider credential at dispatch;
                    # neither authentication nor the token-file reader is replaced.
                    admitted_hashes.append(hashlib.sha256(provider._api_key.encode()).hexdigest())
                    first = len(records) == 1
                    if first:
                        rotated = identity.issue_resource()
                        issued.append(rotated)
                        _publish(token_file, rotated)
                    value = {
                        "id": f"fixture-{len(records)}",
                        "type": "message",
                        "role": "assistant",
                        "model": "fixture",
                        "stop_reason": "tool_use" if first else "end_turn",
                        "content": (
                            [
                                {
                                    "type": "tool_use",
                                    "id": "read-once",
                                    "name": "read",
                                    "input": {"path": str(target)},
                                }
                            ]
                            if first
                            else [{"type": "text", "text": "READY"}]
                        ),
                        "usage": {
                            "input_tokens": 2,
                            "output_tokens": 4,
                            "cache_read_input_tokens": 6,
                            "cache_creation_input_tokens": 2,
                        },
                    }
                    payload = json.dumps(value).encode()
                    self.send_response(200)
                    self.send_header("request-id", f"origin-{len(records)}")
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)

            origin = ThreadingHTTPServer(("127.0.0.1", 0), Origin)
            thread = threading.Thread(target=origin.serve_forever, daemon=True)
            thread.start()
            try:
                admin = bootstrap["access_token"]
                source = f"http://127.0.0.1:{origin.server_port}"
                status, _ = _http(
                    endpoint,
                    "/admin/keys/reference",
                    admin,
                    {"provider": "anthropic", "base_url": source},
                )
                assert status in (200, 201)
                definition = {
                    "id": "demo",
                    "provider": "fixture-messages",
                    "credential_ref": "anthropic:default",
                    "base_url": source,
                    "endpoint": {"endpoint_family": "anthropic_messages"},
                    "models": [
                        {
                            "id": "fixture",
                            "max_input_tokens": 100000,
                            "max_output_tokens": 128,
                            "capabilities": {"streaming": True},
                        }
                    ],
                }
                status, _ = _http(endpoint, "/admin/backends", admin, {"definition": definition})
                assert status in (200, 201)
                settings = Settings(
                    working_directory=str(tmp_path),
                    analytics_enabled=False,
                    use_semantic_tool_selection=False,
                    preload_embeddings=False,
                    enable_planning=False,
                    conversation_memory_enabled=False,
                    conversation_embeddings_enabled=False,
                    tool_cache_enabled=False,
                    chat_max_iterations=3,
                )
                fresh = identity.issue_resource()
                issued.append(fresh)
                _publish(token_file, fresh)
                gateway = {
                    "url": endpoint,
                    "grant": "demo",
                    "oidc": {
                        "token_file": str(token_file),
                        "issuer": identity.issuer,
                        "audience": RESOURCE,
                        "subject": identity.subject,
                    },
                }
                provider = SandhiOpenAIProvider(gateway=gateway)
                orch = AgentOrchestrator(
                    settings=settings,
                    provider=provider,
                    model="fixture",
                    max_tokens=64,
                    provider_name="openai",
                )
                orch.set_enabled_tools({"read"})
                agent = Agent.from_orchestrator(orch)
                try:
                    result = await asyncio.wait_for(
                        agent.run(
                            "Read the requested fixture file once, then report READY.",
                            runtime_context_overrides={"max_iterations": 3, "tool_budget": 2},
                        ),
                        timeout=30,
                    )
                    assert "READY" in result.content
                finally:
                    await agent.close()
                assert len(records) == 2, "positive agent loop must not replay inference"
                issued_count = len(issued)
                assert issued_count == 2
                hashes = [
                    hashlib.sha256(item["access_token"].encode()).hexdigest() for item in issued
                ]
                assert hashes[0] != hashes[1]
                assert admitted_hashes == hashes
                receipts = [identity.receipt(digest) for digest in hashes]
                assert all(len(receipt) == 1 for receipt in receipts)
                probe = SandhiOpenAIProvider(gateway=gateway)

                async def denied(envelope, *, local_expiry=False):
                    _publish(token_file, envelope)
                    before = len(records)
                    if not local_expiry:
                        status, _ = _http(
                            endpoint,
                            "/v1/chat/completions",
                            envelope["access_token"],
                            {
                                "model": "fixture",
                                "messages": [{"role": "user", "content": "auth control"}],
                            },
                        )
                        assert status == 401
                    with pytest.raises(ProviderError) as failure:
                        await probe.chat(
                            [Message(role="user", content="denial control")],
                            model="fixture",
                            max_tokens=16,
                        )
                    if not local_expiry:
                        assert failure.value.status_code == 401
                    assert len(records) == before, "denied credential reached inference or replayed"

                try:
                    baseline = identity.issue_resource()
                    for substitute in (identity.userinfo_token, identity.id_token):
                        await denied({**baseline, "access_token": substitute})
                    other = identity.issue_other_resource()
                    other_active = identity.introspect(other["access_token"])
                    assert other_active[0] == 200 and other_active[1]["active"] is True
                    assert other_active[1]["aud"] != RESOURCE
                    await denied({**other, "audience": RESOURCE})
                    await denied({**baseline, "expires_on": time.time() - 1}, local_expiry=True)
                    # Expire a genuinely issued resource JWT without altering either clock.
                    while time.time() <= baseline["expires_on"] + 1:
                        await asyncio.sleep(min(1, baseline["expires_on"] + 1 - time.time()))
                    status, _ = _http(
                        endpoint,
                        "/v1/chat/completions",
                        baseline["access_token"],
                        {"model": "fixture", "messages": [{"role": "user", "content": "expired"}]},
                    )
                    assert status == 401
                    await denied(baseline, local_expiry=True)
                    renewed = identity.issue_resource()
                    identity.logout()
                    await denied(renewed)
                    revoked = identity.introspect(renewed["access_token"])
                    assert revoked == (200, {"active": False})
                finally:
                    await probe.close()
                with sqlite3.connect(database) as connection:
                    source_rows = connection.execute(
                        "SELECT o.execution_id,c.request_id,o.usage_json FROM budget_terminal_observation o "
                        "JOIN budget_request_correlation c USING(execution_id)"
                    ).fetchall()
                    observations = [json.loads(row[2]) for row in source_rows]
                    events = connection.execute(
                        "SELECT request_id,subject_id,session_id,run_id,step_id FROM usage_events"
                    ).fetchall()
                    settlements = connection.execute(
                        "SELECT charged_tokens FROM budget_settlement_outbox"
                    ).fetchall()
                    attempts = connection.execute(
                        "SELECT count(*) FROM budget_physical_attempt"
                    ).fetchone()[0]
                assert len(observations) == 2 and attempts == 2 and settlements == [(14,), (14,)]
                assert all(
                    (
                        u["tokens_in"],
                        u["cache_read_tokens"],
                        u["cache_creation_tokens"],
                        u["tokens_out"],
                    )
                    == (2, 6, 2, 4)
                    for u in observations
                )
                assert len(events) == 2 and all(event[1] == identity.subject for event in events)
                assert {event[0] for event in events} == {row[1] for row in source_rows}
                joins = []
                for index, digest in enumerate(hashes):
                    origin_id = f"origin-{index+1}"
                    matching = [
                        row
                        for row in source_rows
                        if json.loads(row[2]).get("upstream_request_id") == origin_id
                    ]
                    assert len(matching) == 1
                    joins.append(
                        {
                            "token_sha256": digest,
                            "idm_receipt": receipts[index][0],
                            "origin_request_id": origin_id,
                            "execution_id": matching[0][0],
                            "sandhi_request_id": matching[0][1],
                        }
                    )
                evidence.update(
                    {
                        "resource": RESOURCE,
                        "sandhi_binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                        "sdk_version": importlib.metadata.version("sandhi-gateway"),
                        "idm_issuance_token_hashes": hashes,
                        "idm_receipts": receipts,
                        "source_observations": observations,
                        "issuance_usage_joins": joins,
                        "join_basis": "actual provider credential hash at each origin dispatch, origin request-id, Sandhi durable correlation and exact IDM issuance-hash receipt",
                        "usage_events": events,
                        "charged_tokens": [14, 14],
                        "physical_attempts": attempts,
                        "denials": [
                            "userinfo",
                            "id_token",
                            "wrong_resource",
                            "private_file_expiry",
                            "signed_jwt_expiry",
                            "logout",
                        ],
                    }
                )
            finally:
                origin.shutdown()
                origin.server_close()
                thread.join(timeout=5)
    output = Path(os.environ.get("VICTOR_IDM_EVIDENCE_FILE", tmp_path / "evidence.json"))
    output.write_text(json.dumps(evidence, indent=2) + "\n")
    output.chmod(0o600)

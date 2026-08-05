# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
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

"""FEP-0020 Phase 3 — reverse-proxy serve mode.

A transparent, OpenAI-compatible **reverse proxy** for teams that share one
upstream provider key. The problem it solves (FEP-0020): a team hands out a
single provider API key, so there is no way to see who spent what. Here, each
user gets a **virtual key** (a bearer token); the proxy resolves it to a
subject/team, enforces a token **budget**, forwards the raw request to the real
upstream holding the shared key, streams the response back **byte-for-byte**, and
**meters** usage per subject via the ``sandhi`` gateway — one neutral usage event
per call (AnvaiOps ADR-0047 / Victor ADR-018).

Victor owns only the HTTP ingress + forwarding here; the metering *mechanism*
(usage parsing, the neutral event, the budget ledger) lives in the open-core
``sandhi`` gateway and is reused via its Python binding — no reimplementation.

Requires ``victor[gateway]`` (``sandhi-gateway`` for metering + ``fastapi`` /
``uvicorn`` for serving). The module itself imports only ``httpx`` / ``pydantic``
(both core), so ``import victor`` is unaffected; ``fastapi`` is imported lazily in
:func:`build_gateway_app`.
"""

import json
import logging
import secrets
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, Field, SecretStr

from victor.observability.sandhi_meter import sandhi_available

logger = logging.getLogger(__name__)


class GatewayVirtualKey(BaseModel):
    """One per-user virtual key: the bearer token clients present, the subject/team
    it attributes to, and the real upstream it fans out to."""

    key_id: str  # stable, non-secret id used for metering/attribution + budget scope
    token: SecretStr  # the bearer secret a client sends in `Authorization: Bearer …`
    subject_id: str  # the user
    group_id: Optional[str] = None  # the team (budgets pool at the group when set)
    provider: str = "openai"  # provider slug → selects the sandhi usage parser
    upstream_base_url: str  # real upstream, e.g. https://api.openai.com/v1
    upstream_api_key: SecretStr  # the shared real key the proxy forwards with
    budget_tokens: Optional[int] = None  # cap in neutral tokens; None ⇒ unlimited

    def scope(self) -> str:
        """Budget scope — pools at the team when grouped, else per virtual key.

        Mirrors the sandhi core convention (``group:{g}`` / ``vk:{id}``) so
        ``set_budget`` / ``check_budget`` / ``spent`` line up with what metering
        records.
        """
        return f"group:{self.group_id}" if self.group_id else f"vk:{self.key_id}"


class GatewayConfig(BaseModel):
    """Serve-mode configuration (loaded from a JSON file by ``victor gateway serve``)."""

    host: str = "127.0.0.1"
    port: int = 8600
    sink_path: Optional[str] = None  # JSONL usage-event persistence; None ⇒ in-memory
    virtual_keys: List[GatewayVirtualKey] = Field(default_factory=list)


def _usage_from_data_line(line: str) -> Optional[Dict[str, int]]:
    """Extract an OpenAI ``usage`` object from a single SSE ``data:`` line.

    Returns ``None`` for non-data lines, the ``[DONE]`` sentinel, unparseable
    JSON, or a chunk without a ``usage`` block. Pure + incremental so the stream
    tee stays O(1) memory (only the current line is inspected).
    """
    line = line.strip()
    if not line.startswith("data:"):
        return None
    payload = line[len("data:") :].strip()
    if not payload or payload == "[DONE]":
        return None
    try:
        obj = json.loads(payload)
    except ValueError:
        return None
    usage = obj.get("usage")
    return usage if isinstance(usage, dict) else None


class GatewayRuntime:
    """Holds the sandhi gateway + the token→virtual-key index; the proxy's brain.

    All metering/budget state lives in the sandhi ``Gateway``; this class only
    resolves tokens and translates between Victor's config and the sandhi API.
    """

    def __init__(self, config: GatewayConfig) -> None:
        if not sandhi_available():
            raise RuntimeError(
                "sandhi-gateway is not installed; run `pip install 'victor-ai[gateway]'` "
                "to enable the usage-attribution reverse proxy (FEP-0020 Phase 3)."
            )
        import sandhi_gateway as sg

        self._gw = sg.Gateway(config.sink_path) if config.sink_path else sg.Gateway()
        self._by_token: Dict[str, GatewayVirtualKey] = {}
        for vk in config.virtual_keys:
            # upstream="" — the real key lives in Victor's config for forwarding; sandhi
            # only meters, so we don't copy the secret into its store.
            self._gw.add_virtual_key(vk.key_id, vk.subject_id, vk.group_id, "")
            if vk.budget_tokens is not None:
                self._gw.set_budget(vk.scope(), vk.budget_tokens)
            self._by_token[vk.token.get_secret_value()] = vk

    def resolve(self, token: str) -> Optional[GatewayVirtualKey]:
        """Constant-time-ish bearer-token lookup (compare_digest per candidate)."""
        for candidate, vk in self._by_token.items():
            if secrets.compare_digest(candidate, token):
                return vk
        return None

    def within_budget(self, vk: GatewayVirtualKey) -> bool:
        """Pre-call gate: block once spend has reached the cap (probe with 1 token)."""
        return bool(self._gw.check_budget(vk.scope(), 1))

    def meter_response(
        self,
        vk: GatewayVirtualKey,
        model: str,
        body_text: str,
        session_id: Optional[str] = None,
        route: Optional[str] = None,
    ) -> None:
        """Meter a non-streamed response body (parses usage + records budget). Never raises."""
        try:
            self._gw.meter(vk.key_id, vk.provider, model, body_text, session_id, route)
        except Exception as exc:  # pragma: no cover - defensive, off critical path
            logger.debug("gateway meter failed (ignored): %s", exc)

    def meter_tokens(
        self,
        vk: GatewayVirtualKey,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cache_creation_tokens: int = 0,
        cache_read_tokens: int = 0,
        session_id: Optional[str] = None,
    ) -> None:
        """Meter explicit token counts (streaming path). Records budget. Never raises."""
        try:
            self._gw.meter_tokens(
                vk.key_id,
                vk.provider,
                model,
                int(prompt_tokens),
                int(completion_tokens),
                int(cache_creation_tokens),
                int(cache_read_tokens),
                session_id,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("gateway meter_tokens failed (ignored): %s", exc)

    def spent(self, scope: str) -> int:
        try:
            return int(self._gw.spent(scope))
        except Exception:  # pragma: no cover - defensive
            return 0

    def events(self) -> List[Dict[str, Any]]:
        try:
            return list(self._gw.events())
        except Exception:  # pragma: no cover - defensive
            return []


def build_gateway_app(config: GatewayConfig) -> Any:
    """Build the FastAPI reverse-proxy app. ``fastapi`` is imported here (lazy)."""
    from fastapi import FastAPI, HTTPException, Request, Response
    from fastapi.responses import JSONResponse, StreamingResponse

    runtime = GatewayRuntime(config)
    app = FastAPI(
        title="Victor Usage Gateway",
        description="FEP-0020 reverse proxy — per-user attribution + budgets over a shared key.",
    )
    app.state.runtime = runtime  # exposed for tests / introspection

    def _authenticate(request: Request) -> GatewayVirtualKey:
        header = request.headers.get("authorization", "")
        token = header[len("bearer ") :].strip() if header.lower().startswith("bearer ") else ""
        vk = runtime.resolve(token) if token else None
        if vk is None:
            raise HTTPException(status_code=401, detail="invalid or missing virtual key")
        return vk

    @app.get("/healthz")
    async def healthz() -> Dict[str, Any]:
        return {"status": "ok", "virtual_keys": len(config.virtual_keys)}

    @app.get("/gateway/keys")
    async def list_keys() -> List[Dict[str, Any]]:
        # Never leaks the bearer token or the upstream key.
        return [
            {
                "key_id": vk.key_id,
                "subject_id": vk.subject_id,
                "group_id": vk.group_id,
                "provider": vk.provider,
                "budget_tokens": vk.budget_tokens,
                "spent_tokens": runtime.spent(vk.scope()),
            }
            for vk in config.virtual_keys
        ]

    @app.get("/gateway/usage")
    async def usage() -> Dict[str, Any]:
        events = runtime.events()
        by_subject: Dict[str, int] = {}
        by_group: Dict[str, int] = {}
        for ev in events:
            billable = int(ev.get("tokens_in", 0)) + int(ev.get("tokens_out", 0))
            subject = ev.get("subject_id") or "anonymous"
            by_subject[subject] = by_subject.get(subject, 0) + billable
            group = ev.get("group_id")
            if group:
                by_group[group] = by_group.get(group, 0) + billable
        return {
            "event_count": len(events),
            "by_subject": by_subject,
            "by_group": by_group,
            "events": events,
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Any:
        vk = _authenticate(request)
        if not runtime.within_budget(vk):
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "message": f"token budget exceeded for scope {vk.scope()}",
                        "type": "budget_exceeded",
                    }
                },
            )

        body = await request.body()
        try:
            payload = json.loads(body) if body else {}
        except ValueError:
            payload = {}
        model = str(payload.get("model", "unknown"))
        is_stream = bool(payload.get("stream"))
        session_id = request.headers.get("x-session-id")
        route = "/v1/chat/completions"
        upstream_url = vk.upstream_base_url.rstrip("/") + "/chat/completions"
        headers = {
            "authorization": f"Bearer {vk.upstream_api_key.get_secret_value()}",
            "content-type": "application/json",
        }

        if not is_stream:
            async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
                resp = await client.post(upstream_url, content=body, headers=headers)
            if resp.status_code < 400:
                runtime.meter_response(vk, model, resp.text, session_id, route)
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type=resp.headers.get("content-type", "application/json"),
            )

        async def stream_body() -> Any:
            usage_seen: Dict[str, int] = {}
            pending = bytearray()

            def _scan(chunk: bytes) -> None:
                pending.extend(chunk)
                while b"\n" in pending:
                    raw_line, _, rest = pending.partition(b"\n")
                    del pending[:]
                    pending.extend(rest)
                    parsed = _usage_from_data_line(raw_line.decode("utf-8", "ignore"))
                    if parsed:
                        usage_seen.update(parsed)

            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
                    async with client.stream(
                        "POST", upstream_url, content=body, headers=headers
                    ) as resp:
                        async for chunk in resp.aiter_bytes():
                            yield chunk  # byte-exact pass-through
                            _scan(chunk)
                        tail = _usage_from_data_line(bytes(pending).decode("utf-8", "ignore"))
                        if tail:
                            usage_seen.update(tail)
            finally:
                # A client disconnect raises at ``yield chunk`` (GeneratorExit / the
                # StreamingResponse closing its iterator); a torn upstream raises inside
                # ``aiter_bytes``. Either path used to skip the metering, so a call that
                # streamed real tokens settled zero — the same class of metering hole
                # ADR-0005 D1 closed in sandhi. Settling here, in a finally, captures
                # whatever the tee already parsed. ``meter_tokens`` never raises, so this
                # is safe to run during generator teardown.
                if usage_seen:
                    runtime.meter_tokens(
                        vk,
                        model,
                        usage_seen.get("prompt_tokens", 0),
                        usage_seen.get("completion_tokens", 0),
                        usage_seen.get("cache_creation_input_tokens", 0),
                        usage_seen.get("cache_read_input_tokens", 0),
                        session_id,
                    )

        return StreamingResponse(stream_body(), media_type="text/event-stream")

    return app

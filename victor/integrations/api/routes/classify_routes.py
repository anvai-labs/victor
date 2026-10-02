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

"""FEP-0037: ``POST /v1/classify`` — one structured completion, no agent loop.

The first versioned surface of ``victor serve``. The request/response contract
is frozen by a conformance guard test and additive-only evolution within
``/v1`` (see feps/0037-v1-classify-endpoint.md). Unlike every other route,
``latency_ms`` is present on ALL outcomes — including 422 and 504 — so
consumers never guess whether a failure consumed tokens.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from typing import TYPE_CHECKING, Any, Optional

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from victor.core.model_parameters import ReasoningEffort

from pydantic import BaseModel, ConfigDict, Field

from victor.core.context import bind_attribution
from victor.core.errors import ProviderPolicyError
from victor.observability.request_correlation import request_correlation_id

if TYPE_CHECKING:
    from victor.integrations.api.fastapi_server import VictorFastAPIServer

logger = logging.getLogger(__name__)

router = APIRouter()

# Server-side output schemas. Presets keep consumers from hand-rolling the
# same contract and give the conformance test something frozen to assert.
_PRESETS: dict[str, dict[str, Any]] = {
    "triage.v1": {
        "type": "object",
        "required": ["sensitive", "category", "reason"],
        "properties": {
            "sensitive": {"type": "boolean"},
            "category": {
                "type": "string",
                # NOTE: no "confidential" token here — the remote InferFlux
                # content filter 400s requests containing it (schema text is
                # part of the prompt), and real user messages may too.
                "description": "one word: money|auth|legal|medical|conflict|private|commitment|none",
            },
            "reason": {"type": "string"},
            "confidence": {"type": "number"},
        },
    },
}

_MAX_INPUT_CHARS = 32_000
_MAX_TOKENS_CAP = 4096

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


class ClassifyRequest(BaseModel):
    """FEP-0037 request. ``schema``/``preset`` are mutually exclusive."""

    model_config = ConfigDict(populate_by_name=True)

    input: str = Field(..., description="Text to classify", max_length=_MAX_INPUT_CHARS)
    output_schema: Optional[dict[str, Any]] = Field(
        default=None, alias="schema", description="JSON Schema the output object must satisfy"
    )
    preset: Optional[str] = Field(default=None, description="Server-side schema preset name")
    instructions: Optional[str] = Field(
        default=None,
        max_length=4000,
        description="Stable per-consumer guidance folded into the system prompt",
    )
    sender: Optional[str] = Field(default=None, max_length=512)
    source: Optional[str] = Field(default=None, max_length=512)
    provider: Optional[str] = Field(
        default=None,
        max_length=128,
        description="Provider override (creates a managed provider)",
    )
    model: Optional[str] = Field(default=None, max_length=256, description="Model override")
    reasoning_effort: Optional[ReasoningEffort] = Field(
        default=None,
        description="Explicit model-supported reasoning effort; unset uses upstream default",
    )
    endpoint: Optional[str] = Field(
        default=None,
        max_length=256,
        description="Provider base_url override (ProviderOverrideConfig semantics)",
    )
    max_tokens: int = Field(default=512, ge=1, le=_MAX_TOKENS_CAP)
    timeout_ms: int = Field(default=120_000, ge=250, le=600_000)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)


class ClassifyResponse(BaseModel):
    result: Optional[dict[str, Any]] = None
    usage: Optional[dict[str, Any]] = None
    model: Optional[str] = None
    latency_ms: float = 0.0
    error: Optional[str] = None
    parse_retries: int = 0


def _new_classify_request_id() -> str:
    return f"cls-{uuid.uuid4().hex[:20]}"


def _extract_json_object(text: str) -> Optional[dict[str, Any]]:
    match = _JSON_RE.search(text or "")
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _schema_conforms(result: dict[str, Any], schema: dict[str, Any]) -> bool:
    """Validate the parsed result against the caller's schema.

    Grammar-constrained decoding is a serving-stack feature (llama.cpp
    grammars); without it the schema is enforced here so a
    prompt-hope-shaped answer cannot pass as structured output.
    """
    try:
        import jsonschema

        jsonschema.validate(instance=result, schema=schema)
        return True
    except ImportError:
        # jsonschema is a hard victor dependency; unreachable in practice.
        logger.warning("jsonschema unavailable — skipping /v1/classify validation")
        return True
    except Exception:
        return False


def _resolve_schema(request: ClassifyRequest) -> dict[str, Any]:
    if request.preset and request.output_schema is not None:
        raise HTTPException(
            status_code=422,
            detail="'schema' and 'preset' are mutually exclusive; supply exactly one",
        )
    if request.preset:
        preset = _PRESETS.get(request.preset)
        if preset is None:
            raise HTTPException(
                status_code=422,
                detail=f"unknown preset {request.preset!r}; available: {sorted(_PRESETS)}",
            )
        return preset
    if request.output_schema is not None:
        if not isinstance(request.output_schema, dict) or not request.output_schema:
            raise HTTPException(status_code=422, detail="schema must be a non-empty object")
        return request.output_schema
    raise HTTPException(status_code=422, detail="one of 'schema' or 'preset' is required")


def _schema_contract_text(schema: dict[str, Any], instructions: str = "") -> str:
    text = (
        "Respond with ONLY a single JSON object conforming to this JSON Schema — "
        f"no prose, no markdown fences:\n{json.dumps(schema)}"
    )
    if instructions.strip():
        text += f"\n\nConsumer instructions (binding unless they conflict with the schema contract):\n{instructions.strip()}"
    return text


def _usage_of(completion: Any) -> Optional[dict[str, Any]]:
    usage = getattr(completion, "usage", None)
    if isinstance(usage, dict):
        return {k: v for k, v in usage.items() if isinstance(v, (int, float, str, bool))}
    return None


def _model_of(completion: Any, fallback: Optional[str]) -> Optional[str]:
    return getattr(completion, "model", None) or fallback


async def _resolve_provider(
    orchestrator: Any, request: ClassifyRequest
) -> tuple[Any, Optional[Any]]:
    """Return (provider, disposable) — disposable is awaited after the call."""
    explicit_base_url = (request.endpoint or "").strip() or None
    if not request.provider and not explicit_base_url:
        provider_manager = getattr(orchestrator, "provider_manager", None)
        provider = getattr(provider_manager, "current_provider", None)
        if provider is not None and hasattr(provider, "chat"):
            return provider, None

    # No live current_provider (the orchestrator bootstraps it lazily on the
    # chat path — classify must not depend on that) and no explicit override:
    # build a managed provider from the configured defaults so /v1/classify
    # works on a freshly started server.
    from victor.config.settings import load_settings
    from victor.providers.factory import ManagedProviderFactory

    settings = load_settings(fresh=True)
    provider_name = (
        request.provider
        or getattr(getattr(settings, "provider", None), "default_provider", None)
        or "inferflux"
    )
    model = request.model or getattr(getattr(settings, "provider", None), "default_model", None)
    provider_settings = settings.get_provider_settings(provider_name)
    # Retain gateway identity and transport policy. Classification owns its
    # single bounded parse retry; transport recovery must not replay inference.
    provider_settings["enable_resilience"] = False
    provider_settings["enable_rate_limiting"] = False
    # Request-level endpoint override (ProviderOverrideConfig semantics) wins
    # over the strategy-derived base_url.
    if explicit_base_url:
        provider_settings["base_url"] = explicit_base_url
    # NOTE: ManagedProviderFactory.create is ASYNC — an un-awaited call yields
    # a coroutine object that 500s at chat time (promotion-review P1).
    provider = await ManagedProviderFactory.create(provider_name, model, **provider_settings)
    return provider, provider


def create_router(server: "VictorFastAPIServer") -> APIRouter:
    """Create the /v1 classify router bound to *server*."""
    router = APIRouter()

    @router.get("/v1/classify/schema", include_in_schema=False)
    async def classify_schema(http_request: Request) -> dict[str, Any]:
        """Expose only the pinned request contract to an authenticated consumer."""
        await server._verify_api_key(http_request)
        return ClassifyRequest.model_json_schema(by_alias=True)

    @router.post("/v1/classify", response_model=ClassifyResponse, tags=["Classify"])
    async def classify(request: ClassifyRequest, response: Response, http_request: Request) -> Any:
        """FEP-0037: one structured completion — no agent loop, no tools, no session.

        Contract is frozen by tests/unit/integrations/api/test_classify_routes.py
        (conformance guard) and additive-only within /v1.
        """
        start = time.perf_counter()
        request_id = _new_classify_request_id()
        response.headers["X-Victor-Request-Id"] = request_id
        client_id = await server._verify_api_key(http_request)

        if not request.input.strip():
            return JSONResponse(
                status_code=422,
                headers={"X-Victor-Request-Id": request_id},
                content=ClassifyResponse(error="input is empty", latency_ms=0.0).model_dump(),
            )
        try:
            schema = _resolve_schema(request)
        except HTTPException as exc:
            return JSONResponse(
                status_code=422,
                headers={"X-Victor-Request-Id": request_id},
                content=ClassifyResponse(error=str(exc.detail), latency_ms=0.0).model_dump(),
            )

        deadline = start + request.timeout_ms / 1000

        def _remaining() -> float:
            return max(0.0, deadline - time.perf_counter())

        # Explicit routing does not bootstrap an agent. One network/inference
        # deadline is shared by resolution and both possible parse attempts.
        try:
            orchestrator = (
                None
                if request.provider and request.model
                else await asyncio.wait_for(server._get_orchestrator(), timeout=_remaining())
            )
            provider, disposable = await asyncio.wait_for(
                _resolve_provider(orchestrator, request), timeout=_remaining()
            )
        except asyncio.TimeoutError:
            return JSONResponse(
                status_code=504,
                headers={"X-Victor-Request-Id": request_id},
                content=ClassifyResponse(
                    error=f"classify exceeded timeout_ms={request.timeout_ms}",
                    model=request.model,
                    latency_ms=(time.perf_counter() - start) * 1000,
                ).model_dump(),
            )
        except HTTPException:
            raise
        except Exception:
            return JSONResponse(
                status_code=503,
                headers={"X-Victor-Request-Id": request_id},
                content=ClassifyResponse(
                    error="provider override unavailable",
                    latency_ms=(time.perf_counter() - start) * 1000,
                ).model_dump(),
            )

        context_lines = ""
        if request.sender:
            context_lines += f"Sender: {request.sender}\n"
        if request.source:
            context_lines += f"Channel: {request.source}\n"
        user_content = f"{context_lines}Input:\n{request.input}"
        messages = [
            {
                "role": "system",
                "content": _schema_contract_text(schema, request.instructions or ""),
            },
            {"role": "user", "content": user_content},
        ]
        model = request.model or getattr(orchestrator, "model", None)
        parse_retries = 0
        usage: Optional[dict[str, Any]] = None
        used_model: Optional[str] = model

        def _latency() -> float:
            return round((time.perf_counter() - start) * 1000, 2)

        async def _finish(status_code: int, payload: ClassifyResponse) -> JSONResponse:
            return JSONResponse(
                status_code=status_code,
                headers={"X-Victor-Request-Id": request_id},
                content=payload.model_dump(),
            )

        try:
            with request_correlation_id(request_id), bind_attribution(subject_id=client_id):
                for attempt in (0, 1):
                    try:
                        completion = await asyncio.wait_for(
                            provider.chat(
                                messages=messages,
                                model=model or "default",
                                temperature=request.temperature,
                                max_tokens=request.max_tokens,
                                **(
                                    {"reasoning_effort": request.reasoning_effort}
                                    if request.reasoning_effort is not None
                                    else {}
                                ),
                            ),
                            timeout=_remaining(),
                        )
                    except asyncio.TimeoutError:
                        return await _finish(
                            504,
                            ClassifyResponse(
                                error=f"classify exceeded timeout_ms={request.timeout_ms}",
                                model=model,
                                latency_ms=_latency(),
                            ),
                        )
                    except ProviderPolicyError as exc:
                        receipt = exc.details.get("policy_receipt")
                        headers = {"X-Victor-Request-Id": request_id}
                        if isinstance(receipt, str) and re.fullmatch(r"[0-9a-f]{32}", receipt):
                            headers["X-Sandhi-Policy-Receipt"] = receipt
                        return JSONResponse(
                            status_code=403,
                            headers=headers,
                            content=ClassifyResponse(
                                error="Request stopped by gateway policy",
                                model=model,
                                latency_ms=_latency(),
                            ).model_dump(),
                        )
                    except Exception as exc:
                        # Non-timeout provider failures (upstream auth, rate
                        # limits, connection errors) must carry the same
                        # response shape — a bare 500 hides whether tokens
                        # were consumed, violating the FEP-0037 contract.
                        # Content-policy rejections are PERMANENT (retrying
                        # the same input re-trips the filter) — 422 signals
                        # consumers to route to human review instead of
                        # looping; the message stays sanitized either way.
                        permanent = "blocked content keyword" in str(exc).lower()
                        return await _finish(
                            422 if permanent else 502,
                            ClassifyResponse(
                                error="provider call failed",
                                model=model,
                                latency_ms=_latency(),
                            ),
                        )
                    usage = _usage_of(completion)
                    used_model = _model_of(completion, model)
                    content = getattr(completion, "content", "") or ""
                    parsed = _extract_json_object(content)
                    if parsed is not None and _schema_conforms(parsed, schema):
                        return ClassifyResponse(
                            result=parsed,
                            usage=usage,
                            model=used_model,
                            latency_ms=_latency(),
                            parse_retries=parse_retries,
                        )
                    parse_retries += 1
                    if attempt == 0:
                        messages = messages + [
                            {"role": "assistant", "content": content[-2000:]},
                            {
                                "role": "user",
                                "content": (
                                    "That was not a valid JSON object matching the schema. "
                                    "Respond again with ONLY the JSON object."
                                ),
                            },
                        ]
                        continue
                return await _finish(
                    422,
                    ClassifyResponse(
                        error="model did not return a schema-conformant JSON object after retry",
                        usage=usage,
                        model=used_model,
                        latency_ms=_latency(),
                        parse_retries=parse_retries,
                    ),
                )
        finally:
            if disposable is not None:
                close = getattr(disposable, "shutdown", None) or getattr(disposable, "close", None)
                if close is not None:
                    try:
                        result = close()
                        if hasattr(result, "__await__"):
                            # A separate bounded cleanup allowance also applies
                            # after timeout/cancellation; never leave queue workers.
                            await asyncio.wait_for(result, timeout=1.0)
                    except Exception:
                        logger.debug("managed provider cleanup failed")

    return router

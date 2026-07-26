"""Sandhi-backed provider execution for Victor.

Sandhi is the provider/wire boundary. Victor constructs prompts and tools, submits the
versioned neutral chat contract over the in-process binding, and consumes neutral responses
and stream events directly. There is deliberately no raw provider-JSON FFI, SSE re-encoding,
demotion state, or replay on a second transport.
"""

from __future__ import annotations

import asyncio
import json
import logging
from json import JSONDecodeError
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Type

logger = logging.getLogger(__name__)

from victor.providers.anthropic_provider import AnthropicProvider
from victor.providers.base import (
    BaseProvider,
    CompletionResponse,
    Message,
    ProviderAuthError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    StreamChunk,
    ToolDefinition,
)
from victor.providers.httpx_openai_compat import HttpxOpenAICompatProvider
from victor.providers.google_provider import GoogleProvider
from victor.providers.llamacpp_provider import LlamaCppProvider
from victor.providers.lmstudio_provider import LMStudioProvider
from victor.providers.ollama_provider import OllamaProvider
from victor.providers.openai_provider import OpenAIProvider
from victor.providers.vllm_provider import VLLMProvider
from victor.providers.openai_compat import build_openai_messages, convert_tools_to_openai_format
from victor.providers.usage_parsing import usage_dict_from_neutral

try:
    import sandhi_gateway as _sg  # type: ignore[import-untyped]
except Exception:  # pragma: no cover - diagnosed at provider construction
    _sg = None

# sandhi-gateway >= 0.1.3 raises provider-boundary errors as a dedicated class
# (message = serialized ProviderErrorV1). With it, provider-vs-binding
# classification no longer depends on the message parsing as JSON.
_SANDHI_PROVIDER_ERROR_CLS = getattr(_sg, "SandhiProviderError", None) if _sg else None


# These are deliberately outside TD-0002's admitted typed set because they use a
# different protocol or execution model. Keep the aliases synchronized with the
# registry. Every other Victor-owned provider must resolve to a Sandhi transport.
VICTOR_NATIVE_ONLY_PROVIDER_ALIASES = frozenset(
    {
        "applesilicon",
        "aws",
        "azure",
        "azure-openai",
        "bedrock",
        "hf",
        "huggingface",
        "mlx",
        "mlx-lm",
        "replicate",
        "vertex",
        "vertexai",
    }
)


def sandhi_transport_available() -> bool:
    return _sg is not None and hasattr(_sg, "ProviderRuntime")


def resolve_transport_class(
    name: str, native_cls: Type[BaseProvider], kwargs: Dict[str, Any]
) -> Type[BaseProvider]:
    """Return the Sandhi consumer for every admitted provider family.

    Providers outside the admitted migration set retain their existing implementation. A
    migrated provider never silently falls back: a missing binding is an installation error,
    because replaying after an FFI failure can duplicate a billed/tool-producing request.
    """
    normalized_name = name.lower()
    if normalized_name in VICTOR_NATIVE_ONLY_PROVIDER_ALIASES:
        return native_cls
    if issubclass(native_cls, SandhiTypedProviderMixin):
        variant = native_cls
    else:
        variant = _SANDHI_VARIANTS.get(native_cls)
    if variant is None and issubclass(native_cls, HttpxOpenAICompatProvider):
        variant = _dynamic_httpx_variant(native_cls)
    if variant is None and native_cls.__module__.startswith("victor.providers."):
        raise ProviderConnectionError(
            f"Victor provider {name!r} is not classified as Sandhi-typed or native-only",
            provider=name,
        )
    if variant is None:
        return native_cls
    if not sandhi_transport_available():
        raise ProviderConnectionError(
            "sandhi-gateway 0.1.4 is required for provider transport",
            provider=name,
        )
    return variant


def _typed_error_payload(message: str) -> Optional[Dict[str, Any]]:
    try:
        value = json.loads(message)
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) and isinstance(value.get("code"), str) else None


# The ChatRequestV1/ChatStreamEventV1 wire-contract version Victor speaks.
# Single source for the request "schema_version" field and the handshake below.
EXPECTED_WIRE_CONTRACT = "1"

# Additive rounds of the v1 contract Victor knows how to consume (W3c):
# 3 = wire-truth latency on UsageV2 (sandhi#97), which _latency_fields reads.
# The installed runtime's minor is read once by the handshake below; bindings
# predating chat_contract_minor() report 0 — their documents simply never
# carry the newer fields, which every consumer tolerates by construction.
KNOWN_CONTRACT_MINOR = 3

_wire_contract_checked = False
_installed_contract_minor: int = 0


def installed_chat_contract_minor() -> int:
    """The additive contract round the installed sandhi binding speaks (0 if
    the binding predates the export). Populated by the one-time handshake."""
    _verify_wire_contract()
    return _installed_contract_minor


def _verify_wire_contract() -> None:
    """One-time fail-soft handshake against the installed sandhi binding.

    Sandhi exposes ``wire_contract_version()`` precisely so consumers can
    detect contract drift, but Victor previously hard-coded ``"1"`` and never
    checked — a future contract bump would surface as silent shape drift
    (fields ignored, events unrecognized) instead of one clear warning.
    Feature-detected so bindings predating the surface stay supported.
    """
    global _wire_contract_checked, _installed_contract_minor
    if _wire_contract_checked:
        return
    _wire_contract_checked = True
    try:
        import sandhi_gateway as sg

        minor_fn = getattr(sg, "chat_contract_minor", None)
        if callable(minor_fn):
            _installed_contract_minor = int(minor_fn())
            if _installed_contract_minor > KNOWN_CONTRACT_MINOR:
                logger.info(
                    "sandhi contract minor %d is ahead of victor's known minor %d — "
                    "newer additive fields will be ignored until victor catches up",
                    _installed_contract_minor,
                    KNOWN_CONTRACT_MINOR,
                )
        if not hasattr(sg, "wire_contract_version"):
            return
        actual = str(sg.wire_contract_version())
    except Exception:  # pragma: no cover - handshake must never block transport
        return
    if actual != EXPECTED_WIRE_CONTRACT:
        logger.warning(
            "sandhi wire-contract mismatch: installed binding speaks version %r, "
            "victor speaks %r — typed request/event shapes may drift silently; "
            "align sandhi-gateway and victor versions",
            actual,
            EXPECTED_WIRE_CONTRACT,
        )


def map_sandhi_error(exc: BaseException, provider_name: str, timeout: float) -> ProviderError:
    """Map `ProviderErrorV1` from the FFI without changing retry ownership."""
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return ProviderTimeoutError(
            f"sandhi transport timed out: {exc}", provider=provider_name, timeout=timeout
        )
    typed = _typed_error_payload(str(exc))
    if typed is None:
        if _SANDHI_PROVIDER_ERROR_CLS is not None and isinstance(exc, _SANDHI_PROVIDER_ERROR_CLS):
            # The typed class is authoritative: this IS a provider error even if
            # the payload failed to parse — never misfile it as a binding failure
            # (retry ownership differs).
            return ProviderError(str(exc), provider=provider_name, raw_error=exc)
        return ProviderConnectionError(
            f"sandhi binding failure: {exc}", provider=provider_name, raw_error=exc
        )
    detail = str(typed.get("message") or exc)
    # Sandhi carries the full (capped) upstream error body in details["upstream_body"]
    # (the message holds only a short display snippet). Append it so provider
    # rejections are self-explaining at the surfaced-error level.
    details = typed.get("details")
    if isinstance(details, dict):
        upstream_body = details.get("upstream_body")
        if isinstance(upstream_body, str) and upstream_body and upstream_body[:80] not in detail:
            detail = f"{detail} | upstream body: {upstream_body[:500]}"
    code = typed["code"]
    status = typed.get("http_status")
    if code == "rate_limited":
        return ProviderRateLimitError(detail, provider=provider_name, status_code=429)
    if code == "authentication_error":
        return ProviderAuthError(detail, provider=provider_name, status_code=int(status or 401))
    if code == "timeout":
        return ProviderTimeoutError(detail, provider=provider_name, timeout=timeout)
    if code in {"circuit_open", "transport_error"}:
        return ProviderConnectionError(detail, provider=provider_name, raw_error=exc)
    return ProviderError(
        detail,
        provider=provider_name,
        status_code=(
            int(status) if status is not None else (400 if code == "invalid_request" else None)
        ),
        raw_error=exc,
    )


def _canonical_content(content: Any) -> Any:
    if not isinstance(content, list):
        return "" if content is None else content
    parts: List[Dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        kind = part.get("type")
        if kind == "image_url" and isinstance(part.get("image_url"), dict):
            image = part["image_url"]
            value: Dict[str, Any] = {
                "type": "image_url",
                "image_url": image.get("url", ""),
            }
            if image.get("detail"):
                value["detail"] = image["detail"]
            parts.append(value)
        elif kind == "input_audio" and isinstance(part.get("input_audio"), dict):
            parts.append({"type": "input_audio", **part["input_audio"]})
        elif kind == "file" and isinstance(part.get("file"), dict):
            parts.append({"type": "file", **part["file"]})
        else:
            parts.append(dict(part))
    return parts


def _include_native_response() -> bool:
    """Whether typed requests ask sandhi for the native-body echo (debug-only)."""
    try:
        from victor.config.settings import get_settings

        return bool(get_settings().provider.sandhi_include_native_response)
    except Exception:
        return False  # neutral-only is the safe default


def _typed_request_from_openai_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Translate Victor's normalized prompt into `ChatRequestV1`."""
    messages: List[Dict[str, Any]] = []
    for source in payload.get("messages", []):
        message = dict(source)
        if message.get("role") == "assistant" and message.get("content") is None:
            message.pop("content", None)
        else:
            message["content"] = _canonical_content(message.get("content"))
        if message.get("role") == "assistant" and message.get("tool_calls"):
            message["tool_calls"] = [
                {
                    "id": call.get("id", ""),
                    "name": (call.get("function") or {}).get("name", ""),
                    "arguments": (call.get("function") or {}).get("arguments", ""),
                }
                for call in message["tool_calls"]
            ]
        messages.append(message)

    request: Dict[str, Any] = {
        "schema_version": EXPECTED_WIRE_CONTRACT,
        "model": str(payload.get("model", "")),
        "messages": messages,
    }
    if not _include_native_response():
        # W3a soak (sandhi#90): opt out of the native-body echo — Victor is
        # neutral-contract-only since F1 (#665). Older sandhi runtimes ignore
        # the unknown field and keep including the body, which F1 tolerates.
        request["include_native_response"] = False
    tools = payload.get("tools")
    if isinstance(tools, list):
        request["tools"] = [
            dict(tool.get("function") or {})
            for tool in tools
            if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
        ]
    choice = payload.get("tool_choice")
    if isinstance(choice, str):
        request["tool_choice"] = choice
    elif isinstance(choice, dict):
        name = (choice.get("function") or {}).get("name")
        if name:
            request["tool_choice"] = {"name": name}
    for source, target in (
        ("temperature", "temperature"),
        ("max_tokens", "max_output_tokens"),
        ("max_completion_tokens", "max_output_tokens"),
        ("response_format", "response_format"),
        ("seed", "seed"),
    ):
        if source in payload:
            request[target] = payload[source]
    if "stop" in payload:
        stop = payload["stop"]
        request["stop"] = stop if isinstance(stop, list) else [stop]
    reserved = {
        "model",
        "messages",
        "tools",
        "tool_choice",
        "temperature",
        "max_tokens",
        "max_completion_tokens",
        "response_format",
        "seed",
        "stop",
        "stream",
        "stream_options",
    }
    native = {key: value for key, value in payload.items() if key not in reserved}
    if native:
        request["extensions"] = {"openai": native}
    return request


def _text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def _tool_calls(calls: Any) -> Optional[List[Dict[str, Any]]]:
    if not isinstance(calls, list) or not calls:
        return None
    result: List[Dict[str, Any]] = []
    for call in calls:
        arguments = call.get("arguments", "{}")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except JSONDecodeError:
                pass
        result.append({"id": call.get("id"), "name": call.get("name"), "arguments": arguments})
    return result


def _native_only_usage(raw_usage: Any) -> Dict[str, int]:
    """Extract usage fields that have no neutral home yet, once, at the boundary.

    ``prompt_cache_miss_tokens`` (DeepSeek-style cache accounting) and
    ``cost_in_usd_ticks`` (provider-reported cost) exist only in native bodies.
    Surfacing them into ``metadata["sandhi_usage"]`` here keeps the runtime off
    the native body entirely (foundations strategy F1 — the body becomes
    debug-only, unblocking sandhi's G8 native-body gating). Long term these
    belong in the neutral contract / Victor pricing respectively.
    """
    if not isinstance(raw_usage, dict):
        return {}
    fields: Dict[str, int] = {}
    for source_key, target_key in (
        ("prompt_cache_miss_tokens", "cache_miss_tokens"),
        ("cost_in_usd_ticks", "cost_in_usd_ticks"),
    ):
        try:
            value = int(raw_usage.get(source_key, 0) or 0)
        except (TypeError, ValueError):
            value = 0
        if value:
            fields[target_key] = value
    return fields


def _latency_fields(usage: Any) -> Dict[str, int]:
    """Wire-truth latency measured at sandhi's typed boundary (W3b).

    Present from sandhi-gateway > 0.1.4; tolerant-absent before that. Carried
    on every run (unlike the non-routine diagnostics) so stream metrics can
    prefer wire truth over client wall-clock.
    """
    if not isinstance(usage, dict):
        return {}
    fields: Dict[str, int] = {}
    for key in ("duration_ms", "time_to_first_token_ms"):
        value = usage.get(key)
        if isinstance(value, (int, float)) and value >= 0:
            fields[key] = int(value)
    return fields


def _usage_diagnostics(usage: Any) -> Optional[Dict[str, Any]]:
    """Preserve non-routine typed metering state without polluting legacy token keys."""
    if not isinstance(usage, dict):
        return None
    attempts = int(usage.get("attempts", 1) or 1)
    completeness = usage.get("completeness")
    outcome = usage.get("outcome")
    if (
        attempts <= 1
        and completeness not in {"partial", "unavailable"}
        and outcome
        not in {
            "error",
            "cancelled",
        }
    ):
        return None
    return {
        "attempts": attempts,
        "completeness": completeness,
        "outcome": outcome,
        "upstream_request_id": usage.get("upstream_request_id"),
    }


class SandhiTypedProviderMixin:
    """Shared direct consumer of Sandhi's typed FFI contract."""

    _sandhi_runtime: Any = None
    _sandhi_typed_providers: Optional[Dict[Tuple[str, str, str, str, str, str], Any]] = None
    _SANDHI_WAIT_GRACE_SECS = 5.0

    def _sandhi_slug(self) -> str:
        declared = str(getattr(self, "name", "openai"))
        if _sg is not None and hasattr(_sg, "provider_descriptor_json"):
            try:
                descriptor = json.loads(_sg.provider_descriptor_json(declared))
                return str(descriptor.get("slug") or declared)
            except Exception:
                pass
        return declared

    def _sandhi_timeout(self) -> float:
        try:
            return float(getattr(self, "timeout", 120.0) or 120.0)
        except (TypeError, ValueError):
            return 120.0

    async def discover_capabilities(self, model: str) -> Any:
        """Descriptor-backed discovery for Sandhi-routed providers (gap G6).

        Capability facts (tools/streaming) come from Sandhi's typed descriptor —
        the wire truth the transport actually honors — instead of per-provider
        hardcoded flags; limits stay Victor config policy. Falls back to the
        config-only base implementation when the binding or descriptor is
        absent, so behavior is unchanged for native-only installs.
        """
        descriptor = None
        if _sg is not None and hasattr(_sg, "provider_descriptor_json"):
            try:
                descriptor = json.loads(_sg.provider_descriptor_json(self._sandhi_slug()))
            except Exception:
                descriptor = None
        capabilities = descriptor.get("capabilities") if isinstance(descriptor, dict) else None
        if not isinstance(capabilities, dict):
            return await super().discover_capabilities(model)  # type: ignore[misc]

        from victor.config.config_loaders import get_provider_limits
        from victor.providers.runtime_capabilities import ProviderRuntimeCapabilities

        provider_name = str(getattr(self, "name", self._sandhi_slug()))
        limits = get_provider_limits(provider_name, model)
        supports_tools = getattr(self, "supports_tools", lambda: False)
        supports_streaming = getattr(self, "supports_streaming", lambda: False)
        return ProviderRuntimeCapabilities(
            provider=provider_name,
            model=model,
            context_window=limits.context_window,
            supports_tools=bool(capabilities.get("tools", supports_tools())),
            supports_streaming=bool(capabilities.get("streaming", supports_streaming())),
            source="sandhi_descriptor",
            raw=descriptor,
        )

    def _gateway_overrides(self) -> Optional[Tuple[str, str]]:
        """Return ``(proxy_url, virtual_key)`` when gateway mode is configured.

        Gateway mode (TD-0003 P3) points the Sandhi FFI handle at the Sandhi proxy:
        the proxy URL becomes the handle ``base_url`` and the virtual key becomes
        both the presented credential and a ``bearer`` auth_scheme, so traffic is
        centrally attributed and budget-enforced. The provider slug is preserved so
        the proxy still speaks the right dialect and routes to the vault-resolved
        upstream. Returns ``None`` in direct mode (the default).
        """
        raw = getattr(self, "extra_config", None)
        if not isinstance(raw, dict):
            return None
        gateway = raw.get("gateway")
        if not isinstance(gateway, dict):
            return None
        url = str(gateway.get("url") or "").strip()
        if not url:
            return None
        virtual_key = gateway.get("virtual_key")
        # Duck-type SecretStr without importing pydantic into the transport layer.
        if hasattr(virtual_key, "get_secret_value"):
            virtual_key = virtual_key.get_secret_value()
        return (url, str(virtual_key or ""))

    def _typed_provider(self, model: str) -> Any:
        if not sandhi_transport_available():
            raise ProviderConnectionError(
                "sandhi-gateway 0.1.4 typed runtime is unavailable",
                provider=self._sandhi_slug(),
            )
        _verify_wire_contract()
        if self._sandhi_runtime is None:
            self._sandhi_runtime = _sg.ProviderRuntime()
        if self._sandhi_typed_providers is None:
            self._sandhi_typed_providers = {}
        slug = self._sandhi_slug()
        protocol = str(getattr(self, "_sandhi_protocol", "") or "")
        gateway = self._gateway_overrides()
        if gateway is not None:
            # Gateway mode (TD-0003 P3): the FFI handle targets the Sandhi proxy with
            # the virtual key presented as a bearer token. The slug is preserved so the
            # proxy still speaks the right dialect; no raw HTTP/SSE transport is used.
            proxy_url, virtual_key = gateway
            if not virtual_key:
                raise ProviderConnectionError(
                    f"sandhi gateway mode for provider {slug!r} requires a non-empty "
                    "virtual_key (set providers.<name>.gateway.virtual_key or the "
                    "SANDHI_GATEWAY_VIRTUAL_KEY env var)",
                    provider=slug,
                )
            base_url = proxy_url
            api_key = virtual_key
            # Compat guard for sandhi <= 0.1.4: those bindings REJECT an
            # explicit auth_scheme outside the Anthropic/Gemini protocols
            # (victor#678). sandhi >= 0.1.5 accepts "bearer" family-wide as a
            # no-op (TD-0008 rule 5: reject contradictions, accept redundancy),
            # so once the pin passes 0.1.4 this conditional can collapse to an
            # unconditional "bearer". The real-binding construction conformance
            # suite (test_sandhi_binding_construction.py) holds either way.
            auth_scheme = "bearer" if getattr(self, "_sandhi_auth_scheme", "") else ""
            explicit_base_url = proxy_url
        else:
            base_url = str(getattr(self, "base_url", "") or "")
            # A catalog default is not an override. Omitting it lets Sandhi apply authoritative
            # model-specific routing (notably Moonshot K3's .ai endpoint). Only a genuinely custom
            # endpoint crosses the FFI.
            try:
                catalog_base = str(_sg.provider_spec(slug).get("base_url") or "")
            except Exception:
                catalog_base = ""
            explicit_base_url = (
                base_url if base_url and base_url.rstrip("/") != catalog_base.rstrip("/") else ""
            )
            api_key = str(getattr(self, "_api_key", None) or getattr(self, "api_key", "") or "")
            auth_scheme = str(getattr(self, "_sandhi_auth_scheme", "") or "")
        cache_key = (slug, model, explicit_base_url, api_key, auth_scheme, protocol)
        if cache_key not in self._sandhi_typed_providers:
            kwargs: Dict[str, Any] = {
                "base_url": explicit_base_url or None,
                "timeout_secs": self._sandhi_timeout(),
                "stream_idle_timeout_secs": 90.0,
                "max_retries": max(0, int(getattr(self, "max_retries", 0) or 0)),
            }
            wire_headers = getattr(self, "_wire_headers", None)
            if wire_headers:
                kwargs["headers_json"] = json.dumps(wire_headers)
            if auth_scheme:
                kwargs["auth_scheme"] = auth_scheme
            if protocol:
                kwargs["protocol"] = protocol
            self._sandhi_typed_providers[cache_key] = self._sandhi_runtime.provider(
                slug, model, api_key, **kwargs
            )
        return self._sandhi_typed_providers[cache_key]

    async def _sandhi_complete(self, request: Dict[str, Any]) -> Dict[str, Any]:
        provider = self._typed_provider(str(request.get("model", "")))
        timeout = self._sandhi_timeout()
        try:
            value = await asyncio.wait_for(
                provider.complete_json(json.dumps(request)),
                timeout=timeout + self._SANDHI_WAIT_GRACE_SECS,
            )
            return json.loads(str(value))
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:  # pyo3 panics may subclass BaseException
            raise map_sandhi_error(exc, self._sandhi_slug(), timeout) from exc

    def _completion_from_typed(self, response: Dict[str, Any], model: str) -> CompletionResponse:
        output = response.get("output") or {}
        extensions = response.get("extensions") or {}
        native = extensions.get(self._sandhi_slug())
        if native is None and getattr(self, "_sandhi_protocol", None) in {
            "responses",
            "chatgpt_responses",
        }:
            native = extensions.get("openai_responses")
        if native is None and self._sandhi_slug() not in {
            "anthropic",
            "gemini",
            "cohere",
            "ollama",
        }:
            native = extensions.get("openai")
        reasoning = extensions.get("reasoning")
        if reasoning is None and isinstance(native, dict):
            reasoning = (
                (native.get("choices") or [{}])[0].get("message", {}).get("reasoning_content")
            )
        usage = usage_dict_from_neutral(
            response.get("usage"),
            native.get("usage") if isinstance(native, dict) else None,
            slug="anthropic" if self._sandhi_slug() == "anthropic" else self._sandhi_slug(),
        )
        metadata: Dict[str, Any] = {}
        if reasoning:
            metadata["reasoning_content"] = reasoning
        diagnostics = dict(_usage_diagnostics(response.get("usage")) or {})
        diagnostics.update(_latency_fields(response.get("usage")))
        diagnostics.update(
            _native_only_usage(native.get("usage") if isinstance(native, dict) else None)
        )
        if diagnostics:
            metadata["sandhi_usage"] = diagnostics
        return CompletionResponse(
            content=_text_content(output.get("content")),
            role="assistant",
            tool_calls=_tool_calls(output.get("tool_calls")),
            stop_reason=response.get("finish_reason"),
            usage=usage,
            model=response.get("model") or model,
            # Debug-only: everything load-bearing rides `usage` (neutral) or
            # `metadata["sandhi_usage"]` (boundary-extracted). May be absent
            # once sandhi gates native-body emission (G8).
            raw_response=native if isinstance(native, dict) else response,
            metadata=metadata or None,
        )

    async def _sandhi_stream(self, request: Dict[str, Any]) -> AsyncIterator[StreamChunk]:
        provider = self._typed_provider(str(request.get("model", "")))
        timeout = self._sandhi_timeout()
        calls: Dict[int, Dict[str, Any]] = {}
        finish_reason: Optional[str] = None
        usage: Optional[Dict[str, int]] = None
        usage_diagnostics: Optional[Dict[str, Any]] = None
        try:
            async for event_json in provider.stream_json(json.dumps(request)):
                event = json.loads(str(event_json))
                kind = event.get("event")
                if kind == "text_delta":
                    yield StreamChunk(content=str(event.get("delta", "")))
                elif kind == "reasoning_delta":
                    yield StreamChunk(
                        content="", metadata={"reasoning_content": str(event.get("delta", ""))}
                    )
                elif kind == "refusal_delta":
                    yield StreamChunk(content="", metadata={"refusal": str(event.get("delta", ""))})
                elif kind == "tool_call_start":
                    calls[int(event.get("index", 0))] = {
                        "id": event.get("id"),
                        "name": event.get("name"),
                        "arguments": "",
                    }
                elif kind == "tool_call_arguments_delta":
                    index = int(event.get("index", 0))
                    calls.setdefault(index, {"id": None, "name": "", "arguments": ""})
                    calls[index]["arguments"] += str(event.get("delta", ""))
                elif kind == "finish":
                    finish_reason = str(event.get("reason", "unknown"))
                elif kind == "usage":
                    usage = usage_dict_from_neutral(
                        event.get("usage"), None, slug=self._sandhi_slug()
                    )
                    usage_diagnostics = dict(_usage_diagnostics(event.get("usage")) or {})
                    usage_diagnostics.update(_latency_fields(event.get("usage")))
                    usage_diagnostics = usage_diagnostics or None
                elif kind == "response_start":
                    # Deliberately ignored (TD-0008 consumer-decision row): victor
                    # derives model/id from the request and final chunk.
                    continue
                elif kind == "tool_call_end":
                    # Deliberately ignored: call boundaries are tracked by the
                    # indexed accumulation above; the terminal chunk assembles them.
                    continue
                elif kind == "error":
                    # A typed error EVENT (as opposed to an iterator error) must
                    # surface as the mapped ProviderError, never vanish mid-stream.
                    payload = event.get("error")
                    raise map_sandhi_error(
                        RuntimeError(
                            json.dumps(
                                payload if isinstance(payload, dict) else {"code": "unknown"}
                            )
                        ),
                        self._sandhi_slug(),
                        timeout,
                    )
                else:
                    # Contract-drift alarm (TD-0008 P1): a new ChatStreamEventV1
                    # variant reached a consumer with no consumption decision.
                    logger.warning(
                        "sandhi stream event kind %r has no victor consumption decision "
                        "— contract drift; add a handler or an explicit ignore",
                        kind,
                    )
            yield StreamChunk(
                content="",
                tool_calls=_tool_calls([calls[index] for index in sorted(calls)]),
                stop_reason=finish_reason or "stop",
                is_final=True,
                usage=usage,
                metadata={"sandhi_usage": usage_diagnostics} if usage_diagnostics else None,
            )
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except ProviderError:
            raise
        except BaseException as exc:
            raise map_sandhi_error(exc, self._sandhi_slug(), timeout) from exc


class SandhiHttpxTransportMixin(SandhiTypedProviderMixin):
    """OpenAI-compatible Victor policy hooks backed by Sandhi typed execution."""

    async def _refresh_host_credentials(self) -> None:
        refresh = getattr(self, "_ensure_valid_token", None)
        if callable(refresh):
            await refresh()

    async def chat(
        self,
        messages: List[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        await self._refresh_host_credentials()
        cleaner = getattr(self, "_clean_model_name", None)
        model = cleaner(model) if callable(cleaner) else model
        payload = self._build_request_payload(
            messages, model, temperature, max_tokens, tools, False, **kwargs
        )
        response = await self._sandhi_complete(_typed_request_from_openai_payload(payload))
        return self._completion_from_typed(response, model)

    async def stream(
        self,
        messages: List[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        await self._refresh_host_credentials()
        cleaner = getattr(self, "_clean_model_name", None)
        model = cleaner(model) if callable(cleaner) else model
        payload = self._build_request_payload(
            messages, model, temperature, max_tokens, tools, True, **kwargs
        )
        async for chunk in self._sandhi_stream(_typed_request_from_openai_payload(payload)):
            yield chunk


class SandhiNeutralProviderMixin(SandhiTypedProviderMixin):
    """Build the neutral contract directly for providers with no reusable Victor wire policy."""

    async def _refresh_host_credentials(self) -> None:
        refresh = getattr(self, "_ensure_valid_token", None)
        if callable(refresh):
            await refresh()

    def _neutral_request(
        self,
        messages: List[Message],
        model: str,
        temperature: float,
        max_tokens: int,
        tools: Optional[List[ToolDefinition]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": build_openai_messages(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }
        if tools:
            payload["tools"] = convert_tools_to_openai_format(tools)
            payload.setdefault("tool_choice", "auto")
        request = _typed_request_from_openai_payload(payload)
        slug = self._sandhi_slug()
        if slug not in {"openai"}:
            extensions = request.pop("extensions", {})
            native = extensions.get("openai") if isinstance(extensions, dict) else None
            if native:
                request["extensions"] = {slug: native}
        return request

    async def chat(
        self,
        messages: List[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        await self._refresh_host_credentials()
        request = self._neutral_request(messages, model, temperature, max_tokens, tools, **kwargs)
        return self._completion_from_typed(await self._sandhi_complete(request), model)

    async def stream(
        self,
        messages: List[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        await self._refresh_host_credentials()
        request = self._neutral_request(messages, model, temperature, max_tokens, tools, **kwargs)
        async for chunk in self._sandhi_stream(request):
            yield chunk


class SandhiAnthropicProvider(SandhiTypedProviderMixin, AnthropicProvider):
    """Anthropic prompt policy backed by Sandhi's typed Messages codec and transport."""

    def _anthropic_request(
        self,
        messages: List[Message],
        model: str,
        temperature: float,
        max_tokens: int,
        tools: Optional[List[ToolDefinition]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        native = self._build_request_params(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            **kwargs,
        )
        openai_payload: Dict[str, Any] = {
            "model": model,
            "messages": build_openai_messages(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            openai_payload["tools"] = convert_tools_to_openai_format(tools)
            openai_payload["tool_choice"] = "auto"
        request = _typed_request_from_openai_payload(openai_payload)
        request["extensions"] = {"anthropic": native}
        return request

    async def chat(
        self,
        messages: List[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        await self._ensure_valid_token()
        request = self._anthropic_request(messages, model, temperature, max_tokens, tools, **kwargs)
        return self._completion_from_typed(await self._sandhi_complete(request), model)

    async def stream(
        self,
        messages: List[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        await self._ensure_valid_token()
        request = self._anthropic_request(messages, model, temperature, max_tokens, tools, **kwargs)
        async for chunk in self._sandhi_stream(request):
            yield chunk


class SandhiOpenAIProvider(SandhiNeutralProviderMixin, OpenAIProvider):
    """OpenAI prompt policy with explicit Chat Completions vs Responses selection."""

    def _neutral_request(
        self,
        messages: List[Message],
        model: str,
        temperature: float,
        max_tokens: int,
        tools: Optional[List[ToolDefinition]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        request = super()._neutral_request(
            messages, model, temperature, max_tokens, tools, **kwargs
        )
        if self._is_o_series_model(model):
            request.pop("temperature", None)
        if getattr(self, "_sandhi_protocol", None) in {"responses", "chatgpt_responses"}:
            extensions = request.setdefault("extensions", {})
            native = extensions.pop("openai", {})
            if not isinstance(native, dict):
                native = {}
            effort = native.pop("reasoning_effort", None)
            if effort is not None:
                native["reasoning"] = {"effort": effort}
            extensions["openai_responses"] = native
        return request


class SandhiGoogleProvider(SandhiNeutralProviderMixin, GoogleProvider):
    pass


class SandhiOllamaProvider(SandhiNeutralProviderMixin, OllamaProvider):
    pass


class SandhiLMStudioProvider(SandhiNeutralProviderMixin, LMStudioProvider):
    pass


class SandhiVLLMProvider(SandhiNeutralProviderMixin, VLLMProvider):
    pass


class SandhiLlamaCppProvider(SandhiNeutralProviderMixin, LlamaCppProvider):
    pass


_SANDHI_VARIANTS: Dict[Type[BaseProvider], Type[BaseProvider]] = {
    AnthropicProvider: SandhiAnthropicProvider,
    OpenAIProvider: SandhiOpenAIProvider,
    GoogleProvider: SandhiGoogleProvider,
    OllamaProvider: SandhiOllamaProvider,
    LMStudioProvider: SandhiLMStudioProvider,
    VLLMProvider: SandhiVLLMProvider,
    LlamaCppProvider: SandhiLlamaCppProvider,
}
_DYNAMIC_HTTPX_VARIANTS: Dict[Type[BaseProvider], Type[BaseProvider]] = {}


def _dynamic_httpx_variant(native_cls: Type[BaseProvider]) -> Type[BaseProvider]:
    variant = _DYNAMIC_HTTPX_VARIANTS.get(native_cls)
    if variant is None:
        variant = type(
            f"Sandhi{native_cls.__name__}",
            (SandhiHttpxTransportMixin, native_cls),
            {"__module__": __name__},
        )
        _DYNAMIC_HTTPX_VARIANTS[native_cls] = variant
    return variant


__all__ = [
    "SandhiAnthropicProvider",
    "SandhiHttpxTransportMixin",
    "SandhiNeutralProviderMixin",
    "SandhiOpenAIProvider",
    "SandhiGoogleProvider",
    "SandhiOllamaProvider",
    "SandhiLMStudioProvider",
    "SandhiVLLMProvider",
    "SandhiLlamaCppProvider",
    "SandhiTypedProviderMixin",
    "VICTOR_NATIVE_ONLY_PROVIDER_ALIASES",
    "map_sandhi_error",
    "resolve_transport_class",
    "sandhi_transport_available",
]

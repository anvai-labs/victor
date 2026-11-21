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
import math
import platform
import re
import socket
import ssl

import httpx
from contextlib import aclosing
from json import JSONDecodeError
from collections.abc import Mapping
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Type
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

from victor.core.errors import ErrorCategory, ProviderPolicyError
from victor.providers.gateway_contract import client_protocol, gateway_config, gateway_virtual_key
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
            "sandhi-gateway 0.23.0 is required for provider transport",
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

# Sandhi 0.23 adds a stable tool-prefix hint and exact Ollama thinking toggles
# to the v1 semantic contract. Newer additive minors remain readable.
KNOWN_CONTRACT_MINOR = 2

# Stream-timeout ceilings shared by the transport validators and the
# settings-layer pydantic fields (single source: timeout_profile.py is the
# documented inventory home; these are the enforcement constants).
_SANDHI_STREAM_TIMEOUT_MAX_SECS: float = 600.0
_SANDHI_BODY_TIMEOUT_MAX_SECS: float = 840.0
_SANDHI_BODY_TIMEOUT_MIN_MS = 1

# Internal orchestration keys that must never reach a provider request body
# (W3d/G7 D4). They ride **kwargs from the runtime and would otherwise land in
# extensions["openai"] and cross the wire.
_INTERNAL_KWARG_KEYS = frozenset(
    {
        "execution_mode",
        "provider_hint",
        "escalation_target",
        "topology_action",
        "topology_kind",
        "topology_metadata",
        "task_type",
        "temperature_override",
    }
)

_wire_contract_checked = False
_installed_contract_minor: int = 0


def _normalize_thinking(value: Any) -> Optional[Dict[str, Any]]:
    """Validate Victor's portable thinking intent before sending it to Sandhi."""
    if isinstance(value, bool):
        return {"enabled": value}
    if not isinstance(value, dict) or type(value.get("enabled")) is not bool:
        return None
    if value.keys() - {"enabled", "effort", "budget_tokens"}:
        return None
    effort = value.get("effort")
    budget = value.get("budget_tokens")
    if effort is not None and (not isinstance(effort, str) or not effort):
        return None
    if budget is not None and (type(budget) is not int or budget <= 0):
        return None
    return {key: item for key, item in value.items() if item is not None}


def _validated_thinking_capability(value: Any) -> Dict[str, Any]:
    """Validate the v1 thinking capability shape without inventing defaults."""
    if not isinstance(value, dict) or type(value.get("supported")) is not bool:
        raise ValueError("invalid Sandhi thinking capability")
    effort = value.get("effort")
    if effort is not None:
        values = effort.get("values") if isinstance(effort, dict) else None
        if (
            not isinstance(effort, dict)
            or not isinstance(values, list)
            or not values
            or any(not isinstance(item, str) or not item for item in values)
            or len(set(values)) != len(values)
        ):
            raise ValueError("invalid Sandhi thinking effort capability")
    budget = value.get("budget_tokens")
    if budget is not None:
        minimum = budget.get("min") if isinstance(budget, dict) else None
        maximum = budget.get("max") if isinstance(budget, dict) else None
        if (
            not isinstance(budget, dict)
            or type(minimum) is not int
            or type(maximum) is not int
            or minimum <= 0
            or maximum < minimum
        ):
            raise ValueError("invalid Sandhi thinking budget capability")
    allow_combined = value.get("allow_effort_and_budget")
    can_disable = value.get("can_disable")
    if (allow_combined is not None and type(allow_combined) is not bool) or (
        can_disable is not None and type(can_disable) is not bool
    ):
        raise ValueError("invalid Sandhi thinking capability")
    return value


def _final_thinking_control(kwargs: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Return the semantic thinking control that payload promotion would select."""
    if kwargs.get("thinking") is not None:
        normalized = _normalize_thinking(kwargs["thinking"])
        if normalized is None:
            raise ValueError("thinking must be a boolean or a supported thinking object")
        return normalized
    effort = kwargs.get("reasoning_effort")
    if effort is not None:
        if not isinstance(effort, str) or not effort:
            raise ValueError("reasoning_effort must be a nonempty effort label or None")
        return {"enabled": False} if effort == "none" else {"enabled": True, "effort": effort}
    return None


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


_TLS_PROBE_DEADLINE_SECS = 4.0


def _tls_trust_remediation() -> str:
    """Platform-correct command for importing a CA into the system trust store.

    The remediation differs per OS; a hardcoded macOS command on Linux (or
    anywhere else) sends the operator down a dead end. `<ca.pem>` is the
    gateway's issuing CA certificate, not the endpoint leaf.
    """
    system = platform.system()
    if system == "Darwin":
        return (
            "macOS: sudo security add-trusted-cert -d -r trustRoot -k "
            "/Library/Keychains/System.keychain <ca.pem>"
        )
    if system == "Linux":
        return (
            "Linux: sudo cp <ca.pem> /usr/local/share/ca-certificates/"
            "gateway-ca.crt && sudo update-ca-certificates"
        )
    if system == "Windows":
        return "Windows: certutil -addstore -f ROOT <ca.pem>"
    return f"import the issuing CA into this {system or 'unknown'} platform's trust store"


def _diagnose_endpoint_tls(url: str, connect_timeout: float = 1.5) -> Optional[str]:
    """Classify a transport error against an https endpoint.

    Returns TLS-trust guidance only when the signature matches: TCP connects
    but the default-context handshake fails certificate verification (the
    self-signed/private-CA gateway case). Unreachable endpoints return None —
    the default "check network" hint is then correct — and so do healthy
    handshakes.

    Never raises: every failure mode (unresolvable host, out-of-range port,
    non-OSError resolver errors) classifies as None. Callers must run this
    OFF the event loop (asyncio.to_thread) and cap it (asyncio.wait_for) —
    the socket timeouts bound each operation but DNS resolution is
    unbounded, so the caller owns the hard deadline.
    """
    try:
        parsed = urlparse(url)
        if parsed.scheme != "https" or not parsed.hostname:
            return None
        try:
            port = parsed.port
        except ValueError:
            return None
        if port is None:
            port = 443
        # One connection for both the reachability check and the handshake:
        # create_connection's timeout also bounds the wrap_socket handshake.
        with socket.create_connection((parsed.hostname, port), timeout=connect_timeout) as sock:
            context = ssl.create_default_context()
            try:
                with context.wrap_socket(sock, server_hostname=parsed.hostname):
                    return None
            except ssl.SSLCertVerificationError:
                return (
                    "the endpoint's TLS certificate is not trusted by this "
                    "machine's trust store (private/self-signed CA). Configure "
                    "SSL_CERT_FILE with a trusted CA bundle for the SDK process "
                    "(verify support on the installed SDK/platform), import "
                    f"the issuing CA into the system trust store ({_tls_trust_remediation()}), "
                    "or serve a publicly-trusted certificate"
                )
            except OSError:
                return None
    except Exception:
        # A diagnosis probe must never replace the original transport error
        # with a worse one (unresolvable host, odd URL shapes, resolver
        # errors): no classification is the honest fallback.
        return None


async def map_sandhi_error(
    exc: BaseException, provider_name: str, timeout: float, *, gateway: bool = False
) -> ProviderError:
    """Map `ProviderErrorV1` from the FFI without changing retry ownership.

    Async so the optional TLS diagnosis probe runs off the event loop with a
    hard deadline instead of stalling it while the network is degraded.
    """
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return ProviderTimeoutError(
            f"sandhi transport timed out: {exc}",
            provider=provider_name,
            timeout=timeout,
            recoverable=False,
        )
    typed = _typed_error_payload(str(exc))
    if typed is None:
        if _SANDHI_PROVIDER_ERROR_CLS is not None and isinstance(exc, _SANDHI_PROVIDER_ERROR_CLS):
            # The typed class is authoritative: this IS a provider error even if
            # the payload failed to parse — never misfile it as a binding failure
            # (retry ownership differs).
            return ProviderError(str(exc), provider=provider_name, raw_error=exc, recoverable=False)
        return ProviderConnectionError(
            f"sandhi binding failure: {exc}",
            provider=provider_name,
            raw_error=exc,
            recoverable=False,
        )
    candidates = [typed]
    details = typed.get("details")
    raw = details.get("upstream_body") if isinstance(details, dict) else None
    if isinstance(raw, str) and len(raw) <= 8192:
        try:
            envelope = json.loads(raw)
            if isinstance(envelope, dict) and isinstance(envelope.get("error"), dict):
                candidates.append(envelope["error"])
        except (ValueError, TypeError):
            pass
    for candidate in candidates:
        code = candidate.get("code")
        if isinstance(code, str) and code in {
            "policy_blocked",
            "policy_quarantined",
            "policy_unavailable",
        }:
            receipt = candidate.get("request_id")
            if not isinstance(receipt, str) or not re.fullmatch(r"[0-9a-f]{32}", receipt):
                receipt = None
            status = typed.get("http_status")
            return ProviderPolicyError(
                code,
                provider=provider_name,
                receipt=receipt,
                status_code=status if isinstance(status, int) else None,
            )
    for candidate in candidates:
        # A live gateway can accept the client key while its bound credential
        # was not activated at startup. The transport wraps that terminal 502
        # in a retryable upstream_error; retain the inner setup diagnosis.
        # Match the complete legacy envelope, never arbitrary upstream prose.
        if (
            candidate.get("code") == "invalid_request"
            and candidate.get("http_status") == 502
            and candidate.get("message") == "no upstream registered for this key"
            and candidate.get("retryable") is False
        ):
            hint = (
                "Gateway operator: inspect the key's upstream_ref and startup credential "
                "activation logs. Provision that exact provider/label in the gateway's active "
                "credential backend, then register it with authenticated POST "
                "/admin/keys/reference. For a read-only credstore, provision the service "
                "credential externally and ensure the running service can read it before "
                "registration. "
            )
            if provider_name == "openai":
                hint += (
                    "For a Codex subscription route: run `codex login status` on the login-owner "
                    "machine; run `codex login` if signed out. Restore the bounded access-lease "
                    "publisher and register the same provider/label with scheme=oauth. The "
                    "publisher must refresh the active credential backend and gateway handle; "
                    "a one-time registration expires. "
                )
            hint += (
                "Retry after registration succeeds. Procedure: "
                "docs/architecture/sandhi-subscription-gateway.md#missing-upstream-recovery"
            )
            return ProviderError(
                "Sandhi accepted the client credential, but its configured upstream is not "
                "registered. The request cannot run until gateway setup is repaired.",
                provider=provider_name,
                category=ErrorCategory.CONFIG_MISSING,
                recoverable=False,
                status_code=502,
                recovery_hint=hint,
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
        return ProviderAuthError(
            detail, provider=provider_name, status_code=int(status or 401), recoverable=not gateway
        )
    if code == "timeout":
        # Sandhi may have hit a stream-idle or upstream deadline shorter than
        # Victor's configured overall timeout. Preserve its diagnosis without
        # claiming that the unrelated overall budget actually elapsed.
        return ProviderTimeoutError(
            detail,
            provider=provider_name,
            status_code=status if isinstance(status, int) else None,
            # The binding supplies no authoritative pre-dispatch proof for a
            # timeout. Retrying could duplicate an accepted physical request.
            recoverable=False,
        )
    if code in {"circuit_open", "transport_error"}:
        # "error sending request" wraps both real network failures and TLS
        # trust failures; the default recovery hint is wrong for the latter.
        # Classify once, bounded, so a self-signed/private-CA gateway tells
        # the operator what actually happened.
        if code == "transport_error":
            url_match = re.search(r"https?://[^\s')]+", str(exc))
            if url_match:
                try:
                    diagnosis = await asyncio.wait_for(
                        asyncio.to_thread(_diagnose_endpoint_tls, url_match.group(0)),
                        timeout=_TLS_PROBE_DEADLINE_SECS,
                    )
                except Exception:
                    # The probe is best-effort; never delay or replace the
                    # surfaced transport error with a diagnosis failure.
                    diagnosis = None
                if diagnosis:
                    detail = f"{detail} | TLS diagnosis: {diagnosis}"
        return ProviderConnectionError(
            detail,
            provider=provider_name,
            raw_error=exc,
            recoverable=code == "circuit_open" and typed.get("retryable") is True,
        )
    return ProviderError(
        detail,
        provider=provider_name,
        # A malformed structured response is terminal, even though its HTTP
        # status is 502. Do not turn Sandhi's contract refusal into a retry hint.
        recoverable=(
            typed.get("retryable") is True
            and code not in {"invalid_response", "output_projection_error"}
        ),
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


def _gateway_run_id() -> str:
    """Stable per-agent-run identifier for the sandhi run cost tree.

    Gateway mode stamps this as the ``x-sandhi-run-id`` wire header (sandhi
    PR #149) so the proxy can persist a per-run cost tree queryable at
    ``GET /admin/usage/run/{run_id}``. Reuses victor's session identifier from
    the execution-context correlation spine (``victor.core.context``) — the
    same id the orchestrator binds for the whole agent run. Empty when no
    session is bound (bare provider usage outside an agent run).
    """
    try:
        from victor.core.context import get_session_id

        return str(get_session_id() or "")
    except Exception:  # pragma: no cover - attribution must never block transport
        return ""


def _gateway_family_base(root: str, protocol: str) -> str:
    """Derive the gateway ingress base from the explicit client dialect."""
    root = root.rstrip("/")
    for suffix in ("/v1", "/v1beta"):
        if root.endswith(suffix):
            root = root[: -len(suffix)]
            break
    if protocol == "gemini_generate_content":
        return root + "/v1beta"
    if protocol == "anthropic_messages":
        return root
    return root + "/v1"


def _gateway_wire_headers(headers: Mapping[str, str]) -> Dict[str, str]:
    """Drop headers governed by the gateway configuration or execution owner."""
    blocked = {
        "idempotency-key",
        "authorization",
        "proxy-authorization",
        "x-api-key",
        "api-key",
        "x-goog-api-key",
        "chatgpt-account-id",
        "openai-organization",
        "openai-project",
        "x-sandhi-usage-convention",
        "x-sandhi-grant",
    }
    return {name: value for name, value in headers.items() if name.lower() not in blocked}


def _wire_call_headers() -> Dict[str, str]:
    """Per-call gateway attribution; step changes must not change cache affinity.

    Run/session attribution stays on the cached handle and neutral request metadata.
    The step id rides only the per-call wire headers. Logical call context remains
    local correlation and never becomes an Idempotency-Key or replay authority.
    """
    headers: Dict[str, str] = {}
    try:
        from victor.core.context import get_turn_id

        turn_id = str(get_turn_id() or "")
    except Exception:  # pragma: no cover - attribution must never block transport
        return headers
    if turn_id:
        headers["x-sandhi-step-id"] = turn_id
    return headers


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
        ("cache_intent", "cache_intent"),
    ):
        if source in payload:
            request[target] = payload[source]
    if "stop" in payload:
        stop = payload["stop"]
        request["stop"] = stop if isinstance(stop, list) else [stop]

    thinking = _final_thinking_control(payload)
    if thinking is not None:
        request["thinking"] = thinking

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
        "cache_intent",
        "stop",
        "stream",
        "stream_options",
        "reasoning_effort",
        "thinking",
    }
    # Promoted controls never remain in a dialect-specific extension bucket.
    # Internal orchestration keys never belong on the wire (D4).
    excluded = reserved | _INTERNAL_KWARG_KEYS
    native = {key: value for key, value in payload.items() if key not in excluded}
    if native:
        request["extensions"] = {"openai": native}

    # Conversation affinity (sandhi ADR-0008 D3): the same execution-context session
    # id the run cost tree keys on (`_gateway_run_id`, above) rides in neutral
    # metadata — never the body, where it would break prompt-cache byte-stability.
    # The typed runtime maps it onto catalog-declared vendor affinity headers (e.g.
    # InferFlux's ``x-inferflux-session-id`` KV/prefix-cache key).
    session_id = _gateway_run_id()
    if session_id:
        request["metadata"] = {"session_id": session_id}
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


def _latency_fields(usage: Any) -> Dict[str, Any]:
    """Preserve Sandhi's timing values and per-field measurement provenance.

    Present from contract minor 3, so guaranteed at victor's >= 0.3.0 floor;
    the field-shape checks below stay tolerant-absent defensively. Carried on
    every run (unlike the non-routine diagnostics) so stream metrics can prefer
    wire truth over client wall-clock.
    """
    if not isinstance(usage, dict):
        return {}
    fields: Dict[str, Any] = {}
    for key in ("duration_ms", "time_to_first_token_ms"):
        value = usage.get(key)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and 0 <= value < float("inf")
        ):
            fields[key] = int(value)
            source_key = key.removesuffix("_ms") + "_source"
            if usage.get(source_key) in ("origin", "boundary"):
                fields[source_key] = usage[source_key]
    return fields


def _usage_diagnostics(usage: Any) -> Optional[Dict[str, Any]]:
    """Preserve non-routine typed metering state without polluting legacy token keys."""
    if not isinstance(usage, dict):
        return None
    attempts = int(usage.get("attempts", 1) or 1)
    completeness = usage.get("completeness")
    outcome = usage.get("outcome")
    observation = usage.get("cache_read_observation")
    if (
        not isinstance(observation, dict)
        and attempts <= 1
        and completeness not in {"partial", "unavailable"}
        and outcome
        not in {
            "error",
            "cancelled",
        }
    ):
        return None
    diagnostics = {
        "attempts": attempts,
        "completeness": completeness,
        "outcome": outcome,
        "upstream_request_id": usage.get("upstream_request_id"),
    }
    if isinstance(observation, dict):
        # Sandhi owns validation of the typed status/source contract.
        diagnostics["cache_read_observation"] = dict(observation)
    return diagnostics


class SandhiTypedProviderMixin:
    """Shared direct consumer of Sandhi's typed FFI contract."""

    _sandhi_runtime: Any = None
    _sandhi_typed_providers: Optional[Dict[Tuple[str, ...], Any]] = None
    _gateway_model_capabilities: Optional[Dict[str, Dict[str, Any]]] = None
    _gateway_model_capabilities_key: Optional[Tuple[str, str, Optional[str]]] = None

    @staticmethod
    def _validate_call_options(options: Dict[str, Any]) -> None:
        for name in ("endpoint_profile", "protocol", "peer_role", "gateway"):
            if name in options:
                raise ValueError(f"{name} is a constructor-only provider option")

    _SANDHI_WAIT_GRACE_SECS = 5.0

    async def close(self) -> None:
        """Release this provider's typed handles as well as native client state.

        The binding has no separate close API. Dropping our owned references
        releases idle Rust HTTP pools; active calls retain their local handles.
        Member cleanup must not call this on a borrowed parent provider.
        """
        try:
            await super().close()  # type: ignore[misc]
        finally:
            self._sandhi_typed_providers = None
            self._sandhi_runtime = None

    def _static_sandhi_model_descriptor(self, model: str) -> Optional[Dict[str, Any]]:
        """Return an exact model descriptor from the pinned binding catalog."""
        if not model or _sg is None or not hasattr(_sg, "provider_descriptor_json"):
            return None
        model = self._direct_model_name(model)
        try:
            descriptor = json.loads(_sg.provider_descriptor_json(self._sandhi_slug()))
            models = descriptor.get("models") if isinstance(descriptor, dict) else None
        except Exception:
            return None
        if not isinstance(models, list):
            return None
        try:
            validated: List[Dict[str, Any]] = []
            for candidate in models:
                if not isinstance(candidate, dict):
                    raise ValueError("invalid Sandhi model descriptor")
                _validated_thinking_capability(candidate.get("thinking"))
                validated.append(candidate)
        except ValueError:
            return None
        return next((candidate for candidate in validated if candidate.get("id") == model), None)

    @staticmethod
    def _parse_gateway_model_capabilities(body: Any) -> Dict[str, Dict[str, Any]]:
        """Parse and strictly validate a sanitized model-discovery page."""
        if (
            not isinstance(body, dict)
            or body.get("object") != "list"
            or type(body.get("sandhi_contract_version")) is not int
            or body["sandhi_contract_version"] != 1
        ):
            raise ValueError("unsupported Sandhi discovery contract")
        models = body.get("data")
        if not isinstance(models, list):
            raise ValueError("unsupported Sandhi discovery contract")
        result: Dict[str, Dict[str, Any]] = {}
        for model in models:
            if (
                not isinstance(model, dict)
                or model.get("object") != "model"
                or not isinstance(model.get("id"), str)
                or not model["id"]
            ):
                raise ValueError("invalid Sandhi model descriptor")
            _validated_thinking_capability(model.get("thinking"))
            if model["id"] in result:
                raise ValueError("duplicate Sandhi model descriptor")
            result[model["id"]] = model
        return result

    def _cached_gateway_model_capabilities(
        self, gateway: Tuple[str, str, Optional[str]]
    ) -> Optional[Dict[str, Dict[str, Any]]]:
        """Return discovery data for the current gateway credential and grant."""
        if (
            self._gateway_model_capabilities is not None
            and self._gateway_model_capabilities_key == gateway
        ):
            return self._gateway_model_capabilities
        return None

    def _store_gateway_model_capabilities(
        self, gateway: Tuple[str, str, Optional[str]], capabilities: Dict[str, Dict[str, Any]]
    ) -> None:
        self._gateway_model_capabilities = capabilities
        self._gateway_model_capabilities_key = gateway

    def _gateway_scope(self) -> Optional[Tuple[str, str, Optional[str]]]:
        gateway = self._gateway_overrides()
        if gateway is None:
            return None
        config = gateway_config((getattr(self, "extra_config", None) or {}).get("gateway"))
        return (*gateway, config.get("grant") if config else None)

    @staticmethod
    def _gateway_discovery_headers(credential: str, grant: Optional[str]) -> Dict[str, str]:
        headers = {"Authorization": f"Bearer {credential}"}
        if grant is not None:
            headers["x-sandhi-grant"] = grant
        return headers

    def _load_gateway_model_capabilities(self) -> Dict[str, Dict[str, Any]]:
        """Fetch discovery synchronously for callers outside the event loop.

        A failed page is cached for the credential and grant used to request it.
        A refreshed token or different grant gets a new bounded attempt.
        """
        gateway = self._gateway_scope()
        if gateway is None:
            return {}
        cached = self._cached_gateway_model_capabilities(gateway)
        if cached is not None:
            return cached
        root, credential, grant = gateway
        url = _gateway_family_base(root, "chat_completions") + "/models"
        capabilities: Dict[str, Dict[str, Any]] = {}
        try:
            response = httpx.get(
                url,
                headers=self._gateway_discovery_headers(credential, grant),
                timeout=min(self._sandhi_timeout(), 5.0),
                follow_redirects=False,
            )
            response.raise_for_status()
            capabilities = self._parse_gateway_model_capabilities(response.json())
        except Exception as error:
            logger.warning(
                "Sandhi model discovery unavailable; reasoning effort stays disabled: %s",
                type(error).__name__,
            )
        self._store_gateway_model_capabilities(gateway, capabilities)
        return capabilities

    async def _refresh_gateway_model_capabilities(self) -> None:
        """Refresh a renewable gateway credential, then await discovery I/O."""
        if self._gateway_overrides() is None:
            return
        refresh = getattr(self, "_refresh_host_credentials", None)
        if callable(refresh):
            await refresh()
        gateway = self._gateway_scope()
        if gateway is None:
            return
        cached = self._cached_gateway_model_capabilities(gateway)
        if cached is not None:
            return
        root, credential, grant = gateway
        url = _gateway_family_base(root, "chat_completions") + "/models"
        capabilities: Dict[str, Dict[str, Any]] = {}
        try:
            async with httpx.AsyncClient(
                timeout=min(self._sandhi_timeout(), 5.0),
                follow_redirects=False,
            ) as client:
                response = await client.get(
                    url, headers=self._gateway_discovery_headers(credential, grant)
                )
            response.raise_for_status()
            capabilities = self._parse_gateway_model_capabilities(response.json())
        except Exception as error:
            logger.warning(
                "Sandhi model discovery unavailable; reasoning effort stays disabled: %s",
                type(error).__name__,
            )
        self._store_gateway_model_capabilities(gateway, capabilities)

    def _sandhi_model_descriptor(self, model: Optional[str]) -> Optional[Dict[str, Any]]:
        if not model:
            return None
        gateway = self._gateway_scope()
        if gateway is not None:
            cached = self._cached_gateway_model_capabilities(gateway)
            if cached is None:
                try:
                    asyncio.get_running_loop()
                except RuntimeError:
                    cached = self._load_gateway_model_capabilities()
                else:
                    # The synchronous public capability API must not block the
                    # event loop. Async call sites prewarm with the method below.
                    logger.debug(
                        "Sandhi gateway model discovery awaits async prewarm; "
                        "reasoning effort stays disabled for this synchronous check"
                    )
                    return None
            return cached.get(model)
        return self._static_sandhi_model_descriptor(model)

    def supports_reasoning_effort(self, model: Optional[str] = None) -> bool:
        """Use exact Sandhi model metadata; never infer from provider or prefix."""
        descriptor = self._sandhi_model_descriptor(model)
        thinking = descriptor.get("thinking") if isinstance(descriptor, dict) else None
        effort = thinking.get("effort") if isinstance(thinking, dict) else None
        values = effort.get("values") if isinstance(effort, dict) else None
        return bool(
            isinstance(thinking, dict)
            and thinking.get("supported") is True
            and isinstance(values, list)
            and values
        )

    async def supports_reasoning_effort_async(self, model: Optional[str] = None) -> bool:
        """Exact capability check with async gateway discovery and token refresh."""
        await self._refresh_gateway_model_capabilities()
        return self.supports_reasoning_effort(model)

    async def supports_thinking_disable_async(self, model: str) -> bool:
        await self._refresh_gateway_model_capabilities()
        descriptor = self._sandhi_model_descriptor(model)
        thinking = descriptor.get("thinking") if descriptor else None
        return bool(
            isinstance(thinking, dict)
            and thinking.get("supported") is True
            and thinking.get("can_disable") is True
        )

    def _sandhi_slug(self) -> str:
        declared = str(getattr(self, "name", "openai"))
        if _sg is not None and hasattr(_sg, "provider_descriptor_json"):
            try:
                descriptor = json.loads(_sg.provider_descriptor_json(declared))
                return str(descriptor.get("slug") or declared)
            except Exception:
                pass
        return declared

    @staticmethod
    def _validated_sandhi_timeout(
        value: Any, *, label: str, hi: float, min_ms: int | None = None
    ) -> float:
        """Shared validator for the sandhi extra-config stream-timeout knobs.

        One validation site per knob class: bool rejection, finite-number
        coercion, positive-ceiling bounds, and (for the body budget) the
        one-millisecond floor. Mode-independent — a misconfiguration fails
        the same way in gateway and direct dispatch.
        """
        floor = (
            f"one millisecond and {hi:g} seconds" if min_ms is not None else f"0 and {hi:g} seconds"
        )
        if isinstance(value, bool):
            raise ValueError(f"{label} must be finite and between {floor}")
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{label} must be a finite number of seconds") from None
        if not math.isfinite(numeric) or numeric <= 0 or numeric > hi:
            raise ValueError(f"{label} must be finite and between {floor}")
        if min_ms is not None and numeric * 1000 < min_ms:
            raise ValueError(f"{label} must represent at least {min_ms} milliseconds")
        return numeric

    def _sandhi_stream_idle_timeout(self) -> float:
        value = (getattr(self, "extra_config", None) or {}).get("stream_idle_timeout_secs", 90.0)
        return self._validated_sandhi_timeout(
            value, label="stream idle timeout", hi=_SANDHI_STREAM_TIMEOUT_MAX_SECS
        )

    def _sandhi_stream_setup_timeout(self) -> float:
        """Stream-setup (time-to-first-byte) budget forwarded to Sandhi.

        Mirrors the idle knob: default 90s (the library default since the 30s
        era was measured too tight for cache-miss prefills of large agentic
        prompts), overridable per provider via ``stream_setup_timeout_secs``.
        """
        value = (getattr(self, "extra_config", None) or {}).get("stream_setup_timeout_secs", 90.0)
        return self._validated_sandhi_timeout(
            value, label="stream setup timeout", hi=_SANDHI_STREAM_TIMEOUT_MAX_SECS
        )

    def _sandhi_stream_body_timeout_secs(self) -> Optional[float]:
        """Optional body-duration hint sent to gateway policy preflight.

        Unlike setup/idle, this is deliberately not inferred from the outer request timeout:
        a planning deadline includes non-body work and must not reject an otherwise valid
        stream. Callers opt in when they know a model needs a long body lifetime.
        """
        value = (getattr(self, "extra_config", None) or {}).get("stream_body_timeout_secs")
        if value is None:
            return None
        return self._validated_sandhi_timeout(
            value,
            label="stream body timeout",
            hi=_SANDHI_BODY_TIMEOUT_MAX_SECS,
            min_ms=_SANDHI_BODY_TIMEOUT_MIN_MS,
        )

    def _sandhi_base_wire_headers(self) -> Dict[str, str]:
        """Gateway per-call headers shared by the streaming and buffered paths.

        Single construction site so the two dispatch modes cannot drift
        (blocked-header filtering, step-id headers, etc.)."""
        if self._gateway_overrides() is None:
            return {}
        return _gateway_wire_headers(_wire_call_headers())

    def _sandhi_stream_wire_headers(self) -> Dict[str, str]:
        # Validate unconditionally: a misconfigured knob must fail the same
        # way in direct mode as in gateway mode (fail fast, not silently no-op).
        body_timeout = self._sandhi_stream_body_timeout_secs()
        headers = self._sandhi_base_wire_headers()
        if body_timeout is None:
            return headers
        # round(), not int(): fractional-millisecond budgets must not be
        # silently floored below the caller's requested value.
        headers["x-sandhi-stream-body-budget-ms"] = str(round(body_timeout * 1000))
        return headers

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
        """Return ``(proxy_url, binding_credential)`` for a configured gateway.

        Gateway mode (TD-0003 P3) points the Sandhi FFI handle at the Sandhi proxy:
        the proxy URL becomes the handle ``base_url``. Protected modes present
        their configured token; starter mode uses the public internal marker.
        Server policy owns identity, attribution and budgets. An explicit client dialect is
        independent of server backend selection. Returns ``None`` only in
        direct mode; malformed configured gateway values fail closed.
        """
        raw = getattr(self, "extra_config", None)
        if not isinstance(raw, dict):
            return None
        gateway = gateway_config(raw.get("gateway"))
        if gateway is None:
            return None
        url = str(gateway["url"]).strip()
        virtual_key = (
            getattr(self, "_api_key", None)
            if getattr(self, "_gateway_credential", None) is not None
            else gateway_virtual_key(gateway)
        )
        # Duck-type SecretStr without importing pydantic into the transport layer.
        if hasattr(virtual_key, "get_secret_value"):
            virtual_key = virtual_key.get_secret_value()
        return (url, str(virtual_key or ""))

    def _client_protocol(self) -> str:
        gateway = gateway_config((getattr(self, "extra_config", None) or {}).get("gateway"))
        declared = str(getattr(self, "_sandhi_protocol", "") or "")
        return (
            client_protocol(self._sandhi_slug(), declared, gateway.get("protocol"))
            if gateway is not None
            else declared
        )

    def _direct_model_name(self, model: str) -> str:
        resolve = getattr(self, "_clean_model_name", None)
        return str(resolve(model)) if callable(resolve) else model

    def _usage_slug(self) -> str:
        if self._gateway_overrides() is not None:
            return "anthropic" if self._client_protocol() == "anthropic_messages" else "openai"
        return self._sandhi_slug()

    def _typed_usage_diagnostics(self, usage: Any) -> Dict[str, Any]:
        if not isinstance(usage, dict):
            return {}
        # These are validated typed units. Gateway views are not source receipts.
        # Preserve optional zero/absence and reasoning inclusion exactly.
        fields = (
            "tokens_in",
            "tokens_out",
            "cache_creation_tokens",
            "cache_read_tokens",
            "reasoning_tokens",
            "reasoning_included",
            "completeness",
            "outcome",
            "cache_read_observation",
        )
        typed = {name: usage[name] for name in fields if name in usage}
        if self._gateway_overrides() is not None:
            return {
                "gateway_client_protocol": self._client_protocol(),
                "gateway_client_usage": typed,
                "typed_usage": typed,
                "usage_basis": "gateway_client_projection",
            }
        return {"typed_usage": typed, "usage_basis": "direct_provider"}

    def _typed_provider(self, model: str) -> Any:
        if not sandhi_transport_available():
            raise ProviderConnectionError(
                "sandhi-gateway typed runtime is unavailable",
                provider=self._sandhi_slug(),
            )
        _verify_wire_contract()
        if self._sandhi_runtime is None:
            self._sandhi_runtime = _sg.ProviderRuntime()
        if self._sandhi_typed_providers is None:
            self._sandhi_typed_providers = {}
        slug = self._sandhi_slug()
        protocol = str(getattr(self, "_sandhi_protocol", "") or "")
        config = getattr(self, "extra_config", None)
        endpoint_profile = str(
            getattr(self, "_sandhi_endpoint_profile", None)
            or (config.get("endpoint_profile") if isinstance(config, dict) else "")
            or ""
        )
        gateway = self._gateway_scope()
        grant = None
        # Run cost tree (sandhi PR #149): only gateway mode stamps the run id —
        # direct-provider mode must stay byte-identical on the wire.
        run_id = _gateway_run_id() if gateway is not None else ""
        if gateway is not None:
            # Gateway mode (TD-0003 P3): the FFI handle targets the Sandhi proxy with
            # the configured credential or starter marker. The explicit client
            # dialect is independent of the provider identity and server backend.
            proxy_url, virtual_key, grant = gateway
            if not virtual_key:
                raise ProviderConnectionError(
                    "configured gateway credential is unavailable",
                    provider=slug,
                    recoverable=False,
                )
            protocol = self._client_protocol()
            endpoint_profile = ""
            base_url = _gateway_family_base(proxy_url, protocol)
            api_key = virtual_key
            # sandhi >= 0.3.0 (victor's floor) accepts "bearer" family-wide as a
            # no-op for the gateway virtual key (TD-0008 rule 5: reject
            # contradictions, accept redundancy), so gateway mode presents it
            # unconditionally. Earlier bindings that REJECTED an explicit
            # auth_scheme outside the Anthropic/Gemini protocols (victor#678) are
            # below the pin and no longer reachable. The real-binding
            # construction conformance suite (test_sandhi_binding_construction.py)
            # covers this.
            auth_scheme = "bearer"
            # The derived family base: distinct from the configured root so the
            # FFI dials the right ingress path (the cache key already leads
            # with the slug; this keeps the dialed URL itself correct).
            explicit_base_url = base_url
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
        # run_id participates in the cache key so the handle is reused within an
        # agent run but rebuilt (with a fresh x-sandhi-run-id header) per run.
        cache_key = (
            slug,
            model,
            explicit_base_url,
            api_key,
            auth_scheme,
            protocol,
            endpoint_profile,
            "sandhi_gateway_v1" if gateway is not None else "provider",
            grant or "",
            run_id,
        )
        if cache_key not in self._sandhi_typed_providers:
            kwargs: Dict[str, Any] = {
                "base_url": explicit_base_url or None,
                "timeout_secs": self._sandhi_timeout(),
                "stream_setup_timeout_secs": self._sandhi_stream_setup_timeout(),
                "stream_idle_timeout_secs": self._sandhi_stream_idle_timeout(),
                # One physical dispatch per binding invocation in direct and
                # gateway mode. ResilientProvider owns logical-call retries;
                # an ambiguous FFI/projection failure never triggers a hidden
                # second inference inside this handle.
                "max_retries": 0,
            }
            wire_headers = dict(getattr(self, "_wire_headers", None) or {})
            if gateway is not None:
                wire_headers = _gateway_wire_headers(wire_headers)
            if grant is not None:
                wire_headers["x-sandhi-grant"] = grant
            if run_id:
                # An explicit caller-set header wins; never clobber it.
                wire_headers.setdefault("x-sandhi-run-id", run_id)
                # Conversation affinity on the proxy path (sandhi ADR-0008 D3): the
                # proxy derives its session from ``x-sandhi-session`` and maps it onto
                # catalog-declared vendor affinity headers (InferFlux KV/prefix-cache
                # reuse). Same value as the run id, different consumer.
                wire_headers.setdefault("x-sandhi-session", run_id)
            if wire_headers:
                kwargs["headers_json"] = json.dumps(wire_headers)
            if gateway is not None:
                kwargs["peer_role"] = "sandhi_gateway_v1"
            if auth_scheme:
                kwargs["auth_scheme"] = auth_scheme
            if protocol:
                kwargs["protocol"] = protocol
            if endpoint_profile:
                kwargs["endpoint_profile"] = endpoint_profile
            self._sandhi_typed_providers[cache_key] = self._sandhi_runtime.provider(
                slug, model, api_key, **kwargs
            )
        return self._sandhi_typed_providers[cache_key]

    def _validate_gateway_request(self, request: Dict[str, Any]) -> None:
        if self._gateway_overrides() is None:
            return
        expected = {
            "chat_completions": "openai",
            "anthropic_messages": "anthropic",
            "responses": "openai_responses",
            "gemini_generate_content": "gemini",
        }[self._client_protocol()]
        extensions = request.get("extensions") or {}
        if any(value and key != expected for key, value in extensions.items()):
            raise ValueError(
                "native call options do not match the configured gateway client dialect; use canonical typed options"
            )

    async def _sandhi_complete(self, request: Dict[str, Any]) -> Dict[str, Any]:
        self._validate_gateway_request(request)
        provider = self._typed_provider(str(request.get("model", "")))
        timeout = self._sandhi_timeout()
        # Per-call step headers (gateway mode only — direct mode stays byte-identical).
        call_headers = self._sandhi_base_wire_headers()
        wire_headers = json.dumps(call_headers) if call_headers else None
        try:
            value = await asyncio.wait_for(
                provider.complete_json(json.dumps(request), wire_headers),
                timeout=timeout + self._SANDHI_WAIT_GRACE_SECS,
            )
            return json.loads(str(value))
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:  # pyo3 panics may subclass BaseException
            mapped = await map_sandhi_error(
                exc, self._sandhi_slug(), timeout, gateway=self._gateway_overrides() is not None
            )
            if (
                isinstance(mapped, ProviderPolicyError)
                or mapped.category == ErrorCategory.CONFIG_MISSING
            ):
                raise mapped from None
            raise mapped from exc

    def _completion_from_typed(self, response: Dict[str, Any], model: str) -> CompletionResponse:
        output = response.get("output") or {}
        extensions = response.get("extensions") or {}
        native_key = self._sandhi_slug()
        if self._gateway_overrides() is not None:
            native_key = {
                "chat_completions": "openai",
                "anthropic_messages": "anthropic",
                "responses": "openai_responses",
                "gemini_generate_content": "gemini",
            }[self._client_protocol()]
        native = extensions.get(native_key)
        if native is None and self._client_protocol() in {
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
        # Minor 11 supplies displayable text independently of the native echo.
        # Older bindings retain their explicit legacy extension/native-body path.
        reasoning = output.get("reasoning")
        if reasoning is not None and not isinstance(reasoning, str):
            raise ProviderError(
                "Sandhi output.reasoning must be text or null",
                provider=self._sandhi_slug(),
                recoverable=False,
            )
        typed_reasoning_present = reasoning is not None
        if reasoning is None:
            reasoning = extensions.get("reasoning")
        if reasoning is None and isinstance(native, dict):
            reasoning = (
                (native.get("choices") or [{}])[0].get("message", {}).get("reasoning_content")
            )
        usage = usage_dict_from_neutral(
            response.get("usage"),
            native.get("usage") if isinstance(native, dict) else None,
            slug=self._usage_slug(),
        )
        metadata: Dict[str, Any] = {}
        if reasoning or typed_reasoning_present:
            metadata["reasoning_content"] = reasoning
        diagnostics = dict(_usage_diagnostics(response.get("usage")) or {})
        diagnostics.update(_latency_fields(response.get("usage")))
        diagnostics.update(self._typed_usage_diagnostics(response.get("usage")))
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

    async def _sandhi_stream_with_idle_timeout(
        self,
        provider: Any,
        request: Dict[str, Any],
        wire_headers: Optional[str],
        idle_timeout: float,
    ) -> AsyncIterator[str]:
        """Wrap provider.stream_json() with per-chunk idle timeout.

        If no chunk arrives within ``idle_timeout`` seconds, raises
        asyncio.TimeoutError. This prevents the process from hanging
        indefinitely when the upstream LLM API connection drops.
        """
        stream_iter = provider.stream_json(json.dumps(request), wire_headers).__aiter__()
        while True:
            try:
                event_json = await asyncio.wait_for(stream_iter.__anext__(), timeout=idle_timeout)
                yield event_json
            except asyncio.TimeoutError:
                logger.warning(
                    "Stream idle timeout after %.1fs — no chunk received. "
                    "The upstream LLM API connection may have dropped.",
                    idle_timeout,
                )
                raise asyncio.TimeoutError(
                    f"Stream idle timeout: no chunk received within {idle_timeout}s. "
                    "The upstream LLM API connection may have dropped."
                )
            except StopAsyncIteration:
                break

    async def _sandhi_stream(self, request: Dict[str, Any]) -> AsyncIterator[StreamChunk]:
        self._validate_gateway_request(request)
        provider = self._typed_provider(str(request.get("model", "")))
        timeout = self._sandhi_timeout()
        idle_timeout = self._sandhi_stream_idle_timeout()
        call_headers = self._sandhi_stream_wire_headers()
        wire_headers = json.dumps(call_headers) if call_headers else None
        calls: Dict[int, Dict[str, Any]] = {}
        finish_reason: Optional[str] = None
        reported_model: Optional[str] = None
        response_id: Optional[str] = None
        usage: Optional[Dict[str, int | bool]] = None
        usage_diagnostics: Optional[Dict[str, Any]] = None
        try:
            async for event_json in self._sandhi_stream_with_idle_timeout(
                provider, request, wire_headers, idle_timeout
            ):
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
                    neutral = event.get("usage")
                    incoming = dict(_usage_diagnostics(neutral) or {})
                    incoming.update(_latency_fields(neutral))
                    incoming.update(self._typed_usage_diagnostics(neutral))
                    if isinstance(neutral, dict) and neutral.get("completeness") == "unavailable":
                        # Minor 9 emits metadata-only cache corrections. These are
                        # neither measured zeros nor a replacement numeric verdict.
                        if usage is None:
                            usage_diagnostics = incoming or None
                        elif "cache_read_observation" in incoming:
                            usage_diagnostics = {
                                **(usage_diagnostics or {}),
                                "cache_read_observation": incoming["cache_read_observation"],
                            }
                            for key in ("typed_usage", "gateway_client_usage"):
                                if key in usage_diagnostics:
                                    usage_diagnostics[key] = {
                                        **usage_diagnostics[key],
                                        "cache_read_observation": incoming[
                                            "cache_read_observation"
                                        ],
                                    }
                    else:
                        usage = usage_dict_from_neutral(neutral, None, slug=self._usage_slug())
                        usage_diagnostics = incoming or None
                elif kind == "response_start":
                    # Preserve the typed provider claim separately from the
                    # requested alias; callers must not infer model identity.
                    reported_model = event.get("model")
                    response_id = event.get("id")
                elif kind == "tool_call_end":
                    # Deliberately ignored: call boundaries are tracked by the
                    # indexed accumulation above; the terminal chunk assembles them.
                    continue
                elif kind == "error":
                    # A typed error EVENT (as opposed to an iterator error) must
                    # surface as the mapped ProviderError, never vanish mid-stream.
                    payload = event.get("error")
                    raise await map_sandhi_error(
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
            if finish_reason is None:
                raise ProviderError(
                    "Sandhi stream ended without a validated finish event",
                    provider=self._sandhi_slug(),
                    recoverable=False,
                )
            yield StreamChunk(
                content="",
                tool_calls=_tool_calls([calls[index] for index in sorted(calls)]),
                stop_reason=finish_reason,
                is_final=True,
                usage=usage,
                metadata={
                    **({"sandhi_usage": usage_diagnostics} if usage_diagnostics else {}),
                    "sandhi_stream": {
                        "model": reported_model,
                        "response_id": response_id,
                        "finish_observed": finish_reason is not None,
                    },
                },
            )
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except ProviderError:
            raise
        except BaseException as exc:
            mapped = await map_sandhi_error(
                exc, self._sandhi_slug(), timeout, gateway=self._gateway_overrides() is not None
            )
            if (
                isinstance(mapped, ProviderPolicyError)
                or mapped.category == ErrorCategory.CONFIG_MISSING
            ):
                raise mapped from None
            raise mapped from exc


class SandhiNeutralProviderMixin(SandhiTypedProviderMixin):
    """Send portable Victor prompts through the neutral Sandhi transport."""

    async def _refresh_host_credentials(self) -> None:
        refresh = getattr(self, "_ensure_valid_token", None)
        if callable(refresh) and (
            self._gateway_overrides() is None
            or getattr(self, "_gateway_credential", None) is not None
        ):
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
        direct = self._gateway_overrides() is None
        if direct:
            model = self._direct_model_name(model)
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
        if direct and tools:
            # Tool selection orders stable schemas before stubs. Pass only the
            # stable prefix length and let the transport place any breakpoint.
            for index, tool in enumerate(tools):
                if getattr(tool, "schema_level", None) == "stub":
                    if index > 0:
                        request["stable_tool_prefix_count"] = index
                    break
        native = request.get("extensions", {}).get("openai")
        if native:
            unsupported = ", ".join(sorted(native))
            raise ValueError(
                f"native call options have no portable Sandhi representation: {unsupported}"
            )
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
        self._validate_call_options(kwargs)
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
        self._validate_call_options(kwargs)
        await self._refresh_host_credentials()
        request = self._neutral_request(messages, model, temperature, max_tokens, tools, **kwargs)
        async with aclosing(self._sandhi_stream(request)) as upstream:
            async for chunk in upstream:
                yield chunk


class SandhiHttpxTransportMixin(SandhiNeutralProviderMixin):
    """Reuse the neutral request path with OpenAI-compatible configuration."""

    @staticmethod
    def _validate_call_options(options: Dict[str, Any]) -> None:
        SandhiTypedProviderMixin._validate_call_options(options)
        if "text_response_mode" in options:
            raise ValueError("text_response_mode is a constructor-only provider option")


class SandhiAnthropicProvider(SandhiNeutralProviderMixin, AnthropicProvider):
    pass


class SandhiOpenAIProvider(SandhiNeutralProviderMixin, OpenAIProvider):
    pass


class SandhiGoogleProvider(SandhiNeutralProviderMixin, GoogleProvider):
    pass


class SandhiLocalProviderMixin(SandhiNeutralProviderMixin):
    """Preserve direct discovery; gateway authentication needs no upstream probe."""

    @classmethod
    async def create(cls, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("gateway") is not None:
            # Validate before native factories can probe or read endpoint config.
            gateway_virtual_key(kwargs["gateway"])
            return cls(*args, **kwargs)
        # The concrete local provider supplies this cooperative factory in the MRO.
        return await super().create(*args, **kwargs)  # type: ignore[misc]


class SandhiOllamaProvider(SandhiLocalProviderMixin, OllamaProvider):
    pass


class SandhiLMStudioProvider(SandhiLocalProviderMixin, LMStudioProvider):
    pass


class SandhiVLLMProvider(SandhiLocalProviderMixin, VLLMProvider):
    pass


class SandhiLlamaCppProvider(SandhiLocalProviderMixin, LlamaCppProvider):
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

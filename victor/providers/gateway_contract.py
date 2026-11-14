"""Pure gateway configuration: client dialect never grants backend authority."""

import re
from typing import Any, Optional
from urllib.parse import urlsplit

PROTOCOLS = {"chat_completions", "anthropic_messages", "responses", "gemini_generate_content"}
GATEWAY_NO_AUTH_PLACEHOLDER = "not-needed"


def gateway_auth_mode(config: dict) -> str:
    """Infer auth only from supplied gateway credentials, never upstream accounts.

    Missing credentials deliberately select the starter gateway. Present invalid
    credentials are errors, including null/empty values that normalization must
    not erase. The URL is validated separately by ``gateway_config``.
    """
    mode = config.get("auth_mode")
    if "auth_mode" in config and mode not in ("none", "tokens", "oidc"):
        raise ValueError("gateway auth_mode must be none, tokens or oidc")
    supplied = [name for name in ("virtual_key", "oidc", "credential") if name in config]
    if len(supplied) > 1:
        raise ValueError("gateway requires exactly one credential when authentication is selected")
    if mode == "none" and supplied:
        raise ValueError("gateway auth_mode none cannot include credentials")
    if "virtual_key" in config:
        key = config["virtual_key"]
        if hasattr(key, "get_secret_value"):
            key = key.get_secret_value()
        if not isinstance(key, str) or not key or any(not 33 <= ord(c) <= 126 for c in key):
            raise ValueError("gateway requires a non-empty valid virtual key")
    if "oidc" in config and not isinstance(config["oidc"], dict):
        raise ValueError("gateway oidc must be an identity configuration")
    if "credential" in config and config["credential"] is None:
        raise ValueError("gateway credential must be a TokenCredential")
    if "audience" in config and "credential" not in config:
        raise ValueError("gateway audience requires a TokenCredential")
    inferred = "tokens" if "virtual_key" in supplied else "oidc" if supplied else "none"
    if mode is not None and mode != inferred:
        raise ValueError("gateway auth_mode requires its matching credentials")
    return inferred


def gateway_config(value: Any) -> Optional[dict]:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("gateway must be a configuration object")
    if set(value) - {
        "url",
        "virtual_key",
        "auth_mode",
        "oidc",
        "credential",
        "audience",
        "grant",
        "protocol",
    }:
        raise ValueError("gateway contains unsupported configuration fields")
    url = value.get("url")
    if not isinstance(url, str) or not url.strip():
        raise ValueError("gateway requires a URL")
    try:
        parsed = urlsplit(url.strip().rstrip("/"))
        port = parsed.port
    except ValueError:
        raise ValueError("gateway URL has an invalid host or port") from None
    if (
        not parsed.hostname
        or port == 0
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in ("", "/v1", "/v1beta")
        or not (
            parsed.scheme == "https"
            or (parsed.scheme == "http" and parsed.hostname in ("127.0.0.1", "::1", "localhost"))
        )
    ):
        raise ValueError("gateway URL must use HTTPS or loopback HTTP and a root path")
    protocol = value.get("protocol")
    if protocol is not None and (not isinstance(protocol, str) or protocol not in PROTOCOLS):
        raise ValueError("gateway protocol must be one of the four supported client dialects")
    grant = value.get("grant")
    if grant is not None and (
        not isinstance(grant, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", grant)
    ):
        raise ValueError("gateway grant must be an unambiguous policy name")
    return value


def gateway_virtual_key(value: Any) -> Optional[str]:
    config = gateway_config(value)
    if config is None:
        return None
    mode = gateway_auth_mode(config)
    if mode == "none":
        # The current FFI requires a non-empty bearer even for an unauthenticated
        # gateway. This public marker carries no identity or server authority.
        return GATEWAY_NO_AUTH_PLACEHOLDER
    if config.get("oidc") is not None or config.get("credential") is not None:
        raise ValueError("renewable gateway credentials currently require the OpenAI provider")
    key = config.get("virtual_key")
    if hasattr(key, "get_secret_value"):
        key = key.get_secret_value()
    if not isinstance(key, str) or not key.strip():
        raise ValueError("gateway requires a non-empty virtual key")
    return key


def validate_gateway_auth(value: Any, provider: str) -> None:
    """Validate metadata without reading upstream credentials or broker tokens."""
    config = gateway_config(value)
    if config is None:
        return
    mode = gateway_auth_mode(config)
    if mode == "none":
        return
    if provider != "openai":
        gateway_virtual_key(config)
        return
    key = config.get("virtual_key")
    if hasattr(key, "get_secret_value"):
        key = key.get_secret_value()
    oidc = config.get("oidc")
    credential = config.get("credential")
    if oidc is not None:
        from victor.core.identity.gateway import GatewayTokenFileCredential

        if not isinstance(oidc, dict):
            raise ValueError("gateway oidc must be an identity configuration")
        # Constructor validation is metadata-only. Dispatch owns token reads.
        GatewayTokenFileCredential(oidc)
    elif credential is not None:
        from victor.core.identity.protocols import TokenCredential

        if not isinstance(credential, TokenCredential) or not config.get("audience"):
            raise ValueError("gateway credential requires TokenCredential and audience")
    elif not isinstance(key, str) or not key.strip():
        raise ValueError("gateway requires a non-empty virtual key")


def client_protocol(slug: str, declared: str = "", override: Optional[str] = None) -> str:
    if override is not None:
        return override
    aliases = {
        "chatgpt_responses": "responses",
        "codex_responses": "responses",
        "codex": "responses",
        "openai": "chat_completions",
        "anthropic": "anthropic_messages",
        "gemini": "gemini_generate_content",
    }
    selected = aliases.get(declared, declared)
    if selected:
        if selected not in PROTOCOLS:
            raise ValueError("unsupported gateway client protocol")
        return selected
    return {
        "anthropic": "anthropic_messages",
        "claude": "anthropic_messages",
        "google": "gemini_generate_content",
        "gemini": "gemini_generate_content",
    }.get(slug, "chat_completions")

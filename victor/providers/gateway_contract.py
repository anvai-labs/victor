"""Pure gateway configuration: client dialect never grants backend authority."""

import re
from typing import Any, Optional
from urllib.parse import urlsplit

PROTOCOLS = {"chat_completions", "anthropic_messages", "responses", "gemini_generate_content"}


def gateway_config(value: Any) -> Optional[dict]:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("gateway must be a configuration object")
    url = value.get("url")
    if not isinstance(url, str) or not url.strip():
        raise ValueError("gateway requires a URL")
    parsed = urlsplit(url.strip().rstrip("/"))
    if (
        not parsed.hostname
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
    if config.get("oidc") is not None or config.get("credential") is not None:
        raise ValueError("renewable gateway credentials currently require the OpenAI provider")
    key = config.get("virtual_key")
    if hasattr(key, "get_secret_value"):
        key = key.get_secret_value()
    if not isinstance(key, str) or not key.strip():
        raise ValueError("gateway requires a non-empty virtual key")
    return key


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

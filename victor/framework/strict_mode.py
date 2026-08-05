# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Strict-mode helpers for framework compatibility fallback paths."""

from __future__ import annotations

import os

_STRICT_PRIVATE_FALLBACKS_ENV = "VICTOR_STRICT_FRAMEWORK_PRIVATE_FALLBACKS"
_STRICT_PROTOCOL_FALLBACKS_ENV = "VICTOR_STRICT_FRAMEWORK_PROTOCOL_FALLBACKS"


def _is_truthy_env(value: str) -> bool:
    return value in {"1", "true", "yes", "on"}


def strict_private_fallbacks_enabled() -> bool:
    """Return True when strict private-fallback blocking is enabled."""
    value = os.getenv(_STRICT_PRIVATE_FALLBACKS_ENV, "").strip().lower()
    return _is_truthy_env(value)


def strict_protocol_fallbacks_enabled() -> bool:
    """Return True when protocol-fallback blocking is enabled."""
    value = os.getenv(_STRICT_PROTOCOL_FALLBACKS_ENV, "").strip().lower()
    return _is_truthy_env(value)


def ensure_not_private_fallback(attribute_name: str, *, operation: str) -> None:
    """Raise in strict mode when a compatibility fallback targets private attrs."""
    if not str(attribute_name).startswith("_"):
        return
    if not strict_private_fallbacks_enabled():
        return
    raise RuntimeError(
        f"Private attribute fallback blocked for {operation}: '{attribute_name}'. "
        f"Disable strict mode by unsetting {_STRICT_PRIVATE_FALLBACKS_ENV}."
    )


def ensure_protocol_fallback_allowed(*, operation: str, fallback_target: str) -> None:
    """Raise in strict mode when protocol runtime falls back to duck-typed probes."""
    if not strict_protocol_fallbacks_enabled():
        return
    raise RuntimeError(
        f"Protocol fallback blocked for {operation}: '{fallback_target}'. "
        f"Disable strict mode by unsetting {_STRICT_PROTOCOL_FALLBACKS_ENV}."
    )


__all__ = [
    "ensure_protocol_fallback_allowed",
    "ensure_not_private_fallback",
    "strict_protocol_fallbacks_enabled",
    "strict_private_fallbacks_enabled",
]

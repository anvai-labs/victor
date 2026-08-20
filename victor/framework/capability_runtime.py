# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Shared runtime helpers for capability discovery and invocation."""

from __future__ import annotations

import logging
from typing import Any, Optional

from victor.framework.protocols import CapabilityRegistryProtocol
from victor.framework.strict_mode import ensure_protocol_fallback_allowed

logger = logging.getLogger(__name__)


def check_capability(
    obj: Any,
    capability_name: str,
    *,
    min_version: Optional[str] = None,
) -> bool:
    """Check whether an object exposes a capability.

    Uses CapabilityRegistryProtocol when available and falls back to mapped
    public methods/properties for legacy compatibility.
    """
    if isinstance(obj, CapabilityRegistryProtocol):
        return obj.has_capability(capability_name, min_version=min_version)

    if min_version is not None:
        logger.debug(
            "Version check requested for '%s' but object does not implement "
            "CapabilityRegistryProtocol; using public method fallback.",
            capability_name,
        )

    from victor.core.capability_registry import get_method_for_capability

    method_name = get_method_for_capability(capability_name)
    ensure_protocol_fallback_allowed(
        operation=f"capability check '{capability_name}'",
        fallback_target=method_name,
    )
    attr = getattr(obj, method_name, None)
    return hasattr(obj, method_name) and (callable(attr) or not callable(attr))


def invoke_capability(
    obj: Any,
    capability_name: str,
    *args: Any,
    min_version: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Invoke a capability on an object via registry or mapped public method."""
    if isinstance(obj, CapabilityRegistryProtocol):
        try:
            return obj.invoke_capability(capability_name, *args, min_version=min_version, **kwargs)
        except (KeyError, TypeError) as e:
            logger.debug("Registry invoke failed for '%s': %s", capability_name, e)

    if min_version is not None:
        logger.debug(
            "Version check requested for '%s' but object does not implement "
            "CapabilityRegistryProtocol; invoking without version check.",
            capability_name,
        )

    from victor.core.capability_registry import get_method_for_capability

    method_name = get_method_for_capability(capability_name)
    ensure_protocol_fallback_allowed(
        operation=f"capability invocation '{capability_name}'",
        fallback_target=method_name,
    )
    method = getattr(obj, method_name, None)
    if callable(method):
        return method(*args, **kwargs)

    raise AttributeError(
        f"Cannot invoke capability '{capability_name}' on {type(obj).__name__}. "
        f"Expected method '{method_name}' not found. "
        "Object should implement CapabilityRegistryProtocol."
    )


__all__ = [
    "check_capability",
    "invoke_capability",
]

"""Detect a gateway enforcement boundary through the standard provider wrappers."""

from typing import Any


def uses_gateway(provider: Any) -> bool:
    """Inspect wrapper-owned fields without expanding arbitrary dynamic attributes.

    A malformed or empty explicit gateway still establishes an enforcement boundary.
    Traversal is bounded and cycle-safe; excessive nesting fails closed.
    """
    pending = [provider]
    seen: set[int] = set()
    while pending and len(seen) < 16:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        fields = getattr(current, "__dict__", {})
        config = fields.get("extra_config")
        if isinstance(config, dict) and config.get("gateway") is not None:
            return True
        for name in ("_base_provider", "_active_provider", "provider"):
            child = fields.get(name)
            if child is not None:
                pending.append(child)
    return bool(pending)

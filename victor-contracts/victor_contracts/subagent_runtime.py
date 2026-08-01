"""SDK host adapters for subagent role-provider runtime helpers."""

from __future__ import annotations

import importlib
import warnings
from typing import TYPE_CHECKING, Any

from victor_contracts._deprecation import deprecated_bridge_message

if TYPE_CHECKING:
    from victor.agent.subagents.protocols import (
        DefaultRoleToolProvider,
        RoleToolProvider,
        get_role_tool_provider,
        set_role_tool_provider,
    )

__all__ = [
    "RoleToolProvider",
    "DefaultRoleToolProvider",
    "get_role_tool_provider",
    "set_role_tool_provider",
]

_LAZY_IMPORTS = {
    "RoleToolProvider": "victor.agent.subagents.protocols",
    "DefaultRoleToolProvider": "victor.agent.subagents.protocols",
    "get_role_tool_provider": "victor.agent.subagents.protocols",
    "set_role_tool_provider": "victor.agent.subagents.protocols",
}


def __getattr__(name: str) -> Any:
    """Resolve subagent runtime helpers lazily from the Victor host runtime."""
    warnings.warn(
        deprecated_bridge_message("subagent_runtime"),
        DeprecationWarning,
        stacklevel=2,
    )
    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(
            f"module 'victor_contracts.subagent_runtime' has no attribute {name!r}"
        )

    module = importlib.import_module(module_name)
    return getattr(module, name)

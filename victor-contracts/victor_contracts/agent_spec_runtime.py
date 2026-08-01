"""SDK host adapters for agent-spec runtime models."""

from __future__ import annotations

import importlib
import warnings
from typing import TYPE_CHECKING, Any

from victor_contracts._deprecation import deprecated_bridge_message

if TYPE_CHECKING:
    from victor.agent.specs.models import (
        AgentCapabilities,
        AgentConstraints,
        AgentSpec,
        DelegationPolicy,
        ModelPreference,
        OutputFormat,
    )

__all__ = [
    "AgentSpec",
    "AgentCapabilities",
    "AgentConstraints",
    "ModelPreference",
    "OutputFormat",
    "DelegationPolicy",
]

_LAZY_IMPORTS = {
    "AgentSpec": "victor.agent.specs.models",
    "AgentCapabilities": "victor.agent.specs.models",
    "AgentConstraints": "victor.agent.specs.models",
    "ModelPreference": "victor.agent.specs.models",
    "OutputFormat": "victor.agent.specs.models",
    "DelegationPolicy": "victor.agent.specs.models",
}


def __getattr__(name: str) -> Any:
    """Resolve agent-spec helpers lazily from the Victor host runtime."""
    warnings.warn(
        deprecated_bridge_message("agent_spec_runtime"),
        DeprecationWarning,
        stacklevel=2,
    )
    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(
            f"module 'victor_contracts.agent_spec_runtime' has no attribute {name!r}"
        )

    module = importlib.import_module(module_name)
    return getattr(module, name)

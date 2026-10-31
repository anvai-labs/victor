"""victor.framework.protocols / config re-export the SDK definitions they forked.

Round-4 dedup (SDK<->core forks): each core import path must yield the SAME
object as its ``victor_contracts`` definition, so ``isinstance``/``issubclass``
checks and enum identity/equality agree across both paths. Behaviour of the
re-exported definitions (validation, enum values/order, runtime_checkable
matching) must be unchanged.
"""

from __future__ import annotations

import dataclasses
import importlib
from typing import Any, Dict, Optional

import pytest

# (core module, core name, SDK module, SDK name)
IDENTITY_PAIRS = [
    (
        "victor.framework.protocols",
        "CapabilityConfigScopePortProtocol",
        "victor_contracts.capabilities.runtime",
        "CapabilityConfigScopePortProtocol",
    ),
    (
        "victor.framework.protocols",
        "CapabilityType",
        "victor_contracts.capabilities.runtime",
        "CapabilityType",
    ),
    (
        "victor.framework.config",
        "SafetyLevel",
        "victor_contracts.safety.framework",
        "SafetyLevel",
    ),
    (
        "victor.framework.config",
        "SafetyLevel",
        "victor_contracts.safety_policy",
        "SafetyLevel",
    ),
    (
        "victor.framework.capability_loader",
        "CapabilityEntry",
        "victor_contracts.capabilities.runtime",
        "CapabilityEntry",
    ),
    (
        "victor.framework",
        "CapabilityEntry",
        "victor_contracts.capabilities.runtime",
        "CapabilityEntry",
    ),
]


@pytest.mark.parametrize(
    "core_mod, core_name, sdk_mod, sdk_name",
    IDENTITY_PAIRS,
    ids=[f"{c}.{n}<->{s}" for c, n, s, _ in IDENTITY_PAIRS],
)
def test_core_name_is_the_sdk_object(
    core_mod: str, core_name: str, sdk_mod: str, sdk_name: str
) -> None:
    core_obj = getattr(importlib.import_module(core_mod), core_name)
    sdk_obj = getattr(importlib.import_module(sdk_mod), sdk_name)
    assert core_obj is sdk_obj


# --------------------------------------------------------------------------
# OrchestratorCapability: framework subclass of the SDK dataclass
# --------------------------------------------------------------------------


def test_orchestrator_capability_extends_sdk_definition() -> None:
    from victor.framework.protocols import CapabilityType, OrchestratorCapability
    from victor_contracts.capabilities import OrchestratorCapability as SDKCapability

    assert issubclass(OrchestratorCapability, SDKCapability)
    # Validation is inherited, not re-declared.
    assert "__post_init__" not in vars(OrchestratorCapability)
    assert "_is_valid_version" not in vars(OrchestratorCapability)
    assert "is_compatible_with" in vars(OrchestratorCapability)

    cap = OrchestratorCapability(
        name="x", capability_type=CapabilityType.TOOL, getter="get_x", version="1.10"
    )
    assert dataclasses.is_dataclass(cap)
    assert [f.name for f in dataclasses.fields(cap)] == [
        "name",
        "capability_type",
        "version",
        "setter",
        "getter",
        "attribute",
        "description",
        "required",
        "deprecated",
        "deprecated_message",
    ]
    assert cap.is_compatible_with("1.9")
    assert not cap.is_compatible_with("1.11")
    replaced = dataclasses.replace(cap, name="y")
    assert type(replaced) is OrchestratorCapability
    assert replaced == dataclasses.replace(cap, name="y")
    assert repr(cap).startswith("OrchestratorCapability(")


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"version": "1"}, "invalid version"),
        ({"version": "a.b"}, "invalid version"),
        ({"version": "1.0.0"}, "invalid version"),
        ({"getter": None}, "must specify at least one of"),
    ],
)
def test_orchestrator_capability_still_rejects_malformed_declarations(
    overrides: Dict[str, Any], message: str
) -> None:
    from victor.framework.protocols import CapabilityType, OrchestratorCapability

    kwargs: Dict[str, Any] = {
        "name": "x",
        "capability_type": CapabilityType.TOOL,
        "getter": "get_x",
    }
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=message):
        OrchestratorCapability(**kwargs)


def test_sdk_capability_is_not_mistaken_for_framework_capability() -> None:
    from victor.framework.capability_loader import _coerce_capability
    from victor.framework.protocols import CapabilityType, OrchestratorCapability
    from victor_contracts.capabilities import OrchestratorCapability as SDKCapability

    sdk_cap = SDKCapability(name="s", capability_type=CapabilityType.MODE, setter="set_s")
    assert not isinstance(sdk_cap, OrchestratorCapability)
    coerced = _coerce_capability(sdk_cap)
    assert type(coerced) is OrchestratorCapability
    assert coerced.capability_type is CapabilityType.MODE


# --------------------------------------------------------------------------
# Enum shape and runtime_checkable behaviour preserved
# --------------------------------------------------------------------------


def test_enum_members_values_and_order_preserved() -> None:
    from victor.framework.config import SafetyLevel
    from victor.framework.protocols import CapabilityType

    assert [(m.name, m.value) for m in CapabilityType] == [
        ("TOOL", "tool"),
        ("PROMPT", "prompt"),
        ("MODE", "mode"),
        ("SAFETY", "safety"),
        ("RL", "rl"),
        ("TEAM", "team"),
        ("WORKFLOW", "workflow"),
        ("VERTICAL", "vertical"),
    ]
    assert issubclass(CapabilityType, str)
    assert [m.value for m in SafetyLevel] == ["off", "low", "medium", "high"]
    # SafetyLevel is now a str-Enum (additive str-ness from the SDK unification):
    # a member compares equal to its raw value. str()/f-string still render the
    # member name path, not the value (pinned in the SDK suite).
    assert issubclass(SafetyLevel, str)
    assert SafetyLevel.LOW == "low"


def test_capability_config_scope_port_runtime_checkable_preserved() -> None:
    from victor.framework.protocols import CapabilityConfigScopePortProtocol
    from victor_contracts.capabilities import resolve_capability_config_scope_key

    class _Scoped:
        def get_capability_config_scope_key(self) -> Optional[str]:
            return "  session-1 "

    class _Unscoped:
        pass

    class _NoneScope:
        def get_capability_config_scope_key(self) -> Optional[str]:
            return None

    assert isinstance(_Scoped(), CapabilityConfigScopePortProtocol)
    assert not isinstance(_Unscoped(), CapabilityConfigScopePortProtocol)
    assert resolve_capability_config_scope_key(_Scoped()) == "session-1"
    # Optional[str] implementations (as in AgentOrchestrator) still normalize.
    assert resolve_capability_config_scope_key(_NoneScope()) == resolve_capability_config_scope_key(
        _Unscoped()
    )


# --------------------------------------------------------------------------
# CapabilityEntry: framework loader re-exports the SDK dataclass
# --------------------------------------------------------------------------


def test_capability_entry_shape_preserved() -> None:
    import dataclasses

    from victor.framework.capability_loader import CapabilityEntry
    from victor.framework.protocols import CapabilityType, OrchestratorCapability

    assert [f.name for f in dataclasses.fields(CapabilityEntry)] == [
        "capability",
        "handler",
        "getter_handler",
        "source_module",
        "metadata",
    ]
    cap = OrchestratorCapability(
        name="x", capability_type=CapabilityType.TOOL, getter="get_x", version="1.10"
    )
    entry = CapabilityEntry(capability=cap, source_module="m", metadata={"k": "v"})
    assert entry.name == "x"
    assert entry.version == "1.10"
    assert entry.capability_type is CapabilityType.TOOL
    assert type(dataclasses.replace(entry, source_module="n")) is CapabilityEntry


def test_loader_built_entry_isinstance_sdk_capability_entry() -> None:
    from victor.framework.capability_loader import CapabilityLoader
    from victor.framework.protocols import CapabilityType
    from victor_contracts.capabilities import CapabilityEntry as SDKCapabilityEntry

    loader = CapabilityLoader()
    loader.register_capability(
        name="sdk_identity_cap",
        handler=lambda payload: payload,
        capability_type=CapabilityType.TOOL,
    )
    entry = loader.get_capability("sdk_identity_cap")
    assert entry is not None
    # Loader-built entries must satisfy SDK-side isinstance checks (verticals
    # import CapabilityEntry from victor_contracts.capabilities).
    assert isinstance(entry, SDKCapabilityEntry)
    assert type(entry) is SDKCapabilityEntry

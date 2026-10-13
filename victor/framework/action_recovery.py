# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
# Licensed under the Apache License, Version 2.0.
"""Typed receipt evidence for trusted tool adapters, not a retry or dispatch API."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Protocol


def _identifier(value: Any) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", value):
        raise ValueError("Recovery identifiers must be bounded nonsecret identifiers")


def _hash(value: Any) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
        raise ValueError("Recovery requires an exact action and binding digest")


@dataclass(frozen=True)
class RecoveryIdentity:
    """Adapter semantics version and account/tenant/environment backend namespace.

    These are nonsecret stable IDs, not provider names or credential-bearing URLs.
    A configured adapter must never reuse one identity for a different backend.
    """

    adapter_id: str
    backend_id: str
    version: int = 1

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise ValueError("Unsupported recovery identity version")
        _identifier(self.adapter_id)
        _identifier(self.backend_id)

    def to_dict(self) -> dict[str, Any]:
        self.__post_init__()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Any) -> RecoveryIdentity:
        if not isinstance(value, dict) or set(value) != {"version", "adapter_id", "backend_id"}:
            raise ValueError("Malformed recovery identity")
        return cls(**value)


@dataclass(frozen=True)
class ActionLookup:
    """Runtime-owned identity passed to effect and receipt lookup implementations."""

    action_id: str
    binding_digest: str
    identity: RecoveryIdentity

    def __post_init__(self) -> None:
        _hash(self.action_id)
        _hash(self.binding_digest)
        if type(self.identity) is not RecoveryIdentity:
            raise ValueError("Recovery identity is required")
        self.identity.__post_init__()


@dataclass(frozen=True)
class BackendReceipt:
    """An authoritative committed effect; absence never proves nonexecution."""

    action_id: str
    binding_digest: str
    identity: RecoveryIdentity
    receipt_id: str
    outcome: str = "committed"
    version: int = 1

    def __post_init__(self) -> None:
        ActionLookup(self.action_id, self.binding_digest, self.identity)
        _identifier(self.receipt_id)
        if type(self.version) is not int or self.version != 1 or self.outcome != "committed":
            raise ValueError("Unsupported backend receipt")

    def to_dict(self) -> dict[str, Any]:
        self.__post_init__()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Any) -> BackendReceipt:
        if not isinstance(value, dict) or set(value) != {
            "version",
            "action_id",
            "binding_digest",
            "identity",
            "receipt_id",
            "outcome",
        }:
            raise ValueError("Malformed backend receipt")
        return cls(**{**value, "identity": RecoveryIdentity.from_dict(value["identity"])})


class ActionRecovery(Protocol):
    """Trusted, cancellation-cooperative backend lookup on an existing tool.

    The adapter authenticates to its exact backend and only returns receipts that
    were atomically committed with the bound effect. No tool execution, retries or
    model-provided receipt data belong here. None means unknown, never not-executed.
    """

    @property
    def identity(self) -> RecoveryIdentity: ...

    async def lookup(self, request: ActionLookup) -> BackendReceipt | None: ...


def recovery_identity(tool: Any) -> dict[str, Any] | None:
    """Snapshot a configured tool's trusted capability, rejecting malformed setup."""
    capability = getattr(tool, "action_recovery", None)
    if capability is None:
        return None
    identity = capability.identity
    if type(identity) is not RecoveryIdentity or not callable(getattr(capability, "lookup", None)):
        raise ValueError("Malformed tool recovery capability")
    return identity.to_dict()

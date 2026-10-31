"""Framework-style safety rule helpers for external vertical compatibility.

Host bridge
-----------
``SafetyRule``, ``SafetyConfig`` and ``SafetyEnforcer`` lazily bind their
``SafetyLevel`` objects to the host runtime (``victor.framework.config``) the
first time one is constructed with the host importable, mirroring the
``victor_contracts.safety_policy`` host adapter. The SDK never imports
``victor`` at module scope: the binding goes through
:func:`_host_safety_module`, an ``importlib`` hook resolved once per process.
With the host registered, SDK-created safety objects evaluate levels with the
host's enum, so host and SDK enforcers reach identical allow/block decisions
(both fail closed when a rule predicate raises). Without a host, the objects
keep self-contained SDK semantics, which are fail-closed as well.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import importlib
from typing import Any, Callable, Dict, Optional, Tuple


class SafetyLevel(str, Enum):
    """Safety enforcement levels for operation checks.

    This is the single ``SafetyLevel`` definition in the SDK: it is re-exported
    from the ``victor_contracts.safety`` top level and bound as
    ``victor.framework.config.SafetyLevel`` by the host runtime, so every import
    path yields the same enum object. The ``str`` mixin is additive: members
    compare equal to (and hash like) their lowercase value strings, while
    ``str()``/f-strings still render as ``SafetyLevel.LOW`` on Python 3.12+.
    """

    OFF = "off"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class _Unresolved:
    """Sentinel for the not-yet-resolved host hook."""


_UNRESOLVED = _Unresolved()
_host_config_module: Any = _UNRESOLVED


def _host_safety_module() -> Optional[Any]:
    """Return the host ``victor.framework.config`` module, or ``None``.

    The import is attempted once per process and the result — including an
    import FAILURE — is cached permanently (no retry): once resolved host-less,
    the SDK keeps its self-contained fail-closed semantics for the rest of the
    process. Tests can simulate a host-less environment by monkeypatching
    ``_host_config_module`` to ``None``.
    """
    global _host_config_module
    if _host_config_module is _UNRESOLVED:
        try:
            _host_config_module = importlib.import_module("victor.framework.config")
        except Exception:
            _host_config_module = None
    return _host_config_module


def _coerce_level(level: Any) -> Any:
    """Normalize a level to the SDK enum, then to the host enum when registered.

    ``SafetyLevel`` instances pass through UNCHANGED: the host re-exports this
    very enum, so re-binding them would only swap objects when a foreign host
    module resolves (e.g. a cross-tree import), which breaks identity asserts.
    Strings and foreign enums are parsed via :func:`parse_safety_level` (friendly
    ``ValueError`` for invalid values) and bound to the host enum when the host
    hook is registered; a partial or incompatible host import (missing or
    value-mismatched ``SafetyLevel``) keeps the self-contained SDK enum.
    """
    if isinstance(level, SafetyLevel):
        return level

    coerced = parse_safety_level(getattr(level, "value", level))
    host_module = _host_safety_module()
    if host_module is None:
        return coerced
    try:
        return host_module.SafetyLevel(coerced.value)
    except (AttributeError, ValueError):
        return coerced


def parse_safety_level(value: Any) -> SafetyLevel:
    """Parse a level string (case-insensitive) into :class:`SafetyLevel`.

    Shared by the SDK and host ``SafetyConfig.from_dict`` implementations so
    both accept the same inputs and raise the same friendly ``ValueError``.
    """
    try:
        # Case-insensitive lookup
        return SafetyLevel(value.lower())
    except (ValueError, KeyError, AttributeError) as e:
        raise ValueError(
            f"Invalid safety level: {value}. Must be one of "
            f"{[level.value for level in SafetyLevel]}"
        ) from e


def safety_config_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    """Map a safety-config dictionary to ``SafetyConfig`` constructor kwargs.

    Shared by the SDK and host ``SafetyConfig.from_dict`` implementations so
    both honor the same defaults and level parsing.
    """
    level = config.get("level", "medium")
    # Dict payloads are serialized data (``to_dict`` emits the value string):
    # the level must be a plain string. Enum members are rejected explicitly —
    # with the str mixin they ARE str instances, so the isinstance(str) gate
    # alone would let them through (pre-unification they raised only by
    # accident: plain-Enum members have no .lower()).
    if isinstance(level, SafetyLevel) or not isinstance(level, str):
        raise ValueError(
            f"Invalid safety level: {level!r}. Must be one of "
            f"{[lvl.value for lvl in SafetyLevel]}"
        )
    return {
        "level": parse_safety_level(level),
        "require_confirmation": config.get("require_confirmation", False),
        "blocked_operations": config.get("blocked_operations", []),
        "audit_log": config.get("audit_log", True),
        "dry_run": config.get("dry_run", False),
    }


@dataclass
class SafetyConfig:
    """Configuration for framework-style safety enforcement."""

    level: SafetyLevel = SafetyLevel.MEDIUM
    require_confirmation: bool = False
    blocked_operations: list[str] = field(default_factory=list)
    audit_log: bool = True
    dry_run: bool = False

    def __post_init__(self) -> None:
        """Use the host runtime SafetyLevel enum when available."""
        self.level = _coerce_level(self.level)

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "SafetyConfig":
        """Create a safety config from a dictionary.

        Level parsing matches the host ``SafetyConfig.from_dict``: the lookup is
        case-insensitive and invalid levels raise a friendly ``ValueError``.
        """
        return cls(**safety_config_kwargs(config))

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable dictionary representation."""
        return {
            "level": self.level.value,
            "require_confirmation": self.require_confirmation,
            "blocked_operations": list(self.blocked_operations),
            "audit_log": self.audit_log,
            "dry_run": self.dry_run,
        }


@dataclass
class SafetyRule:
    """Operation-level safety rule with a predicate callback."""

    name: str
    description: str
    check_fn: Callable[[str], bool]
    level: Any = SafetyLevel.MEDIUM
    allow_override: bool = False

    def __post_init__(self) -> None:
        """Use the host runtime SafetyLevel enum when available."""
        self.level = _coerce_level(self.level)


class SafetyEnforcer:
    """Framework-style operation safety enforcer."""

    def __init__(self, config: SafetyConfig):
        self.config = config
        # Delegate level evaluation to the host hook when registered (the same
        # lazy binding SafetyRule/SafetyConfig apply to themselves): the
        # enforcer normalizes its config's level so SDK and host objects share
        # one enum and reach identical decisions. The mutation is idempotent
        # and intended: SDK configs are already bound by their own
        # ``__post_init__`` (a no-op pass-through), while foreign or duck-typed
        # configs are upgraded in place.
        level = getattr(config, "level", None)
        if level is not None:
            config.level = _coerce_level(level)
        self.rules: list[SafetyRule] = []

    @staticmethod
    def _level_value(level: Any) -> str:
        return str(getattr(level, "value", level))

    def add_rule(self, rule: SafetyRule) -> None:
        """Register a safety rule."""
        if any(existing.name == rule.name for existing in self.rules):
            raise ValueError(f"Rule '{rule.name}' already registered")
        self.rules.append(rule)

    def remove_rule(self, name: str) -> bool:
        """Remove a rule by name."""
        for index, rule in enumerate(self.rules):
            if rule.name == name:
                self.rules.pop(index)
                return True
        return False

    def check_operation(
        self,
        operation: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Optional[str]]:
        """Check whether an operation is allowed.

        Fails closed: a registered rule whose ``check_fn`` raises is treated as
        unevaluable and blocks the operation (matching the host twin in
        ``victor.framework.config.SafetyEnforcer``).
        """
        del context
        if self.config.dry_run:
            return True, f"[DRY RUN] Would execute: {operation}"

        for blocked in self.config.blocked_operations:
            if blocked.lower() in operation.lower():
                return False, f"Blocked by configuration: '{blocked}'"

        for rule in self.rules:
            try:
                if not rule.check_fn(operation):
                    continue
            except Exception:
                return False, f"Safety rule could not be evaluated: {rule.name}"

            rule_level = self._level_value(rule.level)
            config_level = self._level_value(self.config.level)

            if rule_level == SafetyLevel.HIGH.value:
                if rule.allow_override and config_level == SafetyLevel.LOW.value:
                    continue
                return (
                    False,
                    f"Blocked by safety rule: {rule.name} - {rule.description}",
                )

            if rule_level == SafetyLevel.MEDIUM.value:
                if config_level == SafetyLevel.LOW.value:
                    continue
                return (
                    False,
                    f"Blocked by safety rule: {rule.name} - {rule.description}",
                )

            if rule_level == SafetyLevel.LOW.value and config_level == SafetyLevel.HIGH.value:
                return (
                    False,
                    f"Blocked by safety rule: {rule.name} - {rule.description}",
                )

        return True, None

    def get_rules_by_level(self, level: Any) -> list[SafetyRule]:
        """Return all rules at the requested level."""
        level_value = self._level_value(level)
        return [rule for rule in self.rules if self._level_value(rule.level) == level_value]


__all__ = [
    "SafetyConfig",
    "SafetyEnforcer",
    "SafetyLevel",
    "SafetyRule",
    "parse_safety_level",
    "safety_config_kwargs",
]

"""Framework-style safety rule helpers for external vertical compatibility."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import importlib
from typing import Any, Callable, Dict, Optional, Tuple


class SafetyLevel(Enum):
    """Safety enforcement levels for operation checks."""

    OFF = "off"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class SafetyConfig:
    """Configuration for framework-style safety enforcement."""

    level: SafetyLevel = SafetyLevel.MEDIUM
    require_confirmation: bool = False
    blocked_operations: list[str] = field(default_factory=list)
    audit_log: bool = True
    dry_run: bool = False

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "SafetyConfig":
        """Create a safety config from a dictionary."""
        return cls(
            level=SafetyLevel(config.get("level", "medium")),
            require_confirmation=config.get("require_confirmation", False),
            blocked_operations=config.get("blocked_operations", []),
            audit_log=config.get("audit_log", True),
            dry_run=config.get("dry_run", False),
        )

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

        self.level = SafetyLevel(getattr(self.level, "value", self.level))
        try:
            runtime_config = importlib.import_module("victor.framework.config")
            runtime_level = runtime_config.SafetyLevel
        except Exception:
            return

        level_value = getattr(self.level, "value", self.level)
        self.level = runtime_level(level_value)


class SafetyEnforcer:
    """Framework-style operation safety enforcer."""

    def __init__(self, config: SafetyConfig):
        self.config = config
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
        """Check whether an operation is allowed."""
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
]

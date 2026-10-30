"""SDK host adapters for framework safety policy helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from victor_contracts._lazy import make_lazy_getattr

if TYPE_CHECKING:
    from victor.framework.config import (
        SafetyConfig,
        SafetyEnforcer,
        SafetyLevel,
        SafetyRule,
    )
    from victor.framework.safety import (
        create_file_safety_rules,
        create_git_safety_rules,
    )

__all__ = [
    "SafetyConfig",
    "SafetyEnforcer",
    "SafetyLevel",
    "SafetyRule",
    "create_file_safety_rules",
    "create_git_safety_rules",
]

_LAZY_IMPORTS = {
    "SafetyConfig": "victor.framework.config",
    "SafetyEnforcer": "victor.framework.config",
    "SafetyLevel": "victor.framework.config",
    "SafetyRule": "victor.framework.config",
    "create_file_safety_rules": "victor.framework.safety",
    "create_git_safety_rules": "victor.framework.safety",
}


__getattr__ = make_lazy_getattr(__name__, _LAZY_IMPORTS)

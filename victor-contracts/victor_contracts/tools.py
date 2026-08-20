# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SDK Tool Contract — declarable, stable tool metadata (FEP-0009).

External, SDK-first verticals import only ``victor_contracts``; this module gives them a
stable, **data-only** way to declare a tool's traits (``access_mode``, ``danger_level``,
``execution_category``, ``cost_tier``, ``category``, plus selection hints) without
importing ``victor.tools.*`` (a boundary violation). The framework's
``victor.tools.contract.resolve_contract`` bridges a declared :class:`ToolContract` into
the richer internal ``victor.tools.metadata.ToolMetadata`` at unchanged precedence.

Design (FEP-0009, Phase 1):
- **Data-only**: no imports from ``victor.*`` (only the in-package ``CapabilityContract``).
- **Frozen + tuples**: the contract is hashable and immutable, safe to share.
- **``str``-valued enums** whose values mirror the framework enums exactly, so the bridge
  maps by value and the wire form is plain strings tolerant of version skew.
- **Declarable intent only**: ranking/loop-detection knobs (priority, mandatory_keywords,
  signature_params) deliberately stay internal — see FEP-0009 Q2.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from victor_contracts.core.capability_contract import CapabilityContract


class ToolCategory(str, Enum):
    """Canonical tool-category vocabulary (FEP-0009 Q1: the SDK owns this).

    ``victor.framework.tools.ToolCategory`` re-exports these names (adding only a legacy
    ``REFACTOR`` alias at the framework layer). Values mirror
    ``victor/config/tool_categories.yaml`` (pinned equal by a framework guard test).
    """

    CORE = "core"
    FILESYSTEM = "filesystem"
    GIT = "git"
    SEARCH = "search"
    WEB = "web"
    DATABASE = "database"
    DOCKER = "docker"
    TESTING = "testing"
    REFACTORING = "refactoring"
    DOCUMENTATION = "documentation"
    ANALYSIS = "analysis"
    COMMUNICATION = "communication"
    NOTEBOOK = "notebook"
    TASK_MANAGEMENT = "task_management"
    VERIFICATION = "verification"
    CUSTOM = "custom"


class AccessMode(str, Enum):
    """How a tool accesses resources (drives approval tracking). Mirrors the framework
    ``victor.tools.enums.AccessMode`` values."""

    READONLY = "readonly"
    WRITE = "write"
    EXECUTE = "execute"
    NETWORK = "network"
    MIXED = "mixed"


class DangerLevel(str, Enum):
    """Risk level for warnings/approval. Mirrors ``victor.tools.enums.DangerLevel``."""

    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ExecutionCategory(str, Enum):
    """Side-effect class for parallel-execution conflict analysis. Mirrors
    ``victor.tools.enums.ExecutionCategory``. (A distinct axis from :class:`ToolCategory` —
    see PR #238.)"""

    READ_ONLY = "read_only"
    WRITE = "write"
    COMPUTE = "compute"
    NETWORK = "network"
    EXECUTE = "execute"
    MIXED = "mixed"


class CostTier(str, Enum):
    """Relative cost of invoking a tool. Mirrors ``victor.tools.enums.CostTier``."""

    FREE = "free"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ArgumentKind(str, Enum):
    """Semantic kind of a tool argument (FEP-0024).

    Drives per-argument behavior — notably, only ``EXECUTABLE_CODE`` arguments are
    eligible for code correction; ``FILE_CONTENT`` is an authored document that is
    never auto-mutated (validated read-only at most).
    """

    EXECUTABLE_CODE = "executable_code"
    FILE_CONTENT = "file_content"
    DATA = "data"


@dataclass(frozen=True)
class ToolContract:
    """Declarable, stable subset of a tool's metadata (the SDK boundary type).

    Carries only *declarable intent*: the safety/economics/identity traits that drive
    capping, parallel-execution, and prompt-splitting, plus the author's selection hints.
    It does NOT carry selection-*engine* knobs (priority/priority_hints,
    mandatory_keywords, signature_params) — those stay internal to ``ToolMetadata`` and are
    owned by the selection engine / RL, not the tool author (FEP-0009 Q2).

    Attach to a tool as a class attribute ``contract = ToolContract(...)``; the framework's
    ``resolve_contract`` honors it below an explicitly set ``.metadata`` and above autogen.
    """

    category: ToolCategory | str = ToolCategory.CUSTOM
    access_mode: AccessMode = AccessMode.READONLY
    danger_level: DangerLevel = DangerLevel.SAFE
    execution_category: ExecutionCategory = ExecutionCategory.READ_ONLY
    cost_tier: CostTier = CostTier.LOW
    keywords: tuple[str, ...] = ()
    use_cases: tuple[str, ...] = ()
    task_types: tuple[str, ...] = ()
    stages: tuple[str, ...] = ()
    # Per-argument semantic kind: sequence of (param-name, ArgumentKind) pairs. Frozen +
    # hashable (not a dict). Empty by default — the tool-level access_mode gate applies.
    argument_kinds: tuple[tuple[str, ArgumentKind], ...] = ()

    def category_value(self) -> str:
        """The category as a plain string (accepts either the enum or a raw str)."""
        return self.category.value if isinstance(self.category, ToolCategory) else self.category


# Per-capability version gate (FEP-0009 Q3). External packages assert this contract — not
# the raw victor-ai package range — to detect the feature floor. Floor is the contracts
# release that first ships this module (0.7.1), shipped additively in the 0.7.x line.
CONTRACT = CapabilityContract(name="tools", version=1, min_sdk_version=">=0.7.1")


__all__ = [
    "ToolCategory",
    "AccessMode",
    "DangerLevel",
    "ExecutionCategory",
    "CostTier",
    "ToolContract",
    "CONTRACT",
]

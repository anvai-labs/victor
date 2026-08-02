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

"""Unified severity/risk classification.

Consolidates six near-identical severity enums into a single
canonical type with bidirectional mapping to each domain-specific
variant. This eliminates cognitive overhead from learning
DangerLevel vs OperationalRiskLevel vs RiskLevel vs CVESeverity
vs SecretSeverity vs PIISeverity when they all mean the same thing.

The domain-specific enums are NOT removed (that would be a breaking
change). Instead, each gains a ``to_severity()`` method and the
unified enum gains ``from_*()`` classmethods.

Usage:
    from victor.core.severity import SeverityLevel

    # Convert from any domain enum
    level = SeverityLevel.from_danger_level(DangerLevel.HIGH)
    level = SeverityLevel.from_risk_level(RiskLevel.CRITICAL)
    level = SeverityLevel.from_cve_severity(CVESeverity.MEDIUM)

    # Use in unified comparisons
    if level >= SeverityLevel.HIGH:
        require_confirmation()
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any


class SeverityLevel(IntEnum):
    """Unified severity/risk classification across all Victor subsystems.

    Ordered by increasing severity so that comparisons work naturally:
        SeverityLevel.LOW < SeverityLevel.HIGH  # True

    Maps to all domain-specific enums:
        DangerLevel         (tools/enums.py)       SAFE→CRITICAL
        OperationalRiskLevel (agent/safety.py)     SAFE→CRITICAL
        RiskLevel           (security/safety)       LOW→CRITICAL
        CVESeverity         (security/protocol.py)  NONE→CRITICAL
        SecretSeverity      (security/safety)       LOW→CRITICAL
        PIISeverity         (security/safety)       LOW→CRITICAL
    """

    NONE = 0  # No risk / not applicable
    LOW = 1  # Minor, easily reversible
    MEDIUM = 2  # Moderate, effort to reverse
    HIGH = 3  # Significant, difficult to reverse
    CRITICAL = 4  # Irreversible or system-wide impact

    @property
    def requires_confirmation(self) -> bool:
        """Whether this severity level warrants user confirmation."""
        return self >= SeverityLevel.MEDIUM

    @property
    def label(self) -> str:
        """Human-readable label."""
        return self.name.lower()

    # ------------------------------------------------------------------
    # Mapping FROM domain-specific enums
    # ------------------------------------------------------------------

    @classmethod
    def from_danger_level(cls, dl: Any) -> SeverityLevel:
        """Map from victor.tools.enums.DangerLevel."""
        return _MAP_BY_VALUE.get(getattr(dl, "value", str(dl)).lower(), cls.CRITICAL)

    @classmethod
    def from_operational_risk(cls, orl: Any) -> SeverityLevel:
        """Map from victor.agent.safety.OperationalRiskLevel."""
        return _MAP_BY_VALUE.get(getattr(orl, "value", str(orl)).lower(), cls.CRITICAL)

    @classmethod
    def from_risk_level(cls, rl: Any) -> SeverityLevel:
        """Map from victor.security.safety.code_patterns.RiskLevel."""
        return _MAP_BY_VALUE.get(getattr(rl, "value", str(rl)).lower(), cls.CRITICAL)

    @classmethod
    def from_cve_severity(cls, cs: Any) -> SeverityLevel:
        """Map from victor.security.protocol.CVESeverity."""
        return _MAP_BY_VALUE.get(getattr(cs, "value", str(cs)).lower(), cls.CRITICAL)

    @classmethod
    def from_secret_severity(cls, ss: Any) -> SeverityLevel:
        """Map from victor.security.safety.secrets.SecretSeverity."""
        return _MAP_BY_VALUE.get(getattr(ss, "value", str(ss)).lower(), cls.CRITICAL)

    @classmethod
    def from_pii_severity(cls, ps: Any) -> SeverityLevel:
        """Map from victor.security.safety.pii.PIISeverity."""
        return _MAP_BY_VALUE.get(getattr(ps, "value", str(ps)).lower(), cls.CRITICAL)

    @classmethod
    def from_any(cls, value: Any) -> SeverityLevel:
        """Best-effort mapping from any severity/risk enum or string.

        Accepts enum instances, string values ("high", "CRITICAL"),
        or integer weights (0-4).
        """
        if isinstance(value, SeverityLevel):
            return value
        if isinstance(value, int):
            try:
                return cls(value)
            except ValueError:
                return cls.CRITICAL
        raw = getattr(value, "value", str(value)).lower()
        return _MAP_BY_VALUE.get(raw, cls.CRITICAL)


# Shared value→level mapping covering all naming conventions.
_MAP_BY_VALUE: dict[str, SeverityLevel] = {
    "none": SeverityLevel.NONE,
    "safe": SeverityLevel.NONE,
    "low": SeverityLevel.LOW,
    "medium": SeverityLevel.MEDIUM,
    "high": SeverityLevel.HIGH,
    "critical": SeverityLevel.CRITICAL,
}

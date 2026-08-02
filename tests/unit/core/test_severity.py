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

"""Tests for the unified severity/risk classification."""

from victor.core.severity import SeverityLevel


class TestSeverityLevel:
    def test_ordering(self):
        assert (
            SeverityLevel.NONE
            < SeverityLevel.LOW
            < SeverityLevel.MEDIUM
            < SeverityLevel.HIGH
            < SeverityLevel.CRITICAL
        )

    def test_requires_confirmation(self):
        assert not SeverityLevel.NONE.requires_confirmation
        assert not SeverityLevel.LOW.requires_confirmation
        assert SeverityLevel.MEDIUM.requires_confirmation
        assert SeverityLevel.HIGH.requires_confirmation
        assert SeverityLevel.CRITICAL.requires_confirmation

    def test_label(self):
        assert SeverityLevel.NONE.label == "none"
        assert SeverityLevel.LOW.label == "low"
        assert SeverityLevel.MEDIUM.label == "medium"
        assert SeverityLevel.HIGH.label == "high"
        assert SeverityLevel.CRITICAL.label == "critical"


class TestFromAny:
    def test_from_string(self):
        assert SeverityLevel.from_any("high") == SeverityLevel.HIGH
        assert SeverityLevel.from_any("CRITICAL") == SeverityLevel.CRITICAL
        assert SeverityLevel.from_any("safe") == SeverityLevel.NONE
        assert SeverityLevel.from_any("none") == SeverityLevel.NONE
        assert SeverityLevel.from_any("low") == SeverityLevel.LOW
        assert SeverityLevel.from_any("medium") == SeverityLevel.MEDIUM

    def test_from_int(self):
        assert SeverityLevel.from_any(0) == SeverityLevel.NONE
        assert SeverityLevel.from_any(1) == SeverityLevel.LOW
        assert SeverityLevel.from_any(2) == SeverityLevel.MEDIUM
        assert SeverityLevel.from_any(3) == SeverityLevel.HIGH
        assert SeverityLevel.from_any(4) == SeverityLevel.CRITICAL

    def test_from_invalid_int(self):
        assert SeverityLevel.from_any(99) == SeverityLevel.CRITICAL

    def test_from_identity(self):
        assert SeverityLevel.from_any(SeverityLevel.HIGH) == SeverityLevel.HIGH
        assert SeverityLevel.from_any(SeverityLevel.NONE) == SeverityLevel.NONE

    def test_unknown_string_defaults_critical(self):
        assert SeverityLevel.from_any("banana") == SeverityLevel.CRITICAL


class TestDomainEnumMappings:
    def test_danger_level_to_severity(self):
        from victor.tools.enums import DangerLevel

        assert DangerLevel.SAFE.to_severity() == SeverityLevel.NONE
        assert DangerLevel.LOW.to_severity() == SeverityLevel.LOW
        assert DangerLevel.MEDIUM.to_severity() == SeverityLevel.MEDIUM
        assert DangerLevel.HIGH.to_severity() == SeverityLevel.HIGH
        assert DangerLevel.CRITICAL.to_severity() == SeverityLevel.CRITICAL

    def test_operational_risk_level_to_severity(self):
        from victor.agent.safety import OperationalRiskLevel

        assert OperationalRiskLevel.SAFE.to_severity() == SeverityLevel.NONE
        assert OperationalRiskLevel.LOW.to_severity() == SeverityLevel.LOW
        assert OperationalRiskLevel.MEDIUM.to_severity() == SeverityLevel.MEDIUM
        assert OperationalRiskLevel.HIGH.to_severity() == SeverityLevel.HIGH
        assert OperationalRiskLevel.CRITICAL.to_severity() == SeverityLevel.CRITICAL

    def test_risk_level_to_severity(self):
        from victor.security.safety.code_patterns import RiskLevel

        assert RiskLevel.LOW.to_severity() == SeverityLevel.LOW
        assert RiskLevel.MEDIUM.to_severity() == SeverityLevel.MEDIUM
        assert RiskLevel.HIGH.to_severity() == SeverityLevel.HIGH
        assert RiskLevel.CRITICAL.to_severity() == SeverityLevel.CRITICAL

    def test_cve_severity_to_severity(self):
        from victor.security.protocol import CVESeverity

        assert CVESeverity.NONE.to_severity() == SeverityLevel.NONE
        assert CVESeverity.LOW.to_severity() == SeverityLevel.LOW
        assert CVESeverity.MEDIUM.to_severity() == SeverityLevel.MEDIUM
        assert CVESeverity.HIGH.to_severity() == SeverityLevel.HIGH
        assert CVESeverity.CRITICAL.to_severity() == SeverityLevel.CRITICAL

    def test_secret_severity_to_severity(self):
        from victor.security.safety.secrets import SecretSeverity

        assert SecretSeverity.LOW.to_severity() == SeverityLevel.LOW
        assert SecretSeverity.MEDIUM.to_severity() == SeverityLevel.MEDIUM
        assert SecretSeverity.HIGH.to_severity() == SeverityLevel.HIGH
        assert SecretSeverity.CRITICAL.to_severity() == SeverityLevel.CRITICAL

    def test_pii_severity_to_severity(self):
        from victor.security.safety.pii import PIISeverity

        assert PIISeverity.LOW.to_severity() == SeverityLevel.LOW
        assert PIISeverity.MEDIUM.to_severity() == SeverityLevel.MEDIUM
        assert PIISeverity.HIGH.to_severity() == SeverityLevel.HIGH
        assert PIISeverity.CRITICAL.to_severity() == SeverityLevel.CRITICAL


class TestFromClassmethods:
    def test_from_danger_level(self):
        from victor.tools.enums import DangerLevel

        assert SeverityLevel.from_danger_level(DangerLevel.SAFE) == SeverityLevel.NONE
        assert SeverityLevel.from_danger_level(DangerLevel.HIGH) == SeverityLevel.HIGH

    def test_from_operational_risk(self):
        from victor.agent.safety import OperationalRiskLevel

        assert SeverityLevel.from_operational_risk(OperationalRiskLevel.SAFE) == SeverityLevel.NONE
        assert (
            SeverityLevel.from_operational_risk(OperationalRiskLevel.CRITICAL)
            == SeverityLevel.CRITICAL
        )

    def test_from_risk_level(self):
        from victor.security.safety.code_patterns import RiskLevel

        assert SeverityLevel.from_risk_level(RiskLevel.LOW) == SeverityLevel.LOW
        assert SeverityLevel.from_risk_level(RiskLevel.CRITICAL) == SeverityLevel.CRITICAL

    def test_from_cve_severity(self):
        from victor.security.protocol import CVESeverity

        assert SeverityLevel.from_cve_severity(CVESeverity.NONE) == SeverityLevel.NONE
        assert SeverityLevel.from_cve_severity(CVESeverity.HIGH) == SeverityLevel.HIGH

    def test_from_secret_severity(self):
        from victor.security.safety.secrets import SecretSeverity

        assert SeverityLevel.from_secret_severity(SecretSeverity.CRITICAL) == SeverityLevel.CRITICAL

    def test_from_pii_severity(self):
        from victor.security.safety.pii import PIISeverity

        assert SeverityLevel.from_pii_severity(PIISeverity.HIGH) == SeverityLevel.HIGH

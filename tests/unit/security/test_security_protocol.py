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

"""Tests for security scanning protocol types and data structures."""

import pytest
from datetime import datetime
from pathlib import Path

from victor.security.protocol import (
    CVE,
    CVSSMetrics,
    Dependency,
    SecurityPolicy,
    SecurityScanResult,
    Severity,
    Vulnerability,
    VulnerabilityStatus,
)

# =============================================================================
# ENUM TESTS
# =============================================================================


class TestSeverity:
    """Tests for Severity enum."""

    def test_none_severity(self):
        """Test none severity."""
        assert Severity.NONE.value == "none"

    def test_low_severity(self):
        """Test low severity."""
        assert Severity.LOW.value == "low"

    def test_medium_severity(self):
        """Test medium severity."""
        assert Severity.MEDIUM.value == "medium"

    def test_high_severity(self):
        """Test high severity."""
        assert Severity.HIGH.value == "high"

    def test_critical_severity(self):
        """Test critical severity."""
        assert Severity.CRITICAL.value == "critical"

    def test_from_cvss_zero(self):
        """Test from_cvss with zero score."""
        assert Severity.from_cvss(0.0) == Severity.NONE

    def test_from_cvss_low(self):
        """Test from_cvss with low score."""
        assert Severity.from_cvss(1.0) == Severity.LOW
        assert Severity.from_cvss(3.9) == Severity.LOW

    def test_from_cvss_medium(self):
        """Test from_cvss with medium score."""
        assert Severity.from_cvss(4.0) == Severity.MEDIUM
        assert Severity.from_cvss(6.9) == Severity.MEDIUM

    def test_from_cvss_high(self):
        """Test from_cvss with high score."""
        assert Severity.from_cvss(7.0) == Severity.HIGH
        assert Severity.from_cvss(8.9) == Severity.HIGH

    def test_from_cvss_critical(self):
        """Test from_cvss with critical score."""
        assert Severity.from_cvss(9.0) == Severity.CRITICAL
        assert Severity.from_cvss(10.0) == Severity.CRITICAL


class TestVulnerabilityStatus:
    """Tests for VulnerabilityStatus enum."""

    def test_open_status(self):
        """Test open status."""
        assert VulnerabilityStatus.OPEN.value == "open"

    def test_in_progress_status(self):
        """Test in_progress status."""
        assert VulnerabilityStatus.IN_PROGRESS.value == "in_progress"

    def test_fixed_status(self):
        """Test fixed status."""
        assert VulnerabilityStatus.FIXED.value == "fixed"

    def test_ignored_status(self):
        """Test ignored status."""
        assert VulnerabilityStatus.IGNORED.value == "ignored"

    def test_false_positive_status(self):
        """Test false_positive status."""
        assert VulnerabilityStatus.FALSE_POSITIVE.value == "false_positive"


# =============================================================================
# CVSS METRICS TESTS
# =============================================================================


class TestCVSSMetrics:
    """Tests for CVSSMetrics dataclass."""

    def test_creation_minimal(self):
        """Test minimal CVSS metrics creation."""
        metrics = CVSSMetrics(score=7.5)
        assert metrics.score == 7.5
        assert metrics.vector == ""
        assert metrics.attack_vector == ""

    def test_creation_full(self):
        """Test full CVSS metrics creation."""
        metrics = CVSSMetrics(
            score=9.8,
            vector="AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
            attack_vector="Network",
            attack_complexity="Low",
            privileges_required="None",
            user_interaction="None",
            scope="Unchanged",
            confidentiality_impact="High",
            integrity_impact="High",
            availability_impact="High",
        )
        assert metrics.score == 9.8
        assert metrics.attack_vector == "Network"
        assert metrics.confidentiality_impact == "High"


# =============================================================================
# CVE TESTS
# =============================================================================


class TestCVE:
    """Tests for CVE dataclass."""

    def test_creation_minimal(self):
        """Test minimal CVE creation."""
        cve = CVE(
            cve_id="CVE-2021-44228",
            description="Log4j vulnerability",
            severity=Severity.CRITICAL,
        )
        assert cve.cve_id == "CVE-2021-44228"
        assert cve.severity == Severity.CRITICAL
        assert cve.cvss is None
        assert cve.references == []
        assert cve.cwe_ids == []

    def test_creation_full(self):
        """Test full CVE creation."""
        now = datetime.now()
        cvss = CVSSMetrics(score=10.0)
        cve = CVE(
            cve_id="CVE-2021-44228",
            description="Log4j vulnerability",
            severity=Severity.CRITICAL,
            cvss=cvss,
            published_date=now,
            modified_date=now,
            references=["https://nvd.nist.gov/vuln/detail/CVE-2021-44228"],
            cwe_ids=["CWE-917"],
            affected_products=["log4j"],
        )
        assert cve.cvss == cvss
        assert len(cve.references) == 1
        assert "CWE-917" in cve.cwe_ids

    def test_cvss_score_with_cvss(self):
        """Test cvss_score property with CVSS."""
        cve = CVE(
            cve_id="CVE-2021-44228",
            description="Test",
            severity=Severity.CRITICAL,
            cvss=CVSSMetrics(score=10.0),
        )
        assert cve.cvss_score == 10.0

    def test_cvss_score_without_cvss(self):
        """Test cvss_score property without CVSS."""
        cve = CVE(
            cve_id="CVE-2021-44228",
            description="Test",
            severity=Severity.CRITICAL,
        )
        assert cve.cvss_score == 0.0


# =============================================================================
# DEPENDENCY TESTS
# =============================================================================


class TestDependency:
    """Tests for Dependency dataclass."""

    def test_creation_minimal(self):
        """Test minimal dependency creation."""
        dep = Dependency(
            name="requests",
            version="2.28.0",
            ecosystem="pypi",
        )
        assert dep.name == "requests"
        assert dep.version == "2.28.0"
        assert dep.ecosystem == "pypi"
        assert dep.is_direct is True
        assert dep.source_file is None
        assert dep.license is None

    def test_creation_full(self):
        """Test full dependency creation."""
        dep = Dependency(
            name="express",
            version="4.18.2",
            ecosystem="npm",
            source_file=Path("package.json"),
            is_direct=False,
            license="MIT",
        )
        assert dep.source_file == Path("package.json")
        assert dep.is_direct is False
        assert dep.license == "MIT"

    def test_package_url(self):
        """Test package_url property."""
        dep = Dependency(
            name="requests",
            version="2.28.0",
            ecosystem="pypi",
        )
        assert dep.package_url == "pkg:pypi/requests@2.28.0"

    def test_package_url_npm(self):
        """Test package_url for npm."""
        dep = Dependency(
            name="express",
            version="4.18.2",
            ecosystem="npm",
        )
        assert dep.package_url == "pkg:npm/express@4.18.2"


# =============================================================================
# VULNERABILITY TESTS
# =============================================================================


class TestVulnerability:
    """Tests for Vulnerability dataclass."""

    @pytest.fixture
    def sample_cve(self):
        """Create sample CVE."""
        return CVE(
            cve_id="CVE-2021-44228",
            description="Log4j vulnerability",
            severity=Severity.CRITICAL,
        )

    @pytest.fixture
    def sample_dependency(self):
        """Create sample dependency."""
        return Dependency(
            name="log4j",
            version="2.14.0",
            ecosystem="maven",
        )

    def test_creation_minimal(self, sample_cve, sample_dependency):
        """Test minimal vulnerability creation."""
        vuln = Vulnerability(
            cve=sample_cve,
            dependency=sample_dependency,
        )
        assert vuln.cve == sample_cve
        assert vuln.dependency == sample_dependency
        assert vuln.fixed_version is None
        assert vuln.status == VulnerabilityStatus.OPEN
        assert vuln.notes == ""

    def test_creation_full(self, sample_cve, sample_dependency):
        """Test full vulnerability creation."""
        vuln = Vulnerability(
            cve=sample_cve,
            dependency=sample_dependency,
            fixed_version="2.17.0",
            status=VulnerabilityStatus.FIXED,
            notes="Upgraded in PR #123",
        )
        assert vuln.fixed_version == "2.17.0"
        assert vuln.status == VulnerabilityStatus.FIXED
        assert vuln.notes == "Upgraded in PR #123"

    def test_is_fixable_true(self, sample_cve, sample_dependency):
        """Test is_fixable when fix available."""
        vuln = Vulnerability(
            cve=sample_cve,
            dependency=sample_dependency,
            fixed_version="2.17.0",
        )
        assert vuln.is_fixable is True

    def test_is_fixable_false(self, sample_cve, sample_dependency):
        """Test is_fixable when no fix available."""
        vuln = Vulnerability(
            cve=sample_cve,
            dependency=sample_dependency,
        )
        assert vuln.is_fixable is False

    def test_upgrade_path_with_fix(self, sample_cve, sample_dependency):
        """Test upgrade_path when fix available."""
        vuln = Vulnerability(
            cve=sample_cve,
            dependency=sample_dependency,
            fixed_version="2.17.0",
        )
        assert vuln.upgrade_path == "log4j: 2.14.0 -> 2.17.0"

    def test_upgrade_path_no_fix(self, sample_cve, sample_dependency):
        """Test upgrade_path when no fix available."""
        vuln = Vulnerability(
            cve=sample_cve,
            dependency=sample_dependency,
        )
        assert vuln.upgrade_path == "log4j: No fix available"


# =============================================================================
# SECURITY SCAN RESULT TESTS
# =============================================================================


class TestSecurityScanResult:
    """Tests for SecurityScanResult dataclass."""

    @pytest.fixture
    def sample_dependencies(self):
        """Create sample dependencies."""
        return [
            Dependency("requests", "2.28.0", "pypi"),
            Dependency("flask", "2.0.0", "pypi"),
            Dependency("django", "3.2.0", "pypi"),
        ]

    @pytest.fixture
    def sample_vulnerabilities(self):
        """Create sample vulnerabilities."""
        critical_cve = CVE("CVE-2021-44228", "Critical", Severity.CRITICAL)
        high_cve = CVE("CVE-2022-1234", "High", Severity.HIGH)
        medium_cve = CVE("CVE-2022-5678", "Medium", Severity.MEDIUM)
        low_cve = CVE("CVE-2022-9999", "Low", Severity.LOW)

        dep1 = Dependency("log4j", "2.14.0", "maven")
        dep2 = Dependency("requests", "2.20.0", "pypi")

        return [
            Vulnerability(critical_cve, dep1),
            Vulnerability(high_cve, dep1),
            Vulnerability(medium_cve, dep2),
            Vulnerability(low_cve, dep2),
        ]

    def test_creation_empty(self):
        """Test empty scan result creation."""
        result = SecurityScanResult()
        assert result.dependencies == []
        assert result.vulnerabilities == []
        assert result.errors == []

    def test_creation_full(self, sample_dependencies, sample_vulnerabilities):
        """Test full scan result creation."""
        result = SecurityScanResult(
            dependencies=sample_dependencies,
            vulnerabilities=sample_vulnerabilities,
            scan_duration_ms=1500.0,
        )
        assert len(result.dependencies) == 3
        assert len(result.vulnerabilities) == 4

    def test_total_dependencies(self, sample_dependencies):
        """Test total_dependencies property."""
        result = SecurityScanResult(dependencies=sample_dependencies)
        assert result.total_dependencies == 3

    def test_vulnerable_dependencies(self, sample_vulnerabilities):
        """Test vulnerable_dependencies property."""
        result = SecurityScanResult(vulnerabilities=sample_vulnerabilities)
        # Two unique dependencies: log4j and requests
        assert result.vulnerable_dependencies == 2

    def test_total_vulnerabilities(self, sample_vulnerabilities):
        """Test total_vulnerabilities property."""
        result = SecurityScanResult(vulnerabilities=sample_vulnerabilities)
        assert result.total_vulnerabilities == 4

    def test_critical_count(self, sample_vulnerabilities):
        """Test critical_count property."""
        result = SecurityScanResult(vulnerabilities=sample_vulnerabilities)
        assert result.critical_count == 1

    def test_high_count(self, sample_vulnerabilities):
        """Test high_count property."""
        result = SecurityScanResult(vulnerabilities=sample_vulnerabilities)
        assert result.high_count == 1

    def test_medium_count(self, sample_vulnerabilities):
        """Test medium_count property."""
        result = SecurityScanResult(vulnerabilities=sample_vulnerabilities)
        assert result.medium_count == 1

    def test_low_count(self, sample_vulnerabilities):
        """Test low_count property."""
        result = SecurityScanResult(vulnerabilities=sample_vulnerabilities)
        assert result.low_count == 1

    def test_get_by_severity(self, sample_vulnerabilities):
        """Test get_by_severity method."""
        result = SecurityScanResult(vulnerabilities=sample_vulnerabilities)
        critical = result.get_by_severity(Severity.CRITICAL)
        assert len(critical) == 1
        assert critical[0].cve.cve_id == "CVE-2021-44228"

    def test_get_fixable(self):
        """Test get_fixable method."""
        cve1 = CVE("CVE-2021-44228", "Critical", Severity.CRITICAL)
        cve2 = CVE("CVE-2022-1234", "High", Severity.HIGH)
        dep = Dependency("log4j", "2.14.0", "maven")

        result = SecurityScanResult(
            vulnerabilities=[
                Vulnerability(cve1, dep, fixed_version="2.17.0"),
                Vulnerability(cve2, dep),  # No fix available
            ]
        )
        fixable = result.get_fixable()
        assert len(fixable) == 1
        assert fixable[0].fixed_version == "2.17.0"


# =============================================================================
# SECURITY POLICY TESTS
# =============================================================================


class TestSecurityPolicy:
    """Tests for SecurityPolicy dataclass."""

    def test_default_policy(self):
        """Test default policy values."""
        policy = SecurityPolicy()
        assert policy.fail_on_critical is True
        assert policy.fail_on_high is True
        assert policy.fail_on_medium is False
        assert policy.fail_on_low is False
        assert policy.max_critical == 0
        assert policy.max_high == 0
        assert policy.max_medium == -1  # unlimited
        assert policy.max_low == -1

    def test_custom_policy(self):
        """Test custom policy configuration."""
        policy = SecurityPolicy(
            fail_on_critical=True,
            fail_on_high=False,
            fail_on_medium=True,
            max_medium=5,
            ignored_cves=["CVE-2021-44228"],
            ignored_dependencies=["test-package"],
        )
        assert policy.fail_on_high is False
        assert policy.fail_on_medium is True
        assert policy.max_medium == 5
        assert "CVE-2021-44228" in policy.ignored_cves

    def test_check_pass_empty_result(self):
        """Test policy check with empty result."""
        policy = SecurityPolicy()
        result = SecurityScanResult()
        passed, failures = policy.check(result)
        assert passed is True
        assert failures == []

    def test_check_fail_on_critical(self):
        """Test policy fails on critical vulnerability."""
        policy = SecurityPolicy(fail_on_critical=True)
        cve = CVE("CVE-2021-44228", "Critical", Severity.CRITICAL)
        dep = Dependency("log4j", "2.14.0", "maven")
        result = SecurityScanResult(vulnerabilities=[Vulnerability(cve, dep)])

        passed, failures = policy.check(result)
        assert passed is False
        assert "1 critical" in failures[0]

    def test_check_fail_on_high(self):
        """Test policy fails on high vulnerability."""
        policy = SecurityPolicy(fail_on_high=True)
        cve = CVE("CVE-2022-1234", "High", Severity.HIGH)
        dep = Dependency("requests", "2.20.0", "pypi")
        result = SecurityScanResult(vulnerabilities=[Vulnerability(cve, dep)])

        passed, failures = policy.check(result)
        assert passed is False
        assert "1 high" in failures[0]

    def test_check_fail_on_medium(self):
        """Test policy fails on medium when configured."""
        policy = SecurityPolicy(fail_on_medium=True)
        cve = CVE("CVE-2022-5678", "Medium", Severity.MEDIUM)
        dep = Dependency("flask", "2.0.0", "pypi")
        result = SecurityScanResult(vulnerabilities=[Vulnerability(cve, dep)])

        passed, failures = policy.check(result)
        assert passed is False
        assert "1 medium" in failures[0]

    def test_check_fail_on_low(self):
        """Test policy fails on low when configured."""
        policy = SecurityPolicy(fail_on_low=True)
        cve = CVE("CVE-2022-9999", "Low", Severity.LOW)
        dep = Dependency("package", "1.0.0", "pypi")
        result = SecurityScanResult(vulnerabilities=[Vulnerability(cve, dep)])

        passed, failures = policy.check(result)
        assert passed is False
        assert "1 low" in failures[0]

    def test_check_exceeds_max_critical(self):
        """Test policy fails when exceeding max critical."""
        policy = SecurityPolicy(fail_on_critical=False, max_critical=0)
        cve = CVE("CVE-2021-44228", "Critical", Severity.CRITICAL)
        dep = Dependency("log4j", "2.14.0", "maven")
        result = SecurityScanResult(vulnerabilities=[Vulnerability(cve, dep)])

        passed, failures = policy.check(result)
        assert passed is False
        assert "exceeds limit" in failures[0]

    def test_check_exceeds_max_high(self):
        """Test policy fails when exceeding max high."""
        policy = SecurityPolicy(fail_on_high=False, max_high=0)
        cve = CVE("CVE-2022-1234", "High", Severity.HIGH)
        dep = Dependency("requests", "2.20.0", "pypi")
        result = SecurityScanResult(vulnerabilities=[Vulnerability(cve, dep)])

        passed, failures = policy.check(result)
        assert passed is False
        assert "exceeds limit" in failures[0]

    def test_check_exceeds_max_medium(self):
        """Test policy fails when exceeding max medium."""
        policy = SecurityPolicy(max_medium=0)
        cve = CVE("CVE-2022-5678", "Medium", Severity.MEDIUM)
        dep = Dependency("flask", "2.0.0", "pypi")
        result = SecurityScanResult(vulnerabilities=[Vulnerability(cve, dep)])

        passed, failures = policy.check(result)
        assert passed is False
        assert "exceeds limit" in failures[0]

    def test_check_exceeds_max_low(self):
        """Test policy fails when exceeding max low."""
        policy = SecurityPolicy(max_low=0)
        cve = CVE("CVE-2022-9999", "Low", Severity.LOW)
        dep = Dependency("package", "1.0.0", "pypi")
        result = SecurityScanResult(vulnerabilities=[Vulnerability(cve, dep)])

        passed, failures = policy.check(result)
        assert passed is False
        assert "exceeds limit" in failures[0]

    def test_check_ignored_cve(self):
        """Test ignored CVE is excluded from policy check."""
        policy = SecurityPolicy(
            fail_on_critical=True,
            ignored_cves=["CVE-2021-44228"],
        )
        cve = CVE("CVE-2021-44228", "Critical", Severity.CRITICAL)
        dep = Dependency("log4j", "2.14.0", "maven")
        result = SecurityScanResult(vulnerabilities=[Vulnerability(cve, dep)])

        passed, failures = policy.check(result)
        assert passed is True
        assert failures == []

    def test_check_ignored_dependency(self):
        """Test ignored dependency is excluded from policy check."""
        policy = SecurityPolicy(
            fail_on_critical=True,
            ignored_dependencies=["log4j"],
        )
        cve = CVE("CVE-2021-44228", "Critical", Severity.CRITICAL)
        dep = Dependency("log4j", "2.14.0", "maven")
        result = SecurityScanResult(vulnerabilities=[Vulnerability(cve, dep)])

        passed, failures = policy.check(result)
        assert passed is True
        assert failures == []

    def test_check_multiple_failures(self):
        """Test policy returns multiple failure messages."""
        policy = SecurityPolicy(
            fail_on_critical=True,
            fail_on_high=True,
        )
        cve1 = CVE("CVE-2021-44228", "Critical", Severity.CRITICAL)
        cve2 = CVE("CVE-2022-1234", "High", Severity.HIGH)
        dep = Dependency("log4j", "2.14.0", "maven")
        result = SecurityScanResult(
            vulnerabilities=[
                Vulnerability(cve1, dep),
                Vulnerability(cve2, dep),
            ]
        )

        passed, failures = policy.check(result)
        assert passed is False
        assert len(failures) == 2

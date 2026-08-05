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

"""Security scanning protocol types.

Defines data structures for vulnerability scanning, CVE tracking,
and security reporting.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from victor.core.severity import SeverityLevel


class CVESeverity(Enum):
    """CVE severity levels (CVSS v3).

    This is the CANONICAL Severity enum for CVE/vulnerability classification.

    Renamed from Severity to be semantically distinct from other severity types:
    - CVESeverity (here): CVE/CVSS-based severity (NONE, LOW, MEDIUM, HIGH, CRITICAL)
    - AuditSeverity: Audit event severity (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - IaCSeverity: IaC issue severity (CRITICAL, HIGH, MEDIUM, LOW, INFO)
    - ReviewSeverity: Code review severity (ERROR, WARNING, INFO, HINT)
    """

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @classmethod
    def from_cvss(cls, score: float) -> "CVESeverity":
        """Convert CVSS score to severity level.

        Args:
            score: CVSS v3 score (0.0 - 10.0)

        Returns:
            Severity level
        """
        if score == 0.0:
            return cls.NONE
        elif score < 4.0:
            return cls.LOW
        elif score < 7.0:
            return cls.MEDIUM
        elif score < 9.0:
            return cls.HIGH
        else:
            return cls.CRITICAL

    def to_severity(self) -> "SeverityLevel":
        """Convert to unified SeverityLevel."""
        from victor.core.severity import SeverityLevel

        return SeverityLevel.from_cve_severity(self)


# Backward compatibility alias
Severity = CVESeverity


class VulnerabilityStatus(Enum):
    """Status of a vulnerability finding."""

    OPEN = "open"  # Not yet addressed
    IN_PROGRESS = "in_progress"  # Being fixed
    FIXED = "fixed"  # Fixed in code
    IGNORED = "ignored"  # Intentionally ignored
    FALSE_POSITIVE = "false_positive"  # Not actually vulnerable


@dataclass
class CVSSMetrics:
    """CVSS v3 metrics."""

    score: float
    vector: str = ""  # e.g., "AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"
    attack_vector: str = ""  # Network, Adjacent, Local, Physical
    attack_complexity: str = ""  # Low, High
    privileges_required: str = ""  # None, Low, High
    user_interaction: str = ""  # None, Required
    scope: str = ""  # Unchanged, Changed
    confidentiality_impact: str = ""  # None, Low, High
    integrity_impact: str = ""  # None, Low, High
    availability_impact: str = ""  # None, Low, High


@dataclass
class CVE:
    """Common Vulnerabilities and Exposures (CVE) entry."""

    cve_id: str  # e.g., "CVE-2021-44228"
    description: str
    severity: Severity
    cvss: Optional[CVSSMetrics] = None
    published_date: Optional[datetime] = None
    modified_date: Optional[datetime] = None
    references: list[str] = field(default_factory=list)
    cwe_ids: list[str] = field(default_factory=list)  # Weakness types
    affected_products: list[str] = field(default_factory=list)

    @property
    def cvss_score(self) -> float:
        """Get CVSS score."""
        return self.cvss.score if self.cvss else 0.0


@dataclass
class SecurityDependency:
    """A software dependency for security scanning.

    Renamed from Dependency to be semantically distinct:
    - SecurityDependency (here): Security scanning with ecosystem, license, package_url
    - PackageDependency (victor.deps.protocol): Package management with version tracking
    """

    name: str
    version: str
    ecosystem: str  # pypi, npm, cargo, maven, etc.
    source_file: Optional[Path] = None
    is_direct: bool = True  # Direct or transitive dependency
    license: Optional[str] = None

    @property
    def package_url(self) -> str:
        """Get Package URL (purl) format."""
        return f"pkg:{self.ecosystem}/{self.name}@{self.version}"


# Backward compatibility alias
Dependency = SecurityDependency


@dataclass
class Vulnerability:
    """A vulnerability finding in a dependency."""

    cve: CVE
    dependency: Dependency
    fixed_version: Optional[str] = None  # Version that fixes the vulnerability
    status: VulnerabilityStatus = VulnerabilityStatus.OPEN
    notes: str = ""

    @property
    def is_fixable(self) -> bool:
        """Whether a fix is available."""
        return self.fixed_version is not None

    @property
    def upgrade_path(self) -> str:
        """Suggested upgrade path."""
        if self.fixed_version:
            return f"{self.dependency.name}: {self.dependency.version} -> {self.fixed_version}"
        return f"{self.dependency.name}: No fix available"


@dataclass
class SecurityScanResult:
    """Result of a security scan."""

    dependencies: list[Dependency] = field(default_factory=list)
    vulnerabilities: list[Vulnerability] = field(default_factory=list)
    scan_timestamp: datetime = field(default_factory=datetime.now)
    scan_duration_ms: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def total_dependencies(self) -> int:
        """Total number of dependencies scanned."""
        return len(self.dependencies)

    @property
    def vulnerable_dependencies(self) -> int:
        """Number of dependencies with vulnerabilities."""
        vuln_deps = {v.dependency.name for v in self.vulnerabilities}
        return len(vuln_deps)

    @property
    def total_vulnerabilities(self) -> int:
        """Total number of vulnerabilities found."""
        return len(self.vulnerabilities)

    @property
    def critical_count(self) -> int:
        """Number of critical severity vulnerabilities."""
        return sum(1 for v in self.vulnerabilities if v.cve.severity == Severity.CRITICAL)

    @property
    def high_count(self) -> int:
        """Number of high severity vulnerabilities."""
        return sum(1 for v in self.vulnerabilities if v.cve.severity == Severity.HIGH)

    @property
    def medium_count(self) -> int:
        """Number of medium severity vulnerabilities."""
        return sum(1 for v in self.vulnerabilities if v.cve.severity == Severity.MEDIUM)

    @property
    def low_count(self) -> int:
        """Number of low severity vulnerabilities."""
        return sum(1 for v in self.vulnerabilities if v.cve.severity == Severity.LOW)

    def get_by_severity(self, severity: Severity) -> list[Vulnerability]:
        """Get vulnerabilities by severity.

        Args:
            severity: Severity level to filter by

        Returns:
            List of vulnerabilities with that severity
        """
        return [v for v in self.vulnerabilities if v.cve.severity == severity]

    def get_fixable(self) -> list[Vulnerability]:
        """Get vulnerabilities that have fixes available."""
        return [v for v in self.vulnerabilities if v.is_fixable]


@dataclass
class SecurityPolicy:
    """Security policy configuration."""

    # Fail thresholds
    fail_on_critical: bool = True
    fail_on_high: bool = True
    fail_on_medium: bool = False
    fail_on_low: bool = False

    # Maximum allowed vulnerabilities by severity
    max_critical: int = 0
    max_high: int = 0
    max_medium: int = -1  # -1 = unlimited
    max_low: int = -1

    # Ignored CVEs
    ignored_cves: list[str] = field(default_factory=list)

    # Ignored dependencies
    ignored_dependencies: list[str] = field(default_factory=list)

    def check(self, result: SecurityScanResult) -> tuple[bool, list[str]]:
        """Check if scan result passes policy.

        Args:
            result: Scan result to check

        Returns:
            Tuple of (passed, list of failure messages)
        """
        failures = []

        # Filter out ignored vulnerabilities
        active_vulns = [
            v
            for v in result.vulnerabilities
            if v.cve.cve_id not in self.ignored_cves
            and v.dependency.name not in self.ignored_dependencies
        ]

        critical = sum(1 for v in active_vulns if v.cve.severity == Severity.CRITICAL)
        high = sum(1 for v in active_vulns if v.cve.severity == Severity.HIGH)
        medium = sum(1 for v in active_vulns if v.cve.severity == Severity.MEDIUM)
        low = sum(1 for v in active_vulns if v.cve.severity == Severity.LOW)

        if self.fail_on_critical and critical > 0:
            failures.append(f"Found {critical} critical vulnerabilities")
        elif self.max_critical >= 0 and critical > self.max_critical:
            failures.append(
                f"Critical vulnerabilities ({critical}) exceeds limit ({self.max_critical})"
            )

        if self.fail_on_high and high > 0:
            failures.append(f"Found {high} high severity vulnerabilities")
        elif self.max_high >= 0 and high > self.max_high:
            failures.append(f"High vulnerabilities ({high}) exceeds limit ({self.max_high})")

        if self.fail_on_medium and medium > 0:
            failures.append(f"Found {medium} medium severity vulnerabilities")
        elif self.max_medium >= 0 and medium > self.max_medium:
            failures.append(f"Medium vulnerabilities ({medium}) exceeds limit ({self.max_medium})")

        if self.fail_on_low and low > 0:
            failures.append(f"Found {low} low severity vulnerabilities")
        elif self.max_low >= 0 and low > self.max_low:
            failures.append(f"Low vulnerabilities ({low}) exceeds limit ({self.max_low})")

        return len(failures) == 0, failures

from __future__ import annotations

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

"""Security manager for orchestrating security scanning operations.

Provides a high-level API for security scanning and vulnerability management.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from victor.security.protocol import (
    SecurityPolicy,
    SecurityScanResult,
    Severity,
)
from victor.security.scanner import get_scanner

logger = logging.getLogger(__name__)


class SecurityManager:
    """High-level manager for security scanning operations."""

    def __init__(
        self,
        project_root: Optional[Path] = None,
        policy: Optional[SecurityPolicy] = None,
        offline: bool = False,
    ):
        """Initialize the security manager.

        Args:
            project_root: Root directory of the project
            policy: Security policy configuration
            offline: Whether to use offline/air-gapped mode
        """
        self.project_root = project_root or Path.cwd()
        self.policy = policy or SecurityPolicy()
        self._scanner = get_scanner(offline=offline)
        self._current_result: Optional[SecurityScanResult] = None

    @property
    def current_result(self) -> Optional[SecurityScanResult]:
        """Get the most recent scan result."""
        return self._current_result

    async def scan(
        self,
        include_dev: bool = True,
    ) -> SecurityScanResult:
        """Scan the project for vulnerabilities.

        Args:
            include_dev: Whether to include dev dependencies

        Returns:
            SecurityScanResult with findings
        """
        result = await self._scanner.scan(
            self.project_root,
            include_dev=include_dev,
        )
        self._current_result = result
        return result

    async def scan_file(self, file_path: Path) -> SecurityScanResult:
        """Scan a single dependency file.

        Args:
            file_path: Path to dependency file

        Returns:
            SecurityScanResult
        """
        result = await self._scanner.scan_file(file_path)
        self._current_result = result
        return result

    def check_policy(
        self,
        result: Optional[SecurityScanResult] = None,
    ) -> tuple[bool, list[str]]:
        """Check if scan result passes security policy.

        Args:
            result: Scan result to check (uses current if not provided)

        Returns:
            Tuple of (passed, list of failure messages)
        """
        result = result or self._current_result
        if result is None:
            return False, ["No scan result available"]

        return self.policy.check(result)

    def generate_report(
        self,
        result: Optional[SecurityScanResult] = None,
        format: str = "text",
    ) -> str:
        """Generate a security report.

        Args:
            result: Scan result (uses current if not provided)
            format: Report format (text, json, markdown)

        Returns:
            Report string
        """
        result = result or self._current_result
        if result is None:
            return "No scan result available"

        if format == "json":
            return self._generate_json_report(result)
        elif format == "markdown":
            return self._generate_markdown_report(result)
        else:
            return self._generate_text_report(result)

    def _generate_text_report(self, result: SecurityScanResult) -> str:
        """Generate text report."""
        lines = []

        lines.append("=" * 70)
        lines.append("SECURITY SCAN REPORT")
        lines.append("=" * 70)
        lines.append("")

        # Summary
        lines.append("SUMMARY")
        lines.append("-" * 40)
        lines.append(f"Dependencies scanned: {result.total_dependencies}")
        lines.append(f"Vulnerable packages:  {result.vulnerable_dependencies}")
        lines.append(f"Total vulnerabilities: {result.total_vulnerabilities}")
        lines.append("")

        # Severity breakdown
        lines.append("SEVERITY BREAKDOWN")
        lines.append("-" * 40)
        lines.append(f"  Critical: {result.critical_count}")
        lines.append(f"  High:     {result.high_count}")
        lines.append(f"  Medium:   {result.medium_count}")
        lines.append(f"  Low:      {result.low_count}")
        lines.append("")

        # Policy check
        passed, failures = self.policy.check(result)
        if failures:
            lines.append("POLICY VIOLATIONS")
            lines.append("-" * 40)
            for failure in failures:
                lines.append(f"  FAIL: {failure}")
            lines.append("")
        else:
            lines.append("Policy Check: PASS")
            lines.append("")

        # Vulnerabilities by severity
        for severity in [
            Severity.CRITICAL,
            Severity.HIGH,
            Severity.MEDIUM,
            Severity.LOW,
        ]:
            vulns = result.get_by_severity(severity)
            if vulns:
                lines.append(f"{severity.value.upper()} VULNERABILITIES")
                lines.append("-" * 40)
                for vuln in vulns:
                    lines.append(
                        f"  [{vuln.cve.cve_id}] {vuln.dependency.name}@{vuln.dependency.version}"
                    )
                    lines.append(f"    {vuln.cve.description[:100]}...")
                    if vuln.fixed_version:
                        lines.append(f"    Fix: Upgrade to {vuln.fixed_version}")
                    lines.append("")

        # Errors
        if result.errors:
            lines.append("ERRORS")
            lines.append("-" * 40)
            for error in result.errors:
                lines.append(f"  {error}")
            lines.append("")

        lines.append("=" * 70)
        lines.append(f"Scan completed in {result.scan_duration_ms:.0f}ms")

        return "\n".join(lines)

    def _generate_markdown_report(self, result: SecurityScanResult) -> str:
        """Generate markdown report."""
        lines = []

        lines.append("# Security Scan Report")
        lines.append("")

        # Summary
        lines.append("## Summary")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Dependencies | {result.total_dependencies} |")
        lines.append(f"| Vulnerable | {result.vulnerable_dependencies} |")
        lines.append(f"| Vulnerabilities | {result.total_vulnerabilities} |")
        lines.append(f"| Critical | {result.critical_count} |")
        lines.append(f"| High | {result.high_count} |")
        lines.append(f"| Medium | {result.medium_count} |")
        lines.append(f"| Low | {result.low_count} |")
        lines.append("")

        # Policy
        passed, failures = self.policy.check(result)
        lines.append("## Policy Status")
        lines.append("")
        if passed:
            lines.append("✅ **PASS** - All security policies met")
        else:
            lines.append("❌ **FAIL** - Policy violations found:")
            for failure in failures:
                lines.append(f"- {failure}")
        lines.append("")

        # Vulnerabilities
        if result.vulnerabilities:
            lines.append("## Vulnerabilities")
            lines.append("")

            for severity in [
                Severity.CRITICAL,
                Severity.HIGH,
                Severity.MEDIUM,
                Severity.LOW,
            ]:
                vulns = result.get_by_severity(severity)
                if vulns:
                    lines.append(f"### {severity.value.capitalize()}")
                    lines.append("")

                    for vuln in vulns:
                        lines.append(f"#### {vuln.cve.cve_id}")
                        lines.append("")
                        lines.append(
                            f"**Package:** {vuln.dependency.name}@{vuln.dependency.version}"
                        )
                        lines.append("")
                        lines.append(vuln.cve.description)
                        lines.append("")
                        if vuln.fixed_version:
                            lines.append(f"**Fix:** Upgrade to version {vuln.fixed_version}")
                            lines.append("")
                        if vuln.cve.references:
                            lines.append("**References:**")
                            for ref in vuln.cve.references[:3]:
                                lines.append(f"- {ref}")
                            lines.append("")

        return "\n".join(lines)

    def _generate_json_report(self, result: SecurityScanResult) -> str:
        """Generate JSON report."""
        passed, failures = self.policy.check(result)

        data = {
            "summary": {
                "dependencies": result.total_dependencies,
                "vulnerable_dependencies": result.vulnerable_dependencies,
                "total_vulnerabilities": result.total_vulnerabilities,
                "by_severity": {
                    "critical": result.critical_count,
                    "high": result.high_count,
                    "medium": result.medium_count,
                    "low": result.low_count,
                },
                "scan_duration_ms": result.scan_duration_ms,
                "timestamp": result.scan_timestamp.isoformat(),
            },
            "policy": {
                "passed": passed,
                "failures": failures,
            },
            "vulnerabilities": [
                {
                    "cve_id": v.cve.cve_id,
                    "severity": v.cve.severity.value,
                    "cvss_score": v.cve.cvss_score,
                    "description": v.cve.description,
                    "package": v.dependency.name,
                    "version": v.dependency.version,
                    "ecosystem": v.dependency.ecosystem,
                    "fixed_version": v.fixed_version,
                    "references": v.cve.references,
                }
                for v in result.vulnerabilities
            ],
            "errors": result.errors,
        }

        return json.dumps(data, indent=2)

    def save_report(
        self,
        output_path: Path,
        format: str = "json",
        result: Optional[SecurityScanResult] = None,
    ) -> None:
        """Save report to file.

        Args:
            output_path: Output file path
            format: Report format
            result: Scan result (uses current if not provided)
        """
        report = self.generate_report(result, format)
        output_path.write_text(report)
        logger.info(f"Report saved to {output_path}")

    def get_fix_commands(
        self,
        result: Optional[SecurityScanResult] = None,
    ) -> dict[str, list[str]]:
        """Get commands to fix vulnerabilities by ecosystem.

        Args:
            result: Scan result (uses current if not provided)

        Returns:
            Dict mapping ecosystem to list of fix commands
        """
        result = result or self._current_result
        if result is None:
            return {}

        commands: dict[str, list[str]] = {}
        fixable = result.get_fixable()

        for vuln in fixable:
            eco = vuln.dependency.ecosystem
            if eco not in commands:
                commands[eco] = []

            if eco == "pypi":
                commands[eco].append(f"pip install {vuln.dependency.name}=={vuln.fixed_version}")
            elif eco == "npm":
                commands[eco].append(f"npm install {vuln.dependency.name}@{vuln.fixed_version}")
            elif eco == "cargo":
                commands[eco].append(f"cargo update -p {vuln.dependency.name}")
            elif eco == "go":
                commands[eco].append(f"go get {vuln.dependency.name}@v{vuln.fixed_version}")

        return commands


# Global manager
_security_manager: Optional[SecurityManager] = None


def get_security_manager(
    project_root: Optional[Path] = None,
    offline: bool = False,
) -> SecurityManager:
    """Get the global security manager.

    Args:
        project_root: Project root directory
        offline: Whether to use offline mode

    Returns:
        SecurityManager instance
    """
    global _security_manager
    if _security_manager is None or (
        project_root and _security_manager.project_root != project_root
    ):
        _security_manager = SecurityManager(
            project_root=project_root,
            offline=offline,
        )
    return _security_manager


def reset_security_manager() -> None:
    """Reset the global security manager."""
    global _security_manager
    _security_manager = None

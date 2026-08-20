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

"""IaC Scanner Manager - Orchestrates security scanning of IaC files.

This module provides the IaCManager class that coordinates multiple
scanners to provide comprehensive security analysis of infrastructure
configurations.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from victor.config.settings import get_project_paths, load_settings

from .protocol import (
    Category,
    IaCConfig,
    IaCFinding,
    IaCPlatform,
    ScanPolicy,
    ScanResult,
    Severity,
)
from .scanners import get_all_scanners, get_scanner

logger = logging.getLogger(__name__)


class IaCManager:
    """Manager for IaC security scanning.

    This class orchestrates multiple scanners to provide:
    - Multi-platform IaC security scanning
    - Policy-based filtering
    - Historical tracking of findings
    - Remediation guidance

    Configuration is driven by settings.py for consistency with Victor.
    """

    def __init__(self, root_path: str | Path | None = None):
        """Initialize the IaC manager.

        Args:
            root_path: Root directory of the project. Defaults to current directory.
        """
        self.root_path = Path(root_path) if root_path else Path.cwd()
        self._settings = load_settings()
        self._paths = get_project_paths(self.root_path)
        self._findings_history_file = self._paths.project_victor_dir / "iac_findings.json"
        self._policy_file = self._paths.project_victor_dir / "iac_policy.json"

    async def detect_platforms(self) -> list[IaCPlatform]:
        """Detect which IaC platforms are used in the project.

        Returns:
            List of detected IaC platforms
        """
        detected = []
        for scanner in get_all_scanners():
            files = await scanner.detect_files(self.root_path)
            if files:
                detected.append(scanner.platform)
                logger.debug(f"Detected {scanner.platform.value}: {len(files)} file(s)")

        return detected

    async def scan(
        self,
        platforms: list[IaCPlatform] | None = None,
        policy: ScanPolicy | None = None,
    ) -> ScanResult:
        """Perform comprehensive IaC security scan.

        Args:
            platforms: Specific platforms to scan. If None, auto-detects all.
            policy: Scan policy for filtering. If None, uses default or stored policy.

        Returns:
            Complete scan result with findings
        """
        start_time = time.time()
        configs: list[IaCConfig] = []
        findings: list[IaCFinding] = []

        # Load policy
        if policy is None:
            policy = await self._load_policy()

        # Determine platforms to scan
        if platforms is None:
            platforms = await self.detect_platforms()

        # Filter by enabled platforms in policy
        platforms = [p for p in platforms if p in policy.enabled_platforms]

        # Scan each platform
        files_scanned = 0
        total_resources = 0

        for platform in platforms:
            scanner = get_scanner(platform)
            if not scanner:
                continue

            config_files = await scanner.detect_files(self.root_path)

            # Filter by excluded paths
            config_files = [
                f for f in config_files if not any(excl in str(f) for excl in policy.excluded_paths)
            ]

            for config_file in config_files:
                try:
                    config = await scanner.parse_config(config_file)
                    configs.append(config)
                    files_scanned += 1
                    total_resources += len(config.resources)

                    # Scan for findings
                    file_findings = await scanner.scan(config, policy)

                    # Filter findings by policy
                    filtered_findings = self._filter_findings(file_findings, policy)
                    findings.extend(filtered_findings)

                    logger.info(
                        f"Scanned {config_file.name}: "
                        f"{len(config.resources)} resources, "
                        f"{len(filtered_findings)} findings"
                    )

                except Exception as e:
                    logger.error(f"Error scanning {config_file}: {e}")
                    findings.append(
                        IaCFinding(
                            rule_id="SCAN-ERROR",
                            severity=Severity.INFO,
                            category=Category.BEST_PRACTICE,
                            message=f"Failed to scan file: {e}",
                            description=str(e),
                            file_path=config_file,
                        )
                    )

        duration_ms = int((time.time() - start_time) * 1000)

        result = ScanResult(
            configs=configs,
            findings=findings,
            files_scanned=files_scanned,
            total_resources=total_resources,
            scan_duration_ms=duration_ms,
        )

        # Store findings for comparison
        await self._save_findings(result)

        return result

    async def scan_file(
        self, file_path: str | Path, policy: ScanPolicy | None = None
    ) -> list[IaCFinding]:
        """Scan a single IaC file.

        Args:
            file_path: Path to the file to scan
            policy: Optional scan policy

        Returns:
            List of findings for the file
        """
        path = Path(file_path)
        if not path.exists():
            return [
                IaCFinding(
                    rule_id="FILE-NOT-FOUND",
                    severity=Severity.INFO,
                    category=Category.BEST_PRACTICE,
                    message=f"File not found: {file_path}",
                    description="The specified file does not exist",
                    file_path=path,
                )
            ]

        # Detect platform
        for scanner in get_all_scanners():
            files = await scanner.detect_files(path.parent)
            if path in files:
                config = await scanner.parse_config(path)
                findings = await scanner.scan(config, policy)
                return self._filter_findings(findings, policy or ScanPolicy())

        return [
            IaCFinding(
                rule_id="UNKNOWN-FORMAT",
                severity=Severity.INFO,
                category=Category.BEST_PRACTICE,
                message=f"Unknown IaC format: {path.name}",
                description="File format not recognized as IaC",
                file_path=path,
            )
        ]

    def _filter_findings(self, findings: list[IaCFinding], policy: ScanPolicy) -> list[IaCFinding]:
        """Filter findings based on policy.

        Args:
            findings: Raw findings from scanner
            policy: Scan policy

        Returns:
            Filtered findings
        """
        severity_order = [
            Severity.INFO,
            Severity.LOW,
            Severity.MEDIUM,
            Severity.HIGH,
            Severity.CRITICAL,
        ]
        min_index = severity_order.index(policy.min_severity)

        filtered = []
        for finding in findings:
            # Filter by severity
            if severity_order.index(finding.severity) < min_index:
                continue

            # Filter by excluded rules
            if finding.rule_id in policy.excluded_rules:
                continue

            filtered.append(finding)

        return filtered

    async def compare_scans(self, baseline: ScanResult | None = None) -> dict[str, Any]:
        """Compare current scan against baseline.

        Args:
            baseline: Previous scan result. If None, uses stored history.

        Returns:
            Comparison results
        """
        current = await self.scan()

        if baseline is None:
            baseline = await self._load_previous_scan()

        if baseline is None:
            return {
                "error": "No baseline scan available",
                "current": current.to_dict(),
            }

        # Calculate deltas
        current_rules = {f.rule_id for f in current.findings}
        baseline_rules = {f.rule_id for f in baseline.findings}

        new_findings = [f for f in current.findings if f.rule_id not in baseline_rules]
        resolved_findings = [f for f in baseline.findings if f.rule_id not in current_rules]

        return {
            "current_total": len(current.findings),
            "baseline_total": len(baseline.findings),
            "new_findings": len(new_findings),
            "resolved_findings": len(resolved_findings),
            "critical_delta": current.critical_count - baseline.critical_count,
            "high_delta": current.high_count - baseline.high_count,
            "new_findings_details": [f.to_dict() for f in new_findings[:10]],
            "resolved_findings_details": [f.to_dict() for f in resolved_findings[:10]],
            "improved": len(new_findings) < len(resolved_findings),
        }

    async def get_summary(self) -> dict[str, Any]:
        """Get a high-level security summary.

        Returns:
            Summary dictionary with key metrics
        """
        result = await self.scan()

        # Group findings by category
        by_category: dict[str, int] = {}
        for finding in result.findings:
            by_category[finding.category.value] = by_category.get(finding.category.value, 0) + 1

        # Group by file
        by_file: dict[str, int] = {}
        for finding in result.findings:
            file_name = finding.file_path.name
            by_file[file_name] = by_file.get(file_name, 0) + 1

        # Top rules triggered
        by_rule: dict[str, int] = {}
        for finding in result.findings:
            by_rule[finding.rule_id] = by_rule.get(finding.rule_id, 0) + 1
        top_rules = sorted(by_rule.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "files_scanned": result.files_scanned,
            "total_resources": result.total_resources,
            "scan_duration_ms": result.scan_duration_ms,
            "total_findings": len(result.findings),
            "by_severity": {
                "critical": result.critical_count,
                "high": result.high_count,
                "medium": result.medium_count,
                "low": result.low_count,
                "info": result.info_count,
            },
            "by_category": by_category,
            "by_file": dict(sorted(by_file.items(), key=lambda x: x[1], reverse=True)[:5]),
            "top_rules": top_rules,
            "risk_score": self._calculate_risk_score(result),
        }

    def _calculate_risk_score(self, result: ScanResult) -> int:
        """Calculate overall risk score (0-100).

        Args:
            result: Scan result

        Returns:
            Risk score from 0 (low risk) to 100 (high risk)
        """
        if result.files_scanned == 0:
            return 0

        # Weight by severity
        weighted_score = (
            result.critical_count * 40
            + result.high_count * 20
            + result.medium_count * 5
            + result.low_count * 1
        )

        # Normalize to 0-100
        max_possible = result.files_scanned * 100
        score = min(100, int((weighted_score / max(max_possible, 1)) * 100))

        return score

    async def _load_policy(self) -> ScanPolicy:
        """Load scan policy from storage."""
        if not self._policy_file.exists():
            return ScanPolicy()

        try:
            with open(self._policy_file, encoding="utf-8") as f:
                data = json.load(f)
                return ScanPolicy(
                    enabled_platforms=[IaCPlatform(p) for p in data.get("enabled_platforms", [])],
                    min_severity=Severity(data.get("min_severity", "low")),
                    excluded_rules=data.get("excluded_rules", []),
                    excluded_paths=data.get("excluded_paths", []),
                    fail_on_severity=Severity(data.get("fail_on_severity", "high")),
                )
        except Exception as e:
            logger.warning(f"Failed to load policy: {e}")
            return ScanPolicy()

    async def save_policy(self, policy: ScanPolicy) -> None:
        """Save scan policy to storage.

        Args:
            policy: Policy to save
        """
        self._policy_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self._policy_file, "w", encoding="utf-8") as f:
            json.dump(policy.to_dict(), f, indent=2)

        logger.info(f"Saved IaC scan policy to {self._policy_file}")

    async def _save_findings(self, result: ScanResult) -> None:
        """Save scan findings to history."""
        self._findings_history_file.parent.mkdir(parents=True, exist_ok=True)

        # Load existing history
        history = []
        if self._findings_history_file.exists():
            try:
                with open(self._findings_history_file, encoding="utf-8") as f:
                    history = json.load(f)
            except Exception:
                pass

        # Add current scan
        history.append(
            {
                "scanned_at": result.scanned_at.isoformat(),
                "files_scanned": result.files_scanned,
                "total_findings": len(result.findings),
                "critical": result.critical_count,
                "high": result.high_count,
                "medium": result.medium_count,
                "low": result.low_count,
            }
        )

        # Keep last 30 scans
        history = history[-30:]

        with open(self._findings_history_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    async def _load_previous_scan(self) -> ScanResult | None:
        """Load previous scan from history."""
        if not self._findings_history_file.exists():
            return None

        try:
            with open(self._findings_history_file, encoding="utf-8") as f:
                history = json.load(f)
                if len(history) < 2:
                    return None

                # Return second-to-last entry as a minimal ScanResult
                prev = history[-2]
                return ScanResult(
                    configs=[],
                    findings=[],  # We don't store full findings
                    files_scanned=prev.get("files_scanned", 0),
                    total_resources=0,
                    scan_duration_ms=0,
                    scanned_at=datetime.fromisoformat(prev["scanned_at"]),
                    critical_count=prev.get("critical", 0),
                    high_count=prev.get("high", 0),
                    medium_count=prev.get("medium", 0),
                    low_count=prev.get("low", 0),
                )
        except Exception as e:
            logger.warning(f"Failed to load previous scan: {e}")
            return None

    async def should_fail_ci(self, result: ScanResult | None = None) -> tuple[bool, str]:
        """Determine if CI should fail based on findings.

        Args:
            result: Scan result to check. If None, performs new scan.

        Returns:
            Tuple of (should_fail, reason)
        """
        if result is None:
            result = await self.scan()

        policy = await self._load_policy()
        severity_order = [
            Severity.INFO,
            Severity.LOW,
            Severity.MEDIUM,
            Severity.HIGH,
            Severity.CRITICAL,
        ]
        fail_index = severity_order.index(policy.fail_on_severity)

        blocking_findings = [
            f for f in result.findings if severity_order.index(f.severity) >= fail_index
        ]

        if blocking_findings:
            reason = (
                f"Found {len(blocking_findings)} findings at or above "
                f"{policy.fail_on_severity.value} severity"
            )
            return True, reason

        return False, "No blocking findings"

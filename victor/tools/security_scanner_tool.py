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

"""Security scanning tool for code analysis.

Features:
- Secret and credential detection
- Dependency vulnerability scanning
- Configuration security checks
- Comprehensive security audits
"""

import re
from victor.core.json_utils import json_loads
from pathlib import Path
from typing import Any, Dict, List
import logging

from victor.tools.base import AccessMode, DangerLevel, Priority
from victor.tools.decorators import tool
from victor.tools.common import gather_files_by_pattern

logger = logging.getLogger(__name__)


# Common secret patterns
SECRET_PATTERNS = {
    "api_key": r"(?i)(api[_-]?key|apikey)\s*[:=]\s*['\"]([a-zA-Z0-9_\-]{20,})['\"]",
    "password": r"(?i)(password|passwd|pwd)\s*[:=]\s*['\"]([^'\"]{8,})['\"]",
    "token": r"(?i)(token|auth[_-]?token)\s*[:=]\s*['\"]([a-zA-Z0-9_\-\.]{20,})['\"]",
    "private_key": r"-----BEGIN (RSA |EC )?PRIVATE KEY-----",
    "aws_key": r"(?i)(aws[_-]?access[_-]?key[_-]?id)\s*[:=]\s*['\"]([A-Z0-9]{20})['\"]",
    "github_token": r"(?i)(github[_-]?token)\s*[:=]\s*['\"]([a-zA-Z0-9_]{40})['\"]",
}

# Configuration security checks
CONFIG_CHECKS = {
    "debug_mode": r"(?i)(debug|DEBUG)\s*[:=]\s*(true|True|1)",
    "insecure_protocol": r"(?i)http://",
    "hardcoded_ip": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
}


def _scan_file_for_secrets(file_path: Path) -> List[Dict[str, Any]]:
    """Scan a file for potential secrets."""
    findings = []

    try:
        content = file_path.read_text()

        for secret_type, pattern in SECRET_PATTERNS.items():
            matches = re.finditer(pattern, content)
            for match in matches:
                findings.append(
                    {
                        "type": secret_type,
                        "file": str(file_path),
                        "line": content[: match.start()].count("\n") + 1,
                        "severity": "high",
                        "message": f"Potential {secret_type} detected",
                    }
                )
    except Exception as e:
        logger.warning(f"Error scanning {file_path}: {e}")

    return findings


def _scan_file_for_config_issues(file_path: Path) -> List[Dict[str, Any]]:
    """Scan a file for configuration security issues."""
    findings = []

    try:
        content = file_path.read_text()

        for check_name, pattern in CONFIG_CHECKS.items():
            matches = re.finditer(pattern, content)
            for match in matches:
                findings.append(
                    {
                        "type": check_name,
                        "file": str(file_path),
                        "line": content[: match.start()].count("\n") + 1,
                        "severity": "medium",
                        "message": f"Configuration issue: {check_name}",
                    }
                )
    except Exception as e:
        logger.warning(f"Error scanning {file_path}: {e}")

    return findings


@tool(
    category="security",
    priority=Priority.MEDIUM,  # Task-specific security analysis
    access_mode=AccessMode.READONLY,  # Only reads files for scanning
    danger_level=DangerLevel.SAFE,  # No side effects
    keywords=["security", "scan", "vulnerability", "secret", "audit"],
    mandatory_keywords=[
        "security scan",
        "scan for vulnerabilities",
        "check security",
    ],  # Force inclusion
    task_types=["security", "analysis"],  # Classification-aware selection
    stages=["analysis"],  # Conversation stages where relevant
)
async def scan(
    path: str,
    scan_types: List[str] = None,
    file_pattern: str = "*.py",
    severity_threshold: str = "low",
    requirements_file: str = "requirements.txt",
    dependency_scan: bool = False,
    iac_scan: bool = False,
) -> Dict[str, Any]:
    """
    Comprehensive security scanning for code analysis.

    Performs security scans including secret detection, dependency vulnerability
    checks, and configuration security analysis. Consolidates multiple scan types
    into a single unified interface.

    Args:
        path: Directory or file path to scan.
        scan_types: List of scan types to perform. Options: "secrets", "dependencies",
            "config", "all". Defaults to ["all"].
        file_pattern: Glob pattern for files to scan (default: *.py).
        severity_threshold: Minimum severity to report: "low", "medium", "high".
            Defaults to "low" (report all).
        requirements_file: Path to requirements file for dependency scan
            (default: requirements.txt).

    Returns:
        Dictionary containing:
        - success: Whether scan completed
        - scan_types_performed: List of scan types executed
        - results: Dictionary with results for each scan type
        - total_issues: Total security issues found across all scans
        - issues_by_severity: Count of issues grouped by severity
        - formatted_report: Human-readable comprehensive security report
        - error: Error message if failed

    Examples:
        # Scan for secrets only
        security_scan("./src", scan_types=["secrets"])

        # Comprehensive scan
        security_scan("./", scan_types=["all"])

        # Secrets and config, high severity only
        security_scan("./src", scan_types=["secrets", "config"], severity_threshold="high")
    """
    if not path:
        return {"success": False, "error": "Missing required parameter: path"}

    if scan_types is None:
        scan_types = ["all"]

    # Expand "all" to all scan types
    if "all" in scan_types:
        scan_types = ["secrets", "dependencies", "config"]

    path_obj = Path(path)
    if not path_obj.exists():
        return {"success": False, "error": f"Path not found: {path}"}

    results = {}
    all_findings = []

    # Secrets scan
    if "secrets" in scan_types:
        if path_obj.is_file():
            findings = _scan_file_for_secrets(path_obj)
            results["secrets"] = {
                "files_scanned": 1,
                "findings": findings,
                "count": len(findings),
            }
            all_findings.extend(findings)
        else:
            # Use gather_files_by_pattern to exclude venv, node_modules, etc.
            files = gather_files_by_pattern(path_obj, file_pattern)
            findings = []
            for file in files:
                findings.extend(_scan_file_for_secrets(file))
            results["secrets"] = {
                "files_scanned": len(files),
                "findings": findings,
                "count": len(findings),
            }
            all_findings.extend(findings)

    # Configuration scan
    if "config" in scan_types:
        if path_obj.is_file():
            findings = _scan_file_for_config_issues(path_obj)
            results["config"] = {
                "files_scanned": 1,
                "findings": findings,
                "count": len(findings),
            }
            all_findings.extend(findings)
        else:
            # Use gather_files_by_pattern to exclude venv, node_modules, etc.
            files = gather_files_by_pattern(path_obj, file_pattern)
            findings = []
            for file in files:
                findings.extend(_scan_file_for_config_issues(file))
            results["config"] = {
                "files_scanned": len(files),
                "findings": findings,
                "count": len(findings),
            }
            all_findings.extend(findings)

    # Dependency scan (optional external tool)
    if "dependencies" in scan_types and dependency_scan:
        req_path = Path(requirements_file)
        if not req_path.is_absolute():
            req_path = (
                path_obj / requirements_file
                if path_obj.is_dir()
                else path_obj.parent / requirements_file
            )

        if req_path.exists():
            try:
                # Route through the shared subprocess runner (co-design
                # review item 15) instead of a bare communicate(): this
                # previously had NO timeout at all, so pip-audit hanging
                # (network lookup, huge dependency tree) hung the tool call.
                from victor.config.timeouts import ProcessTimeouts
                from victor.tools.subprocess_executor import run_managed_process

                stdout, stderr, returncode, timed_out, _capped = await run_managed_process(
                    argv=["pip-audit", "-r", str(req_path), "-f", "json"],
                    timeout=ProcessTimeouts.BASH_LONG,
                )
                if timed_out:
                    results["dependencies"] = {
                        "error": f"pip-audit timed out after {ProcessTimeouts.BASH_LONG}s",
                        "count": 0,
                    }
                elif returncode != 0:
                    results["dependencies"] = {
                        "error": f"pip-audit failed: {stderr.decode('utf-8', 'replace')}",
                        "count": 0,
                    }
                else:
                    audit = json_loads(stdout.decode("utf-8"))
                    vulns = []
                    for item in audit.get("dependencies", []):
                        for vuln in item.get("vulns", []):
                            vulns.append(
                                {
                                    "type": "dependency_vulnerability",
                                    "package": item.get("name"),
                                    "version": item.get("version"),
                                    "id": vuln.get("id"),
                                    "fix_versions": vuln.get("fix_versions"),
                                    "severity": vuln.get("severity", "medium"),
                                    "message": vuln.get("description", "Vulnerability detected"),
                                }
                            )
                    results["dependencies"] = {
                        "packages_checked": len(audit.get("dependencies", [])),
                        "vulnerabilities": vulns,
                        "count": len(vulns),
                    }
                    all_findings.extend(vulns)
            except FileNotFoundError:
                results["dependencies"] = {
                    "error": "pip-audit not installed. Install with: pip install pip-audit",
                    "count": 0,
                }
        else:
            results["dependencies"] = {
                "error": f"Requirements file not found: {req_path}",
                "count": 0,
            }

    # IaC scan (optional, semgrep or bandit)
    if "config" in scan_types and iac_scan:
        try:
            # Route through the shared subprocess runner (co-design review
            # item 15) instead of a bare communicate(): this previously had
            # NO timeout at all, so bandit hanging on a huge tree hung the
            # tool call.
            from victor.config.timeouts import ProcessTimeouts
            from victor.tools.subprocess_executor import run_managed_process

            stdout, stderr, returncode, timed_out, _capped = await run_managed_process(
                argv=["bandit", "-r", str(path_obj), "-f", "json"],
                timeout=ProcessTimeouts.BASH_LONG,
            )
            if timed_out:
                results.setdefault("config", {})
                results["config"]["error"] = f"bandit timed out after {ProcessTimeouts.BASH_LONG}s"
            elif returncode != 0:
                results.setdefault("config", {})
                results["config"]["error"] = f"bandit failed: {stderr.decode('utf-8', 'replace')}"
            else:
                bandit = json_loads(stdout.decode("utf-8"))
                findings = []
                for res in bandit.get("results", []):
                    findings.append(
                        {
                            "type": "iac_issue",
                            "file": res.get("filename"),
                            "line": res.get("line_number"),
                            "severity": res.get("issue_severity", "medium").lower(),
                            "message": res.get("issue_text"),
                            "test_id": res.get("test_id"),
                        }
                    )
                results.setdefault("config", {"findings": [], "count": 0})
                results["config"]["findings"].extend(findings)
                results["config"]["count"] = len(results["config"]["findings"])
                all_findings.extend(findings)
        except FileNotFoundError:
            results.setdefault("config", {})
            results["config"]["error"] = "bandit not installed. Install with: pip install bandit"

    # Filter by severity threshold
    severity_levels = {"low": 0, "medium": 1, "high": 2}
    threshold_level = severity_levels.get(severity_threshold, 0)

    filtered_findings = [
        f
        for f in all_findings
        if severity_levels.get(f.get("severity", "low"), 0) >= threshold_level
    ]

    # Count issues by severity
    issues_by_severity = {"high": 0, "medium": 0, "low": 0}
    for finding in filtered_findings:
        severity = finding.get("severity", "low")
        issues_by_severity[severity] = issues_by_severity.get(severity, 0) + 1

    # Build comprehensive report
    report = []
    report.append("Security Scan Report")
    report.append("=" * 70)
    report.append("")
    report.append(f"Scan types: {', '.join(scan_types)}")
    report.append(f"Severity threshold: {severity_threshold}")
    report.append("")
    report.append(f"Total issues: {len(filtered_findings)}")
    report.append(f"  High: {issues_by_severity['high']}")
    report.append(f"  Medium: {issues_by_severity['medium']}")
    report.append(f"  Low: {issues_by_severity['low']}")
    report.append("")

    # Secrets section
    if "secrets" in results:
        report.append("Secrets Detection:")
        report.append(f"  Files scanned: {results['secrets']['files_scanned']}")
        secrets_filtered = [
            f
            for f in results["secrets"]["findings"]
            if severity_levels.get(f.get("severity", "low"), 0) >= threshold_level
        ]
        report.append(f"  Secrets found: {len(secrets_filtered)}")
        if secrets_filtered:
            report.append("  Critical: Hardcoded secrets detected!")
            for finding in secrets_filtered[:5]:
                report.append(f"    {finding['file']} (line {finding['line']}): {finding['type']}")
            if len(secrets_filtered) > 5:
                report.append(f"    ... and {len(secrets_filtered) - 5} more")
        report.append("")

    # Config section
    if "config" in results:
        report.append("Configuration Issues:")
        report.append(f"  Files scanned: {results['config']['files_scanned']}")
        config_filtered = [
            f
            for f in results["config"]["findings"]
            if severity_levels.get(f.get("severity", "low"), 0) >= threshold_level
        ]
        report.append(f"  Issues found: {len(config_filtered)}")
        if config_filtered:
            for finding in config_filtered[:5]:
                report.append(f"    {finding['file']} (line {finding['line']}): {finding['type']}")
            if len(config_filtered) > 5:
                report.append(f"    ... and {len(config_filtered) - 5} more")
        report.append("")

    # Dependencies section
    if "dependencies" in results:
        report.append("Dependency Vulnerabilities:")
        if "error" in results["dependencies"]:
            report.append(f"  {results['dependencies']['error']}")
        else:
            report.append(f"  Packages checked: {results['dependencies']['packages_checked']}")
            report.append(f"  Vulnerabilities: {len(results['dependencies']['vulnerabilities'])}")
            if results["dependencies"]["vulnerabilities"]:
                for vuln in results["dependencies"]["vulnerabilities"][:5]:
                    report.append(f"    {vuln['package']}: {vuln['message']}")
                if len(results["dependencies"]["vulnerabilities"]) > 5:
                    report.append(
                        f"    ... and {len(results['dependencies']['vulnerabilities']) - 5} more"
                    )
        report.append("")

    # Summary
    if len(filtered_findings) > 0:
        report.append("Security issues require attention!")
    else:
        report.append("No critical security issues detected")

    return {
        "success": True,
        "scan_types_performed": scan_types,
        "results": results,
        "total_issues": len(filtered_findings),
        "issues_by_severity": issues_by_severity,
        "findings": filtered_findings,
        "formatted_report": "\n".join(report),
    }

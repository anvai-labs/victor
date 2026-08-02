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

"""Unified dependency management tool for Python projects.

NOTE: For standard pip operations, PREFER the ``shell`` tool:
    shell(cmd="pip list")
    shell(cmd="pip show requests")
    shell(cmd="pip install black")
    shell(cmd="pip install --upgrade requests")
    shell(cmd="pipdeptree")

This tool wraps ``pip`` and ``pipdeptree`` CLI commands via subprocess.
Use it only for the ``security`` action (vulnerability scanning) or
``generate`` action (requirements.txt generation). For list/show/install/tree
operations, ``shell(cmd="pip ...")`` is more efficient.

Consolidates all dependency operations into a single tool for better token efficiency.
Supports: list, outdated, security, generate, update, tree, check.
"""

from victor.core.json_utils import json_loads
from json import JSONDecodeError
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

from victor.tools.base import AccessMode, DangerLevel, Priority
from victor.tools.decorators import tool
from victor.tools.subprocess_executor import run_pip_async, run_command_async

logger = logging.getLogger(__name__)

# Known vulnerability databases
VULN_PACKAGES = {
    "django": {"<2.2.28": "CVE-2022-28346", "<3.2.13": "CVE-2022-28347"},
    "pillow": {"<9.0.0": "CVE-2022-22817"},
    "requests": {"<2.26.0": "CVE-2021-33503"},
}


def _parse_version(version: str) -> tuple:
    """Parse version string into comparable tuple."""
    try:
        parts = re.findall(r"\d+", version)
        return tuple(int(p) for p in parts)
    except (ValueError, TypeError, AttributeError):
        return (0, 0, 0)


def _version_satisfies(current: str, constraint: str) -> bool:
    """Check if current version satisfies constraint."""
    if constraint.startswith("<"):
        return _parse_version(current) < _parse_version(constraint[1:])
    return False


async def _do_list() -> Dict[str, Any]:
    """List all installed packages."""
    success, stdout, stderr = await run_pip_async("list", "--format=json")
    if not success:
        return {"success": False, "error": f"Failed to list packages: {stderr}"}

    try:
        packages = json_loads(stdout)
    except JSONDecodeError as e:
        return {"success": False, "error": f"Failed to parse package list: {e}"}

    report = []
    report.append("Installed Packages")
    report.append("=" * 70)
    report.append(f"\nTotal: {len(packages)} packages\n")

    by_letter: Dict[str, List[Dict]] = {}
    for pkg in packages:
        letter = pkg["name"][0].upper()
        if letter not in by_letter:
            by_letter[letter] = []
        by_letter[letter].append(pkg)

    for letter in sorted(by_letter.keys()):
        pkgs = by_letter[letter]
        report.append(f"{letter}:")
        for pkg in sorted(pkgs, key=lambda x: x["name"])[:5]:
            report.append(f"  {pkg['name']} ({pkg['version']})")
        if len(pkgs) > 5:
            report.append(f"  ... and {len(pkgs) - 5} more")
        report.append("")

    return {
        "success": True,
        "packages": packages,
        "count": len(packages),
        "formatted_report": "\n".join(report),
    }


async def _do_outdated() -> Dict[str, Any]:
    """Check for outdated packages."""
    success, stdout, stderr = await run_pip_async("list", "--outdated", "--format=json")
    if not success:
        return {
            "success": False,
            "error": f"Failed to check outdated packages: {stderr}",
        }

    try:
        outdated = json_loads(stdout)
    except JSONDecodeError as e:
        return {"success": False, "error": f"Failed to parse outdated list: {e}"}

    if not outdated:
        return {
            "success": True,
            "outdated": [],
            "count": 0,
            "message": "All packages are up to date!",
        }

    major_updates = []
    minor_updates = []
    patch_updates = []

    for pkg in outdated:
        current = _parse_version(pkg["version"])
        latest = _parse_version(pkg["latest_version"])
        if latest[0] > current[0]:
            major_updates.append(pkg)
        elif latest[1] > current[1]:
            minor_updates.append(pkg)
        else:
            patch_updates.append(pkg)

    report = []
    report.append("Outdated Packages")
    report.append("=" * 70)
    report.append(f"\nTotal: {len(outdated)} packages need updating\n")

    if major_updates:
        report.append(f"Major updates ({len(major_updates)}):")
        for pkg in major_updates[:5]:
            report.append(f"  {pkg['name']}: {pkg['version']} -> {pkg['latest_version']}")
        if len(major_updates) > 5:
            report.append(f"  ... and {len(major_updates) - 5} more")
        report.append("")

    if minor_updates:
        report.append(f"Minor updates ({len(minor_updates)}):")
        for pkg in minor_updates[:5]:
            report.append(f"  {pkg['name']}: {pkg['version']} -> {pkg['latest_version']}")
        if len(minor_updates) > 5:
            report.append(f"  ... and {len(minor_updates) - 5} more")
        report.append("")

    if patch_updates:
        report.append(f"Patch updates ({len(patch_updates)}):")
        for pkg in patch_updates[:5]:
            report.append(f"  {pkg['name']}: {pkg['version']} -> {pkg['latest_version']}")
        if len(patch_updates) > 5:
            report.append(f"  ... and {len(patch_updates) - 5} more")

    return {
        "success": True,
        "outdated": outdated,
        "count": len(outdated),
        "by_severity": {
            "major": major_updates,
            "minor": minor_updates,
            "patch": patch_updates,
        },
        "formatted_report": "\n".join(report),
    }


async def _do_security() -> Dict[str, Any]:
    """Check for security vulnerabilities."""
    success, stdout, stderr = await run_pip_async("list", "--format=json")
    if not success:
        return {"success": False, "error": f"Security audit failed: {stderr}"}

    try:
        packages = json_loads(stdout)
    except JSONDecodeError as e:
        return {"success": False, "error": f"Failed to parse package list: {e}"}

    vulnerabilities = []

    for pkg in packages:
        pkg_name = pkg["name"].lower()
        pkg_version = pkg["version"]
        if pkg_name in VULN_PACKAGES:
            for constraint, cve in VULN_PACKAGES[pkg_name].items():
                if _version_satisfies(pkg_version, constraint):
                    vulnerabilities.append(
                        {
                            "package": pkg_name,
                            "version": pkg_version,
                            "cve": cve,
                            "constraint": constraint,
                        }
                    )

    report = []
    report.append("Security Audit")
    report.append("=" * 70)
    report.append("")

    if not vulnerabilities:
        report.append("No known vulnerabilities found!")
        report.append("\nNote: This is a basic check. For comprehensive auditing, use: pip-audit")
    else:
        report.append(f"Found {len(vulnerabilities)} potential vulnerabilities:\n")
        for vuln in vulnerabilities:
            report.append(f"  {vuln['package']} {vuln['version']}")
            report.append(f"    CVE: {vuln['cve']}")
            report.append(f"    Affected: {vuln['constraint']}\n")

    return {
        "success": True,
        "vulnerabilities": vulnerabilities,
        "count": len(vulnerabilities),
        "formatted_report": "\n".join(report),
    }


async def _do_generate(output: str) -> Dict[str, Any]:
    """Generate requirements file."""
    success, stdout, stderr = await run_pip_async("freeze")
    if not success:
        return {"success": False, "error": f"Failed to generate requirements: {stderr}"}

    requirements = stdout.strip()
    package_count = len(
        [line for line in requirements.split("\n") if line and not line.startswith("#")]
    )

    try:
        output_path = Path(output)
        output_path.write_text(requirements)

        return {
            "success": True,
            "file": str(output_path),
            "packages_count": package_count,
            "message": f"Generated {output} with {package_count} packages",
        }
    except IOError as e:
        return {"success": False, "error": f"Failed to write file: {str(e)}"}


async def _do_update(packages: List[str], dry_run: bool) -> Dict[str, Any]:
    """Update packages."""
    if not packages:
        return {"success": False, "error": "No packages specified for update"}

    if dry_run:
        return {
            "success": True,
            "would_update": packages,
            "message": f"Dry run: Would update {len(packages)} packages: {', '.join(packages)}",
        }

    updated = []
    for package in packages:
        success, stdout, stderr = await run_pip_async("install", "--upgrade", package)
        if not success:
            return {
                "success": False,
                "error": f"Failed to update packages: {stderr}",
                "partially_updated": updated,
            }
        updated.append(package)

    return {
        "success": True,
        "updated": updated,
        "message": f"Successfully updated {len(updated)} packages",
    }


async def _do_tree(package: Optional[str]) -> Dict[str, Any]:
    """Show dependency tree."""
    # Check if pipdeptree is installed
    success, stdout, stderr = await run_pip_async("show", "pipdeptree")
    if not success:
        return {
            "success": False,
            "error": "pipdeptree not installed. Install with: pip install pipdeptree",
        }

    cmd = "pipdeptree"
    if package:
        cmd += f" --packages {package}"

    result = await run_command_async(cmd, timeout=60, check_dangerous=False)
    if not result.success:
        return {
            "success": False,
            "error": f"Failed to show dependency tree: {result.stderr}",
        }

    return {"success": True, "tree": result.stdout, "package": package}


async def _do_check(requirements_file: str) -> Dict[str, Any]:
    """Check requirements satisfaction."""
    req_path = Path(requirements_file)
    if not req_path.exists():
        return {
            "success": False,
            "error": f"Requirements file not found: {requirements_file}",
        }

    success, stdout, stderr = await run_pip_async("list", "--format=json")
    if not success:
        return {"success": False, "error": f"Failed to check requirements: {stderr}"}

    try:
        installed = {pkg["name"].lower(): pkg["version"] for pkg in json_loads(stdout)}
    except JSONDecodeError as e:
        return {"success": False, "error": f"Failed to parse package list: {e}"}

    try:
        requirements = req_path.read_text().strip().split("\n")
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to read requirements file: {str(e)}",
        }

    missing = []
    mismatched = []
    satisfied = []

    for req in requirements:
        req = req.strip()
        if not req or req.startswith("#"):
            continue
        match = re.match(r"([a-zA-Z0-9_-]+)(==|>=|<=)?([\d.]+)?", req)
        if not match:
            continue
        pkg_name = match.group(1).lower()
        operator = match.group(2)
        version = match.group(3)

        if pkg_name not in installed:
            missing.append(req)
        elif operator == "==" and version and installed[pkg_name] != version:
            mismatched.append(
                {
                    "package": pkg_name,
                    "required": version,
                    "installed": installed[pkg_name],
                }
            )
        else:
            satisfied.append(pkg_name)

    report = []
    report.append(f"Requirements Check: {requirements_file}")
    report.append("=" * 70)
    report.append("")

    if not missing and not mismatched:
        report.append("All requirements satisfied!")
    else:
        if missing:
            report.append(f"Missing packages ({len(missing)}):")
            for pkg in missing:
                report.append(f"  {pkg}")
            report.append("")
        if mismatched:
            report.append(f"Version mismatches ({len(mismatched)}):")
            for mm in mismatched:
                report.append(
                    f"  {mm['package']}: required {mm['required']}, installed {mm['installed']}"
                )
            report.append("")
        report.append(f"Satisfied: {len(satisfied)} packages")

    return {
        "success": len(missing) == 0 and len(mismatched) == 0,
        "missing": missing,
        "mismatched": mismatched,
        "satisfied_count": len(satisfied),
        "formatted_report": "\n".join(report),
    }


@tool(
    category="deps",
    priority=Priority.MEDIUM,  # Task-specific dependency management
    access_mode=AccessMode.MIXED,  # Can read and update packages
    danger_level=DangerLevel.MEDIUM,  # Package updates can affect system
    keywords=["dependency", "package", "requirements", "version", "npm", "pip"],
)
async def dependency(
    action: str,
    packages: Optional[List[str]] = None,
    package: Optional[str] = None,
    output: str = "requirements.txt",
    requirements_file: str = "requirements.txt",
    dry_run: bool = True,
) -> Dict[str, Any]:
    """Python dependency management: list, outdated, security, generate, update, tree, check.

    Actions: list (packages), outdated, security (vulns), generate (requirements.txt),
    update (packages, dry_run), tree (package), check (requirements_file).
    """
    action_lower = action.lower().strip()

    if action_lower == "list":
        return await _do_list()

    elif action_lower == "outdated":
        return await _do_outdated()

    elif action_lower == "security":
        return await _do_security()

    elif action_lower == "generate":
        return await _do_generate(output)

    elif action_lower == "update":
        return await _do_update(packages or [], dry_run)

    elif action_lower == "tree":
        return await _do_tree(package)

    elif action_lower == "check":
        return await _do_check(requirements_file)

    else:
        return {
            "success": False,
            "error": f"Unknown action: {action}. Valid actions: list, outdated, security, generate, update, tree, check",
        }

"""Fail closed on incomplete scans and enforce scoped, expiring exceptions.

pip-audit does not supply severity: every unexcepted Python advisory blocks.
Trivy reports all severities; the caller selects the blocking severity floor.
"""

from __future__ import annotations

import argparse
from datetime import date
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, cast

SEVERITIES = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def text(value: Any, label: str) -> str:
    require(isinstance(value, str) and bool(value.strip()), f"Missing/invalid {label}")
    return cast(str, value)


def sequence(value: Any, label: str) -> list[Any]:
    require(isinstance(value, list), f"Missing/invalid {label}")
    return cast(list[Any], value)


def package_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def expected_targets(root: Path) -> set[str]:
    """Discover supported lockfiles; a parse failure must not erase scan coverage.

    Other Python requirements/pyproject manifests need resolved-environment audits;
    the root requirements.txt is the repository's pinned inventory.
    """
    targets = set()
    for directory, subdirs, files in os.walk(root):
        subdirs[:] = [
            name
            for name in subdirs
            if name not in {".git", ".worktrees", ".venv", "venv", "node_modules", "target"}
        ]
        for name in files:
            relative = (Path(directory) / name).relative_to(root).as_posix()
            if name in {"Cargo.lock", "package-lock.json"} or relative == "requirements.txt":
                targets.add(relative)
    return targets


def read_exceptions(path: Path, today: date) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    require(isinstance(data, dict) and data.get("version") == 1, "Unknown exception schema")
    entries = sequence(data.get("exceptions"), "exceptions")
    for entry in entries:
        require(isinstance(entry, dict), "Invalid exception")
        for field in ("package", "owner", "reason", "expires"):
            text(entry.get(field), f"exception {field}")
        require(date.fromisoformat(entry["expires"]) >= today, "Expired security exception")
        for field in ("advisories", "versions", "scopes"):
            values = sequence(entry.get(field), f"exception {field}")
            require(bool(values), f"Empty exception {field}")
            for value in values:
                text(value, f"exception {field}")
        require(
            set(entry["scopes"]) <= {"python", "filesystem", "container"},
            "Unknown exception scope",
        )
        require(
            all(re.fullmatch(r"(?:CVE|GHSA|PYSEC)-[\w-]+", a) for a in entry["advisories"]),
            "Invalid exception advisory",
        )
        require(
            all(not any(c in version for c in "*<>=!") for version in entry["versions"]),
            "Exceptions require exact package versions",
        )
    return entries


def accepted(
    entries: list[dict[str, Any]], scope: str, package: str, version: str, ids: set[str]
) -> bool:
    return any(
        scope in entry["scopes"]
        and package_name(package) == package_name(entry["package"])
        and version in entry["versions"]
        and bool(ids.intersection(entry["advisories"]))
        for entry in entries
    )


def python_findings(report: Any, status: int) -> list[dict[str, Any]]:
    require(isinstance(report, dict), "Unknown pip-audit report schema")
    dependencies = sequence(report.get("dependencies"), "pip-audit dependencies")
    require(bool(dependencies), "No dependencies audited")
    require(status in (0, 1), f"pip-audit failed with status {status}")
    findings = []
    for dependency in dependencies:
        require(isinstance(dependency, dict), "Invalid dependency")
        require("skip_reason" not in dependency, f"Unaudited dependency: {dependency.get('name')}")
        name = text(dependency.get("name"), "dependency name")
        version = text(dependency.get("version"), "dependency version")
        for vuln in sequence(dependency.get("vulns"), "dependency vulns"):
            require(isinstance(vuln, dict), "Invalid vulnerability")
            advisory = text(vuln.get("id"), "advisory id")
            aliases = sequence(vuln.get("aliases", []), "advisory aliases")
            ids = {advisory, *(text(alias, "alias") for alias in aliases)}
            findings.append({"package": name, "version": version, "ids": ids})
    require((status == 1) == bool(findings), "pip-audit status/report mismatch")
    return findings


def trivy_findings(report: Any, status: int, scope: str) -> list[dict[str, Any]]:
    require(status == 0, f"Trivy failed with status {status}")
    require(isinstance(report, dict) and report.get("SchemaVersion") == 2, "Unknown Trivy schema")
    expected = "container_image" if scope == "container" else "filesystem"
    require(report.get("ArtifactType") == expected, "Unexpected Trivy artifact type")
    results = sequence(report.get("Results"), "Trivy Results")
    require(bool(results), "No package targets scanned")
    if scope == "container":
        require(
            {r.get("Class") for r in results if isinstance(r, dict)} >= {"os-pkgs", "lang-pkgs"},
            "Container scan must cover OS and application packages",
        )
    findings = []
    for result in results:
        require(isinstance(result, dict), "Invalid Trivy target")
        require(result.get("Class") in ("os-pkgs", "lang-pkgs"), "Unexpected Trivy target class")
        text(result.get("Target"), "Trivy target")
        text(result.get("Type"), "Trivy package type")
        packages = sequence(result.get("Packages"), "scanned packages")
        require(bool(packages), "No packages scanned")
        inventory = set()
        synthetic_roots: list[str] = []
        for package in packages:
            require(isinstance(package, dict), "Invalid scanned package")
            if (
                result["Type"] == "cargo"
                and package.get("AnalyzedBy") == "cargo"
                and package.get("Relationship") == "root"
                and "Name" not in package
                and "Version" not in package
            ):
                # Trivy represents a Cargo workspace by a synthetic root. It is
                # not a dependency; every referenced member must still be present.
                text(package.get("ID"), "workspace root id")
                members = sequence(package.get("DependsOn"), "workspace members")
                require(bool(members), "Empty workspace inventory")
                synthetic_roots.extend(text(member, "workspace member") for member in members)
                continue
            inventory.add(
                (
                    text(package.get("Name"), "package name"),
                    text(package.get("Version"), "package version"),
                )
            )
        require(bool(inventory), "No named packages scanned")
        ids = {
            package.get("ID") for package in packages if package.get("Name") and package.get("ID")
        }
        require(set(synthetic_roots) <= ids, "Missing Cargo workspace members")
        for vuln in sequence(result.get("Vulnerabilities", []), "Trivy vulnerabilities"):
            require(isinstance(vuln, dict), "Invalid Trivy vulnerability")
            severity = text(vuln.get("Severity"), "vulnerability severity")
            require(severity in {*SEVERITIES, "UNKNOWN"}, "Unrecognized vulnerability severity")
            require(
                (vuln.get("PkgName"), vuln.get("InstalledVersion")) in inventory,
                "Vulnerability missing from package inventory",
            )
            findings.append(
                {
                    "package": text(vuln.get("PkgName"), "package name"),
                    "version": text(vuln.get("InstalledVersion"), "installed version"),
                    "ids": {text(vuln.get("VulnerabilityID"), "advisory id")},
                    "severity": severity,
                }
            )
    return findings


def check(
    report: Any,
    entries: list[dict[str, Any]],
    scope: str,
    status: int,
    minimum: str = "HIGH",
    targets: set[str] | None = None,
) -> list[str]:
    require(minimum in SEVERITIES, "Invalid severity floor")
    require(scope in {"python", "filesystem", "container"}, "Unknown scan scope")
    findings = (
        python_findings(report, status)
        if scope == "python"
        else trivy_findings(report, status, scope)
    )
    if scope == "filesystem" and targets is not None:
        observed = {result["Target"].removeprefix("./") for result in report["Results"]}
        require(not targets - observed, f"Unaudited manifests: {sorted(targets - observed)}")
    if scope == "container":
        require(
            any(result.get("Type") == "python-pkg" for result in report["Results"]),
            "Victor container scan is missing Python packages",
        )
    blocked = []
    for finding in findings:
        severity = finding.get("severity")
        if severity in SEVERITIES and SEVERITIES[severity] < SEVERITIES[minimum]:
            continue
        if not accepted(entries, scope, finding["package"], finding["version"], finding["ids"]):
            blocked.append(
                f"{finding['package']}=={finding['version']}: "
                f"{', '.join(sorted(finding['ids']))} ({severity or 'pip-audit advisory'})"
            )
    print(f"Validated {len(findings)} findings; {len(blocked)} unexcepted blockers")
    return blocked


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--scope", choices=("python", "filesystem", "container"), required=True)
    parser.add_argument("--scanner-status", type=int, required=True)
    parser.add_argument("--minimum", choices=tuple(SEVERITIES), default="HIGH")
    parser.add_argument("--exceptions", type=Path, default=Path(".github/security/exceptions.json"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    try:
        entries = read_exceptions(args.exceptions, date.today())
        blocked = check(
            json.loads(args.report.read_text()),
            entries,
            args.scope,
            args.scanner_status,
            args.minimum,
            expected_targets(args.root) if args.scope == "filesystem" else None,
        )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"Security scan incomplete: {exc}", file=sys.stderr)
        return 2
    for finding in blocked:
        print(finding, file=sys.stderr)
    return int(bool(blocked))


if __name__ == "__main__":
    raise SystemExit(main())

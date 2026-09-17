"""Validate Semgrep execution evidence while keeping findings advisory.

Only PartialParsing warnings are nonfatal: they are reported as coverage gaps,
not silently counted as a complete parse. Raw JSON and SARIF remain unchanged.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from typing import Any, cast
from urllib.parse import unquote


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"Invalid/missing {label}")
    return cast(dict[str, Any], value)


def sequence(value: Any, label: str) -> list[Any]:
    require(isinstance(value, list), f"Invalid/missing {label}")
    return cast(list[Any], value)


def text(value: Any, label: str) -> str:
    require(isinstance(value, str) and bool(value.strip()), f"Invalid/missing {label}")
    return cast(str, value)


def line(value: Any) -> int:
    require(type(value) is int and value > 0, "Invalid/missing finding line")
    return cast(int, value)


def in_source_suppressed(result: dict[str, Any]) -> bool:
    suppressions = sequence(result.get("suppressions", []), "SARIF suppressions")
    for item in suppressions:
        suppression = mapping(item, "SARIF suppression")
        if suppression.get("kind") == "inSource" and suppression.get("status") in (
            None,
            "accepted",
        ):
            return True
    return False


def validate(report: Any, sarif: Any, status: int) -> tuple[dict[str, Any], list[str], str]:
    require(status == 0, f"Semgrep operational failure (exit {status})")
    report = mapping(report, "Semgrep JSON report")
    version = text(report.get("version"), "Semgrep version")
    paths = mapping(report.get("paths"), "scan paths")
    scanned = sequence(paths.get("scanned"), "scanned paths")
    require(bool(scanned), "Semgrep scanned no files")
    for path in scanned:
        text(path, "scanned path")
    require(not sequence(report.get("skipped_rules"), "skipped rules"), "Semgrep skipped rules")

    warnings = []
    diagnostic_messages: Counter[str] = Counter()
    for item in sequence(report.get("errors"), "Semgrep errors"):
        diagnostic = mapping(item, "Semgrep diagnostic")
        kind = diagnostic.get("type")
        if isinstance(kind, list) and kind:
            kind = kind[0]
        kind = text(kind, "diagnostic type")
        message = text(diagnostic.get("message"), "diagnostic message")
        require(
            diagnostic.get("level") == "warn" and kind == "PartialParsing",
            f"Semgrep scan incomplete: {kind}: {message}",
        )
        warnings.append(f"PartialParsing coverage gap: {message}")
        diagnostic_messages[message] += 1

    expected: Counter[tuple[str, str, int, int]] = Counter()
    for item in sequence(report.get("results"), "JSON results"):
        result = mapping(item, "JSON finding")
        extra = mapping(result.get("extra"), "JSON finding details")
        text(extra.get("message"), "finding message")
        require(type(extra.get("is_ignored", False)) is bool, "Invalid is_ignored flag")
        key = (
            text(result.get("check_id"), "finding rule"),
            text(result.get("path"), "finding path").removeprefix("./"),
            line(mapping(result.get("start"), "finding start").get("line")),
            line(mapping(result.get("end"), "finding end").get("line")),
        )
        if not extra.get("is_ignored", False):
            expected[key] += 1

    sarif = mapping(sarif, "SARIF report")
    require(sarif.get("version") == "2.1.0", "Unsupported SARIF version")
    runs = sequence(sarif.get("runs"), "SARIF runs")
    require(len(runs) == 1, "Expected one Semgrep SARIF run")
    run = mapping(runs[0], "SARIF run")
    driver = mapping(mapping(run.get("tool"), "SARIF tool").get("driver"), "SARIF driver")
    require(text(driver.get("name"), "scanner name").startswith("Semgrep"), "Not Semgrep SARIF")
    require(driver.get("semanticVersion") == version, "JSON/SARIF scanner version mismatch")
    invocations = sequence(run.get("invocations"), "SARIF invocations")
    require(bool(invocations), "Missing Semgrep invocation evidence")
    sarif_diagnostics: Counter[str] = Counter()
    for item in invocations:
        invocation = mapping(item, "SARIF invocation")
        require(invocation.get("executionSuccessful") is True, "Unsuccessful SARIF invocation")
        for item in sequence(
            invocation.get("toolExecutionNotifications"), "SARIF execution notifications"
        ):
            notification = mapping(item, "SARIF execution notification")
            message = text(
                mapping(notification.get("message"), "notification message").get("text"),
                "notification text",
            )
            level = notification.get("level", "warning")
            require(level in {"warning", "note"}, f"SARIF operational diagnostic: {message}")
            if level == "warning":
                sarif_diagnostics[message] += 1
    require(
        sarif_diagnostics == diagnostic_messages,
        "JSON/SARIF diagnostic inventory mismatch",
    )

    observed: Counter[tuple[str, str, int, int]] = Counter()
    kept = []
    raw_results = sequence(run.get("results"), "SARIF results")
    for item in raw_results:
        result = mapping(item, "SARIF finding")
        rule = text(result.get("ruleId"), "SARIF rule")
        text(mapping(result.get("message"), "SARIF message").get("text"), "SARIF message text")
        locations = sequence(result.get("locations"), "SARIF locations")
        require(bool(locations), "Missing SARIF finding location")
        location = mapping(
            mapping(locations[0], "SARIF location").get("physicalLocation"), "physical location"
        )
        artifact = mapping(location.get("artifactLocation"), "artifact location")
        region = mapping(location.get("region"), "finding region")
        key = (
            rule,
            unquote(text(artifact.get("uri"), "artifact URI")).removeprefix("./"),
            line(region.get("startLine")),
            line(region.get("endLine")),
        )
        if not in_source_suppressed(result):
            kept.append(result)
            observed[key] += 1
    require(observed == expected, "JSON/SARIF finding inventory mismatch")
    filtered = deepcopy(sarif)
    filtered["runs"][0]["results"] = deepcopy(kept)
    summary = (
        f"Semgrep {version}: {len(scanned)} scanned files, {len(kept)} advisory findings, "
        f"{len(raw_results) - len(kept)} in-source suppressions, "
        f"{len(warnings)} partial-parsing coverage warnings. Raw reports retained."
    )
    return filtered, warnings, summary


def annotation(message: str) -> str:
    return message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("sarif", type=Path)
    parser.add_argument("--status", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        require(
            args.output.resolve() not in {args.report.resolve(), args.sarif.resolve()},
            "Output must not overwrite raw evidence",
        )
        filtered, warnings, summary = validate(
            json.loads(args.report.read_text()), json.loads(args.sarif.read_text()), args.status
        )
        args.output.write_text(json.dumps(filtered) + "\n")
        print(summary)
        for warning in warnings:
            print(f"::warning title=Semgrep coverage::{annotation(warning)}")
        if summary_path := os.environ.get("GITHUB_STEP_SUMMARY"):
            with Path(summary_path).open("a") as stream:
                stream.write(summary + "\n")
        return 0
    except (OSError, ValueError, TypeError) as error:
        print(f"::error title=Semgrep scan evidence::{annotation(str(error))}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

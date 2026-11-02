"""Optional pinned jscpd token-fragment evidence, isolated from project config.

The local executable is explicit/trusted configuration. Only bounded snapshots
of selected source files enter its scratch directory; it never runs npx,
downloads a package, executes source, or writes into the analyzed project.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Any

PINNED_JSCPD_VERSION = "5.4.2"
MAX_REPORT_BYTES = 64 * 1024 * 1024
MAX_TOKEN_PAIRS = 100_000
MAX_SOURCE_BYTES = 128 * 1024 * 1024
MAX_BACKEND_RSS = 768 * 1024 * 1024
MAX_SOURCE_FILES = 10_000


def _run_detector(arguments: list[str], work: Path, timeout: float) -> tuple[int, str]:
    """Use shared owned-process limits without replaying a detector call."""
    from victor.core.graph_rag.bounded_process import OwnedProcess

    try:
        return OwnedProcess(
            arguments,
            work,
            timeout,
            max_rss=MAX_BACKEND_RSS,
            artifacts={
                work / "report" / "jscpd-report.json": MAX_REPORT_BYTES,
                work / "report" / "jscpd-report.sarif": MAX_REPORT_BYTES,
                work / "report" / "jscpd-complexity.json": MAX_REPORT_BYTES,
                work / "backend.log": MAX_REPORT_BYTES,
            },
        ).run_sync()
    except ValueError as exc:
        raise ValueError(str(exc).replace("External tool", "Token backend")) from exc


@dataclass(frozen=True)
class TokenOptions:
    binary: str = "jscpd"
    min_tokens: int = 300
    min_lines: int = 3
    normalize_identifiers: bool = False
    timeout_seconds: float = 60
    fingerprints: bool = False

    def __post_init__(self) -> None:
        if not 1 <= self.min_tokens <= 1_000_000 or not 0 <= self.min_lines <= 100_000:
            raise ValueError("Invalid token fragment size")
        if not 0 < self.timeout_seconds <= 300:
            raise ValueError("Invalid token backend timeout")
        if type(self.fingerprints) is not bool:
            raise ValueError("Token fingerprint selection must be boolean")


def _account_for_omissions(
    missing: set[str],
    manifest: dict[str, dict[str, Any]],
    sources: Path,
    work: Path,
    config: Path,
    binary: str,
    options: TokenOptions,
    started: float,
    max_file_bytes: int,
    errors: list[dict[str, str]],
) -> int:
    """Account for filtered inputs without treating eligible unit totals as files."""
    from victor.core.graph_rag.token_coverage import physical_file_summary, proven_below_threshold

    if not missing:
        return 0
    control = work / "coverage"
    control_sources, output = control / "source", control / "report"
    control_sources.mkdir(parents=True)
    for name in sorted(missing):
        target = control_sources / name
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(sources / name, target)  # Link only owned snapshot bytes, never user source files.
    remaining = options.timeout_seconds - (time.monotonic() - started)
    if remaining <= 0:
        raise ValueError("Token backend exceeded its total time budget")
    arguments = [
        binary,
        "--config",
        str(config),
        "--mode",
        "weak",
        "--min-tokens",
        "0",
        "--min-lines",
        "0",
        "--max-size",
        str(max_file_bytes),
        "--complexity",
        "--summary-top",
        str(MAX_SOURCE_FILES),
        "--reporters",
        "json",
        "--output",
        str(output),
        "--no-gitignore",
        "--silent",
        str(control_sources),
    ]
    if options.normalize_identifiers:
        arguments.insert(-1, "--ignore-identifiers")
    code, _ = _run_detector(arguments, control, remaining)
    report = output / "jscpd-complexity.json"
    if code != 0 or not report.is_file() or report.stat().st_size > MAX_REPORT_BYTES:
        raise ValueError("Token backend did not produce bounded physical-file accounting")
    data = report.read_bytes()
    if len(data) > MAX_REPORT_BYTES:
        raise ValueError("Token physical-file accounting exceeded its byte budget")
    accounted = physical_file_summary(
        json.loads(data), {name: manifest[name] for name in missing}, metric="complexity"
    )
    verified = 0
    for name in sorted(missing):
        if name not in accounted:
            errors.append({"file": name, "reason": "backend_unaccounted_input"})
        elif not proven_below_threshold(
            accounted[name],
            manifest[name],
            min_tokens=options.min_tokens,
            min_lines=options.min_lines,
        ):
            errors.append({"file": name, "reason": "backend_unexplained_omission"})
        else:
            verified += 1
    return verified


def scan_tokens(
    root: Path,
    *,
    options: TokenOptions = TokenOptions(),
    scope: str | None = None,
    include_tests: bool = False,
    include_docs: bool = False,
    include_same_file: bool = False,
    max_file_bytes: int = 1_000_000,
) -> dict[str, Any]:
    """Run the pinned fragment detector on an explicit read-only source snapshot."""
    from victor.core.graph_rag.structural_clones import _source_files, _in_scope
    from victor.core.graph_rag.exclude_patterns import UNIVERSAL_EXCLUDE_PATTERNS, is_path_excluded
    from victor.core.graph_rag.source_reader import project_source_reader

    if not root.is_dir() or max_file_bytes < 1:
        raise ValueError("Token analysis requires a project directory and a positive file bound")
    binary = shutil.which(options.binary)
    if binary is None:
        raise ValueError(
            f"Install jscpd {PINNED_JSCPD_VERSION} locally and pass --jscpd-binary; executable unavailable: {options.binary}"
        )
    binary = str(Path(binary).resolve())
    try:
        version = (
            subprocess.run(
                [binary, "--version"], capture_output=True, text=True, timeout=10, check=True
            )
            .stdout.strip()
            .removeprefix("jscpd ")
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("Cannot verify the local jscpd executable") from exc
    if version != PINNED_JSCPD_VERSION:
        raise ValueError(f"Unsupported jscpd version {version!r}; expected {PINNED_JSCPD_VERSION}")
    selection = {
        "scope": scope,
        "include_tests": include_tests,
        "include_docs": include_docs,
        "include_same_file": include_same_file,
    }
    with project_source_reader(root) as read_source:
        selected = [
            rel
            for rel in _source_files(root, include_docs, root_descriptor=read_source.descriptor)
            if _in_scope(rel, scope, include_tests)
            and not is_path_excluded(root / rel, root, UNIVERSAL_EXCLUDE_PATTERNS)
        ]
        read_source.verify_root_path()
        manifest: dict[str, dict[str, Any]] = {}
        errors: list[dict[str, str]] = []
        total_bytes = 0
        with tempfile.TemporaryDirectory(prefix="victor-token-clones-") as directory:
            work = Path(directory)
            sources, output = work / "source", work / "report"
            sources.mkdir()
            for rel in selected:
                try:
                    data, source_stat = read_source(rel, max_file_bytes)
                    if source_stat.st_size > max_file_bytes or len(data) > max_file_bytes:
                        errors.append({"file": rel, "reason": "oversized"})
                        continue
                    data.decode("utf-8")
                    total_bytes += len(data)
                    if total_bytes > MAX_SOURCE_BYTES or len(manifest) >= MAX_SOURCE_FILES:
                        raise ValueError(
                            "Token source snapshot exceeds its total budget; narrow --scope"
                        )
                    destination = sources / rel
                    # Git inventories and filesystem inventories must remain local.
                    if not destination.resolve().is_relative_to(sources.resolve()):
                        raise ValueError("Invalid token source path")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(data)
                    manifest[rel] = {
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "lines": len(data.splitlines()),
                        "bytes": len(data),
                        "native_lines": data.count(b"\n")
                        + int(bool(data) and not data.endswith(b"\n")),
                    }
                except (OSError, UnicodeError) as exc:
                    errors.append({"file": rel, "reason": type(exc).__name__})
            if not manifest:
                return _payload([], manifest, errors, options, analyzed=0, selection=selection)
            config = work / "config.json"
            config.write_text("{}", encoding="utf-8")
            arguments = [
                binary,
                "--config",
                str(config),
                "--mode",
                "weak",
                "--min-tokens",
                str(options.min_tokens),
                "--min-lines",
                str(options.min_lines),
                "--max-size",
                str(max_file_bytes),
                "--summary",
                "--summary-top",
                str(MAX_SOURCE_FILES),
                "--reporters",
                "json,sarif" if options.fingerprints else "json",
                "--output",
                str(output),
                "--no-gitignore",
                "--silent",
                str(sources),
            ]
            if options.normalize_identifiers:
                arguments.insert(-1, "--ignore-identifiers")
            fingerprint_rows = None
            started = time.monotonic()
            try:
                exit_code, memory_scope = _run_detector(arguments, work, options.timeout_seconds)
            except (OSError, subprocess.SubprocessError) as exc:
                raise ValueError("Token backend failed or exceeded its time budget") from exc
            report = output / "jscpd-report.json"
            if (
                exit_code not in {0, 1}
                or not report.is_file()
                or report.stat().st_size > MAX_REPORT_BYTES
            ):
                raise ValueError("Token backend did not produce a bounded valid report")
            try:
                raw = json.loads(report.read_text(encoding="utf-8"))
                duplicates = raw["duplicates"]
                analyzed = raw["statistics"]["total"]["sources"]
                if (
                    type(analyzed) is not int
                    or analyzed < 0
                    or analyzed > len(manifest) * 256
                    or not isinstance(duplicates, list)
                    or len(duplicates) > MAX_TOKEN_PAIRS
                ):
                    raise ValueError("Token backend report exceeds its source/pair contract")
                from victor.core.graph_rag.token_coverage import physical_file_summary

                primary_files = physical_file_summary(raw, manifest, metric="tokens")
                pairs = []
                for duplicate in duplicates:
                    if duplicate.get("kind", "exact") not in {"exact", "renamed"}:
                        raise ValueError("Unexpected token evidence kind")
                    members = []
                    for key in ("firstFile", "secondFile"):
                        item = duplicate[key]
                        name = Path(item["name"])
                        rel = (
                            name.relative_to(sources).as_posix()
                            if name.is_absolute()
                            else name.as_posix()
                        )
                        if (
                            rel not in manifest
                            or rel not in primary_files
                            or type(item["start"]) is not int
                            or type(item["end"]) is not int
                            or not 1 <= item["start"] <= item["end"] <= manifest[rel]["lines"]
                        ):
                            raise ValueError("Token span is outside the selected source snapshot")
                        members.append(
                            {
                                "file": rel,
                                "line": item["start"],
                                "end_line": item["end"],
                                "source_hash": manifest[rel]["sha256"],
                            }
                        )
                    if not include_same_file and members[0]["file"] == members[1]["file"]:
                        continue
                    fragment = duplicate["fragment"]
                    if (
                        not isinstance(fragment, str)
                        or type(duplicate["tokens"]) is not int
                        or duplicate["tokens"] < options.min_tokens
                    ):
                        raise ValueError("Missing token fragment evidence")
                    pairs.append(
                        {
                            "relation": (
                                "renamed_token_fragment"
                                if options.normalize_identifiers
                                else "literal_token_fragment"
                            ),
                            "evidence": "token_sequence",
                            "equivalence_verified": False,
                            "tokens": duplicate["tokens"],
                            "fragment_hash": hashlib.sha256(fragment.encode()).hexdigest(),
                            "members": members,
                        }
                    )
                if options.fingerprints:
                    from victor.core.graph_rag.token_fingerprints import parse_fingerprints

                    carrier = output / "jscpd-report.sarif"
                    if not carrier.is_file() or carrier.stat().st_size > MAX_REPORT_BYTES:
                        raise ValueError(
                            "Token backend did not produce a bounded fingerprint carrier"
                        )
                    fingerprint_rows = parse_fingerprints(
                        json.loads(carrier.read_text(encoding="utf-8")),
                        duplicates,
                        manifest,
                        sources,
                        expected_version=PINNED_JSCPD_VERSION,
                        include_same_file=include_same_file,
                        min_tokens=options.min_tokens,
                    )
            except (KeyError, TypeError, UnicodeError, json.JSONDecodeError) as exc:
                raise ValueError("Malformed token backend report") from exc
            missing = set(manifest) - set(primary_files)
            accounted = len(primary_files) + _account_for_omissions(
                missing,
                manifest,
                sources,
                work,
                config,
                binary,
                options,
                started,
                max_file_bytes,
                errors,
            )
        result = _payload(
            pairs,
            manifest,
            errors,
            options,
            analyzed=accounted,
            eligible_units=analyzed,
            selection=selection,
        )
        result["stats"]["backend_control_files"] = len(missing)
        result["backend"]["coverage_basis"] = "physical_file_summary_and_proven_ineligible_controls"
        result["backend"]["memory_monitor_scope"] = memory_scope
        result["backend"]["memory_budget_bytes"] = MAX_BACKEND_RSS
        if fingerprint_rows is not None:
            result["fingerprints"] = fingerprint_rows
            result["fingerprints_complete"] = result["coverage_complete"]
        return result


def _payload(
    pairs: list[dict[str, Any]],
    manifest: dict[str, Any],
    errors: list[dict[str, str]],
    options: TokenOptions,
    *,
    analyzed: int,
    eligible_units: int = 0,
    selection: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "evidence": "token_sequence",
        "equivalence_verified": False,
        "coverage_complete": bool(manifest) and not errors and analyzed == len(manifest),
        "backend": {
            "name": "jscpd",
            "version": PINNED_JSCPD_VERSION,
            "mode": "weak",
            "min_tokens": options.min_tokens,
            "min_lines": options.min_lines,
            "normalize_identifiers": options.normalize_identifiers,
            "project_config_used": False,
        },
        "stats": {
            "source_files": len(manifest),
            "backend_analyzed_files": analyzed,
            "backend_reported_source_units": eligible_units,
            "fragment_pairs": len(pairs),
        },
        "errors": errors,
        "source_manifest": manifest,
        "source_selection": selection,
        "fragments": pairs,
    }
    if options.fingerprints:
        from victor.core.graph_rag.token_fingerprints import FINGERPRINT_SCHEMA

        payload.update(
            {
                "fingerprint_schema": FINGERPRINT_SCHEMA,
                "fingerprints": [],
                "fingerprints_complete": False,
            }
        )
    return payload

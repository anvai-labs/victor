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
import signal
import tempfile
import time
from typing import Any

PINNED_JSCPD_VERSION = "5.4.2"
MAX_REPORT_BYTES = 64 * 1024 * 1024
MAX_TOKEN_PAIRS = 100_000
MAX_SOURCE_BYTES = 128 * 1024 * 1024
MAX_BACKEND_RSS = 768 * 1024 * 1024


def _run_detector(arguments: list[str], work: Path, timeout: float) -> tuple[int, str]:
    """Bound an owned process tree's runtime, RSS and emitted artifacts."""
    import psutil

    report = work / "report" / "jscpd-report.json"
    log_path = work / "backend.log"
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            arguments,
            cwd=work,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=os.name == "posix",
        )
        started = time.monotonic()
        memory_scope = "owned-tree"
        try:
            parent = psutil.Process(process.pid)
            while process.poll() is None:
                if time.monotonic() - started > timeout:
                    raise ValueError("Token backend exceeded its time budget")
                if any(
                    path.exists() and path.stat().st_size > MAX_REPORT_BYTES
                    for path in (report, log_path)
                ):
                    raise ValueError("Token backend exceeded its artifact budget")
                try:
                    try:
                        owned = [parent, *parent.children(recursive=True)]
                    except (psutil.AccessDenied, PermissionError):
                        # Sandboxed macOS may prohibit global PID enumeration.
                        # Direct-child RSS, time and artifact limits still apply.
                        owned = [parent]
                        memory_scope = "direct-process"
                    if sum(child.memory_info().rss for child in owned) > MAX_BACKEND_RSS:
                        raise ValueError("Token backend exceeded its memory budget")
                except psutil.NoSuchProcess:
                    pass
                time.sleep(0.05)
            return process.wait(), memory_scope
        finally:
            if process.poll() is None:
                if os.name == "posix":
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    try:
                        for child in parent.children(recursive=True):
                            child.kill()
                    except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError):
                        pass
                    process.kill()
                process.wait()


@dataclass(frozen=True)
class TokenOptions:
    binary: str = "jscpd"
    min_tokens: int = 300
    min_lines: int = 3
    normalize_identifiers: bool = False
    timeout_seconds: float = 60

    def __post_init__(self) -> None:
        if not 1 <= self.min_tokens <= 1_000_000 or not 0 <= self.min_lines <= 100_000:
            raise ValueError("Invalid token fragment size")
        if not 0 < self.timeout_seconds <= 300:
            raise ValueError("Invalid token backend timeout")


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

    if not root.is_dir() or max_file_bytes < 1:
        raise ValueError("Token analysis requires a project directory and a positive file bound")
    binary = shutil.which(options.binary)
    if binary is None:
        raise ValueError(
            f"Install jscpd {PINNED_JSCPD_VERSION} locally and pass --jscpd-binary; executable unavailable: {options.binary}"
        )
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
    selected = [
        rel
        for rel in _source_files(root, include_docs)
        if _in_scope(rel, scope, include_tests)
        and not is_path_excluded(root / rel, root, UNIVERSAL_EXCLUDE_PATTERNS)
    ]
    manifest: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, str]] = []
    total_bytes = 0
    with tempfile.TemporaryDirectory(prefix="victor-token-clones-") as directory:
        work = Path(directory)
        sources, output = work / "source", work / "report"
        sources.mkdir()
        for rel in selected:
            origin = root / rel
            try:
                with origin.open("rb") as stream:
                    data = stream.read(max_file_bytes + 1)
                if len(data) > max_file_bytes:
                    errors.append({"file": rel, "reason": "oversized"})
                    continue
                data.decode("utf-8")
                total_bytes += len(data)
                if total_bytes > MAX_SOURCE_BYTES or len(manifest) >= 10_000:
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
                }
            except (OSError, UnicodeError) as exc:
                errors.append({"file": rel, "reason": type(exc).__name__})
        if not manifest:
            return _payload([], manifest, errors, options, analyzed=0)
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
            "--reporters",
            "json",
            "--output",
            str(output),
            "--no-gitignore",
            "--silent",
            str(sources),
        ]
        if options.normalize_identifiers:
            arguments.insert(-1, "--ignore-identifiers")
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
                or analyzed > len(manifest)
                or not isinstance(duplicates, list)
                or len(duplicates) > MAX_TOKEN_PAIRS
            ):
                raise ValueError("Token backend report exceeds its source/pair contract")
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
        except (KeyError, TypeError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("Malformed token backend report") from exc
    result = _payload(pairs, manifest, errors, options, analyzed=analyzed)
    result["backend"]["memory_monitor_scope"] = memory_scope
    result["backend"]["memory_budget_bytes"] = MAX_BACKEND_RSS
    return result


def _payload(
    pairs: list[dict[str, Any]],
    manifest: dict[str, Any],
    errors: list[dict[str, str]],
    options: TokenOptions,
    *,
    analyzed: int,
) -> dict[str, Any]:
    return {
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
            "fragment_pairs": len(pairs),
        },
        "errors": errors,
        "source_manifest": manifest,
        "fragments": pairs,
    }

"""Bounded baseline artifacts and SARIF for source-bound token evidence."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
from typing import Any, Iterator
from urllib.parse import quote

from victor.core.graph_rag.copy_baseline import (
    CopyBaseline,
    CopyEvidence,
    CopyOccurrence,
    Group,
    canonical,
    compare_copies,
    make_baseline,
    parse_copy_report,
    CopyProfile,
    require_current_copy_profile,
)
from victor.core.graph_rag.dedup import settled_thread
from victor.core.graph_rag.source_reader import read_project_source

MAX_BASELINE_BYTES = 16 * 1024 * 1024


def _unique(items: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in items:
        if key in value:
            raise ValueError("Duplicate copy baseline JSON field")
        value[key] = item
    return value


def _decode(data: bytes) -> CopyBaseline:
    if len(data) > MAX_BASELINE_BYTES:
        raise ValueError("Copy baseline exceeds its byte budget")
    try:
        return CopyBaseline.from_dict(json.loads(data, object_pairs_hook=_unique))
    except (RecursionError, UnicodeError) as exc:
        raise ValueError("Malformed copy baseline JSON") from exc


def _load(path: Path) -> CopyBaseline:
    data, info = read_project_source(path.absolute().parent, path.name, MAX_BASELINE_BYTES)
    if info.st_size > MAX_BASELINE_BYTES:
        raise ValueError("Copy baseline exceeds its byte budget")
    return _decode(data)


async def load_copy_baseline(path: Path) -> CopyBaseline:
    """Read a bounded regular file without following a baseline symlink."""
    return await asyncio.to_thread(_load, path)


@contextmanager
def _parent_directory(path: Path) -> Iterator[int]:
    if os.name != "posix" or os.replace not in os.supports_dir_fd:
        # os.replace may not be advertised in supports_dir_fd on qualified
        # POSIX interpreters although rename's identical dir_fd API is present.
        if os.name != "posix" or os.rename not in os.supports_dir_fd:
            raise ValueError("Secure baseline publication is unavailable")
    current = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in path.parent.parts[1:]:
            try:
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=current)
            except FileNotFoundError:
                os.mkdir(part, mode=0o700, dir_fd=current)
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=current)
            os.close(current)
            current = child
        yield current
    finally:
        os.close(current)


def _existing(directory: int, name: str) -> os.stat_result | None:
    try:
        info = os.stat(name, dir_fd=directory, follow_symlinks=False)
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError("Baseline destination must be an exclusive regular file without links")
    return info


def _identity(info: os.stat_result | None) -> tuple[int, ...] | None:
    return (
        (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        if info is not None
        else None
    )


def _publish(path: Path, evidence: CopyEvidence, root: Path) -> None:
    require_current_copy_profile(evidence.profile)
    path = Path(os.path.abspath(path))
    root = root.resolve()
    protected = {root / file for file in evidence.source_files}
    protected.update(
        root / ".victor" / ("project.db" + suffix) for suffix in ("", "-wal", "-shm", "-journal")
    )
    protected.update(
        Path.home() / ".victor" / ("victor.db" + suffix)
        for suffix in ("", "-wal", "-shm", "-journal")
    )
    if str(path).casefold() in {str(item).casefold() for item in protected}:
        raise ValueError("Copy baseline cannot replace source inputs or Victor databases")
    data = (
        json.dumps(make_baseline(evidence).to_dict(), indent=2, allow_nan=False) + "\n"
    ).encode()
    if len(data) > MAX_BASELINE_BYTES:
        raise ValueError("Copy baseline exceeds its byte budget")
    with _parent_directory(path) as directory:
        original = _existing(directory, path.name)
        if original is not None:
            for source in protected:
                try:
                    source_info = source.stat()
                except FileNotFoundError:
                    continue
                if (source_info.st_dev, source_info.st_ino) == (original.st_dev, original.st_ino):
                    raise ValueError(
                        "Copy baseline cannot replace a source/database filesystem alias"
                    )
            descriptor = os.open(
                path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory
            )
            try:
                info = os.fstat(descriptor)
                if _identity(info) != _identity(original) or info.st_size > MAX_BASELINE_BYTES:
                    raise ValueError("Copy baseline destination changed or exceeds its byte budget")
                chunks, remaining = [], MAX_BASELINE_BYTES + 1
                while remaining:
                    chunk = os.read(descriptor, min(remaining, 64 * 1024))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    remaining -= len(chunk)
                _decode(b"".join(chunks))  # Only replace a recognized baseline, never foreign JSON.
            finally:
                os.close(descriptor)
        temporary = f".{path.name}.{secrets.token_hex(12)}.tmp"
        descriptor = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            if _identity(_existing(directory, path.name)) != _identity(original):
                raise ValueError("Copy baseline destination changed before publication")
            parent = path.parent.lstat()
            owned = os.fstat(directory)
            if not stat.S_ISDIR(parent.st_mode) or (parent.st_dev, parent.st_ino) != (
                owned.st_dev,
                owned.st_ino,
            ):
                raise ValueError("Copy baseline directory changed before publication")
            require_current_copy_profile(evidence.profile)
            os.replace(temporary, path.name, src_dir_fd=directory, dst_dir_fd=directory)
        finally:
            try:
                os.unlink(temporary, dir_fd=directory)
            except FileNotFoundError:
                pass


async def write_copy_baseline(path: Path, evidence: CopyEvidence, *, source_root: Path) -> None:
    """Publish once and settle that publication before propagating cancellation."""
    await settled_thread(lambda: _publish(path, evidence, source_root))


async def complete_copy_workflow(
    root: Path,
    report: dict[str, Any],
    profile: CopyProfile,
    *,
    baseline_path: Path | None = None,
    write_path: Path | None = None,
    changed_files: set[str] | None = None,
) -> tuple[CopyEvidence, dict[str, Any] | None]:
    """Apply artifact policy only after a complete scan, with no automatic update."""
    if write_path is not None and (baseline_path is not None or changed_files is not None):
        raise ValueError(
            "Create/update a baseline separately from comparison and changed-file filtering"
        )
    if changed_files is not None and baseline_path is None:
        raise ValueError("Changed-file comparison requires a baseline")
    require_current_copy_profile(profile)
    evidence = parse_copy_report(report, profile)
    comparison = None
    if baseline_path is not None:
        baseline = await load_copy_baseline(baseline_path)
        comparison = compare_copies(evidence, baseline, changed_files=changed_files)
    if write_path is not None:
        await write_copy_baseline(write_path, evidence, source_root=root)
    require_current_copy_profile(profile)
    return evidence, comparison


def copy_sarif(
    root: Path,
    evidence: CopyEvidence | None,
    *,
    comparison: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Export actual token evidence, with grouped count deltas and line-only spans.

    Count deltas cannot identify a unique new physical edge; baselineState is
    deliberately absent. Incomplete scans publish unsuccessful execution.
    """
    successful = evidence is not None and error is None
    run: dict[str, Any] = {
        "tool": {
            "driver": {
                "name": "victor-graph-similar",
                "rules": [
                    {
                        "id": f"victor/token-copy-{kind}",
                        "shortDescription": {"text": f"{kind.title()} token-copy candidates"},
                    }
                    for kind in ("exact", "renamed")
                ],
            }
        },
        "originalUriBaseIds": {"%SRCROOT%": {"uri": root.absolute().as_uri().rstrip("/") + "/"}},
        "invocations": [
            {
                "executionSuccessful": successful,
                "toolExecutionNotifications": (
                    []
                    if successful
                    else [
                        {
                            "level": "error",
                            "message": {"text": (error or "Incomplete token evidence")[:1500]},
                        }
                    ]
                ),
            }
        ],
        "artifacts": [],
        "results": [],
    }
    if successful and evidence is not None:
        profile = evidence.profile.to_dict()
        profile_digest = hashlib.sha256(canonical(profile).encode()).hexdigest()
        grouped: dict[Group, list[CopyOccurrence]] = {}
        for row in evidence.occurrences:
            grouped.setdefault(row.group, []).append(row)
        files = {
            member.file: member.source_hash
            for row in evidence.occurrences
            for member in row.members
        }
        indices = {file: index for index, file in enumerate(sorted(files))}
        run["artifacts"] = [
            {
                "location": {"uri": quote(file, safe="/"), "uriBaseId": "%SRCROOT%"},
                "hashes": {"sha-256": files[file]},
            }
            for file in sorted(files)
        ]
        deltas = (
            {
                (item["fingerprint"], item["kind"], item["format"]): item
                for item in comparison["regressions"]
            }
            if comparison is not None
            else {}
        )
        for group, rows in sorted(grouped.items()):
            members = sorted({member for row in rows for member in row.members})
            local = {member: index + 1 for index, member in enumerate(members)}
            locations = [
                {
                    "id": local[member],
                    "physicalLocation": {
                        "artifactLocation": {
                            "uri": quote(member.file, safe="/"),
                            "uriBaseId": "%SRCROOT%",
                            "index": indices[member.file],
                        },
                        "region": {"startLine": member.line, "endLine": member.end_line},
                    },
                    "properties": {
                        "backend_start_column": member.backend_start_column,
                        "backend_end_column": member.backend_end_column,
                        "backend_columns_qualified": False,
                    },
                }
                for member in members
            ]
            run["results"].append(
                {
                    "ruleId": f"victor/token-copy-{group[1]}",
                    "level": "warning",
                    "message": {
                        "text": f"{len(rows)} token-copy pairs share this backend fingerprint. Review required; behavioral equivalence is unverified."
                    },
                    "locations": locations[:1],
                    "relatedLocations": locations[1:],
                    "partialFingerprints": {
                        "jscpdCloneHash/v1": group[0],
                        "victorCopyGroup/v1": hashlib.sha256(
                            canonical([profile_digest, group]).encode()
                        ).hexdigest(),
                    },
                    "properties": {
                        "evidence": "token_sequence",
                        "equivalence_verified": False,
                        "format": group[2],
                        "copy_pair_count": len(rows),
                        "count_delta": deltas.get(group),
                        "pairs": [
                            {
                                "evidence_id": row.evidence_id,
                                "tokens": row.tokens,
                                "locations": [local[member] for member in row.members],
                            }
                            for row in rows
                        ],
                    },
                }
            )
        run["properties"] = {
            "copy_profile": profile,
            "source_digest": evidence.source_digest,
            "source_count": evidence.source_count,
        }
    return {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [run],
    }

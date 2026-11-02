"""Bounded, read-only SARIF for independent similarity evidence channels.

Families remain single findings. Related locations identify family members,
not verified pair edges. Token copy-count gating has its separate strict exporter.
"""

from __future__ import annotations

import hashlib
from dataclasses import fields
import json
import math
from pathlib import Path
import re
from typing import Any
from urllib.parse import quote

from victor.core.graph_rag.copy_baseline import source_path

_CHANNELS = {"structural", "body", "semantic", "tokens"}
_TIERS = {"exact", "template", "near", "cross_language"}
_MAX_MEMBERS = 100_000
_MAX_BYTES = 64 * 1024**2


def _json_size(value: Any, limit: int) -> int:
    """Count ASCII JSON bytes with bounded scalar allocations and traversal.

    JSONEncoder.iterencode still materializes a whole escaped scalar. Check a
    string's lower bound first, then encode at most 4096 characters at a time.
    Limit nesting/nodes before Python recursion or oversized container work.
    """
    size, nodes = 0, 0
    active: set[int] = set()

    def add(count: int) -> None:
        nonlocal size
        size += count
        if size > limit:
            raise ValueError("SARIF byte budget exceeded")

    def text_size(text: str) -> None:
        if len(text) + 2 > limit - size:
            raise ValueError("SARIF scalar exceeds its byte budget")
        add(2)
        for start in range(0, len(text), 4096):
            add(len(json.dumps(text[start : start + 4096], ensure_ascii=True)) - 2)

    def walk(item: Any, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if depth > 64 or nodes > 2_000_000:
            raise ValueError("SARIF JSON depth/node budget exceeded")
        if isinstance(item, str):
            text_size(item)
        elif item is None:
            add(4)
        elif type(item) is bool:
            add(4 if item else 5)
        elif type(item) in {int, float}:
            if type(item) is int and item.bit_length() > max(0, limit - size) * 4:
                raise ValueError("SARIF number exceeds its byte budget")
            add(len(json.dumps(item, allow_nan=False)))
        elif isinstance(item, (dict, list, tuple)):
            identity = id(item)
            if identity in active:
                raise ValueError("Circular SARIF JSON data")
            active.add(identity)
            add(2)
            entries = item.items() if isinstance(item, dict) else enumerate(item)
            for index, (key, child) in enumerate(entries):
                if index:
                    add(2)  # Default JSON comma and space.
                if isinstance(item, dict):
                    if not isinstance(key, str):
                        raise ValueError("SARIF JSON keys must be strings")
                    text_size(key)
                    add(2)  # Default JSON colon and space.
                walk(child, depth + 1)
            active.remove(identity)
        else:
            raise ValueError("Unsupported SARIF JSON value")

    walk(value, 0)
    return size


def _record(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Expected an evidence object")
    return value


def _failed_output() -> dict[str, Any]:
    return {
        "version": "2.1.0",
        "runs": [
            {
                "tool": {"driver": {"name": "victor-graph-similar"}},
                "invocations": [
                    {
                        "executionSuccessful": False,
                        "toolExecutionNotifications": [
                            {
                                "level": "error",
                                "message": {
                                    "text": "SARIF output exceeds its JSON serialization contract or report budget"
                                },
                            }
                        ],
                    }
                ],
                "results": [],
            }
        ],
    }


def bounded_sarif_text(report: dict[str, Any]) -> tuple[str, bool]:
    """Return the exact bounded stdout representation, without Rich reparsing.

    Reserve two bytes for the platform's final newline. Token-copy and broader
    reports share this last boundary; display formatting cannot expand its size.
    """
    try:
        _json_size(report, _MAX_BYTES - 2)
        successful = report["runs"][0]["invocations"][0]["executionSuccessful"] is True
    except (KeyError, IndexError, TypeError, ValueError, OverflowError, RecursionError):
        report, successful = _failed_output(), False
    return json.dumps(report, allow_nan=False), successful


def _count(value: Any, maximum: int = _MAX_MEMBERS) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError("Invalid evidence count")
    return value


def _rows(value: Any, maximum: int) -> list[Any]:
    if not isinstance(value, list) or len(value) > maximum:
        raise ValueError("Evidence exceeds the SARIF report budget")
    return value


def _hash(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Invalid source/profile hash")
    return value


def _envelope(data: dict[str, Any], channel: str) -> None:
    expected = {
        "structural": "normalized_syntax",
        "body": "function_body_embeddings",
        "tokens": "token_sequence",
        "semantic": "symbol-summary-v1",
    }
    complete = (
        data.get("analysis_complete") if channel == "semantic" else data.get("coverage_complete")
    )
    has_findings = any(data.get(key) for key in ("families", "suppressed", "fragments"))
    needs_evidence = complete is True or has_findings
    for key in ("families", "suppressed", "fragments"):
        if key in data:
            _rows(data[key], 200)
    if (needs_evidence or "evidence" in data) and data.get("evidence") != expected[channel]:
        raise ValueError("Missing or unknown channel evidence contract")
    if complete is True:
        _rows(data.get("fragments" if channel == "tokens" else "families"), 200)
    if channel != "semantic" and needs_evidence and not _record(data.get("source_manifest")):
        raise ValueError("Complete source evidence requires acquired source hashes")
    if channel in {"structural", "tokens"} and complete is True:
        if _count(_record(data.get("stats")).get("source_files")) != len(data["source_manifest"]):
            raise ValueError("Selected-source counts conflict with acquired source hashes")
    if channel == "tokens" and (needs_evidence or "backend" in data):
        from victor.core.graph_rag.token_clones import PINNED_JSCPD_VERSION

        backend = _record(data.get("backend"))
        if backend.get("name") != "jscpd" or backend.get("version") != PINNED_JSCPD_VERSION:
            raise ValueError("Unknown native token backend profile")
    if channel == "body" and (needs_evidence or "profile" in data):
        from victor.core.graph_rag.body_clones import BodyOptions
        from victor.core.graph_rag.body_provenance import ProducerProfile

        profile = _record(data.get("profile"))
        options = BodyOptions(**{field.name: profile[field.name] for field in fields(BodyOptions)})
        producer = dict(_record(profile.get("producer")))
        identity = producer.pop("identity", None)
        declared = ProducerProfile(json.dumps(producer, sort_keys=True, separators=(",", ":")))
        if (
            profile.get("identity") != options.identity
            or profile.get("local_files_only") is not True
            or declared.identity != identity
            or producer.get("input_profile") != options.identity
        ):
            raise ValueError("Body profile/producer identity conflicts with its input contract")
    if channel == "semantic":
        ids = _rows(data.get("embedding_profile_ids", []), 1)
        for identity in ids:
            _hash(identity)


def _manifest(value: Any) -> tuple[dict[str, str], dict[str, int]]:
    raw = _record(value)
    _count(len(raw))
    _json_size(raw, _MAX_BYTES)
    hashes = {
        source_path(name): _hash(entry.get("sha256") if isinstance(entry, dict) else entry)
        for name, entry in raw.items()
    }
    line_limits = {
        name: _count(entry["lines"], 2**31 - 1)
        for name, entry in raw.items()
        if isinstance(entry, dict) and "lines" in entry
    }
    return hashes, line_limits


def _member(value: Any, *, summary: bool) -> dict[str, Any]:
    member = dict(_record(value))
    source_path(member.get("file"))
    line = _count(member.get("line"), 2**31 - 1)
    if not summary and line == 0:
        raise ValueError("Source evidence requires a positive line")
    if "end_line" in member and _count(member["end_line"], 2**31 - 1) < line:
        raise ValueError("Reversed source region")
    digest = member.get("source_hash")
    if not summary or digest is not None:
        _hash(digest)
    return member


def _family(
    value: Any, channel: str, *, suppressed: bool = False
) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    row = _record(value)
    members = [
        _member(m, summary=channel == "semantic") for m in _rows(row.get("members"), _MAX_MEMBERS)
    ]
    if len(members) < 2:
        raise ValueError("Similarity evidence requires at least two members")
    if channel == "structural":
        tier = row.get("tier")
        if tier not in _TIERS:
            raise ValueError("Unknown structural evidence tier")
        rule = f"victor/structural-{tier}"
    elif channel == "tokens":
        tier = row.get("kind")
        if tier not in {"exact", "renamed"} or len(members) != 2:
            raise ValueError("Invalid native token pair")
        rule = f"victor/token-copy-{tier}"
    else:
        rule = f"victor/{'summary' if channel == 'semantic' else 'body'}-similarity"
    expected_evidence = {
        "structural": (
            "portable_operation_shape"
            if row.get("tier") == "cross_language"
            else "normalized_syntax"
        ),
        "body": "function_body_embeddings",
        "semantic": "symbol-summary-v1",
        "tokens": "token_sequence",
    }[channel]
    expected_basis = (
        "normalized_hash_equality"
        if channel == "structural" and row.get("tier") in {"exact", "template"}
        else "native_token_sequence" if channel == "tokens" else "maximum_matching_edge"
    )
    if (
        row.get("evidence", expected_evidence) != expected_evidence
        or row.get("score_basis", expected_basis) != expected_basis
    ):
        raise ValueError("Unknown family evidence/score contract")
    if (
        row.get("cohesion") is not None
        and _record(row["cohesion"]).get("pairwise_minimum_verified", False) is not False
    ):
        raise ValueError("Current vector explanations do not verify a global pairwise minimum")
    properties: dict[str, Any] = {
        key: row[key]
        for key in (
            "evidence",
            "tier",
            "relation",
            "kind",
            "score",
            "similarity",
            "score_basis",
            "matching_edges",
            "matching_pairs",
            "min_edge_similarity",
            "cohesion",
            "summary_input_variants",
            "tokens",
            "format",
            "fragment_hash",
        )
        if key in row
    }
    for key in ("score", "similarity", "min_edge_similarity"):
        score = properties.get(key)
        if score is not None and (
            type(score) not in {int, float} or not math.isfinite(score) or not -1 <= score <= 1
        ):
            raise ValueError("Invalid similarity score")
    for key in ("matching_edges", "matching_pairs"):
        if key in properties:
            _count(properties[key], 1_000_000_000)
    if channel == "tokens":
        if (
            _count(row.get("tokens"), 1_000_000_000) == 0
            or not isinstance(row.get("format"), str)
            or not row["format"]
        ):
            raise ValueError("Missing native token count or format")
        _hash(row.get("fragment_hash"))
    else:
        score_key = "similarity" if channel == "semantic" else "score"
        count_key = "matching_pairs" if channel == "semantic" else "matching_edges"
        if properties.get(score_key) is None:
            raise ValueError("Missing family score")
        count = _count(row.get(count_key), 1_000_000_000)
        if count > len(members) * (len(members) - 1) // 2:
            raise ValueError("Matching-edge count exceeds available member pairs")
        if expected_basis == "maximum_matching_edge" and count < len(members) - 1:
            raise ValueError("Matching-edge count cannot connect this family")
        minimum = properties.get("min_edge_similarity")
        tolerance = 0.000050000001 if channel in {"structural", "semantic"} else 1e-12
        if minimum is not None and minimum > properties[score_key] + tolerance:
            raise ValueError("Minimum matching score exceeds the reported maximum")
        if channel in {"structural", "semantic"}:
            properties["score_rounding_decimals"] = 4
    properties.update(channel=channel, member_count=len(members), equivalence_verified=False)
    properties.setdefault("evidence", expected_evidence)
    properties.setdefault("score_basis", expected_basis)
    if suppressed:
        properties["suppressed"] = str(row.get("suppressed", "policy"))[:1500]
    _json_size(properties, _MAX_BYTES)
    return rule, properties, members


def similarity_sarif(root: Path, payload: dict[str, Any], *, mode: str) -> dict[str, Any]:
    """Export family evidence and explicit failures without inspecting live source.

    A successful summary-index analysis does not establish current source
    coverage. Output limits are reported separately from acquisition failures.
    Malformed/oversized channel evidence fails execution and preserves other
    valid findings. Resource exhaustion never becomes an empty passing report.
    """
    notifications: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    artifact_indices: dict[tuple[str, str | None], int] = {}
    rules: dict[str, dict[str, Any]] = {}
    channel_metadata: dict[str, Any] = {}
    seen_hashes: dict[str, str] = {}
    source_selection: set[str] | None = None
    remaining_bytes = _MAX_BYTES // 2  # Reserve wrapper, relation IDs and delimiters.
    properties: dict[str, Any] = {
        "mode": mode,
        "equivalence_verified": False,
        "coverage_complete": payload.get("coverage_complete") if mode != "semantic" else None,
        "source_freshness": (
            "unverified_index_snapshot" if mode == "semantic" else "acquired_source"
        ),
        "snapshot_conflicts": [],
        "channels": channel_metadata,
        "report_truncated": False,
    }
    successful = (
        payload.get("analysis_complete") is True
        if mode == "semantic"
        else payload.get("coverage_complete") is True
    )

    def failure(message: str) -> None:
        nonlocal successful
        successful = False
        notifications.append({"level": "error", "message": {"text": message[:1500]}})

    def observe_hash(name: str, digest: str) -> None:
        previous = seen_hashes.setdefault(name, digest)
        if previous != digest and name not in properties["snapshot_conflicts"]:
            properties["snapshot_conflicts"].append(name)
            failure("Source hash conflicts between evidence channels")

    def location(member: dict[str, Any]) -> dict[str, Any]:
        nonlocal remaining_bytes
        name, digest = member["file"], member.get("source_hash")
        key = name, digest
        uri = {"uri": quote(name, safe="/"), "uriBaseId": "%SRCROOT%"}
        artifact = {"location": uri, **({"hashes": {"sha-256": digest}} if digest else {})}
        new_artifact = key not in artifact_indices
        physical: dict[str, Any] = {
            "artifactLocation": {**uri, "index": artifact_indices.get(key, len(artifacts))}
        }
        if member["line"]:
            physical["region"] = {"startLine": member["line"]}
            if member.get("end_line"):
                physical["region"]["endLine"] = member["end_line"]
        result = {
            "physicalLocation": physical,
            "properties": {
                key: member[key]
                for key in (
                    "id",
                    "symbol",
                    "language",
                    "source_hash",
                    "input_hash",
                    "source_kind",
                    "region_index",
                    "recovered_from_partial_file",
                    "truncated",
                )
                if key in member
            },
        }
        remaining_bytes -= _json_size(result, remaining_bytes)
        if new_artifact:
            remaining_bytes -= _json_size(artifact, remaining_bytes)
            artifact_indices[key] = len(artifacts)
            artifacts.append(artifact)
        return result

    if mode not in _CHANNELS | {"hybrid"}:
        failure("Unknown similarity mode")
    if not successful:
        failure(str(payload.get("error") or "Similarity analysis has incomplete coverage"))
    if payload.get("snapshot_conflicts"):
        failure("Source snapshots differ between evidence channels")
    channels = payload.get("channels", {}) if mode == "hybrid" else {mode: payload}
    try:
        properties["snapshot_conflicts"] = [
            source_path(name) for name in _rows(payload.get("snapshot_conflicts", []), _MAX_MEMBERS)
        ]
        channels = _record(channels)
        if not channels or len(channels) > 4:
            raise ValueError("Invalid evidence channel set")
        if mode == "hybrid" and set(channels) != {"structural", "body", "tokens"}:
            failure(
                "Hybrid report is missing an independent source channel or includes an unexpected channel"
            )
        consumed = 0
        for channel, raw in channels.items():
            try:
                if channel not in _CHANNELS:
                    raise ValueError("Unknown evidence channel")
                data = _record(raw)
                metadata: dict[str, Any] = {
                    key: data[key]
                    for key in (
                        "evidence",
                        "coverage_complete",
                        "error",
                        "errors",
                        "stats",
                        "profile",
                        "source_coverage",
                        "source_selection",
                        "backend",
                        "ffi_links",
                        "threshold",
                        "symbols_indexed",
                        "neighbors",
                        "embedding_profile_ids",
                        "grouping",
                        "near_grouping",
                        "similarity_basis",
                    )
                    if key in data
                }
                remaining_bytes -= _json_size(metadata, remaining_bytes)
                channel_metadata[channel] = metadata
                _envelope(data, channel)
                manifest, line_limits = (
                    _manifest(data["source_manifest"]) if "source_manifest" in data else (None, {})
                )
                if "source_manifest" in data:
                    assert manifest is not None
                    if source_selection is None:
                        source_selection = set(manifest)
                    else:
                        missing = source_selection.symmetric_difference(manifest)
                        if missing:
                            properties["snapshot_conflicts"] = sorted(
                                set(properties["snapshot_conflicts"]) | missing
                            )
                            failure("Source selections differ between evidence channels")
                    metadata["source_manifest_files"] = len(manifest)
                    hasher = hashlib.sha256()
                    for chunk in json.JSONEncoder(sort_keys=True, separators=(",", ":")).iterencode(
                        manifest
                    ):
                        hasher.update(chunk.encode())
                    metadata["source_manifest_sha256"] = hasher.hexdigest()
                    for name, digest in manifest.items():
                        observe_hash(name, digest)
                if channel == "semantic":
                    metadata["profile_provenance"] = (
                        "declared_index_profile_identity"
                        if data.get("embedding_profile_ids")
                        else "unavailable"
                    )
                    if not data.get("embedding_profile_ids"):
                        notifications.append(
                            {
                                "level": "warning",
                                "message": {
                                    "text": "Summary profile identity unavailable; model provenance remains unqualified"
                                },
                            }
                        )
                if "ffi_links" in data:
                    links = _rows(data["ffi_links"], 200)
                    total_links = _count(data.get("total_ffi_links", len(links)), 1_000_000_000)
                    if total_links < len(links):
                        raise ValueError("Reported FFI count contradicts output")
                    metadata["total_ffi_links"] = total_links
                    properties["report_truncated"] |= total_links > len(links)
                if channel != "semantic" and data.get("coverage_complete") is not True:
                    failure(f"{channel}: " + str(data.get("error") or "incomplete coverage"))
                if data.get("error") or data.get("errors"):
                    failure(f"{channel}: analysis reported errors")
                for field in ("families", "suppressed", "fragments"):
                    rows = _rows(data.get(field, []), 200)
                    total = _count(data.get("total_" + field, len(rows)), 1_000_000_000)
                    if total < len(rows):
                        raise ValueError("Reported evidence count contradicts output")
                    metadata["total_" + field] = total
                    metadata["reported_" + field] = len(rows)
                    properties["report_truncated"] |= total > len(rows)
                    for row in rows:
                        _json_size(row, _MAX_BYTES)
                        rule, details, members = _family(
                            row, channel, suppressed=field == "suppressed"
                        )
                        consumed += len(members)
                        if consumed > _MAX_MEMBERS:
                            raise ValueError("SARIF member budget exceeded")
                        for member in members:
                            member_digest = member.get("source_hash")
                            if (
                                manifest is not None
                                and manifest.get(member["file"]) != member_digest
                            ):
                                raise ValueError(
                                    "Family source hash conflicts with acquired manifest"
                                )
                            if member_digest is not None:
                                observe_hash(member["file"], member_digest)
                            line_limit = line_limits.get(member["file"])
                            if (
                                line_limit is not None
                                and max(member["line"], member.get("end_line", member["line"]))
                                > line_limit
                            ):
                                raise ValueError(
                                    "Evidence region exceeds acquired source line count"
                                )
                        remaining_bytes -= _json_size(details, remaining_bytes)
                        locations = [location(member) for member in members]
                        rules[rule] = {
                            "id": rule,
                            "shortDescription": {
                                "text": f"{channel.title()} similarity candidates"
                            },
                        }
                        results.append(
                            {
                                "ruleId": rule,
                                "level": "note",
                                "kind": "review",
                                "message": {
                                    "text": f"{channel.title()} candidate {'pair' if channel == 'tokens' else 'family'} ({len(members)} members). Review evidence and contracts before refactoring."
                                },
                                "locations": locations[:1],
                                "relatedLocations": [
                                    {"id": i, **item} for i, item in enumerate(locations[1:], 1)
                                ],
                                "properties": details,
                            }
                        )
            except (
                ImportError,
                KeyError,
                TypeError,
                ValueError,
                OverflowError,
                RecursionError,
            ) as exc:
                failure(f"{channel}: {exc}")
    except (TypeError, ValueError, RecursionError) as exc:
        failure(str(exc))
    report: dict[str, Any] = {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {"driver": {"name": "victor-graph-similar", "rules": list(rules.values())}},
                "originalUriBaseIds": {
                    "%SRCROOT%": {"uri": root.absolute().as_uri().rstrip("/") + "/"}
                },
                "invocations": [
                    {"executionSuccessful": successful, "toolExecutionNotifications": notifications}
                ],
                "results": results,
                "artifacts": artifacts,
                "properties": properties,
            }
        ],
    }
    try:
        _json_size(report, _MAX_BYTES - 2)
    except (TypeError, ValueError, OverflowError, RecursionError):
        report = _failed_output()
    return report

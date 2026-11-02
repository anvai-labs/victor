"""Strict adapter for the pinned backend's SARIF fingerprint carrier.

This is not a general SARIF reader. Every carrier result must match the JSON
token report, including its source-relative spans and token count. Columns are
used only to correlate backend records; returned source references use lines.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
import re
from typing import Any

FINGERPRINT_SCHEMA = "jscpdCloneHash/v1"
MAX_FINGERPRINTS = 100_000
MAX_POSITION = 128 * 1024 * 1024
Location = tuple[str, int, int, int, int]
Signature = tuple[tuple[Location, Location], int]


def _integer(value: Any, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value <= MAX_POSITION:
        raise ValueError("Invalid token fingerprint position/count")
    return value


def _carrier_path(value: Any) -> str:
    # Pinned jscpd 5.4.2 emits raw POSIX filenames in SARIF URI fields. Do not
    # decode percent sequences or interpret #/? as URI delimiters. A future
    # standards-compliant writer must encode these paths when exporting SARIF.
    if (
        not isinstance(value, str)
        or not value
        or "\0" in value
        or PurePosixPath(value).is_absolute()
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise ValueError("Invalid source-relative token fingerprint path")
    return value


def _location(
    file: str,
    start: Any,
    end: Any,
    start_column: Any,
    end_column: Any,
    manifest: dict[str, dict[str, Any]],
) -> Location:
    start, end = _integer(start, 1), _integer(end, 1)
    first, last = _integer(start_column), _integer(end_column)
    if file not in manifest or not start <= end <= manifest[file]["lines"]:
        raise ValueError("Token fingerprint span is outside the selected source snapshot")
    if start == end and last < first:
        raise ValueError("Invalid token fingerprint column range")
    return file, start, end, first, last


def _json_location(item: Any, manifest: dict[str, dict[str, Any]], sources: Path) -> Location:
    path = Path(item["name"])
    file = path.relative_to(sources).as_posix() if path.is_absolute() else path.as_posix()
    if (
        _integer(item["startLoc"]["line"], 1) != item["start"]
        or _integer(item["endLoc"]["line"], 1) != item["end"]
    ):
        raise ValueError("Token JSON span metadata disagrees")
    return _location(
        file,
        item["start"],
        item["end"],
        item["startLoc"]["column"],
        item["endLoc"]["column"],
        manifest,
    )


def _sarif_location(item: Any, manifest: dict[str, dict[str, Any]]) -> Location:
    physical = item["physicalLocation"]
    artifact, region = physical["artifactLocation"], physical["region"]
    if artifact["uriBaseId"] != "%SRCROOT%":
        raise ValueError("Unknown token fingerprint URI base")
    file = _carrier_path(artifact["uri"])
    return _location(
        file,
        region["startLine"],
        region["endLine"],
        _integer(region["startColumn"], 1) - 1,
        _integer(region["endColumn"], 1) - 1,
        manifest,
    )


def _signature(left: Location, right: Location, tokens: int) -> Signature:
    first, second = sorted((left, right))
    if first == second:
        raise ValueError("Token fingerprint cannot reference a physical self-pair")
    return (first, second), tokens


def parse_fingerprints(
    raw: Any,
    duplicates: list[dict[str, Any]],
    manifest: dict[str, dict[str, Any]],
    sources: Path,
    *,
    expected_version: str,
    include_same_file: bool = False,
    min_tokens: int = 1,
) -> list[dict[str, Any]]:
    """Cross-check bounded native carriers; never infer identities from filenames."""
    try:
        if raw["version"] != "2.1.0" or not isinstance(raw["runs"], list) or len(raw["runs"]) != 1:
            raise ValueError("Invalid pinned token fingerprint carrier version/runs")
        run = raw["runs"][0]
        driver = run["tool"]["driver"]
        if driver["name"] != "jscpd" or driver["version"] != expected_version:
            raise ValueError("Token fingerprint producer does not match the pinned backend")
        results = run["results"]
        if (
            not isinstance(results, list)
            or len(results) != len(duplicates)
            or len(results) > MAX_FINGERPRINTS
        ):
            raise ValueError("Token fingerprint and JSON result counts disagree")
        if not results:
            return []
        base = run["originalUriBaseIds"]["%SRCROOT%"]["uri"]
        if (
            not isinstance(base, str)
            or not base.endswith("/")
            or not base.startswith("file:///")
            or Path(base.removeprefix("file://")).resolve() != sources.resolve()
        ):
            raise ValueError("Token fingerprint carrier belongs to a different source root")
        expected: dict[Signature, tuple[str, str]] = {}
        for duplicate in duplicates:
            kind, format_ = duplicate.get("kind", "exact"), duplicate["format"]
            if (
                kind not in {"exact", "renamed"}
                or not isinstance(format_, str)
                or not re.fullmatch(r"[\w.+-]{1,100}", format_)
            ):
                raise ValueError("Invalid token fingerprint kind/format")
            key = _signature(
                _json_location(duplicate["firstFile"], manifest, sources),
                _json_location(duplicate["secondFile"], manifest, sources),
                _integer(duplicate["tokens"], min_tokens),
            )
            if key in expected:
                raise ValueError("Duplicate or ambiguous physical token evidence")
            expected[key] = (kind, format_)
        parsed = []
        for result in results:
            fingerprint = result["partialFingerprints"][FINGERPRINT_SCHEMA]
            if not isinstance(fingerprint, str) or not re.fullmatch(r"[0-9a-f]{16}", fingerprint):
                raise ValueError("Invalid pinned token fingerprint")
            if (
                result["ruleId"] != "jscpd/duplicate-code"
                or result["properties"]["clone_hash"] != fingerprint
            ):
                raise ValueError("Token fingerprint metadata disagrees")
            if len(result["locations"]) != 1 or len(result["relatedLocations"]) != 1:
                raise ValueError("Token fingerprint requires exactly two source locations")
            key = _signature(
                _sarif_location(result["locations"][0], manifest),
                _sarif_location(result["relatedLocations"][0], manifest),
                _integer(result["properties"]["token_count"], min_tokens),
            )
            if key not in expected:
                raise ValueError("Token fingerprint has no matching JSON token evidence")
            kind, format_ = expected.pop(key)
            locations, tokens = key
            if not include_same_file and locations[0][0] == locations[1][0]:
                continue
            parsed.append(
                {
                    "fingerprint": fingerprint,
                    "fingerprint_schema": FINGERPRINT_SCHEMA,
                    "format": format_,
                    "kind": kind,
                    "tokens": tokens,
                    "members": [
                        {
                            "file": file,
                            "line": start,
                            "end_line": end,
                            "source_hash": manifest[file]["sha256"],
                        }
                        for file, start, end, _, _ in locations
                    ],
                }
            )
        if expected:
            raise ValueError("JSON token evidence is missing its fingerprint carrier")
        return parsed
    except (KeyError, TypeError, AttributeError, UnicodeError) as exc:
        raise ValueError("Malformed pinned token fingerprint carrier") from exc

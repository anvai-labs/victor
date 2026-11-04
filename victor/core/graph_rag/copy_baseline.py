"""Compatible token-copy count baselines; semantic families are never CI gates."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from victor.core.graph_rag.token_clones import PINNED_JSCPD_VERSION, TokenOptions
from victor.core.graph_rag.token_fingerprints import FINGERPRINT_SCHEMA, source_evidence_id

SCHEMA = "victor-copy-baseline-v1"
MAX_PAIRS = 100_000
MAX_FILES = 20_000
Group = tuple[str, str, str]  # opaque native fingerprint, evidence kind, format
_PRODUCERS = (
    "token_clones.py",
    "token_fingerprints.py",
    "token_coverage.py",
    "source_inventory.py",
    "bounded_process.py",
    "source_reader.py",
    "structural_clones.py",
    "source_regions.py",
    "language_parsers.py",
    "exclude_patterns.py",
    "copy_baseline.py",
)


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _integer(value: Any, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError("Invalid bounded copy count or position")
    return value


def _hash(value: Any, length: int = 64) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{%d}" % length, value):
        raise ValueError("Invalid copy identity")
    return value


def source_path(value: Any) -> str:
    """Validate raw POSIX source names, preserving percent and URI delimiters."""
    if (
        not isinstance(value, str)
        or len(value.encode("utf-8")) > 16_384
        or "\0" in value
        or value.startswith("/")
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise ValueError("Expected a source-relative copy path")
    return value


@dataclass(frozen=True)
class CopyProfile:
    min_tokens: int
    min_lines: int
    normalize_identifiers: bool
    scope: str | None
    include_tests: bool
    include_docs: bool
    include_same_file: bool
    source_policy_digest: str
    producers: tuple[tuple[str, str], ...]
    exclude_patterns: tuple[str, ...] = ()
    profile_version: int = 2

    def __post_init__(self) -> None:
        _integer(self.min_tokens, 1, 1_000_000)
        _integer(self.min_lines, 0, 100_000)
        _hash(self.source_policy_digest)
        if type(self.profile_version) is not int or self.profile_version not in {1, 2}:
            raise ValueError("Unsupported copy profile version")
        checked = TokenOptions(fingerprints=True, exclude_patterns=self.exclude_patterns)
        if checked.exclude_patterns != self.exclude_patterns or (
            self.profile_version == 1 and self.exclude_patterns
        ):
            raise ValueError("Noncanonical or unsupported copy exclusions")
        if any(
            type(value) is not bool
            for value in (
                self.normalize_identifiers,
                self.include_tests,
                self.include_docs,
                self.include_same_file,
            )
        ):
            raise ValueError("Copy profile flags must be boolean")
        if self.scope is not None:
            source_path(self.scope.removesuffix("/"))
        if tuple(name for name, _ in self.producers) != _PRODUCERS:
            raise ValueError("Unknown copy producer profile")
        for _, digest in self.producers:
            _hash(digest)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": f"victor-copy-profile-v{self.profile_version}",
            "fingerprint_schema": FINGERPRINT_SCHEMA,
            "source_policy_digest": self.source_policy_digest,
            "backend": {
                "name": "jscpd",
                "version": PINNED_JSCPD_VERSION,
                "mode": "weak",
                "min_tokens": self.min_tokens,
                "min_lines": self.min_lines,
                "normalize_identifiers": self.normalize_identifiers,
            },
            "selection": {
                "scope": self.scope,
                "include_tests": self.include_tests,
                "include_docs": self.include_docs,
                "include_same_file": self.include_same_file,
            },
            "producers": dict(self.producers),
        }
        if self.profile_version == 2:
            result["selection"]["exclude_patterns"] = list(self.exclude_patterns)
        return result

    @classmethod
    def from_dict(cls, value: Any) -> CopyProfile:
        try:
            backend, selection, producers = value["backend"], value["selection"], value["producers"]
            if value["schema"] not in {"victor-copy-profile-v1", "victor-copy-profile-v2"}:
                raise ValueError("Unsupported copy profile schema")
            legacy = value["schema"] == "victor-copy-profile-v1"
            profile = cls(
                backend["min_tokens"],
                backend["min_lines"],
                backend["normalize_identifiers"],
                selection["scope"],
                selection["include_tests"],
                selection["include_docs"],
                selection["include_same_file"],
                value["source_policy_digest"],
                tuple((name, producers[name]) for name in _PRODUCERS),
                () if legacy else tuple(selection["exclude_patterns"]),
                1 if legacy else 2,
            )
            if canonical(value) != canonical(profile.to_dict()):
                raise ValueError("Unsupported or ambiguous copy profile")
            return profile
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError("Malformed copy profile") from exc


def copy_profile(
    options: TokenOptions,
    *,
    scope: str | None = None,
    include_tests: bool = False,
    include_docs: bool = False,
    include_same_file: bool = False,
) -> CopyProfile:
    """Capture count semantics and source-selection producer generation.

    Executable installation paths and runtime budgets do not alter copy identity.
    Changed producer sources conservatively require a new reviewed baseline.
    """
    if not options.fingerprints:
        raise ValueError("Copy workflows require pinned token fingerprints")
    directory = Path(__file__).parent
    from victor.core.graph_rag.exclude_patterns import UNIVERSAL_EXCLUDE_PATTERNS
    from victor.core.graph_rag.language_parsers import SOURCE_LANGUAGES, LANGUAGE_MODULES
    from victor.core.graph_rag.semantic_clones import TEST_PATH_MARKERS
    from victor.core.graph_rag.source_regions import DOC_SUFFIXES, EMBEDDED_SUFFIXES

    policy = {
        "source_languages": SOURCE_LANGUAGES,
        "region_languages": sorted(LANGUAGE_MODULES),
        "embedded_suffixes": sorted(EMBEDDED_SUFFIXES),
        "doc_suffixes": sorted(DOC_SUFFIXES) if include_docs else [],
        "exclude_patterns": sorted(UNIVERSAL_EXCLUDE_PATTERNS),
        "test_markers": sorted(TEST_PATH_MARKERS),
    }
    return CopyProfile(
        options.min_tokens,
        options.min_lines,
        options.normalize_identifiers,
        scope,
        include_tests,
        include_docs,
        include_same_file,
        _digest(policy),
        tuple(
            (name, hashlib.sha256((directory / name).read_bytes()).hexdigest())
            for name in _PRODUCERS
        ),
        exclude_patterns=options.exclude_patterns,
    )


def require_current_copy_profile(profile: CopyProfile) -> None:
    """Reject selected producer-source changes across a workflow boundary."""
    current = copy_profile(
        TokenOptions(
            min_tokens=profile.min_tokens,
            min_lines=profile.min_lines,
            normalize_identifiers=profile.normalize_identifiers,
            fingerprints=True,
            exclude_patterns=profile.exclude_patterns,
        ),
        scope=profile.scope,
        include_tests=profile.include_tests,
        include_docs=profile.include_docs,
        include_same_file=profile.include_same_file,
    )
    if current != profile:
        raise ValueError("Copy producer sources changed; run a fresh scan")


@dataclass(frozen=True, order=True)
class CopyLocation:
    file: str
    line: int
    end_line: int
    source_hash: str
    backend_start_column: int
    backend_end_column: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "line": self.line,
            "end_line": self.end_line,
            "source_hash": self.source_hash,
            "backend_start_column": self.backend_start_column,
            "backend_end_column": self.backend_end_column,
        }


@dataclass(frozen=True)
class CopyOccurrence:
    evidence_id: str
    group: Group
    tokens: int
    members: tuple[CopyLocation, CopyLocation]


@dataclass(frozen=True)
class CopyEvidence:
    profile: CopyProfile
    source_digest: str
    source_count: int
    source_files: tuple[str, ...]
    occurrences: tuple[CopyOccurrence, ...]


def _group(row: Any) -> Group:
    fingerprint, kind, format_ = _hash(row["fingerprint"], 16), row["kind"], row["format"]
    if (
        kind not in {"exact", "renamed"}
        or not isinstance(format_, str)
        or not re.fullmatch(r"[\w.+-]{1,100}", format_)
    ):
        raise ValueError("Invalid copy evidence kind/format")
    return fingerprint, kind, format_


def parse_copy_report(raw: Any, profile: CopyProfile) -> CopyEvidence:
    """Freeze complete source-bound evidence; never accept a truncated carrier."""
    try:
        if (
            raw["coverage_complete"] is not True
            or raw["fingerprints_complete"] is not True
            or raw["evidence"] != "token_sequence"
            or raw["fingerprint_schema"] != FINGERPRINT_SCHEMA
            or raw["errors"] != []
        ):
            raise ValueError("Copy workflows require complete error-free token evidence")
        if (
            canonical({key: raw["backend"].get(key) for key in profile.to_dict()["backend"]})
            != canonical(profile.to_dict()["backend"])
            or raw["backend"].get("project_config_used") is not False
        ):
            raise ValueError("Copy evidence backend differs from its profile")
        if canonical(raw["source_selection"]) != canonical(profile.to_dict()["selection"]):
            raise ValueError("Copy evidence source selection differs from its profile")
        manifest, rows, stats = raw["source_manifest"], raw["fingerprints"], raw["stats"]
        if (
            not isinstance(manifest, dict)
            or not 1 <= len(manifest) <= MAX_FILES
            or not isinstance(rows, list)
            or len(rows) > MAX_PAIRS
        ):
            raise ValueError("Copy evidence exceeds its source/pair budget")
        for name, info in manifest.items():
            source_path(name)
            _hash(info["sha256"])
            _integer(info["lines"], 0, 128 * 1024 * 1024)
        if (
            _integer(stats["source_files"], 1, MAX_FILES) != len(manifest)
            or _integer(stats["backend_analyzed_files"], 1, MAX_FILES) != len(manifest)
            or _integer(stats["fragment_pairs"], 0, MAX_PAIRS) != len(rows)
        ):
            raise ValueError("Copy evidence coverage/count metadata disagrees")
        occurrences = []
        seen = set()
        physical = set()
        for row in rows:
            evidence_id = _hash(row["evidence_id"])
            if evidence_id in seen:
                raise ValueError("Duplicate physical copy evidence identity")
            seen.add(evidence_id)
            if row["fingerprint_schema"] != FINGERPRINT_SCHEMA or len(row["members"]) != 2:
                raise ValueError("Copy evidence needs two source-bound locations")
            group, members = _group(row), []
            for item in row["members"]:
                file = source_path(item["file"])
                info = manifest[file]
                first = _integer(item["line"], 1, info["lines"])
                last = _integer(item["end_line"], first, info["lines"])
                if _hash(item["source_hash"]) != info["sha256"]:
                    raise ValueError("Copy member is not bound to its source snapshot")
                first_column = _integer(item["backend_start_column"], 0, 128 * 1024 * 1024)
                last_column = _integer(item["backend_end_column"], 0, 128 * 1024 * 1024)
                if first == last and last_column < first_column:
                    raise ValueError("Invalid backend copy column span")
                members.append(
                    CopyLocation(file, first, last, item["source_hash"], first_column, last_column)
                )
            signature = tuple(sorted(members))
            if members[0] == members[1] or signature in physical:
                raise ValueError("Duplicate or self-paired physical copy signature")
            physical.add(signature)
            if source_evidence_id(row) != evidence_id:
                raise ValueError(
                    "Copy evidence identity is not bound to its source/span/producer evidence"
                )
            if not profile.include_same_file and members[0].file == members[1].file:
                raise ValueError("Unexpected same-file copy evidence")
            occurrences.append(
                CopyOccurrence(
                    evidence_id,
                    group,
                    _integer(row["tokens"], profile.min_tokens, 128 * 1024 * 1024),
                    (members[0], members[1]),
                )
            )
        return CopyEvidence(
            profile, _digest(manifest), len(manifest), tuple(sorted(manifest)), tuple(occurrences)
        )
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("Malformed copy evidence") from exc


@dataclass(frozen=True)
class CopyBaseline:
    profile: CopyProfile
    source_digest: str
    source_count: int
    counts: tuple[tuple[Group, int], ...]

    def __post_init__(self) -> None:
        _hash(self.source_digest)
        _integer(self.source_count, 1, MAX_FILES)
        seen = set()
        total = 0
        for group, count in self.counts:
            if _group(dict(zip(("fingerprint", "kind", "format"), group, strict=True))) in seen:
                raise ValueError("Duplicate baseline copy group")
            seen.add(group)
            total += _integer(count, 1, MAX_PAIRS)
        if total > MAX_PAIRS:
            raise ValueError("Baseline exceeds its copy budget")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "profile": self.profile.to_dict(),
            "profile_digest": _digest(self.profile.to_dict()),
            "source_digest": self.source_digest,
            "source_count": self.source_count,
            "counts": [
                {"fingerprint": key[0], "kind": key[1], "format": key[2], "count": count}
                for key, count in sorted(self.counts)
            ],
        }

    @classmethod
    def from_dict(cls, value: Any) -> CopyBaseline:
        try:
            if not isinstance(value["counts"], list) or len(value["counts"]) > MAX_PAIRS:
                raise ValueError("Baseline exceeds its copy budget")
            baseline = cls(
                CopyProfile.from_dict(value["profile"]),
                value["source_digest"],
                value["source_count"],
                tuple((_group(row), row["count"]) for row in value["counts"]),
            )
            if canonical(value) != canonical(baseline.to_dict()):
                raise ValueError("Unsupported or ambiguous copy baseline")
            return baseline
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError("Malformed copy baseline") from exc


def make_baseline(evidence: CopyEvidence) -> CopyBaseline:
    counts = Counter(row.group for row in evidence.occurrences)
    return CopyBaseline(
        evidence.profile,
        evidence.source_digest,
        evidence.source_count,
        tuple(sorted(counts.items())),
    )


def compare_copies(
    current: CopyEvidence, baseline: CopyBaseline, *, changed_files: set[str] | None = None
) -> dict[str, Any]:
    """Attribute count increases to entire fingerprint groups, not arbitrary edges."""
    if current.profile != baseline.profile:
        raise ValueError(
            "Copy baseline requires a compatible producer/options/source-selection profile"
        )
    if changed_files is not None:
        if len(changed_files) > MAX_FILES:
            raise ValueError("Changed source paths exceed their budget")
        for name in changed_files:
            source_path(name)
    previous = dict(baseline.counts)
    grouped: dict[Group, list[CopyOccurrence]] = {}
    for row in current.occurrences:
        grouped.setdefault(row.group, []).append(row)
    regressions = []
    for key, rows in sorted(grouped.items()):
        delta = len(rows) - previous.get(key, 0)
        if delta <= 0:
            continue
        touches = changed_files is None or any(
            member.file in changed_files for row in rows for member in row.members
        )
        regressions.append(
            {
                "fingerprint": key[0],
                "kind": key[1],
                "format": key[2],
                "previous_count": previous.get(key, 0),
                "current_count": len(rows),
                "new_copy_pairs": delta,
                "touches_changed_files": touches,
            }
        )
    return {
        "schema": "victor-copy-comparison-v1",
        "profile_compatible": True,
        "new_copy_pairs": sum(item["new_copy_pairs"] for item in regressions),
        "changed_new_copy_pairs": sum(
            item["new_copy_pairs"] for item in regressions if item["touches_changed_files"]
        ),
        "changed_files_filter": sorted(changed_files) if changed_files is not None else None,
        "regressions": regressions,
    }

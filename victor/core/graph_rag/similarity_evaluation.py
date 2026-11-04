"""Reviewed-pair evaluation with explicit abstentions and split isolation.

Scores classify only labelled pairs. They do not measure whole-repository retrieval
recall, certify equivalence, or turn a connected component into pairwise evidence.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from itertools import groupby
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import PurePosixPath
import re
from types import MappingProxyType
from typing import Any, Sequence

SCHEMA = "victor-similarity-corpus-v1"
CHANNELS = {"syntax", "portable", "body"}
RELATIONS = {"copy", "structure", "behavior"}
MAX_UNITS = 10_000
MAX_PAIRS = 50_000


def digest(value: Any) -> str:
    """Canonical identity for a manifest, detector profile or observation set."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _text(value: Any) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 4096:
        raise ValueError("Expected bounded nonempty text")
    return value


def _hash(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Expected a lowercase SHA-256 identity")
    return value


def _record(value: Any, keys: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(keys.split()):
        raise ValueError("Unexpected or missing corpus fields")
    return value


@dataclass(frozen=True)
class Project:
    id: str
    repository: str
    revision: str
    split: str
    origin: str


@dataclass(frozen=True)
class Unit:
    id: str
    project: str
    file: str
    language: str
    symbol: str
    line: int
    end_line: int
    source_hash: str


@dataclass(frozen=True)
class Pair:
    id: str
    left: str
    right: str
    relation: str
    positive: bool
    family: str
    rationale: str
    domain: str
    reviewed_by: str


@dataclass(frozen=True)
class Corpus:
    projects: Mapping[str, Project]
    units: Mapping[str, Unit]
    pairs: tuple[Pair, ...]
    identity: str

    @classmethod
    def from_dict(cls, raw: Any) -> Corpus:
        """Reject ambiguous labels and declared repository/family/source leakage."""
        data = _record(raw, "schema projects units pairs")
        if data["schema"] != SCHEMA:
            raise ValueError("Unsupported evaluation corpus schema")
        for key, limit in (("projects", 1000), ("units", MAX_UNITS), ("pairs", MAX_PAIRS)):
            if not isinstance(data[key], list) or not 1 <= len(data[key]) <= limit:
                raise ValueError(f"Invalid {key} corpus budget")
        projects, units, pairs = {}, {}, []
        isolation: dict[tuple[str, str], str] = {}

        def isolate(kind: str, key: str, split: str) -> None:
            previous = isolation.setdefault((kind, key), split)
            if previous != split:
                raise ValueError(f"Cross-split {kind} leakage: {key}")

        for value in data["projects"]:
            item = _record(value, "id repository revision split origin")
            project = Project(**{key: _text(val) for key, val in item.items()})
            if len(project.id) > 256:
                raise ValueError("Project identity exceeds its budget")
            if project.id in projects or project.split not in {"calibration", "test"}:
                raise ValueError("Duplicate project or unknown split")
            if project.origin not in {"controlled_fixture", "repository"}:
                raise ValueError("Unknown corpus origin")
            if project.origin == "repository" and not re.fullmatch(
                r"[0-9a-f]{40}", project.revision
            ):
                raise ValueError("Repository sources require an immutable Git revision")
            projects[project.id] = project
            isolate("repository", project.repository, project.split)
        snapshots: dict[tuple[str, str, str], tuple[str, str]] = {}
        references_seen: set[tuple[str, str, str, str, int, int]] = set()
        for value in data["units"]:
            item = _record(value, "id project file language symbol line end_line source_hash")
            for key in ("id", "project", "file", "language", "symbol"):
                _text(item[key])
            if any(type(item[key]) is not int for key in ("line", "end_line")) or not (
                1 <= item["line"] <= item["end_line"] <= 10_000_000
            ):
                raise ValueError("Invalid source span")
            path = PurePosixPath(item["file"])
            if (
                path.is_absolute()
                or "\\" in item["file"]
                or any(part in {"", ".", ".."} for part in item["file"].split("/"))
            ):
                raise ValueError("Source paths must be project-relative POSIX paths")
            unit = Unit(**item)
            if len(unit.id) > 256 or len(unit.language) > 64:
                raise ValueError("Unit/language identity exceeds its budget")
            if unit.id in units or unit.project not in projects or "|" in unit.language:
                raise ValueError("Duplicate unit or unknown project/language")
            project = projects[unit.project]
            location = (
                project.repository,
                project.revision,
                unit.file,
                unit.symbol,
                unit.line,
                unit.end_line,
            )
            if location in references_seen:
                raise ValueError("Aliased canonical source function reference")
            references_seen.add(location)
            snapshot = (unit.source_hash, unit.language)
            if (
                snapshots.setdefault((project.repository, project.revision, unit.file), snapshot)
                != snapshot
            ):
                raise ValueError("Conflicting file snapshot or language labels")
            units[unit.id] = unit
            isolate("source", _hash(unit.source_hash), projects[unit.project].split)
        seen, references, relationships = set(), set(), set()
        for value in data["pairs"]:
            item = _record(
                value, "id left right relation positive family rationale domain reviewed_by"
            )
            for key in set(item) - {"positive"}:
                _text(item[key])
            pair = Pair(**item)
            if type(pair.positive) is not bool or pair.relation not in RELATIONS:
                raise ValueError("Labels require a boolean and an explicit relation")
            if (
                pair.id in seen
                or pair.left == pair.right
                or any(unit not in units for unit in (pair.left, pair.right))
            ):
                raise ValueError("Duplicate pair, self-pair or unknown endpoint")
            key = (pair.relation, *sorted((pair.left, pair.right)))
            if key in relationships:
                raise ValueError("Duplicate or conflicting pair relationship")
            splits = {projects[units[uid].project].split for uid in (pair.left, pair.right)}
            if len(splits) != 1:
                raise ValueError("Cross-split endpoint leakage")
            isolate("family", pair.family, next(iter(splits)))
            seen.add(pair.id)
            relationships.add(key)
            references.update((pair.left, pair.right))
            pairs.append(pair)
        if references != set(units):
            raise ValueError("Unlabelled corpus units are not an evaluation universe")
        if {projects[units[pair.left].project].split for pair in pairs} != {"calibration", "test"}:
            raise ValueError("Both calibration and held-out test pairs are required")
        result = cls(MappingProxyType(projects), MappingProxyType(units), tuple(pairs), "")
        return cls(result.projects, result.units, result.pairs, digest(result.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "projects": [asdict(item) for _, item in sorted(self.projects.items())],
            "units": [asdict(item) for _, item in sorted(self.units.items())],
            "pairs": [asdict(item) for item in sorted(self.pairs, key=lambda pair: pair.id)],
        }

    def split(self, pair: Pair) -> str:
        return self.projects[self.units[pair.left].project].split

    def facet(self, pair: Pair) -> str:
        return "|".join(
            (
                pair.relation,
                *sorted((self.units[pair.left].language, self.units[pair.right].language)),
            )
        )


@dataclass(frozen=True)
class Observation:
    pair_id: str
    score: float | None
    error: str | None = None

    def __post_init__(self) -> None:
        _text(self.pair_id)
        if self.score is None:
            _text(self.error)
        elif (
            type(self.score) not in {int, float}
            or not math.isfinite(self.score)
            or not -1 <= self.score <= 1
            or self.error is not None
        ):
            raise ValueError("Scores must be finite unit-range numbers or explicit abstentions")


@dataclass(frozen=True)
class Policy:
    channel: str
    profile: str
    corpus: str
    calibration_scores: str
    minimum_precision: float
    minimum_support: int
    thresholds: Mapping[str, float]
    rejected: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "profile": self.profile,
            "corpus": self.corpus,
            "calibration_scores": self.calibration_scores,
            "minimum_precision": self.minimum_precision,
            "minimum_support": self.minimum_support,
            "thresholds": dict(self.thresholds),
            "rejected": dict(self.rejected),
        }


def _observations(corpus: Corpus, observations: Sequence[Observation]) -> dict[str, Observation]:
    by_id = {item.pair_id: item for item in observations}
    if len(by_id) != len(observations) or set(by_id) != {pair.id for pair in corpus.pairs}:
        raise ValueError("Observations must cover exactly the labelled pairs, once each")
    return by_id


def _calibration_digest(corpus: Corpus, scores: dict[str, Observation]) -> str:
    return digest(
        [
            asdict(scores[pair.id])
            for pair in sorted(corpus.pairs, key=lambda p: p.id)
            if corpus.split(pair) == "calibration"
        ]
    )


def calibrate(
    corpus: Corpus,
    observations: Sequence[Observation],
    *,
    channel: str,
    profile: str,
    minimum_precision: float = 0.95,
    minimum_support: int = 2,
) -> Policy:
    """Fit per-relation/language thresholds using calibration projects only."""
    if (
        channel not in CHANNELS
        or not math.isfinite(minimum_precision)
        or not 0 < minimum_precision <= 1
    ):
        raise ValueError("Invalid calibration channel/precision")
    if type(minimum_support) is not int or not 1 <= minimum_support <= MAX_PAIRS:
        raise ValueError("Invalid calibration support")
    _hash(profile)
    scores = _observations(corpus, observations)
    groups: dict[str, list[tuple[Pair, float]]] = defaultdict(list)
    for pair in corpus.pairs:
        score = scores[pair.id].score
        if channel != "body" and score is not None and score < 0:
            raise ValueError("Syntax and portable scores cannot be negative")
        if corpus.split(pair) == "calibration" and score is not None:
            groups[corpus.facet(pair)].append((pair, score))
    thresholds, rejected = {}, {}
    for facet in sorted({corpus.facet(pair) for pair in corpus.pairs}):
        rows = groups[facet]
        positives = sum(pair.positive for pair, _ in rows)
        negatives = len(rows) - positives
        if min(positives, negatives) < minimum_support:
            rejected[facet] = "insufficient_calibration_support"
            continue
        # Highest recall, then the more conservative threshold. Predicting
        # nothing is never a feasible perfect-precision operating point.
        best: tuple[int, float] | None = None
        tp, selected = 0, 0
        for threshold, bucket in groupby(
            sorted(rows, key=lambda row: -row[1]), key=lambda row: row[1]
        ):
            for pair, _ in bucket:
                selected += 1
                tp += pair.positive
            if tp >= minimum_support and tp / selected >= minimum_precision:
                candidate = (tp, threshold)
                best = max(best, candidate) if best is not None else candidate
        if best is None:
            rejected[facet] = "no_threshold_meets_precision"
        else:
            thresholds[facet] = best[1]
    return Policy(
        channel,
        profile,
        corpus.identity,
        _calibration_digest(corpus, scores),
        minimum_precision,
        minimum_support,
        MappingProxyType(thresholds),
        MappingProxyType(rejected),
    )


def _metrics(rows: list[tuple[Pair, Observation, float | None]], k: int) -> dict[str, Any]:
    scored = [(pair, obs.score) for pair, obs, _ in rows if obs.score is not None]
    decided = [
        (pair, obs.score >= threshold)
        for pair, obs, threshold in rows
        if obs.score is not None and threshold is not None
    ]
    tp = sum(pair.positive and predicted for pair, predicted in decided)
    fp = sum(not pair.positive and predicted for pair, predicted in decided)
    fn = sum(pair.positive and not predicted for pair, predicted in decided)
    tn = len(decided) - tp - fp - fn
    positives = sum(pair.positive for pair, _, _ in rows)
    ranked = sorted(scored, key=lambda row: -float(row[1]))
    count = min(k, len(ranked))
    precision_at_k = None
    if count:
        cutoff = ranked[count - 1][1]
        above = [pair for pair, score in ranked if score > cutoff]
        tied = [pair for pair, score in ranked if score == cutoff]
        precision_at_k = (
            sum(p.positive for p in above)
            + (count - len(above)) * sum(p.positive for p in tied) / len(tied)
        ) / count
    lower = None
    if tp + fp:
        n, z = tp + fp, 1.959963984540054
        p = tp / n
        lower = (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (
            1 + z * z / n
        )
    return {
        "labelled_pairs": len(rows),
        "positive_pairs": positives,
        "scored_pairs": len(scored),
        "abstained_pairs": len(rows) - len(scored),
        "decided_pairs": len(decided),
        "unqualified_pairs": len(scored) - len(decided),
        "coverage": len(scored) / len(rows) if rows else None,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "precision_wilson_lower_95": lower,
        "recall": tp / (tp + fn) if tp + fn else None,
        "end_to_end_recall": tp / positives if positives else None,
        "labelled_pair_precision_at_k": precision_at_k,
        "ranked_pairs_at_k": count,
    }


def evaluate(
    corpus: Corpus,
    observations: Sequence[Observation],
    policy: Policy,
    *,
    profile: str,
    k: int = 10,
) -> dict[str, Any]:
    """Apply the frozen calibration policy; report test decisions and abstentions."""
    if policy.corpus != corpus.identity or profile != policy.profile:
        raise ValueError("Policy corpus/profile mismatch")
    if type(k) is not int or not 1 <= k <= MAX_PAIRS:
        raise ValueError("Invalid review rank budget")
    scores = _observations(corpus, observations)
    if _calibration_digest(corpus, scores) != policy.calibration_scores:
        raise ValueError("Calibration observations changed after policy selection")
    expected = calibrate(
        corpus,
        observations,
        channel=policy.channel,
        profile=profile,
        minimum_precision=policy.minimum_precision,
        minimum_support=policy.minimum_support,
    )
    if expected != policy:
        raise ValueError("Policy thresholds do not match their calibration evidence")
    result: dict[str, Any] = {
        "schema": "victor-similarity-evaluation-v1",
        "corpus": corpus.identity,
        "policy": policy.to_dict(),
        "evaluation_scope": "labelled_pairs_only",
        "production_quality_qualified": False,
        "coverage_complete": all(obs.score is not None for obs in observations),
        "decision_complete": not policy.rejected
        and all(obs.score is not None for obs in observations),
        "rank_ties": "expected_uniform_at_cutoff",
        "rank_k": k,
        "origins": sorted({project.origin for project in corpus.projects.values()}),
        "abstentions": [asdict(obs) for obs in observations if obs.score is None],
    }
    for split in ("calibration", "test"):
        rows = [
            (pair, scores[pair.id], policy.thresholds.get(corpus.facet(pair)))
            for pair in corpus.pairs
            if corpus.split(pair) == split
        ]
        groups: dict[str, list[tuple[Pair, Observation, float | None]]] = defaultdict(list)
        for row in rows:
            groups[corpus.facet(row[0])].append(row)
        overall = _metrics(rows, k)
        # Scores from different relation/language facets are not a common rank.
        overall["labelled_pair_precision_at_k"] = None
        overall["ranked_pairs_at_k"] = None
        result[split] = {
            "overall": overall,
            "facets": {facet: _metrics(group, k) for facet, group in sorted(groups.items())},
        }
    return result

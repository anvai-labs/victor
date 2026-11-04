"""Independent full-cohort checks for bounded body-cache qualification.

No production retrieval code is used by the dense reference. This validates
captured inputs and public components/counts/extrema, not model accuracy or raw
matching-edge identities (which the production report does not expose).
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import re
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.body_qualification_io import cache_database, source_parts, MAX_CACHE_ROWS
else:
    _spec = importlib.util.spec_from_file_location(
        "body_qualification_metrics_io", Path(__file__).with_name("body_qualification_io.py")
    )
    if _spec is None or _spec.loader is None:
        raise RuntimeError("Qualification I/O owner is unavailable")
    _io = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_io)
    cache_database, source_parts, MAX_CACHE_ROWS = (
        _io.cache_database,
        _io.source_parts,
        _io.MAX_CACHE_ROWS,
    )

TOLERANCE = 1e-6
GLOBAL_OPTIONS = {
    "threshold": 0.9,
    "include_same_file": False,
    "top_k": None,
    "comparison_scope": "all",
}


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_observation(observation: dict[str, Any], *, allow_incomplete: bool = False) -> None:
    """Bind both backends to the independently captured, full function cohort."""
    try:
        _validate_observation(observation, allow_incomplete=allow_incomplete)
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        raise ValueError("Malformed qualification cohort or metrics") from exc


def _validate_observation(observation: dict[str, Any], *, allow_incomplete: bool) -> None:
    units = observation.get("unit_manifest")
    if not isinstance(units, list) or not 2 <= len(units) <= 300:
        raise ValueError("Invalid or missing independent function cohort")
    n = len(units)
    if observation.get("unit_manifest_sha256") != _digest(units):
        raise ValueError("Independent function cohort digest changed")
    sources = observation.get("source_manifest")
    if not isinstance(sources, dict) or not 1 <= len(sources) <= 100:
        raise ValueError("Invalid selected-source manifest")
    for name, value in sources.items():
        if (
            not isinstance(name, str)
            or not isinstance(value, str)
            or not re.fullmatch(r"[0-9a-f]{64}", value)
        ):
            raise ValueError("Invalid selected-source manifest")
        source_parts(name)
    hashes = [item["input_hash"] for item in units]
    if (
        observation.get("unit_input_hashes") != hashes
        or type(observation.get("functions")) is not int
        or observation["functions"] != n
    ):
        raise ValueError("Requested function cohort changed")
    if observation["stats"]["functions"] != n or observation["stats"]["unique_inputs"] != len(
        set(hashes)
    ):
        raise ValueError("Requested function cohort statistics changed")
    if type(observation.get("coverage_complete")) is not bool:
        raise ValueError("Missing selected-source coverage status")
    if not allow_incomplete and (
        not observation.get("coverage_complete") or observation.get("errors")
    ):
        raise ValueError("Incomplete selected-source coverage cannot be qualified")
    global_result = observation.get("global_result")
    if observation.get("global_retrieval_benchmark") is not True or not isinstance(
        global_result, dict
    ):
        raise ValueError("Global retrieval observation is unavailable")
    for stats in (observation["stats"], global_result["stats"]):
        if any(
            type(stats.get(key)) is not int or stats[key] < 0
            for key in (
                "functions",
                "unique_inputs",
                "cache_hits",
                "model_load_attempts",
                "inferred_windows",
            )
        ):
            raise ValueError("Invalid qualification counters")
    if (
        global_result["profile"] != observation["profile"]
        or global_result["source_manifest"] != observation["source_manifest"]
        or global_result["coverage_complete"] != observation["coverage_complete"]
        or global_result["errors"] != observation["errors"]
    ):
        raise ValueError("Global producer/profile or source coverage changed")
    profile = observation["profile"]
    if (
        observation.get("global_options") != GLOBAL_OPTIONS
        or profile.get("comparison_scope") != "all"
        or global_result["stats"].get("neighbors") is not None
        or "neighbors" not in global_result["stats"]
        or observation["stats"].get("retrieval") != "requested_pair_cosine"
    ):
        raise ValueError("Global/requested retrieval configuration changed")
    families = global_result.get("families")
    if (
        not isinstance(families, list)
        or len(families) > n // 2
        or any(not isinstance(family, dict) for family in families)
    ):
        raise ValueError("Invalid global retrieval families")
    for family in families:
        members = family.get("members")
        if (
            not isinstance(members, list)
            or not 2 <= len(members) <= n
            or any(
                not isinstance(member, dict) or not isinstance(member.get("id"), str)
                for member in members
            )
        ):
            raise ValueError("Invalid global retrieval member cohort")
    dimension = profile.get("dimension")
    if type(dimension) is not int or not 1 <= dimension <= 4096:
        raise ValueError("Invalid vector profile dimension")
    expected = []
    for unit in units:
        if (
            not isinstance(unit, dict)
            or not isinstance(unit.get("file"), str)
            or not isinstance(unit.get("symbol"), str)
            or not unit["symbol"]
            or type(unit.get("line")) is not int
            or unit["line"] < 1
            or any(
                type(unit.get(key)) is not int or unit[key] < 0
                for key in ("region_index", "source_start")
            )
            or not isinstance(unit.get("input_hash"), str)
        ):
            raise ValueError("Invalid source-bound function cohort")
        source_parts(unit["file"])
        if unit["file"] not in observation["source_manifest"] or not re.fullmatch(
            r"[0-9a-f]{64}", unit["input_hash"]
        ):
            raise ValueError("Invalid source-bound function cohort")
        identity = hashlib.sha256(
            json.dumps(
                [
                    unit["file"],
                    unit["line"],
                    unit["region_index"],
                    unit["source_start"],
                    unit["input_hash"],
                ]
            ).encode()
        ).hexdigest()
        expected.append((identity, unit["file"], unit["symbol"], unit["line"], unit["input_hash"]))

    def cohort(inputs: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
        if (
            not isinstance(inputs, list)
            or len(inputs) != n
            or any(not isinstance(item, dict) for item in inputs)
        ):
            raise ValueError("Global/requested function cohort was truncated or duplicated")
        return [
            (item["id"], item["file"], item["symbol"], item["line"], item["input_hash"])
            for item in inputs
        ]

    if (
        cohort(observation.get("requested_inputs", [])) != expected
        or cohort(global_result["inputs"]) != expected
        or len({item[0] for item in expected}) != n
        or len(
            {
                (item["file"], item["line"], item["region_index"], item["source_start"])
                for item in units
            }
        )
        != n
        or global_result["stats"]["functions"] != n
        or global_result["stats"]["unique_inputs"] != len(set(hashes))
    ):
        raise ValueError("Global/requested function cohort was truncated or duplicated")
    if any(
        global_result["stats"].get(key) != 0 for key in ("model_load_attempts", "inferred_windows")
    ) or global_result["stats"].get("cache_hits") != len(set(hashes)):
        raise ValueError("Global retrieval did not use a fresh zero-load cached encoder")
    if global_result["stats"].get("retrieval") != "tiled_exact_cosine":
        raise ValueError("Invalid global retrieval scope")
    rows = observation.get("cached_vectors")
    if not isinstance(rows, list) or len(rows) > MAX_CACHE_ROWS:
        raise ValueError("Missing or over-budget vector manifest")
    seen = set()
    for row in rows:
        if (
            not isinstance(row, list)
            or len(row) != 3
            or row[0] in seen
            or not all(isinstance(item, str) for item in row)
            or not re.fullmatch(r"[0-9a-f]{64}", row[0])
            or not re.fullmatch(r"[0-9a-f]{64}", row[1])
            or len(row[2]) > 16384
        ):
            raise ValueError("Invalid vector manifest row")
        seen.add(row[0])
    if observation.get("vector_manifest_sha256") != _digest(rows):
        raise ValueError("Vector manifest digest changed")
    if not set(hashes) <= seen:
        raise ValueError("Active vector coverage is incomplete")
    pairs = observation.get("pair_scores")
    universe = list(itertools.combinations(range(n), 2))
    if not isinstance(pairs, list) or len(pairs) != len(universe):
        raise ValueError("Requested pair coverage is incomplete")
    actual_pairs = []
    for row in pairs:
        left, right, score = row.get("left"), row.get("right"), row.get("score")
        if (
            type(left) is not int
            or type(right) is not int
            or not 0 <= left < right < n
            or type(score) not in (int, float)
            or not math.isfinite(score)
            or not -1 <= score <= 1
        ):
            raise ValueError("Invalid requested pair index or score")
        actual_pairs.append((left, right))
    if sorted(actual_pairs) != universe:
        raise ValueError("Requested pair coverage is duplicated or incomplete")
    if type(observation.get("pair_count")) is not int or observation["pair_count"] != len(universe):
        raise ValueError("Requested pair count disagrees with function cohort")
    if (
        observation.get("pair_scores_sha256") != _digest(pairs)
        or observation.get("pair_abstentions") != 0
    ):
        raise ValueError("Requested score digest or abstention count changed")


def _scores(observation: dict[str, Any]) -> dict[tuple[str, str], float]:
    hashes = observation["unit_input_hashes"]
    answer: dict[tuple[str, str], float] = {}
    for pair in observation["pair_scores"]:
        left, right = sorted((hashes[pair["left"]], hashes[pair["right"]]))
        key = (left, right)
        value = float(pair["score"])
        if key in answer and answer[key] != value:
            raise ValueError("Inconsistent requested score for identical model inputs")
        answer[key] = value
    return answer


def compare_unchanged(
    before: dict[str, Any], after: dict[str, Any], *, allow_incomplete: bool = False
) -> dict[str, Any]:
    validate_observation(before, allow_incomplete=allow_incomplete)
    validate_observation(after, allow_incomplete=allow_incomplete)
    if before["profile"] != after["profile"]:
        raise ValueError("Body producer or input profile changed during qualification")
    for name, digest in before["source_manifest"].items():
        if after["source_manifest"].get(name) == digest:
            left_units = [item for item in before["unit_manifest"] if item["file"] == name]
            right_units = [item for item in after["unit_manifest"] if item["file"] == name]
            if left_units != right_units:
                raise ValueError("Unchanged source function cohort or locations drifted")
    shared = set(before["unit_input_hashes"]) & set(after["unit_input_hashes"])
    old = {row[0]: row[1:] for row in before["cached_vectors"]}
    new = {row[0]: row[1:] for row in after["cached_vectors"]}
    if any(old[key] != new[key] for key in shared):
        raise ValueError("Unchanged input vector bytes or spans drifted")
    old_scores, new_scores = _scores(before), _scores(after)
    # Per-function universes were validated above. Compare every hash pair that
    # exists in both cohorts; a duplicate's self-pair can disappear on deletion.
    expected = old_scores.keys() & new_scores.keys()
    for key in expected:
        if old_scores[key] != new_scores[key]:
            raise ValueError("Unchanged input requested score drifted")
    return {
        "shared_inputs": len(shared),
        "shared_input_pairs": len(expected),
        "selected_source_complete": before["coverage_complete"] and after["coverage_complete"],
        "vector_bytes_and_spans_identical": True,
        "requested_scores_identical": True,
    }


def _cache_records(
    cache: Path, profile: str, dimension: int
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    import numpy as np

    if type(dimension) is not int or not 1 <= dimension <= 4096:
        raise ValueError("Invalid vector profile dimension")
    with cache_database(cache) as db:
        rows = db.execute(
            "SELECT input_hash,vector,spans,checksum FROM body_vectors_v1 WHERE profile=?",
            (profile,),
        ).fetchmany(MAX_CACHE_ROWS + 1)
    if len(rows) > MAX_CACHE_ROWS:
        raise ValueError("Vector cache exceeds its row budget")
    result = {}
    manifest = {}
    for digest, raw, spans, checksum in rows:
        if (
            not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or digest in result
            or not isinstance(raw, bytes)
            or len(raw) != dimension * 4
            or not isinstance(spans, str)
            or len(spans) > 16384
            or not isinstance(checksum, str)
        ):
            raise ValueError("Invalid vector cache row dimension or schema")
        if hashlib.sha256(raw + spans.encode()).hexdigest() != checksum:
            raise ValueError("Cached vector checksum failed")
        try:
            intervals = json.loads(spans)
        except (ValueError, RecursionError) as exc:
            raise ValueError("Invalid cached spans") from exc
        if (
            not isinstance(intervals, list)
            or not 1 <= len(intervals) <= 128
            or any(
                not isinstance(item, list)
                or len(item) != 2
                or any(type(index) is not int for index in item)
                or not 0 <= item[0] < item[1] <= 65536
                for item in intervals
            )
        ):
            raise ValueError("Invalid cached spans")
        vector = np.frombuffer(raw, dtype="<f4").copy()
        with np.errstate(over="ignore", invalid="ignore"):
            norm = np.linalg.norm(vector)
        if not np.isfinite(vector).all() or not np.isfinite(norm) or norm <= 0:
            raise ValueError("Invalid cached vector norm")
        result[digest] = vector
        manifest[digest] = [hashlib.sha256(raw).hexdigest(), spans]
    return result, manifest


def cache_vectors(cache: Path, profile: str, dimension: int) -> dict[str, Any]:
    return _cache_records(cache, profile, dimension)[0]


def global_oracle(observation: dict[str, Any], cache: Path) -> dict[str, Any]:
    """Check public component/count/extrema outputs; no raw edge-list claim."""
    import numpy as np

    validate_observation(observation)
    result = observation["global_result"]
    inputs = result["inputs"]
    vectors, actual_manifest = _cache_records(
        cache, result["profile"]["producer"]["identity"], result["profile"]["dimension"]
    )
    manifest = {row[0]: row[1:] for row in observation["cached_vectors"]}
    matrix = np.array([vectors[item["input_hash"]] for item in inputs], dtype=np.float64)
    for item in inputs:
        digest = item["input_hash"]
        if manifest[digest] != actual_manifest[digest]:
            raise ValueError("Active vector manifest differs from actual cache")
    matrix /= np.linalg.norm(matrix, axis=1)[:, None]
    scores = np.clip(matrix @ matrix.T, -1, 1)
    maximum_requested_delta = max(
        abs(float(row["score"]) - float(scores[row["left"], row["right"]]))
        for row in observation["pair_scores"]
    )
    if maximum_requested_delta > TOLERANCE:
        raise ValueError("Requested scores disagree with independent dense cosine oracle")
    neighbors: dict[int, set[int]] = defaultdict(set)
    edges = []
    for i, j in itertools.combinations(range(len(inputs)), 2):
        if inputs[i]["file"] != inputs[j]["file"] and scores[i, j] >= 0.9:
            neighbors[i].add(j)
            neighbors[j].add(i)
            edges.append((i, j))
    remaining = set(neighbors)
    expected = {}
    while remaining:
        stack, group = [min(remaining)], set()
        while stack:
            item = stack.pop()
            if item not in group:
                group.add(item)
                stack.extend(neighbors[item] - group)
        remaining -= group
        key = tuple(sorted(inputs[index]["id"] for index in group))
        values = [float(scores[i, j]) for i, j in edges if i in group and j in group]
        expected[key] = (len(values), max(values), min(values))
    actual = {}
    indexed = {item["id"]: item for item in inputs}
    for family in result["families"]:
        ids = tuple(sorted(member["id"] for member in family["members"]))
        if ids in actual or len(ids) != len(set(ids)):
            raise ValueError("Duplicated global retrieval family or member")
        if any(member != indexed.get(member["id"]) for member in family["members"]):
            raise ValueError("Global retrieval retained stale member metadata")
        count, maximum, minimum = (
            family.get("matching_edges"),
            family.get("score"),
            family.get("min_edge_similarity"),
        )
        if (
            type(count) is not int
            or not 1 <= count <= len(inputs) * (len(inputs) - 1) // 2
            or any(
                type(value) not in (int, float) or not math.isfinite(value) or not -1 <= value <= 1
                for value in (maximum, minimum)
            )
            or minimum > maximum
        ):
            raise ValueError("Invalid global retrieval edge count or extrema")
        actual[ids] = (count, maximum, minimum)
    if actual.keys() != expected.keys() or any(
        actual[key][0] != expected[key][0]
        or max(abs(actual[key][i] - expected[key][i]) for i in (1, 2)) > 1e-6
        for key in expected
    ):
        raise ValueError("Tiled global retrieval disagrees with dense graph oracle")
    margins = [
        abs(float(scores[i, j]) - 0.9)
        for i, j in itertools.combinations(range(len(inputs)), 2)
        if inputs[i]["file"] != inputs[j]["file"]
    ]
    return {
        "functions": len(inputs),
        "families": len(expected),
        "matching_edges": len(edges),
        "oracle": "float64_dense_cosine_independent_threshold_components",
        "threshold": 0.9,
        "include_same_file": False,
        "minimum_threshold_margin": min(margins) if margins else None,
        "component_and_edge_count_sha256": _digest(
            [(key, value[0]) for key, value in sorted(expected.items())]
        ),
        "raw_matching_edge_identity_verified": False,
        "maximum_requested_dense_score_delta": maximum_requested_delta,
        "absolute_tolerance": TOLERANCE,
    }


def compare_cold(
    warm: dict[str, Any], cold: dict[str, Any], warm_cache: Path, cold_cache: Path
) -> dict[str, Any]:
    """Cold rebuilds may differ by roundoff; cache reuse itself stays bit-exact."""
    import numpy as np

    validate_observation(warm)
    validate_observation(cold)
    if cold["state"] != "cold" or cold["stats"]["cache_hits"] != 0:
        raise ValueError("Cold reference unexpectedly reused cached inputs")
    if warm["profile"] != cold["profile"] or warm["source_manifest"] != cold["source_manifest"]:
        raise ValueError("Cold rebuild does not describe the same source and producer")
    if warm["unit_manifest"] != cold["unit_manifest"]:
        raise ValueError("Cold rebuild function cohort or locations changed")
    global_oracle(warm, warm_cache)
    global_oracle(cold, cold_cache)
    left, right = _scores(warm), _scores(cold)
    if left.keys() != right.keys():
        raise ValueError("Cold rebuild requested pair coverage changed")
    score_delta = max((abs(left[key] - right[key]) for key in left), default=0.0)
    profile = warm["profile"]["producer"]["identity"]
    a, b = cache_vectors(warm_cache, profile, warm["profile"]["dimension"]), cache_vectors(
        cold_cache, profile, cold["profile"]["dimension"]
    )
    active = set(warm["unit_input_hashes"])
    vector_delta = max(float(np.max(np.abs(a[key] - b[key]))) for key in active)
    if score_delta > 1e-6 or vector_delta > 1e-6:
        raise ValueError("Cold rebuild exceeds declared numerical parity tolerance")

    def groups(value: dict[str, Any]) -> list[Any]:
        return sorted(
            (tuple(sorted(member["id"] for member in family["members"])), family["matching_edges"])
            for family in value["global_result"]["families"]
        )

    if groups(warm) != groups(cold):
        raise ValueError("Cold rebuild global families or matching edges changed")
    return {
        "maximum_score_delta": score_delta,
        "maximum_vector_delta": vector_delta,
        "absolute_tolerance": 1e-6,
        "global_families_and_edges_identical": True,
    }

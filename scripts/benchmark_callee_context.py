"""A bounded requested-pair ablation of syntax and resolved Python call targets.

Authored identities and actual SCIP output are separate observations. No project
source is executed, indexer started, model loaded, threshold trained or ranking
fusion installed. An index requires a byte/hash-bound source-generation record.
"""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import hashlib
import inspect
from itertools import groupby
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import re
import secrets
import stat
import sys
import time
from typing import Any, cast

from victor.core.graph_rag.source_reader import project_source_reader, read_project_source
from victor.core.graph_rag.structural_clones import (
    FunctionUnit,
    OccurrenceMap,
    extract_units,
    fresh_occurrence_map,
)

MAX_FILE = 64 * 1024
MAX_SOURCE_BYTES = 128 * 1024
MAX_INDEX = 2 * 1024 * 1024
MAX_REPORT = 2 * 1024 * 1024
ENGINE_FILES = (
    "structural_clones.py",
    "ast_fingerprints.py",
    "clone_bindings.py",
    "source_reader.py",
    "copy_artifacts.py",
    "_scip/scip_pb2.py",
)


def _unique(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise ValueError("Duplicate manifest field")
        result[key] = value
    return result


def _read(path: Path, limit: int) -> bytes:
    data, info = read_project_source(path.absolute().parent, path.name, limit)
    if info.st_size > limit or len(data) > limit:
        raise ValueError("Ablation input exceeds its byte budget")
    return cast(bytes, data)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(_read(path, MAX_FILE), object_pairs_hook=_unique)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def _text(value: Any) -> bool:
    return isinstance(value, str) and 0 < len(value) <= 4096


def _hash(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _fixture(raw: dict[str, Any]) -> dict[str, Any]:
    if (
        set(raw) != {"schema", "oracle_domain", "source_hashes", "units", "pairs"}
        or raw["schema"] != "victor-callee-context-controls-v1"
    ):
        raise ValueError("Unknown callee-control manifest")
    if not _text(raw["oracle_domain"]):
        raise ValueError("Missing restricted oracle domain")
    hashes = raw["source_hashes"]
    if not isinstance(hashes, dict) or not 1 <= len(hashes) <= 32:
        raise ValueError("Invalid source file budget")
    for name, digest in hashes.items():
        # One flat authored Python project, without namespace/path aliases.
        if not isinstance(name, str) or re.fullmatch(r"control_[a-z_]+\.py", name) is None:
            raise ValueError("Unsafe or unsupported source name")
        if not _hash(digest):
            raise ValueError("Invalid source hash")
    if not isinstance(raw["units"], list) or not 1 <= len(raw["units"]) <= 64:
        raise ValueError("Invalid unit budget")
    units: dict[str, dict[str, Any]] = {}
    locations = set()
    for unit in raw["units"]:
        if not isinstance(unit, dict) or set(unit) != {
            "id",
            "file",
            "symbol",
            "line",
            "end_line",
            "controlled_targets",
        }:
            raise ValueError("Invalid unit fields")
        if (
            not all(_text(unit[key]) for key in ("id", "file", "symbol"))
            or unit["id"] in units
            or unit["file"] not in hashes
            or type(unit["line"]) is not int
            or type(unit["end_line"]) is not int
            or not 1 <= unit["line"] <= unit["end_line"] <= MAX_FILE
            or not isinstance(unit["controlled_targets"], list)
            or len(unit["controlled_targets"]) > 1024
            or any(
                target is not None and not _text(target) for target in unit["controlled_targets"]
            )
        ):
            raise ValueError("Invalid or duplicate unit")
        location = tuple(unit[key] for key in ("file", "symbol", "line", "end_line"))
        if location in locations:
            raise ValueError("Aliased source unit")
        locations.add(location)
        units[unit["id"]] = unit
    if not isinstance(raw["pairs"], list) or not 1 <= len(raw["pairs"]) <= 256:
        raise ValueError("Invalid pair budget")
    ids: set[str] = set()
    endpoints: set[tuple[str, ...]] = set()
    referenced: set[str] = set()
    for pair in raw["pairs"]:
        if not isinstance(pair, dict) or set(pair) != {
            "id",
            "left",
            "right",
            "positive",
            "rationale",
        }:
            raise ValueError("Invalid pair fields")
        if (
            not all(_text(pair[key]) for key in ("id", "left", "right", "rationale"))
            or pair["id"] in ids
            or pair["left"] not in units
            or pair["right"] not in units
            or pair["left"] == pair["right"]
            or (pair["positive"] is not None and type(pair["positive"]) is not bool)
        ):
            raise ValueError("Invalid pair label or endpoint")
        key = tuple(sorted((pair["left"], pair["right"])))
        if key in endpoints:
            raise ValueError("Duplicate or conflicting pair")
        ids.add(pair["id"])
        endpoints.add(key)
        referenced.update(key)
    if referenced != units.keys():
        raise ValueError("Unreferenced control units")
    return raw


def load_fixture(path: Path) -> dict[str, Any]:
    """Validate a small frozen control universe before any extraction/scoring."""
    return _fixture(_json(path))


def _calls(tree: ast.Module, unit: dict[str, Any]) -> list[tuple[tuple[int, int], int] | None]:
    selected = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and (node.name, node.lineno, node.end_lineno)
        == (unit["symbol"], unit["line"], unit["end_line"])
    ]
    if len(selected) != 1:
        raise ValueError("Unit span does not identify one function")
    calls: list[tuple[tuple[int, int], int] | None] = []
    for statement in selected[0].body:
        for node in ast.walk(statement):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            if isinstance(function, ast.Name):
                calls.append(((function.lineno, function.col_offset), len(function.id.encode())))
            elif isinstance(function, ast.Attribute) and function.end_col_offset is not None:
                size = len(function.attr.encode())
                calls.append(
                    ((function.end_lineno or function.lineno, function.end_col_offset - size), size)
                )
            else:
                calls.append(None)
    if len(calls) > 1024 or len(calls) != len(unit["controlled_targets"]):
        raise ValueError("Call-site/control-target count changed")
    return calls


class _Rows(list[tuple[int, int, int, str]]):
    def __init__(self, encoding: int, text: str) -> None:
        super().__init__()
        self.position_encoding = encoding
        self.source_text = text


def _index_maps(
    index_path: Path, binding_path: Path, sources: dict[str, str], source_hashes: dict[str, str]
) -> tuple[dict[str, OccurrenceMap], dict[str, Any]]:
    data = _read(index_path, MAX_INDEX)
    binding = _json(binding_path)
    if (
        set(binding) != {"schema", "index_sha256", "source_hashes", "producer"}
        or binding["schema"] != "victor-callee-index-binding-v1"
    ):
        raise ValueError("Unknown index/source generation binding")
    if binding["index_sha256"] != hashlib.sha256(data).hexdigest():
        raise ValueError("Index hash does not match generation binding")
    if not isinstance(binding["producer"], dict) or not binding["producer"].get("name"):
        raise ValueError("Missing index producer declaration")
    descriptor = {
        "index_sha256": binding["index_sha256"],
        "bytes": len(data),
        "declared_producer": binding["producer"],
        "source_freshness": "source_hash_sidecar_matches; curator provenance, not authentication",
    }
    if binding["source_hashes"] != source_hashes:
        return {}, {
            **descriptor,
            "status": "stale_source_binding",
            "source_freshness": "source_hash_sidecar_mismatch",
        }
    try:
        from victor.core.graph_rag._scip import scip_pb2
    except ImportError:
        return {}, {**descriptor, "status": "protobuf_runtime_unavailable"}
    except RuntimeError:
        return {}, {**descriptor, "status": "protobuf_runtime_incompatible"}
    index = scip_pb2.Index()
    try:
        index.ParseFromString(data)
    except Exception as exc:
        raise ValueError("Malformed bounded SCIP index") from exc
    if any(
        binding["producer"].get(key) != getattr(index.metadata.tool_info, key)
        for key in ("name", "version")
    ):
        raise ValueError("Declared index producer differs from SCIP metadata")
    if len(index.documents) > 32 or sum(len(doc.occurrences) for doc in index.documents) > 20_000:
        raise ValueError("Index exceeds document/occurrence budget")
    maps: dict[str, OccurrenceMap] = {}
    seen: set[str] = set()
    for doc in index.documents:
        if doc.relative_path in seen:
            raise ValueError("Duplicate SCIP document path")
        seen.add(doc.relative_path)
        if doc.relative_path not in sources:
            raise ValueError("Index contains a document outside the frozen source universe")
        rows = _Rows(doc.position_encoding, doc.text)
        symbols: dict[tuple[int, int, int], set[str]] = defaultdict(set)
        for occurrence in doc.occurrences:
            values = list(occurrence.range)
            if len(values) == 4 and values[0] == values[2]:
                values = [values[0], values[1], values[3]]
            if len(values) != 3:
                continue
            line, column, end = values
            rows.append((line, column, end, occurrence.symbol))
            symbols[(line, column, end)].add(occurrence.symbol)
        lines = sources[doc.relative_path].splitlines()
        mapping = fresh_occurrence_map(lines, rows)
        if mapping is None:
            continue
        # Never choose an identity by occurrence ordering for an ambiguous span.
        for key, targets in symbols.items():
            if len(targets) <= 1:
                continue
            single = _Rows(rows.position_encoding, rows.source_text)
            single.append((*key, next(iter(targets))))
            for translated in fresh_occurrence_map(lines, single) or {}:
                mapping.pop(translated, None)
        maps[doc.relative_path] = mapping
    return maps, {
        **descriptor,
        "status": "available",
        "metadata_producer": {
            "name": index.metadata.tool_info.name,
            "version": index.metadata.tool_info.version,
        },
        "document_count": len(index.documents),
        "occurrence_count": sum(len(doc.occurrences) for doc in index.documents),
    }


def _resolved(symbol: str) -> bool:
    # This ablation covers qualified Python function/method targets, not local
    # placeholders, callable values or guessed constructor/runtime dispatch.
    parts = symbol.split(" ", 4)
    return (
        0 < len(symbol) <= 4096
        and len(parts) == 5
        and parts[:2] == ["scip-python", "python"]
        and all(parts[2:])
        and "/" in parts[4]
        and parts[4].endswith("().")
    )


def _metric(left: FunctionUnit, right: FunctionUnit, status: str = "available") -> dict[str, Any]:
    a, b = set(left.subtree_hashes), set(right.subtree_hashes)
    if not a or not b:
        return _missing("empty_syntax_fingerprint")
    return {
        "score": len(a & b) / len(a | b),
        "exact": left.exact_hash == right.exact_hash,
        "status": status,
        "reason": None,
    }


def _missing(reason: str) -> dict[str, Any]:
    return {"score": None, "exact": None, "status": "abstained", "reason": reason}


def _summary(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    labelled = [row for row in rows if row["positive"] is not None]
    decided = [row for row in labelled if row[variant]["score"] is not None]
    tp = sum(row["positive"] and row[variant]["exact"] for row in decided)
    fp = sum(not row["positive"] and row[variant]["exact"] for row in decided)
    positives = sum(row["positive"] for row in labelled)
    fn = sum(row["positive"] and not row[variant]["exact"] for row in decided)
    tn = len(decided) - tp - fp - fn
    # Retain every possible review budget instead of choosing a flattering k.
    # A cutoff inside a score tie has expected uniform ordering, never ID order.
    curve = []
    seen = positive_seen = 0
    ordered = sorted(decided, key=lambda row: row[variant]["score"], reverse=True)
    for _, items in groupby(ordered, key=lambda row: row[variant]["score"]):
        group = list(items)
        positive = sum(row["positive"] for row in group)
        for offset in range(1, len(group) + 1):
            curve.append((positive_seen + offset * positive / len(group)) / (seen + offset))
        positive_seen += positive
        seen += len(group)
    return {
        "labelled_pairs": len(labelled),
        "unknown_oracle_pairs": len(rows) - len(labelled),
        "scored_labelled_pairs": len(decided),
        "abstained_labelled_pairs": len(labelled) - len(decided),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "conditional_recall": tp / (tp + fn) if tp + fn else None,
        "end_to_end_recall": tp / positives if positives else None,
        "labelled_pair_precision_by_k": curve,
        "rank_scope": "all budgets 1..scored labelled pairs; expected uniform score ties",
    }


def _implementation() -> dict[str, str]:
    import victor.core.graph_rag.structural_clones as engine

    root = Path(engine.__file__).parent
    return {
        **{name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in ENGINE_FILES},
        "benchmark_callee_context.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def _protobuf_version() -> str | None:
    try:
        return version("protobuf")
    except PackageNotFoundError:
        return None


def _peak_rss() -> int | None:
    try:
        import resource
    except ImportError:
        return None
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (
        1 if sys.platform == "darwin" else 1024
    )


def _publish(directory: int, output: Path, data: bytes, protected: list[Path]) -> None:
    """Publish a fresh report inside the parent owned before measurement.

    Descriptor-relative exclusive linking never replaces a raced-in file. Names
    are removed only while they still identify this call's private temporary.
    """
    parent = os.fstat(directory)
    parent_identity = parent.st_dev, parent.st_ino

    def verify() -> None:
        current = output.parent.lstat()
        if not stat.S_ISDIR(current.st_mode) or (current.st_dev, current.st_ino) != parent_identity:
            raise ValueError("Report output parent changed during measurement")
        for path in protected:
            if path.name != output.name:
                continue
            try:
                other = path.absolute().parent.stat()
            except FileNotFoundError:
                continue
            if (other.st_dev, other.st_ino) == parent_identity:
                raise ValueError("Output cannot replace an ablation input")

    verify()
    try:
        os.stat(output.name, dir_fd=directory, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        raise ValueError("Ablation reports require a fresh output destination")
    temporary = ".callee-ablation-" + secrets.token_hex(16)
    identity: tuple[int, int] | None = None

    def owned(name: str) -> bool:
        if identity is None:
            return False
        try:
            selected = os.stat(name, dir_fd=directory, follow_symlinks=False)
        except FileNotFoundError:
            return False
        return (selected.st_dev, selected.st_ino) == identity

    published = False
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory
    )
    try:
        info = os.fstat(descriptor)
        identity = info.st_dev, info.st_ino
        view = memoryview(data)
        while view:
            count = os.write(descriptor, view)
            if count <= 0:
                raise OSError("Could not write the owned report")
            view = view[count:]
        os.fsync(descriptor)
        verify()
        if not owned(temporary):
            raise ValueError("Report temporary identity changed")
        os.link(
            temporary,
            output.name,
            src_dir_fd=directory,
            dst_dir_fd=directory,
            follow_symlinks=False,
        )
        published = True
        if not owned(output.name):
            raise ValueError("Report publication identity changed")
        verify()
    except BaseException:
        if published and owned(output.name):
            os.unlink(output.name, dir_fd=directory)
        raise
    finally:
        try:
            if identity is None:
                # One read-only cleanup probe may recover identity after transient
                # metadata failure. Never unlink an identity we cannot establish.
                try:
                    info = os.fstat(descriptor)
                    identity = info.st_dev, info.st_ino
                except OSError:
                    pass
        finally:
            try:
                os.close(descriptor)
            finally:
                if owned(temporary):
                    os.unlink(temporary, dir_fd=directory)


def measure(
    root: Path, manifest: Path, *, index: Path | None = None, binding: Path | None = None
) -> dict[str, Any]:
    """Capture frozen bytes and compare only declared pairs, retaining abstentions."""
    started, cpu = time.perf_counter(), time.process_time()
    implementation = _implementation()
    manifest_data = _read(manifest, MAX_FILE)
    parsed = json.loads(manifest_data, object_pairs_hook=_unique)
    if not isinstance(parsed, dict):
        raise ValueError("Expected a control manifest object")
    fixture = _fixture(parsed)
    if (index is None) != (binding is None):
        raise ValueError("An index and its source-generation binding are required together")
    sources: dict[str, str] = {}
    total_bytes = 0
    with project_source_reader(root) as reader:
        for name, expected in fixture["source_hashes"].items():
            data, info = reader(name, MAX_FILE)
            if info.st_size > MAX_FILE or len(data) > MAX_FILE:
                raise ValueError("Source exceeds its byte budget")
            total_bytes += len(data)
            if total_bytes > MAX_SOURCE_BYTES:
                raise ValueError("Source universe exceeds its total byte budget")
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError("Source hash mismatch; control labels require re-review")
            sources[name] = data.decode("utf-8", errors="strict")
        reader.verify_root_path()
    trees = {name: ast.parse(source) for name, source in sources.items()}
    sites = {unit["id"]: _calls(trees[unit["file"]], unit) for unit in fixture["units"]}
    controlled: dict[str, OccurrenceMap] = defaultdict(dict)
    for unit in fixture["units"]:
        for site, target in zip(sites[unit["id"]], unit["controlled_targets"]):
            if site is not None and target is not None:
                controlled[unit["file"]][site[0]] = (site[1], target)
    indexed: dict[str, OccurrenceMap] = {}
    indexed_descriptor: dict[str, Any] = {"status": "not_requested"}
    indexed_started = time.perf_counter()
    if index is not None and binding is not None:
        indexed, indexed_descriptor = _index_maps(index, binding, sources, fixture["source_hashes"])
    # Add only resolved call-target leaves, never unrelated type/data identities.
    projected: dict[str, OccurrenceMap] = defaultdict(dict)
    for spec in fixture["units"]:
        for site in sites[spec["id"]]:
            if site is not None:
                hit = indexed.get(spec["file"], {}).get(site[0])
                if hit and hit[0] == site[1] and _resolved(hit[1]):
                    projected[spec["file"]][site[0]] = hit
    variants = {"syntax": {}, "controlled": controlled, "indexed": projected}
    units: dict[str, dict[str, FunctionUnit]] = {}
    features: dict[str, dict[str, Any]] = {}
    timings: dict[str, float] = {
        "indexed_capture_and_projection_seconds": (
            time.perf_counter() - indexed_started if index is not None else 0.0
        )
    }
    for variant, occurrences in variants.items():
        stamp = time.perf_counter()
        extracted = {
            name: {
                (unit.name, unit.line, unit.end_line): unit
                for unit in extract_units(
                    name,
                    source,
                    occurrences.get(name),
                    min_nodes=1,
                    strict_parse=True,
                    with_subtrees=True,
                )
            }
            for name, source in sources.items()
        }
        units[variant], features[variant] = {}, {}
        for spec in fixture["units"]:
            uid, file = spec["id"], spec["file"]
            units[variant][uid] = extracted[file][(spec["symbol"], spec["line"], spec["end_line"])]
            selected: OccurrenceMap = {}
            for site in sites[uid]:
                if site is not None:
                    hit = occurrences.get(file, {}).get(site[0])
                    if hit and hit[0] == site[1] and _resolved(hit[1]):
                        selected[site[0]] = hit
            features[variant][uid] = {
                "call_sites": len(sites[uid]),
                "resolved_call_sites": len(selected),
                "targets": [hit[1] for hit in selected.values()],
            }
        timings[variant + "_extract_seconds"] = time.perf_counter() - stamp
    rows = []
    for pair in fixture["pairs"]:
        row = dict(pair)
        for variant in variants:
            left, right = pair["left"], pair["right"]
            if variant == "indexed" and indexed_descriptor["status"] != "available":
                row[variant] = _missing(indexed_descriptor["status"])
            elif variant != "syntax" and any(
                features[variant][uid]["resolved_call_sites"] < features[variant][uid]["call_sites"]
                for uid in (left, right)
            ):
                row[variant] = _missing("unresolved_or_ambiguous_call_target")
            else:
                status = "no_call_sites" if not sites[left] and not sites[right] else "available"
                row[variant] = _metric(units[variant][left], units[variant][right], status)
        rows.append(row)
    if implementation != _implementation():
        raise ValueError("Implementation changed during ablation")
    if _read(manifest, MAX_FILE) != manifest_data:
        raise ValueError("Manifest changed during ablation")
    return {
        "schema": "victor-callee-context-ablation-v1",
        "production_quality_qualified": False,
        "weighted_fusion": False,
        "program_semantics_verified": False,
        "evaluation_scope": "authored_requested_pair_controls; not independent-project or retrieval accuracy",
        "decision_rule": "pre-existing exact normalized-token identity; no trained threshold",
        "oracle_domain": fixture["oracle_domain"],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "protobuf_runtime": _protobuf_version(),
        "implementation": implementation,
        "engine_module": inspect.getfile(extract_units),
        "fixture_sha256": hashlib.sha256(manifest_data).hexdigest(),
        "source_hashes": fixture["source_hashes"],
        "source_files": len(sources),
        "source_bytes": total_bytes,
        "compared_units": len(fixture["units"]),
        "requested_pairs": len(fixture["pairs"]),
        "indexed": indexed_descriptor,
        "controlled_provenance": "curator-authored target identities; not compiler output",
        "features": features,
        "pairs": rows,
        "summary": {variant: _summary(rows, variant) for variant in variants},
        "timings": timings,
        "wall_seconds": time.perf_counter() - started,
        "parent_cpu_seconds": time.process_time() - cpu,
        "process_peak_rss_bytes": _peak_rss(),
        "rss_scope": "process_lifetime_peak; not incremental context or child memory",
        "timing_scope": "source capture, extraction, requested scoring; excludes index generation and serialization",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--binding", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        fixture = load_fixture(args.manifest)
        protected = [args.manifest, *(args.root / name for name in fixture["source_hashes"])]
        protected.extend(
            args.root / ".victor" / ("project.db" + suffix)
            for suffix in ("", "-wal", "-shm", "-journal")
        )
        protected.extend(path for path in (args.index, args.binding) if path is not None)
        for path in protected:
            if str(args.output.resolve()).casefold() == str(path.resolve()).casefold() or (
                args.output.exists() and path.exists() and args.output.samefile(path)
            ):
                raise ValueError("Output cannot overwrite an ablation input")
        from victor.core.graph_rag.copy_artifacts import _parent_directory

        if os.link not in os.supports_dir_fd or os.link not in os.supports_follow_symlinks:
            raise ValueError("Secure report publication is unavailable")
        output = args.output.absolute()
        with _parent_directory(output) as directory:
            result = measure(args.root, args.manifest, index=args.index, binding=args.binding)
            data = (json.dumps(result, indent=2, allow_nan=False) + "\n").encode()
            if len(data) > MAX_REPORT:
                raise ValueError("Report exceeds its byte budget")
            _publish(directory, output, data, protected)
        print(json.dumps(result["summary"], allow_nan=False))
        return (
            0
            if args.index is None
            or (
                result["indexed"]["status"] == "available"
                and result["summary"]["indexed"]["abstained_labelled_pairs"] == 0
            )
            else 1
        )
    except (
        ImportError,
        OSError,
        ValueError,
        TypeError,
        KeyError,
        RecursionError,
        SyntaxError,
    ) as exc:
        print(f"Callee ablation failed: {str(exc)[:500]}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

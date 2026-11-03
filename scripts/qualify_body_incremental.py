"""Qualify cache changes and bounded exact retrieval on owned source copies.

This script never downloads models or executes source. Each observation uses a
fresh process; requested pairs and global retrieval have separate timers. The
caller supplies an immutable source manifest, a Git workload revision and a new
output directory. Historical cache seeds are copied with SQLite's backup API.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, TYPE_CHECKING

from victor.core.graph_rag.bounded_process import OwnedProcess

if TYPE_CHECKING:
    from scripts.body_qualification_io import (
        OwnedWork,
        bounded_read,
        cache_database,
        source_parts,
        MAX_CACHE_ROWS,
        MAX_MANIFEST_BYTES,
    )
else:
    _spec = importlib.util.spec_from_file_location(
        "body_qualification_io", Path(__file__).with_name("body_qualification_io.py")
    )
    if _spec is None or _spec.loader is None:
        raise RuntimeError("Qualification I/O owner is unavailable")
    _io = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_io)
    OwnedWork, bounded_read, cache_database, source_parts = (
        _io.OwnedWork,
        _io.bounded_read,
        _io.cache_database,
        _io.source_parts,
    )
    MAX_CACHE_ROWS, MAX_MANIFEST_BYTES = _io.MAX_CACHE_ROWS, _io.MAX_MANIFEST_BYTES


if TYPE_CHECKING:
    from scripts.body_qualification_metrics import (
        _digest,
        _scores,
        validate_observation,
        compare_unchanged,
        cache_vectors,
        global_oracle,
        compare_cold,
    )
else:
    _metrics_spec = importlib.util.spec_from_file_location(
        "body_qualification_metrics", Path(__file__).with_name("body_qualification_metrics.py")
    )
    if _metrics_spec is None or _metrics_spec.loader is None:
        raise RuntimeError("Qualification metrics owner is unavailable")
    _metrics = importlib.util.module_from_spec(_metrics_spec)
    _metrics_spec.loader.exec_module(_metrics)
    _digest, _scores = _metrics._digest, _metrics._scores
    validate_observation, compare_unchanged = (
        _metrics.validate_observation,
        _metrics.compare_unchanged,
    )
    cache_vectors, global_oracle, compare_cold = (
        _metrics.cache_vectors,
        _metrics.global_oracle,
        _metrics.compare_cold,
    )


def _environment() -> dict[str, str]:
    return {
        **{key: value for key, value in os.environ.items() if not key.upper().startswith("GIT_")},
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_NO_LAZY_FETCH": "1",
        "LC_ALL": "C",
    }


def _owned_call(
    arguments: list[str],
    descriptor: int,
    *,
    timeout: float = 10,
    max_rss: int = 256 * 1024 * 1024,
    max_output: int = 4 * 1024 * 1024,
    environment: dict[str, str] | None = None,
    recursive: bool = True,
) -> tuple[int, bytes, bytes]:
    with tempfile.TemporaryDirectory(prefix="victor-body-observer-") as directory:
        work = Path(directory)
        code, _ = OwnedProcess(
            arguments,
            work,
            timeout,
            max_rss=max_rss,
            artifacts={work / "backend.log": max_output, work / "backend.stderr": 256 * 1024},
            cwd_descriptor=descriptor,
            pass_fds=(descriptor,),
            separate_stderr=True,
            recursive=recursive,
            environment=environment or _environment(),
        ).run_sync()
        return (
            code,
            bounded_read(work / "backend.log", max_output),
            bounded_read(work / "backend.stderr", 256 * 1024),
        )


def _binary() -> str:
    binary = shutil.which("git")
    if binary is None:
        raise ValueError("Qualification requires Git")
    return str(Path(binary).resolve())


def _git(owned: OwnedWork, *arguments: str) -> None:
    owned.verify()
    code, _, errors = _owned_call(
        [
            _binary(),
            "--no-pager",
            "--literal-pathspecs",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.untrackedCache=false",
            "-c",
            "core.hooksPath=" + os.devnull,
            "-c",
            "core.attributesFile=" + os.devnull,
            "-c",
            "gc.auto=0",
            "-c",
            "maintenance.auto=false",
            "--git-dir=.git",
            "--work-tree=.",
            *arguments,
        ],
        owned.source_fd,
        recursive=False,
    )
    if code:
        raise ValueError(
            "Owned Git mutation failed: " + errors.decode("utf-8", errors="replace")[:1000]
        )
    owned.verify()


def export_blob(repository: Path, revision: str, relative: str) -> bytes:
    from victor.core.graph_rag.source_reader import project_source_reader

    with project_source_reader(repository) as reader:
        code, raw, _ = _owned_call(
            [
                _binary(),
                "--no-pager",
                "-c",
                "core.fsmonitor=false",
                "show",
                f"{revision}:{relative}",
            ],
            reader.descriptor,
            recursive=False,
        )
        reader.verify_root_path()
        if code:
            raise ValueError("Immutable workload export failed")
        return raw


def _observe(
    owned: OwnedWork, output_name: str, state: str, cache_name: str = "body.sqlite"
) -> dict[str, Any]:
    if len(source_parts(cache_name)) != 1 or not cache_name.endswith(".sqlite"):
        raise ValueError("Observer cache must be one owned relative filename")
    if not re.fullmatch(r"[a-z_]+(?:-cold)?", output_name):
        raise ValueError("Observer report must be one owned relative filename")
    owned.verify()
    repository = Path(__file__).resolve().parents[1]
    environment = {
        **_environment(),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONPATH": os.pathsep.join(
            str(path)
            for path in (
                repository,
                repository / "victor-codegraph",
                repository / "victor-contracts",
            )
        ),
    }
    code, raw, errors = _owned_call(
        [
            sys.executable,
            str(repository / "scripts/benchmark_body_cache.py"),
            "source",
            "--cache",
            cache_name,
            "--state",
            state,
            "--details",
            "--owned-cache",
        ],
        owned.fd,
        timeout=180,
        max_rss=4 * 1024**3,
        max_output=32 * 1024 * 1024,
        environment=environment,
    )
    owned.verify()
    owned.write(output_name + ".stderr.log", errors)
    if code:
        raise ValueError(f"Qualification observation failed ({code}); see {output_name}.stderr.log")
    value: dict[str, Any] = json.loads(raw)
    validate_observation(value, allow_incomplete=not value["coverage_complete"])
    value["global_oracle"] = (
        global_oracle(value, owned.path / cache_name) if value["coverage_complete"] else None
    )
    owned.write(output_name + ".json", json.dumps(value).encode())
    return value


def run(
    repository: Path,
    manifest: dict[str, str],
    revision: str,
    output: Path,
    seed: Path | None = None,
    *,
    expected_functions: int | None = None,
) -> dict[str, Any]:
    if (
        not isinstance(manifest, dict)
        or not 1 <= len(manifest) <= 100
        or not re.fullmatch(r"[0-9a-f]{40}", revision)
    ):
        raise ValueError("Use a bounded source manifest and immutable revision")
    for relative, digest in manifest.items():
        source_parts(relative)
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Invalid qualification source manifest")
    implementation = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in (
            "qualify_body_incremental.py",
            "body_qualification_io.py",
            "body_qualification_metrics.py",
            "benchmark_body_cache.py",
        )
    }
    with OwnedWork(output) as owned:
        total = 0
        for relative, expected in manifest.items():
            raw = export_blob(repository, revision, relative)
            total += len(raw)
            if total > 64 * 1024 * 1024 or hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError("Workload export mismatched its manifest or byte budget")
            owned.write_source(relative, raw)
        _git(owned, "init", "--template=", "-q")
        _git(owned, "add", "--all")
        seed_profiles = {}
        if seed is not None:
            with cache_database(seed) as db:
                seed_profiles = dict(
                    db.execute(
                        "SELECT profile,COUNT(*) FROM body_vectors_v1 GROUP BY profile LIMIT ?",
                        (MAX_CACHE_ROWS + 1,),
                    ).fetchmany(MAX_CACHE_ROWS + 1)
                )
            owned.seed_cache(seed)
        baseline = _observe(owned, "baseline", "warm" if seed else "cold")
        validate_observation(baseline)
        if (
            baseline["source_manifest"] != manifest
            or expected_functions is not None
            and baseline["functions"] != expected_functions
        ):
            raise ValueError(
                "Baseline selected-source/function coverage differs from immutable workload"
            )
        current_profile = baseline["profile"]["producer"]["identity"]
        if not seed_profiles.get(current_profile, 0) and baseline["stats"]["cache_hits"]:
            raise ValueError("Foreign producer rows were improperly reused")
        selection = None
        for item in baseline["global_result"]["inputs"]:
            if item["language"] != "python":
                continue
            source = owned.read_source(item["file"]).decode()
            for node in ast.parse(source).body:
                if (
                    isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.lineno == item["line"]
                    and sum(
                        isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                        for child in ast.walk(node)
                    )
                    == 1
                ):
                    segment = ast.get_source_segment(source, node)
                    if (
                        segment is None
                        or hashlib.sha256(segment.encode()).hexdigest() != item["input_hash"]
                    ):
                        continue
                    selection = (
                        item["file"],
                        node.body[0].lineno - 1,
                        node.body[0].col_offset,
                        node.lineno,
                    )
                    break
            if selection:
                break
        if selection is None:
            raise ValueError("Workload has no eligible top-level leaf Python function")
        relative, line, indent, definition_line = selection
        original = owned.read_source(relative)
        previous = baseline
        stages = []
        expected_manifest = dict(manifest)
        expected_count = baseline["functions"]
        removed_count = sum(item["file"] == relative for item in baseline["unit_manifest"])
        edited_hash: str | None = None

        def observe(name: str, complete: bool = True) -> dict[str, Any]:
            nonlocal previous
            current = _observe(owned, name, "warm")
            if current["coverage_complete"] is not complete:
                raise ValueError(f"Unexpected source selection status in {name}")
            if (
                current["source_manifest"] != expected_manifest
                or current["functions"] != expected_count
            ):
                raise ValueError(f"Selected cohort differs from the controlled snapshot in {name}")
            if current["source_coverage"].get("unreadable_files", 0) != (0 if complete else 1):
                raise ValueError("Unexpected selected-source unreadable count")
            validate_observation(current, allow_incomplete=not complete)
            parity = compare_unchanged(
                previous,
                current,
                allow_incomplete=not complete or not previous["coverage_complete"],
            )
            stats = current["stats"]
            new_inputs = set(current["unit_input_hashes"]) - set(previous["unit_input_hashes"])
            if name != "edit" and (stats["inferred_windows"] or stats["model_load_attempts"]):
                raise ValueError(f"Cached inputs unexpectedly loaded a model in {name}")
            if name == "edit" and (
                new_inputs != {edited_hash}
                or not stats["inferred_windows"]
                or stats["cache_hits"] != stats["unique_inputs"] - 1
            ):
                raise ValueError("Controlled edit did not invalidate exactly one input")
            cold_parity = cold_stats = None
            if complete:
                cold_name = name + "-cold.sqlite"
                cold = _observe(owned, name + "-cold", "cold", cold_name)
                cold_parity = compare_cold(
                    current, cold, owned.path / "body.sqlite", owned.path / cold_name
                )
                cold_stats = cold["stats"]
            stages.append(
                {
                    "state": name,
                    "coverage_complete": complete,
                    "source_manifest": current["source_manifest"],
                    "stats": stats,
                    "global_stats": current["global_result"]["stats"],
                    "cold_stats": cold_stats,
                    "collection_wall_seconds": current["collection_wall_seconds"],
                    "collection_cpu_seconds": current["collection_cpu_seconds"],
                    "requested_wall_seconds": current["wall_seconds"],
                    "requested_cpu_seconds": current["cpu_seconds"],
                    "global_wall_seconds": current["global_wall_seconds"],
                    "global_cpu_seconds": current["global_cpu_seconds"],
                    "global_timing_state": "warm_vectors_fresh_encoder_after_requested_scoring",
                    "process_peak_rss_bytes": current["process_peak_rss_bytes"],
                    "cached_rows_in_current_profile": len(current["cached_vectors"]),
                    "active_unique_inputs": len(set(current["unit_input_hashes"])),
                    "unchanged": parity,
                    "cold_rebuild": cold_parity,
                    "global_oracle": current["global_oracle"],
                }
            )
            previous = current
            return current

        observe("warm")
        lines = original.decode().splitlines(keepends=True)
        lines.insert(line, " " * indent + "# Incremental qualification changes this input only.\n")
        edited = "".join(lines)
        node = next(
            node
            for node in ast.parse(edited).body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.lineno == definition_line
        )
        segment = ast.get_source_segment(edited, node)
        if segment is None:
            raise ValueError("Edited model input cannot be independently captured")
        edited_hash = hashlib.sha256(segment.encode()).hexdigest()
        owned.write_source(relative, edited.encode())
        expected_manifest[relative] = hashlib.sha256(edited.encode()).hexdigest()
        observe("edit")
        owned.unlink_source(relative)
        del expected_manifest[relative]
        expected_count -= removed_count
        observe("unstaged_delete", False)
        _git(owned, "rm", "--cached", "--force", "--", relative)
        observe("delete")
        owned.write_source(relative, original)
        _git(owned, "add", "--", relative)
        expected_manifest = dict(manifest)
        expected_count = baseline["functions"]
        restored = observe("restore")
        if restored["pair_scores_sha256"] != baseline["pair_scores_sha256"]:
            raise ValueError("Restored workload requested scores differ from baseline")
        renamed = str(Path(relative).with_name("r1_renamed_" + Path(relative).name))
        owned.move_source(relative, renamed)
        _git(owned, "update-index", "--force-remove", "--", relative)
        _git(owned, "add", "--", renamed)
        expected_manifest[renamed] = expected_manifest.pop(relative)
        moved = observe("rename")
        if any(item["file"] == relative for item in moved["global_result"]["inputs"]):
            raise ValueError("Renamed source retained stale output locations")
        owned.move_source(renamed, relative)
        _git(owned, "update-index", "--force-remove", "--", renamed)
        _git(owned, "add", "--", relative)
        expected_manifest[relative] = expected_manifest.pop(renamed)
        observe("rename_restore")
        for name, expected in implementation.items():
            if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != expected:
                raise ValueError("Qualification implementation changed during measurement")
        result = {
            "schema": "victor-incremental-body-qualification-v2",
            "revision": revision,
            "implementation_sha256": implementation,
            "source_manifest": manifest,
            "producer": baseline["profile"],
            "seed_profiles": seed_profiles,
            "seed_current_generation_rows": seed_profiles.get(current_profile, 0),
            "baseline_stats": baseline["stats"],
            "baseline_observation": {
                key: baseline[key]
                for key in (
                    "wall_seconds",
                    "cpu_seconds",
                    "collection_wall_seconds",
                    "collection_cpu_seconds",
                    "global_wall_seconds",
                    "global_cpu_seconds",
                    "process_peak_rss_bytes",
                    "rss_scope",
                    "python",
                    "platform",
                    "implementation",
                    "global_options",
                    "global_oracle",
                )
            },
            "baseline_pair_count": baseline["pair_count"],
            "baseline_vector_manifest_sha256": baseline["vector_manifest_sha256"],
            "baseline_pair_scores_sha256": baseline["pair_scores_sha256"],
            "mutation": {
                "file": relative,
                "comment_line": line + 1,
                "indent": indent,
                "renamed_to": renamed,
            },
            "stages": stages,
            "model_downloads": False,
            "accuracy_benchmark": False,
            "source_collection_incremental": False,
            "retrieval": "current_tiled_exact_threshold_join",
            "limits": [
                "one local CPU/runtime and bounded actual-source workload",
                "global timers use cached vectors after requested scoring, not standalone cold global inference",
                "syntax source collection rescans every stage; stale cached rows are retained outside active outputs",
                "public components/counts/extrema checked; raw edge identities are not exposed",
                "no ANN or retrieval rewrite; no broad accuracy or release claim",
            ],
        }
        owned.write("qualification.json", (json.dumps(result, indent=2) + "\n").encode())
        owned.publish()
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-cache", type=Path)
    args = parser.parse_args()
    document = json.loads(bounded_read(args.manifest, MAX_MANIFEST_BYTES))
    manifest = document["source_manifest"]
    result = run(
        args.repository.resolve(),
        manifest,
        args.revision,
        args.output.absolute(),
        args.seed_cache,
        expected_functions=document.get("comparison", {}).get("functions"),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

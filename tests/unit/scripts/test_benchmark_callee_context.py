"""Callee identities are an ablation, with freshness and unknowns kept explicit."""

import importlib.util
import hashlib
from itertools import permutations
import json
import os
from pathlib import Path
import shutil
import stat
import statistics
import sys

import pytest


@pytest.fixture
def callee_benchmark():
    path = Path(__file__).resolve().parents[3] / "scripts/benchmark_callee_context.py"
    spec = importlib.util.spec_from_file_location("benchmark_callee_context", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fixture():
    return Path(__file__).resolve().parents[2] / "fixtures/graph_context"


def _scip_runtime():
    try:
        from victor.core.graph_rag._scip import scip_pb2
    except (ImportError, RuntimeError) as exc:
        if os.environ.get("VICTOR_REQUIRE_SCIP_READER_TESTS") == "1":
            pytest.fail(f"Required SCIP ablation runtime unavailable: {exc}")
        pytest.skip("optional protobuf runtime unavailable")
    return scip_pb2


def test_required_runtime_flag_cannot_silently_skip_actual_fixture(monkeypatch):
    import builtins

    original = builtins.__import__

    def missing(name, *args, **kwargs):
        if name == "victor.core.graph_rag._scip":
            raise ImportError("controlled absent runtime")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing)
    monkeypatch.setenv("VICTOR_REQUIRE_SCIP_READER_TESTS", "1")
    with pytest.raises(pytest.fail.Exception, match="Required SCIP ablation runtime unavailable"):
        _scip_runtime()


def test_controlled_callee_ablation_preserves_counterexamples_and_unknowns(
    callee_benchmark, fixture
):
    report = callee_benchmark.measure(fixture, fixture / "manifest.json")
    pairs = {row["id"]: row for row in report["pairs"]}
    assert report["production_quality_qualified"] is False
    assert report["weighted_fusion"] is False
    assert pairs["same-spelling-different-owner"]["syntax"]["exact"] is True
    assert pairs["same-spelling-different-owner"]["controlled"]["exact"] is False
    assert pairs["aliased-same-target"]["syntax"]["exact"] is False
    assert pairs["aliased-same-target"]["controlled"]["exact"] is True
    assert pairs["changed-literal-same-target"]["controlled"]["exact"] is True
    assert pairs["unresolved-any-receiver"]["controlled"]["score"] is None
    assert "unresolved" in pairs["unresolved-any-receiver"]["controlled"]["reason"]
    assert pairs["no-calls"]["controlled"]["status"] == "no_call_sites"
    assert report["indexed"]["status"] == "not_requested"


def test_actual_indexer_matches_controlled_decisions_with_explicit_provenance(
    callee_benchmark, fixture
):
    _scip_runtime()
    report = callee_benchmark.measure(
        fixture,
        fixture / "manifest.json",
        index=fixture / "compiler.scip",
        binding=fixture / "compiler-binding.json",
    )
    assert (
        report["fixture_sha256"]
        == "c4a4d6e30e8ee8e43fdb1cae06d74869964c3f7e7be4e72454652a312794b26c"
    )
    assert report["indexed"]["status"] == "available"
    assert report["indexed"]["metadata_producer"] == {"name": "scip-python", "version": "0.6.8"}
    assert report["summary"]["indexed"] == report["summary"]["controlled"]
    assert report["summary"]["indexed"]["tp"] == 3
    assert report["summary"]["indexed"]["fp"] == 1
    assert report["summary"]["syntax"]["fp"] == 3
    assert "not authentication" in report["indexed"]["source_freshness"]


def test_changed_source_is_not_scored_under_frozen_labels(callee_benchmark, fixture, tmp_path):
    root = tmp_path / "source"
    shutil.copytree(fixture, root)
    (root / "control_methods.py").write_bytes((root / "control_methods.py").read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Source hash mismatch"):
        callee_benchmark.measure(root, root / "manifest.json")


@pytest.mark.parametrize("change", ["alias", "duplicate_pair", "integer_label", "unknown_field"])
def test_malformed_fixture_cannot_inflate_or_relabel_controls(
    callee_benchmark, fixture, tmp_path, change
):
    raw = json.loads((fixture / "manifest.json").read_text())
    if change == "alias":
        raw["units"].append({**raw["units"][0], "id": "alias"})
    elif change == "duplicate_pair":
        raw["pairs"].append({**raw["pairs"][0], "id": "duplicate"})
    elif change == "integer_label":
        raw["pairs"][0]["positive"] = 1
    else:
        raw["score_tuned_threshold"] = 0.5
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        callee_benchmark.load_fixture(path)


def test_duplicate_json_fields_fail_closed(callee_benchmark, tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text('{"schema":"one","schema":"two"}')
    with pytest.raises(ValueError, match="Duplicate manifest field"):
        callee_benchmark.load_fixture(path)


def test_unqualified_target_spelling_remains_unresolved(callee_benchmark, fixture, tmp_path):
    raw = json.loads((fixture / "manifest.json").read_text())
    for unit in raw["units"]:
        if unit["id"] == "registry_read":
            unit["controlled_targets"] = ["get"]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(raw))
    report = callee_benchmark.measure(fixture, path)
    row = next(row for row in report["pairs"] if row["id"] == "renamed-same-owner")
    assert row["syntax"]["score"] == 1.0
    assert row["controlled"]["score"] is None


def test_total_source_budget_is_enforced_before_extraction(callee_benchmark, fixture, monkeypatch):
    monkeypatch.setattr(callee_benchmark, "MAX_SOURCE_BYTES", 1)
    with pytest.raises(ValueError, match="total byte budget"):
        callee_benchmark.measure(fixture, fixture / "manifest.json")


def test_rank_curve_averages_cutoff_ties_and_ignores_unknown_labels(callee_benchmark):
    rows = [
        {"positive": True, "channel": {"score": 1.0, "exact": True}},
        {"positive": False, "channel": {"score": 1.0, "exact": True}},
        {"positive": True, "channel": {"score": 0.5, "exact": False}},
        {"positive": None, "channel": {"score": 1.0, "exact": True}},
    ]
    for ordering in (rows, list(reversed(rows))):
        assert callee_benchmark._summary(ordering, "channel")["labelled_pair_precision_by_k"] == [
            0.5,
            0.5,
            2 / 3,
        ]


def test_stale_index_source_binding_is_not_negative_evidence(callee_benchmark, fixture, tmp_path):
    binding = json.loads((fixture / "compiler-binding.json").read_text())
    binding["source_hashes"]["control_methods.py"] = "0" * 64
    path = tmp_path / "binding.json"
    path.write_text(json.dumps(binding))
    report = callee_benchmark.measure(
        fixture,
        fixture / "manifest.json",
        index=fixture / "compiler.scip",
        binding=path,
    )
    assert report["indexed"]["status"] == "stale_source_binding"
    assert report["indexed"]["source_freshness"] == "source_hash_sidecar_mismatch"
    assert report["summary"]["indexed"]["scored_labelled_pairs"] == 0
    assert report["summary"]["indexed"]["tn"] == 0
    assert report["summary"]["indexed"]["precision"] is None


def test_unavailable_protobuf_does_not_become_compiler_evidence(
    callee_benchmark, fixture, monkeypatch
):
    import builtins

    original = builtins.__import__

    def unavailable(name, *args, **kwargs):
        if name == "victor.core.graph_rag._scip":
            raise ImportError("controlled missing optional runtime")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", unavailable)
    report = callee_benchmark.measure(
        fixture,
        fixture / "manifest.json",
        index=fixture / "compiler.scip",
        binding=fixture / "compiler-binding.json",
    )
    assert report["indexed"]["status"] == "protobuf_runtime_unavailable"
    assert report["summary"]["controlled"]["scored_labelled_pairs"] == 7
    assert report["summary"]["indexed"]["scored_labelled_pairs"] == 0


def test_index_hash_and_byte_budget_are_checked_before_parsing(
    callee_benchmark, fixture, tmp_path, monkeypatch
):
    binding = json.loads((fixture / "compiler-binding.json").read_text())
    binding["index_sha256"] = "0" * 64
    path = tmp_path / "binding.json"
    path.write_text(json.dumps(binding))
    with pytest.raises(ValueError, match="Index hash"):
        callee_benchmark.measure(
            fixture,
            fixture / "manifest.json",
            index=fixture / "compiler.scip",
            binding=path,
        )
    monkeypatch.setattr(callee_benchmark, "MAX_INDEX", 1)
    with pytest.raises(ValueError, match="byte budget"):
        callee_benchmark.measure(
            fixture,
            fixture / "manifest.json",
            index=fixture / "compiler.scip",
            binding=path,
        )


@pytest.mark.parametrize("mutation", ["duplicate_document", "ambiguous_target", "wrong_producer"])
def test_mutated_compiler_input_cannot_invent_context(
    callee_benchmark, fixture, tmp_path, mutation
):
    scip_pb2 = _scip_runtime()

    index = scip_pb2.Index()
    index.ParseFromString((fixture / "compiler.scip").read_bytes())
    if mutation == "duplicate_document":
        index.documents.add().CopyFrom(index.documents[0])
    elif mutation == "wrong_producer":
        index.metadata.tool_info.version = "different-version"
    else:
        doc = next(doc for doc in index.documents if doc.relative_path == "control_methods.py")
        target = next(
            occurrence for occurrence in doc.occurrences if list(occurrence.range) == [8, 18, 21]
        )
        extra = doc.occurrences.add()
        extra.CopyFrom(target)
        extra.symbol = extra.symbol.replace("Registry", "Unrelated")
    data = index.SerializeToString()
    path = tmp_path / "mutated.scip"
    path.write_bytes(data)
    binding = json.loads((fixture / "compiler-binding.json").read_text())
    binding["index_sha256"] = hashlib.sha256(data).hexdigest()
    binding_path = tmp_path / "binding.json"
    binding_path.write_text(json.dumps(binding))
    if mutation != "ambiguous_target":
        with pytest.raises(ValueError):
            callee_benchmark.measure(
                fixture, fixture / "manifest.json", index=path, binding=binding_path
            )
    else:
        report = callee_benchmark.measure(
            fixture, fixture / "manifest.json", index=path, binding=binding_path
        )
        row = next(row for row in report["pairs"] if row["id"] == "renamed-same-owner")
        assert row["indexed"]["score"] is None
        assert row["controlled"]["exact"] is True


def test_manifest_change_during_measurement_invalidates_report(
    callee_benchmark, fixture, tmp_path, monkeypatch
):
    path = tmp_path / "manifest.json"
    path.write_bytes((fixture / "manifest.json").read_bytes())
    original = callee_benchmark._calls

    def change_manifest(*args):
        path.write_bytes(path.read_bytes() + b"\n")
        return original(*args)

    monkeypatch.setattr(callee_benchmark, "_calls", change_manifest)
    with pytest.raises(ValueError, match="Manifest changed"):
        callee_benchmark.measure(fixture, path)


@pytest.mark.parametrize("destination", ["source", "manifest", "index", "binding", "database"])
def test_cli_cannot_replace_its_inputs_or_project_database(
    callee_benchmark, fixture, tmp_path, destination
):
    root = tmp_path / "source"
    shutil.copytree(fixture, root)
    outputs = {
        "source": root / "control_methods.py",
        "manifest": root / "manifest.json",
        "index": root / "compiler.scip",
        "binding": root / "compiler-binding.json",
        "database": root / ".victor/project.db-wal",
    }
    output = outputs[destination]
    output.parent.mkdir(exist_ok=True)
    if destination == "database":
        output.write_bytes(b"preserve database sidecar")
    before = output.read_bytes()
    assert (
        callee_benchmark.main(
            [
                "--root",
                str(root),
                "--manifest",
                str(root / "manifest.json"),
                "--index",
                str(root / "compiler.scip"),
                "--binding",
                str(root / "compiler-binding.json"),
                "--output",
                str(output),
            ]
        )
        == 2
    )
    assert output.read_bytes() == before


@pytest.mark.skipif(os.name != "posix", reason="secure report publication currently requires POSIX")
def test_fresh_report_is_published_but_existing_destination_is_preserved(
    callee_benchmark, fixture, tmp_path
):
    output = tmp_path / "reports" / "report.json"
    args = [
        "--root",
        str(fixture),
        "--manifest",
        str(fixture / "manifest.json"),
        "--output",
        str(output),
    ]
    assert callee_benchmark.main(args) == 0
    before = output.read_bytes()
    assert json.loads(before)["summary"]["controlled"]["scored_labelled_pairs"] == 7
    assert callee_benchmark.main(args) == 2
    assert output.read_bytes() == before


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX directory symlink race")
def test_output_parent_swap_cannot_redirect_report_into_protected_source(
    callee_benchmark, fixture, tmp_path, monkeypatch
):
    root, parent = tmp_path / "source", tmp_path / "reports"
    shutil.copytree(fixture, root)
    parent.mkdir()
    source = root / "control_methods.py"
    before = source.read_bytes()
    original = callee_benchmark.measure

    def swap_parent(*args, **kwargs):
        report = original(*args, **kwargs)
        parent.rename(tmp_path / "old-reports")
        parent.symlink_to(root, target_is_directory=True)
        return report

    monkeypatch.setattr(callee_benchmark, "measure", swap_parent)
    assert (
        callee_benchmark.main(
            [
                "--root",
                str(root),
                "--manifest",
                str(root / "manifest.json"),
                "--output",
                str(parent / "control_methods.py"),
            ]
        )
        == 2
    )
    assert source.read_bytes() == before


@pytest.mark.skipif(os.name != "posix", reason="requires descriptor-relative publication")
@pytest.mark.parametrize("failure", [OSError, KeyboardInterrupt])
def test_report_metadata_failure_closes_new_fd_and_owned_temporary(
    callee_benchmark, tmp_path, monkeypatch, failure
):
    parent = tmp_path / "reports"
    parent.mkdir()
    original = os.fstat
    failed = []

    def fail_new_regular_fd(descriptor):
        info = original(descriptor)
        if stat.S_ISREG(info.st_mode) and not failed:
            failed.append(descriptor)
            raise failure("controlled metadata failure")
        return info

    monkeypatch.setattr(callee_benchmark.os, "fstat", fail_new_regular_fd)
    descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(failure, match="controlled metadata failure"):
            callee_benchmark._publish(descriptor, parent / "report.json", b"report", [])
    finally:
        os.close(descriptor)
    assert failed
    with pytest.raises(OSError):
        original(failed[0])
    assert list(parent.iterdir()) == []


def test_retained_observations_bind_frozen_inputs_and_replay_counts_ranks_medians(fixture):
    root = Path(__file__).resolve().parents[3]
    evidence = json.loads(
        (root / "docs/architecture/evidence/callee-context-ablation-2026-11-02.json").read_text()
    )
    assert not evidence["production_quality_qualified"]
    for name, digest in evidence["source_files"].items():
        assert hashlib.sha256((fixture / name).read_bytes()).hexdigest() == digest
    rows = evidence["measurement"]["runs"]
    geometries = []
    for row in rows:
        report = row["observation"]
        assert report["fixture_sha256"] == evidence["fixture_manifest_sha256"]
        assert (
            report["implementation"]["benchmark_callee_context.py"] == evidence["observer_sha256"]
        )
        assert not report["program_semantics_verified"] and not report["weighted_fusion"]
        geometries.append(
            {
                key: report[key]
                for key in ("pairs", "features", "source_hashes", "indexed", "implementation")
            }
        )
    assert all(row == geometries[0] for row in geometries)
    digest = hashlib.sha256(
        json.dumps(geometries[0], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert digest == evidence["measurement"]["geometry_sha256"]
    for key, median in evidence["measurement"]["medians"].items():
        if key == "cli_wall_seconds":
            values = [row[key] for row in rows]
        elif key.endswith("_extract_seconds") or key == "indexed_capture_and_projection_seconds":
            values = [row["observation"]["timings"][key] for row in rows]
        else:
            values = [row["observation"][key] for row in rows]
        assert statistics.median(values) == median
    report = rows[0]["observation"]
    for variant in ("syntax", "controlled", "indexed"):
        labelled = [row for row in report["pairs"] if row["positive"] is not None]
        decided = [row for row in labelled if row[variant]["score"] is not None]
        metrics = report["summary"][variant]
        counts = {(positive, exact): 0 for positive in (True, False) for exact in (True, False)}
        for pair in decided:
            counts[pair["positive"], pair[variant]["exact"]] += 1
        assert [metrics[key] for key in ("tp", "fp", "fn", "tn")] == [
            counts[True, True],
            counts[False, True],
            counts[True, False],
            counts[False, False],
        ]
        # Exhaustive ordering oracle is independent of the grouped rank formula.
        assert len(decided) == 7
        totals = [0.0] * len(decided)
        number = 0
        for ordering in permutations(decided):
            ranked = sorted(ordering, key=lambda pair: pair[variant]["score"], reverse=True)
            positives = 0
            for index, pair in enumerate(ranked):
                positives += pair["positive"]
                totals[index] += positives / (index + 1)
            number += 1
        assert metrics["labelled_pair_precision_by_k"] == pytest.approx(
            [total / number for total in totals]
        )

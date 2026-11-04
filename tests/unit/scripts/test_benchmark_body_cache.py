"""Cache observations preserve repeatable scores and reject stale measurements."""

import hashlib
import importlib.util
import copy
import os
import sqlite3
from pathlib import Path

import pytest


def setup_probe(tmp_path, monkeypatch):
    from victor.core.graph_rag.body_clones import LocalBodyEncoder
    from victor.core.graph_rag.structural_clones import StructuralReport, extract_units

    path = Path(__file__).resolve().parents[3] / "scripts/benchmark_body_cache.py"
    spec = importlib.util.spec_from_file_location("body_cache_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "source"
    root.mkdir()
    file = root / "test.py"
    source = "def f(x):\n return x+1\n\ndef g(x):\n return x-1\n"
    file.write_text(source)
    report = StructuralReport(
        [],
        [],
        {"source_files": 1, "selected_regions": 1},
        units=extract_units("test.py", source, None, min_nodes=1, retain_source=True),
        source_manifest={"test.py": hashlib.sha256(source.encode()).hexdigest()},
    )
    monkeypatch.setattr(module, "scan", lambda *a, **k: report)

    class Encoder(LocalBodyEncoder):
        def windows(self, source):
            return [source], [(0, len(source))]

        def encode(self, texts):
            return [[1] * self.options.dimension for _ in texts]

    monkeypatch.setattr(
        module, "LocalBodyEncoder", lambda options: Encoder(options, producer_id="tests/probe-v1")
    )
    return module, root, file


def test_cold_and_warm_measurement_preserves_vector_and_score_hashes(tmp_path, monkeypatch):
    module, root, _ = setup_probe(tmp_path, monkeypatch)
    cache = tmp_path / "cache.sqlite"
    cold = module.benchmark(root, cache, state="cold")
    warm = module.benchmark(root, cache, state="warm")
    assert cold["stats"]["inferred_windows"] == 2
    assert warm["stats"]["cache_hits"] == 2
    assert warm["stats"]["inferred_windows"] == 0
    assert cold["vector_manifest_sha256"] == warm["vector_manifest_sha256"]
    assert cold["pair_scores_sha256"] == warm["pair_scores_sha256"]
    assert cold["pair_count"] == 1
    assert not cold["accuracy_benchmark"]
    with pytest.raises(ValueError, match="existence"):
        module.benchmark(root, cache, state="cold")


def test_measurement_rejects_changed_source_snapshot(tmp_path, monkeypatch):
    module, root, file = setup_probe(tmp_path, monkeypatch)
    original = module.analyze_bodies

    def change(*args, **kwargs):
        result = original(*args, **kwargs)
        file.write_text("# changed snapshot\n")
        return result

    monkeypatch.setattr(module, "analyze_bodies", change)
    with pytest.raises(RuntimeError, match="snapshot changed"):
        module.benchmark(root, tmp_path / "cache.sqlite", state="cold")


def test_cache_inside_snapshot_and_missing_warm_cache_rejected(tmp_path, monkeypatch):
    module, root, _ = setup_probe(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="outside"):
        module.benchmark(root, root / "cache.sqlite", state="cold")
    with pytest.raises(ValueError, match="existence"):
        module.benchmark(root, tmp_path / "missing.sqlite", state="warm")


def test_incremental_observation_separates_requested_scores_from_global_retrieval(
    tmp_path, monkeypatch
):
    module, root, _ = setup_probe(tmp_path, monkeypatch)
    observed = module.benchmark(root, tmp_path / "cache.sqlite", state="cold", details=True)
    assert len(observed["unit_input_hashes"]) == 2
    assert len(observed["pair_scores"]) == 1
    assert observed["global_result"]["stats"]["retrieval"] == "tiled_exact_cosine"
    assert observed["global_result"]["stats"]["model_load_attempts"] == 0
    assert observed["global_result"]["stats"]["cache_hits"] == 2
    assert observed["global_retrieval_benchmark"] is True
    assert observed["collection_wall_seconds"] >= 0
    assert observed["global_wall_seconds"] >= 0


def load_qualification():
    path = Path(__file__).resolve().parents[3] / "scripts/qualify_body_incremental.py"
    spec = importlib.util.spec_from_file_location("incremental_qualification", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_incremental_checks_reject_vector_score_and_producer_drift(tmp_path, monkeypatch):
    probe, root, _ = setup_probe(tmp_path, monkeypatch)
    cache = tmp_path / "cache.sqlite"
    before = probe.benchmark(root, cache, state="cold", details=True)
    after = probe.benchmark(root, cache, state="warm", details=True)
    qualify = load_qualification()
    assert qualify.compare_unchanged(before, after)["shared_inputs"] == 2
    for kind in ("vector", "score", "producer"):
        invalid = copy.deepcopy(after)
        if kind == "vector":
            invalid["cached_vectors"][0][1] = "corrupt"
        elif kind == "score":
            invalid["pair_scores"][0]["score"] = 0.125
        else:
            invalid["profile"]["producer"]["identity"] = "changed"
        with pytest.raises(ValueError, match=kind):
            qualify.compare_unchanged(before, invalid)


def test_global_oracle_preserves_same_file_filter_and_rejects_false_families(tmp_path, monkeypatch):
    probe, root, _ = setup_probe(tmp_path, monkeypatch)
    cache = tmp_path / "cache.sqlite"
    observation = probe.benchmark(root, cache, state="cold", details=True)
    qualify = load_qualification()
    # Identical vectors in the same file must not form a default global family.
    assert qualify.global_oracle(observation, cache)["matching_edges"] == 0
    invalid = copy.deepcopy(observation)
    invalid["global_result"]["families"] = [
        {"members": observation["global_result"]["inputs"], "matching_edges": 1}
    ]
    with pytest.raises(ValueError, match="global retrieval"):
        qualify.global_oracle(invalid, cache)
    with sqlite3.connect(cache) as db:
        db.execute("UPDATE body_vectors_v1 SET checksum='corrupt'")
    with pytest.raises(ValueError, match="checksum"):
        qualify.global_oracle(observation, cache)


def control_observation(tmp_path):
    """Three authored vectors: two threshold edges, one subthreshold endpoint pair."""
    import itertools
    import math
    import json
    import numpy as np

    metrics = load_metrics()
    cache = tmp_path / "vectors.sqlite"
    units = [
        {
            "file": f"{i}.py",
            "symbol": "f",
            "line": 1,
            "region_index": 0,
            "source_start": 0,
            "input_hash": hashlib.sha256(f"input-{i}".encode()).hexdigest(),
        }
        for i in range(3)
    ]
    inputs = [
        {
            "id": hashlib.sha256(
                json.dumps([u["file"], u["line"], 0, 0, u["input_hash"]]).encode()
            ).hexdigest(),
            "file": u["file"],
            "symbol": u["symbol"],
            "line": 1,
            "input_hash": u["input_hash"],
        }
        for u in units
    ]
    source_manifest = {u["file"]: hashlib.sha256(u["file"].encode()).hexdigest() for u in units}
    rows = []
    with sqlite3.connect(cache) as db:
        db.execute("CREATE TABLE body_vectors_v1(profile,input_hash,vector,spans,checksum)")
        for unit, angle in zip(units, (0, 20, 40)):
            raw = np.array(
                [math.cos(math.radians(angle)), math.sin(math.radians(angle))], dtype="<f4"
            ).tobytes()
            spans = "[[0,1]]"
            db.execute(
                "INSERT INTO body_vectors_v1 VALUES(?,?,?,?,?)",
                (
                    "control",
                    unit["input_hash"],
                    raw,
                    spans,
                    hashlib.sha256(raw + spans.encode()).hexdigest(),
                ),
            )
            rows.append([unit["input_hash"], hashlib.sha256(raw).hexdigest(), spans])
    rows.sort()
    scores = [
        {"left": i, "right": j, "score": math.cos(math.radians(20 * (j - i)))}
        for i, j in itertools.combinations(range(3), 2)
    ]
    profile = {"producer": {"identity": "control"}, "dimension": 2, "comparison_scope": "all"}
    observation = {
        "unit_manifest": units,
        "unit_manifest_sha256": metrics._digest(units),
        "unit_input_hashes": [u["input_hash"] for u in units],
        "functions": 3,
        "stats": {"functions": 3, "unique_inputs": 3, "retrieval": "requested_pair_cosine"},
        "coverage_complete": True,
        "errors": [],
        "source_manifest": source_manifest,
        "profile": profile,
        "global_options": dict(metrics.GLOBAL_OPTIONS),
        "requested_inputs": inputs,
        "cached_vectors": rows,
        "vector_manifest_sha256": metrics._digest(rows),
        "pair_scores": scores,
        "pair_count": 3,
        "pair_scores_sha256": metrics._digest(scores),
        "pair_abstentions": 0,
        "global_retrieval_benchmark": True,
        "global_result": {
            "inputs": inputs,
            "profile": profile,
            "coverage_complete": True,
            "errors": [],
            "source_manifest": source_manifest,
            "stats": {
                "functions": 3,
                "unique_inputs": 3,
                "cache_hits": 3,
                "model_load_attempts": 0,
                "inferred_windows": 0,
                "neighbors": None,
                "retrieval": "tiled_exact_cosine",
            },
            "families": [
                {
                    "members": inputs,
                    "matching_edges": 2,
                    "score": math.cos(math.radians(20)),
                    "min_edge_similarity": math.cos(math.radians(20)),
                }
            ],
        },
    }
    observation["stats"].update(cache_hits=3, model_load_attempts=0, inferred_windows=0)
    return metrics, observation, cache


def test_global_oracle_checks_transitive_components_without_inventing_clique_edges(tmp_path):
    metrics, observation, cache = control_observation(tmp_path)
    assert metrics.global_oracle(observation, cache)["matching_edges"] == 2
    observation["global_result"]["families"][0]["matching_edges"] = 3
    with pytest.raises(ValueError, match="global retrieval"):
        metrics.global_oracle(observation, cache)


@pytest.mark.parametrize("relative", ["../escape.py", "/outside.py", "a/../escape.py", "./a.py"])
def test_invalid_incremental_source_manifest_fails_before_output_or_export(tmp_path, relative):
    qualify = load_qualification()
    output = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="manifest"):
        qualify.run(tmp_path, {relative: "a" * 64}, "1" * 40, output)
    assert not output.exists()


@pytest.mark.parametrize(
    "relative", [".git/config", "folder/.git/config", ".gitattributes", ".gitignore"]
)
def test_incremental_manifest_cannot_install_git_control_files(tmp_path, monkeypatch, relative):
    qualify = load_qualification()
    monkeypatch.setattr(
        qualify, "export_blob", lambda *args: pytest.fail("Git control file reached export")
    )
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="control"):
        qualify.run(tmp_path, {relative: "a" * 64}, "1" * 40, output)
    assert not output.exists()


def test_incomplete_vector_pair_or_global_cohort_cannot_be_certified(tmp_path, monkeypatch):
    probe, root, _ = setup_probe(tmp_path, monkeypatch)
    cache = tmp_path / "cache.sqlite"
    observation = probe.benchmark(root, cache, state="cold", details=True)
    qualify = load_qualification()
    missing = copy.deepcopy(observation)
    missing["cached_vectors"] = []
    missing["pair_scores"] = []
    missing["vector_manifest_sha256"] = qualify._digest([])
    missing["pair_scores_sha256"] = qualify._digest([])
    with pytest.raises(ValueError, match="vector|pair|coverage"):
        qualify.compare_unchanged(missing, missing)
    partial = copy.deepcopy(observation)
    partial["functions"] = 3
    partial["unit_input_hashes"].append("a" * 64)
    with pytest.raises(ValueError, match="cohort"):
        qualify.global_oracle(partial, cache)


def test_global_oracle_rejects_overflowing_vector_norm(tmp_path, monkeypatch):
    import numpy as np

    probe, root, _ = setup_probe(tmp_path, monkeypatch)
    cache = tmp_path / "cache.sqlite"
    observation = probe.benchmark(root, cache, state="cold", details=True)
    raw = np.full(384, 1e30, dtype="<f4").tobytes()
    with sqlite3.connect(cache) as db:
        digest, spans = db.execute(
            "SELECT input_hash,spans FROM body_vectors_v1 LIMIT 1"
        ).fetchone()
        db.execute(
            "UPDATE body_vectors_v1 SET vector=?,checksum=? WHERE input_hash=?",
            (raw, hashlib.sha256(raw + spans.encode()).hexdigest(), digest),
        )
    with pytest.raises(ValueError, match="norm"):
        load_qualification().global_oracle(observation, cache)


@pytest.mark.skipif(os.name != "posix", reason="qualification mutations require POSIX descriptors")
def test_owned_source_alias_cannot_overwrite_original(tmp_path):
    qualify = load_qualification()
    original = tmp_path / "original"
    original.mkdir()
    (original / "a.py").write_bytes(b"preserve original")
    output = tmp_path / "publication"
    with qualify.OwnedWork(output) as owned:
        source = owned.path / "source"
        source.rename(owned.path / "saved")
        source.symlink_to(original, target_is_directory=True)
        with pytest.raises((OSError, ValueError), match="changed|link"):
            owned.write_source("a.py", b"replacement")
    assert (original / "a.py").read_bytes() == b"preserve original"
    assert not (output / "QUALIFICATION_COMPLETE").exists()


def load_metrics():
    path = Path(__file__).resolve().parents[3] / "scripts/body_qualification_metrics.py"
    spec = importlib.util.spec_from_file_location("body_qualification_metrics_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_metric_owner_requires_every_pair_and_active_vector(tmp_path, monkeypatch):
    probe, root, _ = setup_probe(tmp_path, monkeypatch)
    observation = probe.benchmark(root, tmp_path / "cache.sqlite", state="cold", details=True)
    metrics = load_metrics()
    for field in ("cached_vectors", "pair_scores", "requested_inputs"):
        invalid = copy.deepcopy(observation)
        invalid[field] = []
        refresh_metric_digests(invalid, metrics)
        with pytest.raises(ValueError):
            metrics.validate_observation(invalid)


def refresh_metric_digests(observation, metrics):
    observation["unit_input_hashes"] = [u["input_hash"] for u in observation["unit_manifest"]]
    observation["unit_manifest_sha256"] = metrics._digest(observation["unit_manifest"])
    observation["pair_scores_sha256"] = metrics._digest(observation["pair_scores"])
    observation["vector_manifest_sha256"] = metrics._digest(observation["cached_vectors"])


def test_duplicate_body_inputs_still_require_every_function_pair(tmp_path):
    import json

    metrics, observation, cache = control_observation(tmp_path)
    units = observation["unit_manifest"]
    units[2]["input_hash"] = units[0]["input_hash"]
    record = observation["requested_inputs"][2]
    record["input_hash"] = units[2]["input_hash"]
    record["id"] = hashlib.sha256(
        json.dumps([record["file"], 1, 0, 0, record["input_hash"]]).encode()
    ).hexdigest()
    observation["stats"]["unique_inputs"] = 2
    observation["global_result"]["stats"].update(unique_inputs=2, cache_hits=2)
    observation["pair_scores"][1]["score"] = 1.0
    observation["global_result"]["families"][0].update(matching_edges=3, score=1.0)
    refresh_metric_digests(observation, metrics)
    assert metrics.global_oracle(observation, cache)["matching_edges"] == 3
    observation["pair_scores"].pop()
    observation["pair_count"] = 2
    refresh_metric_digests(observation, metrics)
    with pytest.raises(ValueError, match="pair coverage"):
        metrics.validate_observation(observation)


def test_forged_scores_fail_dense_oracle_even_with_recomputed_digest(tmp_path):
    metrics, observation, cache = control_observation(tmp_path)
    observation["pair_scores"][0]["score"] = 0.7
    refresh_metric_digests(observation, metrics)
    metrics.validate_observation(observation)
    with pytest.raises(ValueError, match="Requested scores disagree"):
        metrics.global_oracle(observation, cache)


@pytest.mark.parametrize("kind", ["nan", "stale_location", "missing_cohort", "wrong_options"])
def test_global_certificate_rejects_invalid_public_metrics_and_cohort(tmp_path, kind):
    metrics, observation, cache = control_observation(tmp_path)
    if kind == "nan":
        observation["global_result"]["families"][0]["score"] = float("nan")
    elif kind == "stale_location":
        members = copy.deepcopy(observation["global_result"]["families"][0]["members"])
        members[0]["line"] = 99
        observation["global_result"]["families"][0]["members"] = members
    elif kind == "missing_cohort":
        observation["global_result"]["inputs"] = observation["global_result"]["inputs"][:2]
    else:
        observation["global_options"]["top_k"] = 1
    with pytest.raises(ValueError, match="global retrieval|stale|cohort|configuration"):
        metrics.global_oracle(observation, cache)


def test_unchanged_source_cannot_silently_change_declared_function_cohort(tmp_path, monkeypatch):
    probe, root, _ = setup_probe(tmp_path, monkeypatch)
    cache = tmp_path / "cache.sqlite"
    before = probe.benchmark(root, cache, state="cold", details=True)
    after = copy.deepcopy(before)
    metrics = load_metrics()
    after["unit_manifest"][0]["symbol"] = "forged_name"
    after["requested_inputs"][0]["symbol"] = "forged_name"
    after["global_result"]["inputs"][0]["symbol"] = "forged_name"
    refresh_metric_digests(after, metrics)
    metrics.validate_observation(after)
    with pytest.raises(ValueError, match="Unchanged source function cohort"):
        metrics.compare_unchanged(before, after)


@pytest.mark.parametrize("kind", ["dimension", "spans"])
def test_checksum_valid_cache_rows_still_require_declared_vector_and_window_shapes(
    tmp_path, monkeypatch, kind
):
    import json

    probe, root, _ = setup_probe(tmp_path, monkeypatch)
    cache = tmp_path / "cache.sqlite"
    observed = probe.benchmark(root, cache, state="cold", details=True)
    with sqlite3.connect(cache) as db:
        key, raw, spans = db.execute(
            "SELECT input_hash,vector,spans FROM body_vectors_v1 LIMIT 1"
        ).fetchone()
        if kind == "dimension":
            raw = raw[:-4]
        else:
            spans = json.dumps([[0, True]])
        checksum = hashlib.sha256(raw + spans.encode()).hexdigest()
        db.execute(
            "UPDATE body_vectors_v1 SET vector=?,spans=?,checksum=? WHERE input_hash=?",
            (raw, spans, checksum, key),
        )
    with pytest.raises(ValueError, match="dimension|spans"):
        load_metrics().global_oracle(observed, cache)


def test_cold_reference_checks_independent_global_oracle_and_same_source_cohort(
    tmp_path, monkeypatch
):
    probe, root, _ = setup_probe(tmp_path, monkeypatch)
    left, right = tmp_path / "left.sqlite", tmp_path / "right.sqlite"
    warm = probe.benchmark(root, left, state="cold", details=True)
    cold = probe.benchmark(root, right, state="cold", details=True)
    metrics = load_metrics()
    assert metrics.compare_cold(warm, cold, left, right)["global_families_and_edges_identical"]
    cold["pair_scores"][0]["score"] = 0.5
    refresh_metric_digests(cold, metrics)
    with pytest.raises(ValueError, match="Requested scores disagree"):
        metrics.compare_cold(warm, cold, left, right)


def test_aliasing_one_physical_function_cannot_inflate_the_global_cohort(tmp_path):
    import json

    metrics, observation, _ = control_observation(tmp_path)
    original = observation["unit_manifest"][0]
    forged = observation["unit_manifest"][1]
    for key in ("file", "line", "region_index", "source_start"):
        forged[key] = original[key]
    record = observation["requested_inputs"][1]
    record["file"] = original["file"]
    record["id"] = hashlib.sha256(
        json.dumps([record["file"], 1, 0, 0, record["input_hash"]]).encode()
    ).hexdigest()
    refresh_metric_digests(observation, metrics)
    with pytest.raises(ValueError, match="cohort"):
        metrics.validate_observation(observation)

"""Body input provenance, bounded windows, batching and cache isolation."""

import sqlite3

import pytest

pytest.importorskip("victor_codegraph.embeddings")

from victor.core.graph_rag.body_clones import BodyOptions, LocalBodyEncoder, analyze_bodies
from victor.core.graph_rag.structural_clones import scan


class Tokenizer:
    def num_special_tokens_to_add(self, pair=False):
        return 2

    def __call__(self, text, *, add_special_tokens=False, **kwargs):
        result = {"input_ids": list(range(len(text) + (2 if add_special_tokens else 0)))}
        if kwargs.get("return_offsets_mapping"):
            result["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return result


class Model:
    tokenizer = Tokenizer()
    max_seq_length = 512

    def __init__(self):
        self.calls = []

    def get_sentence_embedding_dimension(self):
        return 3

    def encode(self, texts, **kwargs):
        self.calls.extend(texts)
        return [[1, text.count("+") + 1, text.count("-") + 1] for text in texts]


def options(**kwargs):
    return BodyOptions(
        model_id="test/model", revision="0" * 40, dimension=3, max_tokens=32, overlap=4, **kwargs
    )


def extractor(root):
    for name in ("a.py", "b.py"):
        (root / name).write_text("def compute(x):\n value=x+1\n return value*3+4\n")
    return scan(root, min_nodes=5, retain_source=True)


def test_full_bodies_are_windowed_without_silent_truncation(tmp_path):
    report = extractor(tmp_path)
    model = Model()
    received = []
    encoder = LocalBodyEncoder(
        options(), model_factory=lambda *args, **kwargs: received.append(kwargs) or model
    )
    payload = analyze_bodies(report, options=options(), encoder=encoder, threshold=0.99)
    assert payload["coverage_complete"]
    assert payload["stats"]["unique_inputs"] == 1
    assert payload["stats"]["inferred_windows"] == 2
    assert payload["evidence"] == "function_body_embeddings"
    assert received[0]["local_files_only"] is True
    assert received[0]["trust_remote_code"] is False
    assert all(len(text) + 2 <= 32 for text in model.calls)
    members = payload["families"][0]["members"]
    assert all(not m["truncated"] for m in members)
    assert members[0]["window_token_spans"][0][0] == 0
    assert members[0]["window_token_spans"][-1][1] == len(report.units[0].source_code)


def test_cache_reuses_vectors_and_invalidates_changed_body(tmp_path):
    report = extractor(tmp_path)
    model = Model()
    encoder = LocalBodyEncoder(options(), model_factory=lambda *a, **k: model)
    cache = tmp_path / "analysis" / "bodies.sqlite"
    first = analyze_bodies(report, options=options(), encoder=encoder, cache_path=cache)
    second = analyze_bodies(report, options=options(), cache_path=cache)
    assert first["stats"]["inferred_windows"] == 2
    assert second["stats"]["inferred_windows"] == 0
    assert second["stats"]["cache_hits"] == 1
    (tmp_path / "b.py").write_text("def compute(x):\n value=x-1\n return value*3+4\n")
    third = analyze_bodies(
        scan(tmp_path, min_nodes=5, retain_source=True),
        options=options(),
        encoder=encoder,
        cache_path=cache,
    )
    assert third["stats"]["unique_inputs"] == 2
    assert third["stats"]["inferred_windows"] > 0
    assert any("x-1" in text for text in model.calls)


def test_invalid_inference_never_publishes_zero_vectors(tmp_path):
    report = extractor(tmp_path)
    model = Model()
    model.encode = lambda texts, **kw: [[0, 0, 0] for text in texts]
    cache = tmp_path / "bodies.sqlite"
    with pytest.raises(ValueError, match="zero vector"):
        analyze_bodies(
            report,
            options=options(),
            encoder=LocalBodyEncoder(options(), model_factory=lambda *a, **k: model),
            cache_path=cache,
        )
    with sqlite3.connect(cache) as db:
        assert db.execute("SELECT count(*) FROM body_vectors_v1").fetchone()[0] == 0


def test_window_budget_failure_is_explicit(tmp_path):
    report = extractor(tmp_path)
    limited = options(max_windows=1)
    with pytest.raises(ValueError, match="window budget"):
        analyze_bodies(
            report,
            options=limited,
            encoder=LocalBodyEncoder(limited, model_factory=lambda *a, **k: Model()),
        )


def test_cache_profile_is_separate_from_summary_embeddings(tmp_path):
    body = options()
    assert body.text_contract == "function-source-token-windows-v1"
    assert body.identity != options(max_functions=1).revision
    assert body.identity == options(max_functions=1).identity
    assert (
        body.identity
        != BodyOptions(
            model_id="test/model", revision="1" * 40, dimension=3, max_tokens=32, overlap=4
        ).identity
    )


def test_body_extraction_is_required(tmp_path):
    extractor(tmp_path)
    result = analyze_bodies(scan(tmp_path, min_nodes=5), options=options())
    assert not result["coverage_complete"]
    assert result["errors"]


def test_body_collection_does_not_load_compiler_dumps(tmp_path, monkeypatch):
    from victor.core.graph_rag import structural_clones

    extractor(tmp_path)

    def unnecessary(*args, **kwargs):
        raise AssertionError("body vectors do not consume compiler identities")

    monkeypatch.setattr(structural_clones, "load_scip_occurrences", unnecessary)
    report = scan(
        tmp_path, min_nodes=5, retain_source=True, collect_only=True, near_metric="subtree"
    )
    assert len(report.units) == 2
    assert all(not unit.subtree_hashes for unit in report.units)
    assert not report.families


def test_cache_entries_cannot_cross_model_revisions(tmp_path):
    report = extractor(tmp_path)
    cache = tmp_path / "bodies.sqlite"
    original = options()
    changed = BodyOptions(
        model_id="test/model", revision="1" * 40, dimension=3, max_tokens=32, overlap=4
    )
    for profile in (original, changed):
        model = Model()
        payload = analyze_bodies(
            report,
            options=profile,
            cache_path=cache,
            encoder=LocalBodyEncoder(profile, model_factory=lambda *a, **k: model),
        )
        assert payload["stats"]["cache_hits"] == 0
        assert model.calls
    with sqlite3.connect(cache) as db:
        assert db.execute("SELECT count(DISTINCT profile) FROM body_vectors_v1").fetchone()[0] == 2


def test_corrupt_cache_is_not_accepted_as_model_evidence(tmp_path):
    report = extractor(tmp_path)
    cache = tmp_path / "bodies.sqlite"
    analyze_bodies(
        report,
        options=options(),
        cache_path=cache,
        encoder=LocalBodyEncoder(options(), model_factory=lambda *a, **k: Model()),
    )
    with sqlite3.connect(cache) as db:
        db.execute("UPDATE body_vectors_v1 SET checksum='incorrect'")
    with pytest.raises(ValueError, match="Corrupt"):
        analyze_bodies(report, options=options(), cache_path=cache)


def test_failure_after_a_valid_batch_does_not_publish_partial_inputs(tmp_path):
    for i in range(20):
        (tmp_path / f"source_{i}.py").write_text(
            f"def compute_{i}(x):\n value=x+1\n return value*3+4\n"
        )
    report = scan(tmp_path, min_nodes=5, retain_source=True, collect_only=True)
    model = Model()
    calls = []

    def encode(texts, **kwargs):
        calls.append(len(texts))
        return [[1, 2, 3] if len(calls) == 1 else [0, 0, 0] for text in texts]

    model.encode = encode
    cache = tmp_path / "bodies.sqlite"
    with pytest.raises(ValueError, match="zero vector"):
        analyze_bodies(
            report,
            options=options(),
            cache_path=cache,
            encoder=LocalBodyEncoder(options(), model_factory=lambda *a, **k: model),
        )
    assert calls[0] == 32
    assert len(calls) == 2
    with sqlite3.connect(cache) as db:
        assert db.execute("SELECT count(*) FROM body_vectors_v1").fetchone()[0] == 0


def test_requested_pairs_keep_actual_cosine_without_family_clique(tmp_path):
    import numpy as np
    from dataclasses import replace

    report = extractor(tmp_path)
    report.units = [
        replace(report.units[0], file_path=f"{i}.py", source_code=f"body-{i}") for i in range(3)
    ]
    model = Model()
    vectors = {"body-0": [1.0, 0.0, 0.0], "body-1": [0.8, 0.6, 0.0], "body-2": [0.28, 0.96, 0.0]}
    model.encode = lambda texts, **kwargs: [vectors[text] for text in texts]
    encoder = LocalBodyEncoder(options(), model_factory=lambda *a, **k: model)
    result = analyze_bodies(
        report, options=options(), encoder=encoder, requested_pairs=[(0, 1), (1, 2), (0, 2)]
    )
    scores = [item["score"] for item in result["pair_scores"]]
    assert np.allclose(scores, [0.8, 0.8, 0.28])
    assert not result["families"]
    assert result["stats"]["retrieval"] == "requested_pair_cosine"


def test_requested_pairs_report_missing_body_and_reuse_cache(tmp_path):
    from dataclasses import replace

    report = extractor(tmp_path)
    report.units[1] = replace(report.units[1], source_code="")
    encoder = LocalBodyEncoder(options(), model_factory=lambda *a, **k: Model())
    result = analyze_bodies(
        report,
        options=options(),
        encoder=encoder,
        requested_pairs=[(0, 1)],
        cache_path=tmp_path / "body.sqlite",
    )
    assert result["pair_scores"] == [
        {"left": 0, "right": 1, "score": None, "error": "body_unavailable"}
    ]
    assert not result["coverage_complete"]
    again = analyze_bodies(
        report, options=options(), requested_pairs=[(0, 0)], cache_path=tmp_path / "body.sqlite"
    )
    assert again["pair_scores"][0]["score"] == pytest.approx(1)
    assert again["stats"]["inferred_windows"] == 0


@pytest.mark.parametrize("pairs", [[(-1, 0)], [(0, 2)], [(True, 0)], [(0,)]])
def test_requested_pair_bounds_are_checked_before_inference(tmp_path, pairs):
    with pytest.raises(ValueError, match="requested"):
        analyze_bodies(extractor(tmp_path), options=options(), requested_pairs=pairs)


@pytest.mark.parametrize("alias", ["direct", "hardlink", "symlink"])
def test_body_cache_cannot_claim_a_foreign_database(tmp_path, alias):
    import os

    report = extractor(tmp_path)
    database = tmp_path / "project.db"
    with sqlite3.connect(database) as db:
        db.execute("CREATE TABLE graph_node(node_id TEXT)")
    before = database.read_bytes()
    cache = database
    if alias != "direct":
        cache = tmp_path / "cache.sqlite"
        if alias == "hardlink":
            os.link(database, cache)
        else:
            cache.symlink_to(database)
    with pytest.raises(ValueError, match="cache"):
        analyze_bodies(report, options=options(), cache_path=cache)
    assert database.read_bytes() == before


def test_sqlite_like_wildcards_cannot_hide_foreign_tables(tmp_path):
    database = tmp_path / "cache.sqlite"
    with sqlite3.connect(database) as db:
        db.execute("CREATE TABLE sqliteHiddenGraph(value)")
    before = database.read_bytes()
    with pytest.raises(ValueError, match="unrelated SQLite"):
        analyze_bodies(extractor(tmp_path), options=options(), cache_path=database)
    assert database.read_bytes() == before

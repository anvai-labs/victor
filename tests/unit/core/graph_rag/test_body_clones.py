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
    encoder = LocalBodyEncoder(
        options(), model_factory=lambda *a, **k: model, producer_id="tests/character-v1"
    )
    cache = tmp_path / "analysis" / "bodies.sqlite"
    first = analyze_bodies(report, options=options(), encoder=encoder, cache_path=cache)
    second = analyze_bodies(
        report,
        options=options(),
        cache_path=cache,
        encoder=LocalBodyEncoder(
            options(), model_factory=lambda *a, **k: Model(), producer_id="tests/character-v1"
        ),
    )
    assert first["stats"]["inferred_windows"] == 2
    assert second["stats"]["inferred_windows"] == 0
    assert second["stats"]["cache_hits"] == 1
    assert second["stats"]["model_load_attempts"] == 0
    assert second["stats"]["model_load_failed"] is False
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


def test_cached_body_vectors_invalidate_after_runtime_generation_changes(tmp_path, monkeypatch):
    from victor.core.graph_rag import body_provenance

    report = extractor(tmp_path)
    path = tmp_path / "bodies.sqlite"
    model = Model()
    monkeypatch.setattr(body_provenance, "library_versions", lambda: {"torch": "old"})
    first = analyze_bodies(
        report,
        options=options(),
        cache_path=path,
        encoder=LocalBodyEncoder(
            options(), model_factory=lambda *a, **k: model, producer_id="tests/character-v1"
        ),
    )
    monkeypatch.setattr(body_provenance, "library_versions", lambda: {"torch": "new"})
    second = analyze_bodies(
        report,
        options=options(),
        cache_path=path,
        encoder=LocalBodyEncoder(
            options(), model_factory=lambda *a, **k: Model(), producer_id="tests/character-v1"
        ),
    )
    assert second["stats"]["cache_hits"] == 0
    assert second["stats"]["inferred_windows"] > 0
    assert first["profile"]["producer"]["identity"] != second["profile"]["producer"]["identity"]


def test_undeclared_factories_never_read_another_encoder_instances_cache(tmp_path):
    report = extractor(tmp_path)
    path = tmp_path / "bodies.sqlite"
    first = analyze_bodies(
        report,
        options=options(),
        cache_path=path,
        encoder=LocalBodyEncoder(options(), model_factory=lambda *a, **k: Model()),
    )
    second = analyze_bodies(
        report,
        options=options(),
        cache_path=path,
        encoder=LocalBodyEncoder(options(), model_factory=lambda *a, **k: Model()),
    )
    assert second["stats"]["cache_hits"] == 0
    assert first["profile"]["producer"]["cache_reuse_scope"] == "encoder_instance"


def test_subclass_state_is_not_assumed_to_be_the_standard_hub_producer():
    class CustomEncoder(LocalBodyEncoder):
        pass

    first, second = CustomEncoder(options()), CustomEncoder(options())
    assert first.cache_producer.reuse_scope == "encoder_instance"
    assert first.cache_producer.identity != second.cache_producer.identity


def test_instance_method_overrides_do_not_share_standard_producer_cache():
    from types import MethodType

    def encode(self, texts):
        return [self.vector for _ in texts]

    first, second = LocalBodyEncoder(options()), LocalBodyEncoder(options())
    for encoder, vector in ((first, [1, 0, 0]), (second, [0, 1, 0])):
        encoder.vector = vector
        encoder.encode = MethodType(encode, encoder)
    assert first.cache_producer.reuse_scope == "encoder_instance"
    assert first.cache_producer.identity != second.cache_producer.identity


def test_missing_framework_source_is_instance_scoped_even_with_declared_model(
    tmp_path, monkeypatch
):
    from victor.core.graph_rag import body_clones

    def unavailable(*args):
        raise OSError("no source")

    monkeypatch.setattr(body_clones.inspect, "getsource", unavailable)
    report = extractor(tmp_path)
    cache = tmp_path / "cache.sqlite"

    def encoder():
        return LocalBodyEncoder(
            options(), model_factory=lambda *a, **k: Model(), producer_id="tests/character-v1"
        )

    first = analyze_bodies(report, options=options(), encoder=encoder(), cache_path=cache)
    second = analyze_bodies(report, options=options(), encoder=encoder(), cache_path=cache)
    assert second["stats"]["cache_hits"] == 0
    assert first["profile"]["producer"]["cache_reuse_scope"] == "encoder_instance"
    assert first["profile"]["producer"]["custom_id"] == "tests/character-v1"


@pytest.mark.parametrize(
    "field,value", [("options", options()), ("model", Model()), ("model_factory", Model)]
)
def test_encoder_configuration_cannot_be_replaced_after_provenance_capture(field, value):
    encoder = LocalBodyEncoder(options())
    encoder.cache_producer
    with pytest.raises(AttributeError):
        setattr(encoder, field, value)


def test_producer_must_be_bound_to_the_requested_input_profile(tmp_path):
    from victor.core.graph_rag.body_provenance import producer_profile

    class WrongProducer(LocalBodyEncoder):
        @property
        def cache_producer(self):
            return producer_profile("f" * 64, component_hash="e" * 64)

    with pytest.raises(ValueError, match="producer input profile"):
        analyze_bodies(extractor(tmp_path), options=options(), encoder=WrongProducer(options()))


def test_encoder_without_provenance_gets_an_analysis_scoped_cache_identity(tmp_path):
    options_ = options()
    wrapped = LocalBodyEncoder(options_, model_factory=lambda *a, **k: Model())

    class UndeclaredEncoder:
        options = options_
        windows = wrapped.windows
        encode = wrapped.encode

    report = extractor(tmp_path)
    cache = tmp_path / "cache.sqlite"
    first = analyze_bodies(report, options=options_, encoder=UndeclaredEncoder(), cache_path=cache)
    second = analyze_bodies(report, options=options_, encoder=UndeclaredEncoder(), cache_path=cache)
    assert second["stats"]["cache_hits"] == 0
    assert first["profile"]["producer"]["identity"] != second["profile"]["producer"]["identity"]


def test_changed_pooling_invalidates_a_long_lived_encoders_cache(tmp_path, monkeypatch):
    import numpy as np
    from victor.core.graph_rag import body_clones

    report = extractor(tmp_path)
    cache = tmp_path / "bodies.sqlite"
    encoder = LocalBodyEncoder(
        options(), model_factory=lambda *a, **k: Model(), producer_id="tests/character-v1"
    )
    before = analyze_bodies(report, options=options(), cache_path=cache, encoder=encoder)
    original = body_clones._pool_body_windows

    def changed(windows):
        return original(windows) * np.asarray([-1, 1, 1])

    monkeypatch.setattr(body_clones, "_pool_body_windows", changed)
    after = analyze_bodies(report, options=options(), cache_path=cache, encoder=encoder)
    assert after["stats"]["cache_hits"] == 0
    assert after["stats"]["inferred_windows"] > 0
    assert before["profile"]["producer"]["identity"] != after["profile"]["producer"]["identity"]


@pytest.mark.parametrize("wrapped", [False, True])
def test_validator_generation_cannot_mix_cached_and_changed_body_vectors(
    tmp_path, monkeypatch, wrapped
):
    from functools import wraps
    import victor_codegraph.embeddings as embeddings

    report = extractor(tmp_path)
    options_ = options()

    def run(report, cache):
        return analyze_bodies(
            report,
            options=options_,
            cache_path=cache,
            requested_pairs=[(0, 1)],
            encoder=LocalBodyEncoder(
                options_, model_factory=lambda *a, **k: Model(), producer_id="tests/character-v1"
            ),
        )

    cache = tmp_path / "warm.sqlite"
    before = run(report, cache)
    original = embeddings.validate_vectors

    def changed(*args, **kwargs):
        return [[-row[0], *row[1:]] for row in original(*args, **kwargs)]

    if wrapped:
        changed = wraps(original)(changed)
    monkeypatch.setattr(embeddings, "validate_vectors", changed)
    (tmp_path / "b.py").write_text("def compute(x):\n value=x-1\n return value*3-4\n")
    report = scan(tmp_path, min_nodes=5, retain_source=True)
    mixed = run(report, cache)
    cold = run(report, tmp_path / "cold.sqlite")
    assert mixed["stats"]["cache_hits"] == 0
    assert mixed["pair_scores"] == cold["pair_scores"]
    assert before["profile"]["producer"]["identity"] != mixed["profile"]["producer"]["identity"]


def test_producer_changed_during_inference_never_publishes_vectors(tmp_path, monkeypatch):
    from victor.core.graph_rag import body_clones

    report = extractor(tmp_path)
    cache = tmp_path / "cache.sqlite"
    original = body_clones._pool_body_windows
    model = Model()
    encode = model.encode

    def changed(windows):
        return -original(windows)

    def infer(texts, **kwargs):
        monkeypatch.setattr(body_clones, "_pool_body_windows", changed)
        return encode(texts, **kwargs)

    model.encode = infer
    with pytest.raises(ValueError, match="producer changed"):
        analyze_bodies(
            report,
            options=options(),
            cache_path=cache,
            encoder=LocalBodyEncoder(options(), model_factory=lambda *a, **k: model),
        )
    with sqlite3.connect(cache) as db:
        assert db.execute("SELECT count(*) FROM body_vectors_v1").fetchone()[0] == 0


def test_legacy_input_only_cache_rows_are_not_relabelled_as_current_producer(tmp_path):
    import hashlib
    import json
    import numpy as np

    report = extractor(tmp_path)
    path = tmp_path / "bodies.sqlite"
    vector = np.asarray([1, 0, 0], dtype="<f4").tobytes()
    spans = json.dumps([[0, len(report.units[0].source_code)]])
    digest = hashlib.sha256(report.units[0].source_code.encode()).hexdigest()
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE body_vectors_v1 (profile TEXT, input_hash TEXT, vector BLOB, spans TEXT, checksum TEXT, PRIMARY KEY(profile,input_hash))"
        )
        db.execute(
            "INSERT INTO body_vectors_v1 VALUES(?,?,?,?,?)",
            (
                options().identity,
                digest,
                vector,
                spans,
                hashlib.sha256(vector + spans.encode()).hexdigest(),
            ),
        )
    result = analyze_bodies(
        report,
        options=options(),
        cache_path=path,
        encoder=LocalBodyEncoder(
            options(), model_factory=lambda *a, **k: Model(), producer_id="tests/character-v1"
        ),
    )
    assert result["stats"]["cache_hits"] == 0
    assert result["stats"]["inferred_windows"] > 0
    with sqlite3.connect(path) as db:
        assert (
            db.execute(
                "SELECT count(*) FROM body_vectors_v1 WHERE profile=?", (options().identity,)
            ).fetchone()[0]
            == 1
        )


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
        encoder=LocalBodyEncoder(
            options(), model_factory=lambda *a, **k: Model(), producer_id="tests/character-v1"
        ),
    )
    with sqlite3.connect(cache) as db:
        db.execute("UPDATE body_vectors_v1 SET checksum='incorrect'")
    with pytest.raises(ValueError, match="Corrupt"):
        analyze_bodies(
            report,
            options=options(),
            cache_path=cache,
            encoder=LocalBodyEncoder(
                options(), model_factory=lambda *a, **k: Model(), producer_id="tests/character-v1"
            ),
        )


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
    encoder = LocalBodyEncoder(
        options(), model_factory=lambda *a, **k: Model(), producer_id="tests/character-v1"
    )
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
        report,
        options=options(),
        requested_pairs=[(0, 0)],
        cache_path=tmp_path / "body.sqlite",
        encoder=LocalBodyEncoder(
            options(), model_factory=lambda *a, **k: Model(), producer_id="tests/character-v1"
        ),
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


def test_invalid_model_profile_constructs_once_per_encoder(tmp_path):
    model = Model()
    loads = []
    bad_options = BodyOptions(
        model_id="test/model", revision="0" * 40, dimension=4, max_tokens=32, overlap=4
    )
    encoder = LocalBodyEncoder(bad_options, model_factory=lambda *a, **k: loads.append(1) or model)
    result = analyze_bodies(extractor(tmp_path), options=bad_options, encoder=encoder)
    assert not result["coverage_complete"]
    assert len(result["errors"]) == 2
    assert len(loads) == 1
    assert result["stats"]["model_load_attempts"] == 1
    assert result["stats"]["model_load_failed"] is True
    assert all("dimension" in error["reason"] for error in result["errors"])


@pytest.mark.parametrize("failure", ["dimension", "context", "missing_metadata"])
def test_failed_load_is_latched_without_replaying_the_factory(failure):
    model = Model()
    loads = []
    if failure == "dimension":
        model.get_sentence_embedding_dimension = lambda: 4
    elif failure == "context":
        model.max_seq_length = 8
    else:
        model.get_sentence_embedding_dimension = None
    encoder = LocalBodyEncoder(options(), model_factory=lambda *a, **k: loads.append(1) or model)
    for source in ("first", "second", "third"):
        with pytest.raises(ValueError):
            encoder.windows(source)
    assert len(loads) == 1


def test_fresh_encoder_can_retry_after_model_metadata_is_repaired():
    model = Model()
    model.max_seq_length = 8

    def factory(*args, **kwargs):
        return model

    failed = LocalBodyEncoder(options(), model_factory=factory)
    with pytest.raises(ValueError, match="context"):
        failed.windows("source")
    model.max_seq_length = 512
    with pytest.raises(ValueError, match="context"):
        failed.windows("source")
    fresh = LocalBodyEncoder(options(), model_factory=factory)
    assert fresh.windows("source")[0] == ["source"]


@pytest.mark.parametrize("error", [ImportError, OSError, RuntimeError, ValueError])
def test_constructor_failures_are_bounded_per_encoder(error):
    attempts = []

    def factory(*args, **kwargs):
        attempts.append(1)
        raise error("unavailable local model")

    encoder = LocalBodyEncoder(options(), model_factory=factory)
    for _ in range(3):
        with pytest.raises(error, match="unavailable local model"):
            encoder.windows("source")
    assert len(attempts) == 1


def test_cancelled_model_load_is_not_latched():
    attempts = []

    def factory(*args, **kwargs):
        attempts.append(1)
        raise KeyboardInterrupt

    encoder = LocalBodyEncoder(options(), model_factory=factory)
    for _ in range(2):
        with pytest.raises(KeyboardInterrupt):
            encoder.windows("source")
    assert len(attempts) == 2
    assert not encoder.load_failed

"""One semantic input/provenance contract for local and hosted consumers."""

from dataclasses import replace
import pytest
from victor_codegraph import parse
from victor_codegraph.embeddings import (
    BGE_SMALL_PROFILE,
    embedding_text,
    embed_symbols,
    validate_vectors,
)


def test_same_symbols_produce_store_independent_embedding_evidence():
    symbols = parse(
        'def pay(x):\n    """Charge a customer."""\n    return x\n', file_path="billing.py"
    ).symbols
    seen = []

    def encode(texts):
        seen.extend(texts)
        return [[1.0] + [0.0] * 383 for _ in texts]

    documents = embed_symbols(symbols, encode, BGE_SMALL_PROFILE)
    assert seen == [embedding_text(s.simple_name, s.signature, s.documentation) for s in symbols]
    assert [d.symbol_id for d in documents] == [s.id for s in symbols]
    assert all(d.profile_id == BGE_SMALL_PROFILE.identity for d in documents)
    assert all(len(d.input_hash) == 64 and len(d.source_hash) == 64 for d in documents)
    assert all(len(d.values) == 384 for d in documents)


@pytest.mark.parametrize(
    "vectors",
    [[], [[0.0, 0.0]], [[True, 1.0]], [[float("nan"), 1.0]], [[1.0]], [[float("inf"), 1.0]]],
)
def test_invalid_or_missing_vectors_fail_without_substitution(vectors):
    with pytest.raises(ValueError):
        validate_vectors(vectors, count=1, dimension=2)


def test_model_preprocessing_changes_invalidate_profile_identity():
    p = BGE_SMALL_PROFILE
    assert (
        len(
            {
                p.identity,
                replace(p, revision="a" * 40).identity,
                replace(p, normalize=True).identity,
                replace(p, query_prefix="query: ").identity,
                replace(p, max_tokens=256).identity,
            }
        )
        == 5
    )
    with pytest.raises(ValueError, match="revision"):
        replace(p, revision="main")


def test_failed_encoder_propagates_and_duplicate_symbol_ids_are_rejected():
    symbols = parse("def f(): pass", file_path="m.py").symbols

    def fail(texts):
        raise RuntimeError("model unavailable")

    with pytest.raises(RuntimeError, match="model unavailable"):
        embed_symbols(symbols, fail, BGE_SMALL_PROFILE)
    with pytest.raises(ValueError, match="duplicate"):
        embed_symbols(symbols * 2, fail, BGE_SMALL_PROFILE)


def test_unbounded_encoder_outputs_are_rejected_after_one_overflow_item():
    from itertools import repeat

    with pytest.raises(ValueError, match="dimension"):
        validate_vectors([repeat(1.0)], count=1, dimension=2)
    with pytest.raises(ValueError, match="exactly one"):
        validate_vectors(repeat([1.0, 2.0]), count=1, dimension=2)


def test_optional_encoder_owns_one_pinned_model_and_query_preprocessing(monkeypatch):
    import asyncio
    import sys
    from types import SimpleNamespace
    from victor_codegraph.embeddings import SentenceTransformerEncoder

    calls = []

    class Model:
        max_seq_length = 0

        def __init__(self, model_id, **kwargs):
            calls.append((model_id, kwargs))

        def get_sentence_embedding_dimension(self):
            return 3

        def encode(self, texts, **kwargs):
            calls.append((texts, kwargs, self.max_seq_length))
            return [[1.0, 2.0, 3.0] for _ in texts]

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=Model)
    )
    profile = replace(BGE_SMALL_PROFILE, dimension=3, query_prefix="query: ", max_tokens=256)
    encoder = SentenceTransformerEncoder(profile)
    assert encoder.encode([]) == [] and calls == []
    assert encoder.encode_query("permissions") == [1.0, 2.0, 3.0]
    assert asyncio.run(encoder.embed_text("symbol")) == [1.0, 2.0, 3.0]
    assert len(calls) == 3  # one model load, two bounded inference calls
    assert calls[0] == (
        profile.model_id,
        {"revision": profile.revision, "device": "cpu", "trust_remote_code": False},
    )
    assert calls[1][0] == ["query: permissions"]
    assert calls[1][1]["normalize_embeddings"] is False
    assert calls[1][2] == 256
    with pytest.raises(ValueError, match="512"):
        encoder.encode(["x"] * 513)
    with pytest.raises(ValueError, match="dimension"):
        SentenceTransformerEncoder(BGE_SMALL_PROFILE).encode(["x"])


def test_optional_encoder_missing_dependency_is_explicit(monkeypatch):
    import sys
    from victor_codegraph.embeddings import SentenceTransformerEncoder

    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(ImportError):
        SentenceTransformerEncoder().encode(["symbol"])


@pytest.mark.parametrize(
    "changes",
    [
        {"model_id": ""},
        {"revision": "main"},
        {"dimension": True},
        {"max_tokens": 0},
        {"text_contract": "body-v1"},
        {"normalize": 1},
    ],
)
def test_invalid_profile_cannot_claim_semantic_compatibility(changes):
    with pytest.raises(ValueError):
        replace(BGE_SMALL_PROFILE, **changes)

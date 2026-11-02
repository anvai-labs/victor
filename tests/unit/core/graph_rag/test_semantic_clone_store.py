"""Semantic analysis accepts compatible evidence and filters tests consistently."""

import json

import pytest

from victor_codegraph.embeddings import EmbeddingProfile
from victor.core.graph_rag.semantic_clones import load_embedding_symbols

lancedb = pytest.importorskip("lancedb")
pa = pytest.importorskip("pyarrow")


def _store(root, *, mixed=False, zero=False, evidence=True, profile_document=None):
    profile = EmbeddingProfile("test", "a" * 40, 2)
    rows = []
    for i, (path, content) in enumerate(
        [
            ("src/a.rs", "run ()"),
            ("tests/a.rs", "run ()"),
            ("src/b.rs", "test_stats #[cfg(test)] ()"),
            ("src/testdata.rs", "run ()"),
            ("src/prefix_match/a.rs", "run ()"),
            ("src/prefixXmatch/a.rs", "run ()"),
        ]
    ):
        rows.append(
            {
                "id": str(i),
                "file_path": path,
                "symbol_name": "run",
                "line_number": 1,
                "content": content,
                "vector": [0.0, 0.0] if zero else [1.0, 0.0],
                "embedding_kind": "structural" if mixed and i == 0 else "semantic",
                "embedding_profile_id": profile.identity,
                "embedding_input_hash": "b" * 64,
            }
        )
    table = pa.Table.from_pylist(rows)
    table = table.set_column(
        table.schema.get_field_index("vector"),
        "vector",
        pa.array([r["vector"] for r in rows], type=pa.list_(pa.float32(), 2)),
    )
    if evidence:
        from dataclasses import asdict

        table = table.replace_schema_metadata(
            {
                b"victor_codegraph.embedding_profile": json.dumps(
                    asdict(profile) if profile_document is None else profile_document
                ).encode()
            }
        )
    lancedb.connect(str(root / ".victor" / "embeddings")).create_table("embeddings", table)


def test_root_and_inline_tests_are_excluded_and_vectors_align(tmp_path):
    _store(tmp_path)
    records, matrix = load_embedding_symbols(tmp_path)
    assert {r.file_path for r in records} == {
        "src/a.rs",
        "src/testdata.rs",
        "src/prefix_match/a.rs",
        "src/prefixXmatch/a.rs",
    }
    assert matrix.shape == (4, 2)
    assert matrix[:, 0].tolist() == [1.0] * 4
    assert len(load_embedding_symbols(tmp_path, include_tests=True)[0]) == 6
    assert len(load_embedding_symbols(tmp_path, scope="src/prefix_match/")[0]) == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mixed": True},
        {"zero": True},
        {"evidence": False},
        {"profile_document": []},
        {"profile_document": {"unknown": "field"}},
    ],
)
def test_invalid_or_unknown_evidence_is_rejected(tmp_path, kwargs):
    _store(tmp_path, **kwargs)
    with pytest.raises(ValueError):
        load_embedding_symbols(tmp_path)

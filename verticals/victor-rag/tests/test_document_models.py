"""Document model types have exactly one definition.

``document_store`` historically defined Document/DocumentChunk/
DocumentSearchResult/DocumentStoreConfig, and ``document_models`` held an
identical, never-imported copy. Both import paths must keep working and
must yield the SAME classes (isinstance across paths), with models
importable without touching the storage backend.
"""

import subprocess
import sys
from pathlib import Path

import pytest

NAMES = ("Document", "DocumentChunk", "DocumentSearchResult", "DocumentStoreConfig")


@pytest.mark.parametrize("name", NAMES)
def test_store_and_models_export_the_same_class(name):
    from victor_rag import document_models, document_store

    assert getattr(document_store, name) is getattr(document_models, name)


def test_chunk_from_one_path_is_instance_of_the_other():
    from victor_rag.document_models import DocumentChunk as ModelChunk
    from victor_rag.document_store import DocumentChunk as StoreChunk

    chunk = StoreChunk(id="c", doc_id="d", content="x", embedding=[0.0])
    assert isinstance(chunk, ModelChunk)


def test_model_behaviour_preserved():
    from victor_rag.document_store import (
        Document,
        DocumentChunk,
        DocumentSearchResult,
        DocumentStoreConfig,
    )

    doc = Document(id="1", content="hello", source="a.md")
    assert doc.doc_type == "text" and doc.metadata == {}
    assert doc.content_hash == "2cf24dba5fb0a30e"  # sha256("hello")[:16]
    chunk = DocumentChunk(id="c", doc_id="1", content="body", embedding=[1.0], metadata={"k": 1})
    result = DocumentSearchResult(chunk=chunk, score=0.5)
    assert (result.content, result.doc_id, result.metadata) == ("body", "1", {"k": 1})
    cfg = DocumentStoreConfig()
    assert (cfg.table_name, cfg.embedding_dim, str(cfg.path)) == ("documents", 384, ".victor/rag")


def test_models_import_without_storage_backend():
    package_root = Path(__file__).resolve().parents[1]
    script = """
import sys
sys.path.insert(0, sys.argv[1])
import victor_rag.document_models
for name in ('lancedb', 'pyarrow'):
    assert name not in sys.modules, name
"""
    subprocess.run([sys.executable, "-c", script, str(package_root)], check=True)

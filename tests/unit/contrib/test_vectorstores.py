# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for victor.contrib.vectorstores package."""

import pytest

from victor.contrib.vectorstores import InMemoryVectorStore


class TestInMemoryVectorStore:
    """Test InMemoryVectorStore implementation."""

    def test_store_info(self) -> None:
        """Test store metadata retrieval."""
        store = InMemoryVectorStore()
        info = store.get_store_info()

        assert info["type"] == "in-memory"
        assert info["document_count"] == 0
        assert info["persistence"] is False
        assert "note" in info["info"]

    @pytest.mark.asyncio
    async def test_add_documents(self) -> None:
        """Test adding documents to the store."""
        store = InMemoryVectorStore()
        documents = ["doc1", "doc2"]
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

        doc_ids = await store.add_documents(documents, embeddings)

        assert len(doc_ids) == 2
        assert all(doc_id.startswith("doc_") for doc_id in doc_ids)

    @pytest.mark.asyncio
    async def test_add_documents_with_metadata(self) -> None:
        """Test adding documents with metadata."""
        store = InMemoryVectorStore()
        documents = ["doc1"]
        embeddings = [[0.1, 0.2]]
        metadata = [{"category": "test", "id": 1}]

        doc_ids = await store.add_documents(documents, embeddings, metadata)

        assert len(doc_ids) == 1
        info = store.get_store_info()
        assert info["document_count"] == 1

    @pytest.mark.asyncio
    async def test_add_documents_mismatch_raises(self) -> None:
        """Test that mismatched documents and embeddings raise error."""
        store = InMemoryVectorStore()
        documents = ["doc1", "doc2"]
        embeddings = [[0.1, 0.2]]  # Only one embedding

        with pytest.raises(ValueError, match="must match"):
            await store.add_documents(documents, embeddings)

    @pytest.mark.asyncio
    async def test_search_empty_store(self) -> None:
        """Test searching an empty store."""
        store = InMemoryVectorStore()
        query = [0.1, 0.2, 0.3]

        results = await store.search(query, top_k=5)

        assert results == []

    @pytest.mark.asyncio
    async def test_search(self) -> None:
        """Test searching for similar documents."""
        store = InMemoryVectorStore()
        documents = ["apple pie", "car engine", "banana split"]
        embeddings = [
            [1.0, 0.0, 0.0],  # apple
            [0.0, 1.0, 0.0],  # car
            [0.9, 0.0, 0.1],  # banana (similar to apple)
        ]
        await store.add_documents(documents, embeddings)

        query = [1.0, 0.0, 0.0]  # Similar to apple
        results = await store.search(query, top_k=2)

        assert len(results) == 2
        # First result should be "apple pie" (highest similarity)
        assert results[0].text == "apple pie"
        assert results[0].score > results[1].score

    @pytest.mark.asyncio
    async def test_search_with_metadata_filter(self) -> None:
        """Test searching with metadata filter."""
        store = InMemoryVectorStore()
        documents = ["doc1", "doc2", "doc3"]
        embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        metadata = [
            {"category": "A"},
            {"category": "B"},
            {"category": "A"},
        ]
        await store.add_documents(documents, embeddings, metadata)

        query = [0.1, 0.2]
        results = await store.search(
            query,
            top_k=10,
            filter_metadata={"category": "A"},
        )

        # Should only return documents with category "A"
        assert len(results) == 2
        assert all(r.metadata.get("category") == "A" for r in results)

    @pytest.mark.asyncio
    async def test_search_top_k(self) -> None:
        """Test top_k parameter limits results."""
        store = InMemoryVectorStore()
        documents = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        embeddings = [[0.1 * i, 0.2 * i] for i in range(5)]
        await store.add_documents(documents, embeddings)

        query = [0.1, 0.2]
        results = await store.search(query, top_k=3)

        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_delete(self) -> None:
        """Test deleting documents."""
        store = InMemoryVectorStore()
        documents = ["doc1", "doc2", "doc3"]
        embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

        doc_ids = await store.add_documents(documents, embeddings)
        deleted = await store.delete(doc_ids[:2])

        assert deleted is True

        # Search should only find the remaining document
        results = await store.search([0.5, 0.6], top_k=10)
        assert len(results) == 1
        assert results[0].text == "doc3"

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self) -> None:
        """Test deleting non-existent documents doesn't raise error."""
        store = InMemoryVectorStore()

        deleted = await store.delete(["nonexistent_id"])
        assert deleted is True

    @pytest.mark.asyncio
    async def test_search_result_structure(self) -> None:
        """Test that search results have correct structure."""
        store = InMemoryVectorStore()
        documents = ["test doc"]
        embeddings = [[0.1, 0.2, 0.3]]
        metadata = {"key": "value"}

        doc_ids = await store.add_documents(documents, embeddings, [metadata])
        results = await store.search([0.1, 0.2, 0.3], top_k=1)

        assert len(results) == 1
        result = results[0]
        assert result.document_id == doc_ids[0]
        assert result.text == "test doc"
        assert isinstance(result.score, float)
        assert result.metadata == metadata

    @pytest.mark.asyncio
    async def test_cosine_similarity_identical(self) -> None:
        """Test cosine similarity for identical vectors."""
        store = InMemoryVectorStore()
        documents = ["test"]
        embeddings = [[0.5, 0.5, 0.5, 0.5]]
        await store.add_documents(documents, embeddings)

        # Query with same vector should give score of 1.0
        results = await store.search([0.5, 0.5, 0.5, 0.5], top_k=1)

        assert len(results) == 1
        assert abs(results[0].score - 1.0) < 0.001

    @pytest.mark.asyncio
    async def test_cosine_similarity_orthogonal(self) -> None:
        """Test cosine similarity for orthogonal vectors."""
        store = InMemoryVectorStore()
        documents = ["test"]
        embeddings = [[1.0, 0.0, 0.0]]
        await store.add_documents(documents, embeddings)

        # Query with orthogonal vector should give score of 0.0
        results = await store.search([0.0, 1.0, 0.0], top_k=1)

        assert len(results) == 1
        assert abs(results[0].score - 0.0) < 0.001

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

"""Unit tests for victor.contrib.embeddings package."""

import pytest

from victor.contrib.embeddings import BasicEmbeddingsProvider


class TestBasicEmbeddingsProvider:
    """Test BasicEmbeddingsProvider implementation."""

    def test_provider_info(self) -> None:
        """Test provider metadata retrieval."""
        embeddings = BasicEmbeddingsProvider()
        info = embeddings.get_model_info()

        assert info["name"] == "basic-hash"
        assert info["dimension"] == 384
        assert info["type"] == "hash-based"
        assert "note" in info["info"]

    def test_custom_dimension(self) -> None:
        """Test custom embedding dimension."""
        embeddings = BasicEmbeddingsProvider(dimension=128)
        assert embeddings.get_dimension() == 128

    @pytest.mark.asyncio
    async def test_embed_text(self) -> None:
        """Test embedding a single text."""
        embeddings = BasicEmbeddingsProvider(dimension=128)
        vector = await embeddings.embed_text("Hello world")

        assert len(vector) == 128
        assert all(isinstance(v, float) for v in vector)
        # Values should be in [-1, 1] range
        assert all(-1 <= v <= 1 for v in vector)

    @pytest.mark.asyncio
    async def test_embed_text_deterministic(self) -> None:
        """Test that embeddings are deterministic for same text."""
        embeddings = BasicEmbeddingsProvider(dimension=64)

        vector1 = await embeddings.embed_text("Same text")
        vector2 = await embeddings.embed_text("Same text")

        assert vector1 == vector2

    @pytest.mark.asyncio
    async def test_embed_text_different(self) -> None:
        """Test that different texts produce different embeddings."""
        embeddings = BasicEmbeddingsProvider(dimension=64)

        vector1 = await embeddings.embed_text("Text one")
        vector2 = await embeddings.embed_text("Text two")

        assert vector1 != vector2

    @pytest.mark.asyncio
    async def test_embed_batch(self) -> None:
        """Test embedding multiple texts."""
        embeddings = BasicEmbeddingsProvider(dimension=64)
        texts = ["text1", "text2", "text3"]

        vectors = await embeddings.embed_batch(texts)

        assert len(vectors) == 3
        assert all(len(v) == 64 for v in vectors)
        assert all(isinstance(v, list) for v in vectors)

    @pytest.mark.asyncio
    async def test_embed_batch_empty(self) -> None:
        """Test embedding empty list."""
        embeddings = BasicEmbeddingsProvider()
        vectors = await embeddings.embed_batch([])

        assert vectors == []

    @pytest.mark.asyncio
    async def test_embed_single_vs_batch(self) -> None:
        """Test that batch produces same results as individual calls."""
        embeddings = BasicEmbeddingsProvider(dimension=32)
        texts = ["hello", "world"]

        # Individual calls
        vec1 = await embeddings.embed_text(texts[0])
        vec2 = await embeddings.embed_text(texts[1])

        # Batch call
        batch_vecs = await embeddings.embed_batch(texts)

        assert vec1 == batch_vecs[0]
        assert vec2 == batch_vecs[1]

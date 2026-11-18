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

"""Chunk-source intent-tier embedding contract: same staleness discipline
as SymbolEmbedding, distinct profile identity, offline fake encoder."""

import pytest

from victor_codegraph import BGE_SMALL_PROFILE, CHUNK_SOURCE_PROFILE, chunk_path
from victor_codegraph.embeddings import embed_chunks

SOURCE = (
    "def rate_limiter():\n"
    "    '''Token bucket.'''\n"
    "    bucket = 10\n"
    "    return bucket\n"
    "\n"
    "\n"
    "def unrelated():\n"
    "    return 42\n"
)


def _write(tmp_path, text):
    (tmp_path / "svc.py").write_text(text)
    return tmp_path / "svc.py"


def _chunks(tmp_path):
    return chunk_path(_write(tmp_path, SOURCE), repo_root=tmp_path)


def _encode(seen=None):
    def encode(texts):
        if seen is not None:
            seen.extend(texts)
        return [[1.0] + [0.0] * 383 for _ in texts]

    return encode


class TestEmbedChunks:
    def test_chunks_embedded_with_deterministic_identity(self, tmp_path):
        chunks = _chunks(tmp_path)
        assert chunks, "chunker must produce chunks for the fixture"
        docs = embed_chunks(chunks, _encode(), CHUNK_SOURCE_PROFILE)
        assert [d.chunk_id for d in docs] == [c.chunk_id for c in chunks]
        assert [d.symbol_id for d in docs] == [c.symbol_id for c in chunks]
        assert all(d.profile_id == CHUNK_SOURCE_PROFILE.identity for d in docs)
        assert all(len(d.input_hash) == 64 and len(d.source_hash) == 64 for d in docs)
        assert all(len(d.values) == 384 for d in docs)

    def test_input_hash_is_text_deterministic(self, tmp_path):
        chunks = _chunks(tmp_path)
        d1 = embed_chunks(chunks, _encode(), CHUNK_SOURCE_PROFILE)
        d2 = embed_chunks(chunks, _encode(), CHUNK_SOURCE_PROFILE)
        assert [d.input_hash for d in d1] == [d.input_hash for d in d2]
        mutated = embed_chunks(
            chunk_path(
                _write(tmp_path, SOURCE.replace("bucket = 10", "bucket = 11")),
                repo_root=tmp_path,
            ),
            _encode(),
            CHUNK_SOURCE_PROFILE,
        )
        assert mutated[0].input_hash != d1[0].input_hash

    def test_source_hash_propagates_from_manifest(self, tmp_path):
        chunks = _chunks(tmp_path)
        docs = embed_chunks(
            chunks,
            _encode(),
            CHUNK_SOURCE_PROFILE,
            source_hashes={"svc.py": "abc"},
        )
        assert all(d.source_hash == "abc" for d in docs)  # manifest hash stored verbatim

    def test_profile_identity_differs_from_symbol_summary_tier(self):
        assert CHUNK_SOURCE_PROFILE.identity != BGE_SMALL_PROFILE.identity

    def test_duplicate_chunk_id_rejected(self, tmp_path):
        chunks = _chunks(tmp_path)
        doubled = list(chunks) + [chunks[0]]
        with pytest.raises(ValueError, match="duplicate"):
            embed_chunks(doubled, _encode(), CHUNK_SOURCE_PROFILE)

    def test_batch_validation_propagates(self, tmp_path):
        chunks = _chunks(tmp_path)

        def bad_encode(texts):
            return [[1.0] for _ in texts]  # wrong dimension

        with pytest.raises(ValueError, match="dimension"):
            embed_chunks(chunks, bad_encode, CHUNK_SOURCE_PROFILE)

    def test_batch_size_respected(self, tmp_path):
        chunks = _chunks(tmp_path)
        batch_sizes = []

        def encode(texts):
            batch_sizes.append(len(texts))
            return [[1.0] + [0.0] * 383 for _ in texts]

        embed_chunks(chunks, encode, CHUNK_SOURCE_PROFILE, batch_size=1)
        assert all(size == 1 for size in batch_sizes)
        assert len(batch_sizes) == len(chunks)

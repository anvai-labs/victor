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

"""Merkle projection tests: derived from the flat manifest, stdlib-only,
pure. Subtree minimality is the incremental-sync contract."""

import pytest

from victor_codegraph import (
    MerkleFormatError,
    build_merkle_tree,
    changed_subtrees,
    deserialize_tree,
    diff_merkle,
    serialize_tree,
)


def _manifest(**hashes: str) -> dict[str, str]:
    return {path: f"{hashes[path]:064d}" for path in hashes}


class TestBuildMerkleTree:
    def test_deterministic(self):
        m = _manifest(**{"src/a.py": 1, "src/b.py": 2, "top.py": 3})
        t1 = build_merkle_tree(m)
        t2 = build_merkle_tree(dict(reversed(list(m.items()))))
        assert t1.root == t2.root
        assert t1.nodes == t2.nodes

    def test_single_file_repo(self):
        tree = build_merkle_tree(_manifest(**{"main.py": 7}))
        assert tree.root
        assert tree.nodes[""] == tree.root
        assert tree.leaf_count == 1

    def test_empty_manifest(self):
        tree = build_merkle_tree({})
        assert tree.leaf_count == 0
        assert tree.root == tree.nodes[""]

    def test_directory_entry_shape(self):
        tree = build_merkle_tree(_manifest(**{"src/a.py": 1}))
        assert "src/" in tree.nodes
        assert tree.leaves["src/a.py"] == "1".zfill(64)


class TestDiffMerkle:
    def test_unchanged_root_short_circuit(self):
        m = _manifest(**{"src/a.py": 1, "lib/b.py": 2})
        t = build_merkle_tree(m)
        diff = diff_merkle(t, t)
        assert diff.changed_dirs == []
        assert not diff.added_files and not diff.changed_files and not diff.removed_files
        assert changed_subtrees(t, t) == frozenset()

    def test_change_detected(self):
        old = build_merkle_tree(_manifest(**{"src/a.py": 1}))
        new = build_merkle_tree(_manifest(**{"src/a.py": 9}))
        diff = diff_merkle(new, old)
        assert diff.changed_files == ["src/a.py"]
        assert "src/" in changed_subtrees(new, old)

    def test_add_and_remove(self):
        old = build_merkle_tree(_manifest(**{"src/a.py": 1, "old/x.py": 2}))
        new = build_merkle_tree(_manifest(**{"src/a.py": 1, "new/y.py": 3}))
        diff = diff_merkle(new, old)
        assert diff.added_files == ["new/y.py"]
        assert diff.removed_files == ["old/x.py"]

    def test_rename_is_add_plus_remove(self):
        old = build_merkle_tree(_manifest(**{"a.py": 1}))
        new = build_merkle_tree(_manifest(**{"b.py": 1}))
        diff = diff_merkle(new, old)
        assert diff.added_files == ["b.py"]
        assert diff.removed_files == ["a.py"]
        assert diff.changed_files == []

    def test_subtree_minimality_deep_edit(self):
        """One deep edit lists only the deepest changed directory — not the
        full ancestor chain (pruning walks use the frontier)."""
        files = {
            "a/b/c/deep.py": 5,
            "a/b/other.py": 6,
            "a/top.py": 7,
            "z/unrelated.py": 8,
        }
        old = build_merkle_tree(_manifest(**files))
        files["a/b/c/deep.py"] = 99
        new = build_merkle_tree(_manifest(**files))
        diff = diff_merkle(new, old)
        assert diff.changed_files == ["a/b/c/deep.py"]
        assert diff.changed_dirs == ["a/b/c/"]
        assert changed_subtrees(new, old) == frozenset({"a/b/c/"})

    def test_empty_to_nonempty_repo(self):
        old = build_merkle_tree({})
        new = build_merkle_tree(_manifest(**{"root.py": 1}))
        diff = diff_merkle(new, old)
        assert diff.added_files == ["root.py"]
        assert "" in changed_subtrees(new, old)


class TestSerialization:
    def test_roundtrip(self):
        tree = build_merkle_tree(_manifest(**{"src/a.py": 1, "b.py": 2}))
        restored = deserialize_tree(serialize_tree(tree))
        assert restored == tree

    def test_corrupt_json_rejected(self):
        with pytest.raises(MerkleFormatError):
            deserialize_tree("not json at all")

    def test_wrong_shape_rejected(self):
        with pytest.raises(MerkleFormatError):
            deserialize_tree('{"root": "x"}')


class TestChunkGranularity:
    """The tree extends directory -> file -> CONSTITUENT CHUNK so
    chunk-relevant graph identity is represented at leaf granularity."""

    def _chunked(self, deep_digest: str):
        manifest = _manifest(**{"a/b/c/deep.py": 5, "a/b/other.py": 6, "z/u.py": 8})
        chunks = {
            "a/b/c/deep.py": {
                "sym_a#body#0": str(deep_digest).zfill(64),
                "sym_a#body#1": f"{21:064d}",
            },
            "a/b/other.py": {"sym_b#body#0": f"{22:064d}"},
        }
        return manifest, chunks

    def test_file_digest_rolls_chunk_digests(self):
        manifest, chunks = self._chunked(51)
        t1 = build_merkle_tree(manifest, chunks)
        # Same file content hash but a different chunk digest -> the FILE
        # digest changes even though the manifest leaf is unchanged.
        manifest2 = dict(manifest)
        manifest2["a/b/c/deep.py"] = manifest["a/b/c/deep.py"]
        t2 = build_merkle_tree(
            manifest2,
            {**chunks, "a/b/c/deep.py": {**chunks["a/b/c/deep.py"], "sym_a#body#0": f"{52:064d}"}},
        )
        assert t1.root != t2.root
        # And the manifest leaf itself is preserved as the integrity check.
        assert t1.leaves["a/b/c/deep.py"] == t2.leaves["a/b/c/deep.py"]

    def test_chunk_ids_present_at_leaf_granularity(self):
        manifest, chunks = self._chunked("51")
        tree = build_merkle_tree(manifest, chunks)
        assert set(tree.chunks["a/b/c/deep.py"]) == {"sym_a#body#0", "sym_a#body#1"}
        assert "sym_b#body#0" in tree.chunks["a/b/other.py"]

    def test_chunk_precise_diff_lists_only_changed_chunk(self):
        manifest, chunks = self._chunked("51")
        old = build_merkle_tree(manifest, chunks)
        new = build_merkle_tree(
            manifest,
            {**chunks, "a/b/c/deep.py": {**chunks["a/b/c/deep.py"], "sym_a#body#0": f"{99:064d}"}},
        )
        diff = diff_merkle(new, old)
        assert diff.changed_chunks == [("a/b/c/deep.py", "sym_a#body#0")]
        # The manifest leaf did not change (file content hash is the
        # integrity check), so file-level changes are empty.
        assert diff.changed_files == []

    def test_chunk_add_and_remove(self):
        manifest, chunks = self._chunked("51")
        old = build_merkle_tree(manifest, chunks)
        grown = {**chunks["a/b/c/deep.py"], "sym_a#body#2": f"{23:064d}"}
        new = build_merkle_tree(manifest, {**chunks, "a/b/c/deep.py": grown})
        diff = diff_merkle(new, old)
        assert ("a/b/c/deep.py", "sym_a#body#2") in diff.changed_chunks
        shrunk = {k: v for k, v in grown.items() if k != "sym_a#body#2"}
        newer = build_merkle_tree(manifest, {**chunks, "a/b/c/deep.py": shrunk})
        diff2 = diff_merkle(newer, new)
        assert ("a/b/c/deep.py", "sym_a#body#2") in diff2.changed_chunks

    def test_unchanged_file_chunks_not_in_work_set(self):
        manifest, chunks = self._chunked("51")
        old = build_merkle_tree(manifest, chunks)
        touched = {**chunks, "a/b/other.py": {"sym_b#body#0": f"{77:064d}"}}
        new = build_merkle_tree(manifest, touched)
        diff = diff_merkle(new, old)
        assert all(path == "a/b/other.py" for path, _ in diff.changed_chunks)
        assert ("a/b/c/deep.py", "sym_a#body#0") not in diff.changed_chunks

    def test_chunked_roundtrip(self):
        manifest, chunks = self._chunked("51")
        tree = build_merkle_tree(manifest, chunks)
        restored = deserialize_tree(serialize_tree(tree))
        assert restored == tree
        assert restored.chunks == tree.chunks

    def test_files_without_chunks_fall_back_to_content_hash(self):
        manifest, chunks = self._chunked("51")
        tree = build_merkle_tree(manifest, chunks)
        # z/u.py has no chunk entry: its file digest is the manifest hash.
        flat = build_merkle_tree(manifest)
        assert tree.nodes["z/"] == flat.nodes["z/"]

    def test_manifest_hash_change_with_same_chunks_keeps_root(self):
        """Chunk digests are the leaf truth once a file is chunked; the
        manifest content hash moves to ``leaves`` as the coverage-drift
        integrity check (identical chunks + changed content hash = chunk
        coverage drift, detectable by comparing leaves)."""
        manifest, chunks = self._chunked("51")
        old = build_merkle_tree(manifest, chunks)
        m2 = dict(manifest)
        m2["a/b/c/deep.py"] = f"{6:064d}"
        new = build_merkle_tree(m2, chunks)
        assert new.root == old.root  # chunk digests unchanged -> no rework
        assert new.leaves["a/b/c/deep.py"] != old.leaves["a/b/c/deep.py"]  # drift signal
        assert diff_merkle(new, old).changed_files == []
        assert diff_merkle(new, old).changed_chunks == []

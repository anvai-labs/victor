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

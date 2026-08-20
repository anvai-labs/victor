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

"""Tests for victor.core.aot_manifest module."""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from victor.core.aot_manifest import (
    MANIFEST_VERSION,
    AOTManifest,
    AOTManifestManager,
    EntryPointEntry,
)


class TestEntryPointEntry:
    """Tests for EntryPointEntry dataclass."""

    def test_entry_point_entry_creation(self):
        """EntryPointEntry should store all fields correctly."""
        entry = EntryPointEntry(
            name="security",
            module="victor_security.vertical",
            attr="SecurityAssistant",
            group="victor.verticals",
        )
        assert entry.name == "security"
        assert entry.module == "victor_security.vertical"
        assert entry.attr == "SecurityAssistant"
        assert entry.group == "victor.verticals"

    def test_entry_point_entry_equality(self):
        """Two EntryPointEntry with same values should be equal."""
        entry1 = EntryPointEntry("test", "mod", "attr", "group")
        entry2 = EntryPointEntry("test", "mod", "attr", "group")
        assert entry1 == entry2

    def test_entry_point_entry_with_empty_attr(self):
        """EntryPointEntry should handle empty attr (module-only entry points)."""
        entry = EntryPointEntry(
            name="plugin",
            module="victor_plugin",
            attr="",
            group="victor.plugins",
        )
        assert entry.attr == ""


class TestAOTManifest:
    """Tests for AOTManifest dataclass."""

    def test_manifest_creation(self):
        """AOTManifest should store version, hash, and entries."""
        entries = {"victor.verticals": [EntryPointEntry("sec", "mod", "attr", "victor.verticals")]}
        manifest = AOTManifest(
            version="1.0",
            env_hash="abc123",
            entries=entries,
        )
        assert manifest.version == "1.0"
        assert manifest.env_hash == "abc123"
        assert len(manifest.entries["victor.verticals"]) == 1

    def test_manifest_to_dict(self):
        """to_dict should serialize manifest to JSON-compatible dict."""
        entry = EntryPointEntry("test", "module.path", "MyClass", "group.name")
        manifest = AOTManifest("1.0", "hash123", {"group.name": [entry]})

        result = manifest.to_dict()

        assert result["version"] == "1.0"
        assert result["env_hash"] == "hash123"
        assert result["entries"]["group.name"][0] == {
            "name": "test",
            "module": "module.path",
            "attr": "MyClass",
            "group": "group.name",
        }

    def test_manifest_from_dict(self):
        """from_dict should deserialize dict to AOTManifest."""
        data = {
            "version": "1.0",
            "env_hash": "hash456",
            "entries": {
                "group1": [
                    {
                        "name": "entry1",
                        "module": "mod1",
                        "attr": "Attr1",
                        "group": "group1",
                    }
                ],
                "group2": [
                    {
                        "name": "entry2",
                        "module": "mod2",
                        "attr": "Attr2",
                        "group": "group2",
                    }
                ],
            },
        }

        manifest = AOTManifest.from_dict(data)

        assert manifest.version == "1.0"
        assert manifest.env_hash == "hash456"
        assert len(manifest.entries) == 2
        assert manifest.entries["group1"][0].name == "entry1"
        assert manifest.entries["group2"][0].attr == "Attr2"

    def test_manifest_round_trip(self):
        """to_dict/from_dict should preserve all data."""
        original_entries = {
            "victor.verticals": [
                EntryPointEntry("security", "sec.mod", "SecClass", "victor.verticals"),
                EntryPointEntry("ml", "ml.mod", "MLClass", "victor.verticals"),
            ],
            "victor.providers": [
                EntryPointEntry("custom", "custom.prov", "Provider", "victor.providers"),
            ],
        }
        original = AOTManifest("1.0", "original_hash", original_entries)

        # Round trip
        data = original.to_dict()
        restored = AOTManifest.from_dict(data)

        assert restored.version == original.version
        assert restored.env_hash == original.env_hash
        assert len(restored.entries) == len(original.entries)
        for group in original.entries:
            assert len(restored.entries[group]) == len(original.entries[group])
            for i, entry in enumerate(original.entries[group]):
                assert restored.entries[group][i] == entry

    def test_manifest_from_dict_with_empty_entries(self):
        """from_dict should handle empty entries dict."""
        data = {"version": "1.0", "env_hash": "empty", "entries": {}}
        manifest = AOTManifest.from_dict(data)
        assert manifest.entries == {}

    def test_manifest_from_dict_with_missing_entries(self):
        """from_dict should handle missing entries key."""
        data = {"version": "1.0", "env_hash": "missing"}
        manifest = AOTManifest.from_dict(data)
        assert manifest.entries == {}


class TestAOTManifestManagerComputeEnvHash:
    """Tests for AOTManifestManager.compute_env_hash."""

    def test_compute_env_hash_returns_16_char_hex(self, tmp_path):
        """compute_env_hash should return a 16-character hex string."""
        manager = AOTManifestManager(cache_dir=tmp_path)
        hash_value = manager.compute_env_hash()

        # Should be exactly 16 characters
        assert len(hash_value) == 16
        # Should be valid hex
        int(hash_value, 16)

    def test_compute_env_hash_is_deterministic(self, tmp_path):
        """compute_env_hash should return same value on repeated calls."""
        manager = AOTManifestManager(cache_dir=tmp_path)
        hash1 = manager.compute_env_hash()
        hash2 = manager.compute_env_hash()
        assert hash1 == hash2

    def test_compute_env_hash_returns_unknown_on_error(self, tmp_path):
        """compute_env_hash should return 'unknown' if distributions fails."""
        manager = AOTManifestManager(cache_dir=tmp_path)

        with patch("victor.core.aot_manifest.AOTManifestManager.compute_env_hash") as mock_compute:
            # Simulate internal error
            mock_compute.return_value = "unknown"
            result = manager.compute_env_hash()
            assert result == "unknown"


class TestAOTManifestManagerSaveLoad:
    """Tests for AOTManifestManager save/load functionality."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create manager with temp cache directory."""
        return AOTManifestManager(cache_dir=tmp_path)

    @pytest.fixture
    def sample_manifest(self, manager):
        """Create a sample manifest for testing."""
        entries = {
            "victor.verticals": [
                EntryPointEntry("test", "test.mod", "TestClass", "victor.verticals")
            ]
        }
        return AOTManifest(MANIFEST_VERSION, manager.compute_env_hash(), entries)

    def test_save_manifest_creates_directory(self, tmp_path):
        """save_manifest should create cache directory if needed."""
        nested_path = tmp_path / "nested" / "cache" / "dir"
        manager = AOTManifestManager(cache_dir=nested_path)
        manifest = AOTManifest(MANIFEST_VERSION, "hash", {})

        manager.save_manifest(manifest)

        assert nested_path.exists()
        assert manager.manifest_path.exists()

    def test_save_manifest_writes_valid_json(self, manager, sample_manifest):
        """save_manifest should write valid JSON."""
        manager.save_manifest(sample_manifest)

        with open(manager.manifest_path) as f:
            data = json.load(f)

        assert data["version"] == MANIFEST_VERSION
        assert "env_hash" in data
        assert "entries" in data

    def test_load_manifest_returns_none_if_no_file(self, manager):
        """load_manifest should return None if no cache file exists."""
        result = manager.load_manifest()
        assert result is None

    def test_load_manifest_returns_manifest_if_valid(self, manager, sample_manifest):
        """load_manifest should return manifest if cache is valid."""
        manager.save_manifest(sample_manifest)

        result = manager.load_manifest()

        assert result is not None
        assert result.version == sample_manifest.version
        assert result.env_hash == sample_manifest.env_hash

    def test_load_manifest_returns_none_if_env_hash_changed(self, manager, tmp_path):
        """load_manifest should return None if env_hash doesn't match."""
        # Save manifest with different hash
        manifest = AOTManifest(MANIFEST_VERSION, "old_hash", {})
        manager.save_manifest(manifest)

        # Load should fail because current hash won't match "old_hash"
        result = manager.load_manifest()
        assert result is None

    def test_load_manifest_returns_none_if_version_mismatch(self, manager, tmp_path):
        """load_manifest should return None if version doesn't match."""
        # Write manifest with wrong version directly
        with open(manager.manifest_path, "w") as f:
            json.dump(
                {
                    "version": "0.1",
                    "env_hash": manager.compute_env_hash(),
                    "entries": {},
                },
                f,
            )

        result = manager.load_manifest()
        assert result is None

    def test_load_manifest_returns_none_if_invalid_json(self, manager, tmp_path):
        """load_manifest should return None if JSON is invalid."""
        manager.cache_dir.mkdir(parents=True, exist_ok=True)
        with open(manager.manifest_path, "w") as f:
            f.write("not valid json {{{")

        result = manager.load_manifest()
        assert result is None

    def test_load_manifest_returns_none_if_missing_keys(self, manager, tmp_path):
        """load_manifest should return None if required keys are missing."""
        manager.cache_dir.mkdir(parents=True, exist_ok=True)
        with open(manager.manifest_path, "w") as f:
            json.dump({"partial": "data"}, f)

        result = manager.load_manifest()
        assert result is None

    def test_save_load_round_trip(self, manager, sample_manifest):
        """save/load should preserve manifest data."""
        manager.save_manifest(sample_manifest)
        loaded = manager.load_manifest()

        assert loaded is not None
        assert loaded.version == sample_manifest.version
        assert loaded.env_hash == sample_manifest.env_hash
        assert len(loaded.entries) == len(sample_manifest.entries)


class TestAOTManifestManagerBuildManifest:
    """Tests for AOTManifestManager.build_manifest."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create manager with temp cache directory."""
        return AOTManifestManager(cache_dir=tmp_path)

    def test_build_manifest_with_empty_groups(self, manager):
        """build_manifest should handle empty group list."""
        manifest = manager.build_manifest([])

        assert manifest.version == MANIFEST_VERSION
        assert manifest.env_hash == manager.compute_env_hash()
        assert manifest.entries == {}

    def test_build_manifest_returns_aot_manifest(self, manager):
        """build_manifest should return AOTManifest instance."""
        # Use a group that probably has no entries
        manifest = manager.build_manifest(["nonexistent.group"])

        assert isinstance(manifest, AOTManifest)
        assert manifest.version == MANIFEST_VERSION
        assert "nonexistent.group" in manifest.entries
        assert manifest.entries["nonexistent.group"] == []

    def test_build_manifest_parses_entry_point_value(self, manager):
        """build_manifest should correctly parse module:attr format."""
        # Mock entry_points to return controlled data
        mock_ep = MagicMock()
        mock_ep.name = "test_entry"
        mock_ep.value = "some.module:SomeClass"

        def mock_entry_points_fn(group=None):
            if group == "test.group":
                return [mock_ep]
            return []

        with patch("importlib.metadata.entry_points", side_effect=mock_entry_points_fn):
            manifest = manager.build_manifest(["test.group"])

        assert len(manifest.entries["test.group"]) == 1
        entry = manifest.entries["test.group"][0]
        assert entry.name == "test_entry"
        assert entry.module == "some.module"
        assert entry.attr == "SomeClass"
        assert entry.group == "test.group"

    def test_build_manifest_handles_module_only_entry_point(self, manager):
        """build_manifest should handle entry points without :attr."""
        mock_ep = MagicMock()
        mock_ep.name = "module_only"
        mock_ep.value = "just.module.path"

        def mock_entry_points_fn(group=None):
            if group == "test.group":
                return [mock_ep]
            return []

        with patch("importlib.metadata.entry_points", side_effect=mock_entry_points_fn):
            manifest = manager.build_manifest(["test.group"])

        entry = manifest.entries["test.group"][0]
        assert entry.module == "just.module.path"
        assert entry.attr == ""

    def test_build_manifest_multiple_groups(self, manager):
        """build_manifest should handle multiple groups."""
        mock_ep1 = MagicMock()
        mock_ep1.name = "ep1"
        mock_ep1.value = "mod1:Cls1"

        mock_ep2 = MagicMock()
        mock_ep2.name = "ep2"
        mock_ep2.value = "mod2:Cls2"

        def mock_entry_points_fn(group=None):
            if group == "group1":
                return [mock_ep1]
            elif group == "group2":
                return [mock_ep2]
            return []

        with patch("importlib.metadata.entry_points", side_effect=mock_entry_points_fn):
            manifest = manager.build_manifest(["group1", "group2"])

        assert "group1" in manifest.entries
        assert "group2" in manifest.entries
        assert manifest.entries["group1"][0].name == "ep1"
        assert manifest.entries["group2"][0].name == "ep2"


class TestAOTManifestManagerInvalidate:
    """Tests for AOTManifestManager.invalidate."""

    def test_invalidate_removes_manifest_file(self, tmp_path):
        """invalidate should remove the manifest file."""
        manager = AOTManifestManager(cache_dir=tmp_path)
        manifest = AOTManifest(MANIFEST_VERSION, "hash", {})
        manager.save_manifest(manifest)

        assert manager.manifest_path.exists()
        result = manager.invalidate()

        assert result is True
        assert not manager.manifest_path.exists()

    def test_invalidate_returns_false_if_no_file(self, tmp_path):
        """invalidate should return False if no file exists."""
        manager = AOTManifestManager(cache_dir=tmp_path)

        result = manager.invalidate()

        assert result is False


class TestAOTManifestManagerCacheInvalidation:
    """Tests for cache invalidation when environment changes."""

    def test_cache_invalidation_on_env_hash_change(self, tmp_path):
        """Cache should be invalidated when environment hash changes."""
        manager = AOTManifestManager(cache_dir=tmp_path)

        # Save manifest with current hash
        manifest = manager.build_manifest([])
        manager.save_manifest(manifest)

        # Verify it loads correctly
        assert manager.load_manifest() is not None

        # Simulate environment change by modifying the cached hash
        with open(manager.manifest_path) as f:
            data = json.load(f)
        data["env_hash"] = "different_hash"
        with open(manager.manifest_path, "w") as f:
            json.dump(data, f)

        # Now load should fail because hash doesn't match
        result = manager.load_manifest()
        assert result is None


class TestAOTManifestManagerDefaultCacheDir:
    """Tests for default cache directory behavior."""

    def test_default_cache_dir_is_home_victor_cache(self):
        """Default cache_dir should be ~/.victor/cache."""
        manager = AOTManifestManager()
        expected = Path.home() / ".victor" / "cache"
        assert manager.cache_dir == expected

    def test_manifest_path_is_entrypoints_json(self, tmp_path):
        """manifest_path should be cache_dir/entrypoints.json."""
        manager = AOTManifestManager(cache_dir=tmp_path)
        assert manager.manifest_path == tmp_path / "entrypoints.json"

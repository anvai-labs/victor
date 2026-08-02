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

"""Ahead-of-time entry point manifest for faster startup.

This module provides AOT (ahead-of-time) caching of Python entry points
to avoid the overhead of scanning installed packages on every startup.
The manifest is invalidated when the installed package set changes.

Usage:
    manager = AOTManifestManager()

    # Try to load cached manifest
    manifest = manager.load_manifest()
    if manifest is None:
        # Build and cache new manifest
        manifest = manager.build_manifest(["victor.plugins", "victor.providers"])
        manager.save_manifest(manifest)

    # Use cached entries
    for entry in manifest.entries.get("victor.plugins", []):
        module = importlib.import_module(entry.module)
        attr = getattr(module, entry.attr)
"""

import json
import hashlib
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MANIFEST_VERSION = "1.0"


@dataclass
class EntryPointEntry:
    """Represents a single entry point discovered from a package.

    Attributes:
        name: The entry point name (e.g., "security" for a vertical).
        module: The module path to import (e.g., "victor_security.vertical").
        attr: The attribute to get from the module (e.g., "SecurityAssistant").
        group: The entry point group (e.g., "victor.plugins").
    """

    name: str
    module: str
    attr: str
    group: str


@dataclass
class AOTManifest:
    """Container for all cached entry points.

    Attributes:
        version: Manifest format version for compatibility checking.
        env_hash: Hash of installed packages to detect environment changes.
        entries: Dict mapping group names to lists of EntryPointEntry.
    """

    version: str
    env_hash: str
    entries: Dict[str, List[EntryPointEntry]]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize manifest to a JSON-compatible dictionary."""
        return {
            "version": self.version,
            "env_hash": self.env_hash,
            "entries": {
                group: [asdict(e) for e in entries] for group, entries in self.entries.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AOTManifest":
        """Deserialize manifest from a dictionary.

        Args:
            data: Dictionary with version, env_hash, and entries keys.

        Returns:
            AOTManifest instance.
        """
        entries = {
            group: [EntryPointEntry(**e) for e in elist]
            for group, elist in data.get("entries", {}).items()
        }
        return cls(data["version"], data["env_hash"], entries)


class AOTManifestManager:
    """Manages AOT entry point manifest for fast discovery.

    The manager handles:
    - Computing environment hashes to detect package changes
    - Loading/saving manifest from disk cache
    - Building fresh manifests from entry points
    - Automatic cache invalidation when environment changes

    Attributes:
        cache_dir: Directory where manifest is stored.
        manifest_path: Full path to the manifest JSON file.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize the manifest manager.

        Args:
            cache_dir: Directory for cache files. Defaults to ~/.victor/cache
        """
        self.cache_dir = cache_dir or Path.home() / ".victor" / "cache"
        self.manifest_path = self.cache_dir / "entrypoints.json"

    def compute_env_hash(self) -> str:
        """Compute a hash of the current Python environment.

        The hash is based on all installed package names and versions,
        sorted alphabetically. This ensures the cache is invalidated
        when packages are installed, removed, or upgraded.

        Returns:
            16-character hex hash of the environment, or "unknown" on error.
        """
        try:
            from importlib.metadata import distributions

            packages = sorted(f"{d.name}=={d.version}" for d in distributions())
            return hashlib.sha256("\n".join(packages).encode()).hexdigest()[:16]
        except Exception:
            logger.debug("Failed to compute environment hash", exc_info=True)
            return "unknown"

    def load_manifest(self) -> Optional[AOTManifest]:
        """Load and validate the cached manifest.

        The manifest is only returned if:
        - The file exists
        - It can be parsed as valid JSON
        - The env_hash matches the current environment

        Returns:
            AOTManifest if valid cache exists, None otherwise.
        """
        if not self.manifest_path.exists():
            logger.debug("No manifest cache found at %s", self.manifest_path)
            return None

        try:
            with open(self.manifest_path) as f:
                data = json.load(f)

            manifest = AOTManifest.from_dict(data)

            # Validate version compatibility
            if manifest.version != MANIFEST_VERSION:
                logger.debug(
                    "Manifest version mismatch: %s != %s",
                    manifest.version,
                    MANIFEST_VERSION,
                )
                return None

            # Validate environment hash
            current_hash = self.compute_env_hash()
            if manifest.env_hash != current_hash:
                logger.debug(
                    "Environment hash mismatch: %s != %s",
                    manifest.env_hash,
                    current_hash,
                )
                return None

            logger.debug("Loaded valid manifest from cache")
            return manifest

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.debug("Failed to load manifest: %s", e)
            return None

    def save_manifest(self, manifest: AOTManifest) -> None:
        """Save manifest to disk cache.

        Creates the cache directory if it doesn't exist.

        Args:
            manifest: The manifest to save.
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with open(self.manifest_path, "w") as f:
            json.dump(manifest.to_dict(), f, indent=2)
        logger.debug("Saved manifest to %s", self.manifest_path)

    def build_manifest(self, groups: List[str]) -> AOTManifest:
        """Build a fresh manifest from entry points.

        Scans the specified entry point groups and creates entries
        for each discovered entry point.

        Args:
            groups: List of entry point group names to scan
                   (e.g., ["victor.plugins", "victor.providers"]).

        Returns:
            New AOTManifest with current environment hash and entries.
        """
        from importlib.metadata import entry_points

        entries: Dict[str, List[EntryPointEntry]] = {}

        for group in groups:
            eps = entry_points(group=group)
            group_entries = []

            for ep in eps:
                # Parse the entry point value (format: "module:attr")
                if ":" in ep.value:
                    module, attr = ep.value.split(":", 1)
                else:
                    module = ep.value
                    attr = ""

                group_entries.append(
                    EntryPointEntry(
                        name=ep.name,
                        module=module,
                        attr=attr,
                        group=group,
                    )
                )

            entries[group] = group_entries
            logger.debug("Found %d entry points for group %s", len(group_entries), group)

        return AOTManifest(MANIFEST_VERSION, self.compute_env_hash(), entries)

    def invalidate(self) -> bool:
        """Remove the cached manifest file.

        Returns:
            True if file was removed, False if it didn't exist.
        """
        if self.manifest_path.exists():
            self.manifest_path.unlink()
            logger.debug("Invalidated manifest cache")
            return True
        return False

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

"""Artifact management for experiment tracking.

This module provides functionality for storing, retrieving, and managing
artifacts associated with experiment runs.
"""

from __future__ import annotations

import logging
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from victor.experiments.entities import Artifact, ArtifactType
from victor.experiments.storage import IStorageBackend

logger = logging.getLogger(__name__)


class ArtifactManager:
    """Manager for storing and retrieving artifacts.

    This class handles artifact storage using a local filesystem backend,
    with support for different artifact types and metadata tracking.

    Args:
        artifact_root: Root directory for artifact storage
                        (default: ~/.victor/artifacts)
        storage: Storage backend for artifact metadata

    Example:
        manager = ArtifactManager()
        artifact = manager.log_artifact(
            run_id="run-123",
            file_path="/path/to/model.pkl",
            artifact_type=ArtifactType.MODEL,
        )
    """

    def __init__(
        self,
        artifact_root: Optional[str] = None,
        storage: Optional[IStorageBackend] = None,
    ) -> None:
        """Initialize artifact manager.

        Args:
            artifact_root: Root directory for artifact storage
            storage: Storage backend for metadata
        """
        if artifact_root is None:
            victor_dir = Path.home() / ".victor"
            victor_dir.mkdir(parents=True, exist_ok=True)
            artifact_root = str(victor_dir / "artifacts")

        self.artifact_root = Path(artifact_root).expanduser()
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self._storage = storage

    def log_artifact(
        self,
        run_id: str,
        file_path: str,
        artifact_type: Union[ArtifactType, str] = ArtifactType.CUSTOM,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Artifact:
        """Log an artifact for a run.

        Copies the file to the artifact storage directory and creates
        an artifact record in the database.

        Args:
            run_id: Run ID
            file_path: Path to the artifact file
            artifact_type: Type of artifact
            metadata: Optional metadata dictionary

        Returns:
            Created artifact entity

        Raises:
            FileNotFoundError: If source file doesn't exist
            IOError: If file copy fails
        """
        source_path = Path(file_path).expanduser()
        if not source_path.exists():
            raise FileNotFoundError(f"Artifact file not found: {file_path}")

        # Convert string to enum if needed
        if isinstance(artifact_type, str):
            artifact_type = ArtifactType(artifact_type)

        # Generate unique artifact ID
        artifact_id = str(uuid.uuid4())

        # Create destination path: <root>/<run_id>/<artifact_id>/<filename>
        dest_dir = self.artifact_root / run_id / artifact_id
        dest_dir.mkdir(parents=True, exist_ok=True)

        dest_path = dest_dir / source_path.name

        # Copy file to artifact storage
        shutil.copy2(source_path, dest_path)

        # Get file size
        file_size = dest_path.stat().st_size

        # Create artifact entity
        artifact = Artifact(
            artifact_id=artifact_id,
            run_id=run_id,
            artifact_type=artifact_type,
            filename=source_path.name,
            file_path=str(dest_path),
            file_size_bytes=file_size,
            created_at=datetime.now(timezone.utc),
            metadata=metadata or {},
        )

        # Store artifact metadata in database
        if self._storage:
            self._storage.log_artifact(artifact)

        logger.info(
            f"Logged artifact {artifact_id} for run {run_id}: "
            f"{source_path.name} ({file_size} bytes)"
        )

        return artifact

    def download_artifact(
        self,
        artifact_id: str,
        dest_path: str,
        storage: Optional[IStorageBackend] = None,
    ) -> Path:
        """Download an artifact to a local path.

        Args:
            artifact_id: Artifact ID
            dest_path: Destination directory or file path
            storage: Storage backend for metadata lookup

        Returns:
            Path to downloaded file

        Raises:
            FileNotFoundError: If artifact doesn't exist
        """
        if storage is None:
            storage = self._storage

        if storage is None:
            raise ValueError("Storage backend required for artifact lookup")

        # Get artifact metadata
        artifact = storage.get_artifact(artifact_id)
        if artifact is None:
            raise FileNotFoundError(f"Artifact not found: {artifact_id}")

        source_path = Path(artifact.file_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Artifact file not found: {artifact.file_path}")

        # Determine destination path
        dest = Path(dest_path).expanduser()
        if dest.is_dir():
            dest = dest / artifact.filename

        # Copy file to destination
        shutil.copy2(source_path, dest)

        logger.info(f"Downloaded artifact {artifact_id} to {dest}")
        return dest

    def get_artifact_path(self, artifact_id: str) -> Optional[Path]:
        """Get the filesystem path for an artifact.

        Args:
            artifact_id: Artifact ID

        Returns:
            Path to artifact file, or None if not found
        """
        if self._storage is None:
            return None

        artifact = self._storage.get_artifact(artifact_id)
        if artifact is None:
            return None

        return Path(artifact.file_path)

    def list_artifacts(
        self, run_id: str, storage: Optional[IStorageBackend] = None
    ) -> List[Artifact]:
        """List all artifacts for a run.

        Args:
            run_id: Run ID
            storage: Storage backend

        Returns:
            List of artifacts
        """
        if storage is None:
            storage = self._storage

        if storage is None:
            raise ValueError("Storage backend required")

        return storage.get_artifacts(run_id)

    def delete_artifact(self, artifact_id: str, storage: Optional[IStorageBackend] = None) -> bool:
        """Delete an artifact.

        Removes both the database record and the file from disk.

        Args:
            artifact_id: Artifact ID
            storage: Storage backend

        Returns:
            True if deleted, False if not found
        """
        if storage is None:
            storage = self._storage

        if storage is None:
            raise ValueError("Storage backend required")

        # Get artifact metadata
        artifact = storage.get_artifact(artifact_id)
        if artifact is None:
            return False

        # Delete file from disk
        artifact_path = Path(artifact.file_path)
        if artifact_path.exists():
            # Remove the file
            artifact_path.unlink()
            # Try to remove parent directories if empty
            try:
                artifact_path.parent.rmdir()  # artifact_id directory
            except OSError:
                pass  # Directory not empty, that's okay

        # Note: Database deletion not implemented in IStorageBackend
        # This would need to be added to the protocol
        logger.info(f"Deleted artifact {artifact_id}")

        return True

    def get_artifact_uri(self, artifact_id: str) -> Optional[str]:
        """Get a URI for an artifact (e.g., for external access).

        Args:
            artifact_id: Artifact ID

        Returns:
            URI string (file://), or None if not found
        """
        path = self.get_artifact_path(artifact_id)
        if path is None:
            return None

        return f"file://{path.absolute()}"

    def cleanup_run_artifacts(self, run_id: str, storage: Optional[IStorageBackend] = None) -> int:
        """Delete all artifacts for a run.

        Args:
            run_id: Run ID
            storage: Storage backend

        Returns:
            Number of artifacts deleted
        """
        if storage is None:
            storage = self._storage

        if storage is None:
            raise ValueError("Storage backend required")

        artifacts = storage.get_artifacts(run_id)
        count = 0

        for artifact in artifacts:
            try:
                if self.delete_artifact(artifact.artifact_id, storage):
                    count += 1
            except Exception as e:
                logger.warning(f"Failed to delete artifact {artifact.artifact_id}: {e}")

        # Delete run directory
        run_dir = self.artifact_root / run_id
        if run_dir.exists():
            shutil.rmtree(run_dir)

        logger.info(f"Cleaned up {count} artifacts for run {run_id}")
        return count

    def get_storage_usage(self, run_id: Optional[str] = None) -> Dict[str, int]:
        """Get storage usage statistics.

        Args:
            run_id: Optional run ID to get usage for specific run

        Returns:
            Dictionary with storage stats (bytes, file_count)
        """
        if run_id:
            run_dir = self.artifact_root / run_id
            if not run_dir.exists():
                return {"bytes": 0, "file_count": 0}

            file_count = sum(1 for _ in run_dir.rglob("*") if _.is_file())
            total_bytes = sum(f.stat().st_size for f in run_dir.rglob("*") if f.is_file())

            return {"bytes": total_bytes, "file_count": file_count}

        # Get total usage across all runs
        file_count = sum(1 for _ in self.artifact_root.rglob("*") if _.is_file())
        total_bytes = sum(f.stat().st_size for f in self.artifact_root.rglob("*") if f.is_file())

        return {"bytes": total_bytes, "file_count": file_count}


# Singleton instance for global access
_artifact_manager: Optional[ArtifactManager] = None


def get_artifact_manager() -> ArtifactManager:
    """Get the global artifact manager instance.

    Returns:
        Shared ArtifactManager instance
    """
    global _artifact_manager
    if _artifact_manager is None:
        _artifact_manager = ArtifactManager()
    return _artifact_manager

# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Shared loader for manifest-driven benchmark datasets (JSONL, list, or object)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Optional

if TYPE_CHECKING:
    from victor.evaluation.protocol import BenchmarkMetadata


def load_benchmark_manifest(
    dataset_path: Path,
    *,
    dataset_label: str,
    manifest_label: str,
) -> tuple[list[dict[str, Any]], Optional[dict[str, Any]], dict[str, Any]]:
    """Read a benchmark manifest and validate its object form.

    Accepted shapes: a ``.jsonl`` file with one task record per non-blank
    line, a JSON list of task records, or a JSON object with optional
    ``tasks`` (list), ``defaults`` (object) and ``metadata`` (object) keys.

    Args:
        dataset_path: Path to the manifest file.
        dataset_label: Prefix for the "payload must be a list or object"
            error, e.g. ``"Browser benchmark"``.
        manifest_label: Prefix for the per-key shape errors, e.g.
            ``"Browser benchmark"``.

    Returns:
        ``(records, defaults, metadata)``. ``defaults`` is ``None`` unless the
        manifest is an object (only object manifests carry defaults);
        ``metadata`` is ``{}`` for JSONL and list manifests.

    Raises:
        FileNotFoundError: If ``dataset_path`` does not exist.
        ValueError: If an object manifest has a key of the wrong shape, or
            the JSON payload is neither a list nor an object.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Benchmark dataset not found: {dataset_path}")

    if dataset_path.suffix == ".jsonl":
        records = []
        for line in dataset_path.read_text().splitlines():
            if line.strip():
                records.append(json.loads(line))
        return records, None, {}

    payload = json.loads(dataset_path.read_text())
    if isinstance(payload, list):
        return payload, None, {}

    if not isinstance(payload, dict):
        raise ValueError(f"{dataset_label} dataset payload must be a list or object manifest")

    tasks = payload.get("tasks", [])
    if not isinstance(tasks, list):
        raise ValueError(f"{manifest_label} manifest 'tasks' must be a list")

    defaults = payload.get("defaults", {}) or {}
    if not isinstance(defaults, dict):
        raise ValueError(f"{manifest_label} manifest 'defaults' must be an object")

    metadata = payload.get("metadata", {}) or {}
    if not isinstance(metadata, dict):
        raise ValueError(f"{manifest_label} manifest 'metadata' must be an object")

    return tasks, defaults, metadata


class ManifestBenchmarkMixin:
    """``_load_manifest`` for runners whose datasets are benchmark manifests.

    Subclasses set the two error-message labels and implement
    ``_build_manifest_metadata``; object manifests populate ``_defaults``.
    """

    _manifest_dataset_label: ClassVar[str]
    _manifest_label: ClassVar[str]
    _defaults: dict[str, Any]

    if TYPE_CHECKING:

        def _build_manifest_metadata(
            self, metadata: dict[str, Any], records: list[dict[str, Any]]
        ) -> BenchmarkMetadata: ...

    def _load_manifest(self, dataset_path: Path) -> tuple[list[dict[str, Any]], BenchmarkMetadata]:
        records, defaults, metadata = load_benchmark_manifest(
            dataset_path,
            dataset_label=self._manifest_dataset_label,
            manifest_label=self._manifest_label,
        )
        if defaults is not None:
            self._defaults = defaults
        return records, self._build_manifest_metadata(metadata, records)

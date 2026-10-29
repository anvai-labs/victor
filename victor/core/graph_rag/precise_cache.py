"""ADR-032 tier 3: on-disk cache for SCIP dump artifacts (derived state).

A full pyright dump of a large monorepo takes minutes; re-paying it on every
harvest (any corpus change re-fires it via the ``graph_changed`` gate) is the
dominant tier-3 cost. The artifact is pure derived state — deterministic for
a given (tool, corpus, schema) — so it caches on disk like every other
derived index output:

- ``<root>/.victor/precise/<provider>.scip`` — the artifact (atomic write:
  tmp file + ``os.replace``, then the sidecar — never a half artifact)
- ``<root>/.victor/precise/<provider>.scip.json`` — the sidecar carrying the
  cache key and the artifact size (truncation guard)

Cache key = sha256 over ``{schema, provider, tool_path, tool_stat
[mtime_ns, size], files: the language's (path, content_hash) manifest}``.
Any add/change/delete in THAT language's files re-dumps only that language
(editing ``.rs`` reuses the ~8-minute pyright artifact); a tool binary change
re-dumps; a schema bump busts everything. Tool identity is the binary's
stat, not a version probe (no subprocess per run — the ``prepared_id``
precedent folds identity signals, it does not probe). Bust the cache with
``rm -rf .victor/precise``; there is no eviction or size bound (the feature
is opt-in and the python artifact is ~336 MB — called out in the CHANGELOG).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CACHE_SCHEMA = "precise-dump-cache-v1"


def dump_cache_key(
    provider_name: str,
    tool_path: str,
    file_hashes: dict[str, str],
) -> str:
    """The cache key for one provider's artifact.

    ``file_hashes`` is the per-language slice of the codegraph manifest
    (path → content_hash). Sorted for order-independence.
    """

    tool = Path(tool_path)
    try:
        stat = tool.stat()
        tool_stat = [stat.st_mtime_ns, stat.st_size]
    except OSError:
        tool_stat = [0, 0]
    evidence = {
        "schema": CACHE_SCHEMA,
        "provider": provider_name,
        "tool_path": str(tool),
        "tool_stat": tool_stat,
        "files": sorted(file_hashes.items()),
    }
    raw = json.dumps(evidence, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def cache_paths(root: Path, provider_name: str) -> tuple[Path, Path]:
    """(artifact, sidecar) paths under the project's .victor/precise/."""

    base = root / ".victor" / "precise"
    return base / f"{provider_name}.scip", base / f"{provider_name}.scip.json"


def load_cached(
    root: Path,
    provider_name: str,
    key: str,
) -> Optional[Path]:
    """A reusable artifact for this key, or None.

    Reuse requires: sidecar parses, key matches, artifact exists, and the
    artifact's size equals the sidecar's recorded size (a truncated or
    partially-written artifact never validates). Best-effort: any malformed
    sidecar is a miss, never an error.
    """

    artifact, sidecar = cache_paths(root, provider_name)
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        if payload.get("key") != key:
            return None
        expected_bytes = int(payload["artifact_bytes"])
        if not artifact.exists() or artifact.stat().st_size != expected_bytes:
            return None
    except (OSError, ValueError, KeyError, TypeError):
        return None
    logger.info("precise-index: reusing cached %s dump (key %s..)", provider_name, key[:12])
    return artifact


def store(
    root: Path,
    provider_name: str,
    artifact: Path,
    key: str,
) -> None:
    """Persist the sidecar for an artifact already written to its cache path.

    Atomicity: the dump is written to a unique ``.tmp`` sibling and
    ``os.replace``d into place BEFORE this sidecar lands — a crash between
    the two leaves a keyless artifact that never validates as a hit. A
    zero-byte artifact is refused: a truncated protobuf parses as a VALID
    empty index, which would silently harvest zero edges (observed when two
    concurrent dumps raced a fixed tmp name — the hazard this guards).
    """

    size = artifact.stat().st_size
    if size <= 0:
        raise ValueError(f"refusing to cache a zero-byte {provider_name} artifact")
    _, sidecar = cache_paths(root, provider_name)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": CACHE_SCHEMA, "key": key, "artifact_bytes": size}
    tmp = sidecar.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, sidecar)


def atomic_dump_path(root: Path, provider_name: str) -> tuple[Path, Path]:
    """(final_path, tmp_path) the provider should dump to: write tmp, then
    ``os.replace(tmp, final)`` before ``store`` records the sidecar.

    The tmp name is PROCESS-UNIQUE: two concurrent harvests on the same root
    would otherwise interleave writes on one fixed tmp file and ``replace``
    a corrupt mixture — and a truncated protobuf parses as a valid EMPTY
    index, poisoning the cache with a zero-edge artifact.
    """

    artifact, _ = cache_paths(root, provider_name)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    return artifact, artifact.with_name(f"{artifact.name}.{os.getpid()}.tmp")


def slice_manifest_for_extensions(
    manifest: Optional[dict[str, str]],
    extensions: frozenset[str],
) -> dict[str, str]:
    """The manifest entries for one language's files (path → content_hash)."""

    if not manifest:
        return {}
    return {
        path: digest for path, digest in manifest.items() if Path(path).suffix.lower() in extensions
    }

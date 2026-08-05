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

"""Shared load-validate-save helper for metadata-wrapped pickle caches.

Several components persist small derived datasets (tool embeddings, static
collection embeddings) as a single pickled ``dict`` that wraps both the payload
and validation metadata (cache version, content hash, embedding model, embedding
dimensions). They all follow the same lifecycle:

1. Load the pickle if the file exists.
2. Run an ordered sequence of validators (version, hash, model, dims,
   integrity). Each validator decides whether the cache is valid and, if not,
   whether the stale cache file should be deleted.
3. Delete the file and return ``None`` on any unpickling/generic error.
4. Save the payload plus metadata back to disk.

This module extracts that shared flow. It is deliberately minimal: it owns only
the file I/O, the delete-on-mismatch bookkeeping, and the exception handling.
The per-site validation semantics are supplied by callers as validator
callables, so each site keeps its exact checks, order, and on-mismatch
behavior.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

__all__ = [
    "CacheValidation",
    "valid",
    "invalid",
    "PickleCacheValidator",
    "load_validated_pickle",
    "save_pickle_with_metadata",
    "delete_cache_file",
]


@dataclass(frozen=True)
class CacheValidation:
    """Outcome of a single cache validator.

    Attributes:
        ok: True if the validator considers the cached data valid.
        delete: When ``ok`` is False, whether the stale cache file should be
            deleted. Ignored when ``ok`` is True.
        reason: Short human-readable reason used for the deletion log message.
    """

    ok: bool
    delete: bool = False
    reason: str = ""


def valid() -> CacheValidation:
    """Return a passing :class:`CacheValidation`."""
    return CacheValidation(ok=True)


def invalid(*, delete: bool, reason: str = "") -> CacheValidation:
    """Return a failing :class:`CacheValidation`.

    Args:
        delete: Whether the stale cache file should be deleted on this failure.
        reason: Short reason string used in the deletion log message.
    """
    return CacheValidation(ok=False, delete=delete, reason=reason)


# A validator inspects the loaded cache dict and returns a ``CacheValidation``.
PickleCacheValidator = Callable[[Dict[str, Any]], CacheValidation]


def delete_cache_file(path: Path, reason: str, logger: logging.Logger, *, label: str) -> None:
    """Delete a stale or corrupt cache file, logging success and failure.

    Args:
        path: Cache file path.
        reason: Reason for deletion (included in the log message).
        logger: Logger used for info/warning output.
        label: Human-readable prefix identifying the cache (e.g. the collection
            name) so log messages match the original call sites.
    """
    try:
        if path.exists():
            path.unlink()
            logger.info(f"{label}: deleted stale cache ({reason})")
    except Exception as e:  # noqa: BLE001 - deletion is best-effort
        logger.warning(f"{label}: failed to delete cache: {e}")


def load_validated_pickle(
    path: Path,
    *,
    validators: List[PickleCacheValidator],
    logger: logging.Logger,
    label: str,
    missing_message: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Load a metadata-wrapped pickle cache, validating it before returning.

    Reproduces the shared load-validate flow: the file is unpickled, then each
    validator runs in order. The first failing validator stops evaluation; if it
    requests deletion the file is removed. Unpickling errors and generic load
    errors always delete the file. On any failure ``None`` is returned so the
    caller can rebuild.

    Args:
        path: Cache file path.
        validators: Ordered validators. Each receives the loaded cache ``dict``
            and returns a :class:`CacheValidation`. Order and per-validator
            delete behavior fully determine the invalidation semantics.
        logger: Logger for debug/info/warning output.
        label: Human-readable prefix identifying the cache in log messages.
        missing_message: Optional debug message logged when the file is absent.

    Returns:
        The validated cache ``dict`` if every validator passed, otherwise
        ``None``.
    """
    if not path.exists():
        if missing_message:
            logger.debug(missing_message)
        return None

    try:
        with open(path, "rb") as f:
            cache_data = pickle.load(f)

        for validator in validators:
            result = validator(cache_data)
            if not result.ok:
                if result.delete:
                    delete_cache_file(path, result.reason, logger, label=label)
                return None

        return cache_data

    except (pickle.UnpicklingError, EOFError) as e:
        logger.warning(f"{label}: cache file corrupted: {e}")
        delete_cache_file(path, "unpickling error", logger, label=label)
        return None
    except Exception as e:  # noqa: BLE001 - any load failure is recoverable
        logger.warning(f"{label}: failed to load cache: {e}")
        delete_cache_file(path, "load error", logger, label=label)
        return None


def save_pickle_with_metadata(
    path: Path,
    data: Dict[str, Any],
    *,
    logger: logging.Logger,
    label: str,
) -> bool:
    """Pickle ``data`` to ``path``, swallowing and logging any write error.

    Args:
        path: Cache file path.
        data: The metadata-wrapped payload dict to persist.
        logger: Logger for warning output on failure.
        label: Human-readable prefix identifying the cache in log messages.

    Returns:
        True if the file was written, False if saving failed.
    """
    try:
        with open(path, "wb") as f:
            pickle.dump(data, f)
        return True
    except Exception as e:  # noqa: BLE001 - saving is best-effort
        logger.warning(f"{label}: failed to save cache: {e}")
        return False

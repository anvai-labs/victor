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

"""Atomic, data-only persistence for rebuildable metadata and embedding caches.

Legacy pickle files are not deserialized. Numeric NumPy arrays use a raw byte
buffer with validated dtype and shape; other values use explicit JSON data tags.
"""

from __future__ import annotations

import logging
import base64
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

__all__ = [
    "CacheValidation",
    "valid",
    "invalid",
    "CacheValidator",
    "load_validated_data",
    "save_data_with_metadata",
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
CacheValidator = Callable[[Dict[str, Any]], CacheValidation]


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


def load_validated_data(
    path: Path,
    *,
    validators: List[CacheValidator],
    logger: logging.Logger,
    label: str,
    missing_message: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Load a metadata-wrapped data cache, validating it before returning.

    Reproduces the shared load-validate flow: the file is decoded as data, then each
    validator runs in order. The first failing validator stops evaluation; if it
    requests deletion the file is removed. Decoding errors and generic load
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
        cache_data = read_cache_data(path)
        if not isinstance(cache_data, dict):
            raise ValueError("Cache metadata must be a dictionary")

        for validator in validators:
            result = validator(cache_data)
            if not result.ok:
                if result.delete:
                    delete_cache_file(path, result.reason, logger, label=label)
                return None

        return cache_data

    except Exception as e:  # noqa: BLE001 - any load failure is recoverable
        logger.warning(f"{label}: failed to load cache: {e}")
        delete_cache_file(path, "load error", logger, label=label)
        return None


def save_data_with_metadata(
    path: Path,
    data: Dict[str, Any],
    *,
    logger: logging.Logger,
    label: str,
) -> bool:
    """Atomically write data to ``path``, logging any write error.

    Args:
        path: Cache file path.
        data: The metadata-wrapped payload dict to persist.
        logger: Logger for warning output on failure.
        label: Human-readable prefix identifying the cache in log messages.

    Returns:
        True if the file was written, False if saving failed.
    """
    try:
        write_cache_data(path, data)
        return True
    except Exception as e:  # noqa: BLE001 - saving is best-effort
        logger.warning(f"{label}: failed to save cache: {e}")
        return False


def _encode_data(value: Any) -> Any:
    if value is None or type(value) in (bool, int, float, str):
        return value
    if type(value) is bytes:
        return ["bytes", base64.b64encode(value).decode("ascii")]
    if type(value) in (list, tuple):
        return [type(value).__name__, [_encode_data(item) for item in value]]
    if type(value) is dict:
        return ["dict", [[_encode_data(k), _encode_data(v)] for k, v in value.items()]]
    # Keep numpy off the scalar-only load path. Never load a class named by data.
    import numpy as np

    if type(value) is np.ndarray and value.dtype.kind in "biufc":
        return [
            "ndarray",
            {
                "dtype": value.dtype.str,
                "shape": list(value.shape),
                "bytes": base64.b64encode(value.tobytes()).decode("ascii"),
            },
        ]
    raise TypeError(f"Unsupported cache data type: {type(value).__name__}")


def _decode_data(value: Any) -> Any:
    if value is None or type(value) in (bool, int, float, str):
        return value
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("Malformed cache value")
    tag, payload = value
    if tag in ("list", "tuple") and isinstance(payload, list):
        items = [_decode_data(item) for item in payload]
        return tuple(items) if tag == "tuple" else items
    if tag == "dict" and isinstance(payload, list):
        return {_decode_data(k): _decode_data(v) for k, v in payload}
    if tag == "bytes" and isinstance(payload, str):
        return base64.b64decode(payload, validate=True)
    if tag == "ndarray" and isinstance(payload, dict):
        import numpy as np

        shape = payload.get("shape")
        if (
            not isinstance(payload.get("dtype"), str)
            or not isinstance(shape, list)
            or len(shape) > 32
            or any(type(size) is not int or size < 0 for size in shape)
        ):
            raise ValueError("Invalid cache array metadata")
        dtype = np.dtype(payload["dtype"])
        if dtype.kind not in "biufc":
            raise ValueError("Cache arrays must contain numeric data")
        raw = base64.b64decode(payload["bytes"], validate=True)
        # frombuffer/reshape validate the exact byte count without allocating an
        # array sized by an untrusted shape. No pickle or Python object loader.
        return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
    raise ValueError("Unknown cache data tag")


def read_cache_data(path: Path) -> Any:
    """Read data without constructing arbitrary Python objects."""
    with path.open(encoding="utf-8") as stream:
        envelope = json.load(stream)
    if not isinstance(envelope, dict) or envelope.get("format") != "victor-data-cache-v1":
        raise ValueError("Unknown cache format; legacy pickle must be rebuilt")
    return _decode_data(envelope["data"])


def write_cache_data(path: Path, data: Any) -> None:
    """Atomically replace a cache with a private file, preserving it on failure."""
    encoded = json.dumps(
        {"format": "victor-data-cache-v1", "data": _encode_data(data)},
        allow_nan=False,
        separators=(",", ":"),
    )
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            temporary = stream.name
            stream.write(encoded)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)

"""Deprecated cache helper names; persisted data now uses JSON, never pickle.

Internal consumers use :mod:`victor.core.data_cache`. These aliases preserve the
old import surface while refusing to deserialize legacy Python objects.
"""

from victor.core.data_cache import (
    CacheValidation,
    CacheValidator as PickleCacheValidator,
    delete_cache_file,
    invalid,
    load_validated_data as load_validated_pickle,
    save_data_with_metadata as save_pickle_with_metadata,
    valid,
)

__all__ = [
    "CacheValidation",
    "PickleCacheValidator",
    "delete_cache_file",
    "invalid",
    "load_validated_pickle",
    "save_pickle_with_metadata",
    "valid",
]

"""Strict lifecycle cleanup is additive to legacy warning-only close behavior."""

from unittest.mock import MagicMock

import pytest

from victor.storage.cache.config import CacheConfig
from victor.storage.cache.tiered_cache import TieredCache


@pytest.mark.parametrize("strict", [False, True])
def test_disk_close_failure_policy(tmp_path, strict, caplog):
    cache = TieredCache(CacheConfig(disk_path=tmp_path))
    disk = cache._disk_cache
    real_close = disk.close
    disk.close = MagicMock(side_effect=OSError("disk close failed"))
    try:
        if strict:
            with pytest.raises(OSError, match="disk close failed"):
                cache.close(strict=True)
        else:
            cache.close()
            assert "Error closing disk cache" in caplog.text
    finally:
        real_close()

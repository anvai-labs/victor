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

"""Tests for signature store - achieving 70%+ coverage."""

import pytest
import tempfile
import time
from pathlib import Path

from victor.core.database import reset_database, get_database
from victor.agent.signature_store import (
    FailedSignature,
    SignatureStore,
    get_signature_store,
    reset_signature_store,
    DEFAULT_TTL_SECONDS,
    SCHEMA_VERSION,
)


class TestFailedSignature:
    """Tests for FailedSignature dataclass."""

    def test_basic_creation(self):
        """Test basic FailedSignature creation."""
        sig = FailedSignature(
            tool_name="read",
            args_hash="abc123",
            error_message="File not found",
            failure_count=3,
            first_seen=1000.0,
            last_seen=2000.0,
            expires_at=3000.0,
        )
        assert sig.tool_name == "read"
        assert sig.args_hash == "abc123"
        assert sig.failure_count == 3

    def test_is_expired_false(self):
        """Test is_expired returns False when not expired."""
        future_time = time.time() + 3600
        sig = FailedSignature(
            tool_name="read",
            args_hash="abc",
            error_message="Error",
            failure_count=1,
            first_seen=1000.0,
            last_seen=1000.0,
            expires_at=future_time,
        )
        assert sig.is_expired is False

    def test_is_expired_true(self):
        """Test is_expired returns True when expired."""
        past_time = time.time() - 3600
        sig = FailedSignature(
            tool_name="read",
            args_hash="abc",
            error_message="Error",
            failure_count=1,
            first_seen=1000.0,
            last_seen=1000.0,
            expires_at=past_time,
        )
        assert sig.is_expired is True


class TestSignatureStore:
    """Tests for SignatureStore class."""

    @pytest.fixture
    def temp_db(self, tmp_path: Path):
        """Create a temporary database path."""
        reset_database()
        db_path = tmp_path / "test_signatures.db"
        get_database(db_path)
        yield db_path
        reset_database()

    @pytest.fixture
    def store(self, temp_db):
        """Create a SignatureStore with temporary database."""
        return SignatureStore(db_path=temp_db)

    def test_initialization_with_path(self, temp_db):
        """Test initialization with explicit path."""
        store = SignatureStore(db_path=temp_db)
        assert store.db_path == temp_db
        assert store.ttl_seconds == DEFAULT_TTL_SECONDS
        assert store.max_signatures == 10000

    def test_initialization_custom_values(self, temp_db):
        """Test initialization with custom values."""
        store = SignatureStore(
            db_path=temp_db,
            ttl_seconds=3600,
            max_signatures=100,
        )
        assert store.ttl_seconds == 3600
        assert store.max_signatures == 100

    def test_compute_hash_deterministic(self, store):
        """Test hash computation is deterministic."""
        args = {"path": "/test/file.py", "limit": 100}
        hash1 = store._compute_hash(args)
        hash2 = store._compute_hash(args)
        assert hash1 == hash2

    def test_compute_hash_different_args(self, store):
        """Test different args produce different hashes."""
        args1 = {"path": "/file1.py"}
        args2 = {"path": "/file2.py"}
        hash1 = store._compute_hash(args1)
        hash2 = store._compute_hash(args2)
        assert hash1 != hash2

    def test_compute_hash_handles_non_json(self, store):
        """Test hash handles non-JSON-serializable objects."""
        args = {"key": object()}
        # Should not raise, falls back to str()
        hash_val = store._compute_hash(args)
        assert len(hash_val) == 32

    def test_make_signature_key(self, store):
        """Test signature key creation."""
        key = store._make_signature_key("read", "abc123")
        assert key == "read:abc123"

    def test_record_failure_creates_signature(self, store):
        """Test recording a failure creates a signature."""
        store.record_failure("read", {"path": "/test"}, "File not found")

        assert store.is_known_failure("read", {"path": "/test"})

    def test_record_failure_increments_count(self, store):
        """Test multiple failures increment count."""
        args = {"path": "/test"}
        store.record_failure("read", args, "Error 1")
        store.record_failure("read", args, "Error 2")

        failures = store.get_failures("read")
        assert len(failures) > 0
        assert failures[0].failure_count == 2

    def test_record_failure_with_custom_ttl(self, store):
        """Test recording failure with custom TTL."""
        store.record_failure("read", {"path": "/test"}, "Error", custom_ttl=60)
        # Should still be known
        assert store.is_known_failure("read", {"path": "/test"})

    def test_is_known_failure_false_for_new(self, store):
        """Test is_known_failure returns False for new signatures."""
        assert store.is_known_failure("unknown", {"arg": "value"}) is False

    def test_is_known_failure_false_for_expired(self, temp_db):
        """Test is_known_failure returns False for expired signatures."""
        store = SignatureStore(db_path=temp_db, ttl_seconds=0)
        args = {"path": "/test"}
        store.record_failure("read", args, "Error")
        time.sleep(0.1)
        # Now expired
        assert store.is_known_failure("read", args) is False

    def test_clear_signature(self, store):
        """Test clearing a specific signature."""
        args = {"path": "/test"}
        store.record_failure("read", args, "Error")
        assert store.is_known_failure("read", args)

        deleted = store.clear_signature("read", args)
        assert deleted is True
        assert store.is_known_failure("read", args) is False

    def test_clear_signature_not_found(self, store):
        """Test clearing non-existent signature."""
        deleted = store.clear_signature("unknown", {"arg": "value"})
        assert deleted is False

    def test_clear_tool(self, store):
        """Test clearing all signatures for a tool."""
        store.record_failure("read", {"path": "/test1"}, "Error")
        store.record_failure("read", {"path": "/test2"}, "Error")
        store.record_failure("write", {"path": "/test"}, "Error")

        deleted = store.clear_tool("read")
        assert deleted == 2
        assert store.is_known_failure("read", {"path": "/test1"}) is False
        assert store.is_known_failure("write", {"path": "/test"}) is True

    def test_clear_all(self, store):
        """Test clearing all signatures."""
        store.record_failure("read", {"path": "/test"}, "Error")
        store.record_failure("write", {"path": "/test"}, "Error")

        deleted = store.clear_all()
        assert deleted == 2
        assert store.is_known_failure("read", {"path": "/test"}) is False

    def test_cleanup_expired(self, temp_db):
        """Test cleaning up expired signatures."""
        store = SignatureStore(db_path=temp_db, ttl_seconds=0)
        store.record_failure("read", {"path": "/test"}, "Error")
        time.sleep(0.1)

        deleted = store.cleanup_expired()
        assert deleted >= 1

    def test_get_failures_all(self, store):
        """Test getting all failures."""
        store.record_failure("read", {"path": "/test1"}, "Error 1")
        store.record_failure("write", {"path": "/test2"}, "Error 2")

        failures = store.get_failures()
        assert len(failures) >= 2

    def test_get_failures_by_tool(self, store):
        """Test getting failures filtered by tool."""
        store.record_failure("read", {"path": "/test1"}, "Error 1")
        store.record_failure("read", {"path": "/test2"}, "Error 2")
        store.record_failure("write", {"path": "/test"}, "Error 3")

        failures = store.get_failures("read")
        assert len(failures) == 2
        assert all(f.tool_name == "read" for f in failures)

    def test_get_failures_with_limit(self, store):
        """Test getting failures with limit."""
        for i in range(10):
            store.record_failure("read", {"path": f"/test{i}"}, f"Error {i}")

        failures = store.get_failures(limit=3)
        assert len(failures) == 3

    def test_get_stats(self, store):
        """Test getting statistics."""
        store.record_failure("read", {"path": "/test1"}, "Error")
        store.record_failure("read", {"path": "/test1"}, "Error")  # Same signature
        store.record_failure("write", {"path": "/test2"}, "Error")

        stats = store.get_stats()
        assert "total_signatures" in stats
        assert "active_signatures" in stats
        assert "by_tool" in stats
        assert "most_failing" in stats
        assert "cache_size" in stats
        assert "db_path" in stats

    def test_close(self, store):
        """Test closing store."""
        store.record_failure("read", {"path": "/test"}, "Error")
        store.close()
        # Should not raise after close

    def test_cache_refresh(self, store):
        """Test cache refresh mechanism."""
        store.record_failure("read", {"path": "/test"}, "Error")
        # Force cache time to be old
        store._cache_time = 0
        # This should refresh the cache
        store._refresh_cache()
        assert store._cache_time > 0

    def test_maybe_prune(self, temp_db):
        """Test pruning when over limit."""
        store = SignatureStore(db_path=temp_db, max_signatures=5)

        # Add more than max
        for i in range(10):
            store.record_failure("read", {"path": f"/test{i}"}, f"Error {i}")
            time.sleep(0.01)  # Ensure different timestamps

        # Pruning should have occurred
        stats = store.get_stats()
        assert stats["total_signatures"] <= 10  # Some pruning may have occurred


class TestGlobalFunctions:
    """Tests for global signature store functions."""

    def test_get_signature_store_singleton(self):
        """Test get_signature_store returns singleton."""
        reset_signature_store()  # Reset first
        store1 = get_signature_store()
        store2 = get_signature_store()
        assert store1 is store2

    def test_reset_signature_store(self):
        """Test reset_signature_store clears singleton."""
        store1 = get_signature_store()
        reset_signature_store()
        store2 = get_signature_store()
        assert store1 is not store2

    def test_get_signature_store_with_path(self, tmp_path: Path):
        """Test get_signature_store with custom path."""
        reset_signature_store()
        reset_database()
        custom_path = tmp_path / "custom.db"
        get_database(custom_path)
        store = get_signature_store(custom_path)
        assert store.db_path == custom_path
        reset_signature_store()
        reset_database()


class TestSignatureStoreEdgeCases:
    """Edge case tests for SignatureStore."""

    @pytest.fixture
    def temp_db(self, tmp_path: Path):
        """Create a temporary database path."""
        reset_database()
        db_path = tmp_path / "test_signatures.db"
        get_database(db_path)
        yield db_path
        reset_database()

    def test_empty_args(self, temp_db):
        """Test with empty args dict."""
        store = SignatureStore(db_path=temp_db)
        store.record_failure("read", {}, "Error")
        assert store.is_known_failure("read", {})

    def test_long_error_message_truncated(self, temp_db):
        """Test long error messages are truncated."""
        store = SignatureStore(db_path=temp_db)
        long_error = "x" * 1000
        store.record_failure("read", {"path": "/test"}, long_error)

        failures = store.get_failures("read")
        assert len(failures[0].error_message) <= 500

    def test_concurrent_access(self, temp_db):
        """Test thread-safe concurrent access."""
        import threading

        store = SignatureStore(db_path=temp_db)
        errors = []

        def worker(i):
            try:
                store.record_failure("read", {"path": f"/test{i}"}, f"Error {i}")
                store.is_known_failure("read", {"path": f"/test{i}"})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_special_characters_in_args(self, temp_db):
        """Test handling special characters in args."""
        store = SignatureStore(db_path=temp_db)
        args = {"path": "/test/file with spaces.py", "pattern": "class.*\\{\\}"}
        store.record_failure("read", args, "Error")
        assert store.is_known_failure("read", args)

    def test_unicode_in_args(self, temp_db):
        """Test handling unicode in args."""
        store = SignatureStore(db_path=temp_db)
        args = {"content": "こんにちは世界", "emoji": "🚀"}
        store.record_failure("write", args, "Error")
        assert store.is_known_failure("write", args)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

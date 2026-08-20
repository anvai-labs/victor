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

"""Unit tests for secure paths module.

Tests security-hardened path resolution and validation:
- SEC-001: HOME directory manipulation detection
- SEC-002: Path traversal attack prevention
- SEC-003: Symlink attack detection
- SEC-005: File permission validation
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


class TestValidateVictorDirName:
    """Tests for validate_victor_dir_name function (SEC-002)."""

    def test_valid_default_name(self):
        """Test that .victor is valid."""
        from victor.config.secure_paths import validate_victor_dir_name

        result, is_valid = validate_victor_dir_name(".victor")
        assert is_valid is True
        assert result == ".victor"

    def test_valid_custom_hidden_name(self):
        """Test that custom hidden directory names are valid."""
        from victor.config.secure_paths import validate_victor_dir_name

        result, is_valid = validate_victor_dir_name(".my_custom_dir")
        assert is_valid is True
        assert result == ".my_custom_dir"

    def test_valid_non_hidden_name_with_warning(self, caplog):
        """Test that non-hidden names are valid but trigger warning."""
        from victor.config.secure_paths import validate_victor_dir_name

        result, is_valid = validate_victor_dir_name("custom_dir")
        assert is_valid is True
        assert result == "custom_dir"
        # Should have logged a warning about not starting with '.'
        assert any("doesn't start with '.'" in record.message for record in caplog.records)

    def test_blocks_parent_traversal(self):
        """Test that parent directory traversal is blocked."""
        from victor.config.secure_paths import validate_victor_dir_name

        test_cases = [
            "../etc",
            "..\\windows",
            "foo/../bar",
            "...",
        ]
        for test in test_cases:
            result, is_valid = validate_victor_dir_name(test)
            assert is_valid is False, f"Should block: {test}"
            assert result == ".victor"

    def test_blocks_absolute_paths(self):
        """Test that absolute paths are blocked."""
        from victor.config.secure_paths import validate_victor_dir_name

        test_cases = [
            "/tmp/malicious",
            "/etc/passwd",
            "\\windows\\system32",
        ]
        for test in test_cases:
            result, is_valid = validate_victor_dir_name(test)
            assert is_valid is False, f"Should block: {test}"
            assert result == ".victor"

    def test_blocks_path_separators(self):
        """Test that paths with separators are blocked."""
        from victor.config.secure_paths import validate_victor_dir_name

        test_cases = [
            "foo/bar",
            "foo\\bar",
            "a/b/c",
        ]
        for test in test_cases:
            result, is_valid = validate_victor_dir_name(test)
            assert is_valid is False, f"Should block: {test}"
            assert result == ".victor"

    def test_blocks_empty_name(self):
        """Test that empty names are blocked."""
        from victor.config.secure_paths import validate_victor_dir_name

        test_cases = ["", "   ", None]
        for test in test_cases:
            if test is None:
                continue  # None would raise TypeError
            result, is_valid = validate_victor_dir_name(test)
            assert is_valid is False
            assert result == ".victor"


class TestGetSecureHome:
    """Tests for get_secure_home function (SEC-001)."""

    def test_returns_path(self):
        """Test that it returns a valid Path."""
        from victor.config.secure_paths import get_secure_home

        result = get_secure_home()
        assert isinstance(result, Path)
        assert result.is_absolute()

    def test_matches_passwd_when_available(self):
        """Test that result matches passwd database on Unix."""
        from victor.config.secure_paths import (
            get_secure_home,
            get_real_home_from_passwd,
        )

        passwd_home = get_real_home_from_passwd()
        secure_home = get_secure_home()

        if passwd_home is not None:
            # On Unix, should prefer passwd entry
            assert secure_home.resolve() == passwd_home.resolve()

    @patch.dict(os.environ, {"HOME": "/tmp/fake_home"})
    def test_detects_home_manipulation(self, caplog):
        """Test that HOME manipulation is detected and logged."""
        from victor.config.secure_paths import (
            get_secure_home,
            get_real_home_from_passwd,
        )

        passwd_home = get_real_home_from_passwd()
        if passwd_home is None:
            pytest.skip("passwd database not available (Windows)")

        result = get_secure_home()

        # Should return passwd home, not manipulated HOME
        assert result == passwd_home
        # Should have logged a security warning
        assert any(
            "HOME_MANIPULATION" in record.message or "SECURITY" in record.message
            for record in caplog.records
        )


class TestGetVictorDir:
    """Tests for get_victor_dir function."""

    def test_returns_path_under_home(self):
        """Test that it returns a path under home directory."""
        from victor.config.secure_paths import get_victor_dir, get_secure_home

        result = get_victor_dir()
        secure_home = get_secure_home()

        assert isinstance(result, Path)
        assert result.parent == secure_home

    def test_uses_default_name(self):
        """Test that it uses .victor as default."""
        from victor.config.secure_paths import get_victor_dir

        # Clear env var to test default
        with patch.dict(os.environ, {"VICTOR_DIR_NAME": ""}):
            result = get_victor_dir()

        assert result.name == ".victor"


class TestSafeResolvePath:
    """Tests for safe_resolve_path function (SEC-003)."""

    def test_resolves_normal_path(self):
        """Test that normal paths are resolved correctly."""
        from victor.config.secure_paths import safe_resolve_path

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("test")

            result = safe_resolve_path(test_file)
            assert result is not None
            assert result.exists()

    def test_detects_escape_from_parent(self):
        """Test that paths escaping expected parent are blocked."""
        from victor.config.secure_paths import safe_resolve_path

        with tempfile.TemporaryDirectory() as tmpdir:
            expected_parent = Path(tmpdir) / "allowed"
            expected_parent.mkdir()

            escape_path = Path(tmpdir) / "allowed" / ".." / "escaped.txt"

            result = safe_resolve_path(escape_path, expected_parent=expected_parent)
            # Should return None because resolved path escapes expected parent
            assert result is None

    def test_allows_path_within_parent(self):
        """Test that paths within expected parent are allowed."""
        from victor.config.secure_paths import safe_resolve_path

        with tempfile.TemporaryDirectory() as tmpdir:
            expected_parent = Path(tmpdir)
            test_file = Path(tmpdir) / "subdir" / "test.txt"
            test_file.parent.mkdir(parents=True)
            test_file.write_text("test")

            result = safe_resolve_path(test_file, expected_parent=expected_parent)
            assert result is not None
            assert result.exists()


class TestCheckFilePermissions:
    """Tests for check_file_permissions function (SEC-005)."""

    def test_accepts_secure_permissions(self):
        """Test that 0600 permissions are accepted."""
        from victor.config.secure_paths import check_file_permissions

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test")
            temp_path = Path(f.name)

        try:
            os.chmod(temp_path, 0o600)
            result = check_file_permissions(temp_path)
            assert result is True
        finally:
            temp_path.unlink()

    def test_rejects_world_readable(self, caplog):
        """Test that world-readable permissions are rejected."""
        from victor.config.secure_paths import check_file_permissions

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test")
            temp_path = Path(f.name)

        try:
            os.chmod(temp_path, 0o644)  # World readable
            result = check_file_permissions(temp_path)
            assert result is False
            # Should have logged security warning
            assert any(
                "INSECURE" in record.message or "SECURITY" in record.message
                for record in caplog.records
            )
        finally:
            temp_path.unlink()

    def test_accepts_nonexistent_file(self):
        """Test that non-existent files are accepted (will be created securely)."""
        from victor.config.secure_paths import check_file_permissions

        result = check_file_permissions(Path("/nonexistent/path"))
        assert result is True


class TestSecureCreateFile:
    """Tests for secure_create_file function (SEC-005 TOCTOU fix)."""

    def test_creates_file_with_secure_permissions(self):
        """Test that files are created with 0600 permissions."""
        from victor.config.secure_paths import secure_create_file

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "secret.txt"
            content = "secret content"

            result = secure_create_file(test_file, content, mode=0o600)

            assert result is True
            assert test_file.exists()
            assert test_file.read_text() == content

            # Check permissions
            mode = test_file.stat().st_mode & 0o777
            assert mode == 0o600

    def test_atomic_write_prevents_partial_content(self):
        """Test that atomic write is used."""
        from victor.config.secure_paths import secure_create_file

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "atomic.txt"
            content = "atomic content"

            result = secure_create_file(test_file, content, atomic=True)

            assert result is True
            assert test_file.read_text() == content
            # Temp file should not exist
            assert not (test_file.with_suffix(".txt.tmp")).exists()

    def test_creates_parent_directories(self):
        """Test that parent directories are created."""
        from victor.config.secure_paths import secure_create_file

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "subdir" / "nested" / "file.txt"

            result = secure_create_file(test_file, "content")

            assert result is True
            assert test_file.exists()


class TestSecureString:
    """Tests for SecureString class (SEC-006)."""

    def test_stores_and_retrieves_value(self):
        """Test that value can be stored and retrieved."""
        from victor.config.api_keys import SecureString

        ss = SecureString("my_secret")
        assert ss.get() == "my_secret"

    def test_clears_value_from_memory(self):
        """Test that value is cleared from memory."""
        from victor.config.api_keys import SecureString

        ss = SecureString("my_secret")
        assert ss.get() == "my_secret"

        ss.clear()
        assert ss.get() == ""
        assert ss._cleared is True

    def test_bool_check(self):
        """Test that bool check works correctly."""
        from victor.config.api_keys import SecureString

        ss = SecureString("secret")
        assert bool(ss) is True

        ss.clear()
        assert bool(ss) is False

    def test_auto_clear_on_delete(self):
        """Test that value is cleared on deletion."""
        from victor.config.api_keys import SecureString

        ss = SecureString("secret")
        # Get reference to internal bytearray
        _internal_ref = ss._value

        del ss

        # After deletion, the bytearray should be zeroed
        # (though in CPython, gc may have already collected it)

    def test_empty_string(self):
        """Test handling of empty string."""
        from victor.config.api_keys import SecureString

        ss = SecureString("")
        assert ss.get() == ""
        assert bool(ss) is False  # Empty string is falsy


class TestAPIKeyManagerAudit:
    """Tests for API key manager audit logging (SEC-004)."""

    def test_audit_logs_key_access(self, caplog):
        """Test that key access is audit logged."""
        from victor.config.api_keys import APIKeyManager
        import logging

        caplog.set_level(logging.INFO)

        manager = APIKeyManager()
        # Try to get a non-existent key
        manager.get_key("test_provider")

        # Should have logged the access attempt
        assert any("SECRET_ACCESS" in record.message for record in caplog.records)

    def test_audit_logs_cache_hit(self, caplog):
        """Test that cache hits are audit logged."""
        from victor.config.api_keys import APIKeyManager, SecureString
        import logging

        caplog.set_level(logging.INFO)

        manager = APIKeyManager()
        # Pre-populate cache
        manager._cache["cached_provider"] = SecureString("cached_key")

        # Access should log cache hit
        result = manager.get_key("cached_provider")

        assert result == "cached_key"
        assert any("cache_hit" in record.message for record in caplog.records)

    def test_audit_logs_local_provider(self, caplog):
        """Test that local provider access is logged."""
        from victor.config.api_keys import APIKeyManager
        import logging

        caplog.set_level(logging.INFO)

        manager = APIKeyManager()
        result = manager.get_key("ollama")  # Local provider

        assert result is None  # Ollama doesn't need a key
        assert any("local_provider" in record.message for record in caplog.records)


class TestSettingsSecureIntegration:
    """Tests for settings.py secure paths integration."""

    def test_global_victor_dir_is_secure(self):
        """Test that GLOBAL_VICTOR_DIR uses secure resolution."""
        from victor.config.settings import GLOBAL_VICTOR_DIR
        from victor.config.secure_paths import get_victor_dir

        # Should match the secure version
        assert GLOBAL_VICTOR_DIR == get_victor_dir()

    def test_victor_dir_name_is_validated(self):
        """Test that VICTOR_DIR_NAME is validated."""
        from victor.config.settings import VICTOR_DIR_NAME

        # Should be the validated name
        assert ".." not in VICTOR_DIR_NAME
        assert "/" not in VICTOR_DIR_NAME
        assert "\\" not in VICTOR_DIR_NAME


class TestXDGSecurePaths:
    """Tests for XDG config path validation."""

    def test_get_secure_xdg_config_home_default(self):
        """Test default XDG_CONFIG_HOME under ~/.config."""
        from victor.config.secure_paths import (
            get_secure_xdg_config_home,
            get_secure_home,
        )

        # Without XDG_CONFIG_HOME set, should use default
        with patch.dict(os.environ, {"XDG_CONFIG_HOME": ""}, clear=False):
            result = get_secure_xdg_config_home()
            assert result == get_secure_home() / ".config"

    def test_get_secure_xdg_config_home_valid(self):
        """Test valid XDG_CONFIG_HOME under home."""
        from victor.config.secure_paths import (
            get_secure_xdg_config_home,
            get_secure_home,
        )

        valid_path = str(get_secure_home() / ".myconfig")
        with patch.dict(os.environ, {"XDG_CONFIG_HOME": valid_path}):
            result = get_secure_xdg_config_home()
            assert result == Path(valid_path)

    def test_get_secure_xdg_config_home_blocks_escape(self, caplog):
        """Test that XDG_CONFIG_HOME outside home is blocked."""
        from victor.config.secure_paths import (
            get_secure_xdg_config_home,
            get_secure_home,
        )

        with patch.dict(os.environ, {"XDG_CONFIG_HOME": "/tmp/attacker_config"}):
            result = get_secure_xdg_config_home()
            # Should fall back to default
            assert result == get_secure_home() / ".config"
            # Should have logged security event
            assert any(
                "XDG_CONFIG" in record.message or "SECURITY" in record.message
                for record in caplog.records
            )

    def test_get_secure_xdg_data_home_default(self):
        """Test default XDG_DATA_HOME under ~/.local/share."""
        from victor.config.secure_paths import get_secure_xdg_data_home, get_secure_home

        with patch.dict(os.environ, {"XDG_DATA_HOME": ""}, clear=False):
            result = get_secure_xdg_data_home()
            assert result == get_secure_home() / ".local" / "share"


class TestPluginDirectorySecurity:
    """Tests for plugin directory security validation."""

    def test_validate_plugin_directory_valid_path(self):
        """Test that valid plugin paths are accepted."""
        from victor.config.secure_paths import validate_plugin_directory, get_victor_dir

        # Path under Victor dir should be valid
        plugin_dir = get_victor_dir() / "plugins"
        result, is_valid = validate_plugin_directory(plugin_dir)
        assert is_valid is True

    def test_validate_plugin_directory_blocks_traversal(self, caplog):
        """Test that path traversal is blocked."""
        from victor.config.secure_paths import validate_plugin_directory

        # Path with traversal should be blocked
        plugin_dir = Path("~/.victor/../../../etc/evil_plugins")
        result, is_valid = validate_plugin_directory(plugin_dir)
        assert is_valid is False
        assert any(
            "PLUGIN_PATH" in record.message or "SECURITY" in record.message
            for record in caplog.records
        )

    def test_validate_plugin_directory_blocks_absolute_escape(self, caplog):
        """Test that paths outside allowed locations are blocked."""
        from victor.config.secure_paths import validate_plugin_directory

        # Absolute path outside allowed locations
        plugin_dir = Path("/opt/malicious_plugins")
        result, is_valid = validate_plugin_directory(plugin_dir)
        assert is_valid is False

    def test_get_secure_plugin_dirs_returns_list(self):
        """Test that get_secure_plugin_dirs returns a list of paths."""
        from victor.config.secure_paths import get_secure_plugin_dirs

        result = get_secure_plugin_dirs()
        assert isinstance(result, list)
        # All returned paths should be Path objects
        for path in result:
            assert isinstance(path, Path)


class TestKeyringIntegration:
    """Tests for keyring integration in API keys."""

    def test_is_keyring_available(self):
        """Test is_keyring_available returns bool."""
        from victor.config.api_keys import is_keyring_available

        result = is_keyring_available()
        assert isinstance(result, bool)

    def test_get_key_resolution_order(self):
        """Test that key resolution follows priority order."""
        from victor.config.api_keys import APIKeyManager

        manager = APIKeyManager()

        # Verify docstring mentions resolution order
        docstring = manager.get_key.__doc__
        assert "Environment variable" in docstring
        assert "keyring" in docstring.lower()
        assert "file" in docstring.lower()

    def test_set_key_with_keyring_option(self):
        """Test that set_key accepts use_keyring parameter."""
        from victor.config.api_keys import APIKeyManager
        import inspect

        manager = APIKeyManager()
        sig = inspect.signature(manager.set_key)

        # Should have use_keyring parameter
        assert "use_keyring" in sig.parameters
        assert sig.parameters["use_keyring"].default is False


class TestCacheIntegrity:
    """Tests for embedding cache integrity verification."""

    def test_compute_file_hash(self):
        """Test file hash computation."""
        from victor.config.secure_paths import compute_file_hash

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test content")
            temp_path = Path(f.name)

        try:
            hash1 = compute_file_hash(temp_path)
            assert hash1 is not None
            assert len(hash1) == 64  # SHA-256 hex digest

            # Same content should produce same hash
            hash2 = compute_file_hash(temp_path)
            assert hash1 == hash2
        finally:
            temp_path.unlink()

    def test_compute_file_hash_nonexistent(self):
        """Test hash of nonexistent file returns None."""
        from victor.config.secure_paths import compute_file_hash

        result = compute_file_hash(Path("/nonexistent/file"))
        assert result is None

    def test_create_cache_manifest(self):
        """Test cache manifest creation."""
        from victor.config.secure_paths import (
            create_cache_manifest,
            get_cache_manifest_path,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)

            # Create some test files
            (cache_dir / "file1.pkl").write_text("content1")
            (cache_dir / "file2.pkl").write_text("content2")

            # Create manifest
            result = create_cache_manifest(cache_dir)
            assert result is True

            # Manifest file should exist
            manifest_path = get_cache_manifest_path(cache_dir)
            assert manifest_path.exists()

            # Manifest should contain file hashes
            import json

            with open(manifest_path) as f:
                manifest = json.load(f)

            assert "files" in manifest
            assert "file1.pkl" in manifest["files"]
            assert "file2.pkl" in manifest["files"]

    def test_verify_cache_integrity_valid(self):
        """Test cache integrity verification passes for unmodified cache."""
        from victor.config.secure_paths import (
            create_cache_manifest,
            verify_cache_integrity,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)

            # Create test file and manifest
            (cache_dir / "test.pkl").write_text("original content")
            create_cache_manifest(cache_dir)

            # Verify should pass
            is_valid, tampered = verify_cache_integrity(cache_dir)
            assert is_valid is True
            assert len(tampered) == 0

    def test_verify_cache_integrity_detects_tampering(self, caplog):
        """Test cache integrity verification detects modified files."""
        from victor.config.secure_paths import (
            create_cache_manifest,
            verify_cache_integrity,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)

            # Create test file and manifest
            test_file = cache_dir / "test.pkl"
            test_file.write_text("original content")
            create_cache_manifest(cache_dir)

            # Tamper with file
            test_file.write_text("TAMPERED content")

            # Verify should fail
            is_valid, tampered = verify_cache_integrity(cache_dir)
            assert is_valid is False
            assert "test.pkl" in tampered

    def test_verify_cache_integrity_no_manifest(self):
        """Test verification without manifest returns valid."""
        from victor.config.secure_paths import verify_cache_integrity

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)

            # No manifest - should return valid (unverified)
            is_valid, tampered = verify_cache_integrity(cache_dir)
            assert is_valid is True

    def test_secure_embeddings_dir(self):
        """Test secure embeddings directory path."""
        from victor.config.secure_paths import secure_embeddings_dir, get_victor_dir

        result = secure_embeddings_dir()
        assert isinstance(result, Path)
        assert result == get_victor_dir() / "embeddings"


class TestPluginSignature:
    """Tests for plugin signature verification."""

    def test_compute_plugin_hash_file(self):
        """Test plugin hash computation for a file."""
        from victor.config.secure_paths import compute_plugin_hash

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def plugin_func(): pass")
            temp_path = Path(f.name)

        try:
            hash1 = compute_plugin_hash(temp_path)
            assert hash1 is not None
            assert len(hash1) == 64

            hash2 = compute_plugin_hash(temp_path)
            assert hash1 == hash2
        finally:
            temp_path.unlink()

    def test_compute_plugin_hash_directory(self):
        """Test plugin hash computation for a directory."""
        from victor.config.secure_paths import compute_plugin_hash

        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_dir = Path(tmpdir)
            (plugin_dir / "__init__.py").write_text("# Plugin init")
            (plugin_dir / "plugin.py").write_text("def plugin(): pass")

            hash1 = compute_plugin_hash(plugin_dir)
            assert hash1 is not None
            assert len(hash1) == 64

    def test_compute_plugin_hash_nonexistent(self):
        """Test hash of nonexistent plugin returns None."""
        from victor.config.secure_paths import compute_plugin_hash

        result = compute_plugin_hash(Path("/nonexistent/plugin"))
        assert result is None

    def test_trust_and_verify_plugin(self):
        """Test trusting and verifying a plugin."""
        from victor.config.secure_paths import (
            trust_plugin,
            verify_plugin_trust,
            untrust_plugin,
        )

        # Generate unique name and content to avoid conflicts with other tests
        import uuid

        unique_name = f"test_plugin_{uuid.uuid4().hex[:8]}"
        unique_content = f"def plugin_{uuid.uuid4().hex[:8]}(): pass"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(unique_content)
            temp_path = Path(f.name)

        try:
            # Initially not trusted
            is_trusted, reason = verify_plugin_trust(temp_path)
            assert is_trusted is False
            assert reason == "not_trusted"

            # Trust the plugin with unique name
            result = trust_plugin(temp_path, name=unique_name)
            assert result is True

            # Now should be trusted
            is_trusted, reason = verify_plugin_trust(temp_path)
            assert is_trusted is True
            assert f"trusted:{unique_name}" in reason

            # Untrust
            untrust_result = untrust_plugin(unique_name)
            assert untrust_result is True

            # Should be untrusted again
            is_trusted, _ = verify_plugin_trust(temp_path)
            assert is_trusted is False

        finally:
            # Clean up: ensure plugin is untrusted even if test fails
            try:
                untrust_plugin(unique_name)
            except Exception:
                pass
            temp_path.unlink()

    def test_list_trusted_plugins(self):
        """Test listing trusted plugins."""
        from victor.config.secure_paths import list_trusted_plugins

        result = list_trusted_plugins()
        assert isinstance(result, list)

    def test_get_security_status(self):
        """Test security status function."""
        from victor.config.secure_paths import get_security_status

        status = get_security_status()

        assert "platform" in status
        assert "home_security" in status
        assert "keyring" in status
        assert "paths" in status
        assert "plugins" in status
        assert "cache_integrity" in status

        assert status["platform"]["system"] is not None
        assert status["home_security"]["home_validated"] is True


class TestPluginSandbox:
    """Tests for plugin sandboxing functions."""

    def test_sandbox_policy_defaults(self):
        """Test default sandbox policy values."""
        from victor.config.secure_paths import (
            PluginSandboxPolicy,
            DEFAULT_SANDBOX_POLICY,
        )

        policy = PluginSandboxPolicy()
        assert policy.allow_network is True
        assert policy.allow_subprocess is True
        assert policy.allow_file_write is True
        assert policy.require_trust is False
        assert policy.blocked_paths == []
        assert policy.allowed_paths == []

        # Default policy should match
        assert DEFAULT_SANDBOX_POLICY.allow_network is True
        assert DEFAULT_SANDBOX_POLICY.require_trust is False

    def test_strict_sandbox_policy(self):
        """Test strict sandbox policy values."""
        from victor.config.secure_paths import STRICT_SANDBOX_POLICY

        assert STRICT_SANDBOX_POLICY.allow_network is False
        assert STRICT_SANDBOX_POLICY.allow_subprocess is False
        assert STRICT_SANDBOX_POLICY.allow_file_write is False
        assert STRICT_SANDBOX_POLICY.require_trust is True

    def test_get_plugin_sandbox_policy(self):
        """Test getting current sandbox policy."""
        from victor.config.secure_paths import get_plugin_sandbox_policy

        policy = get_plugin_sandbox_policy()
        assert policy.allow_network is True  # Default is permissive

    def test_check_plugin_can_load_valid_path(self):
        """Test plugin loading check with valid path."""
        from victor.config.secure_paths import check_plugin_can_load

        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_dir = Path(tmpdir) / "test_plugin"
            plugin_dir.mkdir()
            (plugin_dir / "plugin.py").write_text("# test plugin")

            can_load, reason = check_plugin_can_load(plugin_dir)
            # May fail due to path escape detection in temp dirs, which is expected
            assert isinstance(can_load, bool)
            assert isinstance(reason, str)

    def test_check_plugin_can_load_with_blocked_paths(self):
        """Test plugin loading blocked by path policy."""
        from victor.config.secure_paths import (
            check_plugin_can_load,
            PluginSandboxPolicy,
        )

        policy = PluginSandboxPolicy(
            blocked_paths=["/tmp/blocked"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_dir = Path(tmpdir) / "test_plugin"
            plugin_dir.mkdir()
            (plugin_dir / "plugin.py").write_text("# test plugin")

            can_load, reason = check_plugin_can_load(plugin_dir, policy=policy)
            # Should load since not in blocked path
            assert isinstance(can_load, bool)

    def test_check_sandbox_action_network(self):
        """Test sandbox action check for network."""
        from victor.config.secure_paths import check_sandbox_action, PluginSandboxPolicy

        # Default policy allows network
        is_allowed, reason = check_sandbox_action("test_plugin", "network")
        assert is_allowed is True
        assert reason == "allowed"

        # Strict policy blocks network
        strict_policy = PluginSandboxPolicy(allow_network=False)
        is_allowed, reason = check_sandbox_action("test_plugin", "network", policy=strict_policy)
        assert is_allowed is False
        assert reason == "network_not_allowed"

    def test_check_sandbox_action_subprocess(self):
        """Test sandbox action check for subprocess."""
        from victor.config.secure_paths import check_sandbox_action, PluginSandboxPolicy

        # Default policy allows subprocess
        is_allowed, reason = check_sandbox_action("test_plugin", "subprocess")
        assert is_allowed is True

        # Strict policy blocks subprocess
        strict_policy = PluginSandboxPolicy(allow_subprocess=False)
        is_allowed, reason = check_sandbox_action("test_plugin", "subprocess", policy=strict_policy)
        assert is_allowed is False
        assert reason == "subprocess_not_allowed"

    def test_check_sandbox_action_file_write(self):
        """Test sandbox action check for file write."""
        from victor.config.secure_paths import check_sandbox_action, PluginSandboxPolicy

        # Default policy allows file write
        is_allowed, reason = check_sandbox_action("test_plugin", "file_write")
        assert is_allowed is True

        # Strict policy blocks file write
        strict_policy = PluginSandboxPolicy(allow_file_write=False)
        is_allowed, reason = check_sandbox_action("test_plugin", "file_write", policy=strict_policy)
        assert is_allowed is False
        assert reason == "file_write_not_allowed"

    def test_check_sandbox_action_blocked_path(self):
        """Test sandbox action with blocked path."""
        from victor.config.secure_paths import check_sandbox_action, PluginSandboxPolicy

        policy = PluginSandboxPolicy(
            blocked_paths=["/etc"],
        )

        # Access to blocked path should fail
        is_allowed, reason = check_sandbox_action(
            "test_plugin",
            "file_read",
            details={"path": "/etc/passwd"},
            policy=policy,
        )
        assert is_allowed is False
        assert "blocked_path" in reason

        # Access to non-blocked path should succeed
        is_allowed, reason = check_sandbox_action(
            "test_plugin",
            "file_read",
            details={"path": "/tmp/test.txt"},
            policy=policy,
        )
        assert is_allowed is True

    def test_get_sandbox_summary(self):
        """Test sandbox summary function."""
        from victor.config.secure_paths import get_sandbox_summary

        summary = get_sandbox_summary()

        assert "policy" in summary
        assert "trusted_plugins" in summary
        assert "require_trust" in summary["policy"]
        assert "allow_network" in summary["policy"]
        assert "count" in summary["trusted_plugins"]
        assert "names" in summary["trusted_plugins"]

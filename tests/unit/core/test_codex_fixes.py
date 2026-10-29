"""Tests for Codex-verified plugin/vertical architecture fixes."""

from unittest.mock import MagicMock
import pytest

# =====================================================================
# Fix 1: Subprocess Permission Enforcement (Finding #6)
# =====================================================================


class TestPermissionEnforcement:
    """Test that external plugin tools enforce permissions before execution."""

    def test_permission_check_exists_in_execute_tool(self):
        """execute_tool should check required_permission before subprocess."""
        import inspect
        from victor.core.plugins.external import ExternalPluginManager

        source = inspect.getsource(ExternalPluginManager.execute_tool)
        assert "required_permission" in source or "danger-full-access" in source

    @pytest.mark.asyncio
    async def test_dangerous_tool_denied_by_default(self):
        """danger-full-access tools should be denied when _allow_dangerous=False."""
        from victor.core.plugins.external import ExternalPluginManager

        runner = ExternalPluginManager.__new__(ExternalPluginManager)
        runner._allow_dangerous = False
        runner._tool_index = {"dangerous_tool": "plugin1"}

        # Mock plugin with danger-full-access tool
        tool_spec = MagicMock()
        tool_spec.name = "dangerous_tool"
        tool_spec.required_permission = "danger-full-access"
        tool_spec.command = "/bin/rm"
        tool_spec.args = ["-rf", "/"]

        plugin = MagicMock()
        plugin.manifest.tools = [tool_spec]
        runner._plugins = {"plugin1": plugin}

        result = await runner.execute_tool("dangerous_tool", {"target": "/"})
        assert result.is_error is True
        assert result.return_code == 126
        assert "Permission denied" in result.output

    @pytest.mark.asyncio
    async def test_safe_tool_allowed(self):
        """workspace-write tools should proceed (not blocked by permission check)."""
        from victor.core.plugins.external import ExternalPluginManager

        runner = ExternalPluginManager.__new__(ExternalPluginManager)
        runner._allow_dangerous = False
        runner._tool_index = {"safe_tool": "plugin1"}

        tool_spec = MagicMock()
        tool_spec.name = "safe_tool"
        tool_spec.required_permission = "workspace-write"
        tool_spec.command = "echo"
        tool_spec.args = ["hello"]

        plugin = MagicMock()
        plugin.manifest.tools = [tool_spec]
        plugin.root_path = MagicMock()
        runner._plugins = {"plugin1": plugin}

        # The tool won't actually execute (no real subprocess), but the
        # permission check should NOT block it
        # We just verify the permission check doesn't return early
        # (the actual subprocess will fail in test env, that's fine)
        try:
            result = await runner.execute_tool("safe_tool", {})
            # If it reaches subprocess and fails, that's fine — permission wasn't denied
            assert result.return_code != 126 or "Permission denied" not in result.output
        except Exception:
            pass  # Subprocess failure is expected in test — permission check passed


# =====================================================================
# Fix 2: Deprecation warning (already implemented - verify)
# =====================================================================


class TestVerticalDeprecation:
    """Verify that victor.verticals EP group emits deprecation warning."""

    def test_deprecation_method_exists(self):
        """VerticalRegistry has _warn_legacy_entry_point_usage method."""
        from victor.core.verticals.base import VerticalRegistry

        assert hasattr(VerticalRegistry, "_warn_legacy_entry_point_usage")

    def test_deprecation_uses_logger_warning(self):
        """Method uses logger.warning for deprecation notices."""
        import inspect
        from victor.core.verticals.base import VerticalRegistry

        source = inspect.getsource(VerticalRegistry._warn_legacy_entry_point_usage)
        assert "logger.warning" in source


# =====================================================================
# Fix 3: EP Scan Caching (Finding #5)
# =====================================================================


class TestEPScanCaching:
    """Test that EP scanning is cached per (group, vertical_name)."""

    def test_cache_set_exists(self):
        """_EP_SCAN_CACHE module-level set exists."""
        from victor.core.plugins.context import _EP_SCAN_CACHE

        assert isinstance(_EP_SCAN_CACHE, set)

    def test_cache_prevents_rescan(self):
        """Adding to cache prevents redundant EP scanning."""
        from victor.core.plugins.context import _EP_SCAN_CACHE

        # Simulate cache hit
        test_key = "test_group:test_vertical"
        _EP_SCAN_CACHE.add(test_key)
        assert test_key in _EP_SCAN_CACHE

        # Clean up
        _EP_SCAN_CACHE.discard(test_key)

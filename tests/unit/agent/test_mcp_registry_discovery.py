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

"""Tests for MCP Registry auto-discovery functionality."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

from victor.integrations.mcp.registry import MCPRegistry, MCPServerConfig, ServerStatus


class TestMCPServerConfig:
    """Tests for MCPServerConfig model."""

    def test_full_config(self):
        """Test creating config with all fields."""
        config = MCPServerConfig(
            name="full_server",
            command=["node", "server.js"],
            description="Full test server",
            auto_connect=False,
            health_check_interval=60,
            max_retries=5,
            retry_delay=10,
            enabled=True,
            tags=["database", "test"],
            env={"DEBUG": "true"},
        )
        assert config.name == "full_server"
        assert config.description == "Full test server"
        assert config.auto_connect is False
        assert config.health_check_interval == 60
        assert config.tags == ["database", "test"]
        assert config.env["DEBUG"] == "true"


class TestMCPRegistryDiscovery:
    """Tests for MCPRegistry.discover_servers() method."""

    def test_discover_no_config_found(self):
        """Test discovery when no config files and no built-in executables exist."""
        with patch.dict("os.environ", {}, clear=True):
            with patch.object(Path, "exists", return_value=False):
                # Isolate from ambient machine state: well-known MCP executables
                # on PATH (e.g. anthropics_mcp-server-fetch via npx) are
                # auto-discovered, so neutralise shutil.which to exercise the
                # "no config" path only.
                with patch("shutil.which", return_value=None):
                    registry = MCPRegistry.discover_servers()
                    assert isinstance(registry, MCPRegistry)
                    assert len(registry.list_servers()) == 0

    def test_discover_from_env_variable(self):
        """Test discovery from VICTOR_MCP_CONFIG environment variable."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
servers:
  - name: env_server
    command: ["python", "-m", "env_server"]
""")
            f.flush()

            with patch.dict("os.environ", {"VICTOR_MCP_CONFIG": f.name}):
                registry = MCPRegistry.discover_servers()
                servers = registry.list_servers()
                assert "env_server" in servers

    def test_discover_from_project_local(self, isolated_project_victor_dir):
        """Test discovery from project-local .victor/mcp.yaml."""
        config_file = isolated_project_victor_dir / "mcp.yaml"
        config_file.write_text("""
servers:
  - name: local_server
    command: ["python", "-m", "local_server"]
""")
        with patch.dict("os.environ", {"VICTOR_MCP_CONFIG": ""}):
            registry = MCPRegistry.discover_servers()
            servers = registry.list_servers()
            assert "local_server" in servers

    def test_discover_priority_order(self):
        """Test that env variable takes priority over local config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create env config
            env_config = Path(tmpdir) / "env_mcp.yaml"
            env_config.write_text("""
servers:
  - name: env_priority
    command: ["python", "-m", "env"]
""")

            # Create local config
            local_dir = Path(tmpdir) / "local" / ".victor"
            local_dir.mkdir(parents=True)
            local_config = local_dir / "mcp.yaml"
            local_config.write_text("""
servers:
  - name: local_server
    command: ["python", "-m", "local"]
""")

            import os

            old_cwd = os.getcwd()
            try:
                os.chdir(Path(tmpdir) / "local")
                with patch.dict("os.environ", {"VICTOR_MCP_CONFIG": str(env_config)}):
                    registry = MCPRegistry.discover_servers()
                    servers = registry.list_servers()
                    # Env config should take priority
                    assert "env_priority" in servers
                    assert "local_server" not in servers
            finally:
                os.chdir(old_cwd)


class TestMCPRegistryBasics:
    """Tests for basic MCPRegistry functionality."""

    def test_register_server(self):
        """Test registering a server."""
        registry = MCPRegistry()
        config = MCPServerConfig(
            name="test_server",
            command=["python", "server.py"],
        )
        registry.register_server(config)
        assert "test_server" in registry.list_servers()

    def test_unregister_server(self):
        """Test unregistering a server."""
        registry = MCPRegistry()
        config = MCPServerConfig(
            name="test_server",
            command=["python", "server.py"],
        )
        registry.register_server(config)
        result = registry.unregister_server("test_server")
        assert result is True
        assert "test_server" not in registry.list_servers()

    def test_unregister_nonexistent_server(self):
        """Test unregistering a server that doesn't exist."""
        registry = MCPRegistry()
        result = registry.unregister_server("nonexistent")
        assert result is False

    def test_list_servers(self):
        """Test listing registered servers."""
        registry = MCPRegistry()
        registry.register_server(MCPServerConfig(name="server1", command=["python", "s1.py"]))
        registry.register_server(MCPServerConfig(name="server2", command=["python", "s2.py"]))
        servers = registry.list_servers()
        assert "server1" in servers
        assert "server2" in servers

    def test_get_server_status_not_found(self):
        """Test getting status of non-existent server."""
        registry = MCPRegistry()
        status = registry.get_server_status("nonexistent")
        assert status is None

    def test_get_server_status(self):
        """Test getting server status."""
        registry = MCPRegistry()
        registry.register_server(MCPServerConfig(name="test", command=["python", "s.py"]))
        status = registry.get_server_status("test")
        assert status is not None
        assert status["name"] == "test"
        assert status["status"] == "DISCONNECTED"


class TestMCPRegistryFromConfig:
    """Tests for MCPRegistry.from_config() method."""

    def test_from_yaml_config(self):
        """Test loading from YAML config file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
health_check_enabled: false
default_health_interval: 60
servers:
  - name: yaml_server
    command: ["python", "-m", "yaml_server"]
    description: Test YAML server
    tags:
      - test
      - yaml
""")
            f.flush()

            registry = MCPRegistry.from_config(Path(f.name))
            assert "yaml_server" in registry.list_servers()

    def test_from_json_config(self):
        """Test loading from JSON config file."""
        import json

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config = {
                "servers": [
                    {
                        "name": "json_server",
                        "command": ["python", "-m", "json_server"],
                    }
                ]
            }
            json.dump(config, f)
            f.flush()

            registry = MCPRegistry.from_config(Path(f.name))
            assert "json_server" in registry.list_servers()

    def test_from_nonexistent_config(self):
        """Test loading from non-existent config file."""
        registry = MCPRegistry.from_config(Path("/nonexistent/config.yaml"))
        assert isinstance(registry, MCPRegistry)
        assert len(registry.list_servers()) == 0


class TestMCPRegistryEvents:
    """Tests for MCPRegistry event handling."""

    def test_on_event_callback(self):
        """Test registering event callback."""
        registry = MCPRegistry()
        events_received = []

        def callback(event_type, server_name, data):
            events_received.append((event_type, server_name, data))

        registry.on_event(callback)
        assert len(registry._event_callbacks) == 1

    def test_reset_server(self):
        """Test resetting a failed server."""
        registry = MCPRegistry()
        registry.register_server(MCPServerConfig(name="test", command=["python", "s.py"]))

        # Manually set failure state
        entry = registry._servers["test"]
        entry.status = ServerStatus.FAILED
        entry.consecutive_failures = 3
        entry.error_message = "Connection failed"

        # Reset
        result = registry.reset_server("test")
        assert result is True
        assert entry.status == ServerStatus.DISCONNECTED
        assert entry.consecutive_failures == 0
        assert entry.error_message is None

    def test_reset_nonexistent_server(self):
        """Test resetting a server that doesn't exist."""
        registry = MCPRegistry()
        result = registry.reset_server("nonexistent")
        assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

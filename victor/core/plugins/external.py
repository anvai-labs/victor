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

"""External plugin manager for subprocess-based plugins.

Provides discovery, lifecycle management, and tool execution for external
plugins defined via plugin.json manifests. Plugins execute tools as
subprocesses with JSON I/O, enabling language-agnostic extensibility.

Extends Victor's entry-point-based plugin system with language-agnostic
subprocess plugins, supporting any executable as a tool provider.

Plugin directory structure:
    plugin-root/
    ├── plugin.json              (or .victor-plugin/plugin.json)
    ├── hooks/
    │   ├── pre.sh
    │   └── post.sh
    ├── lifecycle/
    │   ├── init.sh
    │   └── shutdown.sh
    ├── tools/
    │   └── *.sh
    └── commands/
        └── *.sh
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from victor.core.plugins.manifest import (
    ManifestValidationError,
    PluginHooksSpec,
    PluginKind,
    PluginManifest,
    PluginToolSpec,
    sanitize_plugin_id,
)

logger = logging.getLogger(__name__)


@dataclass
class PluginToolResult:
    """Result from executing an external plugin tool."""

    output: str
    is_error: bool = False
    return_code: int = 0


@dataclass
class RegisteredPlugin:
    """A plugin loaded into the registry."""

    plugin_id: str
    kind: PluginKind
    manifest: PluginManifest
    root_path: Path
    enabled: bool = False

    @property
    def name(self) -> str:
        return self.manifest.name

    @property
    def version(self) -> str:
        return self.manifest.version


@dataclass
class InstalledPluginRecord:
    """Registry entry for an installed external plugin."""

    plugin_id: str
    kind: str
    name: str
    version: str
    description: str
    install_path: str
    source_type: str = ""
    source_url: str = ""
    installed_at_unix_ms: int = 0
    updated_at_unix_ms: int = 0


class ExternalPluginManager:
    """Manages external plugins with subprocess-based tool execution.

    Handles plugin discovery, lifecycle (init/shutdown), tool execution
    via subprocess with JSON I/O, and hook aggregation.
    """

    def __init__(
        self,
        config_home: Optional[Path] = None,
        enabled_plugins: Optional[Dict[str, bool]] = None,
        external_dirs: Optional[List[Path]] = None,
        bundled_root: Optional[Path] = None,
    ) -> None:
        """Initialize the external plugin manager.

        Args:
            config_home: Base configuration directory (default: ~/.victor).
            enabled_plugins: Map of plugin_id -> enabled state.
            external_dirs: Additional directories to scan for plugins.
            bundled_root: Directory containing bundled plugins.
        """
        self._config_home = config_home or Path.home() / ".victor"
        self._install_root = self._config_home / "plugins" / "installed"
        self._registry_path = self._config_home / "plugins" / "installed.json"
        self._enabled_plugins = enabled_plugins or {}
        self._external_dirs = external_dirs or []
        self._bundled_root = bundled_root
        self._plugins: Dict[str, RegisteredPlugin] = {}
        self._tool_index: Dict[str, str] = {}  # tool_name -> plugin_id

    @property
    def plugins(self) -> Dict[str, RegisteredPlugin]:
        """Return all registered plugins."""
        return dict(self._plugins)

    def discover_plugins(self) -> List[RegisteredPlugin]:
        """Discover and register all available plugins.

        Scans bundled, installed, and external directories for plugin.json files.

        Returns:
            List of discovered RegisteredPlugin instances.
        """
        discovered: List[RegisteredPlugin] = []

        # Bundled plugins
        if self._bundled_root and self._bundled_root.is_dir():
            for path in sorted(self._bundled_root.iterdir()):
                if path.is_dir():
                    plugin = self._load_plugin(path, PluginKind.BUNDLED)
                    if plugin:
                        discovered.append(plugin)

        # Installed external plugins
        if self._install_root.is_dir():
            for path in sorted(self._install_root.iterdir()):
                if path.is_dir():
                    plugin = self._load_plugin(path, PluginKind.EXTERNAL)
                    if plugin:
                        discovered.append(plugin)

        # Additional external directories
        for ext_dir in self._external_dirs:
            if ext_dir.is_dir():
                plugin = self._load_plugin(ext_dir, PluginKind.EXTERNAL)
                if plugin:
                    discovered.append(plugin)

        # Register all discovered plugins
        for plugin in discovered:
            self._register_plugin(plugin)

        logger.info(
            "Discovered %d plugins (%d enabled)",
            len(discovered),
            sum(1 for p in discovered if p.enabled),
        )
        return discovered

    def _load_plugin(self, path: Path, kind: PluginKind) -> Optional[RegisteredPlugin]:
        """Load a plugin from a directory.

        Searches for plugin.json at the root or under .victor-plugin/.

        Args:
            path: Plugin root directory.
            kind: Origin type (builtin, bundled, external).

        Returns:
            RegisteredPlugin if valid, None otherwise.
        """
        manifest_path = path / "plugin.json"
        if not manifest_path.exists():
            manifest_path = path / ".victor-plugin" / "plugin.json"
        if not manifest_path.exists():
            return None

        try:
            manifest = PluginManifest.from_file(manifest_path)
        except (ManifestValidationError, json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning("Failed to load plugin from %s: %s", path, e)
            return None

        plugin_id = sanitize_plugin_id(manifest.name, kind.value)
        enabled = self._resolve_enabled(plugin_id, manifest, kind)

        return RegisteredPlugin(
            plugin_id=plugin_id,
            kind=kind,
            manifest=manifest,
            root_path=path,
            enabled=enabled,
        )

    def _resolve_enabled(self, plugin_id: str, manifest: PluginManifest, kind: PluginKind) -> bool:
        """Determine if a plugin should be enabled.

        Uses explicit setting if available, otherwise falls back to manifest default.
        External plugins default to disabled unless explicitly enabled.

        Args:
            plugin_id: The plugin identifier.
            manifest: The plugin manifest.
            kind: The plugin origin type.

        Returns:
            Whether the plugin should be enabled.
        """
        if plugin_id in self._enabled_plugins:
            return self._enabled_plugins[plugin_id]
        if kind == PluginKind.EXTERNAL:
            return False
        return manifest.default_enabled

    def _register_plugin(self, plugin: RegisteredPlugin) -> None:
        """Register a plugin and index its tools."""
        self._plugins[plugin.plugin_id] = plugin
        if plugin.enabled:
            for tool in plugin.manifest.tools:
                if tool.name in self._tool_index:
                    existing = self._tool_index[tool.name]
                    logger.warning(
                        "Tool name conflict: '%s' in plugin '%s' " "already registered by '%s'",
                        tool.name,
                        plugin.plugin_id,
                        existing,
                    )
                else:
                    self._tool_index[tool.name] = plugin.plugin_id

    async def initialize(self) -> None:
        """Run init lifecycle for all enabled plugins."""
        for plugin in self._plugins.values():
            if not plugin.enabled:
                continue
            for cmd in plugin.manifest.lifecycle.init:
                try:
                    await self._run_lifecycle_command(cmd, plugin.root_path)
                    logger.debug("Init command succeeded for %s", plugin.plugin_id)
                except Exception as e:
                    logger.error("Init failed for plugin %s: %s", plugin.plugin_id, e)

    async def shutdown(self) -> None:
        """Run shutdown lifecycle for all enabled plugins in reverse order."""
        for plugin in reversed(list(self._plugins.values())):
            if not plugin.enabled:
                continue
            for cmd in plugin.manifest.lifecycle.shutdown:
                try:
                    await self._run_lifecycle_command(cmd, plugin.root_path)
                except Exception as e:
                    logger.warning("Shutdown failed for plugin %s: %s", plugin.plugin_id, e)

    async def execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> PluginToolResult:
        """Execute an external plugin tool via subprocess.

        Args:
            tool_name: Name of the tool to execute.
            tool_input: Tool arguments as a dictionary.

        Returns:
            PluginToolResult with output and error status.

        Raises:
            KeyError: If tool is not registered.
        """
        plugin_id = self._tool_index.get(tool_name)
        if not plugin_id:
            raise KeyError(f"No plugin tool registered with name '{tool_name}'")

        plugin = self._plugins[plugin_id]
        tool_spec = next(t for t in plugin.manifest.tools if t.name == tool_name)

        # Permission enforcement (Codex finding #6 — was advisory only)
        required_perm = getattr(tool_spec, "required_permission", "workspace-write")
        if required_perm == "danger-full-access":
            allow_dangerous = getattr(self, "_allow_dangerous", False)
            if not allow_dangerous:
                logger.warning(
                    f"[permission] Tool '{tool_name}' requires 'danger-full-access' "
                    f"but dangerous execution is not enabled. Denied."
                )
                return PluginToolResult(
                    output=f"Permission denied: '{tool_name}' requires 'danger-full-access'",
                    is_error=True,
                    return_code=126,
                )

        # Resolve command path
        command = tool_spec.command
        if command.startswith("./") or command.startswith("../"):
            command = str(plugin.root_path / command)

        args = tool_spec.args or []
        input_json = json.dumps(tool_input)

        env = {
            "VICTOR_PLUGIN_ID": plugin_id,
            "VICTOR_PLUGIN_NAME": plugin.name,
            "VICTOR_TOOL_NAME": tool_name,
            "VICTOR_TOOL_INPUT": input_json,
            "VICTOR_PLUGIN_ROOT": str(plugin.root_path),
        }

        try:
            proc = await asyncio.create_subprocess_exec(
                command,
                *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(plugin.root_path),
                env={**dict(__import__("os").environ), **env},
            )
            stdout, stderr = await proc.communicate(input=input_json.encode())
            return_code = proc.returncode or 0

            if return_code != 0:
                error_msg = stderr.decode().strip() or stdout.decode().strip()
                return PluginToolResult(
                    output=error_msg or f"Tool exited with code {return_code}",
                    is_error=True,
                    return_code=return_code,
                )

            return PluginToolResult(
                output=stdout.decode().strip(),
                is_error=False,
                return_code=0,
            )

        except FileNotFoundError:
            return PluginToolResult(
                output=f"Plugin tool command not found: {command}",
                is_error=True,
                return_code=127,
            )
        except Exception as e:
            return PluginToolResult(
                output=f"Plugin tool execution failed: {e}",
                is_error=True,
                return_code=1,
            )

    def has_tool(self, tool_name: str) -> bool:
        """Check if a tool is provided by any enabled plugin."""
        return tool_name in self._tool_index

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Return JSON Schema tool definitions for all enabled plugin tools.

        Returns:
            List of tool definition dicts suitable for LLM tool_use.
        """
        definitions = []
        for plugin in self._plugins.values():
            if not plugin.enabled:
                continue
            for tool in plugin.manifest.tools:
                definitions.append(
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.input_schema,
                        "metadata": {
                            "plugin_id": plugin.plugin_id,
                            "required_permission": tool.required_permission,
                        },
                    }
                )
        return definitions

    def get_aggregated_hooks(self) -> PluginHooksSpec:
        """Aggregate hooks from all enabled plugins.

        Returns:
            Merged PluginHooksSpec with all hook commands.
        """
        merged = PluginHooksSpec()
        for plugin in self._plugins.values():
            if plugin.enabled:
                merged = merged.merged_with(plugin.manifest.hooks)
        return merged

    async def install_plugin(
        self,
        source: str,
        source_type: str = "local_path",
    ) -> RegisteredPlugin:
        """Install a plugin from a source path or git URL.

        Args:
            source: Local path or git URL.
            source_type: "local_path" or "git_url".

        Returns:
            The installed RegisteredPlugin.

        Raises:
            ManifestValidationError: If plugin manifest is invalid.
            FileNotFoundError: If source does not exist.
        """
        if source_type == "git_url" or source.endswith(".git"):
            source_path = await self._clone_git_plugin(source)
        else:
            source_path = Path(source)

        if not source_path.is_dir():
            raise FileNotFoundError(f"Plugin source not found: {source}")

        manifest = PluginManifest.from_file(
            source_path / "plugin.json"
            if (source_path / "plugin.json").exists()
            else source_path / ".victor-plugin" / "plugin.json"
        )

        plugin_id = sanitize_plugin_id(manifest.name, "external")
        install_path = self._install_root / plugin_id.replace("@", "-")
        install_path.mkdir(parents=True, exist_ok=True)

        # Copy plugin files
        if install_path.exists():
            shutil.rmtree(install_path)
        shutil.copytree(source_path, install_path)

        # Update registry
        self._save_registry_entry(
            InstalledPluginRecord(
                plugin_id=plugin_id,
                kind="external",
                name=manifest.name,
                version=manifest.version,
                description=manifest.description,
                install_path=str(install_path),
                source_type=source_type,
                source_url=source,
                installed_at_unix_ms=int(time.time() * 1000),
                updated_at_unix_ms=int(time.time() * 1000),
            )
        )

        plugin = RegisteredPlugin(
            plugin_id=plugin_id,
            kind=PluginKind.EXTERNAL,
            manifest=manifest,
            root_path=install_path,
            enabled=False,
        )
        self._register_plugin(plugin)
        return plugin

    def enable_plugin(self, plugin_id: str) -> bool:
        """Enable a plugin by ID."""
        if plugin_id not in self._plugins:
            return False
        self._plugins[plugin_id].enabled = True
        self._enabled_plugins[plugin_id] = True
        # Re-index tools
        for tool in self._plugins[plugin_id].manifest.tools:
            self._tool_index[tool.name] = plugin_id
        return True

    def disable_plugin(self, plugin_id: str) -> bool:
        """Disable a plugin by ID."""
        if plugin_id not in self._plugins:
            return False
        self._plugins[plugin_id].enabled = False
        self._enabled_plugins[plugin_id] = False
        # Remove tools from index
        for tool in self._plugins[plugin_id].manifest.tools:
            if self._tool_index.get(tool.name) == plugin_id:
                del self._tool_index[tool.name]
        return True

    def uninstall_plugin(self, plugin_id: str) -> bool:
        """Uninstall an external plugin.

        Bundled plugins cannot be uninstalled, only disabled.

        Args:
            plugin_id: The plugin identifier.

        Returns:
            True if uninstalled, False if not found or is bundled.
        """
        plugin = self._plugins.get(plugin_id)
        if not plugin:
            return False
        if plugin.kind == PluginKind.BUNDLED:
            logger.warning("Cannot uninstall bundled plugin %s, use disable", plugin_id)
            return False

        self.disable_plugin(plugin_id)
        del self._plugins[plugin_id]

        if plugin.root_path.exists():
            shutil.rmtree(plugin.root_path)

        self._remove_registry_entry(plugin_id)
        return True

    async def _run_lifecycle_command(self, command: str, cwd: Path) -> None:
        """Run a lifecycle command in the plugin's directory."""
        if command.startswith("./") or command.startswith("../"):
            command = str(cwd / command)

        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"Lifecycle command failed (exit {proc.returncode}): " f"{stderr.decode().strip()}"
            )

    async def _clone_git_plugin(self, url: str) -> Path:
        """Clone a git repository to a temporary directory."""
        import tempfile

        tmp_dir = Path(tempfile.mkdtemp(prefix="victor-plugin-"))
        proc = await asyncio.create_subprocess_exec(
            "git",
            "clone",
            "--depth=1",
            url,
            str(tmp_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"Git clone failed: {stderr.decode().strip()}")
        return tmp_dir

    def _save_registry_entry(self, record: InstalledPluginRecord) -> None:
        """Save or update a plugin in the installed.json registry."""
        self._registry_path.parent.mkdir(parents=True, exist_ok=True)

        registry: Dict[str, Any] = {"plugins": {}}
        if self._registry_path.exists():
            with open(self._registry_path) as f:
                registry = json.load(f)

        registry.setdefault("plugins", {})[record.plugin_id] = {
            "kind": record.kind,
            "id": record.plugin_id,
            "name": record.name,
            "version": record.version,
            "description": record.description,
            "install_path": record.install_path,
            "source": {"type": record.source_type, "url": record.source_url},
            "installed_at_unix_ms": record.installed_at_unix_ms,
            "updated_at_unix_ms": record.updated_at_unix_ms,
        }

        with open(self._registry_path, "w") as f:
            json.dump(registry, f, indent=2)

    def _remove_registry_entry(self, plugin_id: str) -> None:
        """Remove a plugin from the installed.json registry."""
        if not self._registry_path.exists():
            return
        with open(self._registry_path) as f:
            registry = json.load(f)
        registry.get("plugins", {}).pop(plugin_id, None)
        with open(self._registry_path, "w") as f:
            json.dump(registry, f, indent=2)

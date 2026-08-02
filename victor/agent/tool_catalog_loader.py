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

"""Tool catalog loader for dynamic tool discovery.

This module handles dynamic tool discovery from the victor/tools directory,
extracted from ToolRegistrar as part of SRP compliance refactoring.

Single Responsibility: Load and register tools from the shared tool registry.

Design Pattern: Loader Pattern
- Discovers tools from SharedToolRegistry
- Loads tool configurations from settings
- Registers tools with ToolRegistry

Usage:
    from victor.agent.tool_catalog_loader import ToolCatalogLoader

    loader = ToolCatalogLoader(
        registry=tool_registry,
        settings=settings,
        airgapped_mode=False,
    )
    count = loader.load()
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from victor.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class ToolCatalogConfig:
    """Configuration for tool catalog loading.

    Attributes:
        airgapped_mode: Disable web tools
        enabled_tools: List of explicitly enabled tool names (if set, others disabled)
        disabled_tools: List of tool names to disable
    """

    airgapped_mode: bool = False
    enabled_tools: List[str] = field(default_factory=list)
    disabled_tools: List[str] = field(default_factory=list)
    lazy_startup: bool = True


@dataclass
class CatalogLoadResult:
    """Result of tool catalog loading.

    Attributes:
        tools_loaded: Number of tools successfully loaded
        tools_disabled: Number of tools disabled by configuration
        errors: List of errors encountered during loading
    """

    tools_loaded: int = 0
    tools_disabled: int = 0
    errors: List[str] = field(default_factory=list)
    full_catalog_loaded: bool = False


class ToolCatalogLoader:
    """Loads tools from shared registry into tool registry.

    Single Responsibility: Dynamic tool discovery and registration.

    This class handles:
    - Loading tools from SharedToolRegistry
    - Applying enabled/disabled configuration
    - Respecting airgapped mode for web tools

    Extracted from ToolRegistrar for SRP compliance.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        settings: Any,
        config: Optional[ToolCatalogConfig] = None,
    ):
        """Initialize the tool catalog loader.

        Args:
            registry: Tool registry to register tools with
            settings: Application settings for tool configuration
            config: Optional catalog configuration
        """
        self._registry = registry
        self._settings = settings
        self._config = config or ToolCatalogConfig()
        self._loaded = False
        self._bootstrap_loaded = False

    @property
    def is_loaded(self) -> bool:
        """Check if tools have been loaded."""
        return self._loaded

    def load(self) -> CatalogLoadResult:
        """Load tools from shared registry.

        This method:
        1. Gets pre-discovered tools from SharedToolRegistry
        2. Registers each tool with the target registry
        3. Applies enabled/disabled configuration

        Returns:
            CatalogLoadResult with loading statistics
        """
        result = CatalogLoadResult()

        # Register dynamic tools
        result.tools_loaded = self._register_from_shared_registry(full=True)
        result.full_catalog_loaded = True

        # Apply configuration
        result.tools_disabled = self._apply_configuration()

        self._loaded = True
        self._bootstrap_loaded = True
        logger.debug(
            f"ToolCatalogLoader: loaded {result.tools_loaded} tools, "
            f"disabled {result.tools_disabled}"
        )

        return result

    def load_bootstrap(self) -> CatalogLoadResult:
        """Load the compact startup tool set.

        This keeps session startup from importing/registering the full specialty
        catalog. Use ``load`` for the old eager behavior or ``ensure_tools`` for
        targeted hydration.
        """
        result = CatalogLoadResult()
        result.tools_loaded = self._register_from_shared_registry(full=False)
        result.tools_disabled = self._apply_configuration()
        self._bootstrap_loaded = True
        logger.debug(
            f"ToolCatalogLoader: bootstrap-loaded {result.tools_loaded} tools, "
            f"disabled {result.tools_disabled}"
        )
        return result

    def ensure_tools(self, tool_names: List[str]) -> CatalogLoadResult:
        """Register specific known tools without full catalog discovery."""
        result = CatalogLoadResult()
        if not tool_names:
            return result

        from victor.agent.shared_tool_registry import SharedToolRegistry
        from victor.tools.batch_registration import BatchRegistrar

        shared_registry = SharedToolRegistry.get_instance()
        already_registered = {
            getattr(tool, "name", "") for tool in self._registry.list_tools(only_enabled=False)
        }
        needed = [name for name in tool_names if name not in already_registered]
        if not needed:
            return result

        tools_to_register = shared_registry.get_tools_for_names(
            needed,
            airgapped_mode=self._config.airgapped_mode,
        )
        registrar = BatchRegistrar(self._registry)
        batch_result = registrar.register_batch(tools_to_register, fail_fast=False)
        result.tools_loaded = batch_result.success_count
        if batch_result.failure_count > 0:
            result.errors.extend(f"{name}: {error}" for name, error in batch_result.failed)
            for tool_name, error_msg in batch_result.failed:
                logger.debug(f"Skipped registering {tool_name}: {error_msg}")
        if result.tools_loaded:
            self._apply_configuration()
        return result

    def _register_from_shared_registry(self, *, full: bool) -> int:
        """Register tools from the shared tool registry.

        Uses SharedToolRegistry to get pre-discovered tool definitions,
        avoiding redundant discovery across multiple orchestrator instances.

        Returns:
            Number of tools registered
        """
        from victor.agent.shared_tool_registry import SharedToolRegistry

        shared_registry = SharedToolRegistry.get_instance()

        if full:
            tools_to_register = shared_registry.get_all_tools_for_registration(
                airgapped_mode=self._config.airgapped_mode
            )
        else:
            tools_to_register = shared_registry.get_bootstrap_tools_for_registration(
                airgapped_mode=self._config.airgapped_mode
            )

        # Use batch registration for performance (30-50 tools → single cache invalidation)
        from victor.tools.batch_registration import BatchRegistrar

        registrar = BatchRegistrar(self._registry)
        result = registrar.register_batch(tools_to_register, fail_fast=False)

        if result.failure_count > 0:
            for tool_name, error_msg in result.failed:
                logger.debug(f"Skipped registering {tool_name}: {error_msg}")

        return result.success_count

    def _apply_configuration(self) -> int:
        """Apply tool configuration from settings.

        Applies enabled/disabled states to tools based on configuration.

        Returns:
            Number of tools disabled
        """
        disabled_count = 0

        try:
            tool_config = self._settings.load_tool_config()

            if not tool_config:
                return disabled_count

            # Handle enabled list (if set, disable all others)
            enabled_list = tool_config.get("enabled", [])
            if enabled_list:
                for tool in self._registry.list_tools():
                    if tool.name not in enabled_list:
                        self._registry.disable_tool(tool.name)
                        disabled_count += 1
                logger.debug(f"Enabled {len(enabled_list)} tools from config")

            # Handle disabled list
            disabled_list = tool_config.get("disabled", [])
            for tool_name in disabled_list:
                self._registry.disable_tool(tool_name)
                disabled_count += 1
            if disabled_list:
                logger.debug(f"Disabled {len(disabled_list)} tools from config")

            # Handle per-tool configurations
            for tool_name, config in tool_config.items():
                if isinstance(config, dict) and "enabled" in config:
                    if not config["enabled"]:
                        self._registry.disable_tool(tool_name)
                        disabled_count += 1

        except Exception as e:
            logger.warning(f"Failed to load tool configurations: {e}")

        return disabled_count

    def get_tool_config(self) -> Dict[str, Any]:
        """Get tool configuration for context injection.

        Returns:
            Dictionary with tool-specific settings for context injection.
        """
        config: Dict[str, Any] = {}

        # Load web tool config if not air-gapped
        if not self._config.airgapped_mode:
            try:
                tool_config = self._settings.load_tool_config()
                web_cfg = tool_config.get("web_tools", {}) or tool_config.get("web", {}) or {}
                config.update(
                    {
                        "web_fetch_top": web_cfg.get("summarize_fetch_top"),
                        "web_fetch_pool": web_cfg.get("summarize_fetch_pool"),
                        "max_content_length": web_cfg.get("summarize_max_content_length"),
                    }
                )
            except Exception as exc:
                logger.debug(f"Failed to load web tool config: {exc}")

        return config


__all__ = ["ToolCatalogLoader", "ToolCatalogConfig", "CatalogLoadResult"]

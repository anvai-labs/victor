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

"""
Plugin registration for MCP vertical.
"""

from __future__ import annotations

from typing import Any

from victor_contracts.core.plugins import VictorPlugin
from victor_contracts.verticals.protocols.base import VerticalBase
from victor.verticals.mcp_vertical import MCPVertical


class MCPPlugin(VictorPlugin):
    """
    Plugin for MCP vertical registration.

    This plugin registers the MCP integration as a first-class vertical,
    enabling MCP servers to be discovered and managed through the standard
    vertical architecture.
    """

    name = "mcp"
    version = "1.0.0"
    description = "Model Context Protocol (MCP) integration vertical"

    def register(self, context: Any) -> None:
        """
        Register MCP vertical with the framework.

        Args:
            context: Plugin registration context
        """
        context.register_vertical(MCPVertical)


# Entry point: victor.plugins
def plugin() -> MCPPlugin:
    """
    Entry point for MCP plugin.

    Returns:
        MCPPlugin instance
    """
    return MCPPlugin()

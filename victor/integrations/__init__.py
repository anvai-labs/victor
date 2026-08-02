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

"""Victor Integrations Module.

This module consolidates all integration-related functionality:

- api: HTTP API servers for IDE integrations (VS Code, JetBrains)
- mcp: Model Context Protocol client/server implementations
- protocol: Victor Protocol interface for unified client communication
- protocols: Query enhancement protocols (core protocols live in victor.protocols)

Example usage:
    # HTTP API
    from victor.integrations.api import VictorFastAPIServer

    # MCP
    from victor.integrations.mcp import MCPClient, MCPServer, MCPRegistry

    # Protocol interface
    from victor.integrations.protocol import VictorProtocol, DirectProtocolAdapter

    # SOLID protocols
    from victor.protocols import IProviderAdapter, IGroundingStrategy

    # Query enhancement protocols
    from victor.protocols.query_enhancement import EnhancementContext
"""

# Expose submodules for easier access
from victor.integrations import api
from victor.integrations import mcp
from victor.integrations import protocol

__all__ = [
    "api",
    "mcp",
    "protocol",
]

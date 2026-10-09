"""Legacy Python protocol adapters and public types.

These opt-in adapters remain import-compatible for external consumers. Current
CLI, VS Code and MCP paths have separate application-service adapters; this
package is not the canonical shared service contract or a parity guarantee.
"""

from victor.integrations.protocol.interface import (
    VictorProtocol,
    ChatMessage,
    ChatResponse,
    ClientStreamChunk,
    ToolCall,
    ToolResult,
    UndoRedoResult,
    AgentMode,
    AgentStatus,
)
from victor.integrations.search_types import CodeSearchResult
from victor.integrations.protocol.adapters import (
    DirectProtocolAdapter,
    HTTPProtocolAdapter,
)

__all__ = [
    # Protocol interface
    "VictorProtocol",
    # Data types
    "ChatMessage",
    "ChatResponse",
    "ClientStreamChunk",
    "CodeSearchResult",
    "ToolCall",
    "ToolResult",
    "UndoRedoResult",
    "AgentMode",
    "AgentStatus",
    # Adapters
    "DirectProtocolAdapter",
    "HTTPProtocolAdapter",
]

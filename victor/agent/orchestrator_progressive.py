"""Progressive-tools support for the orchestrator.

Moved verbatim from ``victor/agent/orchestrator.py`` (facade decomposition).
Registration still runs at orchestrator import time: that module imports this
one at the top of its import block, preserving the original module-load
ordering semantics.
"""

from typing import Any, List, Set

from victor.tools.progressive_registry import get_progressive_registry
from victor.tools.tool_names import TOOL_ALIASES, ToolNames

# Tools with progressive parameters - different params = progress, not a loop
# Format: tool_name -> list of param names that indicate progress
# NOTE: Includes both canonical short names and legacy names for LLM compatibility
# NOTE: These values are registered with ProgressiveToolsRegistry for extensibility
_PROGRESSIVE_TOOLS_CONFIG = {
    # Canonical short names
    "read": ["path", "offset", "limit"],
    "grep": ["query", "directory"],
    "search": ["query", "directory"],
    "ls": ["path", "recursive"],
    "shell": ["cmd"],
    "git": ["operation", "files", "branch"],
    "http": ["url", "method"],
    "web": ["query"],
    "summarize": ["query"],
    "fetch": ["url"],
    "code_search": ["query", "directory"],
    "semantic_code_search": ["query", "directory"],
    "http_request": ["url", "method"],
    "web_search": ["query"],
    "web_summarize": ["query"],
    "web_fetch": ["url"],
}


def _register_progressive_tools() -> None:
    """Register all progressive tools with the ProgressiveToolsRegistry.

    This function is idempotent and can be called multiple times safely.
    Tools are only registered if not already present in the registry.
    """
    registry = get_progressive_registry()
    for tool_name, params in _PROGRESSIVE_TOOLS_CONFIG.items():
        if not registry.is_progressive(tool_name):
            # Convert list of param names to dict format expected by registry
            progressive_params = dict.fromkeys(params, "any")
            registry.register(tool_name, progressive_params)


def _ensure_progressive_tools_registered() -> None:
    """Ensure progressive tools are registered in the registry.

    This is called lazily on first access to handle cases where
    the registry singleton was reset (e.g., during testing).
    """
    registry = get_progressive_registry()
    # Quick check if already registered by checking for one known tool
    if not registry.is_progressive("read"):
        _register_progressive_tools()


# Register tools at module load time
_register_progressive_tools()


class _ProgressiveToolsProxy:
    """Proxy class that delegates to ProgressiveToolsRegistry for backward compatibility.

    This allows existing code using PROGRESSIVE_TOOLS dict-like access to continue working
    while the actual data is managed by the registry.

    The proxy ensures tools are re-registered if the registry was reset (e.g., during tests).
    """

    def __getitem__(self, key: str) -> List[str]:
        _ensure_progressive_tools_registered()
        registry = get_progressive_registry()
        config = registry.get_config(key)
        if config is None:
            raise KeyError(key)
        return list(config.progressive_params.keys())

    def __contains__(self, key: str) -> bool:
        _ensure_progressive_tools_registered()
        registry = get_progressive_registry()
        return registry.is_progressive(key)

    def get(self, key: str, default: Any = None) -> Any:
        _ensure_progressive_tools_registered()
        registry = get_progressive_registry()
        config = registry.get_config(key)
        if config is None:
            return default
        return list(config.progressive_params.keys())

    def keys(self) -> Set[str]:
        _ensure_progressive_tools_registered()
        registry = get_progressive_registry()
        return registry.list_progressive_tools()

    def items(self):
        _ensure_progressive_tools_registered()
        registry = get_progressive_registry()
        for tool_name in registry.list_progressive_tools():
            config = registry.get_config(tool_name)
            if config:
                yield tool_name, list(config.progressive_params.keys())

    def __iter__(self):
        _ensure_progressive_tools_registered()
        registry = get_progressive_registry()
        return iter(registry.list_progressive_tools())


# Backward-compatible constant that delegates to the registry
PROGRESSIVE_TOOLS = _ProgressiveToolsProxy()

# Build set of all known tool names (canonical + aliases) for detection
_ALL_TOOL_NAMES: Set[str] = set()
for attr in dir(ToolNames):
    if not attr.startswith("_"):
        val = getattr(ToolNames, attr)
        if isinstance(val, str):
            _ALL_TOOL_NAMES.add(val)
_ALL_TOOL_NAMES.update(TOOL_ALIASES.keys())


def _detect_mentioned_tools(text: str) -> List[str]:
    """Detect tool names mentioned in text that model said it would call.

    Looks for patterns like:
    - "let me call read()"
    - "I'll use web_search to"
    - "calling the ls tool"
    - "execute grep"

    Args:
        text: Model response text

    Returns:
        List of mentioned tool names (canonical form)
    """
    import re

    mentioned: List[str] = []
    text_lower = text.lower()

    # Look for tool names followed by common patterns
    for tool_name in _ALL_TOOL_NAMES:
        # Match patterns like: call read, use read, execute read, run read
        # Also: read() or read( with args
        patterns = [
            rf"\b(?:call|use|execute|run|invoke|perform)\s+{re.escape(tool_name)}\b",
            rf"\b{re.escape(tool_name)}\s*\(",  # tool_name( or tool_name (
            rf"\bthe\s+{re.escape(tool_name)}\s+tool\b",  # "the read tool"
        ]
        for pattern in patterns:
            if re.search(pattern, text_lower):
                # Resolve to canonical name
                canonical = TOOL_ALIASES.get(tool_name, tool_name)
                if canonical not in mentioned:
                    mentioned.append(canonical)
                break

    return mentioned

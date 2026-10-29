"""
Provider-Specific Tool Guidance (Strategy Pattern).

This module implements the Strategy pattern to provide provider-specific tool
usage guidance, addressing differences in how various LLM providers handle
tool calling.

SOLID Principles Applied:
- Single Responsibility: Each strategy handles guidance for one provider
- Open/Closed: New providers can be added without modifying existing code
- Liskov Substitution: All strategies are interchangeable
- Interface Segregation: ToolGuidanceStrategy defines minimal interface
- Dependency Inversion: Consumers depend on abstraction, not concrete strategies

Addresses GAP-5: Excessive tool calling (DeepSeek)
Addresses GAP-7: Over-exploration without synthesis
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any
import logging

from victor.tools.tool_names import ToolNames, get_canonical_name

logger = logging.getLogger(__name__)


def _canonical_tool_name(tool_name: str) -> str:
    """Normalize tool aliases before provider-specific scoring."""
    return get_canonical_name(tool_name)


def _canonicalize_tools_for_prompt(available_tools: List[str]) -> List[str]:
    """Return a stable, deduplicated tool list for model-facing guidance."""
    normalized: List[str] = []
    seen: set[str] = set()
    for tool_name in available_tools:
        canonical = _canonical_tool_name(tool_name)
        if canonical and canonical not in seen:
            normalized.append(canonical)
            seen.add(canonical)
    return normalized


class ToolGuidanceStrategy(ABC):
    """
    Abstract strategy interface for provider-specific tool guidance.

    Each provider has different tool calling behaviors that require
    different guidance approaches to optimize performance.
    """

    @abstractmethod
    def get_guidance_prompt(self, task_type: str, available_tools: List[str]) -> str:
        """
        Return provider-specific tool usage guidance.

        Args:
            task_type: Type of task (simple, medium, complex)
            available_tools: List of available tool names

        Returns:
            Guidance text to inject into system prompt
        """
        pass

    @abstractmethod
    def should_consolidate_calls(self, tool_history: List[Dict[str, Any]]) -> bool:
        """
        Determine if recent tool calls should be consolidated.

        Args:
            tool_history: List of recent tool calls with 'tool' and 'args' keys

        Returns:
            True if consolidation/synthesis is recommended
        """
        pass

    @abstractmethod
    def get_max_exploration_depth(self, task_complexity: str) -> int:
        """
        Return max tool calls before forcing synthesis.

        Args:
            task_complexity: Task complexity level (simple, medium, complex)

        Returns:
            Maximum number of tool calls before synthesis checkpoint
        """
        pass

    @abstractmethod
    def get_synthesis_checkpoint_prompt(self, tool_count: int) -> str:
        """
        Return prompt to inject when synthesis checkpoint is reached.

        Args:
            tool_count: Number of tool calls made so far

        Returns:
            Synthesis prompt or empty string if not needed
        """
        pass

    @abstractmethod
    def get_tool_boost(self, tool_name: str) -> float:
        """
        Get boost factor for a specific tool with this provider.

        This allows providers to boost tools that align with their strengths
        (e.g., Gemini boosts code analysis tools, Groq boosts fast operations).

        Args:
            tool_name: Name of the tool

        Returns:
            Boost factor (1.0 = no boost, >1.0 = boost, <1.0 = penalty)
        """
        pass

    def adjust_tool_scores(self, tool_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Adjust tool scores based on provider preferences.

        Args:
            tool_scores: Dictionary mapping tool names to scores

        Returns:
            Adjusted tool scores with provider-specific boosts/penalties
        """
        adjusted = {}
        for tool_name, score in tool_scores.items():
            boost = self.get_tool_boost(tool_name)
            adjusted[tool_name] = score * boost
        return adjusted


class GrokToolGuidance(ToolGuidanceStrategy):
    """
    Grok (xAI) tool guidance strategy.

    Grok handles tools efficiently with minimal redundancy, requiring
    little to no additional guidance.
    """

    PREFERRED_TOOLS = {
        ToolNames.READ,
        "code_search",
        "grep",
        ToolNames.WRITE,
        ToolNames.EDIT,
        ToolNames.LS,
        ToolNames.SHELL,
        "git",
    }

    def get_guidance_prompt(self, task_type: str, available_tools: List[str]) -> str:
        # Grok handles tools well, minimal guidance needed
        return ""

    def should_consolidate_calls(self, tool_history: List[Dict[str, Any]]) -> bool:
        # Grok already consolidates well
        return False

    def get_max_exploration_depth(self, task_complexity: str) -> int:
        depths = {
            "simple": 5,
            "medium": 10,
            "complex": 20,
        }
        return depths.get(task_complexity, 10)

    def get_synthesis_checkpoint_prompt(self, tool_count: int) -> str:
        # Grok typically doesn't need synthesis checkpoints
        return ""

    def get_tool_boost(self, tool_name: str) -> float:
        # Boost fast operations for Grok's speed advantage
        if _canonical_tool_name(tool_name) in self.PREFERRED_TOOLS:
            return 1.15  # 15% boost
        return 1.0


class DeepSeekToolGuidance(ToolGuidanceStrategy):
    """
    DeepSeek tool guidance strategy.

    DeepSeek tends to over-explore with redundant tool calls,
    requiring explicit guidance to minimize tool usage and
    synthesize findings earlier.
    """

    SYNTHESIS_THRESHOLD = 5  # Trigger synthesis after N tool calls
    PREFERRED_TOOLS = {
        "architectural_summary",
        "code_review",
        "refactor_code",
        "plan_implementation",
        "security_scan",
        "code_metrics",
        "shell",
        "git",
    }

    def get_guidance_prompt(self, task_type: str, available_tools: List[str]) -> str:
        base_guidance = """
IMPORTANT - Tool Usage Guidelines:
1. Be efficient with tool calls - minimize redundant operations
2. Check if you already have the information before making a tool call
3. Prefer broad queries over multiple narrow ones
4. Consolidate findings and synthesize before continuing exploration

For git operations (status, diff, log, branch, commit, push):
- PREFER the unified ``git`` tool for version control, NOT raw shell git:
  • git(cmd="status")  /  git(cmd="diff")  /  git(cmd="diff --staged")
  • git(cmd="log -n 10")  /  git(cmd="stage <files>")  /  git(cmd="commit -m \\"msg\\"")
  • git(cmd="commit --ai") to auto-generate a commit message
  • git(cmd="push origin <branch>")
- It delegates to the richer git implementation (AI commit messages, conflict
  analysis, PRs) when available and falls back to plain ``git`` otherwise.
- Reserve ``shell`` for ad-hoc git plumbing the ``git`` tool does not cover.

For executing shell commands:
- shell() defaults to readonly=False — the dangerous-command check is the safety floor.
- shell(cmd=..., readonly=True) validates against the readonly allowlist (opt-in for purely read-only commands).
  • Use ``write`` for new files and ``edit`` for modifications; prefer over shell redirection.
- Use dangerous=True only for genuinely destructive commands such as rm or kill

Code & file search:
- Use ``code grep "query" path`` for literal content search (add --regex / -C).
- Use ``code search "query" --mode semantic`` for semantic code search.
- Use ``shell`` with ``find . -name "pattern"`` to find files by name.
"""

        if task_type == "simple":
            return base_guidance + """
For this simple task, aim to complete with 1-2 tool calls.
"""
        elif task_type == "medium":
            return base_guidance + """
After 5 tool calls, pause to synthesize your findings before continuing.
"""
        else:  # complex
            return base_guidance + """
For complex tasks, regularly synthesize findings (every 5-7 tool calls).
Focus on depth over breadth when exploring.
"""

    def should_consolidate_calls(self, tool_history: List[Dict[str, Any]]) -> bool:
        if len(tool_history) < 2:
            return False

        # Check for same file accessed multiple times
        files_accessed = []
        for entry in tool_history:
            args = entry.get("args", {})
            path = args.get("path") or args.get("file_path") or args.get("file")
            if path:
                files_accessed.append(path)

        # Duplicate file access detected
        if len(files_accessed) != len(set(files_accessed)):
            return True

        # Check for same tool called 3+ times
        if len(tool_history) >= 3:
            recent_tools = [_canonical_tool_name(h.get("tool", "")) for h in tool_history[-5:]]
            tool_counts = {}
            for tool in recent_tools:
                tool_counts[tool] = tool_counts.get(tool, 0) + 1
                if tool_counts[tool] >= 3:
                    return True

        # Check for excessive ls/listing calls
        ls_count = sum(
            1 for h in tool_history if _canonical_tool_name(h.get("tool", "")) == ToolNames.LS
        )
        if ls_count >= 4:
            return True

        return False

    def get_max_exploration_depth(self, task_complexity: str) -> int:
        depths = {
            "simple": 3,
            "medium": 6,
            "complex": 12,
        }
        return depths.get(task_complexity, 5)

    def get_synthesis_checkpoint_prompt(self, tool_count: int) -> str:
        if tool_count < self.SYNTHESIS_THRESHOLD:
            return ""

        return """
SYNTHESIS CHECKPOINT: You've made several tool calls. Before continuing:
1. Summarize what you've learned so far
2. Identify if you have enough information to answer
3. Only continue exploration if absolutely necessary
"""

    def get_tool_boost(self, tool_name: str) -> float:
        # Boost reasoning-heavy tasks for DeepSeek's thinking mode
        if _canonical_tool_name(tool_name) in self.PREFERRED_TOOLS:
            return 1.25  # 25% boost for reasoning tasks
        return 1.0


class OllamaToolGuidance(ToolGuidanceStrategy):
    """
    Ollama (local models) tool guidance strategy.

    Local models often have weaker tool calling capabilities and
    require more explicit guidance and stricter boundaries.
    """

    PREFERRED_TOOLS = {
        ToolNames.READ,
        ToolNames.WRITE,
        ToolNames.EDIT,
        ToolNames.LS,
        "grep",
        ToolNames.SHELL,
        "git",
        "code_search",
    }
    AVOIDED_TOOLS = {"web_search", "web_fetch"}  # Air-gapped environments

    def get_guidance_prompt(self, task_type: str, available_tools: List[str]) -> str:
        tools_str = ", ".join(_canonicalize_tools_for_prompt(available_tools)[:5])

        return f"""
TOOL USAGE INSTRUCTIONS:
Available tools: {tools_str}

Guidelines:
1. Use ONE tool at a time
2. Wait for tool results before deciding next action
3. Prefer simpler tools (read, ls) over complex ones
4. Complete the task with minimal tool calls
5. Synthesize findings clearly in your response
"""

    def should_consolidate_calls(self, tool_history: List[Dict[str, Any]]) -> bool:
        # More aggressive consolidation for local models
        if len(tool_history) < 2:
            return False

        # Consolidate after 3 calls
        return len(tool_history) >= 3

    def get_max_exploration_depth(self, task_complexity: str) -> int:
        depths = {
            "simple": 2,
            "medium": 4,
            "complex": 8,
        }
        return depths.get(task_complexity, 4)

    def get_synthesis_checkpoint_prompt(self, tool_count: int) -> str:
        if tool_count < 3:
            return ""

        return "Summarize your findings and provide an answer based on what you've learned."

    def get_tool_boost(self, tool_name: str) -> float:
        # Prefer offline tools for air-gapped/local deployments
        if _canonical_tool_name(tool_name) in self.PREFERRED_TOOLS:
            return 1.1  # 10% boost for offline-safe tools
        elif tool_name in self.AVOIDED_TOOLS:
            return 0.5  # 50% penalty for tools requiring internet
        return 1.0


class AnthropicToolGuidance(ToolGuidanceStrategy):
    """
    Anthropic (Claude) tool guidance strategy.

    Claude handles tools very well with good judgment about when
    to use them, requiring minimal additional guidance.
    """

    PREFERRED_TOOLS = {
        "code_review",
        "architectural_summary",
        "refactor_code",
        "security_scan",
        ToolNames.WRITE,
        "generate_docs",
        ToolNames.EDIT,
    }

    def get_guidance_prompt(self, task_type: str, available_tools: List[str]) -> str:
        # Claude handles tools well, minimal guidance needed
        return ""

    def should_consolidate_calls(self, tool_history: List[Dict[str, Any]]) -> bool:
        # Claude typically manages tool usage well
        return False

    def get_max_exploration_depth(self, task_complexity: str) -> int:
        depths = {
            "simple": 5,
            "medium": 12,
            "complex": 25,
        }
        return depths.get(task_complexity, 12)

    def get_synthesis_checkpoint_prompt(self, tool_count: int) -> str:
        # Claude typically doesn't need forced synthesis
        return ""

    def get_tool_boost(self, tool_name: str) -> float:
        # Boost reasoning and analysis tools for Claude's strengths
        if _canonical_tool_name(tool_name) in self.PREFERRED_TOOLS:
            return 1.2  # 20% boost for reasoning/analysis tasks
        return 1.0


class OpenAIToolGuidance(ToolGuidanceStrategy):
    """
    OpenAI (GPT-4) tool guidance strategy.

    GPT-4 handles tools reasonably well but can sometimes benefit
    from light guidance on efficiency.
    """

    PREFERRED_TOOLS = {
        ToolNames.WRITE,
        ToolNames.EDIT,
        "plan_implementation",
        ToolNames.SHELL,
        "git",
        ToolNames.READ,
        "grep",
        ToolNames.LS,
    }

    def get_guidance_prompt(self, task_type: str, available_tools: List[str]) -> str:
        if task_type == "simple":
            return "Be efficient with tool calls."
        return ""

    def should_consolidate_calls(self, tool_history: List[Dict[str, Any]]) -> bool:
        # Check for obvious redundancy
        if len(tool_history) < 3:
            return False

        # Same file read twice
        files_read = []
        for entry in tool_history:
            if _canonical_tool_name(entry.get("tool", "")) == ToolNames.READ:
                path = entry.get("args", {}).get("path")
                if path:
                    files_read.append(path)

        return len(files_read) != len(set(files_read))

    def get_max_exploration_depth(self, task_complexity: str) -> int:
        depths = {
            "simple": 4,
            "medium": 8,
            "complex": 18,
        }
        return depths.get(task_complexity, 8)

    def get_synthesis_checkpoint_prompt(self, tool_count: int) -> str:
        if tool_count >= 10:
            return "Consider synthesizing your findings before continuing."
        return ""

    def get_tool_boost(self, tool_name: str) -> float:
        # Boost general-purpose tools for GPT-4's versatility
        if _canonical_tool_name(tool_name) in self.PREFERRED_TOOLS:
            return 1.2  # 20% boost for general-purpose operations
        return 1.0


class DefaultToolGuidance(ToolGuidanceStrategy):
    """
    Default tool guidance for unknown providers.

    Uses conservative settings to work reasonably across
    different providers.
    """

    def get_guidance_prompt(self, task_type: str, available_tools: List[str]) -> str:
        return "Use tools efficiently and synthesize findings clearly."

    def should_consolidate_calls(self, tool_history: List[Dict[str, Any]]) -> bool:
        return len(tool_history) >= 5

    def get_max_exploration_depth(self, task_complexity: str) -> int:
        depths = {
            "simple": 3,
            "medium": 6,
            "complex": 12,
        }
        return depths.get(task_complexity, 6)

    def get_synthesis_checkpoint_prompt(self, tool_count: int) -> str:
        if tool_count >= 5:
            return "Synthesize your findings before continuing."
        return ""

    def get_tool_boost(self, tool_name: str) -> float:
        # No specific preferences for unknown providers
        return 1.0


# Strategy registry (cached instances)
_strategy_cache: Dict[str, ToolGuidanceStrategy] = {}

# Provider name mappings
_provider_mappings: Dict[str, type] = {
    "grok": GrokToolGuidance,
    "xai": GrokToolGuidance,
    "deepseek": DeepSeekToolGuidance,
    "ollama": OllamaToolGuidance,
    "lmstudio": OllamaToolGuidance,
    "vllm": OllamaToolGuidance,
    "anthropic": AnthropicToolGuidance,
    "claude": AnthropicToolGuidance,
    "openai": OpenAIToolGuidance,
    "gpt": OpenAIToolGuidance,
    "google": OpenAIToolGuidance,  # Similar to OpenAI in tool handling
    "gemini": OpenAIToolGuidance,
    "zai": OpenAIToolGuidance,  # Z.AI GLM uses OpenAI-compatible tool calling
    "zhipu": OpenAIToolGuidance,
    "moonshot": OpenAIToolGuidance,  # Kimi uses OpenAI-compatible format
    "cerebras": OpenAIToolGuidance,
    "groqcloud": OpenAIToolGuidance,
    "fireworks": OpenAIToolGuidance,
    "mistral": OpenAIToolGuidance,
    "together": OpenAIToolGuidance,
}


def get_tool_guidance_strategy(provider_name: str) -> ToolGuidanceStrategy:
    """
    Get the tool guidance strategy for a provider.

    Args:
        provider_name: Name of the LLM provider

    Returns:
        Appropriate ToolGuidanceStrategy instance (cached)
    """
    # Normalize provider name
    provider_key = provider_name.lower().strip()

    # Check cache first
    if provider_key in _strategy_cache:
        return _strategy_cache[provider_key]

    # Find strategy class
    strategy_class = _provider_mappings.get(provider_key, DefaultToolGuidance)

    # Create and cache instance
    strategy = strategy_class()
    _strategy_cache[provider_key] = strategy

    logger.debug(f"Created tool guidance strategy for {provider_name}: {strategy_class.__name__}")

    return strategy


def clear_strategy_cache() -> None:
    """Clear the strategy cache (mainly for testing)."""
    _strategy_cache.clear()

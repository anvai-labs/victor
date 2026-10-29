"""
Tests for Provider-Specific Tool Guidance (Strategy Pattern).

Addresses GAP-5: Excessive tool calling by DeepSeek
Addresses GAP-7: Over-exploration without synthesis

SOLID Principles:
- Strategy Pattern for provider-specific behavior
- Open/Closed: New providers can be added without modifying existing code
- Liskov Substitution: All strategies are interchangeable
"""

import pytest


class TestToolGuidanceStrategy:
    """Tests for the abstract ToolGuidanceStrategy interface."""

    def test_strategy_interface_defines_required_methods(self):
        """Strategy interface should define get_guidance_prompt, should_consolidate_calls, get_max_exploration_depth."""
        from victor.agent.provider_tool_guidance import ToolGuidanceStrategy

        # Check abstract methods exist
        assert hasattr(ToolGuidanceStrategy, "get_guidance_prompt")
        assert hasattr(ToolGuidanceStrategy, "should_consolidate_calls")
        assert hasattr(ToolGuidanceStrategy, "get_max_exploration_depth")
        assert hasattr(ToolGuidanceStrategy, "get_synthesis_checkpoint_prompt")

    def test_cannot_instantiate_abstract_strategy(self):
        """Abstract strategy should not be directly instantiable."""
        from victor.agent.provider_tool_guidance import ToolGuidanceStrategy

        with pytest.raises(TypeError):
            ToolGuidanceStrategy()


class TestGrokToolGuidance:
    """Tests for Grok-specific tool guidance strategy."""

    @pytest.fixture
    def grok_strategy(self):
        from victor.agent.provider_tool_guidance import GrokToolGuidance

        return GrokToolGuidance()

    def test_grok_guidance_prompt_is_minimal(self, grok_strategy):
        """Grok handles tools efficiently, needs minimal guidance."""
        prompt = grok_strategy.get_guidance_prompt(
            task_type="analysis", available_tools=["read", "grep", "ls"]
        )
        # Grok needs minimal or no guidance
        assert len(prompt) < 100  # Minimal guidance

    def test_grok_should_not_consolidate_by_default(self, grok_strategy):
        """Grok already consolidates well, shouldn't trigger consolidation."""
        tool_history = [
            {"tool": "read", "args": {"path": "file1.py"}},
            {"tool": "grep", "args": {"pattern": "class"}},
            {"tool": "read", "args": {"path": "file2.py"}},
        ]
        assert grok_strategy.should_consolidate_calls(tool_history) is False

    def test_grok_max_exploration_depth_simple(self, grok_strategy):
        """Simple tasks should have low exploration depth."""
        assert grok_strategy.get_max_exploration_depth("simple") <= 5

    def test_grok_max_exploration_depth_medium(self, grok_strategy):
        """Medium tasks should have moderate exploration depth."""
        depth = grok_strategy.get_max_exploration_depth("medium")
        assert 5 <= depth <= 12

    def test_grok_max_exploration_depth_complex(self, grok_strategy):
        """Complex tasks should have higher exploration depth."""
        depth = grok_strategy.get_max_exploration_depth("complex")
        assert depth >= 10

    def test_grok_synthesis_checkpoint_returns_empty(self, grok_strategy):
        """Grok doesn't need synthesis checkpoints typically."""
        prompt = grok_strategy.get_synthesis_checkpoint_prompt(tool_count=5)
        assert prompt == "" or len(prompt) < 50


class TestDeepSeekToolGuidance:
    """Tests for DeepSeek-specific tool guidance strategy."""

    @pytest.fixture
    def deepseek_strategy(self):
        from victor.agent.provider_tool_guidance import DeepSeekToolGuidance

        return DeepSeekToolGuidance()

    def test_deepseek_guidance_prompt_is_explicit(self, deepseek_strategy):
        """DeepSeek needs explicit guidance to avoid over-exploration."""
        prompt = deepseek_strategy.get_guidance_prompt(
            task_type="analysis", available_tools=["read", "grep", "ls"]
        )
        # Should contain guidance about minimizing tool calls
        assert "minimize" in prompt.lower() or "efficient" in prompt.lower()

    def test_deepseek_guidance_contains_consolidation_hint(self, deepseek_strategy):
        """DeepSeek guidance should mention synthesizing findings."""
        prompt = deepseek_strategy.get_guidance_prompt(
            task_type="analysis", available_tools=["read", "grep", "ls"]
        )
        assert "synthesize" in prompt.lower() or "consolidate" in prompt.lower()

    def test_deepseek_detects_duplicate_file_reads(self, deepseek_strategy):
        """DeepSeek should detect when same file is read multiple times."""
        tool_history = [
            {"tool": "read", "args": {"path": "file1.py"}},
            {"tool": "grep", "args": {"pattern": "class"}},
            {"tool": "read", "args": {"path": "file1.py"}},  # Duplicate
        ]
        assert deepseek_strategy.should_consolidate_calls(tool_history) is True

    def test_deepseek_detects_excessive_ls_calls(self, deepseek_strategy):
        """DeepSeek should detect excessive ls calls."""
        tool_history = [
            {"tool": "ls", "args": {"path": "."}},
            {"tool": "ls", "args": {"path": "./src"}},
            {"tool": "ls", "args": {"path": "./src/utils"}},
            {"tool": "ls", "args": {"path": "./src/models"}},
            {"tool": "ls", "args": {"path": "./tests"}},
        ]
        assert deepseek_strategy.should_consolidate_calls(tool_history) is True

    def test_deepseek_no_consolidation_for_diverse_tools(self, deepseek_strategy):
        """No consolidation needed for diverse tool usage."""
        tool_history = [
            {"tool": "read", "args": {"path": "file1.py"}},
            {"tool": "grep", "args": {"pattern": "class"}},
            {"tool": "symbol", "args": {"symbol_name": "MyClass"}},
        ]
        assert deepseek_strategy.should_consolidate_calls(tool_history) is False

    def test_deepseek_max_exploration_depth_simple(self, deepseek_strategy):
        """Simple tasks should have very low exploration depth for DeepSeek."""
        assert deepseek_strategy.get_max_exploration_depth("simple") <= 3

    def test_deepseek_max_exploration_depth_medium(self, deepseek_strategy):
        """Medium tasks should have lower depth than Grok."""
        depth = deepseek_strategy.get_max_exploration_depth("medium")
        assert 3 <= depth <= 8

    def test_deepseek_max_exploration_depth_complex(self, deepseek_strategy):
        """Complex tasks should still be bounded."""
        depth = deepseek_strategy.get_max_exploration_depth("complex")
        assert 8 <= depth <= 15

    def test_deepseek_synthesis_checkpoint_at_threshold(self, deepseek_strategy):
        """DeepSeek should get synthesis prompt after N tool calls."""
        prompt = deepseek_strategy.get_synthesis_checkpoint_prompt(tool_count=5)
        assert len(prompt) > 0
        # Should contain synthesis-related keywords
        prompt_lower = prompt.lower()
        assert any(
            kw in prompt_lower
            for kw in ["synthesize", "findings", "summarize", "synthesis", "learned"]
        )

    def test_deepseek_no_synthesis_below_threshold(self, deepseek_strategy):
        """No synthesis prompt for low tool counts."""
        prompt = deepseek_strategy.get_synthesis_checkpoint_prompt(tool_count=2)
        assert prompt == ""


class TestOllamaToolGuidance:
    """Tests for Ollama-specific tool guidance (local models)."""

    @pytest.fixture
    def ollama_strategy(self):
        from victor.agent.provider_tool_guidance import OllamaToolGuidance

        return OllamaToolGuidance()

    def test_ollama_guidance_is_strict(self, ollama_strategy):
        """Ollama (local models) need stricter guidance due to weaker tool use."""
        prompt = ollama_strategy.get_guidance_prompt(
            task_type="analysis", available_tools=["read", "grep", "ls"]
        )
        # Should contain explicit tool usage instructions
        assert len(prompt) > 50

    def test_ollama_lower_exploration_depth(self, ollama_strategy):
        """Ollama should have lower exploration depth."""
        assert ollama_strategy.get_max_exploration_depth("complex") <= 10

    def test_ollama_guidance_canonicalizes_available_tool_preview(self, ollama_strategy):
        """Model-facing tool previews should use canonical core tool names."""
        prompt = ollama_strategy.get_guidance_prompt(
            task_type="analysis",
            available_tools=["list_directory", "read_file", "execute_bash", "read"],
        )

        assert "list_directory" not in prompt
        assert "read_file" not in prompt
        assert "execute_bash" not in prompt
        assert "Available tools: ls, read, shell" in prompt


class TestAnthropicToolGuidance:
    """Tests for Anthropic (Claude) tool guidance strategy."""

    @pytest.fixture
    def anthropic_strategy(self):
        from victor.agent.provider_tool_guidance import AnthropicToolGuidance

        return AnthropicToolGuidance()

    def test_anthropic_guidance_is_minimal(self, anthropic_strategy):
        """Anthropic models handle tools well, need minimal guidance."""
        prompt = anthropic_strategy.get_guidance_prompt(
            task_type="analysis", available_tools=["read", "grep", "ls"]
        )
        assert len(prompt) < 100

    def test_anthropic_high_exploration_depth(self, anthropic_strategy):
        """Anthropic can handle higher exploration depth."""
        assert anthropic_strategy.get_max_exploration_depth("complex") >= 15


class TestOpenAIToolGuidance:
    """Tests for OpenAI (GPT-4) tool guidance strategy."""

    @pytest.fixture
    def openai_strategy(self):
        from victor.agent.provider_tool_guidance import OpenAIToolGuidance

        return OpenAIToolGuidance()

    def test_openai_guidance_is_moderate(self, openai_strategy):
        """OpenAI needs moderate guidance."""
        prompt = openai_strategy.get_guidance_prompt(
            task_type="analysis", available_tools=["read", "grep", "ls"]
        )
        assert len(prompt) < 200

    def test_openai_alias_boost_matches_canonical(self, openai_strategy):
        """Legacy tool aliases should resolve to the canonical boost class."""
        assert openai_strategy.get_tool_boost("read_file") == openai_strategy.get_tool_boost("read")


class TestToolGuidanceRegistry:
    """Tests for the strategy registry/factory."""

    def test_get_strategy_for_grok(self):
        """Should return GrokToolGuidance for grok provider."""
        from victor.agent.provider_tool_guidance import (
            get_tool_guidance_strategy,
            GrokToolGuidance,
        )

        strategy = get_tool_guidance_strategy("grok")
        assert isinstance(strategy, GrokToolGuidance)

    def test_get_strategy_for_xai(self):
        """Should return GrokToolGuidance for xai provider (alias)."""
        from victor.agent.provider_tool_guidance import (
            get_tool_guidance_strategy,
            GrokToolGuidance,
        )

        strategy = get_tool_guidance_strategy("xai")
        assert isinstance(strategy, GrokToolGuidance)

    def test_get_strategy_for_deepseek(self):
        """Should return DeepSeekToolGuidance for deepseek provider."""
        from victor.agent.provider_tool_guidance import (
            get_tool_guidance_strategy,
            DeepSeekToolGuidance,
        )

        strategy = get_tool_guidance_strategy("deepseek")
        assert isinstance(strategy, DeepSeekToolGuidance)

    def test_get_strategy_for_ollama(self):
        """Should return OllamaToolGuidance for ollama provider."""
        from victor.agent.provider_tool_guidance import (
            get_tool_guidance_strategy,
            OllamaToolGuidance,
        )

        strategy = get_tool_guidance_strategy("ollama")
        assert isinstance(strategy, OllamaToolGuidance)

    def test_get_strategy_for_anthropic(self):
        """Should return AnthropicToolGuidance for anthropic provider."""
        from victor.agent.provider_tool_guidance import (
            get_tool_guidance_strategy,
            AnthropicToolGuidance,
        )

        strategy = get_tool_guidance_strategy("anthropic")
        assert isinstance(strategy, AnthropicToolGuidance)

    def test_get_strategy_for_openai(self):
        """Should return OpenAIToolGuidance for openai provider."""
        from victor.agent.provider_tool_guidance import (
            get_tool_guidance_strategy,
            OpenAIToolGuidance,
        )

        strategy = get_tool_guidance_strategy("openai")
        assert isinstance(strategy, OpenAIToolGuidance)

    def test_get_strategy_for_unknown_provider(self):
        """Should return default strategy for unknown provider."""
        from victor.agent.provider_tool_guidance import (
            get_tool_guidance_strategy,
            DefaultToolGuidance,
        )

        strategy = get_tool_guidance_strategy("unknown_provider")
        assert isinstance(strategy, DefaultToolGuidance)

    def test_strategy_is_cached(self):
        """Strategies should be cached/singleton."""
        from victor.agent.provider_tool_guidance import get_tool_guidance_strategy

        strategy1 = get_tool_guidance_strategy("grok")
        strategy2 = get_tool_guidance_strategy("grok")
        assert strategy1 is strategy2


class TestConsolidationDetection:
    """Tests for tool call consolidation detection logic."""

    @pytest.fixture
    def deepseek_strategy(self):
        from victor.agent.provider_tool_guidance import DeepSeekToolGuidance

        return DeepSeekToolGuidance()

    def test_detect_same_tool_repeated_3_times(self, deepseek_strategy):
        """Detect when same tool called 3+ times consecutively."""
        tool_history = [
            {"tool": "ls", "args": {"path": "."}},
            {"tool": "ls", "args": {"path": "./src"}},
            {"tool": "ls", "args": {"path": "./tests"}},
        ]
        assert deepseek_strategy.should_consolidate_calls(tool_history) is True

    def test_detect_same_file_accessed_multiple_times(self, deepseek_strategy):
        """Detect when same file accessed multiple times."""
        tool_history = [
            {"tool": "read", "args": {"path": "main.py"}},
            {"tool": "symbol", "args": {"file_path": "main.py", "symbol_name": "foo"}},
            {"tool": "read", "args": {"path": "main.py", "offset": 100}},
        ]
        assert deepseek_strategy.should_consolidate_calls(tool_history) is True

    def test_detect_legacy_aliases_as_redundant(self, deepseek_strategy):
        """Legacy aliases should normalize before redundancy checks."""
        tool_history = [
            {"tool": "list_directory", "args": {"path": "."}},
            {"tool": "ls", "args": {"path": "./src"}},
            {"tool": "list_directory", "args": {"path": "./tests"}},
            {"tool": "ls", "args": {"path": "./docs"}},
        ]
        assert deepseek_strategy.should_consolidate_calls(tool_history) is True

    def test_no_consolidation_for_short_history(self, deepseek_strategy):
        """Don't consolidate for very short tool histories."""
        tool_history = [{"tool": "ls", "args": {"path": "."}}]
        assert deepseek_strategy.should_consolidate_calls(tool_history) is False

    def test_empty_history_no_consolidation(self, deepseek_strategy):
        """Empty history should not trigger consolidation."""
        assert deepseek_strategy.should_consolidate_calls([]) is False


class TestTaskTypeClassification:
    """Tests for task type classification affecting guidance."""

    @pytest.fixture
    def deepseek_strategy(self):
        from victor.agent.provider_tool_guidance import DeepSeekToolGuidance

        return DeepSeekToolGuidance()

    def test_simple_task_guidance(self, deepseek_strategy):
        """Simple tasks should have focused guidance."""
        prompt = deepseek_strategy.get_guidance_prompt(
            task_type="simple", available_tools=["ls", "read"]
        )
        assert "1-2 tool" in prompt.lower() or "minimal" in prompt.lower()

    def test_complex_task_guidance(self, deepseek_strategy):
        """Complex tasks can have more exploration room."""
        prompt = deepseek_strategy.get_guidance_prompt(
            task_type="complex", available_tools=["read", "grep", "symbol", "graph"]
        )
        # Should still have synthesis guidance
        assert "synthesize" in prompt.lower()


class TestToolGuidanceIntegration:
    """Integration tests for tool guidance with prompt builder."""

    def test_guidance_can_be_injected_into_system_prompt(self):
        """Guidance should be injectable into system prompts."""
        from victor.agent.provider_tool_guidance import get_tool_guidance_strategy

        strategy = get_tool_guidance_strategy("deepseek")
        guidance = strategy.get_guidance_prompt(
            task_type="medium", available_tools=["read", "grep"]
        )

        # Guidance should be valid prompt text
        assert isinstance(guidance, str)
        assert len(guidance) > 0

    def test_synthesis_checkpoint_can_be_injected(self):
        """Synthesis checkpoint should be injectable as user message."""
        from victor.agent.provider_tool_guidance import get_tool_guidance_strategy

        strategy = get_tool_guidance_strategy("deepseek")
        checkpoint = strategy.get_synthesis_checkpoint_prompt(tool_count=6)

        assert isinstance(checkpoint, str)
        # Should be actionable instruction
        if checkpoint:  # May be empty for some tool counts
            checkpoint_lower = checkpoint.lower()
            assert any(
                kw in checkpoint_lower
                for kw in ["synthesize", "summary", "summarize", "synthesis", "learned"]
            )

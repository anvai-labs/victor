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

"""Tests for SystemPromptBuilder module."""

from unittest.mock import MagicMock, patch

from victor.agent.prompt_builder import (
    ASI_TOOL_EFFECTIVENESS_GUIDANCE,
    COMPLETION_GUIDANCE,
    CONCISE_MODE_GUIDANCE,
    SystemPromptBuilder,
    GROUNDING_RULES,
    GROUNDING_RULES_EXTENDED,
    LARGE_FILE_PAGINATION_GUIDANCE,
    PARALLEL_READ_GUIDANCE,
    build_system_prompt,
    CLOUD_PROVIDERS,
    LOCAL_PROVIDERS,
)
from victor.agent.prompt_section_texts import (
    ASI_TOOL_EFFECTIVENESS_GUIDANCE as CANONICAL_ASI_TOOL_EFFECTIVENESS_GUIDANCE,
    COMPLETION_GUIDANCE as CANONICAL_COMPLETION_GUIDANCE,
    CONCISE_MODE_GUIDANCE as CANONICAL_CONCISE_MODE_GUIDANCE,
    GROUNDING_RULES as CANONICAL_GROUNDING_RULES,
    GROUNDING_RULES_EXTENDED as CANONICAL_GROUNDING_RULES_EXTENDED,
    LARGE_FILE_PAGINATION_GUIDANCE as CANONICAL_LARGE_FILE_PAGINATION_GUIDANCE,
    PARALLEL_READ_GUIDANCE as CANONICAL_PARALLEL_READ_GUIDANCE,
)


class TestSystemPromptBuilder:
    """Tests for SystemPromptBuilder class."""

    def test_shared_prompt_section_texts_are_reexported_from_canonical_module(self):
        """Prompt builder should consume and re-export canonical shared section text."""
        assert GROUNDING_RULES == CANONICAL_GROUNDING_RULES
        assert PARALLEL_READ_GUIDANCE == CANONICAL_PARALLEL_READ_GUIDANCE
        assert CONCISE_MODE_GUIDANCE == CANONICAL_CONCISE_MODE_GUIDANCE
        assert COMPLETION_GUIDANCE == CANONICAL_COMPLETION_GUIDANCE
        assert GROUNDING_RULES_EXTENDED == CANONICAL_GROUNDING_RULES_EXTENDED
        assert LARGE_FILE_PAGINATION_GUIDANCE == CANONICAL_LARGE_FILE_PAGINATION_GUIDANCE
        assert ASI_TOOL_EFFECTIVENESS_GUIDANCE == CANONICAL_ASI_TOOL_EFFECTIVENESS_GUIDANCE

    def test_is_cloud_provider_returns_true_for_cloud(self):
        """Test cloud provider detection."""
        for provider in ["anthropic", "openai", "google", "xai"]:
            builder = SystemPromptBuilder(provider_name=provider, model="test")
            assert builder.is_cloud_provider()
            assert not builder.is_local_provider()

    def test_is_local_provider_returns_true_for_local(self):
        """Test local provider detection."""
        for provider in ["ollama", "lmstudio", "vllm"]:
            builder = SystemPromptBuilder(provider_name=provider, model="test")
            assert builder.is_local_provider()
            assert not builder.is_cloud_provider()

    def test_is_cloud_provider_case_insensitive(self):
        """Test provider detection is case insensitive."""
        builder = SystemPromptBuilder(provider_name="ANTHROPIC", model="test")
        assert builder.is_cloud_provider()

    def test_has_native_tool_support_for_qwen(self):
        """Test native tool support detection for Qwen models."""
        builder = SystemPromptBuilder(provider_name="ollama", model="qwen2.5-coder:7b")
        assert builder.has_native_tool_support()

        builder = SystemPromptBuilder(provider_name="ollama", model="qwen3:30b")
        assert builder.has_native_tool_support()

    def test_has_native_tool_support_for_llama(self):
        """Test native tool support detection for Llama models."""
        builder = SystemPromptBuilder(provider_name="ollama", model="llama3.1:8b")
        assert builder.has_native_tool_support()

        builder = SystemPromptBuilder(provider_name="ollama", model="llama-3.2:3b")
        assert builder.has_native_tool_support()

    def test_has_native_tool_support_for_mistral(self):
        """Test native tool support detection for Mistral models."""
        builder = SystemPromptBuilder(provider_name="ollama", model="mistral:7b")
        assert builder.has_native_tool_support()

        builder = SystemPromptBuilder(provider_name="ollama", model="mixtral:8x7b")
        assert builder.has_native_tool_support()

    def test_no_native_tool_support_for_other_models(self):
        """Test that other models are not marked as having native support."""
        builder = SystemPromptBuilder(provider_name="ollama", model="codellama:7b")
        assert not builder.has_native_tool_support()

        builder = SystemPromptBuilder(provider_name="ollama", model="phi:2b")
        assert not builder.has_native_tool_support()

    def test_build_returns_string(self):
        """Test that build() returns a string."""
        builder = SystemPromptBuilder(provider_name="anthropic", model="claude-3")
        result = builder.build()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_build_document_renders_same_prompt(self):
        """Canonical document render should match legacy build output."""
        builder = SystemPromptBuilder(provider_name="anthropic", model="claude-3")

        document = builder.build_document()

        assert document.render() == builder.build()

    def test_build_cloud_prompt_mentions_tools(self):
        """Test that cloud prompt mentions tool usage."""
        builder = SystemPromptBuilder(provider_name="anthropic", model="claude-3")
        result = builder._build_cloud_prompt()
        assert "tool" in result.lower()
        assert "read" in result or "ls" in result

    def test_build_cloud_prompt_mentions_graph_call_modes(self):
        """Test that generic cloud prompts mention graph traversal modes."""
        builder = SystemPromptBuilder(provider_name="anthropic", model="claude-3")
        result = builder._build_cloud_prompt()
        assert "graph(mode='callers'|'callees'|'trace')" in result

    def test_build_with_adapter_uses_hints(self):
        """Test that adapter hints are used when available."""
        mock_adapter = MagicMock()
        mock_adapter.get_system_prompt_hints.return_value = "Custom adapter hints"
        mock_adapter.get_capabilities.return_value = None

        builder = SystemPromptBuilder(
            provider_name="test",
            model="test",
            tool_adapter=mock_adapter,
        )
        result = builder._build_with_adapter()
        assert "Custom adapter hints" in result

    def test_build_ollama_prompt_native_support(self):
        """Test Ollama prompt for models with native support."""
        builder = SystemPromptBuilder(provider_name="ollama", model="qwen2.5:7b")
        result = builder._build_ollama_prompt()
        assert "tool calling capability" in result.lower()
        assert "JSON" in result or "json" in result

    def test_build_ollama_prompt_no_native_support(self):
        """Test Ollama prompt for models without native support."""
        builder = SystemPromptBuilder(provider_name="ollama", model="codellama:7b")
        result = builder._build_ollama_prompt()
        assert "EXACTLY" in result or "ONE AT A TIME" in result

    def test_build_ollama_prompt_qwen3_includes_thinking(self):
        """Test Qwen3 prompt includes thinking mode guidance."""
        builder = SystemPromptBuilder(provider_name="ollama", model="qwen3:30b")
        result = builder._build_ollama_prompt()
        assert "QWEN3" in result or "/no_think" in result

    def test_provider_tool_hint_block_uses_ollama_adapter(self):
        """Local-provider hint helper should reuse Ollama adapter guidance."""
        builder = SystemPromptBuilder(provider_name="ollama", model="qwen3:30b")

        hints = builder._get_provider_tool_hint_block("ollama")

        assert "TOOL USAGE:" in hints
        assert "QWEN3 MODE:" in hints

    def test_provider_tool_hint_block_uses_lmstudio_adapter(self):
        """Local-provider hint helper should reuse LMStudio adapter guidance."""
        builder = SystemPromptBuilder(provider_name="lmstudio", model="codellama:7b")

        hints = builder._get_provider_tool_hint_block("lmstudio")

        assert "CRITICAL RULES:" in hints
        assert "After reading 2-3 files" in hints

    def test_provider_tool_hint_block_uses_vllm_adapter(self):
        """Local-provider hint helper should reuse vLLM adapter guidance."""
        builder = SystemPromptBuilder(provider_name="vllm", model="mistral")

        hints = builder._get_provider_tool_hint_block("vllm")

        assert "TOOL USAGE:" in hints
        assert "Provide answers in plain, readable text." in hints

    def test_build_lmstudio_prompt_native_support(self):
        """Test LMStudio prompt for models with native support."""
        builder = SystemPromptBuilder(provider_name="lmstudio", model="qwen2.5-coder:7b")
        result = builder._build_lmstudio_prompt()
        # Native support models get the expanded capabilities prompt
        assert "CAPABILITIES:" in result
        assert "Code generation" in result

    def test_build_vllm_prompt(self):
        """Test vLLM prompt."""
        builder = SystemPromptBuilder(provider_name="vllm", model="mistral")
        result = builder._build_vllm_prompt()
        assert "OpenAI-compatible" in result

    def test_build_vllm_prompt_includes_provider_hint_block(self):
        """vLLM prompt builder should consume adapter-backed hint blocks."""
        builder = SystemPromptBuilder(provider_name="vllm", model="mistral")

        with patch.object(
            builder, "_get_provider_tool_hint_block", return_value="SENTINEL VLLM HINTS"
        ):
            result = builder._build_vllm_prompt()

        assert "SENTINEL VLLM HINTS" in result

    def test_build_google_prompt_mentions_graph_call_modes(self):
        """Test Google prompts mention graph traversal modes."""
        builder = SystemPromptBuilder(provider_name="google", model="gemini-2.5-pro")
        result = builder._build_google_prompt()
        assert "graph(mode='callers'|'callees'|'trace')" in result

    def test_build_deepseek_prompt_mentions_graph_call_modes(self):
        """Test DeepSeek prompts mention graph traversal modes."""
        builder = SystemPromptBuilder(provider_name="deepseek", model="deepseek-coder")
        result = builder._build_deepseek_prompt()
        assert "graph(mode='callers'|'callees'|'trace')" in result

    def test_build_xai_prompt_mentions_graph_call_modes(self):
        """Test xAI prompts mention graph traversal modes."""
        builder = SystemPromptBuilder(provider_name="xai", model="grok-4")
        result = builder._build_xai_prompt()
        assert "graph(mode='callers'|'callees'|'trace')" in result

    def test_build_default_prompt(self):
        """Test default prompt for unknown providers."""
        builder = SystemPromptBuilder(provider_name="unknown", model="test")
        result = builder._build_default_prompt()
        assert "code analyst" in result.lower()
        assert "tool" in result.lower()

    def test_dynamic_tool_guidance_text_includes_goals_and_intent_rationale(self):
        """Dynamic tool hints should expose the current plan rationale without replanning."""
        builder = SystemPromptBuilder(provider_name="anthropic", model="claude-3")

        result = builder.get_dynamic_tool_guidance_text(
            ["workflow", "web_search"],
            goals=["inspect repo changes", "verify docs"],
            current_intent="read_only",
            selection_source="planned_tools",
            tool_rationale={
                "web_search": "Check upstream docs for API changes",
                "workflow": "Coordinate multi-step execution",
            },
        )

        assert "DYNAMIC TOOL HINTS" in result
        assert "web_search, workflow" in result
        assert "web_search (Check upstream docs for API changes)" in result
        assert "workflow (Coordinate multi-step execution)" in result
        assert "Current plan focus: inspect repo changes; verify docs." in result
        assert "Current intent guard: read only." in result
        assert "planned tool sequence" in result


class TestBuildSystemPromptFunction:
    """Tests for the build_system_prompt convenience function."""

    def test_convenience_function_returns_string(self):
        """Test that the convenience function works."""
        result = build_system_prompt(
            provider_name="anthropic",
            model="claude-3",
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_convenience_function_with_adapter(self):
        """Test convenience function with adapter."""
        mock_adapter = MagicMock()
        mock_adapter.get_system_prompt_hints.return_value = "Hints from adapter"
        mock_adapter.get_capabilities.return_value = None

        result = build_system_prompt(
            provider_name="test",
            model="test",
            tool_adapter=mock_adapter,
        )
        assert "Hints from adapter" in result


class TestProviderConstants:
    """Tests for provider constants."""

    def test_cloud_providers_set(self):
        """Test CLOUD_PROVIDERS constant."""
        assert "anthropic" in CLOUD_PROVIDERS
        assert "openai" in CLOUD_PROVIDERS
        assert "google" in CLOUD_PROVIDERS
        assert "xai" in CLOUD_PROVIDERS

    def test_local_providers_set(self):
        """Test LOCAL_PROVIDERS constant."""
        assert "ollama" in LOCAL_PROVIDERS
        assert "lmstudio" in LOCAL_PROVIDERS
        assert "vllm" in LOCAL_PROVIDERS


class TestSystemPromptBuilderEdgeCases:
    """Edge case tests for SystemPromptBuilder."""

    def test_build_for_vllm_provider(self):
        """Test _build_for_provider for vllm (covers line 157-158)."""
        builder = SystemPromptBuilder(provider_name="vllm", model="test")
        result = builder._build_for_provider()
        assert "OpenAI-compatible" in result

    def test_build_for_lmstudio_provider(self):
        """Test _build_for_provider for lmstudio (covers lines 161-162)."""
        builder = SystemPromptBuilder(provider_name="lmstudio", model="qwen2.5:7b")
        result = builder._build_for_provider()
        assert isinstance(result, str)

    def test_build_for_ollama_provider(self):
        """Test _build_for_provider for ollama (covers lines 165-166)."""
        builder = SystemPromptBuilder(provider_name="ollama", model="llama3.1:8b")
        result = builder._build_for_provider()
        assert isinstance(result, str)

    def test_build_for_unknown_provider(self):
        """Test _build_for_provider for unknown provider (covers lines 168-169)."""
        builder = SystemPromptBuilder(provider_name="totally_unknown", model="test")
        result = builder._build_for_provider()
        assert "code analyst" in result.lower()

    def test_build_lmstudio_prompt_no_native_support(self):
        """Test LMStudio prompt for models without native support (covers line 224)."""
        builder = SystemPromptBuilder(provider_name="lmstudio", model="codellama:7b")
        result = builder._build_lmstudio_prompt()
        assert "ONE AT A TIME" in result
        assert "CRITICAL RULES" in result

    def test_build_lmstudio_prompt_uses_provider_hint_block_when_available(self):
        """LMStudio prompt builder should consume adapter-backed hint blocks."""
        builder = SystemPromptBuilder(provider_name="lmstudio", model="codellama:7b")

        with patch.object(
            builder,
            "_get_provider_tool_hint_block",
            return_value="SENTINEL LMSTUDIO HINTS",
        ):
            result = builder._build_lmstudio_prompt()

        assert "SENTINEL LMSTUDIO HINTS" in result

    def test_build_ollama_prompt_uses_provider_hint_block_when_available(self):
        """Ollama prompt builder should consume adapter-backed hint blocks."""
        builder = SystemPromptBuilder(provider_name="ollama", model="codellama:7b")

        with patch.object(
            builder,
            "_get_provider_tool_hint_block",
            return_value="SENTINEL OLLAMA HINTS",
        ):
            result = builder._build_ollama_prompt()

        assert "SENTINEL OLLAMA HINTS" in result

    def test_build_with_adapter_no_hints_no_caps(self):
        """Test _build_with_adapter includes base_prompt and grounding rules when no hints/caps."""
        mock_adapter = MagicMock()
        mock_adapter.get_system_prompt_hints.return_value = None
        mock_adapter.get_capabilities.return_value = None

        builder = SystemPromptBuilder(
            provider_name="test",
            model="test",
            tool_adapter=mock_adapter,
        )
        result = builder._build_with_adapter()
        # Should include base prompt and grounding rules
        assert "expert coding assistant" in result.lower()
        assert "GROUNDING" in result

    def test_build_deepseek_prompt(self):
        """Test DeepSeek prompt includes anti-repetition and grounding rules."""
        builder = SystemPromptBuilder(provider_name="deepseek", model="deepseek-coder")
        result = builder._build_deepseek_prompt()
        # Should include anti-repetition rules
        assert "NEVER read the same file twice" in result
        assert "NEVER call the same tool with identical arguments" in result
        # Should include grounding rules
        assert "GROUNDING" in result
        # Should include tool efficiency guidance
        assert "TOOL EFFICIENCY" in result or "ls" in result

    def test_build_xai_prompt(self):
        """Test xAI/Grok prompt includes task structure and grounding."""
        builder = SystemPromptBuilder(provider_name="xai", model="grok-beta")
        result = builder._build_xai_prompt()
        # Should include effective tool usage guidance
        assert "EFFECTIVE TOOL USAGE" in result or "tool" in result.lower()
        # Should include task approach guidance
        assert "TASK APPROACH" in result or "analysis" in result.lower()
        # Should include grounding rules
        assert "GROUNDING" in result

    def test_build_cloud_prompt_delegates_to_deepseek(self):
        """Test _build_cloud_prompt uses DeepSeek-specific prompt."""
        builder = SystemPromptBuilder(provider_name="deepseek", model="deepseek-coder")
        cloud_result = builder._build_cloud_prompt()
        deepseek_result = builder._build_deepseek_prompt()
        # Both should return the same content
        assert cloud_result == deepseek_result

    def test_build_cloud_prompt_delegates_to_xai(self):
        """Test _build_cloud_prompt uses xAI-specific prompt."""
        builder = SystemPromptBuilder(provider_name="xai", model="grok-beta")
        cloud_result = builder._build_cloud_prompt()
        xai_result = builder._build_xai_prompt()
        # Both should return the same content
        assert cloud_result == xai_result

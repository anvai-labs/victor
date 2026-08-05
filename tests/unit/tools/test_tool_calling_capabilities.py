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

"""Tests for tool calling capability loader and adapters."""

from victor.agent.tool_calling import (
    ModelCapabilityLoader,
    ToolCallingAdapterRegistry,
    get_model_capabilities,
    ToolCallFormat,
)


class TestModelCapabilityLoader:
    """Tests for ModelCapabilityLoader."""

    def test_loader_is_singleton(self):
        """Loader should be a singleton."""
        loader1 = ModelCapabilityLoader()
        loader2 = ModelCapabilityLoader()
        assert loader1 is loader2

    def test_get_provider_names(self):
        """Should return list of configured providers."""
        loader = ModelCapabilityLoader()
        providers = loader.get_provider_names()
        assert "ollama" in providers
        assert "anthropic" in providers
        assert "openai" in providers

    def test_get_model_patterns(self):
        """Should return list of configured model patterns."""
        loader = ModelCapabilityLoader()
        patterns = loader.get_all_model_patterns()
        assert any("llama" in p.lower() for p in patterns)
        assert any("qwen" in p.lower() for p in patterns)

    def test_ollama_provider_defaults(self):
        """Ollama provider should have expected defaults."""
        caps = get_model_capabilities("ollama")
        # Ollama default without specific model - uses provider defaults
        assert caps.json_fallback_parsing is True
        assert caps.xml_fallback_parsing is True

    def test_anthropic_provider_defaults(self):
        """Anthropic provider should have native tool support."""
        caps = get_model_capabilities("anthropic")
        assert caps.native_tool_calls is True
        assert caps.streaming_tool_calls is True
        assert caps.parallel_tool_calls is True

    def test_llama31_model_override(self):
        """Llama 3.1 should have native tool support via model pattern."""
        caps = get_model_capabilities("ollama", "llama3.1:8b")
        assert caps.native_tool_calls is True
        assert caps.requires_strict_prompting is False
        assert caps.recommended_tool_budget == 15

    def test_qwen25_model_override(self):
        """Qwen 2.5 should have native tool support."""
        caps = get_model_capabilities("ollama", "qwen2.5:7b")
        assert caps.native_tool_calls is True
        assert caps.requires_strict_prompting is False

    def test_qwen3_thinking_mode(self):
        """Qwen 3 should have thinking mode enabled."""
        caps = get_model_capabilities("ollama", "qwen3:4b")
        assert caps.thinking_mode is True

    def test_codellama_limited_support(self):
        """CodeLlama should have limited tool support."""
        caps = get_model_capabilities("ollama", "codellama:7b")
        assert caps.native_tool_calls is False
        assert caps.requires_strict_prompting is True
        assert caps.recommended_tool_budget == 8

    def test_vllm_provider_defaults(self):
        """vLLM should have robust tool calling."""
        caps = get_model_capabilities("vllm", "meta-llama/Llama-3.1-8B")
        assert caps.native_tool_calls is True
        assert caps.streaming_tool_calls is True
        assert caps.tool_choice_param is True

    def test_llamacpp_provider_defaults(self):
        """llama.cpp (llama-server) is an OpenAI-compatible native-tool-calling endpoint.

        Like LMStudio/vLLM, llama-server exposes native OpenAI function-calling
        (verified: it emits `finish_reason: tool_calls`), so its provider defaults
        must enable native tool calls rather than inheriting the strict-prompting
        Ollama-style fallback.
        """
        caps = get_model_capabilities("llamacpp")
        assert caps.native_tool_calls is True
        assert caps.streaming_tool_calls is True
        assert caps.tool_choice_param is True
        assert caps.requires_strict_prompting is False

    def test_llamacpp_qwen3_coder_uses_native_tools(self):
        """Regression: qwen3-coder on llama.cpp must use native tool calls, not XML fallback.

        The server emits native OpenAI tool_calls; treating llamacpp as strict-prompted
        (the missing-provider fallback to the global `false` default) drops the tool call
        so the agent narrates instead of acting. Mirrors the lmstudio/vllm entries.
        """
        caps = get_model_capabilities("llamacpp", "qwen3-coder")
        assert caps.native_tool_calls is True
        assert caps.requires_strict_prompting is False

    def test_ollama_timeout_multiplier(self):
        """Ollama should have higher timeout multiplier for slow local models."""
        caps = get_model_capabilities("ollama", "qwen3-coder:30b")
        # Ollama has 2.0x multiplier in provider_defaults
        assert caps.timeout_multiplier == 2.0
        assert caps.exploration_multiplier == 1.5

    def test_lmstudio_timeout_multiplier(self):
        """LMStudio should have timeout multiplier for local inference."""
        caps = get_model_capabilities("lmstudio", "qwen2.5-coder:32b")
        # LMStudio has 1.5x multiplier
        assert caps.timeout_multiplier == 1.5
        assert caps.exploration_multiplier == 1.3

    def test_anthropic_default_timeout(self):
        """Cloud providers should use default timeout multiplier."""
        caps = get_model_capabilities("anthropic", "claude-3-sonnet")
        # Cloud providers use default 1.0x
        assert caps.timeout_multiplier == 1.0

    def test_openai_default_timeout(self):
        """OpenAI should use default timeout multiplier."""
        caps = get_model_capabilities("openai", "gpt-4o")
        assert caps.timeout_multiplier == 1.0


class TestToolCallingAdapterRegistry:
    """Tests for ToolCallingAdapterRegistry."""

    def test_get_ollama_adapter(self):
        """Should return OllamaToolCallingAdapter for ollama."""
        adapter = ToolCallingAdapterRegistry.get_adapter("ollama", "llama3.1:8b")
        assert adapter.provider_name == "ollama"
        caps = adapter.get_capabilities()
        assert caps.native_tool_calls is True

    def test_get_anthropic_adapter(self):
        """Should return AnthropicToolCallingAdapter for anthropic."""
        adapter = ToolCallingAdapterRegistry.get_adapter("anthropic", "claude-3-opus")
        assert adapter.provider_name == "anthropic"
        caps = adapter.get_capabilities()
        assert caps.tool_call_format == ToolCallFormat.ANTHROPIC

    def test_get_openai_adapter(self):
        """Should return OpenAIToolCallingAdapter for openai."""
        adapter = ToolCallingAdapterRegistry.get_adapter("openai", "gpt-4")
        assert adapter.provider_name == "openai"
        caps = adapter.get_capabilities()
        assert caps.tool_call_format == ToolCallFormat.OPENAI

    def test_get_lmstudio_adapter(self):
        """Should return OpenAICompatToolCallingAdapter for lmstudio."""
        adapter = ToolCallingAdapterRegistry.get_adapter("lmstudio", "qwen2.5-coder")
        assert adapter.provider_name == "lmstudio"

    def test_get_vllm_adapter(self):
        """vLLM uses OpenAI adapter by default (OpenAI-compatible format)."""
        adapter = ToolCallingAdapterRegistry.get_adapter("vllm", "mistral-7b")
        # Default: uses OpenAI adapter (not provider-specific)
        assert adapter.provider_name == "openai"

    def test_unknown_provider_fallback(self):
        """Unknown provider should fall back to OpenAI-compatible."""
        adapter = ToolCallingAdapterRegistry.get_adapter("custom_provider", "some-model")
        # Should not raise, returns OpenAI-compatible adapter
        assert adapter is not None

    def test_list_providers(self):
        """Should list registered providers."""
        providers = ToolCallingAdapterRegistry.list_providers()
        assert "ollama" in providers
        assert "anthropic" in providers

    def test_is_registered(self):
        """Should check if provider is registered."""
        assert ToolCallingAdapterRegistry.is_registered("ollama") is True
        assert ToolCallingAdapterRegistry.is_registered("anthropic") is True


class TestAdapterSystemPromptHints:
    """Tests for system prompt hints generation."""

    def test_ollama_native_hints(self):
        """Ollama with native support should have tool usage hints."""
        adapter = ToolCallingAdapterRegistry.get_adapter("ollama", "llama3.1:8b")
        hints = adapter.get_system_prompt_hints()
        assert "TOOL USAGE" in hints
        assert "one at a time" in hints.lower()

    def test_ollama_fallback_hints(self):
        """Ollama without native support should have strict hints."""
        adapter = ToolCallingAdapterRegistry.get_adapter("ollama", "codellama:7b")
        hints = adapter.get_system_prompt_hints()
        assert "CRITICAL" in hints
        assert "ONE AT A TIME" in hints

    def test_qwen3_thinking_hints(self):
        """Qwen3 should include thinking mode hints."""
        adapter = ToolCallingAdapterRegistry.get_adapter("ollama", "qwen3:4b")
        hints = adapter.get_system_prompt_hints()
        assert "QWEN3" in hints or "/no_think" in hints

    def test_anthropic_no_hints(self):
        """Anthropic should have parallel tool calling hint."""
        adapter = ToolCallingAdapterRegistry.get_adapter("anthropic", "claude-3-opus")
        hints = adapter.get_system_prompt_hints()
        assert "MULTIPLE tools in parallel" in hints

    def test_openai_no_hints(self):
        """OpenAI should have parallel tool calling hint."""
        adapter = ToolCallingAdapterRegistry.get_adapter("openai", "gpt-4")
        hints = adapter.get_system_prompt_hints()
        assert "MULTIPLE tools in parallel" in hints

    def test_vllm_has_hints(self):
        """vLLM should have tool calling hints."""
        adapter = ToolCallingAdapterRegistry.get_adapter("vllm", "llama-3.1-8b")
        hints = adapter.get_system_prompt_hints()
        assert "TOOL" in hints


class TestModelCapabilityLoaderEdgeCases:
    """Edge case tests for ModelCapabilityLoader."""

    def test_reload_method(self):
        """Test reload method refreshes config (covers line 80)."""
        loader = ModelCapabilityLoader()
        # Just verify reload doesn't error
        loader.reload()
        # Config should still be valid after reload
        providers = loader.get_provider_names()
        assert len(providers) > 0

    def test_get_capabilities_none_config(self):
        """Test get_capabilities when _config is None (covers line 105)."""
        from unittest.mock import patch

        loader = ModelCapabilityLoader()
        # Force _config to None to test the reload path
        with patch.object(loader, "_config", None):
            with patch.object(loader, "_load_config") as mock_load:
                mock_load.side_effect = lambda: setattr(
                    loader, "_config", {"defaults": {}, "providers": {}, "models": {}}
                )
                caps = loader.get_capabilities("test_provider")
                mock_load.assert_called_once()
                assert caps is not None

    def test_get_capabilities_with_format_hint(self):
        """Test get_capabilities accepts format_hint parameter."""
        loader = ModelCapabilityLoader()
        caps = loader.get_capabilities("ollama", "test", format_hint=ToolCallFormat.OPENAI)
        assert caps is not None

    def test_config_file_not_found(self):
        """Test handling of missing config file (covers lines 66-68)."""
        from unittest.mock import patch
        from pathlib import Path

        loader = ModelCapabilityLoader()
        with patch.object(Path, "exists", return_value=False):
            # Force reload to test missing file path
            loader._config = None
            loader._load_config()
            # Should have default empty config
            assert loader._config is not None
            assert loader._config.get("defaults") == {}

    def test_config_load_exception(self):
        """Test handling of exception during config load (covers lines 74-76)."""
        from unittest.mock import patch, mock_open

        loader = ModelCapabilityLoader()
        # Reset singleton state for this test
        original_config = loader._config

        try:
            with patch("builtins.open", mock_open()) as m:
                m.return_value.read.side_effect = Exception("Read error")
                loader._config = None
                loader._load_config()
                # Should have default empty config after error
                assert loader._config is not None
        finally:
            # Restore original config
            loader._config = original_config

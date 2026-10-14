"""Tests for provider-aware tiered tool selection strategy.

Verifies that:
- BaseProvider.supports_prompt_caching() defaults to False
- AnthropicProvider overrides to True
- _cache_optimization_enabled derives from provider capability
- Q&A detection heuristic is accurate
- Streaming pipeline skips tools for Q&A tasks
"""

from unittest.mock import MagicMock, patch

import pytest

from victor.agent.services.turn_execution_runtime import TurnExecutor

# =====================================================================
# Step 1-2: Provider capability detection
# =====================================================================


class TestSupportsPromptCaching:
    """Test supports_prompt_caching() on BaseProvider and subclasses."""

    def test_base_provider_defaults_to_false(self):
        from victor.providers.base import BaseProvider

        # BaseProvider is abstract; test via a mock that inherits the default method
        mock_provider = MagicMock(spec=BaseProvider)
        # Call the real method via unbound reference
        assert BaseProvider.supports_prompt_caching(mock_provider) is False

    def test_anthropic_provider_returns_true(self):
        from victor.providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider.__new__(AnthropicProvider)
        assert provider.supports_prompt_caching() is True

    @pytest.mark.parametrize(
        "provider_module,provider_class",
        [
            ("victor.providers.openai_provider", "OpenAIProvider"),
            ("victor.providers.google_provider", "GoogleProvider"),
            ("victor.providers.deepseek_provider", "DeepSeekProvider"),
            ("victor.providers.xai_provider", "XAIProvider"),
            ("victor.providers.groq_provider", "GroqProvider"),
            ("victor.providers.bedrock_provider", "BedrockProvider"),
            ("victor.providers.vertex_provider", "VertexAIProvider"),
            ("victor.providers.azure_openai_provider", "AzureOpenAIProvider"),
            ("victor.providers.openrouter_provider", "OpenRouterProvider"),
            # Cloud inference with auto-caching
            ("victor.providers.fireworks_provider", "FireworksProvider"),
            ("victor.providers.together_provider", "TogetherProvider"),
            ("victor.providers.cerebras_provider", "CerebrasProvider"),
        ],
    )
    def test_cloud_caching_providers_return_true(self, provider_module, provider_class):
        """Cloud providers with API-level cached token discounts return True."""
        import importlib

        mod = importlib.import_module(provider_module)
        cls = getattr(mod, provider_class)
        # Query on an uninitialized instance, the way discovery does: the
        # OpenAI-compat shells derive capability facts from validated config via
        # ``getattr(self, "_spec", None) or provider_spec()``, which must be
        # reachable before __init__ (no credentials/HTTP). A MagicMock ``self``
        # defeats that fallback by fabricating a truthy ``_spec``, so build the
        # instance without __init__ (matches test_anthropic_provider_returns_true).
        instance = cls.__new__(cls)
        assert instance.supports_prompt_caching() is True

    @pytest.mark.parametrize(
        "provider_module,provider_class",
        [
            ("victor.providers.ollama_provider", "OllamaProvider"),
            ("victor.providers.vllm_provider", "VLLMProvider"),
            ("victor.providers.lmstudio_provider", "LMStudioProvider"),
            ("victor.providers.llamacpp_provider", "LlamaCppProvider"),
            ("victor.providers.mlx_provider", "MLXProvider"),
        ],
    )
    def test_local_providers_return_false(self, provider_module, provider_class):
        """Local inference providers return False (KV cache != API prompt caching)."""
        import importlib

        mod = importlib.import_module(provider_module)
        cls = getattr(mod, provider_class)
        # Uninitialized instance (see test_cloud_caching_providers_return_true).
        instance = cls.__new__(cls)
        assert instance.supports_prompt_caching() is False

    @pytest.mark.parametrize(
        "provider_module,provider_class",
        [
            # Local providers have KV prefix caching
            ("victor.providers.ollama_provider", "OllamaProvider"),
            ("victor.providers.vllm_provider", "VLLMProvider"),
            ("victor.providers.lmstudio_provider", "LMStudioProvider"),
            ("victor.providers.llamacpp_provider", "LlamaCppProvider"),
            ("victor.providers.mlx_provider", "MLXProvider"),
            # Cloud providers also have KV prefix caching
            ("victor.providers.anthropic_provider", "AnthropicProvider"),
            ("victor.providers.openai_provider", "OpenAIProvider"),
            ("victor.providers.google_provider", "GoogleProvider"),
        ],
    )
    def test_kv_prefix_caching_returns_true(self, provider_module, provider_class):
        """All providers with KV prefix caching return True."""
        import importlib

        mod = importlib.import_module(provider_module)
        cls = getattr(mod, provider_class)
        # Uninitialized instance (see test_cloud_caching_providers_return_true):
        # avoids the MagicMock-self anti-pattern that defeats config fallbacks.
        instance = cls.__new__(cls)
        assert instance.supports_kv_prefix_caching() is True

    def test_base_provider_kv_prefix_defaults_false(self):
        from victor.providers.base import BaseProvider

        mock_provider = MagicMock(spec=BaseProvider)
        assert BaseProvider.supports_kv_prefix_caching(mock_provider) is False

    def test_local_provider_has_kv_but_no_api_caching(self):
        """Local providers: KV=True, API=False (the two concepts are independent)."""
        from victor.providers.ollama_provider import OllamaProvider

        provider = OllamaProvider.__new__(OllamaProvider)
        assert provider.supports_prompt_caching() is False
        assert provider.supports_kv_prefix_caching() is True

    def test_cloud_provider_has_both(self):
        """Cloud providers: KV=True, API=True."""
        from victor.providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider.__new__(AnthropicProvider)
        assert provider.supports_prompt_caching() is True
        assert provider.supports_kv_prefix_caching() is True


# =====================================================================
# Step 3: _cache_optimization_enabled wiring
# =====================================================================


class TestCacheOptimizationWiring:
    """Test that _cache_optimization_enabled derives from provider capability."""

    def _make_orch_mock(self, provider_caches: bool, setting_enabled: bool = True):
        """Create a mock orchestrator with the real _cache_optimization_enabled property.

        The property delegates to _get_prompt_builder_runtime().is_cache_optimization_enabled(),
        so we replace _get_prompt_builder_runtime on the instance with a plain MagicMock
        (bypassing the spec constraint) whose return value is pre-configured.
        """
        from victor.agent.orchestrator import AgentOrchestrator

        orch = MagicMock(spec=AgentOrchestrator)
        # Set up provider
        orch.provider = MagicMock()
        orch.provider.supports_prompt_caching.return_value = provider_caches
        # Set up settings
        orch.settings = MagicMock()
        orch.settings.context = MagicMock()
        orch.settings.context.cache_optimization_enabled = setting_enabled
        # Replace _get_prompt_builder_runtime on the instance (not the spec-constrained mock)
        # so the real property can call it and get a controlled is_cache_optimization_enabled().
        mock_runtime = MagicMock()
        mock_runtime.is_cache_optimization_enabled.return_value = (
            provider_caches and setting_enabled
        )
        orch._get_prompt_builder_runtime = MagicMock(return_value=mock_runtime)
        # Wire up the real property so the delegation path is exercised
        type(orch)._cache_optimization_enabled = AgentOrchestrator._cache_optimization_enabled
        return orch

    def test_disabled_for_non_caching_provider(self):
        orch = self._make_orch_mock(provider_caches=False)
        assert orch._cache_optimization_enabled is False

    def test_enabled_for_caching_provider(self):
        orch = self._make_orch_mock(provider_caches=True)
        assert orch._cache_optimization_enabled is True

    def test_setting_override_disables(self):
        orch = self._make_orch_mock(provider_caches=True, setting_enabled=False)
        assert orch._cache_optimization_enabled is False

    def test_no_provider_defaults_false(self):
        from victor.agent.orchestrator import AgentOrchestrator

        orch = MagicMock(spec=AgentOrchestrator)
        orch.provider = None
        orch.settings = None
        mock_runtime = MagicMock()
        mock_runtime.is_cache_optimization_enabled.return_value = False
        orch._get_prompt_builder_runtime = MagicMock(return_value=mock_runtime)
        type(orch)._cache_optimization_enabled = AgentOrchestrator._cache_optimization_enabled
        assert orch._cache_optimization_enabled is False


# =====================================================================
# Step 7: Q&A detection accuracy
# =====================================================================


class TestQADetection:
    """Test _is_question_only heuristic accuracy."""

    @pytest.mark.parametrize(
        "msg",
        [
            "What is 2 + 2?",
            "How does Python handle memory?",
            "Explain decorators in Python",
            "What is the difference between list and tuple?",
            "Tell me about async/await",
            "Reply with just the number 4.",
            "What are the benefits of Rust?",
            "How can I learn machine learning?",
            "Why is the sky blue?",
            "Describe the observer pattern",
            "Who is Guido van Rossum?",
            "Define polymorphism",
            "Say hello",
            "What harmless marker did I ask you to remember? "
            "Reply with only the marker; do not use tools.",
        ],
    )
    def test_qa_detected(self, msg):
        assert TurnExecutor._is_question_only(msg) is True

    @pytest.mark.parametrize(
        "msg",
        [
            "Fix the bug in main.py",
            "Create a new test file",
            "Refactor the database module",
            "Add error handling to the login function",
            "Write a function to sort an array",
            "Run the tests and fix failures",
            "Update the README",
            "Debug the authentication issue",
            "Delete the old migration files",
            "Install the missing dependency",
            "Build a REST API endpoint",
            "Deploy to staging",
            "Configure the CI pipeline",
            "Look at the code in utils.py and tell me whats wrong",
            "Implement a binary search tree",
        ],
    )
    def test_action_not_qa(self, msg):
        assert TurnExecutor._is_question_only(msg) is False

    def test_empty_string_not_qa(self):
        assert TurnExecutor._is_question_only("") is False

    def test_long_question_with_action_not_qa(self):
        msg = "Can you fix the broken authentication in the login page?"
        assert TurnExecutor._is_question_only(msg) is False

    def test_codebase_question_not_qa(self):
        msg = "What does the auth module do?"
        assert TurnExecutor._is_question_only(msg) is False


# =====================================================================
# Streaming pipeline Q&A bypass
# =====================================================================


class TestStreamingPipelineQABypass:
    """Test that streaming pipeline skips tools for Q&A tasks."""

    def test_qa_context_flag_default_false(self):
        from victor.agent.streaming.context import StreamingChatContext

        ctx = StreamingChatContext(user_message="test")
        assert ctx.is_qa_task is False

    def test_qa_context_flag_can_be_set(self):
        from victor.agent.streaming.context import StreamingChatContext

        ctx = StreamingChatContext(user_message="What is 2+2?", is_qa_task=True)
        assert ctx.is_qa_task is True

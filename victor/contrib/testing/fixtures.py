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

"""Test fixtures for Victor verticals.

This module provides mixin classes with test fixtures for mocking
providers and creating test assistants.
"""

from __future__ import annotations

from typing import Any, Optional
from unittest import mock


class MockProviderMixin:
    """Mixin providing mock provider fixtures for testing.

    Provides utilities for creating mock LLM providers that return
    predefined responses, useful for testing vertical logic without
    making actual API calls.

    Usage:
        class TestMyVertical(VerticalTestCase, MockProviderMixin):
            def test_with_mock_provider(self):
                provider = self.create_mock_provider("Test response")
                # Use provider in tests
    """

    def create_mock_provider(
        self,
        response_text: str = "Test response",
        model: str = "test-model",
    ) -> mock.MagicMock:
        """Create a mock provider that returns a fixed response.

        Args:
            response_text: Text to return from chat completions
            model: Model name to report

        Returns:
            Mock provider object
        """
        provider = mock.MagicMock()

        # Mock chat response
        mock_response = mock.MagicMock()
        mock_response.content = response_text
        mock_response.text = response_text
        provider.chat.return_value = mock_response
        provider.complete.return_value = response_text

        # Mock streaming
        async def mock_stream(*args, **kwargs):
            yield response_text

        provider.stream = mock_stream

        # Mock model info
        provider.model = model
        provider.provider_name = "test"

        return provider

    def create_mock_provider_with_sequence(
        self,
        responses: list[str],
        model: str = "test-model",
    ) -> mock.MagicMock:
        """Create a mock provider that returns a sequence of responses.

        Args:
            responses: List of responses to return in order
            model: Model name to report

        Returns:
            Mock provider object
        """
        provider = mock.MagicMock()

        # Create mock responses for each item in sequence
        mock_responses = []
        for response_text in responses:
            mock_response = mock.MagicMock()
            mock_response.content = response_text
            mock_response.text = response_text
            mock_responses.append(mock_response)

        provider.chat.side_effect = mock_responses
        provider.complete.side_effect = responses

        provider.model = model
        provider.provider_name = "test"

        return provider

    def create_mock_embedding_provider(
        self,
        embedding_dim: int = 1536,
    ) -> mock.MagicMock:
        """Create a mock embedding provider.

        Args:
            embedding_dim: Dimension of embeddings to return

        Returns:
            Mock embedding provider
        """
        import random

        provider = mock.MagicMock()

        def mock_embed(text: str) -> list[float]:
            return [random.random() for _ in range(embedding_dim)]

        provider.embed = mock_embed
        provider.get_embedding_dim.return_value = embedding_dim
        provider.provider_name = "test-embeddings"

        return provider


class TestAssistantMixin:
    """Mixin providing test assistant creation utilities.

    Provides utilities for creating vertical assistant instances
    with test configuration for testing.

    Usage:
        class TestMyVertical(VerticalTestCase, TestAssistantMixin):
            def test_assistant(self):
                assistant = self.create_test_assistant(
                    provider=self.create_mock_provider(),
                )
    """

    def create_test_assistant(
        self,
        provider: Optional[Any] = None,
        vertical_name: str = "test",
        **kwargs,
    ) -> Any:
        """Create a test assistant instance.

        Args:
            provider: Provider to use (creates mock if None)
            vertical_name: Name of the vertical
            **kwargs: Additional assistant configuration

        Returns:
            Assistant instance

        Note:
            This is a placeholder implementation. Verticals should
            override this method to create their specific assistant type.
        """
        if provider is None:
            if hasattr(self, "create_mock_provider"):
                provider = self.create_mock_provider()
            else:
                provider = mock.MagicMock()

        # Return a mock assistant by default
        assistant = mock.MagicMock()
        assistant.provider = provider
        assistant.vertical_name = vertical_name

        return assistant

    def create_test_config(
        self,
        tool_budget: int = 10,
        max_iterations: int = 20,
        temperature: float = 0.7,
        **kwargs,
    ) -> dict:
        """Create a test configuration dictionary.

        Args:
            tool_budget: Tool call budget
            max_iterations: Maximum iterations
            temperature: LLM temperature
            **kwargs: Additional configuration

        Returns:
            Configuration dictionary
        """
        config = {
            "tool_budget": tool_budget,
            "max_iterations": max_iterations,
            "temperature": temperature,
        }
        config.update(kwargs)
        return config


class MockToolMixin:
    """Mixin providing mock tool fixtures for testing.

    Provides utilities for creating mock tools that return
    predefined results.

    Usage:
        class TestMyVertical(VerticalTestCase, MockToolMixin):
            def test_with_mock_tool(self):
                tool = self.create_mock_tool("tool_name", "result")
    """

    def create_mock_tool(
        self,
        name: str,
        return_value: Any = None,
        side_effect: Optional[Any] = None,
    ) -> mock.MagicMock:
        """Create a mock tool.

        Args:
            name: Tool name
            return_value: Value to return when tool is called
            side_effect: Side effect to execute (overrides return_value)

        Returns:
            Mock tool object
        """
        tool = mock.MagicMock()
        tool.name = name
        tool.__name__ = name

        if side_effect is not None:
            tool.side_effect = side_effect
        else:
            tool.return_value = return_value

        return tool

    def create_mock_tool_registry(self, tools: dict[str, Any]) -> mock.MagicMock:
        """Create a mock tool registry.

        Args:
            tools: Dict mapping tool names to tools

        Returns:
            Mock tool registry
        """
        registry = mock.MagicMock()

        def get_tool(name: str):
            return tools.get(name)

        registry.get_tool = get_tool
        registry.list_tools.return_value = list(tools.keys())
        registry.has_tool.side_effect = lambda name: name in tools

        return registry


__all__ = [
    "MockProviderMixin",
    "TestAssistantMixin",
    "MockToolMixin",
]

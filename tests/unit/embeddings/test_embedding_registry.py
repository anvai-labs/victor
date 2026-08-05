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

"""Tests for embedding registry module."""

import pytest

from victor.storage.vector_stores.registry import EmbeddingRegistry
from victor.storage.vector_stores.base import BaseEmbeddingProvider, EmbeddingConfig


class MockEmbeddingProvider(BaseEmbeddingProvider):
    """Mock embedding provider for testing."""

    def __init__(self, config: EmbeddingConfig):
        self._config = config

    async def initialize(self) -> None:
        pass

    async def index_documents(self, documents, collection: str = "default"):
        pass

    async def search(self, query: str, k: int = 10, collection: str = "default"):
        return []

    async def delete_collection(self, collection: str = "default") -> None:
        pass


class TestEmbeddingRegistryRegister:
    """Tests for EmbeddingRegistry.register method."""

    def test_register_valid_provider(self):
        """Test registering a valid provider."""
        # Save original state
        original_providers = EmbeddingRegistry._providers.copy()
        try:
            EmbeddingRegistry.register("test_provider", MockEmbeddingProvider)
            assert "test_provider" in EmbeddingRegistry._providers
            assert EmbeddingRegistry._providers["test_provider"] == MockEmbeddingProvider
        finally:
            # Restore original state
            EmbeddingRegistry._providers = original_providers

    def test_register_invalid_provider_raises_type_error(self):
        """Test registering invalid provider raises TypeError (covers line 48)."""

        class NotAProvider:
            pass

        with pytest.raises(TypeError) as excinfo:
            EmbeddingRegistry.register("invalid", NotAProvider)

        assert "must inherit from BaseEmbeddingProvider" in str(excinfo.value)


class TestEmbeddingRegistryGet:
    """Tests for EmbeddingRegistry.get method."""

    def test_get_registered_provider(self):
        """Test getting a registered provider."""
        original_providers = EmbeddingRegistry._providers.copy()
        try:
            EmbeddingRegistry._providers["test_get"] = MockEmbeddingProvider
            result = EmbeddingRegistry.get("test_get")
            assert result == MockEmbeddingProvider
        finally:
            EmbeddingRegistry._providers = original_providers

    def test_get_unknown_provider_raises_key_error(self):
        """Test getting unknown provider raises KeyError (covers lines 66-73)."""
        with pytest.raises(KeyError) as excinfo:
            EmbeddingRegistry.get("nonexistent_provider_xyz")

        assert "Unknown embedding provider" in str(excinfo.value)
        assert "nonexistent_provider_xyz" in str(excinfo.value)


class TestEmbeddingRegistryListProviders:
    """Tests for EmbeddingRegistry.list_providers method."""

    def test_list_providers(self):
        """Test listing providers (covers line 95)."""
        providers = EmbeddingRegistry.list_providers()
        assert isinstance(providers, list)
        # Should include auto-registered providers if dependencies are available
        # At minimum, lancedb is likely registered since it's included


class TestEmbeddingRegistryIsRegistered:
    """Tests for EmbeddingRegistry.is_registered method."""

    def test_is_registered_true(self):
        """Test is_registered returns True for registered provider (covers line 107)."""
        original_providers = EmbeddingRegistry._providers.copy()
        try:
            EmbeddingRegistry._providers["test_check"] = MockEmbeddingProvider
            assert EmbeddingRegistry.is_registered("test_check") is True
        finally:
            EmbeddingRegistry._providers = original_providers

    def test_is_registered_false(self):
        """Test is_registered returns False for unregistered provider."""
        assert EmbeddingRegistry.is_registered("nonexistent_xyz") is False

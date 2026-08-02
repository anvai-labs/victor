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

"""Tests for the model switcher module."""

import pytest

from victor.agent.model_switcher import (
    ModelSwitcher,
    ModelInfo,
    ModelSwitchEvent,
    SwitchReason,
    get_model_switcher,
    reset_model_switcher,
)


@pytest.fixture
def switcher():
    """Create a fresh model switcher for each test."""
    reset_model_switcher()
    return ModelSwitcher()


class TestSwitchReason:
    """Tests for SwitchReason enum."""

    def test_switch_reasons(self):
        """Test all switch reasons exist."""
        assert SwitchReason.USER_REQUEST.value == "user_request"
        assert SwitchReason.PERFORMANCE.value == "performance"
        assert SwitchReason.COST.value == "cost"
        assert SwitchReason.CAPABILITY.value == "capability"
        assert SwitchReason.FALLBACK.value == "fallback"
        assert SwitchReason.LOAD_BALANCING.value == "load_balancing"


class TestModelInfo:
    """Tests for ModelInfo dataclass."""

    def test_create_model_info(self):
        """Test creating ModelInfo."""
        model = ModelInfo(
            provider="openai",
            model_id="gpt-4",
            display_name="GPT-4",
            context_window=8192,
            supports_tools=True,
        )

        assert model.provider == "openai"
        assert model.model_id == "gpt-4"
        assert model.full_id == "openai:gpt-4"
        assert model.context_window == 8192
        assert model.supports_tools is True
        assert model.is_local is False

    def test_local_model_info(self):
        """Test creating local ModelInfo."""
        model = ModelInfo(
            provider="ollama",
            model_id="llama3:8b",
            display_name="Llama 3 8B",
            is_local=True,
        )

        assert model.is_local is True
        assert model.cost_per_1k_input == 0.0


class TestModelSwitchEvent:
    """Tests for ModelSwitchEvent dataclass."""

    def test_create_switch_event(self):
        """Test creating switch event."""
        event = ModelSwitchEvent(
            from_provider="openai",
            from_model="gpt-4",
            to_provider="anthropic",
            to_model="claude-3",
            reason=SwitchReason.USER_REQUEST,
        )

        assert event.from_provider == "openai"
        assert event.to_provider == "anthropic"
        assert event.reason == SwitchReason.USER_REQUEST
        assert event.context_preserved is True
        assert event.timestamp is not None


class TestModelSwitcher:
    """Tests for ModelSwitcher class."""

    def test_initial_state(self, switcher):
        """Test initial state."""
        assert switcher.current_provider is None
        assert switcher.current_model is None
        assert switcher.current_full_id is None

    def test_set_current(self, switcher):
        """Test setting current model."""
        switcher.set_current("anthropic", "claude-3")

        assert switcher.current_provider == "anthropic"
        assert switcher.current_model == "claude-3"
        assert switcher.current_full_id == "anthropic:claude-3"

    def test_switch_model(self, switcher):
        """Test switching models."""
        switcher.set_current("openai", "gpt-4")

        result = switcher.switch("anthropic", "claude-3")

        assert result is True
        assert switcher.current_provider == "anthropic"
        assert switcher.current_model == "claude-3"

    def test_switch_records_history(self, switcher):
        """Test that switches are recorded."""
        switcher.set_current("openai", "gpt-4")
        switcher.switch("anthropic", "claude-3")

        history = switcher.get_switch_history()

        assert len(history) == 1
        assert history[0]["from"] == "openai:gpt-4"
        assert history[0]["to"] == "anthropic:claude-3"

    def test_switch_with_reason(self, switcher):
        """Test switching with different reasons."""
        switcher.set_current("openai", "gpt-4")
        switcher.switch("ollama", "llama3", reason=SwitchReason.COST)

        history = switcher.get_switch_history()
        assert history[0]["reason"] == "cost"

    def test_switch_by_id(self, switcher):
        """Test switching by full ID."""
        switcher.set_current("openai", "gpt-4")

        result = switcher.switch_by_id("anthropic:claude-sonnet-4-20250514")

        assert result is True
        assert switcher.current_provider == "anthropic"


class TestModelRegistry:
    """Tests for model registration."""

    def test_default_models_registered(self, switcher):
        """Test that default models are registered."""
        models = switcher.get_available_models()
        assert len(models) > 0

    def test_register_model(self, switcher):
        """Test registering a custom model."""
        model = ModelInfo(
            provider="custom",
            model_id="custom-model",
            display_name="Custom Model",
        )

        switcher.register_model(model)

        found = switcher.find_model("custom-model")
        assert found is not None
        assert found.provider == "custom"

    def test_find_model_by_display_name(self, switcher):
        """Test finding model by display name."""
        found = switcher.find_model("Claude Sonnet")

        assert found is not None
        assert "claude" in found.model_id.lower() or "sonnet" in found.display_name.lower()

    def test_find_model_not_found(self, switcher):
        """Test finding non-existent model."""
        found = switcher.find_model("nonexistent-model-xyz")
        assert found is None


class TestModelFiltering:
    """Tests for model filtering."""

    def test_filter_by_provider(self, switcher):
        """Test filtering models by provider."""
        models = switcher.get_available_models(provider="anthropic")

        assert len(models) > 0
        assert all(m.provider == "anthropic" for m in models)

    def test_filter_local_only(self, switcher):
        """Test filtering local models only."""
        models = switcher.get_available_models(local_only=True)

        # Should have some local models (Ollama)
        assert len(models) > 0
        assert all(m.is_local for m in models)

    def test_filter_by_capability(self, switcher):
        """Test filtering models by capability."""
        models = switcher.get_available_models(capability="code")

        assert len(models) > 0
        assert all("code" in m.capabilities for m in models)


class TestFallbackChain:
    """Tests for fallback chain functionality."""

    def test_set_fallback_chain(self, switcher):
        """Test setting fallback chain."""
        switcher.set_fallback_chain(
            [
                "anthropic:claude-sonnet-4-20250514",
                "openai:gpt-4-turbo",
                "ollama:llama3.1:8b",
            ]
        )

        switcher.set_current("anthropic", "claude-sonnet-4-20250514")
        fallback = switcher.get_fallback()

        assert fallback == "openai:gpt-4-turbo"

    def test_get_fallback_at_end(self, switcher):
        """Test getting fallback when at end of chain."""
        switcher.set_fallback_chain(
            [
                "anthropic:claude-sonnet-4-20250514",
                "openai:gpt-4-turbo",
            ]
        )

        switcher.set_current("openai", "gpt-4-turbo")
        fallback = switcher.get_fallback()

        assert fallback is None

    def test_switch_to_fallback(self, switcher):
        """Test switching to fallback."""
        switcher.set_fallback_chain(
            [
                "anthropic:claude-sonnet-4-20250514",
                "openai:gpt-4-turbo",
            ]
        )

        switcher.set_current("anthropic", "claude-sonnet-4-20250514")
        result = switcher.switch_to_fallback()

        assert result is True
        assert switcher.current_provider == "openai"
        assert switcher.current_model == "gpt-4-turbo"

        # Check reason is FALLBACK
        history = switcher.get_switch_history()
        assert history[-1]["reason"] == "fallback"


class TestCallbacks:
    """Tests for switch callbacks."""

    def test_register_callback(self, switcher):
        """Test registering and calling callback."""
        events = []

        def callback(event):
            events.append(event)

        switcher.register_callback(callback)
        switcher.set_current("openai", "gpt-4")
        switcher.switch("anthropic", "claude-3")

        assert len(events) == 1
        assert events[0].to_provider == "anthropic"

    def test_multiple_callbacks(self, switcher):
        """Test multiple callbacks."""
        calls1 = []
        calls2 = []

        switcher.register_callback(lambda e: calls1.append(e))
        switcher.register_callback(lambda e: calls2.append(e))

        switcher.set_current("openai", "gpt-4")
        switcher.switch("anthropic", "claude-3")

        assert len(calls1) == 1
        assert len(calls2) == 1


class TestModelSuggestion:
    """Tests for model suggestion functionality."""

    def test_suggest_model_basic(self, switcher):
        """Test basic model suggestion."""
        suggested = switcher.suggest_model()

        assert suggested is not None
        assert suggested.supports_tools is True

    def test_suggest_local_model(self, switcher):
        """Test suggesting local model."""
        suggested = switcher.suggest_model(prefer_local=True)

        assert suggested is not None
        assert suggested.is_local is True

    def test_suggest_with_context_size(self, switcher):
        """Test suggesting model with minimum context size."""
        # Request large context
        suggested = switcher.suggest_model(context_size=100000)

        assert suggested is not None
        assert suggested.context_window >= 100000

    def test_get_model_for_task(self, switcher):
        """Test getting model for specific task."""
        model = switcher.get_model_for_task("code")

        assert model is not None
        assert "code" in model.capabilities


class TestStatus:
    """Tests for status reporting."""

    def test_get_status_initial(self, switcher):
        """Test getting status in initial state."""
        status = switcher.get_status()

        assert status["current_provider"] is None
        assert status["current_model"] is None
        assert status["switch_count"] == 0

    def test_get_status_after_switch(self, switcher):
        """Test getting status after switches."""
        switcher.set_current("openai", "gpt-4")
        switcher.switch("anthropic", "claude-3")

        status = switcher.get_status()

        assert status["current_provider"] == "anthropic"
        assert status["current_model"] == "claude-3"
        assert status["switch_count"] == 1


class TestGlobalSwitcher:
    """Tests for global switcher functions."""

    def test_get_model_switcher_singleton(self):
        """Test singleton behavior."""
        reset_model_switcher()
        switcher1 = get_model_switcher()
        switcher2 = get_model_switcher()

        assert switcher1 is switcher2

    def test_reset_model_switcher(self):
        """Test resetting switcher."""
        reset_model_switcher()
        switcher1 = get_model_switcher()
        reset_model_switcher()
        switcher2 = get_model_switcher()

        assert switcher1 is not switcher2

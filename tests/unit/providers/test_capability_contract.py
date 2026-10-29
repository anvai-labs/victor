"""TDD tests for ProviderCapabilityContract — Wave 4.

Verifies: frozen Pydantic model, factory from_tool_calling, None-safe construction,
and ProviderState contract field population.
"""

from __future__ import annotations

import pytest


class TestProviderCapabilityContractModel:
    def test_contract_importable(self):
        from victor.providers.capability_contract import ProviderCapabilityContract

        contract = ProviderCapabilityContract(
            provider="ollama",
            model="llama3",
            context_window=8192,
        )
        assert contract.provider == "ollama"
        assert contract.model == "llama3"
        assert contract.context_window == 8192

    def test_contract_is_frozen_pydantic_model(self):
        from victor.providers.capability_contract import ProviderCapabilityContract

        contract = ProviderCapabilityContract(
            provider="anthropic", model="claude-3", context_window=200000
        )
        with pytest.raises(Exception):  # ValidationError or TypeError from frozen model
            contract.provider = "openai"  # type: ignore[misc]

    def test_contract_defaults(self):
        from victor.providers.capability_contract import ProviderCapabilityContract

        contract = ProviderCapabilityContract(provider="x", model="y", context_window=4096)
        assert contract.native_tool_calls is False
        assert contract.streaming_tool_calls is False
        assert contract.parallel_tool_calls is False
        assert contract.json_fallback_parsing is False
        assert contract.thinking_mode is False
        assert contract.supports_streaming is False
        assert contract.source == "config"


class TestFromToolCallingFactory:
    def test_from_tool_calling_with_both_sources(self):
        from victor.providers.capability_contract import ProviderCapabilityContract
        from victor.agent.tool_calling.base import ToolCallingCapabilities
        from victor.providers.runtime_capabilities import ProviderRuntimeCapabilities

        caps = ToolCallingCapabilities(
            native_tool_calls=True,
            streaming_tool_calls=True,
            parallel_tool_calls=False,
            json_fallback_parsing=True,
            thinking_mode=False,
        )
        runtime = ProviderRuntimeCapabilities(
            provider="ollama",
            model="llama3",
            context_window=8192,
            supports_tools=True,
            supports_streaming=True,
            source="discovered",
        )

        contract = ProviderCapabilityContract.from_tool_calling(caps=caps, runtime=runtime)

        assert contract.provider == "ollama"
        assert contract.model == "llama3"
        assert contract.context_window == 8192
        assert contract.native_tool_calls is True
        assert contract.streaming_tool_calls is True
        assert contract.json_fallback_parsing is True
        assert contract.supports_streaming is True
        assert contract.source == "discovered"

    def test_from_tool_calling_handles_none_runtime(self):
        from victor.providers.capability_contract import ProviderCapabilityContract
        from victor.agent.tool_calling.base import ToolCallingCapabilities

        caps = ToolCallingCapabilities(native_tool_calls=True)
        contract = ProviderCapabilityContract.from_tool_calling(
            caps=caps,
            runtime=None,
            provider="anthropic",
            model="claude-3",
            context_window=200000,
        )

        assert contract.native_tool_calls is True
        assert contract.provider == "anthropic"
        assert contract.context_window == 200000

    def test_from_tool_calling_handles_none_caps(self):
        from victor.providers.capability_contract import ProviderCapabilityContract
        from victor.providers.runtime_capabilities import ProviderRuntimeCapabilities

        runtime = ProviderRuntimeCapabilities(
            provider="openai",
            model="gpt-4o",
            context_window=128000,
            supports_tools=True,
            supports_streaming=True,
        )
        contract = ProviderCapabilityContract.from_tool_calling(caps=None, runtime=runtime)

        assert contract.provider == "openai"
        assert contract.context_window == 128000
        assert contract.native_tool_calls is False  # caps was None

    def test_from_tool_calling_handles_both_none_with_explicit_kwargs(self):
        from victor.providers.capability_contract import ProviderCapabilityContract

        contract = ProviderCapabilityContract.from_tool_calling(
            caps=None,
            runtime=None,
            provider="unknown",
            model="unknown",
            context_window=4096,
        )
        assert contract.provider == "unknown"
        assert contract.context_window == 4096


class TestProviderStateContractField:
    def test_provider_state_has_contract_field(self):
        from victor.agent.provider_manager import ProviderState

        fields = {f.name for f in getattr(ProviderState, "__dataclass_fields__", {}).values()}
        assert (
            "contract" in fields
        ), "ProviderState must have a 'contract' field (Optional[ProviderCapabilityContract])"

    def test_provider_state_contract_defaults_none(self):
        from unittest.mock import MagicMock
        from victor.agent.provider_manager import ProviderState

        state = ProviderState(
            provider=MagicMock(),
            provider_name="test",
            model="test-model",
        )
        assert state.contract is None

    def test_provider_state_can_be_created_with_contract(self):
        from unittest.mock import MagicMock
        from victor.agent.provider_manager import ProviderState
        from victor.providers.capability_contract import ProviderCapabilityContract

        contract = ProviderCapabilityContract(
            provider="test", model="test-model", context_window=4096
        )
        state = ProviderState(
            provider=MagicMock(),
            provider_name="test",
            model="test-model",
            contract=contract,
        )
        assert state.contract is contract
        assert state.contract.context_window == 4096

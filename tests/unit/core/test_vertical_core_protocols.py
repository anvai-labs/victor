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

"""Tests for vertical provider protocols.

These tests verify that ChainProviderProtocol, PersonaProviderProtocol,
and CapabilityProviderProtocol are properly defined and runtime_checkable.
"""

import pytest
from typing import Dict, Any, Protocol, runtime_checkable


class TestChainProviderProtocol:
    """Tests for ChainProviderProtocol."""

    def test_chain_provider_protocol_is_runtime_checkable(self):
        """ChainProviderProtocol must be @runtime_checkable."""
        from victor.core.verticals.protocols import ChainProviderProtocol

        # Verify isinstance() doesn't raise TypeError (which happens without @runtime_checkable)
        # With @runtime_checkable, isinstance() works; without it, TypeError is raised
        class DummyClass:
            pass

        try:
            # This should NOT raise TypeError if protocol is @runtime_checkable
            isinstance(DummyClass(), ChainProviderProtocol)
        except TypeError:
            pytest.fail("ChainProviderProtocol must be decorated with @runtime_checkable")

    def test_chain_provider_protocol_has_get_chains(self):
        """ChainProviderProtocol must define get_chains method."""
        from victor.core.verticals.protocols import ChainProviderProtocol

        # Verify the method exists in the protocol
        assert hasattr(
            ChainProviderProtocol, "get_chains"
        ), "ChainProviderProtocol must define get_chains method"

    def test_chain_provider_protocol_isinstance_check(self):
        """Verify isinstance() works with ChainProviderProtocol."""
        from victor.core.verticals.protocols import ChainProviderProtocol

        class ValidChainProvider:
            def get_chains(self) -> Dict[str, Any]:
                return {"test_chain": {}}

        class InvalidProvider:
            pass

        valid = ValidChainProvider()
        invalid = InvalidProvider()

        assert isinstance(
            valid, ChainProviderProtocol
        ), "Valid implementation should pass isinstance check"
        assert not isinstance(
            invalid, ChainProviderProtocol
        ), "Invalid implementation should fail isinstance check"


class TestPersonaProviderProtocol:
    """Tests for PersonaProviderProtocol."""

    def test_persona_provider_protocol_is_runtime_checkable(self):
        """PersonaProviderProtocol must be @runtime_checkable."""
        from victor.core.verticals.protocols import PersonaProviderProtocol

        # Verify isinstance() doesn't raise TypeError (which happens without @runtime_checkable)
        # With @runtime_checkable, isinstance() works; without it, TypeError is raised
        class DummyClass:
            pass

        try:
            # This should NOT raise TypeError if protocol is @runtime_checkable
            isinstance(DummyClass(), PersonaProviderProtocol)
        except TypeError:
            pytest.fail("PersonaProviderProtocol must be decorated with @runtime_checkable")

    def test_persona_provider_protocol_has_get_personas(self):
        """PersonaProviderProtocol must define get_personas method."""
        from victor.core.verticals.protocols import PersonaProviderProtocol

        # Verify the method exists in the protocol
        assert hasattr(
            PersonaProviderProtocol, "get_personas"
        ), "PersonaProviderProtocol must define get_personas method"

    def test_persona_provider_protocol_isinstance_check(self):
        """Verify isinstance() works with PersonaProviderProtocol."""
        from victor.core.verticals.protocols import PersonaProviderProtocol

        class ValidPersonaProvider:
            def get_personas(self) -> Dict[str, Any]:
                return {"developer": {"name": "Developer"}}

        class InvalidProvider:
            pass

        valid = ValidPersonaProvider()
        invalid = InvalidProvider()

        assert isinstance(
            valid, PersonaProviderProtocol
        ), "Valid implementation should pass isinstance check"
        assert not isinstance(
            invalid, PersonaProviderProtocol
        ), "Invalid implementation should fail isinstance check"


class TestCapabilityProviderProtocol:
    """Tests for CapabilityProviderProtocol."""

    def test_capability_provider_protocol_is_runtime_checkable(self):
        """CapabilityProviderProtocol must be @runtime_checkable."""
        from victor.core.verticals.protocols import CapabilityProviderProtocol

        # Verify isinstance() doesn't raise TypeError (which happens without @runtime_checkable)
        # With @runtime_checkable, isinstance() works; without it, TypeError is raised
        class DummyClass:
            pass

        try:
            # This should NOT raise TypeError if protocol is @runtime_checkable
            isinstance(DummyClass(), CapabilityProviderProtocol)
        except TypeError:
            pytest.fail("CapabilityProviderProtocol must be decorated with @runtime_checkable")

    def test_capability_provider_protocol_has_get_capabilities(self):
        """CapabilityProviderProtocol must define get_capabilities method."""
        from victor.core.verticals.protocols import CapabilityProviderProtocol

        # Verify the method exists in the protocol
        assert hasattr(
            CapabilityProviderProtocol, "get_capabilities"
        ), "CapabilityProviderProtocol must define get_capabilities method"

    def test_capability_provider_protocol_isinstance_check(self):
        """Verify isinstance() works with CapabilityProviderProtocol."""
        from victor.core.verticals.protocols import CapabilityProviderProtocol

        class ValidCapabilityProvider:
            def get_capabilities(self) -> Dict[str, Any]:
                return {"code_review": True, "refactoring": True}

        class InvalidProvider:
            pass

        valid = ValidCapabilityProvider()
        invalid = InvalidProvider()

        assert isinstance(
            valid, CapabilityProviderProtocol
        ), "Valid implementation should pass isinstance check"
        assert not isinstance(
            invalid, CapabilityProviderProtocol
        ), "Invalid implementation should fail isinstance check"


class TestProtocolsExported:
    """Tests for protocol exports in __all__."""

    def test_chain_provider_protocol_exported(self):
        """ChainProviderProtocol must be exported in __all__."""
        from victor.core.verticals import protocols

        assert (
            "ChainProviderProtocol" in protocols.__all__
        ), "ChainProviderProtocol must be in __all__"

    def test_persona_provider_protocol_exported(self):
        """PersonaProviderProtocol must be exported in __all__."""
        from victor.core.verticals import protocols

        assert (
            "PersonaProviderProtocol" in protocols.__all__
        ), "PersonaProviderProtocol must be in __all__"

    def test_capability_provider_protocol_exported(self):
        """CapabilityProviderProtocol must be exported in __all__."""
        from victor.core.verticals import protocols

        assert (
            "CapabilityProviderProtocol" in protocols.__all__
        ), "CapabilityProviderProtocol must be in __all__"

# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for public observability wiring ports."""

import logging
from unittest.mock import MagicMock, patch

from victor.observability.integration import ObservabilityIntegration


class TestObservabilityIntegrationPorts:
    """Validate that observability wiring uses public orchestrator ports."""

    def test_wire_orchestrator_prefers_set_observability(self):
        """wire_orchestrator should call set_observability when available."""
        integration = ObservabilityIntegration()
        orchestrator = MagicMock()
        orchestrator.conversation_state = MagicMock()

        with patch.object(integration, "wire_state_machine") as mock_wire_state:
            integration.wire_orchestrator(orchestrator)

        mock_wire_state.assert_called_once_with(orchestrator.conversation_state)
        orchestrator.set_observability.assert_called_once_with(integration)

    def test_wire_orchestrator_falls_back_to_observability_property(self):
        """wire_orchestrator should use observability property when setter method is absent."""

        class PropertyOnlyOrchestrator:
            def __init__(self):
                self._observability = None
                self.conversation_state = MagicMock()

            @property
            def observability(self):
                return self._observability

            @observability.setter
            def observability(self, value):
                self._observability = value

        integration = ObservabilityIntegration()
        orchestrator = PropertyOnlyOrchestrator()

        with patch.object(integration, "wire_state_machine") as mock_wire_state:
            integration.wire_orchestrator(orchestrator)

        mock_wire_state.assert_called_once_with(orchestrator.conversation_state)
        assert orchestrator.observability is integration

    def test_wire_orchestrator_warns_without_public_port(self, caplog):
        """wire_orchestrator should warn when no public observability port exists."""

        class NoObservabilityPortOrchestrator:
            def __init__(self):
                self.conversation_state = MagicMock()

        integration = ObservabilityIntegration()
        orchestrator = NoObservabilityPortOrchestrator()

        with patch.object(integration, "wire_state_machine"):
            with caplog.at_level(logging.WARNING):
                integration.wire_orchestrator(orchestrator)

        assert any("public observability setter" in rec.message for rec in caplog.records)

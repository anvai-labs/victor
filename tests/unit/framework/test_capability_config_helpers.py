# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Unit tests for shared capability config helper utilities."""

import pytest

from victor.framework.capability_config_helpers import (
    load_capability_config,
    resolve_capability_config_scope_key,
    resolve_capability_config_service,
    store_capability_config,
    update_capability_config_section,
)
from victor.framework.capability_config_service import CapabilityConfigService


class _StubContainer:
    def __init__(self, service):
        self._service = service

    def get_optional(self, service_type):
        if isinstance(self._service, service_type):
            return self._service
        return self._service


class _ServiceBackedOrchestrator:
    def __init__(self, service):
        self._container = _StubContainer(service)

    def get_service_container(self):
        return self._container


class _ScopedServiceBackedOrchestrator(_ServiceBackedOrchestrator):
    def __init__(self, service, scope_key: str):
        super().__init__(service)
        self._scope_key = scope_key

    def get_capability_config_scope_key(self):
        return self._scope_key


class _FallbackOrchestrator:
    def __init__(self):
        self.existing_config = {"value": 1}


class TestCapabilityConfigHelpers:
    def test_resolve_service_requires_real_type(self):
        service = CapabilityConfigService()
        orchestrator = _ServiceBackedOrchestrator(service)
        assert resolve_capability_config_service(orchestrator) is service

        non_service_orchestrator = _ServiceBackedOrchestrator(object())
        assert resolve_capability_config_service(non_service_orchestrator) is None

    def test_store_and_load_use_service_first(self):
        service = CapabilityConfigService()
        orchestrator = _ServiceBackedOrchestrator(service)

        stored_in_service = store_capability_config(orchestrator, "alpha", {"x": 1})
        loaded = load_capability_config(orchestrator, "alpha", {"x": 0})

        assert stored_in_service is True
        assert loaded == {"x": 1}

    def test_resolve_scope_key_uses_port_when_available(self):
        service = CapabilityConfigService()
        orchestrator = _ScopedServiceBackedOrchestrator(service, "session-123")
        assert resolve_capability_config_scope_key(orchestrator) == "session-123"

    def test_store_and_load_are_scope_isolated(self):
        service = CapabilityConfigService()
        orchestrator_a = _ScopedServiceBackedOrchestrator(service, "session-a")
        orchestrator_b = _ScopedServiceBackedOrchestrator(service, "session-b")

        store_capability_config(orchestrator_a, "alpha", {"x": 1})
        store_capability_config(orchestrator_b, "alpha", {"x": 9})

        assert load_capability_config(orchestrator_a, "alpha", {"x": 0}) == {"x": 1}
        assert load_capability_config(orchestrator_b, "alpha", {"x": 0}) == {"x": 9}

    def test_load_uses_legacy_service_name_when_present(self):
        service = CapabilityConfigService()
        service.set_config("legacy_name", {"enabled": False})
        orchestrator = _ServiceBackedOrchestrator(service)

        loaded = load_capability_config(
            orchestrator,
            "new_name",
            {"enabled": True},
            legacy_service_names=["legacy_name"],
        )
        assert loaded == {"enabled": False}

    def test_store_fallback_respects_existing_attr_policy(self):
        orchestrator = _FallbackOrchestrator()

        stored = store_capability_config(orchestrator, "missing_config", {"a": 1})
        assert stored is False
        assert not hasattr(orchestrator, "missing_config")

        store_capability_config(
            orchestrator,
            "created_config",
            {"a": 2},
            require_existing_attr=False,
        )
        assert orchestrator.created_config == {"a": 2}

    def test_update_section_merges_root_config(self):
        service = CapabilityConfigService()
        orchestrator = _ServiceBackedOrchestrator(service)

        update_capability_config_section(
            orchestrator,
            root_name="rag_config",
            section_name="indexing",
            section_config={"chunk_size": 512},
            root_defaults={"indexing": {}, "retrieval": {}},
        )
        update_capability_config_section(
            orchestrator,
            root_name="rag_config",
            section_name="retrieval",
            section_config={"top_k": 8},
            root_defaults={"indexing": {}, "retrieval": {}},
        )

        assert service.get_config("rag_config") == {
            "indexing": {"chunk_size": 512},
            "retrieval": {"top_k": 8},
        }

    def test_load_private_fallback_blocked_in_strict_mode(self, monkeypatch):
        orchestrator = _FallbackOrchestrator()
        monkeypatch.setenv("VICTOR_STRICT_FRAMEWORK_PRIVATE_FALLBACKS", "1")

        with pytest.raises(RuntimeError):
            load_capability_config(
                orchestrator,
                "private_config",
                {"enabled": True},
                fallback_attr="_private_config",
            )

    def test_store_private_fallback_blocked_in_strict_mode(self, monkeypatch):
        orchestrator = _FallbackOrchestrator()
        monkeypatch.setenv("VICTOR_STRICT_FRAMEWORK_PRIVATE_FALLBACKS", "true")

        with pytest.raises(RuntimeError):
            store_capability_config(
                orchestrator,
                "private_config",
                {"enabled": True},
                fallback_attr="_private_config",
                require_existing_attr=False,
            )

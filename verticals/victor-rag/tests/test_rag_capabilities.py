# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Unit tests for RAG capability config storage behavior."""

from victor.framework.capability_config_service import CapabilityConfigService
from victor_rag.capabilities import (
    configure_indexing,
    configure_retrieval,
    configure_synthesis,
    get_retrieval_config,
)


class _StubContainer:
    def __init__(self, service: CapabilityConfigService | None = None) -> None:
        self._service = service

    def get_optional(self, service_type):
        if self._service is None:
            return None
        if isinstance(self._service, service_type):
            return self._service
        return None


class _ServiceBackedOrchestrator:
    def __init__(self, service: CapabilityConfigService) -> None:
        self._container = _StubContainer(service)

    def get_service_container(self):
        return self._container


class _LegacyOrchestrator:
    pass


class TestRAGCapabilityConfigStorage:
    """Validate RAG capability config storage migration path."""

    def test_configure_sections_store_in_framework_service_without_clobbering(self):
        service = CapabilityConfigService()
        orchestrator = _ServiceBackedOrchestrator(service)

        configure_indexing(orchestrator, chunk_size=1024, chunk_overlap=100)
        configure_retrieval(orchestrator, top_k=8, search_type="semantic")

        rag_config = service.get_config("rag_config")
        assert rag_config["indexing"]["chunk_size"] == 1024
        assert rag_config["indexing"]["chunk_overlap"] == 100
        assert rag_config["retrieval"]["top_k"] == 8
        assert rag_config["retrieval"]["search_type"] == "semantic"

    def test_get_retrieval_reads_framework_service_first(self):
        service = CapabilityConfigService()
        service.set_config(
            "rag_config",
            {
                "retrieval": {
                    "top_k": 12,
                    "similarity_threshold": 0.8,
                    "search_type": "hybrid",
                    "rerank_enabled": False,
                    "max_context_tokens": 3000,
                }
            },
        )
        orchestrator = _ServiceBackedOrchestrator(service)

        assert get_retrieval_config(orchestrator) == {
            "top_k": 12,
            "similarity_threshold": 0.8,
            "search_type": "hybrid",
            "rerank_enabled": False,
            "max_context_tokens": 3000,
        }

    def test_legacy_fallback_initializes_rag_config_attribute(self):
        orchestrator = _LegacyOrchestrator()

        configure_synthesis(orchestrator, citation_style="footnote", include_sources=False)

        assert orchestrator.rag_config["synthesis"] == {
            "citation_style": "footnote",
            "include_sources": False,
            "max_answer_tokens": 2000,
            "temperature": 0.3,
            "require_verification": True,
        }

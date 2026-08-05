# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Unit tests for framework capability config service."""

from victor.framework.capability_config_service import (
    CapabilityConfigMergePolicy,
    CapabilityConfigService,
)


class TestCapabilityConfigService:
    """Validate capability config storage and merge behavior."""

    def test_set_and_get_config(self):
        service = CapabilityConfigService()
        service.set_config("citation_config", {"style": "apa"})

        assert service.has_config("citation_config") is True
        assert service.get_config("citation_config") == {"style": "apa"}

    def test_shallow_merge_updates_existing_dict(self):
        service = CapabilityConfigService()
        service.set_config("rag_config", {"indexing": {"chunk_size": 512}})
        service.set_config(
            "rag_config",
            {"retrieval": {"top_k": 5}},
            merge_policy=CapabilityConfigMergePolicy.SHALLOW_MERGE,
        )

        assert service.get_config("rag_config") == {
            "indexing": {"chunk_size": 512},
            "retrieval": {"top_k": 5},
        }

    def test_apply_configs_and_clear(self):
        service = CapabilityConfigService()
        service.apply_configs(
            {
                "source_verification_config": {"min_credibility": 0.8},
                "fact_checking_config": {"min_source_count_for_claim": 3},
            }
        )

        assert service.list_names() == [
            "fact_checking_config",
            "source_verification_config",
        ]
        service.clear("fact_checking_config")
        assert service.has_config("fact_checking_config") is False
        service.clear()
        assert service.snapshot() == {}

    def test_scope_isolation_between_sessions(self):
        service = CapabilityConfigService()

        service.set_config("citation_config", {"style": "apa"}, scope_key="session-a")
        service.set_config("citation_config", {"style": "mla"}, scope_key="session-b")

        assert service.get_config("citation_config", scope_key="session-a") == {"style": "apa"}
        assert service.get_config("citation_config", scope_key="session-b") == {"style": "mla"}
        assert service.get_config("citation_config", scope_key="missing", default=None) is None

    def test_clear_scope_only_does_not_remove_other_scopes(self):
        service = CapabilityConfigService()
        service.set_config(
            "source_verification_config",
            {"min_credibility": 0.8},
            scope_key="session-a",
        )
        service.set_config(
            "source_verification_config",
            {"min_credibility": 0.6},
            scope_key="session-b",
        )

        service.clear(scope_key="session-a")

        assert service.snapshot(scope_key="session-a") == {}
        assert service.snapshot(scope_key="session-b") == {
            "source_verification_config": {"min_credibility": 0.6}
        }

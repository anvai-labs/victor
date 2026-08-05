# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for canonical evaluation service exports."""

from victor.evaluation.services import (
    ValidatedSessionTruthServiceProtocol,
    ValidatedSessionTruthService as exported_service,
    create_validated_session_truth_service,
    materialize_validated_session_truth_service,
    parse_validated_session_truth_legacy_kwargs,
    resolve_validated_session_truth_service,
)
from victor.evaluation.validated_session_truth_emitters import (
    ValidatedSessionTruthEmitterRegistry,
)
from victor.evaluation.validated_session_truth_service import (
    ValidatedSessionTruthService as concrete_service,
)


def test_evaluation_services_reexport_validated_session_truth_service():
    """Evaluation services should expose the canonical validated-session service."""
    assert exported_service is concrete_service


def test_create_validated_session_truth_service_uses_supplied_registry():
    registry = ValidatedSessionTruthEmitterRegistry()

    service = create_validated_session_truth_service(registry)

    assert isinstance(service, concrete_service)
    assert service._emitters is registry


def test_validated_session_truth_service_protocol_accepts_duck_typed_service():
    class StubService:
        def persist_evaluation_result(self, result, **kwargs):
            return []

        def persist_validation_result(self, **kwargs):
            return None

    assert isinstance(StubService(), ValidatedSessionTruthServiceProtocol)


def test_resolve_validated_session_truth_service_prefers_explicit_service():
    class StubService:
        def persist_evaluation_result(self, result, **kwargs):
            return []

        def persist_validation_result(self, **kwargs):
            return None

    service = StubService()

    resolved = resolve_validated_session_truth_service(service=service)

    assert resolved is service


def test_resolve_validated_session_truth_service_delegates_to_factory(monkeypatch):
    captured = {}
    stub_service = object()

    def fake_factory(emitters=None):
        captured["emitters"] = emitters
        return stub_service

    monkeypatch.setattr(
        "victor.evaluation.services.create_validated_session_truth_service",
        fake_factory,
    )
    registry = ValidatedSessionTruthEmitterRegistry()

    resolved = resolve_validated_session_truth_service(emitters=registry)

    assert resolved is stub_service
    assert captured["emitters"] is registry


def test_parse_validated_session_truth_legacy_kwargs_returns_registry():
    registry = ValidatedSessionTruthEmitterRegistry()

    parsed = parse_validated_session_truth_legacy_kwargs(
        {"validated_session_truth_emitters": registry}
    )

    assert parsed is registry


def test_parse_validated_session_truth_legacy_kwargs_rejects_unexpected_keys():
    try:
        parse_validated_session_truth_legacy_kwargs({"unexpected": object()})
    except TypeError as exc:
        assert str(exc) == "Unexpected keyword argument(s): unexpected"
    else:
        raise AssertionError("Expected TypeError for unexpected keyword")


def test_materialize_validated_session_truth_service_composes_parser_and_resolver(
    monkeypatch,
):
    captured = {}
    registry = ValidatedSessionTruthEmitterRegistry()
    stub_service = object()

    def fake_parse(legacy_kwargs):
        captured["legacy_kwargs"] = legacy_kwargs
        return registry

    def fake_resolve(*, service=None, emitters=None):
        captured["service"] = service
        captured["emitters"] = emitters
        return stub_service

    monkeypatch.setattr(
        "victor.evaluation.services.parse_validated_session_truth_legacy_kwargs",
        fake_parse,
    )
    monkeypatch.setattr(
        "victor.evaluation.services.resolve_validated_session_truth_service",
        fake_resolve,
    )

    resolved = materialize_validated_session_truth_service(
        service="explicit-service",
        legacy_kwargs={"validated_session_truth_emitters": registry},
    )

    assert resolved is stub_service
    assert captured["legacy_kwargs"] == {"validated_session_truth_emitters": registry}
    assert captured["service"] == "explicit-service"
    assert captured["emitters"] is registry

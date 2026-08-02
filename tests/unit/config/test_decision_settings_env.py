# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License").

"""FEP-0012: the Phase-7 A/B knobs on DecisionServiceSettings are env-overridable
so the classifier/completion A/B can be run as a one-liner without source edits.
"""

from __future__ import annotations

from victor.agent.services.decision_backend import DecisionBackend
from victor.config.decision_settings import DecisionServiceSettings


def test_defaults_preserved(monkeypatch):
    monkeypatch.delenv("VICTOR_DECISION_BACKEND", raising=False)
    monkeypatch.delenv("VICTOR_LOCAL_LEARNING_ENABLED", raising=False)
    monkeypatch.delenv("VICTOR_LOCAL_CLASSIFIER_COMPLETION_SIGNAL", raising=False)
    s = DecisionServiceSettings()
    assert s.decision_backend == DecisionBackend.AUTO
    assert s.local_learning_enabled is True
    assert s.local_classifier_completion_signal is False


def test_env_overrides_apply(monkeypatch):
    monkeypatch.setenv("VICTOR_DECISION_BACKEND", "edge")
    monkeypatch.setenv("VICTOR_LOCAL_LEARNING_ENABLED", "false")
    monkeypatch.setenv("VICTOR_LOCAL_CLASSIFIER_COMPLETION_SIGNAL", "true")
    s = DecisionServiceSettings()
    assert s.decision_backend == DecisionBackend.EDGE
    assert s.local_learning_enabled is False
    assert s.local_classifier_completion_signal is True


def test_env_backend_falls_back_to_auto_on_garbage(monkeypatch):
    monkeypatch.setenv("VICTOR_DECISION_BACKEND", "nonsense")
    assert DecisionServiceSettings().decision_backend == DecisionBackend.AUTO

"""Tests for the classification backend enum + resolver (mirrors FEP-0012).

``use_tiered_classification()`` is the single decision for whether classification
triage routes through the ``TieredDecisionService``. ``AUTO`` (default settings)
defers to the legacy ``USE_TIERED_CLASSIFICATION`` flag (prior behavior); explicit
``TIERED`` forces it regardless of the flag.
"""

import types


from victor.agent.services.classification_backend import (
    ClassificationBackend,
    use_tiered_classification,
)
from victor.core.feature_flags import FeatureFlag, is_feature_enabled


def _set_backend(monkeypatch, value) -> None:
    """Override the settings model read by the resolver."""
    from victor.config import classification_settings as cs

    monkeypatch.setattr(
        cs,
        "ClassificationSettings",
        lambda: types.SimpleNamespace(classification_backend=value),
    )


def _flag(monkeypatch, on: bool) -> None:
    """Set USE_TIERED_CLASSIFICATION via env (auto-cleaned by monkeypatch)."""
    monkeypatch.setenv(
        FeatureFlag.USE_TIERED_CLASSIFICATION.get_env_var_name(),
        "true" if on else "false",
    )


def test_parse():
    assert ClassificationBackend.parse("tiered") is ClassificationBackend.TIERED
    assert ClassificationBackend.parse("auto") is ClassificationBackend.AUTO
    assert ClassificationBackend.parse(ClassificationBackend.TIERED) is ClassificationBackend.TIERED
    assert ClassificationBackend.parse("garbage") is ClassificationBackend.AUTO
    assert ClassificationBackend.parse(None) is ClassificationBackend.AUTO


def test_auto_flag_off_is_direct(monkeypatch):
    _flag(monkeypatch, False)
    assert is_feature_enabled(FeatureFlag.USE_TIERED_CLASSIFICATION) is False
    _set_backend(monkeypatch, ClassificationBackend.AUTO)
    assert use_tiered_classification() is False


def test_auto_flag_on_is_tiered(monkeypatch):
    _flag(monkeypatch, True)
    assert is_feature_enabled(FeatureFlag.USE_TIERED_CLASSIFICATION) is True
    _set_backend(monkeypatch, ClassificationBackend.AUTO)
    assert use_tiered_classification() is True


def test_explicit_tiered_overrides_flag_off(monkeypatch):
    _flag(monkeypatch, False)
    _set_backend(monkeypatch, ClassificationBackend.TIERED)
    assert use_tiered_classification() is True


def test_default_settings_backend_is_auto():
    from victor.config.classification_settings import ClassificationSettings

    assert ClassificationSettings().classification_backend is ClassificationBackend.AUTO

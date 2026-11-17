"""Mirror suite for victor/config/feature_config.py (FOU-D5f)."""

from victor.config.feature_config import DEFAULT_FEATURE_FLAGS


def test_use_stategraph_agentic_loop_default_is_true():
    """FOU-D5f (ADR-036): the StateGraph executor graduated to default-on."""
    assert DEFAULT_FEATURE_FLAGS["use_stategraph_agentic_loop"] is True

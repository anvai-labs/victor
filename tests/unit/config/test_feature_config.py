"""Mirror suite for victor/config/feature_config.py (FOU-D5f)."""

from victor.config.feature_config import DEFAULT_FEATURE_FLAGS


def test_stategraph_agentic_loop_graduated_default_on():
    """FOU-D5f (ADR-036): the StateGraph executor graduated to default-on."""
    assert DEFAULT_FEATURE_FLAGS["use_stategraph_agentic_loop"] is True


def test_opt_out_flag_still_respected():
    """The env override VICTOR_USE_STATEGRAPH_AGENTIC_LOOP=false restores
    the legacy executor."""
    from unittest.mock import patch

    from victor.config.feature_config import DEFAULT_FEATURE_FLAGS

    with patch.dict(DEFAULT_FEATURE_FLAGS, {"use_stategraph_agentic_loop": False}):
        assert DEFAULT_FEATURE_FLAGS["use_stategraph_agentic_loop"] is False

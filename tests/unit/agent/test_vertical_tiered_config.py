"""Tests for VerticalContext.get_vertical_tiered_config (extracted from orchestrator)."""

from unittest.mock import MagicMock, patch


from victor.agent.vertical_context import VerticalContext


class TestGetVerticalTieredConfig:
    def test_returns_config_from_active_vertical(self):
        mock_config = MagicMock()
        mock_vertical = MagicMock()
        mock_vertical.get_tiered_tool_config.return_value = mock_config

        mock_loader = MagicMock()
        mock_loader.active_vertical = mock_vertical

        with patch(
            "victor.core.verticals.vertical_loader.get_vertical_loader",
            return_value=mock_loader,
        ):
            result = VerticalContext.get_vertical_tiered_config()

        assert result is mock_config
        mock_vertical.get_tiered_tool_config.assert_called_once()

    def test_returns_none_when_no_active_vertical(self):
        mock_loader = MagicMock()
        mock_loader.active_vertical = None

        with patch(
            "victor.core.verticals.vertical_loader.get_vertical_loader",
            return_value=mock_loader,
        ):
            result = VerticalContext.get_vertical_tiered_config()

        assert result is None

    def test_returns_none_when_vertical_has_no_getter(self):
        mock_vertical = MagicMock(spec=[])  # No attributes at all
        mock_loader = MagicMock()
        mock_loader.active_vertical = mock_vertical

        with patch(
            "victor.core.verticals.vertical_loader.get_vertical_loader",
            return_value=mock_loader,
        ):
            result = VerticalContext.get_vertical_tiered_config()

        assert result is None

    def test_returns_none_when_getter_not_callable(self):
        mock_vertical = MagicMock()
        mock_vertical.get_tiered_tool_config = "not callable"

        mock_loader = MagicMock()
        mock_loader.active_vertical = mock_vertical

        with patch(
            "victor.core.verticals.vertical_loader.get_vertical_loader",
            return_value=mock_loader,
        ):
            result = VerticalContext.get_vertical_tiered_config()

        assert result is None

    def test_returns_none_on_exception(self):
        with patch(
            "victor.core.verticals.vertical_loader.get_vertical_loader",
            side_effect=RuntimeError("loader broken"),
        ):
            result = VerticalContext.get_vertical_tiered_config()

        assert result is None

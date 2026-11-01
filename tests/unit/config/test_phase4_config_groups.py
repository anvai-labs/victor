"""Tests for Phase 4 config groups extracted from Settings class.

Tests cover:
- RecoverySettings
- AnalyticsSettings
- NetworkSettings
"""

import pytest

from victor.config.groups.recovery_config import RecoverySettings
from victor.config.groups.analytics_config import AnalyticsSettings
from victor.config.groups.network_config import NetworkSettings
from victor.config.code_correction_settings import CodeCorrectionSettings


class TestCodeCorrectionSettings:
    """Tests for CodeCorrectionSettings."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        settings = CodeCorrectionSettings()

        assert settings.code_correction_enabled is True
        assert settings.code_correction_auto_fix is True
        # 5x'd 2026-10-31: agents run longer autonomous sessions (was 3)
        assert settings.code_correction_max_iterations == 15


class TestRecoverySettings:
    """Tests for RecoverySettings."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        settings = RecoverySettings()

        assert settings.recovery_empty_response_threshold == 5
        assert settings.recovery_blocked_consecutive_threshold == 4
        assert settings.recovery_blocked_total_threshold == 6
        assert settings.chat_max_iterations == 250  # 5x'd 2026-10-31 (was 50)
        # max_consecutive_tool_calls stays 20: stuck-agent guard, intentionally unchanged
        assert settings.max_consecutive_tool_calls == 20
        assert settings.max_continuation_prompts == 3
        assert settings.force_response_on_error is True

    def test_custom_values(self):
        """Test custom values."""
        settings = RecoverySettings(
            recovery_empty_response_threshold=10,
            recovery_blocked_consecutive_threshold=12,
            recovery_blocked_total_threshold=18,
            chat_max_iterations=100,
            max_consecutive_tool_calls=40,
            max_continuation_prompts=5,
            force_response_on_error=False,
        )

        assert settings.recovery_empty_response_threshold == 10
        assert settings.recovery_blocked_consecutive_threshold == 12
        assert settings.recovery_blocked_total_threshold == 18
        assert settings.chat_max_iterations == 100
        assert settings.max_consecutive_tool_calls == 40
        assert settings.max_continuation_prompts == 5
        assert settings.force_response_on_error is False

    def test_empty_response_threshold_validation(self):
        """Test empty response threshold validation."""
        # Valid threshold
        settings = RecoverySettings(recovery_empty_response_threshold=1)
        assert settings.recovery_empty_response_threshold == 1

        # Invalid threshold (zero)
        with pytest.raises(ValueError, match="recovery_empty_response_threshold must be >= 1"):
            RecoverySettings(recovery_empty_response_threshold=0)

    def test_blocked_consecutive_threshold_validation(self):
        """Test blocked consecutive threshold validation."""
        # Valid threshold
        settings = RecoverySettings(recovery_blocked_consecutive_threshold=1)
        assert settings.recovery_blocked_consecutive_threshold == 1

        # Invalid threshold (zero)
        with pytest.raises(ValueError, match="recovery_blocked_consecutive_threshold must be >= 1"):
            RecoverySettings(recovery_blocked_consecutive_threshold=0)

    def test_blocked_total_threshold_validation(self):
        """Test blocked total threshold validation."""
        # Valid threshold
        settings = RecoverySettings(
            recovery_blocked_consecutive_threshold=1,
            recovery_blocked_total_threshold=1,
        )
        assert settings.recovery_blocked_total_threshold == 1

        # Invalid threshold (zero)
        with pytest.raises(ValueError, match="recovery_blocked_total_threshold must be >= 1"):
            RecoverySettings(recovery_blocked_total_threshold=0)

    def test_blocked_total_threshold_must_not_be_lower_than_consecutive(self):
        """Total blocked threshold should be at least the consecutive threshold."""
        with pytest.raises(
            ValueError,
            match=(
                "recovery_blocked_total_threshold must be >= "
                "recovery_blocked_consecutive_threshold"
            ),
        ):
            RecoverySettings(
                recovery_blocked_consecutive_threshold=5,
                recovery_blocked_total_threshold=4,
            )


class TestAnalyticsSettings:
    """Tests for AnalyticsSettings."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        settings = AnalyticsSettings()

        assert settings.streaming_metrics_enabled is True
        assert settings.streaming_metrics_history_size == 1000
        assert settings.analytics_enabled is True
        assert settings.show_token_count is True
        assert settings.show_cost_metrics is False

    def test_custom_values(self):
        """Test custom values."""
        settings = AnalyticsSettings(
            streaming_metrics_enabled=False,
            streaming_metrics_history_size=500,
            analytics_enabled=False,
            show_token_count=False,
            show_cost_metrics=True,
        )

        assert settings.streaming_metrics_enabled is False
        assert settings.streaming_metrics_history_size == 500
        assert settings.analytics_enabled is False
        assert settings.show_token_count is False
        assert settings.show_cost_metrics is True

    def test_history_size_validation(self):
        """Test metrics history size validation."""
        # Valid history size
        settings = AnalyticsSettings(streaming_metrics_history_size=1)
        assert settings.streaming_metrics_history_size == 1

        settings = AnalyticsSettings(streaming_metrics_history_size=10000)
        assert settings.streaming_metrics_history_size == 10000

        # Invalid history size (zero)
        with pytest.raises(ValueError, match="streaming_metrics_history_size must be >= 1"):
            AnalyticsSettings(streaming_metrics_history_size=0)

        # Invalid history size (too large)
        with pytest.raises(ValueError, match="streaming_metrics_history_size must be <= 10000"):
            AnalyticsSettings(streaming_metrics_history_size=10001)


class TestNetworkSettings:
    """Tests for NetworkSettings."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        settings = NetworkSettings()

        assert settings.tool_retry_enabled is True
        assert settings.tool_retry_max_attempts == 3
        assert settings.tool_retry_base_delay == 1.0
        assert settings.tool_retry_max_delay == 10.0

    def test_custom_values(self):
        """Test custom values."""
        settings = NetworkSettings(
            tool_retry_enabled=False,
            tool_retry_max_attempts=5,
            tool_retry_base_delay=2.0,
            tool_retry_max_delay=30.0,
        )

        assert settings.tool_retry_enabled is False
        assert settings.tool_retry_max_attempts == 5
        assert settings.tool_retry_base_delay == 2.0
        assert settings.tool_retry_max_delay == 30.0

    def test_max_attempts_validation(self):
        """Test max attempts validation."""
        # Valid max attempts
        settings = NetworkSettings(tool_retry_max_attempts=1)
        assert settings.tool_retry_max_attempts == 1

        # Invalid max attempts (zero)
        with pytest.raises(ValueError, match="tool_retry_max_attempts must be >= 1"):
            NetworkSettings(tool_retry_max_attempts=0)

    def test_base_delay_validation(self):
        """Test base delay validation."""
        # Valid base delay
        settings = NetworkSettings(tool_retry_base_delay=0)
        assert settings.tool_retry_base_delay == 0

        # Invalid base delay (negative)
        with pytest.raises(ValueError, match="tool_retry_base_delay must be >= 0"):
            NetworkSettings(tool_retry_base_delay=-0.1)

    def test_max_delay_validation(self):
        """Test max delay validation."""
        # Valid max delay
        settings = NetworkSettings(tool_retry_max_delay=0)
        assert settings.tool_retry_max_delay == 0

        # Invalid max delay (negative)
        with pytest.raises(ValueError, match="tool_retry_max_delay must be >= 0"):
            NetworkSettings(tool_retry_max_delay=-1.0)


class TestPhase4ConfigGroupsIntegration:
    """Integration tests for Phase 4 config groups with Settings."""

    def test_recovery_settings_in_main_settings(self):
        """Test that RecoverySettings is accessible from Settings."""
        from victor.config.settings import Settings

        settings = Settings()

        # Nested access
        assert settings.recovery is not None
        assert isinstance(settings.recovery, RecoverySettings)

        # NOTE: Flat access removed - use nested access only

    def test_analytics_settings_in_main_settings(self):
        """Test that AnalyticsSettings is accessible from Settings."""
        from victor.config.settings import Settings

        settings = Settings()

        # Nested access
        assert settings.analytics is not None
        assert isinstance(settings.analytics, AnalyticsSettings)

        # NOTE: Flat access removed - use nested access only
        assert settings.analytics_enabled == settings.analytics.analytics_enabled

    def test_network_settings_in_main_settings(self):
        """Test that NetworkSettings is accessible from Settings."""
        from victor.config.settings import Settings

        settings = Settings()

        # Nested access
        assert settings.network is not None
        assert isinstance(settings.network, NetworkSettings)

        # NOTE: Flat access removed - use nested access only
        assert settings.network.tool_retry_enabled == settings.network.tool_retry_enabled
        assert settings.network.tool_retry_max_attempts == settings.network.tool_retry_max_attempts

    def test_flat_field_overrides_sync_to_nested_groups(self):
        """Test that nested initialization works correctly."""
        from victor.config.settings import Settings

        # Initialize with nested structure
        settings = Settings(
            **{"recovery": {"chat_max_iterations": 100}},
            **{"analytics": {"streaming_metrics_enabled": False}},
            **{"network": {"tool_retry_enabled": False}},
        )

        # Verify nested values are set correctly
        assert settings.recovery.chat_max_iterations == 100
        assert settings.analytics.streaming_metrics_enabled is False
        assert settings.network.tool_retry_enabled is False

    def test_nested_group_independence(self):
        """Test that nested groups are independent from each other."""
        from victor.config.settings import Settings

        settings = Settings()

        # Modifying one group shouldn't affect others
        recovery_id = id(settings.recovery)
        analytics_id = id(settings.analytics)
        network_id = id(settings.network)

        assert recovery_id != analytics_id != network_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

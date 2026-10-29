"""Analytics and metrics configuration.

This module contains settings for:
- Streaming metrics and performance monitoring
- Usage analytics
- Cost metrics display
- Token tracking
"""

from pydantic import BaseModel, field_validator


class AnalyticsSettings(BaseModel):
    """Analytics and metrics settings.

    Controls performance monitoring, usage analytics, and cost tracking.
    """

    # ==========================================================================
    # Streaming Metrics (Performance Monitoring)
    # ==========================================================================
    # Enable streaming performance metrics collection and reporting.
    # Provides real-time insights into agent performance, tool execution, and resource usage.
    streaming_metrics_enabled: bool = True
    streaming_metrics_history_size: int = 1000  # Number of metrics samples to retain

    # ==========================================================================
    # Usage Analytics
    # ==========================================================================
    # Enable usage analytics for tracking agent behavior and performance.
    # Logs usage events to JSONL file for analysis and debugging.
    analytics_enabled: bool = True
    # Note: analytics_log_file now uses get_project_paths().global_logs_dir / "usage.jsonl"

    # ==========================================================================
    # UI Display Settings
    # ==========================================================================
    # Controls what metrics are displayed in the UI.
    show_token_count: bool = True
    show_cost_metrics: bool = False  # Show cost in metrics display (e.g., "$0.015")

    @field_validator("streaming_metrics_history_size")
    @classmethod
    def validate_history_size(cls, v: int) -> int:
        """Validate metrics history size is positive and reasonable.

        Args:
            v: History size

        Returns:
            Validated history size

        Raises:
            ValueError: If history size is not positive or too large
        """
        if v < 1:
            raise ValueError("streaming_metrics_history_size must be >= 1")
        if v > 10000:
            raise ValueError("streaming_metrics_history_size must be <= 10000 (unreasonably large)")
        return v

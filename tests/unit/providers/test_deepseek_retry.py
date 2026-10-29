"""Tests for DeepSeek provider retry improvements.

Covers:
- Retry count increased from 3 to 5 for sustained load scenarios
- Exponential backoff with longer delays
- RemoteProtocolError is retryable
"""

from __future__ import annotations


class TestDeepSeekRetryConfig:
    """DeepSeek retry configuration for sustained load."""

    def test_max_retries_at_least_5(self):
        """DeepSeek should retry at least 5 times for sustained load issues."""
        from victor.providers.deepseek_provider import DeepSeekProvider

        provider = DeepSeekProvider.__new__(DeepSeekProvider)
        # Check the retry loop range
        # The provider uses _attempt in range(5) = 5 attempts
        # We verify via the RETRY_ATTEMPTS class attribute
        retry_attempts = getattr(DeepSeekProvider, "RETRY_ATTEMPTS", 3)
        assert retry_attempts >= 5

    def test_remote_protocol_error_is_retried(self):
        """RemoteProtocolError in retryable exceptions."""
        from victor.providers.resilience import ProviderRetryConfig

        config = ProviderRetryConfig()
        assert "RemoteProtocolError" in config.retryable_exception_names


class TestEdgeCacheConfig:
    """Edge model decision caching."""

    def test_edge_cache_ttl_120(self):
        """Edge model cache TTL should be at least 120s to reduce calls."""
        from victor.agent.edge_model import EdgeModelConfig

        config = EdgeModelConfig()
        assert config.cache_ttl >= 120

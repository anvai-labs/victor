"""Co-design review item 18: shared failure-classification vocabulary.

Pins that BaseProvider and ProviderRetryStrategy consume the SAME token
tables from victor.providers.failure_taxonomy (a mutation to the shared
source is visible to both, instead of each holding its own copy that can
drift), and that TimeoutProfile's defaults match the current effective
per-site values it documents.
"""

from unittest.mock import MagicMock

import pytest

from victor.providers import failure_taxonomy
from victor.providers.base import BaseProvider, CompletionResponse, StreamChunk
from victor.providers.resilience import ProviderRetryStrategy
from victor.providers.timeout_profile import TimeoutProfile


class _DummyProvider(BaseProvider):
    def __init__(self) -> None:
        self.timeout = 12

    @property
    def name(self) -> str:
        return "dummy"

    async def chat(
        self,
        messages,
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools=None,
        **kwargs,
    ) -> CompletionResponse:
        return CompletionResponse(content="ok")

    async def stream(
        self,
        messages,
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools=None,
        **kwargs,
    ):
        yield StreamChunk(content="ok")

    async def close(self) -> None:
        pass


class _FakeStatus429Error(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.status_code = 429


class TestHardRateLimitSingleSource:
    def test_base_provider_recognizes_documented_token(self):
        provider = _DummyProvider()
        error = _FakeStatus429Error("Error: quota exceeded for this billing period")
        assert provider._looks_like_hard_rate_limit(error) is True

    def test_retry_strategy_recognizes_same_documented_token(self):
        strategy = ProviderRetryStrategy()
        error = _FakeStatus429Error("Error: quota exceeded for this billing period")
        assert strategy._is_hard_rate_limit(error) is True

    def test_mutating_shared_table_changes_both_consumers(self, monkeypatch):
        """The regression this guards against: two independently-maintained
        copies of the token list silently drifting apart. Both consumers
        must read through the module attribute, not a name bound at import
        time, so a single patch changes both at once."""
        monkeypatch.setattr(failure_taxonomy, "HARD_RATE_LIMIT_TOKENS", ("zzz-new-token",))

        provider = _DummyProvider()
        strategy = ProviderRetryStrategy()
        error = _FakeStatus429Error("Error: zzz-new-token triggered")

        assert provider._looks_like_hard_rate_limit(error) is True
        assert strategy._is_hard_rate_limit(error) is True

        # And the token that used to trigger it, no longer does — proves
        # both consumers are reading the patched table, not a stale copy.
        old_token_error = _FakeStatus429Error("quota exceeded")
        assert provider._looks_like_hard_rate_limit(old_token_error) is False
        assert strategy._is_hard_rate_limit(old_token_error) is False


class TestConnectionTimeoutSingleSource:
    def test_base_provider_connection_classification_uses_shared_tokens(self):
        provider = _DummyProvider()
        assert provider._is_connection_error_like(Exception("connection refused")) is True
        assert provider._is_connection_error_like(Exception("nothing relevant here")) is False

    def test_base_provider_timeout_classification_uses_shared_tokens(self):
        provider = _DummyProvider()
        assert provider._is_timeout_error_like(Exception("request timed out")) is True
        assert provider._is_timeout_error_like(Exception("nothing relevant here")) is False

    def test_mutating_shared_connection_tokens_changes_base_provider(self, monkeypatch):
        monkeypatch.setattr(failure_taxonomy, "CONNECTION_TOKENS", ("zzz-conn-token",))
        provider = _DummyProvider()
        assert provider._is_connection_error_like(Exception("zzz-conn-token seen")) is True
        assert provider._is_connection_error_like(Exception("connection refused")) is False


class TestSmartRouterHealthAwareness:
    @pytest.mark.asyncio
    async def test_score_health_scores_connection_issue_between_critical_and_generic(self):
        from victor.providers.smart_router import RoutingDecisionEngine

        router = RoutingDecisionEngine.__new__(RoutingDecisionEngine)
        health_result = MagicMock(healthy=False, issues=["connection refused by upstream"])
        router.checker = MagicMock()
        router.checker.get_provider_health.return_value = health_result

        score = await router._score_health("dummy")

        assert score == 0.15

    @pytest.mark.asyncio
    async def test_score_health_scores_timeout_issue_between_critical_and_generic(self):
        from victor.providers.smart_router import RoutingDecisionEngine

        router = RoutingDecisionEngine.__new__(RoutingDecisionEngine)
        health_result = MagicMock(healthy=False, issues=["request timed out"])
        router.checker = MagicMock()
        router.checker.get_provider_health.return_value = health_result

        score = await router._score_health("dummy")

        assert score == 0.15

    @pytest.mark.asyncio
    async def test_score_health_still_scores_auth_issue_as_critical(self):
        from victor.providers.smart_router import RoutingDecisionEngine

        router = RoutingDecisionEngine.__new__(RoutingDecisionEngine)
        health_result = MagicMock(healthy=False, issues=["invalid api_key"])
        router.checker = MagicMock()
        router.checker.get_provider_health.return_value = health_result

        score = await router._score_health("dummy")

        assert score == 0.0

    @pytest.mark.asyncio
    async def test_score_health_generic_issue_unchanged(self):
        from victor.providers.smart_router import RoutingDecisionEngine

        router = RoutingDecisionEngine.__new__(RoutingDecisionEngine)
        health_result = MagicMock(healthy=False, issues=["something unrelated broke"])
        router.checker = MagicMock()
        router.checker.get_provider_health.return_value = health_result

        score = await router._score_health("dummy")

        assert score == 0.3


class TestTimeoutProfileDefaults:
    """Defaults must match the CURRENT effective value at each site this
    profile documents — this is a documentation artifact in this phase, not
    wired into behavior, so drift here is silent unless pinned."""

    def test_request_timeout_matches_factory_config_default(self):
        import dataclasses

        from victor.providers.factory import ProviderConfig

        timeout_field = next(f for f in dataclasses.fields(ProviderConfig) if f.name == "timeout")
        assert TimeoutProfile().request_timeout_seconds == timeout_field.default

    def test_sdk_timeout_matches_base_provider_constructor_default(self):
        import inspect

        sig = inspect.signature(BaseProvider.__init__)
        assert TimeoutProfile().sdk_timeout_seconds == sig.parameters["timeout"].default

    def test_circuit_breaker_recovery_matches_base_provider_default(self):
        import inspect

        sig = inspect.signature(BaseProvider.__init__)
        assert (
            TimeoutProfile().circuit_breaker_recovery_seconds
            == sig.parameters["circuit_breaker_recovery_timeout"].default
        )

    def test_retry_delays_match_provider_retry_config_defaults(self):
        from victor.providers.resilience import ProviderRetryConfig

        config = ProviderRetryConfig()
        profile = TimeoutProfile()
        assert profile.retry_base_delay_seconds == config.base_delay_seconds
        assert profile.retry_max_delay_seconds == config.max_delay_seconds

    def test_http_slots_match_dead_timeouts_constants(self):
        from victor.config.timeouts import Timeouts

        profile = TimeoutProfile()
        assert profile.llm_api_http_timeout_seconds == Timeouts.HTTP_LLM_API
        assert profile.embedding_http_timeout_seconds == Timeouts.HTTP_EMBEDDING

    def test_profile_is_frozen(self):
        profile = TimeoutProfile()
        with pytest.raises(Exception):
            profile.request_timeout_seconds = 1.0

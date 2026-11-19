import json

import pytest
from pydantic import ValidationError

from victor.config.settings import ProviderConfig
from victor.providers import sandhi_transport as st
from victor.providers.sandhi_transport import SandhiTypedProviderMixin


def test_stream_body_budget_config_survives_profile_serialization():
    value = ProviderConfig(stream_body_timeout_secs=600).model_dump(exclude_none=True)
    assert value["stream_body_timeout_secs"] == 600


@pytest.mark.parametrize("value", [0, 0.0005, -1, float("nan"), float("inf"), 841, True])
def test_invalid_stream_body_budget_config_is_rejected(value):
    with pytest.raises(ValidationError):
        ProviderConfig(stream_body_timeout_secs=value)


def test_stream_body_budget_is_omitted_and_header_is_not_sent_by_default():
    provider = SandhiTypedProviderMixin()
    provider._gateway_overrides = lambda: None
    assert provider._sandhi_stream_body_timeout_secs() is None
    assert "x-sandhi-stream-body-budget-ms" not in provider._sandhi_stream_wire_headers()


def test_stream_body_budget_becomes_one_integer_gateway_header(monkeypatch):
    provider = SandhiTypedProviderMixin()
    provider._gateway_overrides = lambda: {"url": "https://gateway.test"}
    monkeypatch.setattr(st, "_wire_call_headers", lambda: {"x-sandhi-step-id": "step-1"})
    provider.extra_config = {"stream_body_timeout_secs": 600}
    headers = provider._sandhi_stream_wire_headers()
    assert headers["x-sandhi-step-id"] == "step-1"
    assert headers["x-sandhi-stream-body-budget-ms"] == "600000"
    assert json.dumps(headers).count("x-sandhi-stream-body-budget-ms") == 1


def test_invalid_stream_body_budget_runtime_is_rejected():
    provider = SandhiTypedProviderMixin()
    provider.extra_config = {"stream_body_timeout_secs": True}
    with pytest.raises(ValueError, match="stream body timeout"):
        provider._sandhi_stream_body_timeout_secs()


@pytest.mark.asyncio
async def test_stream_body_budget_reaches_gateway_stream_wire_request(monkeypatch):
    provider = SandhiTypedProviderMixin()
    provider.extra_config = {"stream_body_timeout_secs": 600}
    provider._validate_gateway_request = lambda request: None
    provider._typed_provider = lambda model: object()
    provider._gateway_overrides = lambda: {"url": "https://gateway.test"}
    monkeypatch.setattr(st, "_wire_call_headers", lambda: {"x-sandhi-step-id": "step-1"})
    provider._sandhi_timeout = lambda: 1.0
    provider._sandhi_stream_idle_timeout = lambda: 1.0
    captured = []

    async def stream_wrapper(_provider, _request, wire_headers, _idle_timeout):
        captured.append(wire_headers)
        yield '{"event":"finish","reason":"stop"}'

    provider._sandhi_stream_with_idle_timeout = stream_wrapper
    chunks = [chunk async for chunk in provider._sandhi_stream({"model": "test-model"})]

    assert len(chunks) == 1
    assert captured == [
        json.dumps(
            {
                "x-sandhi-step-id": "step-1",
                "x-sandhi-stream-body-budget-ms": "600000",
            }
        )
    ]


@pytest.mark.asyncio
async def test_stream_body_budget_is_not_sent_on_buffered_complete_request(monkeypatch):
    provider = SandhiTypedProviderMixin()
    provider.extra_config = {"stream_body_timeout_secs": 600}
    provider._validate_gateway_request = lambda request: None
    provider._gateway_overrides = lambda: {"url": "https://gateway.test"}
    monkeypatch.setattr(st, "_wire_call_headers", lambda: {"x-sandhi-step-id": "step-1"})
    provider._sandhi_timeout = lambda: 1.0
    captured = []

    class CompleteProvider:
        @staticmethod
        async def complete_json(_request, wire_headers):
            captured.append(wire_headers)
            return '{"ok":true}'

    provider._typed_provider = lambda model: CompleteProvider()
    result = await provider._sandhi_complete({"model": "test-model"})

    assert result == {"ok": True}
    assert captured == [json.dumps({"x-sandhi-step-id": "step-1"})]


class TestStreamBodyBudgetReviewFixes:
    """Pins for the six code-review findings on #1895."""

    def test_direct_mode_validation_fails_fast(self):
        """Finding 1a: the knob must fail the same way in direct mode as in
        gateway mode — not silently no-op."""
        provider = SandhiTypedProviderMixin()
        provider.extra_config = {"stream_body_timeout_secs": True}
        # Direct mode (no gateway overrides) still validates eagerly.
        with pytest.raises(ValueError, match="stream body timeout"):
            provider._sandhi_stream_body_timeout_secs()

    def test_fractional_ms_rounds_not_truncates(self, monkeypatch):
        """Finding 1e: 1.5ms must round to '2', not truncate to '1'."""
        provider = SandhiTypedProviderMixin()
        provider.extra_config = {"stream_body_timeout_secs": 0.0015}
        headers = provider._sandhi_stream_wire_headers()
        assert headers.get("x-sandhi-stream-body-budget-ms") == "2"

    def test_bool_message_names_millisecond_floor(self):
        """Finding 1 (cosmetic): the bool-rejection message must not say
        'between 0 and 840' — the actual floor is one millisecond."""
        provider = SandhiTypedProviderMixin()
        provider.extra_config = {"stream_body_timeout_secs": True}
        with pytest.raises(ValueError, match="one millisecond"):
            provider._sandhi_stream_body_timeout_secs()

    def test_non_numeric_gives_labeled_error(self):
        """Finding (raw ValueError without context): a string value gets
        a labeled error, not a bare float() conversion failure."""
        provider = SandhiTypedProviderMixin()
        provider.extra_config = {"stream_body_timeout_secs": "ten"}
        with pytest.raises(ValueError, match="stream body timeout"):
            provider._sandhi_stream_body_timeout_secs()

    def test_buffered_path_uses_shared_base_headers(self):
        """Finding 3: the buffered path must use the shared base-header
        construction (single site, no drift)."""
        provider = SandhiTypedProviderMixin()
        provider._gateway_overrides = lambda: {"url": "https://gateway.test"}
        provider.extra_config = {}
        headers = provider._sandhi_base_wire_headers()
        assert headers == provider._sandhi_stream_wire_headers()

    def test_body_budget_is_emitted_under_clean_version_one_pin(self, monkeypatch):
        """The exact 0.22 pin always owns the body-budget preflight header."""
        monkeypatch.setattr(st, "_wire_call_headers", lambda: {"x-sandhi-step-id": "s"})
        provider = SandhiTypedProviderMixin()
        provider.extra_config = {"stream_body_timeout_secs": 600}
        headers = provider._sandhi_stream_wire_headers()
        assert headers["x-sandhi-stream-body-budget-ms"] == "600000"

    def test_idle_and_setup_use_shared_validator(self):
        """Finding 1c: the shared validator enforces the same bounds as the
        old inline checks (regression: no drift between the three knobs)."""
        provider = SandhiTypedProviderMixin()
        provider.extra_config = {"stream_idle_timeout_secs": True}
        with pytest.raises(ValueError, match="stream idle timeout"):
            provider._sandhi_stream_idle_timeout()
        provider.extra_config = {"stream_setup_timeout_secs": "bad"}
        with pytest.raises(ValueError, match="stream setup timeout"):
            provider._sandhi_stream_setup_timeout()

    def test_timeout_profile_documents_body_budget(self):
        """Finding 5: the TimeoutProfile docstring inventory includes the
        body-budget knob."""
        import inspect
        from victor.providers.timeout_profile import TimeoutProfile

        doc = inspect.getdoc(TimeoutProfile)
        assert "stream_body_timeout_seconds" in doc

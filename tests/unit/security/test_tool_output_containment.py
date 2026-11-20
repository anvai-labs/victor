# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FOU-S3a seam tests: the model-bound tool output is enveloped; the user
preview is not. ADR-037 decision 1 — provenance marking is structural."""

import pytest

from victor.agent.services.tool_service import format_and_prune_tool_output
from victor.security.containment import (
    reset_session_canary,
    set_session_canary,
)


@pytest.fixture(autouse=True)
def _clean_canary():
    reset_session_canary()
    yield
    reset_session_canary()


class TestToolOutputEnvelope:
    def test_llm_output_is_enveloped(self):
        set_session_canary("victor-canary-0123456789abcdef")
        preview, llm_output, _, _, _ = format_and_prune_tool_output(
            tool_name="read",
            arguments={"path": "/tmp/x"},
            output="file body text",
        )
        assert llm_output.startswith("<untrusted ")
        assert llm_output.rstrip().endswith("</untrusted>")
        assert 'origin="read"' in llm_output
        assert 'canary="victor-canary-0123456789abcdef"' in llm_output
        assert "file body text" in llm_output

    def test_preview_is_not_enveloped(self):
        preview, _, _, _, _ = format_and_prune_tool_output(
            tool_name="read", arguments={}, output="file body text"
        )
        assert "<untrusted" not in preview
        assert "file body text" in preview

    def test_envelope_carries_canary_even_without_session_state(self):
        _, llm_output, _, _, _ = format_and_prune_tool_output(
            tool_name="search", arguments={}, output="results"
        )
        assert 'canary="victor-canary-' in llm_output

    def test_envelope_wraps_error_output_too(self):
        _, llm_output, _, _, _ = format_and_prune_tool_output(
            tool_name="shell", arguments={}, output="Traceback: permission denied"
        )
        assert llm_output.startswith("<untrusted ")


class TestEnvelopeFailClosed:
    """Finding 2b: an envelope failure must never silently publish untrusted
    text outside the provenance boundary when enforcement is configured."""

    def test_envelope_failure_publishes_inert_placeholder(self, monkeypatch):
        from victor.security import containment

        def _explode(content, **kwargs):
            raise RuntimeError("envelope infrastructure failure")

        monkeypatch.setattr(containment, "wrap_untrusted", _explode)
        preview, llm_output, _, _, _ = format_and_prune_tool_output(
            tool_name="read",
            arguments={"path": "/tmp/x"},
            output="SECRET untrusted body",
        )
        assert llm_output == containment.CONTAINMENT_FAILURE_PLACEHOLDER
        # The raw content never reaches model context.
        assert "SECRET untrusted body" not in llm_output

    def test_envelopes_disabled_delivers_content_unenveloped(self, monkeypatch):
        """Negative: enforcement explicitly OFF — content is delivered raw."""
        from unittest.mock import patch

        from victor.config.tool_settings import ToolSettings
        from victor.security import containment

        settings = ToolSettings(untrusted_content_envelopes=False)
        with patch("victor.config.tool_settings.get_tool_settings", return_value=settings):
            _, llm_output, _, _, _ = format_and_prune_tool_output(
                tool_name="read",
                arguments={"path": "/tmp/x"},
                output="plain body",
            )
        assert "<untrusted" not in llm_output
        assert "plain body" in llm_output
        assert containment.CONTAINMENT_FAILURE_PLACEHOLDER not in llm_output

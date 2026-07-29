# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Tests for the transport-agnostic empty-response retry policy.

Regression cover for session sandhi-cdfbc589 (2026-07-26), where plan generation
retried three times with byte-identical parameters against a reasoning model and
lost ~100s to three guaranteed-identical failures.
"""

import pytest

from victor.agent.recovery.empty_response import (
    DEFAULT_TEMPERATURE_LADDER,
    MAX_ESCALATED_TOKENS,
    EmptyResponseDiagnosis,
    diagnose_empty_response,
    next_retry_parameters,
)


class TestDiagnoseEmptyResponse:
    """Diagnosis must not infer exhaustion from absent evidence."""

    def test_reasoning_chars_indicates_exhaustion(self):
        diagnosis = diagnose_empty_response({"reasoning_chars": 4096, "stop_reason": "length"})

        assert diagnosis.reasoning_exhausted is True
        assert diagnosis.reasoning_chars == 4096
        assert diagnosis.stop_reason == "length"
        assert "reasoning-token exhaustion" in diagnosis.summary

    @pytest.mark.parametrize("diagnostics", [None, {}, {"stop_reason": "stop"}])
    def test_absent_evidence_is_not_exhaustion(self, diagnostics):
        assert diagnose_empty_response(diagnostics).reasoning_exhausted is False

    def test_zero_reasoning_chars_is_not_exhaustion(self):
        assert diagnose_empty_response({"reasoning_chars": 0}).reasoning_exhausted is False

    @pytest.mark.parametrize("bad", ["not-a-number", object()])
    def test_unparseable_reasoning_chars_degrades_quietly(self, bad):
        diagnosis = diagnose_empty_response({"reasoning_chars": bad})

        assert diagnosis.reasoning_chars is None
        assert diagnosis.reasoning_exhausted is False


class TestNextRetryParameters:
    """Every retry must differ from the attempt that failed."""

    def test_reasoning_exhaustion_raises_max_tokens(self):
        retry = next_retry_parameters(
            attempt=0,
            base_max_tokens=3000,
            diagnosis=EmptyResponseDiagnosis(reasoning_exhausted=True, reasoning_chars=9000),
        )

        assert retry.max_tokens == 12000
        assert retry.escalated is True

    def test_max_tokens_escalation_is_capped(self):
        retry = next_retry_parameters(
            attempt=0,
            base_max_tokens=16000,
            diagnosis=EmptyResponseDiagnosis(reasoning_exhausted=True, reasoning_chars=1),
        )

        assert retry.max_tokens == MAX_ESCALATED_TOKENS

    def test_no_token_escalation_without_reasoning_evidence(self):
        retry = next_retry_parameters(attempt=0, base_max_tokens=3000)

        assert retry.max_tokens == 3000
        # Temperature still varies, so the retry is not a byte-identical repeat.
        assert retry.temperature == DEFAULT_TEMPERATURE_LADDER[0]
        assert retry.escalated is True

    def test_temperature_walks_the_ladder_then_holds(self):
        temps = [
            next_retry_parameters(attempt=i, base_max_tokens=3000).temperature for i in range(4)
        ]

        assert temps[0] == DEFAULT_TEMPERATURE_LADDER[0]
        assert temps[1] == DEFAULT_TEMPERATURE_LADDER[1]
        assert temps[2] == DEFAULT_TEMPERATURE_LADDER[-1]
        assert temps[3] == DEFAULT_TEMPERATURE_LADDER[-1]

    def test_reasoning_effort_only_when_supported(self):
        diagnosis = EmptyResponseDiagnosis(reasoning_exhausted=True, reasoning_chars=100)

        supported = next_retry_parameters(
            attempt=0,
            base_max_tokens=3000,
            diagnosis=diagnosis,
            supports_reasoning_effort=True,
        )
        unsupported = next_retry_parameters(
            attempt=0,
            base_max_tokens=3000,
            diagnosis=diagnosis,
            supports_reasoning_effort=False,
        )

        assert supported.reasoning_effort == "low"
        assert unsupported.reasoning_effort is None

    def test_reasoning_effort_not_sent_without_exhaustion(self):
        retry = next_retry_parameters(
            attempt=0, base_max_tokens=3000, supports_reasoning_effort=True
        )

        assert retry.reasoning_effort is None

    def test_nothing_to_escalate_is_reported_not_hidden(self):
        """With no ladder and no diagnosis, a retry would be a pure repeat."""
        retry = next_retry_parameters(attempt=0, base_max_tokens=3000, temperature_ladder=())

        assert retry.escalated is False
        assert "nothing left to escalate" in retry.reason

    def test_the_sandhi_scenario_never_repeats_a_request(self):
        """The exact failure: 3000 tokens, glm-5.2, reasoning-only responses.

        Each escalated retry must differ from the one before it, and once nothing
        can be changed the policy must say so rather than authorise a repeat —
        which is precisely what the original loop did three times over.
        """
        diagnosis = diagnose_empty_response({"reasoning_chars": 12000, "stop_reason": "length"})
        base = 3000

        issued = []
        for i in range(4):
            retry = next_retry_parameters(
                attempt=i,
                base_max_tokens=base,
                diagnosis=diagnosis,
                supports_reasoning_effort=True,
            )
            if not retry.escalated:
                break
            issued.append((retry.max_tokens, retry.temperature, retry.reasoning_effort))
            base = retry.max_tokens

        assert len(issued) == len(set(issued)), "an escalated retry repeated a prior request"
        assert issued[0][0] == 12000, "first retry should quadruple the token budget"
        assert issued[-1][0] == MAX_ESCALATED_TOKENS
        assert all(params[2] == "low" for params in issued)
        # The loop stopped on its own rather than running to the retry limit.
        assert len(issued) < 4

    def test_exhausted_ladder_and_cap_stops_escalating(self):
        """Beyond the ladder with the cap reached, there is nothing left to change."""
        diagnosis = EmptyResponseDiagnosis(reasoning_exhausted=True, reasoning_chars=1)

        retry = next_retry_parameters(
            attempt=len(DEFAULT_TEMPERATURE_LADDER),
            base_max_tokens=MAX_ESCALATED_TOKENS,
            diagnosis=diagnosis,
            supports_reasoning_effort=True,
        )

        assert retry.escalated is False
        assert "nothing left to escalate" in retry.reason

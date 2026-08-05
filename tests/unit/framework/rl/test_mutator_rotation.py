# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
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

"""One provider's rate limit must not stall a whole evolution pass.

Evolution makes several LLM calls per section, back to back, all to one provider.
On a hosted plan that reliably earns a 429 partway through — and a 429 in the
mutate call is not a degraded result, it is *no* result: ``mutate()`` returns its
input and whatever reformatting runs next becomes the candidate's entire diff.
Two full runs were spent diagnosing that as a strategy problem.
"""

import pytest

from victor.framework.rl.mutator_rotation import (
    MutatorRotation,
    MutatorSpec,
    build_rotation,
    is_rate_limit,
    is_worth_another_provider,
)

ZAI = ("zai", "glm-5.2", "zai-glm52-openai")
KIMI = ("moonshot", "kimi-k3", "kimi")
DEEPSEEK = ("deepseek", "deepseek-v4-pro", "deepseek-v4pro-openai")


class TestIsRateLimit:
    @pytest.mark.parametrize(
        "text",
        [
            "[ab2e62a5] rate limited (429)",
            "429 Too Many Requests",
            "rate_limit_exceeded",
            "monthly quota exhausted",
            "TOO MANY REQUESTS",
        ],
    )
    def test_throttling_recognized(self, text):
        assert is_rate_limit(text) is True

    @pytest.mark.parametrize(
        "text",
        ["connection timeout", "invalid api key", "500 internal server error", "", None],
    )
    def test_other_failures_are_not_throttling(self, text):
        assert is_rate_limit(text) is False


class TestRotation:
    def test_round_robins_across_units_of_work(self):
        rotation = build_rotation([ZAI, KIMI, DEEPSEEK])
        seen = [rotation.next_spec().display() for _ in range(6)]
        assert (
            seen
            == [
                "zai-glm52-openai",
                "kimi",
                "deepseek-v4pro-openai",
            ]
            * 2
        )

    def test_duplicates_collapse(self):
        rotation = build_rotation([ZAI, ZAI, KIMI])
        assert len(rotation.specs) == 2

    def test_blank_entries_are_dropped(self):
        assert len(build_rotation([ZAI, ("", "", ""), ("zai", "", "")]).specs) == 1

    def test_empty_rotation_is_falsy(self):
        assert not build_rotation([])

    def test_a_throttled_provider_is_benched_for_the_run(self):
        rotation = build_rotation([ZAI, KIMI, DEEPSEEK])
        zai = rotation.specs[0]

        assert rotation.note_failure(zai, "rate limited (429)") is True

        later = {rotation.next_spec().display() for _ in range(6)}
        assert "zai-glm52-openai" not in later
        assert later == {"kimi", "deepseek-v4pro-openai"}

    def test_non_throttling_failure_keeps_the_provider(self):
        """A timeout says nothing about quota; benching on it strands the run."""
        rotation = build_rotation([ZAI, KIMI])
        assert rotation.note_failure(rotation.specs[0], "connection timeout") is False
        assert len(rotation.available) == 2

    def test_benching_is_idempotent(self):
        rotation = build_rotation([ZAI, KIMI])
        rotation.bench(rotation.specs[0], "429")
        rotation.bench(rotation.specs[0], "429")
        assert len(rotation.available) == 1

    def test_all_benched_returns_none_rather_than_falling_through(self):
        """None must mean stop, not 'use the session default'.

        Falling through to the default is how a 2B local model ended up
        rewriting production prompts unnoticed.
        """
        rotation = build_rotation([ZAI, KIMI])
        for spec in list(rotation.specs):
            rotation.bench(spec, "429")
        assert rotation.next_spec() is None

    def test_summary_names_active_and_benched(self):
        rotation = build_rotation([ZAI, KIMI])
        rotation.bench(rotation.specs[0], "429")
        summary = rotation.summary()
        assert "kimi" in summary
        assert "benched: zai-glm52-openai" in summary

    def test_summary_when_nothing_configured(self):
        assert "no mutator rotation" in MutatorRotation(specs=[]).summary()

    def test_display_falls_back_to_provider_model(self):
        assert MutatorSpec("zai", "glm-5.2").display() == "zai/glm-5.2"
        assert MutatorSpec("zai", "glm-5.2", label="lbl").display() == "lbl"


class TestBenchingAndRetryingAreSeparateQuestions:
    """A timeout should move the call without writing the provider off.

    Conflating the two cost a real reflection: a 420s reasoning call died on
    "sandhi transport timed out", which is correctly not grounds to bench a
    provider — a timeout says nothing about quota — but is exactly when a peer is
    likely to answer. The failover declined, and the reflection was lost.
    """

    @pytest.mark.parametrize(
        "error",
        ["sandhi transport timed out: ", "connection reset by peer", "503 service unavailable"],
    )
    def test_transient_failures_are_worth_a_peer_but_not_a_bench(self, error):
        assert is_worth_another_provider(error) is True
        assert is_rate_limit(error) is False

        rotation = build_rotation([ZAI, KIMI])
        assert rotation.note_failure(rotation.specs[0], error) is False
        assert len(rotation.available) == 2, "a timeout must not bench the provider"

    @pytest.mark.parametrize("error", ["invalid api key", "400 malformed request"])
    def test_our_own_defects_are_worth_neither(self, error):
        assert is_worth_another_provider(error) is False

    def test_a_throttle_is_worth_both(self):
        assert is_worth_another_provider("rate limited (429)") is True
        assert is_rate_limit("rate limited (429)") is True

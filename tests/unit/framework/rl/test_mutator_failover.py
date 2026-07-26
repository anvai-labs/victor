# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
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

"""A throttled mutator call must move to another provider, not give up.

Rotating per section looked sufficient until a real run disproved it: one
section makes several calls back to back, so the first 429 still cost the whole
section and the rotation never even learned about it — ``_call_llm`` caught the
exception and returned None, so nothing above the transport could react.
"""

from typing import Any, List, Optional, Tuple

import pytest

from victor.framework.rl.gepa_service import GEPAService
from victor.framework.rl.mutator_rotation import MutatorSpec, build_rotation

ZAI = ("zai", "glm-5.2", "zai-glm52-openai")
KIMI = ("moonshot", "kimi-k3", "kimi")
DEEPSEEK = ("deepseek", "deepseek-v4-pro", "deepseek-v4pro-openai")

LONG_ENOUGH = "a mutated prompt section that clears the twenty character floor"


class _Response:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeProvider:
    """Answers per model: an Exception instance is raised, a str is returned."""

    def __init__(self, name: str, answers: dict) -> None:
        self.name = name
        self._answers = answers
        self.calls: List[str] = []

    async def chat(self, *, messages: Any, model: str, **kwargs: Any) -> _Response:
        del messages, kwargs
        self.calls.append(model)
        answer = self._answers.get(model, LONG_ENOUGH)
        if isinstance(answer, Exception):
            raise answer
        return _Response(answer)


def make_service(provider: Any, model: str, failover: Optional[Any] = None) -> GEPAService:
    return GEPAService(provider=provider, model=model, tier="balanced", failover=failover)


class TestFailoverTriggering:
    def test_a_throttled_call_retries_on_the_next_provider(self):
        throttled = FakeProvider("zai", {"glm-5.2": RuntimeError("rate limited (429)")})
        healthy = FakeProvider("moonshot", {})
        service = make_service(throttled, "glm-5.2", lambda m, e: (healthy, "kimi-k3"))

        assert service.reflect("traces", "COMPLETION_GUIDANCE", "current") == LONG_ENOUGH
        assert healthy.calls == ["kimi-k3"]

    def test_a_non_throttling_failure_does_not_burn_another_provider(self):
        """A bad key or a bug here fails identically everywhere."""
        broken = FakeProvider("zai", {"glm-5.2": RuntimeError("invalid api key")})
        healthy = FakeProvider("moonshot", {})
        service = make_service(broken, "glm-5.2", lambda m, e: (healthy, "kimi-k3"))

        assert "Reflection unavailable" in service.reflect("t", "COMPLETION_GUIDANCE", "c")
        assert healthy.calls == []

    def test_an_empty_answer_tries_a_peer_only_after_the_budget_escalation(self):
        """Order matters, and it is measured rather than assumed.

        A bigger budget on the same provider is the cheaper fix and the more
        common cause, so it goes first. But an empty answer *is* model-specific:
        on one mutate prompt deepseek-v4-pro returns 0 characters where kimi-k3
        returns a full 1624-character rewrite at the original budget. Giving up
        without trying the peer wastes the section.
        """
        from victor.framework.rl.gepa_service import REASONING_TOKEN_BUDGET

        budgets: List[int] = []

        class AlwaysEmpty:
            async def chat(self, *, messages, model, max_tokens, **kw):
                del messages, model, kw
                budgets.append(max_tokens)
                return _Response("")

        healthy = FakeProvider("moonshot", {})
        service = GEPAService(
            provider=AlwaysEmpty(),
            model="glm-5.2",
            failover=lambda m, e: (healthy, "kimi-k3"),
            max_tokens=1000,
        )

        assert service.mutate("current text", "reflection", "COMPLETION") == LONG_ENOUGH
        assert budgets == [1000, REASONING_TOKEN_BUDGET], "escalate before switching"
        assert healthy.calls == ["kimi-k3"]

    def test_an_empty_answer_gives_up_when_there_is_no_peer(self):
        terse = FakeProvider("zai", {"glm-5.2": "too short"})
        assert make_service(terse, "glm-5.2").mutate("current text", "r", "S") == "current text"

    def test_no_failover_configured_degrades_to_the_old_behaviour(self):
        throttled = FakeProvider("zai", {"glm-5.2": RuntimeError("429")})
        assert make_service(throttled, "glm-5.2").mutate("current", "r", "S") == "current"

    def test_exhausted_failover_stops_rather_than_looping(self):
        throttled = FakeProvider("zai", {"glm-5.2": RuntimeError("429 quota")})
        service = make_service(throttled, "glm-5.2", lambda m, e: None)
        assert "Reflection unavailable" in service.reflect("t", "S", "c")

    def test_paid_attempts_are_bounded(self):
        """A failover that always answers must not spin.

        ``max_failovers`` bounds the calls that cost money. Lookups run one
        higher, because the lookup after the *last* attempt is what reports that
        failure — the rotation needs it to bench the provider for later sections
        even though the replacement it returns goes unused.
        """
        always_429 = FakeProvider("any", {})
        always_429._answers = {f"model-{i}": RuntimeError("429") for i in range(1, 6)}
        lookups: List[str] = []

        def failover(model: str, error: BaseException) -> Tuple[Any, str]:
            lookups.append(model)
            return always_429, f"model-{len(lookups)}"

        first = FakeProvider("zai", {"glm-5.2": RuntimeError("429")})
        service = GEPAService(provider=first, model="glm-5.2", failover=failover, max_failovers=2)

        assert "Reflection unavailable" in service.reflect("t", "S", "c")
        assert len(first.calls) + len(always_429.calls) == 3, "1 + max_failovers attempts"
        assert lookups == ["glm-5.2", "model-1", "model-2"]

    def test_a_broken_failover_does_not_mask_the_failure(self):
        def exploding(model: str, error: BaseException) -> Tuple[Any, str]:
            raise ValueError("failover itself is broken")

        service = make_service(
            FakeProvider("zai", {"glm-5.2": RuntimeError("429")}), "glm-5.2", exploding
        )
        assert "Reflection unavailable" in service.reflect("t", "S", "c")


class TestRotationFailover:
    """The adapter that turns a MutatorRotation into a GEPAService failover."""

    @staticmethod
    def _failover(rotation, build):
        from victor.framework.rl.gepa_tier_manager import _RotationFailover

        return _RotationFailover(rotation, build, "zai", "glm-5.2")

    def test_it_benches_the_live_provider_and_hands_back_the_next(self):
        rotation = build_rotation([ZAI, KIMI])
        built: List[str] = []

        def build(provider: str, base_url: str) -> Any:
            built.append(provider)
            return FakeProvider(provider, {})

        provider, model = self._failover(rotation, build)("glm-5.2", RuntimeError("429"))

        assert model == "kimi-k3"
        assert built == ["moonshot"]
        assert [s.provider for s in rotation.available] == ["moonshot"]

    def test_the_second_failure_benches_the_provider_that_actually_failed(self):
        """It must not keep benching whichever provider it started on."""
        rotation = build_rotation([ZAI, KIMI, DEEPSEEK])
        failover = self._failover(rotation, lambda p, u: FakeProvider(p, {}))

        failover("glm-5.2", RuntimeError("429"))
        failover("kimi-k3", RuntimeError("429"))

        assert [s.provider for s in rotation.available] == ["deepseek"]

    def test_an_unbuildable_provider_is_benched_not_returned_forever(self):
        rotation = build_rotation([ZAI, KIMI, DEEPSEEK])

        def build(provider: str, base_url: str) -> Optional[Any]:
            return None if provider == "moonshot" else FakeProvider(provider, {})

        provider, model = self._failover(rotation, build)("glm-5.2", RuntimeError("429"))

        assert model == "deepseek-v4-pro"
        assert [s.provider for s in rotation.available] == ["deepseek"]

    def test_returns_none_when_nothing_can_be_built(self):
        rotation = build_rotation([ZAI, KIMI])
        failover = self._failover(rotation, lambda p, u: None)
        assert failover("glm-5.2", RuntimeError("429")) is None

    def test_end_to_end_a_429_moves_the_call_to_the_rotation_peer(self):
        """The whole seam: rotation -> failover -> service retry."""
        rotation = build_rotation([ZAI, KIMI])
        healthy = FakeProvider("moonshot", {})
        failover = self._failover(rotation, lambda p, u: healthy)
        service = make_service(
            FakeProvider("zai", {"glm-5.2": RuntimeError("rate limited (429)")}),
            "glm-5.2",
            failover,
        )

        assert service.mutate("current text", "reflection", "COMPLETION") == LONG_ENOUGH
        assert healthy.calls == ["kimi-k3"]


class TestEmptyAnswerEscalatesTheBudget:
    """A reasoning model can spend the whole budget thinking and emit nothing.

    Verified against deepseek-v4-pro: the identical mutate prompt returns 0
    characters at 1000 max_tokens and a real 638-character rewrite at 4000. GEPA's
    default predates reasoning models, so this read as "no improvement offered".
    """

    class BudgetSensitiveProvider:
        """Returns content only once the budget clears a threshold."""

        def __init__(self, needs: int) -> None:
            self._needs = needs
            self.budgets: List[int] = []

        async def chat(self, *, messages: Any, model: str, max_tokens: int, **kw: Any) -> _Response:
            del messages, model, kw
            self.budgets.append(max_tokens)
            return _Response(LONG_ENOUGH if max_tokens >= self._needs else "")

    def test_a_second_attempt_gets_a_reasoning_sized_budget(self):
        from victor.framework.rl.gepa_service import REASONING_TOKEN_BUDGET

        provider = self.BudgetSensitiveProvider(needs=4000)
        service = GEPAService(provider=provider, model="deepseek-v4-pro", max_tokens=1000)

        assert service.mutate("current", "reflection", "COMPLETION") == LONG_ENOUGH
        assert provider.budgets == [1000, REASONING_TOKEN_BUDGET]

    def test_the_deadline_grows_with_the_budget(self):
        """A 16k-token reasoning call takes minutes; 120s would abort it.

        Without this the escalation is just a slower way to fail: the retry gets
        the tokens it needs and then gets killed before answering.
        """
        from victor.framework.rl.gepa_service import REASONING_TIMEOUT_S

        deadlines: List[float] = []

        class SlowProvider:
            async def chat(self, *, messages, model, max_tokens, **kw):
                del messages, model, kw
                return _Response("" if max_tokens < 4000 else LONG_ENOUGH)

        service = GEPAService(provider=SlowProvider(), model="m", max_tokens=1000, timeout_s=120.0)
        original = service._attempt_call

        def spy(system, user, budget, timeout_s=None):
            deadlines.append(timeout_s)
            return original(system, user, budget, timeout_s)

        service._attempt_call = spy  # type: ignore[method-assign]

        assert service.mutate("current", "reflection", "COMPLETION") == LONG_ENOUGH
        assert deadlines == [120.0, REASONING_TIMEOUT_S]
        assert REASONING_TIMEOUT_S > 120.0

    def test_escalation_happens_once(self):
        """Two empty answers means the prompt, not the budget."""
        provider = self.BudgetSensitiveProvider(needs=10**9)
        service = GEPAService(provider=provider, model="m", max_tokens=1000)

        assert service.mutate("current", "reflection", "COMPLETION") == "current"
        assert len(provider.budgets) == 2

    def test_an_already_generous_budget_is_not_re_escalated(self):
        from victor.framework.rl.gepa_service import REASONING_TOKEN_BUDGET

        provider = self.BudgetSensitiveProvider(needs=10**9)
        service = GEPAService(provider=provider, model="m", max_tokens=REASONING_TOKEN_BUDGET)

        service.mutate("current", "reflection", "COMPLETION")
        assert provider.budgets == [REASONING_TOKEN_BUDGET]

    def test_escalation_does_not_consume_a_failover(self):
        """The two recoveries are independent: an empty answer then a 429."""
        calls: List[int] = []

        class EmptyThen429:
            async def chat(self, *, messages, model, max_tokens, **kw):
                del messages, model, kw
                calls.append(max_tokens)
                if len(calls) == 1:
                    return _Response("")
                raise RuntimeError("rate limited (429)")

        healthy = FakeProvider("moonshot", {})
        service = GEPAService(
            provider=EmptyThen429(),
            model="glm-5.2",
            failover=lambda m, e: (healthy, "kimi-k3"),
            max_tokens=1000,
        )

        assert service.mutate("current", "reflection", "COMPLETION") == LONG_ENOUGH
        assert healthy.calls == ["kimi-k3"]


class TestRejectionIsVisible:
    """A discarded candidate must say so, at a level the CLI shows.

    Both hygiene gates returned ``current_text`` while logging at INFO, so a
    mutation the provider had already been paid for vanished into a bare "no
    change" row — indistinguishable from a model with nothing to offer. That is
    the same defect as the 429 being logged at DEBUG.
    """

    def test_structural_rejection_warns_and_names_the_gate(self, caplog):
        bloated = "\n".join(f"- an entirely new rule number {i}" for i in range(400))
        service = make_service(FakeProvider("zai", {"m": bloated}), "m")

        with caplog.at_level("WARNING"):
            out = service.mutate(
                "Rules:\n\n- keep it short", "reflection", "COMPLETION_GUIDANCE", 10**6
            )

        assert out == "Rules:\n\n- keep it short"
        assert "COMPLETION_GUIDANCE" in caplog.text
        assert "hygiene" in caplog.text.lower()
        assert "unchanged" in caplog.text

    def test_prefpo_hygiene_rejection_warns(self, caplog):
        from unittest.mock import patch

        from victor.framework.rl.learners.strategies.prefpo_strategy import PrefPOStrategy

        strategy = PrefPOStrategy()
        report = type("R", (), {"accepted": False, "violations": ["unsupported_additions"]})()

        with (
            patch.object(strategy, "_challenger_factory", return_value="challenger text"),
            patch.object(
                strategy,
                "_judge",
                return_value=("challenger", "Prefer challenger because it adds:\n- x"),
            ),
            patch.object(strategy, "_optimizer", return_value="candidate text"),
            patch(
                "victor.framework.rl.learners.strategies.prefpo_strategy.evaluate_prompt_candidate",
                return_value=report,
            ),
        ):
            with caplog.at_level("WARNING"):
                assert strategy.reflect([object()], "COMPLETION_GUIDANCE", "current") == ""

        assert "PrefPO rejected" in caplog.text
        assert "unsupported_additions" in caplog.text


class TestTierManagerWiring:
    def test_no_rotation_means_no_failover_is_attached(self):
        from victor.framework.rl.gepa_tier_manager import GEPATierManager

        manager = GEPATierManager(config=_Config())
        assert manager._make_failover("zai", "glm-5.2") is None

    def test_arming_a_rotation_invalidates_cached_services(self):
        """A cached service holds a stale failover of None."""
        from victor.framework.rl.gepa_tier_manager import GEPATierManager

        manager = GEPATierManager(config=_Config())
        manager._services["balanced"] = object()
        manager.set_mutator_rotation(build_rotation([ZAI, KIMI]))

        assert manager._services == {}
        assert manager._make_failover("zai", "glm-5.2") is not None


class _Config:
    use_main_model = False
    default_tier = "balanced"
    auto_tier_switch = False
    max_prompt_chars = 1500


@pytest.mark.parametrize(
    "spec,expected",
    [(MutatorSpec("zai", "glm-5.2"), "zai/glm-5.2"), (MutatorSpec("a", "b", "lbl"), "lbl")],
)
def test_spec_display(spec, expected):
    assert spec.display() == expected

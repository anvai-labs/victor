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

"""The control plane may only carry what the model cannot already derive.

Co-design rule, applied to the runtime→model boundary: a signal earns its place
only if it informs a decision the consumer could not make from what it already
has. Restating something the model can see is not neutral — it costs tokens on
every turn, and it introduces a second copy that can disagree with the first.

That disagreement is not hypothetical. In session ``sandhi-cdfbc589`` the
``progress`` reminder reported "3 tools used" after eight calls, because it
appended one name per *batch* while the counter advanced per *call*. The model
compared it against the tool calls in its own context, found it false, and
treated the mismatch as evidence of tampering. A redundant signal was strictly
worse than no signal: it could only ever agree with the wire (adding nothing) or
disagree with it (actively misleading).

Each reminder type must therefore declare what it is for, and types whose
information is already on the wire must not be emitted at all.
"""

from __future__ import annotations

from victor.agent.context_reminder import (
    REMINDER_DECISIONS,
    RETIRED_REMINDERS,
    ContextReminderManager,
    ReminderType,
)


class TestEveryReminderDeclaresItsPurpose:
    """A signal with no stated consumer decision cannot be justified."""

    def test_every_type_is_either_justified_or_explicitly_retired(self):
        """No silent omissions: a new type must argue its case either way."""
        unaccounted = [
            rt.value
            for rt in ReminderType
            if rt is not ReminderType.CUSTOM
            and rt not in REMINDER_DECISIONS
            and rt not in RETIRED_REMINDERS
        ]

        assert not unaccounted, (
            f"reminder types accounted for in neither contract: {unaccounted}. "
            "Add to REMINDER_DECISIONS with the decision it informs, or to "
            "RETIRED_REMINDERS with why the model already has the information."
        )

    def test_the_two_contracts_are_disjoint(self):
        both = [rt.value for rt in REMINDER_DECISIONS if rt in RETIRED_REMINDERS]

        assert not both, f"declared as both emitted and retired: {both}"

    def test_retired_types_state_where_the_information_already_is(self):
        thin = [rt.value for rt, why in RETIRED_REMINDERS.items() if len(why.strip()) < 20]

        assert not thin, f"retirement without a stated reason: {thin}"

    def test_declared_decisions_are_substantive(self):
        thin = [rt.value for rt, why in REMINDER_DECISIONS.items() if len(why.strip()) < 20]

        assert not thin, f"placeholder decisions, not real ones: {thin}"


class TestRedundantSignalsAreNotEmitted:
    """Information already on the wire must not be restated."""

    def _manager(self) -> ContextReminderManager:
        manager = ContextReminderManager(provider="zai")
        manager.update_state(
            observed_files={"a.py", "b.py"},
            executed_tools=["read", "code", "shell"],
            tool_calls=8,
            tool_budget=20,
        )
        return manager

    def test_progress_is_not_emitted(self):
        """The model's own tool calls and results are already in its context.

        `openai_compat.build_openai_messages` serialises `msg.tool_calls` and the
        matching tool results, so a per-turn count restates what is visible — and
        did so incorrectly.
        """
        manager = self._manager()

        assert manager.get_reminder(ReminderType.PROGRESS) is None
        assert manager.should_inject_reminder(ReminderType.PROGRESS) is False

    def test_grounding_is_not_emitted(self):
        """GROUNDING_RULES already ships in the system prompt, every turn."""
        manager = self._manager()

        assert manager.get_reminder(ReminderType.GROUNDING) is None
        assert manager.should_inject_reminder(ReminderType.GROUNDING) is False

    def test_consolidated_reminder_carries_neither(self):
        manager = self._manager()

        reminder = manager.get_consolidated_reminder(force=True) or ""

        assert "Progress:" not in reminder
        assert "tools used" not in reminder
        assert "Ground responses in tool output only" not in reminder


class TestNonDerivableSignalsSurvive:
    """Narrowing must not silence the signals that do earn their place."""

    def test_evidence_still_reported(self):
        """Files read in earlier turns can fall out of a compacted context."""
        manager = ContextReminderManager(provider="zai")
        manager.update_state(observed_files={"crates/sandhi-proxy/src/lib.rs"}, tool_calls=3)

        assert "lib.rs" in (manager.get_reminder(ReminderType.EVIDENCE) or "")

    def test_budget_still_reported(self):
        """The enforcer's counter is not visible to the model at all."""
        manager = ContextReminderManager(provider="zai")
        manager.update_state(tool_calls=18, tool_budget=20)

        assert "2 tool calls remaining" in (manager.get_reminder(ReminderType.BUDGET) or "")

    def test_compaction_still_reported(self):
        """By definition the model cannot see what was removed from its context."""
        manager = ContextReminderManager(provider="zai")
        manager.update_state(tool_calls=1)
        manager.state.compaction_summary = "earlier turns summarised"

        assert "compacted" in (manager.get_reminder(ReminderType.COMPACTION) or "").lower()

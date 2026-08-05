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

"""FEP-0023 Phase 2: ToolResultDeduplicator as an assembler *view* stage.

The load-bearing property: dedup runs on the assembled context copy and NEVER
mutates the source-of-truth message history. These tests pin that invariant plus
the actual token-saving behaviour and the passthrough-when-unwired contract.
"""

from victor.agent.conversation.assembler import TurnBoundaryContextAssembler
from victor.agent.tool_result_deduplicator import ToolResultDeduplicator
from victor.providers.base import Message


def _read_output(path: str, filler: str) -> str:
    # >500 chars so it clears min_content_chars_to_dedup.
    body = (filler * 700)[:700]
    return f'<TOOL_OUTPUT tool="read" path="{path}">{body}</TOOL_OUTPUT>'


def _duplicate_read_history():
    dup = _read_output("/src/config.py", "x")
    return [
        Message(role="user", content=dup),  # older read (should be stubbed in the view)
        Message(role="assistant", content="I looked at the config."),
        Message(role="user", content="read it again"),
        Message(role="user", content=dup),  # newer identical read (kept)
    ]


class TestDedupViewStage:
    def test_view_dedups_identical_reads(self):
        assembler = TurnBoundaryContextAssembler(tool_result_deduplicator=ToolResultDeduplicator())
        out = assembler.assemble(_duplicate_read_history(), max_context_chars=100_000)

        stubs = [m for m in out if m.content.startswith("[Duplicate tool output")]
        assert len(stubs) == 1, f"expected exactly one stubbed duplicate, got {len(stubs)}"
        # The newest full copy is retained.
        assert sum(1 for m in out if "<TOOL_OUTPUT" in m.content) == 1

    def test_source_history_is_never_mutated(self):
        """The critical invariant: assembling must not touch the input list/objects."""
        messages = _duplicate_read_history()
        snapshot = [(id(m), m.role, m.content) for m in messages]

        assembler = TurnBoundaryContextAssembler(tool_result_deduplicator=ToolResultDeduplicator())
        assembler.assemble(messages, max_context_chars=100_000)

        after = [(id(m), m.role, m.content) for m in messages]
        assert after == snapshot, "assemble() mutated the source history (list or Message objects)"
        # No stubs leaked into the source.
        assert not any(m.content.startswith("[Duplicate") for m in messages)

    def test_passthrough_when_no_deduplicator(self):
        """Without a wired deduplicator (flag OFF), duplicates survive unchanged."""
        assembler = TurnBoundaryContextAssembler(tool_result_deduplicator=None)
        out = assembler.assemble(_duplicate_read_history(), max_context_chars=100_000)

        assert not any(m.content.startswith("[Duplicate") for m in out)
        assert sum(1 for m in out if "<TOOL_OUTPUT" in m.content) == 2

    def test_view_dedup_reduces_total_chars(self):
        history = _duplicate_read_history()
        base = TurnBoundaryContextAssembler(tool_result_deduplicator=None).assemble(
            history, max_context_chars=100_000
        )
        deduped = TurnBoundaryContextAssembler(
            tool_result_deduplicator=ToolResultDeduplicator()
        ).assemble(history, max_context_chars=100_000)

        assert sum(len(m.content) for m in deduped) < sum(len(m.content) for m in base)


class TestDeduplicateHistoryView:
    def test_returns_copy_and_leaves_input_untouched(self):
        messages = _duplicate_read_history()
        before = [(m.role, m.content) for m in messages]

        new_list, stubbed = ToolResultDeduplicator().deduplicate_history_view(messages)

        assert stubbed == 1
        assert new_list is not messages
        assert [(m.role, m.content) for m in messages] == before  # input untouched
        assert any(m.content.startswith("[Duplicate") for m in new_list)

    def test_disabled_config_is_passthrough(self):
        from victor.config.orchestrator_constants import DeduplicationConfig

        dedup = ToolResultDeduplicator(config=DeduplicationConfig(enabled=False))
        messages = _duplicate_read_history()
        new_list, stubbed = dedup.deduplicate_history_view(messages)

        assert stubbed == 0
        assert [(m.role, m.content) for m in new_list] == [(m.role, m.content) for m in messages]


class TestFactoryWiring:
    def test_create_context_assembler_threads_deduplicator(self):
        """The factory passes a supplied deduplicator through to the assembler.

        component_assembler supplies one only when USE_TOOL_RESULT_DEDUP is on;
        this pins the plumbing so 'flag on -> view stage active' holds.
        """
        from victor.agent.factory.coordination_builders import CoordinationBuildersMixin

        class _Builder(CoordinationBuildersMixin):
            pass

        dedup = ToolResultDeduplicator()
        assembler = _Builder().create_context_assembler(
            ledger=None, controller=None, tool_result_deduplicator=dedup
        )
        assert assembler._tool_result_deduplicator is dedup

        # Default (no deduplicator) leaves the view stage inert.
        plain = _Builder().create_context_assembler(ledger=None, controller=None)
        assert plain._tool_result_deduplicator is None

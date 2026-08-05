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

"""L1 reference-aware pruning: prune only old, large, *unreferenced* tool results.

Safe-by-construction contract: a tool result is preserved if it is recent (within the window)
OR if any later assistant turn cites a distinctive anchor (file/symbol/path) from it. Only the
remainder — old, large, never-cited — is replaced with a marker. Default OFF changes nothing.
"""

from __future__ import annotations

from victor.agent.fast_pruning import FastPruner, FastPruningConfig
from victor.providers.base import Message


def _tool(content: str, call_id: str = "c") -> Message:
    return Message(role="tool", content=content, tool_call_id=call_id)


def _assistant(content: str) -> Message:
    return Message(role="assistant", content=content)


def _user(content: str) -> Message:
    return Message(role="user", content=content)


# A large blob with a distinctive anchor token the assistant may later cite.
def _blob(anchor: str) -> str:
    return f"def {anchor}():\n" + ("    # filler line of output\n" * 200)


def _ref_pruner(window: int = 1) -> FastPruner:
    return FastPruner(
        FastPruningConfig(enable_reference_tracking=True, reference_window_turns=window)
    )


def test_unreferenced_old_tool_result_is_pruned():
    msgs = [
        _user("look at the helper"),
        _tool(_blob("orphaned_helper")),  # old, large, never cited again
        _assistant("I read it."),  # does NOT cite the anchor
        _tool(_blob("recent_result"), "c2"),  # recent -> protected by window
    ]
    out = _ref_pruner(window=1).prune_old_tool_results(msgs, current_turn=len(msgs))
    assert out[1].content.startswith("[pruned]")
    assert out[3].content == msgs[3].content  # recent one untouched


def test_referenced_tool_result_is_preserved():
    msgs = [
        _user("find the parser"),
        _tool(_blob("parse_tokens")),  # large + old
        _assistant("The function parse_tokens looks buggy, let me fix it."),  # cites anchor
        _tool(_blob("other"), "c2"),
    ]
    out = _ref_pruner(window=1).prune_old_tool_results(msgs, current_turn=len(msgs))
    # Cited by a later assistant message -> must be kept verbatim.
    assert out[1].content == msgs[1].content


def test_recency_window_always_preserved():
    msgs = [
        _tool(_blob("first"), "c1"),
        _assistant("noted"),
        _tool(_blob("second"), "c2"),
        _tool(_blob("third"), "c3"),
    ]
    # Window of 2 protects the last two tool results regardless of references.
    out = _ref_pruner(window=2).prune_old_tool_results(msgs, current_turn=len(msgs))
    assert out[2].content == msgs[2].content
    assert out[3].content == msgs[3].content
    # The first (out of window, uncited) is pruned.
    assert out[0].content.startswith("[pruned]")


def test_small_tool_results_not_pruned():
    msgs = [
        _tool("ok", "c1"),  # tiny -> not worth pruning even if uncited
        _assistant("done"),
        _tool(_blob("recent"), "c2"),
    ]
    out = _ref_pruner(window=1).prune_old_tool_results(msgs, current_turn=len(msgs))
    assert out[0].content == "ok"


def test_pruned_marker_preserves_tool_call_id_pairing():
    msgs = [
        _tool(_blob("zzz"), "call_42"),
        _assistant("moving on"),
        _tool(_blob("recent"), "c2"),
    ]
    out = _ref_pruner(window=1).prune_old_tool_results(msgs, current_turn=len(msgs))
    assert out[0].content.startswith("[pruned]")
    # Provider pairing requires the tool_call_id to survive pruning.
    assert out[0].tool_call_id == "call_42"


def test_flag_off_is_identical_to_size_based_path():
    # With reference tracking disabled, behavior is the legacy size-based pruning (unchanged).
    msgs = [
        _tool(_blob("anything")),
        _assistant("x"),
    ]
    off = FastPruner(FastPruningConfig(enable_reference_tracking=False))
    out = off.prune_old_tool_results(msgs, current_turn=len(msgs))
    # Legacy path prunes large tool results by size threshold (default 1000).
    assert out[0].content.startswith("[pruned]")


def test_estimate_matches_reference_aware_pruning():
    msgs = [
        _tool(_blob("orphan")),
        _assistant("I read it."),
        _tool(_blob("recent"), "c2"),
    ]
    pruner = _ref_pruner(window=1)
    original, estimated = pruner.estimate_size_reduction(msgs, current_turn=len(msgs))
    assert estimated < original  # the orphan is estimated as prunable
    # And the estimate's prune decision matches the actual prune.
    out = pruner.prune_old_tool_results(msgs, current_turn=len(msgs))
    assert out[0].content.startswith("[pruned]")
    assert out[2].content == msgs[2].content


def test_reference_aware_pruning_is_on_by_default():
    # The flip: evidence showed L1 is lossless (0 cited dropped) and recovers most savings in
    # the realistic read-many/cite-few regime, so it is now the default fast-pruning path.
    from victor.agent.context_compactor import CompactorConfig
    from victor.config.orchestrator_constants import CompactionConfig

    assert CompactionConfig().enable_reference_aware_pruning is True
    assert CompactorConfig().enable_reference_aware_pruning is True


def test_default_pruner_preserves_cited_results_losslessly():
    # End-to-end with the shipped default window: a cited result survives, an uncited one is
    # pruned. This is the property that makes the default-on flip safe.
    cfg = FastPruningConfig(enable_reference_tracking=True, reference_window_turns=1)
    msgs = [
        _tool(_blob("cited_unit")),
        _tool(_blob("orphan_unit")),
        _assistant("cited_unit is the one that matters."),
        _tool(_blob("recent"), "c_recent"),
    ]
    out = FastPruner(cfg).prune_old_tool_results(msgs, current_turn=len(msgs))
    assert out[0].content == msgs[0].content  # cited -> preserved (lossless)
    assert out[1].content.startswith("[pruned]")  # uncited -> pruned

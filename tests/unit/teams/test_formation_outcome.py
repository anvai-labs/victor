"""Per-formation aggregation of member results and the paused-run status."""

from __future__ import annotations

import pytest

from victor.teams.formation_outcome import aggregate_member_results, apply_pause_status
from victor.teams.types import MemberResult, TeamFormation


def _r(member_id, success=True, output="", tool_calls=0, error=None, **metadata):
    return MemberResult(
        member_id=member_id,
        success=success,
        output=output,
        error=error,
        metadata=metadata,
        tool_calls_used=tool_calls,
    )


class TestAggregation:
    def test_no_members_is_a_failure(self):
        assert aggregate_member_results(TeamFormation.SEQUENTIAL, [], {}) == (False, "", 0)

    def test_sequential_needs_every_member_and_joins_outputs(self):
        results = [_r("a", output="one", tool_calls=2), _r("b", output="two", tool_calls=3)]
        assert aggregate_member_results(TeamFormation.SEQUENTIAL, results, {}) == (
            True,
            "one\n\ntwo",
            5,
        )
        results.append(_r("c", success=False, output="ignored"))
        success, output, _ = aggregate_member_results(TeamFormation.SEQUENTIAL, results, {})
        assert success is False and output == "one\n\ntwo"

    def test_pipeline_keeps_only_the_last_stage(self):
        results = [_r("a", output="draft"), _r("b", output="final")]
        assert aggregate_member_results(TeamFormation.PIPELINE, results, {})[1] == "final"

    def test_parallel_succeeds_on_any_member_and_lists_failures(self):
        results = [_r("a", output="ok"), _r("b", success=False, error="boom"), _r("c", False)]
        success, output, _ = aggregate_member_results(TeamFormation.PARALLEL, results, {})
        assert success is True
        assert output == "ok\n\nMember failures:\n- b: boom\n- c: execution failed"
        all_failed = [_r("a", False, error="x")]
        assert aggregate_member_results(TeamFormation.PARALLEL, all_failed, {}) == (
            False,
            "Member failures:\n- a: x",
            0,
        )

    @pytest.mark.parametrize(
        "metadata,expected",
        [
            ({"consensus_achieved": True, "consensus_decision": "ship"}, (True, "ship")),
            ({"consensus_achieved": False}, (False, "x")),
            ({"consensus_decision": "maybe"}, (True, "maybe")),
        ],
    )
    def test_consensus_requires_agreement(self, metadata, expected):
        results = [_r("a", output="x", **metadata)]
        success, output, _ = aggregate_member_results(TeamFormation.CONSENSUS, results, {})
        assert (success, output) == expected

    def test_ensemble_and_conversation_metadata_override_output(self):
        ens = [
            _r("a", output="x", ensemble_success=True, ensemble_decision="d", ensemble_error="e")
        ]
        assert aggregate_member_results(TeamFormation.SEQUENTIAL, ens, {})[:2] == (True, "d")
        ens_fail = [
            _r("a", output="x", ensemble_success=False, ensemble_decision="d", ensemble_error="e")
        ]
        assert aggregate_member_results(TeamFormation.SEQUENTIAL, ens_fail, {})[:2] == (False, "e")
        conv = [_r("a", conversation_success=True, conversation_output="chat")]
        assert aggregate_member_results(TeamFormation.SEQUENTIAL, conv, {})[:2] == (True, "chat")

    def test_reflection_metadata_only_counts_with_usage_capture(self):
        results = [_r("a", output="x", reflection_success=True, reflection_output="refined")]
        off = aggregate_member_results(TeamFormation.REFLECTION, results, {})
        on = aggregate_member_results(
            TeamFormation.REFLECTION, results, {"capture_member_usage": True}
        )
        assert off[1] == "x" and on[1] == "refined"


class TestPauseStatus:
    def test_single_member_pause(self):
        result = {"success": True}
        apply_pause_status(
            result,
            {"__awaiting_approval__": {"member_id": "m1", "approval_request": {"tool": "w"}}},
            "t-1",
        )
        assert result == {
            "success": False,
            "status": "awaiting_approval",
            "paused_member_id": "m1",
            "approval_request": {"tool": "w"},
            "thread_id": "t-1",
        }

    def test_multi_member_pause_without_thread(self):
        result = {"success": True}
        apply_pause_status(
            result, {"__awaiting_approvals__": [{"member_id": "a"}, {"member_id": "b"}]}, None
        )
        assert result["paused_member_ids"] == ["a", "b"] and "thread_id" not in result
        assert result["success"] is False and result["status"] == "awaiting_approval"

    def test_no_pause_leaves_result_untouched(self):
        result = {"success": True}
        apply_pause_status(result, {}, "t")
        assert result == {"success": True}

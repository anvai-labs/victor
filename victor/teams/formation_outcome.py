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

"""Pure aggregation of a team formation's member results.

Extracted from ``UnifiedTeamCoordinator._execute_formation`` (co-design backlog
item 30): the per-formation success/output rules and the paused-run status,
as functions with no coordinator state.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

from victor.teams.types import MemberResult, TeamFormation

__all__ = ["aggregate_member_results", "apply_pause_status"]


def aggregate_member_results(
    active_formation: TeamFormation,
    member_results_list: List[MemberResult],
    effective_context: Mapping[str, Any],
) -> Tuple[bool, str, int]:
    """Return ``(success, final_output, total_tool_calls)`` for a formation run."""
    # Build final output
    if not member_results_list:
        success = False
    elif active_formation == TeamFormation.PARALLEL:
        # PARALLEL members are independent by contract (redundancy,
        # diverse perspectives): the team succeeds when at least one
        # member delivered. Failed members stay visible in
        # member_results instead of failing the whole team.
        success = any(r.success for r in member_results_list)
    else:
        success = all(r.success for r in member_results_list)
    final_outputs = [r.output for r in member_results_list if r.success]
    total_tool_calls = sum(r.tool_calls_used for r in member_results_list)

    # Determine final output based on formation
    # For pipeline, use only the last stage's output
    # For other formations, join all outputs
    if active_formation == TeamFormation.PIPELINE and final_outputs:
        final_output = final_outputs[-1]  # Last stage's output only
    else:
        final_output = "\n\n".join(final_outputs)

    if active_formation == TeamFormation.PARALLEL:
        failed = [result for result in member_results_list if not result.success]
        if failed:
            summary = "Member failures:\n" + "\n".join(
                f"- {result.member_id}: {result.error or 'execution failed'}" for result in failed
            )
            final_output = f"{final_output}\n\n{summary}" if final_output else summary
    if active_formation == TeamFormation.CONSENSUS and member_results_list:
        metadata = member_results_list[0].metadata
        success = success and bool(
            metadata.get("consensus_achieved") or "consensus_decision" in metadata
        )
        if "consensus_decision" in metadata:
            final_output = metadata["consensus_decision"]

    if member_results_list and "ensemble_success" in member_results_list[0].metadata:
        metadata = member_results_list[0].metadata
        success = bool(metadata["ensemble_success"]) and all(r.success for r in member_results_list)
        final_output = metadata["ensemble_decision"] if success else metadata["ensemble_error"]

    if member_results_list and "conversation_success" in member_results_list[0].metadata:
        metadata = member_results_list[0].metadata
        success = bool(metadata["conversation_success"]) and all(
            r.success for r in member_results_list
        )
        final_output = metadata["conversation_output"]

    if (
        active_formation == TeamFormation.REFLECTION
        and effective_context.get("capture_member_usage", False)
        and member_results_list
        and "reflection_success" in member_results_list[0].metadata
    ):
        metadata = member_results_list[0].metadata
        success = bool(metadata["reflection_success"]) and all(
            r.success for r in member_results_list
        )
        final_output = metadata["reflection_output"]

    return success, final_output, total_tool_calls


def apply_pause_status(
    result_dict: Dict[str, Any],
    shared_state: Mapping[str, Any],
    thread_id: Optional[Any],
) -> None:
    """Mark ``result_dict`` as a durable paused aggregate when members await approval."""
    # ADR-023 pillar 2b: the formation stopped at a member awaiting approval —
    # surface a durable paused aggregate. The run is resumable on the same thread_id
    # (with an ``approval_decision`` in context) via the persisted pause checkpoint.
    awaiting = shared_state.get("__awaiting_approval__")
    awaiting_many = shared_state.get("__awaiting_approvals__")
    if awaiting:
        result_dict["success"] = False  # team did not complete — it is paused
        result_dict["status"] = "awaiting_approval"
        result_dict["paused_member_id"] = awaiting.get("member_id")
        result_dict["approval_request"] = awaiting.get("approval_request")
        if thread_id:
            result_dict["thread_id"] = str(thread_id)
    elif awaiting_many:
        # ADR-023 concurrent pause: a PARALLEL wave stopped with one or more members
        # awaiting approval — surface the multi-pause aggregate. Resumable on the same
        # thread_id via the persisted pause checkpoint (re-runs exactly the paused set).
        result_dict["success"] = False  # team did not complete — it is paused
        result_dict["status"] = "awaiting_approval"
        result_dict["awaiting_approvals"] = list(awaiting_many)
        result_dict["paused_member_ids"] = [a.get("member_id") for a in awaiting_many]
        if thread_id:
            result_dict["thread_id"] = str(thread_id)

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

"""Service-owned host for RL runtime access.

The RL implementation remains a first-class public framework API in
``victor.framework.rl.coordinator``. This module provides the canonical
service-first import surface for agent runtime code so the live orchestration
path does not depend on framework module paths directly.

Use this module for:
- sync and async access to the global RL coordinator
- benchmark-gated prompt rollout experiment creation from agent runtime code
- prompt rollout analysis and recommendation application from agent runtime code
- benchmark-suite post-processing into rollout and rollout-decision state
"""

from typing import Any, Optional

# Six wrappers below were token-identical to victor.framework.rl's
# (modulo docstrings and Dict/dict annotation spelling — jscpd exact-
# duplicate extraction): re-export the one implementation. The
# process_prompt_candidate_evaluation_suite pair stays LOCAL — it
# returns workflow.to_dict() where the framework returns the workflow
# object (real behavioral drift, not consolidated).
from victor.framework.rl import (  # noqa: F401
    analyze_prompt_rollout_experiment,
    analyze_prompt_rollout_experiment_async,
    apply_prompt_rollout_recommendation,
    apply_prompt_rollout_recommendation_async,
    create_prompt_rollout_experiment,
    create_prompt_rollout_experiment_async,
)
from victor.framework.rl.coordinator import (
    AsyncWriterQueue,
    RLCoordinator,
    get_rl_coordinator,
    get_rl_coordinator_async,
    reset_rl_coordinator,
)


def process_prompt_candidate_evaluation_suite(
    suite: Any,
    *,
    min_pass_rate: float = 0.5,
    promote_best: bool = False,
    create_rollout: bool = False,
    rollout_control_hash: Optional[str] = None,
    rollout_traffic_split: float = 0.1,
    rollout_min_samples_per_variant: int = 100,
    analyze_rollout: bool = False,
    apply_rollout_decision: bool = False,
    rollout_decision_dry_run: bool = False,
) -> Optional[dict[str, Any]]:
    """Process a prompt-candidate benchmark suite via the global RL coordinator."""
    coordinator = get_rl_coordinator()
    workflow = coordinator.process_prompt_candidate_evaluation_suite(
        suite,
        min_pass_rate=min_pass_rate,
        promote_best=promote_best,
        create_rollout=create_rollout,
        rollout_control_hash=rollout_control_hash,
        rollout_traffic_split=rollout_traffic_split,
        rollout_min_samples_per_variant=rollout_min_samples_per_variant,
        analyze_rollout=analyze_rollout,
        apply_rollout_decision=apply_rollout_decision,
        rollout_decision_dry_run=rollout_decision_dry_run,
    )
    return workflow.to_dict()


async def process_prompt_candidate_evaluation_suite_async(
    suite: Any,
    *,
    min_pass_rate: float = 0.5,
    promote_best: bool = False,
    create_rollout: bool = False,
    rollout_control_hash: Optional[str] = None,
    rollout_traffic_split: float = 0.1,
    rollout_min_samples_per_variant: int = 100,
    analyze_rollout: bool = False,
    apply_rollout_decision: bool = False,
    rollout_decision_dry_run: bool = False,
) -> Optional[dict[str, Any]]:
    """Process a prompt-candidate benchmark suite asynchronously via the global coordinator."""
    coordinator = await get_rl_coordinator_async()
    workflow = await coordinator.process_prompt_candidate_evaluation_suite_async(
        suite,
        min_pass_rate=min_pass_rate,
        promote_best=promote_best,
        create_rollout=create_rollout,
        rollout_control_hash=rollout_control_hash,
        rollout_traffic_split=rollout_traffic_split,
        rollout_min_samples_per_variant=rollout_min_samples_per_variant,
        analyze_rollout=analyze_rollout,
        apply_rollout_decision=apply_rollout_decision,
        rollout_decision_dry_run=rollout_decision_dry_run,
    )
    return workflow.to_dict()


__all__ = [
    "AsyncWriterQueue",
    "RLCoordinator",
    "analyze_prompt_rollout_experiment",
    "analyze_prompt_rollout_experiment_async",
    "apply_prompt_rollout_recommendation",
    "apply_prompt_rollout_recommendation_async",
    "process_prompt_candidate_evaluation_suite",
    "process_prompt_candidate_evaluation_suite_async",
    "create_prompt_rollout_experiment",
    "create_prompt_rollout_experiment_async",
    "get_rl_coordinator",
    "get_rl_coordinator_async",
    "reset_rl_coordinator",
]

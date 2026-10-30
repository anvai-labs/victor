"""SDK host adapters for RL runtime helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from victor_contracts._lazy import make_lazy_getattr

if TYPE_CHECKING:
    from victor.framework.rl import (
        RLCoordinator,
        RLManager,
        analyze_prompt_rollout_experiment,
        analyze_prompt_rollout_experiment_async,
        apply_prompt_rollout_recommendation,
        apply_prompt_rollout_recommendation_async,
        process_prompt_candidate_evaluation_suite,
        process_prompt_candidate_evaluation_suite_async,
        create_prompt_rollout_experiment,
        create_prompt_rollout_experiment_async,
        get_rl_coordinator,
        get_rl_coordinator_async,
    )
    from victor.framework.rl.base import RLOutcome

__all__ = [
    "RLCoordinator",
    "RLManager",
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
    "RLOutcome",
]

_LAZY_IMPORTS = {
    "RLOutcome": "victor.framework.rl.base",
    "RLCoordinator": "victor.framework.rl",
    "RLManager": "victor.framework.rl",
    "analyze_prompt_rollout_experiment": "victor.framework.rl",
    "analyze_prompt_rollout_experiment_async": "victor.framework.rl",
    "apply_prompt_rollout_recommendation": "victor.framework.rl",
    "apply_prompt_rollout_recommendation_async": "victor.framework.rl",
    "process_prompt_candidate_evaluation_suite": "victor.framework.rl",
    "process_prompt_candidate_evaluation_suite_async": "victor.framework.rl",
    "create_prompt_rollout_experiment": "victor.framework.rl",
    "create_prompt_rollout_experiment_async": "victor.framework.rl",
    "get_rl_coordinator": "victor.framework.rl",
    "get_rl_coordinator_async": "victor.framework.rl",
}


__getattr__ = make_lazy_getattr(__name__, _LAZY_IMPORTS)

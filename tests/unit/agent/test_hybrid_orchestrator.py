"""Default pins for the hybrid orchestration router.

Added with the 5x'd stamina budgets (2026-10-31): the Bayesian belief state
seeds its task budget from a module default, so the default is pinned here.
"""

from types import SimpleNamespace

import victor.agent.hybrid_orchestrator as hybrid_orchestrator
from victor.agent.hybrid_orchestrator import HybridOrchestrationRouter


class _RecordingBayesianService:
    """Minimal Bayesian service stub capturing belief-state creation."""

    def __init__(self):
        self.create_kwargs = None
        self.cleaned_up = []
        self.reliability_learner = SimpleNamespace(
            get_agent_reliability_stats=lambda agent_id: None
        )

    def create_belief_state(self, **kwargs):
        self.create_kwargs = kwargs
        return SimpleNamespace(belief_id="b1")

    def update_belief_with_message(self, **kwargs):
        return True

    def get_belief_state(self, belief_id):
        return SimpleNamespace(belief_id=belief_id, belief_entropy=0.4)

    def cleanup_belief_state(self, belief_id):
        self.cleaned_up.append(belief_id)


class _StubConsensusBuilder:
    """Replaces the DB-backed consensus builder for the default pin."""

    def __init__(self, service):
        self.service = service

    def compute_consensus(self, **kwargs):
        return {
            "recommended_outcome": "Yes",
            "confidence": 0.9,
            "agreement_level": "high",
        }


def test_bayesian_belief_state_default_tool_budget(monkeypatch):
    """Belief states are seeded with the 5x'd default tool budget (was 10)."""
    monkeypatch.setattr(hybrid_orchestrator, "BayesianConsensusBuilder", _StubConsensusBuilder)
    service = _RecordingBayesianService()
    router = HybridOrchestrationRouter(bayesian_service=service, force_bayesian=True)

    result = router.route_query(
        "Refactor the auth module across services",
        agent_messages={"a1": "yes, proceed with the plan"},
    )

    assert result.orchestration_type == "bayesian"
    assert service.create_kwargs is not None
    # 5x'd 2026-10-31: agents run longer autonomous sessions (was 10)
    assert service.create_kwargs["tool_budget"] == 50

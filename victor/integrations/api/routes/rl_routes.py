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

"""RL routes: /rl/stats, /rl/recommend, /rl/explore, /rl/strategy, /rl/reset."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from victor.integrations.api.fastapi_server import RLExploreRequest, RLStrategyRequest

if TYPE_CHECKING:
    from victor.integrations.api.fastapi_server import VictorFastAPIServer

logger = logging.getLogger(__name__)


def create_router(server: "VictorFastAPIServer") -> APIRouter:
    """Create RL routes bound to *server*."""
    router = APIRouter(tags=["RL"])

    @router.get("/rl/stats")
    async def rl_stats() -> JSONResponse:
        """Get RL model selector statistics."""
        try:
            from victor.framework.rl.coordinator import get_rl_coordinator

            coordinator = get_rl_coordinator()
            learner = coordinator.get_learner("model_selector")

            if not learner:
                return JSONResponse(
                    {"error": "Model selector learner not available"},
                    status_code=503,
                )

            rankings = learner.get_provider_rankings()

            task_q_summary = {}
            for provider, task_q_table in learner._q_table_by_task.items():
                task_q_summary[provider] = {
                    task_type: round(q_val, 3) for task_type, q_val in task_q_table.items()
                }

            stats = {
                "strategy": learner.strategy.value,
                "epsilon": round(learner.epsilon, 3),
                "total_selections": learner._total_selections,
                "num_providers": len(learner._q_table),
                "top_provider": rankings[0]["provider"] if rankings else None,
                "top_q_value": round(rankings[0]["q_value"], 3) if rankings else 0.0,
                "learning_rate": learner.learning_rate,
                "ucb_c": learner.ucb_c,
                "provider_rankings": [
                    {
                        "provider": r["provider"],
                        "q_value": round(r["q_value"], 3),
                        "sessions": r["session_count"],
                        "confidence": round(r["confidence"], 3),
                    }
                    for r in rankings[:5]
                ],
                "task_q_tables": task_q_summary,
                "db_path": str(coordinator.db_path),
            }

            return JSONResponse(stats)

        except Exception:
            logger.exception("RL stats error")
            return JSONResponse({"error": "Internal server error"}, status_code=500)

    @router.get("/rl/recommend")
    async def rl_recommend(
        task_type: Optional[str] = Query(None),
    ) -> JSONResponse:
        """Get model recommendation based on Q-values."""
        try:
            from victor.framework.rl.coordinator import get_rl_coordinator
            import json

            coordinator = get_rl_coordinator()
            learner = coordinator.get_learner("model_selector")

            if not learner:
                return JSONResponse(
                    {"error": "Model selector learner not available"},
                    status_code=503,
                )

            available = list(learner._q_table.keys()) if learner._q_table else ["ollama"]

            recommendation = coordinator.get_recommendation(
                "model_selector",
                json.dumps(available),
                "",
                task_type or "unknown",
            )

            if not recommendation:
                return JSONResponse(
                    {
                        "provider": available[0] if available else "ollama",
                        "model": None,
                        "q_value": 0.5,
                        "confidence": 0.0,
                        "reason": "No recommendation available",
                        "task_type": task_type,
                        "alternatives": [],
                    }
                )

            alternatives = []
            for provider in available:
                if provider != recommendation.value:
                    q_val = learner._get_q_value(provider, task_type)
                    alternatives.append({"provider": provider, "q_value": round(q_val, 3)})
            alternatives.sort(key=lambda x: x["q_value"], reverse=True)

            return JSONResponse(
                {
                    "provider": recommendation.value,
                    "model": None,
                    "q_value": round(learner._get_q_value(recommendation.value, task_type), 3),
                    "confidence": round(recommendation.confidence, 3),
                    "reason": recommendation.reason,
                    "task_type": task_type,
                    "alternatives": alternatives[:3],
                }
            )

        except Exception:
            logger.exception("RL recommend error")
            return JSONResponse({"error": "Internal server error"}, status_code=500)

    @router.post("/rl/explore")
    async def rl_explore(request: RLExploreRequest) -> JSONResponse:
        """Set exploration rate for RL model selector."""
        try:
            from victor.framework.rl.coordinator import get_rl_coordinator

            coordinator = get_rl_coordinator()
            learner = coordinator.get_learner("model_selector")

            if not learner:
                return JSONResponse(
                    {"error": "Model selector learner not available"},
                    status_code=503,
                )

            old_rate = learner.epsilon
            learner.epsilon = request.rate

            return JSONResponse(
                {
                    "success": True,
                    "old_rate": round(old_rate, 3),
                    "new_rate": round(request.rate, 3),
                }
            )

        except Exception:
            logger.exception("RL explore error")
            return JSONResponse({"error": "Internal server error"}, status_code=500)

    @router.post("/rl/strategy")
    async def rl_strategy(request: RLStrategyRequest) -> JSONResponse:
        """Set selection strategy for RL model selector."""
        try:
            from victor.framework.rl.coordinator import get_rl_coordinator
            from victor.framework.rl.learners.model_selector import (
                SelectionStrategy,
            )

            coordinator = get_rl_coordinator()
            learner = coordinator.get_learner("model_selector")

            if not learner:
                return JSONResponse(
                    {"error": "Model selector learner not available"},
                    status_code=503,
                )

            try:
                strategy = SelectionStrategy(request.strategy.lower())
            except ValueError:
                available = [s.value for s in SelectionStrategy]
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown strategy: {request.strategy}. Available: {available}",
                )

            old_strategy = learner.strategy.value
            learner.strategy = strategy

            return JSONResponse(
                {
                    "success": True,
                    "old_strategy": old_strategy,
                    "new_strategy": strategy.value,
                }
            )

        except HTTPException:
            raise
        except Exception:
            logger.exception("RL strategy error")
            return JSONResponse({"error": "Internal server error"}, status_code=500)

    @router.post("/rl/reset")
    async def rl_reset() -> JSONResponse:
        """Reset RL model selector Q-values."""
        try:
            from victor.framework.rl.coordinator import get_rl_coordinator

            coordinator = get_rl_coordinator()
            learner = coordinator.get_learner("model_selector")
            if learner is None:
                return JSONResponse(
                    {"error": "Model selector learner not available"},
                    status_code=503,
                )

            if hasattr(coordinator, "_ensure_db_connection"):
                coordinator._ensure_db_connection()
            cursor = coordinator.db.cursor()
            try:
                cursor.execute("DELETE FROM model_selector_q_values")
                cursor.execute("DELETE FROM model_selector_task_q_values")
                cursor.execute("DELETE FROM model_selector_state")
                coordinator.db.commit()
            finally:
                cursor.close()

            coordinator._learners.pop("model_selector", None)
            learner = coordinator.get_learner("model_selector")

            return JSONResponse(
                {
                    "success": True,
                    "message": "RL model selector reset to initial state",
                }
            )

        except Exception:
            logger.exception("RL reset error")
            return JSONResponse({"error": "Internal server error"}, status_code=500)

    return router

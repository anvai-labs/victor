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

"""Domain facade for orchestrator decomposition.

``OrchestrationFacade`` is the one live runtime-facing migration boundary,
exposing already-initialized components behind a coherent domain boundary
(service-first surfaces ``chat_service``/``tool_service``/``session_service`` and
state-passed surfaces ``exploration_state_passed``/``system_prompt_state_passed``/
``safety_state_passed``). It is read by ``framework/agent.py`` and the turn
execution runtime via ``orchestrator._orchestration_facade``.

The seven per-domain facades (Chat/Tool/Session/Provider/Resilience/Workflow/
Metrics) were removed: they were constructed lazily in the bootstrapper but had
zero production readers (the orchestrator never delegated through them), so they
were dead parallel views. The orchestrator owns its component properties directly.
"""

from victor.agent.facades.orchestration_facade import OrchestrationFacade

__all__ = [
    "OrchestrationFacade",
]

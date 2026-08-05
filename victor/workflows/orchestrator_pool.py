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

"""Orchestrator pool for multi-provider workflows.

Manages multiple orchestrators for different provider profiles.
This is a stub during the SOLID refactoring migration.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class OrchestratorPool:
    """Pool of orchestrators for multi-provider workflows.

    Responsibility (SRP):
    - Manage multiple orchestrators for different providers
    - Create orchestrators on-demand for unique profiles
    - Reuse orchestrators across workflow executions
    - Provide orchestrator lifecycle management

    Non-responsibility:
    - Orchestrator creation (handled by OrchestratorFactory)
    - Profile resolution (handled by profile resolver)
    """

    def __init__(self, settings: Any, container: Optional[Any] = None):
        """Initialize the orchestrator pool.

        Args:
            settings: Application settings
            container: DI container
        """
        self._settings = settings
        self._container = container
        self._orchestrators: Dict[str, Any] = {}

    def get_orchestrator(self, profile: Optional[str] = None) -> Any:
        """Get orchestrator for a profile.

        Args:
            profile: Provider profile name

        Returns:
            AgentOrchestrator instance
        """
        # STUB: Proper orchestrator pooling — future feature for workflow parallelism
        logger.debug(f"OrchestratorPool.get_orchestrator({profile}) - stub implementation")
        return None

    def get_default_orchestrator(self) -> Any:
        """Get the default orchestrator.

        The current stub pool has no provider-specific distinction yet, so the
        default path is the same as requesting an unqualified orchestrator.
        """
        return self.get_orchestrator(None)


__all__ = ["OrchestratorPool"]

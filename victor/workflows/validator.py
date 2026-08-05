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

"""Workflow validator.

Validates workflow structure and semantics.
This is a stub during the SOLID refactoring migration.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class WorkflowValidator:
    """Validates workflow structure and semantics.

    Responsibility (SRP):
    - Validate workflow structure (nodes, edges)
    - Check node type compatibility
    - Validate node properties (tool_budget, timeout)
    - Check for cycles and unreachable nodes

    Non-responsibility:
    - Workflow loading (handled by YAMLWorkflowLoader)
    - Workflow compilation (handled by WorkflowCompiler)
    """

    def __init__(self, strict_mode: bool = False):
        """Initialize the validator.

        Args:
            strict_mode: Whether to enable strict validation
        """
        self._strict_mode = strict_mode

    def validate(self, workflow_def: Dict[str, Any]) -> bool:
        """Validate a workflow definition.

        Args:
            workflow_def: Workflow definition dict

        Returns:
            True if valid

        Raises:
            ValueError: If validation fails
        """
        logger.debug("WorkflowValidator.validate() - stub implementation")
        return True


__all__ = ["WorkflowValidator"]

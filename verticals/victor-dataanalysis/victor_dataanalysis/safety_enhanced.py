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

"""Enhanced safety integration for victor-dataanalysis using SafetyCoordinator."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from victor.framework.extensions import (
    SafetyAction,
    SafetyCategory,
    SafetyCoordinator,
    SafetyRule,
)
from victor.framework.extensions import SafetyExtensionProtocol, SafetyPattern

logger = logging.getLogger(__name__)


class DataAnalysisSafetyRules:
    """Data analysis safety rules."""

    @staticmethod
    def get_all_rules() -> List[SafetyRule]:
        """Get all data analysis safety rules."""
        return [
            SafetyRule(
                rule_id="dataanalysis_delete_data",
                category=SafetyCategory.FILE,
                pattern=r"delete.*csv|drop.*table|truncate.*data",
                description="Delete data files or database tables",
                action=SafetyAction.REQUIRE_CONFIRMATION,
                severity=8,
                confirmation_prompt="This will delete data. Consider backing up first. Continue?",
            ),
            SafetyRule(
                rule_id="dataanalysis_overwrite_original",
                category=SafetyCategory.FILE,
                pattern=r"overwrite.*source|save.*--force.*original",
                description="Overwrite original data source",
                action=SafetyAction.BLOCK,
                severity=10,
            ),
            SafetyRule(
                rule_id="dataanalysis_share_sensitive",
                category=SafetyCategory.SHELL,
                pattern=r"upload.*data|share.*csv|publish.*dataset",
                description="Share or upload potentially sensitive data",
                action=SafetyAction.REQUIRE_CONFIRMATION,
                severity=7,
                confirmation_prompt="Ensure data doesn't contain sensitive information. Continue?",
            ),
        ]


class EnhancedDataAnalysisSafetyExtension(SafetyExtensionProtocol):
    """Enhanced safety extension for DataAnalysis."""

    def __init__(self, strict_mode: bool = False):
        self._coordinator = SafetyCoordinator(strict_mode=strict_mode)
        for rule in DataAnalysisSafetyRules.get_all_rules():
            self._coordinator.register_rule(rule)
        logger.info("EnhancedDataAnalysisSafetyExtension initialized")

    def check_operation(
        self, tool_name: str, args: List[str], context: Optional[Dict[str, Any]] = None
    ) -> Any:
        return self._coordinator.check_safety(tool_name, args, context)

    def is_operation_safe(
        self, tool_name: str, args: List[str], context: Optional[Dict[str, Any]] = None
    ) -> bool:
        return self._coordinator.is_operation_safe(tool_name, args, context)

    def get_bash_patterns(self) -> List[SafetyPattern]:
        return []

    def get_file_patterns(self) -> List[SafetyPattern]:
        return []

    def get_tool_restrictions(self) -> Dict[str, List[str]]:
        return {}

    def get_coordinator(self) -> SafetyCoordinator:
        return self._coordinator

    def get_safety_stats(self) -> Dict[str, Any]:
        return self._coordinator.get_stats_dict()


__all__ = ["DataAnalysisSafetyRules", "EnhancedDataAnalysisSafetyExtension"]

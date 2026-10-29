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
from typing import List

from victor.framework.extensions import (
    SafetyAction,
    SafetyCategory,
    SafetyRule,
)
from victor_contracts.verticals import CoordinatorSafetyExtension

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


class EnhancedDataAnalysisSafetyExtension(CoordinatorSafetyExtension):
    """Enhanced dataanalysis safety extension (shared coordinator kernel)."""

    def __init__(self, strict_mode: bool = False, enable_custom_rules: bool = True):
        super().__init__(
            "dataanalysis",
            DataAnalysisSafetyRules.get_all_rules(),
            strict_mode=strict_mode,
            enable_custom_rules=enable_custom_rules,
        )


__all__ = ["DataAnalysisSafetyRules", "EnhancedDataAnalysisSafetyExtension"]

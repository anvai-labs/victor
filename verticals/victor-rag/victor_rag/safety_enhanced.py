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

"""Enhanced safety integration for victor-rag using SafetyCoordinator."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from victor.framework.extensions import (
    SafetyAction,
    SafetyCategory,
    SafetyCoordinator,
    SafetyRule,
)
from victor_contracts.verticals import CoordinatorSafetyExtension

logger = logging.getLogger(__name__)


class RAGSafetyRules:
    """RAG-specific safety rules."""

    @staticmethod
    def get_all_rules() -> List[SafetyRule]:
        """Get all RAG safety rules."""
        return [
            SafetyRule(
                rule_id="rag_delete_index",
                category=SafetyCategory.FILE,
                pattern=r"delete.*index|drop.*vector|remove.*embeddings",
                description="Delete RAG index or embeddings",
                action=SafetyAction.REQUIRE_CONFIRMATION,
                severity=9,
                confirmation_prompt="This will delete the RAG index. Rebuilding may take time. Continue?",
            ),
            SafetyRule(
                rule_id="rag_bulk_embed",
                category=SafetyCategory.SHELL,
                pattern=r"embed.*--all|index.*bulk|--rebuild.*index",
                description="Bulk rebuild of RAG index",
                action=SafetyAction.WARN,
                severity=6,
            ),
            SafetyRule(
                rule_id="rag_share_corpus",
                category=SafetyCategory.SHELL,
                pattern=r"share.*corpus|upload.*documents|publish.*index",
                description="Share or upload RAG corpus documents",
                action=SafetyAction.REQUIRE_CONFIRMATION,
                severity=7,
                confirmation_prompt="Ensure documents don't contain sensitive information. Continue?",
            ),
        ]


class EnhancedRAGSafetyExtension(CoordinatorSafetyExtension):
    """Enhanced rag safety extension (shared coordinator kernel)."""

    def __init__(self, strict_mode: bool = False, enable_custom_rules: bool = True):
        super().__init__(
            "rag",
            RAGSafetyRules.get_all_rules(),
            strict_mode=strict_mode,
            enable_custom_rules=enable_custom_rules,
        )


__all__ = ["RAGSafetyRules", "EnhancedRAGSafetyExtension"]

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

"""Framework runtime adapter for SDK prompt contribution contracts."""

from typing import Any, List

from victor_contracts.capabilities import (
    PromptContribution,
    PromptContributionCapability as SdkPromptContributionCapability,
)


class PromptContributionCapability(SdkPromptContributionCapability):
    """Runtime-aware prompt contribution capability wrapper."""

    def get_contributors(self) -> List[Any]:
        """Get prompt contributors for vertical.

        Returns:
            List of prompt contributor adapters (if available)
        """
        # Try to import PromptContributorAdapter
        try:
            from victor.core.verticals.prompt_adapter import PromptContributorAdapter

            contributors = []
            for contrib in self.contributions:
                adapter = PromptContributorAdapter.from_dict(
                    task_hints={
                        contrib.task_type: {
                            "hint": contrib.hint,
                            "tool_budget": contrib.tool_budget,
                        }
                    },
                    system_prompt_section=contrib.system_section,
                    priority=contrib.priority,
                )
                contributors.append(adapter)

            return contributors
        except ImportError:
            # Return empty list if adapter not available
            return []


__all__ = [
    "PromptContribution",
    "PromptContributionCapability",
]

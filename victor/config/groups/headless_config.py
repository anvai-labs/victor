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

"""Headless mode configuration settings.

Extracted from victor/config/settings.py to improve maintainability.
Contains configuration for headless/CI/CD automation mode.
"""

from typing import Optional

from pydantic import BaseModel


class HeadlessSettings(BaseModel):
    """Headless mode settings for CI/CD and automation.

    Controls behavior when running without user interaction.
    """

    # Headless Mode Settings (for CI/CD and automation)
    # These can be set via CLI flags or environment variables:
    #   - VICTOR_HEADLESS_MODE=true
    #   - VICTOR_DRY_RUN_MODE=true
    #   - VICTOR_MAX_FILE_CHANGES=10
    headless_mode: bool = False  # Run without prompts, auto-approve safe actions
    dry_run_mode: bool = False  # Preview changes without applying them
    auto_approve_safe: bool = False  # Auto-approve read-only and LOW risk operations
    max_file_changes: Optional[int] = None  # Limit file modifications per session
    one_shot_mode: bool = False  # Exit after completing a single request

    @property
    def is_automated(self) -> bool:
        """Check if running in automated mode.

        Returns:
            True if any automation mode is enabled
        """
        return self.headless_mode or self.dry_run_mode or self.one_shot_mode

    @property
    def has_file_change_limit(self) -> bool:
        """Check if file change limit is set.

        Returns:
            True if max_file_changes is set
        """
        return self.max_file_changes is not None

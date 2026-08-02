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

"""Subprocess configuration settings.

Extracted from victor/config/settings.py to improve maintainability.
Contains configuration for code execution and subprocess resource limits.
"""

from typing import Optional

from pydantic import BaseModel, Field


class SubprocessSettings(BaseModel):
    """Subprocess and code execution configuration.

    Controls code executor sandbox defaults and subprocess
    resource limits (POSIX rlimit for tool subprocesses).
    """

    # Code execution sandbox defaults (used by code_executor_tool)
    code_executor_network_disabled: bool = True
    code_executor_memory_limit: Optional[str] = "512m"
    code_executor_cpu_shares: Optional[int] = 256

    # Subprocess resource limits (POSIX rlimit for tool subprocesses)
    # When enabled, applies memory/CPU/FD limits via preexec_fn.
    # Defaults to False — opt-in to avoid breaking existing workflows.
    subprocess_resource_limits_enabled: bool = False

    @property
    def memory_limit_mb(self) -> Optional[int]:
        """Convert memory limit string to MB.

        Returns:
            Memory limit in MB or None if not set
        """
        if self.code_executor_memory_limit is None:
            return None

        limit_str = self.code_executor_memory_limit.lower()
        if limit_str.endswith("m"):
            return int(limit_str[:-1])
        elif limit_str.endswith("g"):
            return int(limit_str[:-1]) * 1024
        else:
            # Assume MB if no unit
            return int(limit_str)

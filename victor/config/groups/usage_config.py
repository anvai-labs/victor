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

"""Usage analytics configuration settings.

Extracted from victor/config/settings.py to improve maintainability.
Contains configuration for usage logging and semantic sampling.
"""

from pydantic import BaseModel, Field, field_validator


class UsageSettings(BaseModel):
    """Usage analytics and logging configuration.

    Controls usage log semantic sampling to reduce disk I/O
    for noisy events.
    """

    # Usage log semantic sampling (reduces disk I/O for noisy events)
    usage_sampling_enabled: bool = True
    usage_content_sample_rate: int = Field(
        default=10, gt=0, description="Emit 1 in N content-chunk events"
    )
    usage_dedup_window_seconds: float = Field(
        default=5.0, gt=0, description="Dedup window for progress events"
    )

    @field_validator("usage_content_sample_rate")
    @classmethod
    def validate_sample_rate(cls, v: int) -> int:
        """Validate sample rate is positive.

        Args:
            v: Sample rate

        Returns:
            Validated sample rate

        Raises:
            ValueError: If sample rate is not positive
        """
        if v <= 0:
            raise ValueError("usage_content_sample_rate must be positive")
        return v

    @field_validator("usage_dedup_window_seconds")
    @classmethod
    def validate_dedup_window(cls, v: float) -> float:
        """Validate dedup window is positive.

        Args:
            v: Dedup window in seconds

        Returns:
            Validated dedup window

        Raises:
            ValueError: If dedup window is not positive
        """
        if v <= 0:
            raise ValueError("usage_dedup_window_seconds must be positive")
        return v

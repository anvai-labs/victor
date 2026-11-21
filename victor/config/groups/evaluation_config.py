# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Evaluation settings (ADR-036 flag-graduation field data).

The field-data recorder persists per-session flag exposure + outcome markers
to an append-only JSONL store under the Victor state directory so the ADR-036
graduation queue can read real-session evidence. Strictly opt-in: the flag is
default OFF and the recorder is a no-op unless the operator enables it.
"""

from typing import Optional

from pydantic import BaseModel, Field, field_validator


class EvaluationSettings(BaseModel):
    """Evaluation/field-data settings (ADR-036).

    Attributes:
        field_data_enabled: Master switch for the ADR-036 field-data recorder.
            Default OFF — no session is recorded unless opted in (env override
            ``VICTOR_FIELD_DATA_ENABLED``).
        field_data_store_dir: Override for the JSONL store directory. None =
            the Victor state dir ``evaluation/`` subfolder
            (``~/.victor/evaluation``).
        field_data_max_file_bytes: Size cap per daily JSONL segment before
            rotation onto the next numbered segment.
    """

    field_data_enabled: bool = False
    field_data_store_dir: Optional[str] = None
    field_data_max_file_bytes: int = Field(default=5 * 1024 * 1024)

    @field_validator("field_data_max_file_bytes")
    @classmethod
    def validate_max_file_bytes(cls, v: int) -> int:
        """Require a positive cap so rotation can never disable itself."""
        if v <= 0:
            raise ValueError("field_data_max_file_bytes must be positive")
        return v

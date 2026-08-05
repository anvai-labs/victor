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

"""Example state models for StateGraph users."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field


class AgentStateModel(BaseModel):
    """Example Pydantic state model for agent workflows.

    This model is provided as a developer example and compatibility surface.
    Production workflows should typically define a domain-specific state model
    instead of reusing this three-field example directly.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )

    messages: List[str] = Field(default_factory=list, description="Conversation messages")
    task: str = Field(default="", description="Current task being processed")
    result: Optional[str] = Field(default=None, description="Task result")

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value by key (dict-like interface)."""
        return getattr(self, key, default)

    def keys(self) -> List[str]:
        """Return list of keys (dict-like interface)."""
        return ["messages", "task", "result"]

    def values(self) -> List[Any]:
        """Return list of values (dict-like interface)."""
        return [self.messages, self.task, self.result]

    def items(self) -> List[Tuple[str, Any]]:
        """Return list of (key, value) tuples (dict-like interface)."""
        return [
            ("messages", self.messages),
            ("task", self.task),
            ("result", self.result),
        ]

    def __getitem__(self, key: str) -> Any:
        """Get item by key (dict-like subscript access)."""
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Set item by key (dict-like subscript access)."""
        setattr(self, key, value)

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

"""Enhanced conversation management for Victor verticals.

Provides base classes and utilities for implementing enhanced conversation
management in verticals. Verticals can inherit from BaseConversationManager
to get common conversation functionality while adding vertical-specific features.

Usage:
    from victor.contrib.conversation import BaseConversationManager

    class MyVerticalConversationManager(BaseConversationManager):
        def get_vertical_name(self) -> str:
            return "myvertical"

        def get_system_prompt(self) -> str:
            return "You are a helpful assistant for myvertical."
"""

from victor.contrib.conversation.base_manager import BaseConversationManager
from victor.contrib.conversation.vertical_context import VerticalConversationContext

__all__ = [
    "BaseConversationManager",
    "VerticalConversationContext",
]

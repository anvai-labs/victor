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

"""
Tool calling adapters for unified provider handling.

This module provides a maintainable abstraction for tool calling across
different LLM providers, handling the nuances of each provider's API
while presenting a unified interface to the orchestrator.
"""

from victor.agent.tool_calling.base import (
    BaseToolCallingAdapter,
    ToolCall,
    ToolCallingCapabilities,
    ToolCallFormat,
    ToolCallParseResult,
)
from victor.agent.tool_calling.registry import ToolCallingAdapterRegistry
from victor.agent.tool_calling.adapters import (
    AnthropicToolCallingAdapter,
    AzureOpenAIToolCallingAdapter,
    BedrockToolCallingAdapter,
    DeepSeekToolCallingAdapter,
    OpenAIToolCallingAdapter,
    OllamaToolCallingAdapter,
    OpenAICompatToolCallingAdapter,
)
from victor.agent.tool_calling.capabilities import (
    ModelCapabilityLoader,
    get_model_capabilities,
    # Model name normalization utilities
    normalize_model_name,
    get_model_name_variants,
    # MODEL_NAME_ALIASES not exported — uses normalize_model_name() instead
)
from victor.agent.tool_calling.base import HALLUCINATED_ARGUMENTS

__all__ = [
    "BaseToolCallingAdapter",
    "ToolCall",
    "ToolCallingCapabilities",
    "ToolCallFormat",
    "ToolCallParseResult",
    "ToolCallingAdapterRegistry",
    "AnthropicToolCallingAdapter",
    "AzureOpenAIToolCallingAdapter",
    "BedrockToolCallingAdapter",
    "DeepSeekToolCallingAdapter",
    "OpenAIToolCallingAdapter",
    "OllamaToolCallingAdapter",
    "OpenAICompatToolCallingAdapter",
    "ModelCapabilityLoader",
    "get_model_capabilities",
]

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

"""Configuration groups for Victor settings.

This package contains focused configuration modules that were extracted
from the monolithic settings.py file to improve maintainability.

Each config group is a Pydantic BaseModel that can be used independently
or composed together in the main Settings class.
"""

from victor.config.groups.provider_config import (
    ProviderSettings,
    ProviderConfig,
    ModelConfig,
)
from victor.config.groups.agent_config import AgentSettings, PlanningConfig
from victor.config.groups.server_config import ServerSettings
from victor.config.groups.codebase_config import CodebaseSettings
from victor.config.groups.usage_config import UsageSettings
from victor.config.groups.usage_gateway_config import UsageGatewaySettings
from victor.config.groups.subprocess_config import SubprocessSettings
from victor.config.groups.headless_config import HeadlessSettings
from victor.config.groups.workflow_config import WorkflowSettings
from victor.config.groups.response_config import ResponseSettings
from victor.config.groups.cache_config import CacheSettings
from victor.config.groups.recovery_config import RecoverySettings
from victor.config.groups.analytics_config import AnalyticsSettings
from victor.config.groups.network_config import NetworkSettings
from victor.config.groups.embedding_config import EmbeddingSettings
from victor.config.groups.tool_selection_config import ToolSelectionSettings
from victor.config.groups.fuzzy_matching_config import FuzzyMatchingSettings
from victor.config.groups.governance_settings import GovernanceSettings

__all__ = [
    # Provider configuration
    "ProviderSettings",
    "ProviderConfig",
    "ModelConfig",
    # Agent configuration
    "AgentSettings",
    "PlanningConfig",
    # Server configuration
    "ServerSettings",
    # Codebase configuration
    "CodebaseSettings",
    # Usage configuration
    "UsageSettings",
    # Usage-gateway (sandhi) attribution configuration
    "UsageGatewaySettings",
    # Subprocess configuration
    "SubprocessSettings",
    # Headless configuration
    "HeadlessSettings",
    # Workflow configuration
    "WorkflowSettings",
    # Response configuration
    "ResponseSettings",
    # Cache configuration
    "CacheSettings",
    # Recovery configuration
    "RecoverySettings",
    # Analytics configuration
    "AnalyticsSettings",
    # Network configuration
    "NetworkSettings",
    # Embedding configuration
    "EmbeddingSettings",
    # Tool selection configuration
    "ToolSelectionSettings",
    # Fuzzy matching configuration
    "FuzzyMatchingSettings",
    # Governance / policy configuration
    "GovernanceSettings",
]

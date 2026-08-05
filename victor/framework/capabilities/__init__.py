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

"""Generic capability providers for vertical extensions.

This module contains framework-level capability providers that can be reused
across all verticals to promote code reuse and maintain SOLID principles.

Phase 1: Promote Generic Capabilities to Framework

Capabilities:
- BaseCapabilityProvider: Abstract base for vertical capability providers
- CapabilityMetadata: Metadata for capability registration
- FileOperationsCapability: Common file operation tools (read, write, edit, grep)
- PromptContributionCapability: Common prompt hints for task types
- PrivacyCapabilityProvider: Framework-level privacy and PII management (cross-vertical)
- StageBuilderCapability: Generic stage definitions for workflow templates
- GroundingRulesCapability: Centralized grounding rules for vertical constraints
- ValidationCapabilityProvider: Pluggable validation system
- SafetyRulesCapabilityProvider: Reusable safety pattern definitions
- TaskTypeHintCapabilityProvider: Centralized task type hints
- SourceVerificationCapabilityProvider: Source verification for citations
"""

# Base classes for capability providers
from .base import BaseCapabilityProvider, CapabilityMetadata

# Concrete capability implementations
from .file_operations import FileOperationsCapability
from .prompt_contributions import PromptContributionCapability
from .privacy import PrivacyCapabilityProvider

# Phase 1: Generic capabilities promoted to framework
from .stages import (
    StageBuilderCapability,
    StageBuilderPresets,
    StandardStage,
    StagePromptHint,
)
from .grounding_rules import (
    GroundingRulesCapability,
    GroundingRulesPresets,
    RuleCategory,
    GroundingRule,
)
from .validation import (
    ValidationCapabilityProvider,
    Validator,
    FilePathValidator,
    CodeSyntaxValidator,
    ConfigurationValidator,
    OutputFormatValidator,
    CapabilityValidationResult,
    ValidationSeverity,
)
from .safety_rules import (
    SafetyRulesCapabilityProvider,
    SafetyRulesPresets,
    SafetyCategory,
    SafetyAction,
    SafetyRule,
)
from .task_hints import (
    TaskTypeHintCapabilityProvider,
    TaskTypeHintPresets,
    TaskCategory,
    TaskTypeHint,
)
from .source_verification import (
    SourceVerificationCapabilityProvider,
    SourceVerificationPresets,
    SourceType,
    ReliabilityLevel,
    Citation,
    VerificationResult,
)

__all__ = [
    # Base classes
    "BaseCapabilityProvider",
    "CapabilityMetadata",
    # Original concrete implementations
    "FileOperationsCapability",
    "PromptContributionCapability",
    "PrivacyCapabilityProvider",
    # Phase 1: Generic capabilities
    "StageBuilderCapability",
    "StageBuilderPresets",
    "StandardStage",
    "StagePromptHint",
    "GroundingRulesCapability",
    "GroundingRulesPresets",
    "RuleCategory",
    "GroundingRule",
    "ValidationCapabilityProvider",
    "Validator",
    "FilePathValidator",
    "CodeSyntaxValidator",
    "ConfigurationValidator",
    "OutputFormatValidator",
    "CapabilityValidationResult",
    "ValidationSeverity",
    "SafetyRulesCapabilityProvider",
    "SafetyRulesPresets",
    "SafetyCategory",
    "SafetyAction",
    "SafetyRule",
    "TaskTypeHintCapabilityProvider",
    "TaskTypeHintPresets",
    "TaskCategory",
    "TaskTypeHint",
    "SourceVerificationCapabilityProvider",
    "SourceVerificationPresets",
    "SourceType",
    "ReliabilityLevel",
    "Citation",
    "VerificationResult",
]

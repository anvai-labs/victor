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

"""Core protocol interfaces for provider adaptation, grounding, and quality."""

# Provider Adapter
from victor.protocols.provider_adapter import (
    BaseProviderAdapter,
    ContinuationContext,
    IProviderAdapter,
    ProviderCapabilities,
    ToolCallFormat,
    get_provider_adapter,
    register_provider_adapter,
)

# Grounding
from victor.protocols.grounding import (
    AggregatedVerificationResult,
    CompositeGroundingVerifier,
    ContentMatchStrategy,
    FileExistenceStrategy,
    GroundingClaim,
    GroundingClaimType,
    IGroundingStrategy,
    SymbolReferenceStrategy,
    VerificationResult,
)

# Quality
from victor.protocols.quality import (
    BaseQualityAssessor,
    CompositeQualityAssessor,
    DimensionScore,
    IQualityAssessor,
    ProtocolQualityDimension,
    ProviderAwareQualityAssessor,
    QualityScore,
    SimpleQualityAssessor,
)

# Mode Awareness
from victor.protocols.mode_aware import (
    IModeController,
    ModeAwareMixin,
    ModeInfo,
    create_mode_aware_mixin,
)

# Path Resolution
from victor.protocols.path_resolver import (
    IPathResolver,
    PathResolution,
    PathResolver,
    create_path_resolver,
)

# LSP Types - Enumerations
from victor.protocols.lsp_types import (
    DiagnosticSeverity,
    CompletionItemKind,
    SymbolKind,
    DiagnosticTag,
    # LSP Types - Position and Range
    Position,
    Range,
    Location,
    LocationLink,
    # LSP Types - Diagnostics
    DiagnosticRelatedInformation,
    Diagnostic,
    # LSP Types - Completions
    CompletionItem,
    # LSP Types - Hover
    Hover,
    # LSP Types - Symbols
    DocumentSymbol,
    SymbolInformation,
    # LSP Types - Text Edits
    TextEdit,
    TextDocumentIdentifier,
    VersionedTextDocumentIdentifier,
    TextDocumentEdit,
)

# Team Coordination
from victor.protocols.team import (
    IAgent,
    ITeamMember,
    ITeamCoordinator,
    IDelegateFollowUpCoordinator,
    IObservableCoordinator,
    IRLCoordinator,
    IMessageBusProvider,
    ISharedMemoryProvider,
    IEnhancedTeamCoordinator,
    TeamCoordinatorProtocol,
    TeamMemberProtocol,
)

# Search Protocols
from victor.protocols.search import (
    ISemanticSearch,
    IIndexable,
    ISemanticSearchWithIndexing,
)

# Agent Orchestrator Protocol (breaks circular dependencies)
from victor.protocols.agent import (
    IAgentOrchestrator,
    IAgentOrchestratorFactory,
)

# Tool Selector Protocol (unifies tool selection interfaces)
from victor.protocols.tool_selector import (
    IToolSelector,
    IConfigurableToolSelector,
    IToolSelectorFactory,
    ToolSelectionResult,
    ToolSelectionContext,
    ToolSelectionStrategy,
)

__all__ = [
    # Provider Adapter
    "IProviderAdapter",
    "ProviderCapabilities",
    "ToolCallFormat",
    "BaseProviderAdapter",
    "ContinuationContext",
    "get_provider_adapter",
    "register_provider_adapter",
    # Grounding
    "IGroundingStrategy",
    "GroundingClaimType",
    "GroundingClaim",
    "VerificationResult",
    "AggregatedVerificationResult",
    "FileExistenceStrategy",
    "SymbolReferenceStrategy",
    "ContentMatchStrategy",
    "CompositeGroundingVerifier",
    # Quality
    "IQualityAssessor",
    "ProtocolQualityDimension",
    "DimensionScore",
    "QualityScore",
    "BaseQualityAssessor",
    "SimpleQualityAssessor",
    "ProviderAwareQualityAssessor",
    "CompositeQualityAssessor",
    # Mode Awareness
    "IModeController",
    "ModeInfo",
    "ModeAwareMixin",
    "create_mode_aware_mixin",
    # Path Resolution
    "IPathResolver",
    "PathResolution",
    "PathResolver",
    "create_path_resolver",
    # LSP Types - Enumerations
    "DiagnosticSeverity",
    "CompletionItemKind",
    "SymbolKind",
    "DiagnosticTag",
    # LSP Types - Position and Range
    "Position",
    "Range",
    "Location",
    "LocationLink",
    # LSP Types - Diagnostics
    "DiagnosticRelatedInformation",
    "Diagnostic",
    # LSP Types - Completions
    "CompletionItem",
    # LSP Types - Hover
    "Hover",
    # LSP Types - Symbols
    "DocumentSymbol",
    "SymbolInformation",
    # LSP Types - Text Edits
    "TextEdit",
    "TextDocumentIdentifier",
    "VersionedTextDocumentIdentifier",
    "TextDocumentEdit",
    # Team Coordination
    "IAgent",
    "ITeamMember",
    "ITeamCoordinator",
    "IDelegateFollowUpCoordinator",
    "IObservableCoordinator",
    "IRLCoordinator",
    "IMessageBusProvider",
    "ISharedMemoryProvider",
    "IEnhancedTeamCoordinator",
    "TeamCoordinatorProtocol",
    "TeamMemberProtocol",
    # Search Protocols
    "ISemanticSearch",
    "IIndexable",
    "ISemanticSearchWithIndexing",
    # Agent Orchestrator Protocol
    "IAgentOrchestrator",
    "IAgentOrchestratorFactory",
    # Tool Selector Protocol
    "IToolSelector",
    "IConfigurableToolSelector",
    "IToolSelectorFactory",
    "ToolSelectionResult",
    "ToolSelectionContext",
    "ToolSelectionStrategy",
]

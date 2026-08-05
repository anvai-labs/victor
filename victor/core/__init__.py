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

"""Core infrastructure modules for Victor.

This package provides foundational infrastructure:
- Dependency injection container (ServiceContainer)
- Lazy initialization utilities (avoiding circular imports)
- Error handling utilities
- Retry mechanisms
"""

from victor.core.container import (
    AsyncDisposable,
    ContainerFrozenError,
    ServiceContainer,
    ServiceScope,
    ServiceLifetime,
    get_container,
    set_container,
    reset_container,
)

from victor.core.lazy import (
    LazyProperty,
    deferred_import,
    SingletonFactory,
    CircularImportInfo,
    KNOWN_CIRCULAR_IMPORTS,
    get_circular_import_info,
    list_circular_imports,
)

from victor.core.protocols import (
    OrchestratorProtocol,
    TaskClassifierProtocol,
    IntentClassifierProtocol,
    EmbeddingServiceProtocol,
    ToolCallingAdapterProtocol,
    RuntimeIntelligencePipelineProtocol,
    IntelligentPipelineProtocol,
    ProviderProtocol,
    CacheProtocol,
    MetricsCollectorProtocol,
    TaskClassificationResultProtocol,
    IntentClassificationResultProtocol,
    ServiceFactory,
)

from victor.core.health import (
    BaseHealthCheck,
    CacheHealthCheck,
    CallableHealthCheck,
    ComponentHealth,
    HealthChecker,
    HealthCheckProtocol,
    HealthReport,
    HealthStatus,
    MemoryHealthCheck,
    ProviderHealthCheck,
    ToolHealthCheck,
    create_default_health_checker,
)

from victor.core.validation import (
    AgentConfigSchema,
    CacheConfigSchema,
    ConfigurationBuilder,
    ConfigValidator,
    ConfigValidationResult,
    DependencyRule,
    EnumRule,
    ModelConfigSchema,
    ObservabilityConfigSchema,
    PathRule,
    ProviderConfigSchema,
    RangeRule,
    RegexRule,
    ResilienceConfigSchema,
    ToolConfigSchema,
    ValidationIssue,
    ValidationResult,
    ValidationRule,
    ValidationSeverity,
    severity_rank,
    validate_agent_config,
    validate_model_config,
    validate_provider_config,
)

from victor.core.middleware import (
    CachingMiddleware,
    ContextKey,
    ErrorHandlingMiddleware,
    ErrorMiddleware,
    LoggingMiddleware,
    MetricsMiddleware,
    Middleware,
    MiddlewareContext,
    MiddlewarePipeline,
    PipelineBuilder,
    RequestMiddleware,
    ResponseMiddleware,
    RetryMiddleware,
    TimeoutMiddleware,
    TimingMiddleware,
    ValidationMiddleware,
    create_default_pipeline,
    create_resilient_pipeline,
)

from victor.core.event_sourcing import (
    Aggregate,
    ConcurrencyError,
    DomainEvent,
    EventDispatcher,
    EventEnvelope,
    EventSourcedRepository,
    EventStore,
    InMemoryEventStore,
    Projection,
    SQLiteEventStore,
    StateChangedEvent,
    TaskCompletedEvent,
    TaskFailedEvent,
    TaskStartedEvent,
    ToolCalledEvent,
    ToolResultEvent,
)

from victor.core.repository import (
    AndSpecification,
    AttributeSpecification,
    BaseSpecification,
    CachedRepository,
    Entity,
    EntityExistsError,
    EntityNotFoundError,
    InMemoryRepository,
    NotSpecification,
    OrSpecification,
    ReadOnlyRepository,
    Repository,
    RepositoryError,
    SQLiteRepository,
    create_cached_repository,
    create_repository,
)

from victor.core.unit_of_work import (
    CompositeUnitOfWork,
    EntityState,
    SQLiteUnitOfWork,
    TrackedEntity,
    UnitOfWork,
    UnitOfWorkError,
    UnitOfWorkProtocol,
    create_unit_of_work,
    transactional,
)

from victor.core.mode_config import (
    ModeLevel,
    ModeDefinition,
    VerticalModeConfig,
    DEFAULT_MODES,
    DEFAULT_TASK_BUDGETS,
    ModeConfigRegistry,
    RegistryBasedModeConfigProvider,
    get_mode_config,
    get_tool_budget,
    register_vertical_modes,
)

from victor.core.tool_dependency_base import (
    BaseToolDependencyProvider,
    ToolDependencyConfig,
)

from victor.core.tool_types import (
    ToolDependency,
    ToolDependencyProviderProtocol,
)

from victor.core.vertical_types import (
    StageDefinition,
    TaskTypeHint,
    StandardTaskHints,
    StandardGroundingRules,
    MiddlewarePriority as VerticalMiddlewarePriority,
    MiddlewareResult as VerticalMiddlewareResult,
    TieredToolConfig,
    TieredToolTemplate,
    VerticalConfigBase,
)

from victor.core.database import (
    DatabaseManager,
    ProjectDatabaseManager,
    get_database,
    get_project_database,
    reset_database,
    reset_project_database,
    reset_all_databases,
)

from victor.core.schema import (
    Tables,
    LearnerID,
    Schema,
    CURRENT_SCHEMA_VERSION,
    get_migration_sql,
)

# SeverityLevel is lazy-loaded to avoid import-time overhead.
# Access via: from victor.core import SeverityLevel
# or:        from victor.core.severity import SeverityLevel

__all__ = [
    # Container
    "AsyncDisposable",
    "ContainerFrozenError",
    "ServiceContainer",
    "ServiceScope",
    "ServiceLifetime",
    "get_container",
    "set_container",
    "reset_container",
    # Lazy initialization
    "LazyProperty",
    "deferred_import",
    "SingletonFactory",
    "CircularImportInfo",
    "KNOWN_CIRCULAR_IMPORTS",
    "get_circular_import_info",
    "list_circular_imports",
    # Protocols (for interface-based decoupling)
    "OrchestratorProtocol",
    "TaskClassifierProtocol",
    "IntentClassifierProtocol",
    "EmbeddingServiceProtocol",
    "ToolCallingAdapterProtocol",
    "RuntimeIntelligencePipelineProtocol",
    "IntelligentPipelineProtocol",
    "ProviderProtocol",
    "CacheProtocol",
    "MetricsCollectorProtocol",
    "TaskClassificationResultProtocol",
    "IntentClassificationResultProtocol",
    "ServiceFactory",
    # Health checks
    "BaseHealthCheck",
    "CacheHealthCheck",
    "CallableHealthCheck",
    "ComponentHealth",
    "HealthChecker",
    "HealthCheckProtocol",
    "HealthReport",
    "HealthStatus",
    "MemoryHealthCheck",
    "ProviderHealthCheck",
    "ToolHealthCheck",
    "create_default_health_checker",
    # Validation
    "AgentConfigSchema",
    "CacheConfigSchema",
    "ConfigurationBuilder",
    "ConfigValidator",
    "ConfigValidationResult",
    "DependencyRule",
    "EnumRule",
    "ModelConfigSchema",
    "ObservabilityConfigSchema",
    "PathRule",
    "ProviderConfigSchema",
    "RangeRule",
    "RegexRule",
    "ResilienceConfigSchema",
    "ToolConfigSchema",
    "ValidationIssue",
    "ValidationResult",
    "ValidationRule",
    "ValidationSeverity",
    "severity_rank",
    "validate_agent_config",
    "validate_model_config",
    "validate_provider_config",
    # Middleware
    "CachingMiddleware",
    "ContextKey",
    "ErrorHandlingMiddleware",
    "LoggingMiddleware",
    "MetricsMiddleware",
    "Middleware",
    "MiddlewareContext",
    "MiddlewarePipeline",
    "PipelineBuilder",
    "RetryMiddleware",
    "TimeoutMiddleware",
    "TimingMiddleware",
    "ValidationMiddleware",
    "create_default_pipeline",
    "create_resilient_pipeline",
    # Event Sourcing
    "Aggregate",
    "ConcurrencyError",
    "DomainEvent",
    "EventDispatcher",
    "EventEnvelope",
    "EventSourcedRepository",
    "EventStore",
    "InMemoryEventStore",
    "Projection",
    "SQLiteEventStore",
    "StateChangedEvent",
    "TaskCompletedEvent",
    "TaskFailedEvent",
    "TaskStartedEvent",
    "ToolCalledEvent",
    "ToolResultEvent",
    # Repository
    "AndSpecification",
    "AttributeSpecification",
    "BaseSpecification",
    "CachedRepository",
    "Entity",
    "EntityExistsError",
    "EntityNotFoundError",
    "InMemoryRepository",
    "NotSpecification",
    "OrSpecification",
    "ReadOnlyRepository",
    "Repository",
    "RepositoryError",
    "SQLiteRepository",
    "create_cached_repository",
    "create_repository",
    # Unit of Work
    "CompositeUnitOfWork",
    "EntityState",
    "SQLiteUnitOfWork",
    "TrackedEntity",
    "UnitOfWork",
    "UnitOfWorkError",
    "UnitOfWorkProtocol",
    "create_unit_of_work",
    "transactional",
    # Mode Configuration
    "ModeLevel",
    "ModeDefinition",
    "VerticalModeConfig",
    "DEFAULT_MODES",
    "DEFAULT_TASK_BUDGETS",
    "ModeConfigRegistry",
    "RegistryBasedModeConfigProvider",
    "get_mode_config",
    "get_tool_budget",
    "register_vertical_modes",
    # Tool Dependency Base
    "BaseToolDependencyProvider",
    "ToolDependencyConfig",
    # Tool Types
    "ToolDependency",
    "ToolDependencyProviderProtocol",
    # Vertical Types (cross-vertical abstractions)
    "StageDefinition",
    "TaskTypeHint",
    "StandardTaskHints",
    "StandardGroundingRules",
    "VerticalMiddlewarePriority",
    "VerticalMiddlewareResult",
    "TieredToolConfig",
    "TieredToolTemplate",
    "VerticalConfigBase",
    # Database (DatabaseManager now includes async methods directly)
    "DatabaseManager",
    "ProjectDatabaseManager",
    "get_database",
    "get_project_database",
    "reset_database",
    "reset_project_database",
    "reset_all_databases",
    # Schema
    "Tables",
    "LearnerID",
    "Schema",
    "CURRENT_SCHEMA_VERSION",
    "get_migration_sql",
    # Unified severity level (consolidates DangerLevel, RiskLevel, CVESeverity, etc.)
    "SeverityLevel",
]


def __getattr__(name: str):
    """Lazy import for heavy or optional modules."""
    if name == "SeverityLevel":
        from victor.core.severity import SeverityLevel

        return SeverityLevel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

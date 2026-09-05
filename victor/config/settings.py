from __future__ import annotations

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

"""Configuration management for CodingAgent."""

import threading
import logging
import os
import warnings
from pathlib import Path
from typing import Annotated, Any, Callable, ClassVar, Dict, Optional, Union, List

logger = logging.getLogger(__name__)

import yaml
from pydantic import BaseModel, Field, SecretStr, computed_field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict
from victor.config.model_capabilities import _load_tool_capable_patterns_from_yaml
from victor.config.orchestrator_constants import BUDGET_LIMITS, TOOL_SELECTION_PRESETS
from victor.core.constants import DEFAULT_VERTICAL
from victor.config.secrets import reveal_secret, unwrap_secrets

# =============================================================================
# CENTRALIZED PATH CONFIGURATION
# =============================================================================
# All Victor paths are centralized here for consistency and easy configuration.
# Project-local paths are stored in {project_root}/.victor/
# Global paths are stored in ~/.victor/
#
# SECURITY: Uses secure_paths module to protect against:
# - HOME environment variable manipulation (SEC-001)
# - Path traversal attacks via VICTOR_DIR_NAME (SEC-002)
# - Symlink attacks (SEC-003)
# =============================================================================

# Context file name (configurable)
VICTOR_CONTEXT_FILE = os.getenv("VICTOR_CONTEXT_FILE", "init.md")


def _get_secure_victor_dir_name() -> str:
    """Get validated VICTOR_DIR_NAME with path traversal protection.

    Security: Blocks path traversal attempts like '../../../etc' or '/tmp/evil'
    """
    try:
        from victor.config.secure_paths import validate_victor_dir_name

        raw_name = os.getenv("VICTOR_DIR_NAME", ".victor")
        validated_name, is_valid = validate_victor_dir_name(raw_name)
        return validated_name
    except ImportError:
        # Fallback if secure_paths not available yet (during initial import)
        return os.getenv("VICTOR_DIR_NAME", ".victor")


def _get_secure_global_victor_dir() -> Path:
    """Get global Victor directory with secure home resolution.

    Security: Validates HOME against passwd database to detect manipulation.
    """
    try:
        from victor.config.secure_paths import get_victor_dir

        return get_victor_dir()
    except ImportError:
        # Fallback if secure_paths not available yet
        return Path.home() / _get_secure_victor_dir_name()


# Directory name with validation (lazy property to avoid circular imports)
VICTOR_DIR_NAME = _get_secure_victor_dir_name()

# Global config directory with secure home resolution
GLOBAL_VICTOR_DIR = _get_secure_global_victor_dir()


class ProjectPaths:
    """Centralized path management for Victor.

    Provides consistent paths for both project-local and global storage.
    Project-local paths are preferred for isolation between projects.

    Directory structure:
        {project_root}/.victor/
        ├── init.md              # Project context (was .victor.md)
        ├── project.db           # Project-specific database (conversations, graph, entities)
        ├── embeddings/          # Vector embeddings for semantic search
        ├── index_metadata.json  # Codebase index metadata
        ├── backups/             # File edit backups
        ├── changes/             # Undo/redo history
        ├── sessions/            # Session snapshots
        └── mcp.yaml             # MCP server configuration

        ~/.victor/
        ├── victor.db            # Global database (RL, prompt opt, sessions, tool usage)
        ├── profiles.yaml        # Global profiles configuration
        ├── plugins/             # Plugins directory
        ├── cache/               # Global cache
        ├── logs/                # Log files
        └── embeddings/          # Global embedding cache (task classifier, etc.)

    **DATABASE CONSOLIDATION**: All conversation data migrated from conversation.db
    to two consolidated databases:
    - project.db: Project-specific conversations, graph, entities (local to each project)
    - victor.db: Global user data, RL learning, prompt optimization, cross-project sessions
    """

    def __init__(self, project_root: Optional[Path] = None):
        """Initialize paths for a project.

        Args:
            project_root: Project root directory. Defaults to current working directory.
        """
        self._project_root = Path(project_root) if project_root else Path.cwd()

    @property
    def project_root(self) -> Path:
        """Get project root directory."""
        return self._project_root

    # -------------------------------------------------------------------------
    # Project-local paths (stored in {project}/.victor/)
    # -------------------------------------------------------------------------

    @property
    def project_victor_dir(self) -> Path:
        """Get project-local .victor directory."""
        return self._project_root / VICTOR_DIR_NAME

    @property
    def project_context_file(self) -> Path:
        """Get project context file path (.victor/init.md)."""
        return self.project_victor_dir / VICTOR_CONTEXT_FILE

    @property
    def project_db(self) -> Path:
        """Get project-local database path (conversations, state)."""
        return self.project_victor_dir / "project.db"

    @property
    def undo_db(self) -> Path:
        """Get project-local undo/redo history database path.

        A dedicated SQLite file (sibling of ``project.db``) so the per-edit undo
        write has its own write-lock and never contends with the graph indexer's
        continuous reindex stream on ``project.db``. Rebuildable/ephemeral.
        """
        return self.project_victor_dir / "undo.db"

    @property
    def embeddings_dir(self) -> Path:
        """Get project-local embeddings directory."""
        return self.project_victor_dir / "embeddings"

    @property
    def graph_dir(self) -> Path:
        """Get project-local graph directory."""
        return self.project_victor_dir / "graph"

    @property
    def index_metadata(self) -> Path:
        """Get codebase index metadata file path."""
        return self.project_victor_dir / "index_metadata.json"

    @property
    def backups_dir(self) -> Path:
        """Get project-local backups directory."""
        return self.project_victor_dir / "backups"

    @property
    def changes_dir(self) -> Path:
        """Get project-local changes (undo/redo) directory."""
        return self.project_victor_dir / "changes"

    @property
    def sessions_dir(self) -> Path:
        """Get project-local sessions directory."""
        return self.project_victor_dir / "sessions"

    @property
    def conversations_export_dir(self) -> Path:
        """Get project-local conversations export directory."""
        return self.project_victor_dir / "conversations"

    @property
    def analysis_dir(self) -> Path:
        """Get project-local analysis checkpoint directory (.victor/analysis/).

        Stores persistent cross-session analysis artifacts with YAML frontmatter
        manifests for mtime-based staleness detection. Already gitignored via .victor/.
        """
        return self.project_victor_dir / "analysis"

    @property
    def mcp_config(self) -> Path:
        """Get project-local MCP configuration file."""
        return self.project_victor_dir / "mcp.yaml"

    # -------------------------------------------------------------------------
    # Global paths (stored in ~/.victor/)
    # -------------------------------------------------------------------------

    @property
    def global_victor_dir(self) -> Path:
        """Get global .victor directory."""
        return GLOBAL_VICTOR_DIR

    @property
    def global_profiles(self) -> Path:
        """Get global profiles.yaml path."""
        return GLOBAL_VICTOR_DIR / "profiles.yaml"

    @property
    def global_plugins_dir(self) -> Path:
        """Get global plugins directory."""
        return GLOBAL_VICTOR_DIR / "plugins"

    @property
    def global_cache_dir(self) -> Path:
        """Get global cache directory."""
        return GLOBAL_VICTOR_DIR / "cache"

    @property
    def global_logs_dir(self) -> Path:
        """Get global logs directory."""
        return GLOBAL_VICTOR_DIR / "logs"

    @property
    def global_embeddings_dir(self) -> Path:
        """Get global embeddings cache directory (for task classifier, etc.)."""
        return GLOBAL_VICTOR_DIR / "embeddings"

    @property
    def global_mcp_config(self) -> Path:
        """Get global MCP configuration file."""
        return GLOBAL_VICTOR_DIR / "mcp.yaml"

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    def ensure_project_dirs(self) -> None:
        """Create project-local directories if they don't exist."""
        self.project_victor_dir.mkdir(parents=True, exist_ok=True)
        # embeddings_dir is now at project root, not under .victor
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        self.graph_dir.mkdir(parents=True, exist_ok=True)
        self.backups_dir.mkdir(parents=True, exist_ok=True)
        self.changes_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_dir.mkdir(parents=True, exist_ok=True)

    def ensure_global_dirs(self) -> None:
        """Create global directories if they don't exist."""
        self.global_victor_dir.mkdir(parents=True, exist_ok=True)
        self.global_plugins_dir.mkdir(parents=True, exist_ok=True)
        self.global_cache_dir.mkdir(parents=True, exist_ok=True)
        self.global_logs_dir.mkdir(parents=True, exist_ok=True)
        self.global_embeddings_dir.mkdir(parents=True, exist_ok=True)

    def find_context_file(self) -> Optional[Path]:
        """Find project context file at .victor/init.md.

        Returns:
            Path to context file if found, None otherwise.

        Location: .victor/init.md (configurable via VICTOR_DIR_NAME, VICTOR_CONTEXT_FILE)
        """
        if self.project_context_file.exists():
            return self.project_context_file

        return None


# Global singleton for current project
_current_project_paths: Optional[ProjectPaths] = None


def get_project_paths(project_root: Optional[Path] = None) -> ProjectPaths:
    """Get ProjectPaths instance for a project.

    Args:
        project_root: Project root directory. If None, uses cached instance or cwd.

    Returns:
        ProjectPaths instance for the project.
    """
    global _current_project_paths

    if project_root is not None:
        return ProjectPaths(project_root)

    if _current_project_paths is None:
        _current_project_paths = ProjectPaths()

    return _current_project_paths


def set_project_root(project_root: Path) -> ProjectPaths:
    """Set the current project root and return paths.

    Args:
        project_root: Project root directory.

    Returns:
        ProjectPaths instance for the project.
    """
    global _current_project_paths
    _current_project_paths = ProjectPaths(project_root)
    return _current_project_paths


def reset_project_paths() -> None:
    """Reset the cached project paths singleton.

    This is primarily useful for testing when you need to switch project roots.
    """
    global _current_project_paths
    _current_project_paths = None


class ProviderGatewayConfig(BaseModel):
    """Per-provider Sandhi gateway mode (TD-0003 P3).

    When set, the provider's Sandhi FFI handle is pointed at the Sandhi proxy and
    presents the virtual key as a bearer token, so traffic is centrally attributed
    and budget-enforced. The provider slug is preserved so the proxy still speaks the
    right dialect (openai-compat / anthropic / responses) and routes to the
    vault-resolved upstream. Opt-in and default-off: an unset ``gateway`` leaves the
    provider in direct FFI mode (unchanged behavior).
    """

    url: str = Field(
        default="",
        description=(
            "Sandhi proxy ROOT, e.g. http://localhost:8600 — a trailing /v1 or "
            "/v1beta is accepted and stripped (the transport derives the "
            "per-family prefix). May be omitted entirely when "
            "SANDHI_GATEWAY_URL is set."
        ),
    )
    virtual_key: Optional[SecretStr] = Field(
        default=None,
        description=(
            "Virtual key (vk_...) presented to the proxy as a bearer token. Also "
            "resolvable from env (per-provider SANDHI_GATEWAY_VIRTUAL_KEY_<PROVIDER> "
            "or SANDHI_GATEWAY_VIRTUAL_KEY)."
        ),
    )

    @property
    def virtual_key_value(self) -> Optional[str]:
        """Return the plain virtual key for provider construction."""
        return reveal_secret(self.virtual_key)


class ProviderConfig(BaseSettings):
    """Configuration for a specific provider."""

    api_key: Optional[SecretStr] = None
    base_url: Optional[Union[str, List[str]]] = None
    timeout: int = 300  # 5 minutes - increased for CPU-only inference
    max_retries: int = 3
    organization: Optional[str] = None  # For OpenAI
    gateway: Optional[ProviderGatewayConfig] = Field(
        default=None,
        description="Opt-in Sandhi gateway/virtual-key mode (TD-0003 P3).",
    )

    @property
    def api_key_value(self) -> Optional[str]:
        """Return the plain API key for provider construction."""
        return reveal_secret(self.api_key)

    def to_runtime_dict(self) -> Dict[str, Any]:
        """Serialize provider config with secrets safely unwrapped.

        Excludes SecretStr fields from model_dump to prevent intermediate
        plaintext in stack traces, then adds them back explicitly.
        """
        secret_fields = {
            name
            for name, info in self.model_fields.items()
            if info.annotation is SecretStr
            or (
                hasattr(info.annotation, "__args__")
                and SecretStr in getattr(info.annotation, "__args__", ())
            )
        }
        result = self.model_dump(exclude_none=True, exclude=secret_fields)
        for name in secret_fields:
            val = getattr(self, name, None)
            if val is not None:
                result[name] = val.get_secret_value() if isinstance(val, SecretStr) else val
        # Recursively unwrap any SecretStr nested inside non-secret sub-models
        # (e.g. providers.<name>.gateway.virtual_key) so downstream consumers and
        # the transport layer receive plain strings.
        return unwrap_secrets(result)


class ProfileConfig(BaseSettings):
    """Configuration for a model profile."""

    model_config = SettingsConfigDict(extra="allow")

    provider: str = Field(..., description="Provider name (ollama, anthropic, openai, google)")
    model: str = Field(..., description="Model identifier")
    account: Optional[str] = Field(
        None,
        description=(
            "Provider account name (credential identity). Decoupled from the provider "
            "transport slug: multiple accounts can share one provider/dialect (e.g. several "
            "anthropic-dialect upstreams) while each resolves its own credential. When unset, "
            "the provider's default account is used."
        ),
    )
    temperature: float = Field(0.6, ge=0.0, le=2.0)  # ADR-013 default flip 0.7→0.6 (A/B 2026-06-22)
    temperatures: Optional[Dict[str, float]] = Field(
        None,
        description="Per-task temperature overrides (task_type -> temperature), e.g. {'plan': 0.5}. "
        "Highest precedence in the ADR-013 resolution chain; absent keys defer to settings/constants.",
    )
    max_tokens: int = Field(4096, gt=0)
    description: Optional[str] = Field(None, description="Optional profile description")
    tool_selection: Optional[Dict[str, Any]] = Field(
        None, description="Tool selection configuration for adaptive thresholds"
    )

    # -------------------------------------------------------------------------
    # Planning Configuration (P3-1)
    # -------------------------------------------------------------------------
    # Optional override for planning-specific model selection
    # Allows using a different (often more capable or cost-effective) model for
    # structured planning vs regular chat execution
    #
    # Example: Use local model for chat (fast, cheap) but cloud model for planning
    # (better at structured output, more reliable JSON generation)
    #
    # Set planning_provider and planning_model to override defaults for planning
    # If not set, uses the default provider/model from the profile
    # -------------------------------------------------------------------------

    planning_provider: Optional[str] = Field(
        None,
        description="Override provider for planning (e.g., 'deepseek', 'anthropic')",
    )
    planning_model: Optional[str] = Field(
        None,
        description="Override model for planning tasks (e.g., 'deepseek-chat', 'claude-sonnet-4-5')",
    )

    # -------------------------------------------------------------------------
    # Provider Tuning Options (P3-1)
    # -------------------------------------------------------------------------
    # These settings allow fine-tuning agent behavior per provider/model.
    # Override defaults from Settings for provider-specific optimizations.

    # Loop detection thresholds - controls when to detect repetitive tool calls
    loop_repeat_threshold: Optional[int] = Field(
        None, description="Number of identical calls before triggering loop detection"
    )
    max_continuation_prompts: Optional[int] = Field(
        None,
        description="Max consecutive continuation prompts before forcing completion",
    )

    # Quality thresholds - controls response quality requirements
    quality_threshold: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Minimum quality score to accept response"
    )
    grounding_threshold: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for grounding verification",
    )

    # Tool behavior - controls tool usage patterns
    max_tool_calls_per_turn: Optional[int] = Field(
        None, gt=0, description="Maximum tool calls allowed in single model turn"
    )
    tool_cache_enabled: Optional[bool] = Field(
        None, description="Enable/disable tool result caching for this profile"
    )
    tool_deduplication_enabled: Optional[bool] = Field(
        None, description="Enable/disable tool call deduplication"
    )

    # Timeout and session limits
    session_idle_timeout: Optional[int] = Field(
        None,
        gt=0,
        description="Maximum idle time in seconds (resets on provider response/tool execution)",
    )
    # Future: session_time_limit for total session duration cap (regardless of activity)
    timeout: Optional[int] = Field(None, gt=0, description="Request timeout in seconds")

    @field_validator("tool_selection")
    @classmethod
    def validate_tool_selection(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Validate tool_selection configuration.

        Args:
            v: Tool selection configuration dictionary

        Returns:
            Validated configuration with expanded tier shortcuts

        Raises:
            ValueError: If configuration is invalid
        """
        if v is None:
            return None

        # Predefined model size tiers for convenience (from orchestrator_constants)
        TIER_PRESETS = {
            "tiny": TOOL_SELECTION_PRESETS.tiny,  # 0.5B-3B
            "small": TOOL_SELECTION_PRESETS.small,  # 7B-8B
            "medium": TOOL_SELECTION_PRESETS.medium,  # 13B-15B
            "large": TOOL_SELECTION_PRESETS.large,  # 30B+
            "cloud": TOOL_SELECTION_PRESETS.cloud,  # Claude/GPT
        }

        # Expand tier shortcuts
        if "model_size_tier" in v:
            tier = v["model_size_tier"]
            if tier in TIER_PRESETS:
                # Apply preset values, but allow manual overrides
                preset = TIER_PRESETS[tier].copy()
                preset.update(v)  # Manual values override preset
                v = preset

        # Validate base_threshold
        if "base_threshold" in v:
            threshold = v["base_threshold"]
            if not isinstance(threshold, (int, float)):
                raise ValueError(f"base_threshold must be a number, got {type(threshold)}")
            if not (0.0 <= threshold <= 1.0):
                raise ValueError(f"base_threshold must be between 0.0 and 1.0, got {threshold}")

        # Validate base_max_tools
        if "base_max_tools" in v:
            max_tools = v["base_max_tools"]
            if not isinstance(max_tools, int):
                raise ValueError(f"base_max_tools must be an integer, got {type(max_tools)}")
            if max_tools < 1:
                raise ValueError(f"base_max_tools must be positive, got {max_tools}")

        return v


# =============================================================================
# NESTED CONFIG GROUPS
# =============================================================================
# Focused Pydantic models that group related settings by domain.
# Settings composes these and syncs flat field values into them,
# enabling both flat access (settings.default_provider) and
# structured access (settings.provider.default_provider).

# Phase 5: New config groups
from victor.config.groups import (
    ProviderSettings,
    AgentSettings,
    ServerSettings,
    CodebaseSettings,
    UsageSettings,
    UsageGatewaySettings,
    SubprocessSettings,
    HeadlessSettings,
    WorkflowSettings,
    ResponseSettings,
    CacheSettings,
    RecoverySettings,
    AnalyticsSettings,
    NetworkSettings,
    EmbeddingSettings,
    ToolSelectionSettings,
    FuzzyMatchingSettings,
    GovernanceSettings,
)

# Old config imports (to be migrated to groups)
from victor.config.tool_settings import ToolSettings  # noqa: E402
from victor.config.theme_settings import ThemeSettings  # noqa: E402
from victor.config.search_settings import SearchSettings  # noqa: E402
from victor.config.resilience_settings import ResilienceSettings  # noqa: E402
from victor.config.security_settings import SecuritySettings  # noqa: E402
from victor.config.event_settings import EventSettings  # noqa: E402
from victor.config.event_debouncing_settings import (
    EventDebouncingSettings,
)  # noqa: E402
from victor.config.observability_settings import ObservabilitySettings  # noqa: E402
from victor.config.context_settings import ContextSettings  # noqa: E402
from victor.config.checkpoint_settings import CheckpointSettings  # noqa: E402
from victor.config.ui_settings import UISettings  # noqa: E402
from victor.config.pipeline_settings import PipelineSettings  # noqa: E402
from victor.config.feature_flag_settings import FeatureFlagSettings  # noqa: E402
from victor.config.prompt_enrichment_settings import (
    PromptEnrichmentSettings,
)  # noqa: E402
from victor.config.hitl_settings import HITLSettings  # noqa: E402
from victor.config.plugin_settings import PluginSettings  # noqa: E402
from victor.config.prompt_policy_settings import PromptPolicySettings  # noqa: E402
from victor.config.conversation_settings import ConversationSettings  # noqa: E402
from victor.config.exploration_settings import ExplorationSettings  # noqa: E402

from victor.config.serialization_settings import SerializationSettings  # noqa: E402


from victor.config.automation_settings import AutomationSettings  # noqa: E402


from victor.config.code_correction_settings import CodeCorrectionSettings  # noqa: E402


from victor.config.mcp_settings import McpSettings  # noqa: E402


from victor.config.sandbox_settings import SandboxSettings  # noqa: E402


from victor.config.hooks_settings import HooksSettings  # noqa: E402


from victor.config.compaction_settings import CompactionSettings  # noqa: E402


from victor.config.permission_settings import PermissionSettings  # noqa: E402

from victor.config.prompt_optimization_settings import (
    PromptOptimizationSettings,
)  # noqa: E402

from victor.config.credit_assignment_settings import (
    CreditAssignmentSettings,
)  # noqa: E402
from victor.config.temperature_settings import TemperatureSettings  # noqa: E402

# Module-level mapping of group names to nested model classes
_NESTED_GROUPS = {
    "provider": ProviderSettings,
    "tools": ToolSettings,
    "search": SearchSettings,
    "resilience": ResilienceSettings,
    "security": SecuritySettings,
    "events": EventSettings,
    "event_debouncing": EventDebouncingSettings,
    "pipeline": PipelineSettings,
    "observability": ObservabilitySettings,
    "context": ContextSettings,
    "checkpoint": CheckpointSettings,
    "ui": UISettings,
    "feature_flags": FeatureFlagSettings,
    "enrichment": PromptEnrichmentSettings,
    "hitl": HITLSettings,
    "plugins": PluginSettings,
    "prompt_policy": PromptPolicySettings,
    "conversation": ConversationSettings,
    "exploration": ExplorationSettings,
    "serialization": SerializationSettings,
    "automation": AutomationSettings,
    "code_correction": CodeCorrectionSettings,
    "mcp": McpSettings,
    "sandbox": SandboxSettings,
    "hooks": HooksSettings,
    "compaction": CompactionSettings,
    "permissions": PermissionSettings,
    "prompt_optimization": PromptOptimizationSettings,
    "credit_assignment": CreditAssignmentSettings,
    # Phase 5: Additional nested config groups
    "agent": AgentSettings,
    "server": ServerSettings,
    "codebase": CodebaseSettings,
    "usage": UsageSettings,
    "usage_gateway": UsageGatewaySettings,
    "subprocess": SubprocessSettings,
    "headless": HeadlessSettings,
    "workflow": WorkflowSettings,
    "response": ResponseSettings,
    "cache": CacheSettings,
    "recovery": RecoverySettings,
    "analytics": AnalyticsSettings,
    "network": NetworkSettings,
    "embedding": EmbeddingSettings,
    "tool_selection": ToolSelectionSettings,
    "fuzzy_matching": FuzzyMatchingSettings,
    "governance": GovernanceSettings,
    "temperature_policy": TemperatureSettings,
}


class Settings(BaseSettings):
    """Main application settings.

    Consolidates all configuration sources with explicit precedence.
    Use from_sources() classmethod to load with proper precedence.

    Precedence (highest to lowest):
    1. CLI arguments (passed via override)
    2. Environment variables (VICTOR_*)
    3. .env file
    4. ~/.victor/settings.yaml
    5. ~/.victor/profiles.yaml
    6. Default values
    """

    model_config = SettingsConfigDict(
        env_prefix="VICTOR_",
        env_file=".env" if not os.getenv("VICTOR_SKIP_ENV_FILE") else None,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow",
    )

    # Change listeners for runtime config updates.
    # Stored at class level so all instances share the same listener list.
    _change_listeners: ClassVar[List[Callable[[str, Any, Any], None]]] = []

    @classmethod
    def add_change_listener(cls, callback: Callable[[str, Any, Any], None]) -> None:
        """Register callback for settings changes.

        Callbacks receive ``(field_name, old_value, new_value)`` when
        :meth:`notify_change` is called after a configuration update.
        """
        cls._change_listeners.append(callback)

    @classmethod
    def remove_change_listener(cls, callback: Callable) -> None:
        """Remove a previously registered change listener."""
        if callback in cls._change_listeners:
            cls._change_listeners.remove(callback)

    def notify_change(self, field_name: str, old_value: Any, new_value: Any) -> None:
        """Notify all registered listeners of a settings change.

        This must be called explicitly after updating a setting via
        ``model_copy(update=...)`` or direct attribute assignment.
        Automatic interception via ``__setattr__`` is intentionally
        avoided because Pydantic's ``BaseSettings`` relies on its own
        attribute-setting machinery during construction and validation.
        """
        if old_value == new_value:
            return
        for listener in self._change_listeners:
            try:
                listener(field_name, old_value, new_value)
            except Exception:
                pass

    @model_validator(mode="before")
    @classmethod
    def _map_legacy_environment_variables(cls, data: Any) -> Any:
        """Map legacy flat environment variables to nested structure.

        Supports both legacy and new environment variable styles:
        - Legacy: VICTOR_DEFAULT_PROVIDER -> provider.default_provider
        - New: VICTOR_PROVIDER__DEFAULT_PROVIDER -> provider.default_provider

        New style takes precedence when both are set.

        Args:
            data: Input data (dict or other)

        Returns:
            Transformed data with legacy env vars mapped to nested structure
        """
        if not isinstance(data, dict):
            return data

        import os

        # First, extract new-style nested env vars from os.environ
        # These take precedence over old-style flat env vars
        # Format: VICTOR_PROVIDER__DEFAULT_PROVIDER -> provider.default_provider
        new_style_values = {}
        for env_key, env_value in os.environ.items():
            if env_key.startswith("VICTOR_") and "__" in env_key:
                # Convert VICTOR_PROVIDER__DEFAULT_PROVIDER -> provider.default_provider
                parts = env_key.replace("VICTOR_", "", 1).split("__")  # Remove VICTOR_ prefix
                if len(parts) >= 2:
                    # This is a nested env var like PROVIDER__DEFAULT_PROVIDER
                    nested_path = ".".join([p.lower() for p in parts])
                    new_style_values[nested_path] = env_value

        # Legacy flat -> nested environment variable mappings
        LEGACY_ENV_MAPPINGS = {
            # Provider Configuration
            "default_provider": "provider.default_provider",
            "default_model": "provider.default_model",
            "default_temperature": "provider.default_temperature",
            "default_max_tokens": "provider.default_max_tokens",
            "anthropic_api_key": "provider.anthropic_api_key",
            "openai_api_key": "provider.openai_api_key",
            "google_api_key": "provider.google_api_key",
            "moonshot_api_key": "provider.moonshot_api_key",
            "deepseek_api_key": "provider.deepseek_api_key",
            "ollama_base_url": "provider.ollama_base_url",
            "lmstudio_base_urls": "provider.lmstudio_base_urls",
            "vllm_base_url": "provider.vllm_base_url",
            "lmstudio_max_vram_gb": "provider.lmstudio_max_vram_gb",
            # Agent Configuration
            "enable_planning": "agent.enable_planning",
            "planning_min_complexity": "agent.planning_min_complexity",
            "planning_show_plan": "agent.planning_show_plan",
            # Tool Selection Configuration
            "use_semantic_tool_selection": "tool_selection.use_semantic_tool_selection",
            "preload_embeddings": "tool_selection.preload_embeddings",
            "enable_tool_deduplication": "tool_selection.enable_tool_deduplication",
            "tool_deduplication_window_size": "tool_selection.tool_deduplication_window_size",
            "fallback_max_tools": "tool_selection.fallback_max_tools",
            # Tool Configuration
            "tool_cache_enabled": "tools.tool_cache_enabled",
            "tool_cache_ttl": "tools.tool_cache_ttl",
            "tool_cache_allowlist": "tools.tool_cache_allowlist",
            "tool_cache_dir": "tools.tool_cache_dir",
            "generic_result_cache_enabled": "tools.generic_result_cache_enabled",
            "tool_budget_calibration_enabled": "tools.tool_budget_calibration_enabled",
            "tool_budget_calibration_min_confidence": "tools.tool_budget_calibration_min_confidence",
            "generic_result_cache_ttl": "tools.generic_result_cache_ttl",
            "tool_selection_cache_enabled": "tools.tool_selection_cache_enabled",
            "tool_selection_cache_ttl": "tools.tool_selection_cache_ttl",
            "tool_call_budget": "tools.tool_call_budget",
            "tool_call_budget_warning_threshold": "tools.tool_call_budget_warning_threshold",
            "tool_call_budget_warning_pct": "tools.tool_call_budget_warning_pct",
            "tool_call_budget_warning_remaining": "tools.tool_call_budget_warning_remaining",
            "tool_budget_progress_relief_enabled": "tools.tool_budget_progress_relief_enabled",
            "tool_budget_progress_relief_amount": "tools.tool_budget_progress_relief_amount",
            "tool_budget_progress_relief_max_uses": "tools.tool_budget_progress_relief_max_uses",
            "tool_calling_models": "tools.tool_calling_models",
            "tool_exploration_boosts": "tools.tool_exploration_boosts",
            "tool_result_truncation": "tools.tool_result_truncation",
            "tool_retry_enabled": "tools.tool_retry_enabled",
            "tool_retry_max_attempts": "tools.tool_retry_max_attempts",
            "tool_retry_base_delay": "tools.tool_retry_base_delay",
            "tool_retry_max_delay": "tools.tool_retry_max_delay",
            "tool_validation_mode": "tools.tool_validation_mode",
            # Embedding Configuration
            "unified_embedding_model": "embedding.unified_embedding_model",
            "embedding_provider": "embedding.embedding_provider",
            "embedding_model": "embedding.embedding_model",
            # Search Configuration
            "codebase_vector_store": "search.codebase_vector_store",
            "codebase_embedding_provider": "search.codebase_embedding_provider",
            "codebase_embedding_model": "search.codebase_embedding_model",
            "codebase_persist_directory": "search.codebase_persist_directory",
            "codebase_dimension": "search.codebase_dimension",
            "codebase_batch_size": "search.codebase_batch_size",
            "codebase_structural_indexing_enabled": "search.codebase_structural_indexing_enabled",
            "codebase_chunking_strategy": "search.codebase_chunking_strategy",
            "codebase_chunk_size": "search.codebase_chunk_size",
            "codebase_chunk_overlap": "search.codebase_chunk_overlap",
            "codebase_embedding_extra_config": "search.codebase_embedding_extra_config",
            "codebase_graph_store": "search.codebase_graph_store",
            "codebase_graph_path": "search.codebase_graph_path",
            "codebase_graph_writer_mode": "search.codebase_graph_writer_mode",
            "semantic_similarity_threshold": "search.semantic_similarity_threshold",
            "semantic_query_expansion_enabled": "search.semantic_query_expansion_enabled",
            "semantic_max_query_expansions": "search.semantic_max_query_expansions",
            "enable_hybrid_search": "search.enable_hybrid_search",
            "hybrid_search_semantic_weight": "search.hybrid_search_semantic_weight",
            "hybrid_search_keyword_weight": "search.hybrid_search_keyword_weight",
            # Analytics Configuration
            "analytics_enabled": "analytics.analytics_enabled",
            "streaming_metrics_enabled": "analytics.streaming_metrics_enabled",
            "streaming_metrics_history_size": "analytics.streaming_metrics_history_size",
            "show_token_count": "analytics.show_token_count",
            "show_cost_metrics": "analytics.show_cost_metrics",
            # Recovery Configuration
            "chat_max_iterations": "recovery.chat_max_iterations",
            "max_consecutive_tool_calls": "recovery.max_consecutive_tool_calls",
            "session_idle_timeout": "recovery.session_idle_timeout",
            # Response Configuration
            "response_completion_retries": "response.response_completion_retries",
            "response_token_reserve": "response.response_token_reserve",
            # Server Configuration
            "server_api_key": "server.server_api_key",
            "server_session_secret": "server.server_session_secret",
            "server_max_sessions": "server.server_max_sessions",
            "server_max_message_bytes": "server.server_max_message_bytes",
            "server_session_ttl_seconds": "server.server_session_ttl_seconds",
            "render_max_payload_bytes": "server.render_max_payload_bytes",
            "render_timeout_seconds": "server.render_timeout_seconds",
            "render_max_concurrency": "server.render_max_concurrency",
            # Subprocess Configuration
            "code_executor_network_disabled": "subprocess.code_executor_network_disabled",
            "code_executor_memory_limit": "subprocess.code_executor_memory_limit",
            "code_executor_cpu_shares": "subprocess.code_executor_cpu_shares",
            "subprocess_resource_limits_enabled": "subprocess.subprocess_resource_limits_enabled",
            # Usage Configuration
            "usage_sampling_enabled": "usage.usage_sampling_enabled",
            # Headless Configuration
            "headless_mode": "headless.headless_mode",
            "dry_run_mode": "headless.dry_run_mode",
            "auto_approve_safe": "headless.auto_approve_safe",
            "max_file_changes": "headless.max_file_changes",
            "one_shot_mode": "headless.one_shot_mode",
            # Security Configuration
            "docker_allow_dangerous_operations": "security.docker_allow_dangerous_operations",
            # Resilience Configuration
            "resilience_enabled": "resilience.resilience_enabled",
            "circuit_breaker_failure_threshold": "resilience.circuit_breaker_failure_threshold",
            "circuit_breaker_success_threshold": "resilience.circuit_breaker_success_threshold",
            "circuit_breaker_timeout": "resilience.circuit_breaker_timeout",
            "circuit_breaker_half_open_max": "resilience.circuit_breaker_half_open_max",
            "retry_max_attempts": "resilience.retry_max_attempts",
            "retry_base_delay": "resilience.retry_base_delay",
            "retry_max_delay": "resilience.retry_max_delay",
            "retry_exponential_base": "resilience.retry_exponential_base",
            "rate_limiting_enabled": "resilience.rate_limiting_enabled",
            "rate_limit_requests_per_minute": "resilience.rate_limit_requests_per_minute",
            "rate_limit_tokens_per_minute": "resilience.rate_limit_tokens_per_minute",
            "rate_limit_max_concurrent": "resilience.rate_limit_max_concurrent",
            "rate_limit_queue_size": "resilience.rate_limit_queue_size",
            "rate_limit_num_workers": "resilience.rate_limit_num_workers",
            # UI Configuration
            "theme": "ui.theme",
            "stream_responses": "ui.stream_responses",
            # Checkpoint Configuration
            "checkpoint_enabled": "checkpoint.checkpoint_enabled",
            "checkpoint_auto_interval": "checkpoint.checkpoint_auto_interval",
            "checkpoint_max_per_session": "checkpoint.checkpoint_max_per_session",
            "checkpoint_compression_enabled": "checkpoint.checkpoint_compression_enabled",
            "checkpoint_compression_threshold": "checkpoint.checkpoint_compression_threshold",
            # HITL Configuration
            "hitl_default_timeout": "hitl.hitl_default_timeout",
            "hitl_default_fallback": "hitl.hitl_default_fallback",
            "hitl_auto_approve_low_risk": "hitl.hitl_auto_approve_low_risk",
            "hitl_keyboard_shortcuts_enabled": "hitl.hitl_keyboard_shortcuts_enabled",
            # Observability Configuration
            "enable_observability_logging": "observability.enable_observability_logging",
            "observability_log_path": "observability.observability_log_path",
            "log_level": "observability.log_level",
            "log_file": "observability.log_file",
            # Prompt Policy Configuration
            "prompt_policy_enforce_identity": "prompt_policy.prompt_policy_enforce_identity",
            "prompt_policy_enforce_guidelines": "prompt_policy.prompt_policy_enforce_guidelines",
            "prompt_policy_enforce_operating_preamble": "prompt_policy.prompt_policy_enforce_operating_preamble",
            "prompt_policy_enforce_unique_sections": "prompt_policy.prompt_policy_enforce_unique_sections",
            "prompt_policy_protected_sections": "prompt_policy.prompt_policy_protected_sections",
            "prompt_policy_max_section_chars": "prompt_policy.prompt_policy_max_section_chars",
            "prompt_policy_identity": "prompt_policy.prompt_policy_identity",
            "prompt_policy_guidelines": "prompt_policy.prompt_policy_guidelines",
            "prompt_policy_operating_template": "prompt_policy.prompt_policy_operating_template",
            "prompt_policy_fallback_template": "prompt_policy.prompt_policy_fallback_template",
            # Conversation Configuration
            "conversation_memory_enabled": "conversation.conversation_memory_enabled",
            "conversation_embeddings_enabled": "conversation.conversation_embeddings_enabled",
            # Context Configuration
            "max_context_tokens": "context.max_context_tokens",
            # Feature Flags Configuration
            "use_composition_over_inheritance": "feature_flags.use_composition_over_inheritance",
            "use_strategy_based_tool_registration": "feature_flags.use_strategy_based_tool_registration",
            # Workflow Configuration
            "workflow_definition_cache_enabled": "workflow.workflow_definition_cache_enabled",
            "workflow_definition_cache_ttl": "workflow.workflow_definition_cache_ttl",
            "workflow_definition_cache_max_entries": "workflow.workflow_definition_cache_max_entries",
            "stategraph_copy_on_write_enabled": "workflow.stategraph_copy_on_write_enabled",
            # Pipeline Configuration
            "max_exploration_iterations": "pipeline.max_exploration_iterations",
            "max_exploration_iterations_action": "pipeline.max_exploration_iterations_action",
            "max_exploration_iterations_analysis": "pipeline.max_exploration_iterations_analysis",
            "min_content_threshold": "pipeline.min_content_threshold",
            "max_research_iterations": "pipeline.max_research_iterations",
            "recovery_empty_response_threshold": "pipeline.recovery_empty_response_threshold",
            "recovery_blocked_consecutive_threshold": "pipeline.recovery_blocked_consecutive_threshold",
            "recovery_blocked_total_threshold": "pipeline.recovery_blocked_total_threshold",
            "max_continuation_prompts_analysis": "pipeline.max_continuation_prompts_analysis",
            "max_continuation_prompts_action": "pipeline.max_continuation_prompts_action",
            "max_continuation_prompts_default": "pipeline.max_continuation_prompts_default",
            "runtime_intelligence_enabled": "pipeline.runtime_intelligence_enabled",
            "intelligent_pipeline_enabled": "pipeline.runtime_intelligence_enabled",
            "intelligent_quality_scoring": "pipeline.intelligent_quality_scoring",
            "intelligent_mode_learning": "pipeline.intelligent_mode_learning",
            "intelligent_prompt_optimization": "pipeline.intelligent_prompt_optimization",
            "intelligent_grounding_verification": "pipeline.intelligent_grounding_verification",
            "intelligent_min_quality_threshold": "pipeline.intelligent_min_quality_threshold",
            "intelligent_grounding_threshold": "pipeline.intelligent_grounding_threshold",
            "intelligent_exploration_rate": "pipeline.intelligent_exploration_rate",
            "intelligent_learning_rate": "pipeline.intelligent_learning_rate",
            "intelligent_discount_factor": "pipeline.intelligent_discount_factor",
            "serialization_include_format_hint": "pipeline.serialization_include_format_hint",
            "serialization_min_rows_for_tabular": "pipeline.serialization_min_rows_for_tabular",
            "serialization_debug_mode": "pipeline.serialization_debug_mode",
        }

        # Transform legacy flat env vars to nested structure
        # Process in order of precedence:
        # 1. Direct initialization (flat_key in data) - map to nested structure
        # 2. Environment variables (VICTOR_FLAT_KEY) - map to nested structure
        # 3. New-style nested env vars (VICTOR_GROUP__FIELD) - already processed above

        for flat_key, nested_path in LEGACY_ENV_MAPPINGS.items():
            # Skip if already set by new-style env var
            if nested_path in new_style_values:
                continue

            value = None

            # Check if this flat field is in the data dict (direct initialization)
            if flat_key in data:
                value = data[flat_key]
                # Remove flat key from data - it will be replaced with nested structure
                del data[flat_key]
            else:
                # Check if this flat field is in environment variables
                # Environment variables take precedence over direct initialization
                full_env_key = f"VICTOR_{flat_key.upper()}"
                if full_env_key in os.environ:
                    # Get value from environment
                    value = os.environ[full_env_key]
                    pass  # env value set in `value` above

            # If we have a value from either source, map it to nested structure
            if value is not None:
                # Type conversion for boolean and numeric values
                value_lower = value.lower() if isinstance(value, str) else value
                if value_lower in ("true", "false"):
                    value = value_lower == "true"
                elif value_lower == "none":
                    value = None
                elif isinstance(value, str):
                    # Try to convert to int or float
                    try:
                        if "." in value:
                            value = float(value)
                        else:
                            value = int(value)
                    except ValueError:
                        pass  # Keep as string

                # Set nested value from direct initialization or environment variable
                parts = nested_path.split(".")
                current = data

                for i, part in enumerate(parts[:-1]):
                    if part not in current:
                        current[part] = {}
                    elif not isinstance(current[part], dict):
                        # Already set to non-dict value, skip
                        break
                    current = current[part]
                else:
                    # Only set if we didn't break out of the loop
                    if isinstance(current, dict):
                        current[parts[-1]] = value

        # Now add new-style values (these take precedence)
        for nested_path, env_value in new_style_values.items():
            # Type conversion
            value = env_value
            value_lower = env_value.lower() if isinstance(env_value, str) else env_value
            if value_lower in ("true", "false"):
                value = value_lower == "true"
            elif value_lower == "none":
                value = None
            elif isinstance(env_value, str):
                # Try to convert to int or float
                try:
                    if "." in env_value:
                        value = float(env_value)
                    else:
                        value = int(env_value)
                except ValueError:
                    pass  # Keep as string

            parts = nested_path.split(".")
            current = data

            for i, part in enumerate(parts[:-1]):
                if part not in current:
                    current[part] = {}
                elif not isinstance(current[part], dict):
                    break
                current = current[part]
            else:
                if isinstance(current, dict):
                    current[parts[-1]] = value

        return data

    @model_validator(mode="before")
    @classmethod
    def _handle_legacy_field_names(cls, data: Any) -> Any:
        """Handle legacy field names for backward compatibility.

        Maps old field names to new field names:
        - provider → default_provider
        - model → default_model
        - api_key → anthropic_api_key (default)

        This allows tests and code to use the old field names while
        internally using the new structured names.
        """
        if not isinstance(data, dict):
            return data

        # Handle legacy 'provider' field → 'default_provider'
        if "provider" in data and "default_provider" not in data:
            provider_value = data["provider"]
            # Only map if it's a string (not already a ProviderSettings object)
            if isinstance(provider_value, str):
                data["default_provider"] = provider_value
                # Remove the old field name to avoid validation errors
                del data["provider"]

        # Handle legacy 'model' field → 'default_model'
        if "model" in data and "default_model" not in data:
            model_value = data["model"]
            if isinstance(model_value, str):
                data["default_model"] = model_value
                del data["model"]

        # Handle legacy 'api_key' field → specific provider API keys
        if "api_key" in data:
            api_key_value = data["api_key"]
            if isinstance(api_key_value, str):
                # Try to determine which provider based on the default_provider
                default_provider = data.get("default_provider", data.get("provider", "ollama"))
                if default_provider in ["anthropic", "claude"]:
                    data["anthropic_api_key"] = api_key_value
                elif default_provider in ["openai", "gpt"]:
                    data["openai_api_key"] = api_key_value
                elif default_provider in ["google", "gemini"]:
                    data["google_api_key"] = api_key_value
                # Remove the generic api_key field
                del data["api_key"]

        return data

    # Nested config groups (structured access to grouped settings).
    # Auto-synced from flat fields by _sync_nested_groups validator.
    # Use flat access (settings.default_provider) or
    # nested access (settings.provider.default_provider).

    # NOTE: For backward compatibility, these fields accept None during construction
    # and are populated by the _sync_nested_groups validator from flat fields.
    provider: Optional[ProviderSettings] = Field(default=None, exclude=True, repr=False)
    tools: Optional[ToolSettings] = Field(default=None, exclude=True, repr=False)
    search: Optional[SearchSettings] = Field(default=None, exclude=True, repr=False)
    resilience: Optional[ResilienceSettings] = Field(default=None, exclude=True, repr=False)
    security: Optional[SecuritySettings] = Field(default=None, exclude=True, repr=False)
    events: Optional[EventSettings] = Field(default=None, exclude=True, repr=False)
    event_debouncing: Optional[EventDebouncingSettings] = Field(
        default=None, exclude=True, repr=False
    )
    pipeline: Optional[PipelineSettings] = Field(default=None, exclude=True, repr=False)
    observability: Optional[ObservabilitySettings] = Field(default=None, exclude=True, repr=False)
    context: Optional[ContextSettings] = Field(default=None, exclude=True, repr=False)
    checkpoint: Optional[CheckpointSettings] = Field(default=None, exclude=True, repr=False)
    ui: Optional[UISettings] = Field(default=None, exclude=True, repr=False)
    feature_flags: Optional[FeatureFlagSettings] = Field(default=None, exclude=True, repr=False)
    enrichment: Optional[PromptEnrichmentSettings] = Field(default=None, exclude=True, repr=False)
    hitl: Optional[HITLSettings] = Field(default=None, exclude=True, repr=False)
    plugins: Optional[PluginSettings] = Field(default=None, exclude=True, repr=False)
    prompt_policy: Optional[PromptPolicySettings] = Field(default=None, exclude=True, repr=False)
    conversation: Optional[ConversationSettings] = Field(default=None, exclude=True, repr=False)
    exploration: Optional[ExplorationSettings] = Field(default=None, exclude=True, repr=False)
    serialization: Optional[SerializationSettings] = Field(default=None, exclude=True, repr=False)
    automation: Optional[AutomationSettings] = Field(default=None, exclude=True, repr=False)
    code_correction: Optional[CodeCorrectionSettings] = Field(
        default=None, exclude=True, repr=False
    )
    mcp: Optional[McpSettings] = Field(default=None, exclude=True, repr=False)
    sandbox: Optional[SandboxSettings] = Field(default=None, exclude=True, repr=False)
    hooks: Optional[HooksSettings] = Field(default=None, exclude=True, repr=False)
    compaction: Optional[CompactionSettings] = Field(default=None, exclude=True, repr=False)
    permissions: Optional[PermissionSettings] = Field(default=None, exclude=True, repr=False)
    prompt_optimization: Optional[PromptOptimizationSettings] = Field(
        default=None, exclude=True, repr=False
    )
    credit_assignment: Optional[CreditAssignmentSettings] = Field(
        default=None, exclude=True, repr=False
    )

    # Phase 5: Additional nested config groups
    agent: Optional[AgentSettings] = Field(default=None, exclude=True, repr=False)
    server: Optional[ServerSettings] = Field(default=None, exclude=True, repr=False)
    codebase: Optional[CodebaseSettings] = Field(default=None, exclude=True, repr=False)
    usage: Optional[UsageSettings] = Field(default=None, exclude=True, repr=False)
    usage_gateway: Optional[UsageGatewaySettings] = Field(default=None, exclude=True, repr=False)
    subprocess: Optional[SubprocessSettings] = Field(default=None, exclude=True, repr=False)
    headless: Optional[HeadlessSettings] = Field(default=None, exclude=True, repr=False)
    workflow: Optional[WorkflowSettings] = Field(default=None, exclude=True, repr=False)
    response: Optional[ResponseSettings] = Field(default=None, exclude=True, repr=False)
    cache: Optional[CacheSettings] = Field(default=None, exclude=True, repr=False)
    recovery: Optional[RecoverySettings] = Field(default=None, exclude=True, repr=False)
    analytics: Optional[AnalyticsSettings] = Field(default=None, exclude=True, repr=False)
    network: Optional[NetworkSettings] = Field(default=None, exclude=True, repr=False)
    embedding: Optional[EmbeddingSettings] = Field(default=None, exclude=True, repr=False)
    tool_selection: Optional[ToolSelectionSettings] = Field(default=None, exclude=True, repr=False)
    fuzzy_matching: Optional[FuzzyMatchingSettings] = Field(default=None, exclude=True, repr=False)
    governance: Optional[GovernanceSettings] = Field(default=None, exclude=True, repr=False)
    temperature_policy: Optional[TemperatureSettings] = Field(
        default=None, exclude=True, repr=False
    )

    tool_settings: Optional[ToolSettings] = Field(
        default_factory=ToolSettings, exclude=True, repr=False
    )

    theme_settings: Optional[ThemeSettings] = Field(
        default_factory=ThemeSettings, exclude=True, repr=False
    )

    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None

    # Observability Logging (JSONL export for dashboard)
    # When enabled, writes all EventBus events to ~/.victor/metrics/victor.jsonl
    # This allows the dashboard to show events from agent runs
    # Off by default for performance - enable with: victor chat --log-events
    # NOTE: enable_observability_logging, observability_log_path now in observability nested group

    # Privacy and Security
    airgapped_mode: bool = False

    # ==========================================================================
    # Vertical Configuration
    # ==========================================================================
    # Verticals are domain-specific configurations that customize Victor's behavior.
    # Available verticals: coding, research, devops (extensible via plugins)
    default_vertical: str = DEFAULT_VERTICAL  # Default vertical when --vertical not specified
    auto_detect_vertical: bool = False  # Auto-detect vertical from project context (experimental)

    # Server Security (FastAPI/WebSocket layer)
    # NOTE: server_* fields now in server nested group

    # Diagram rendering limits
    # NOTE: render_* fields now in server nested group

    # Code execution sandbox defaults (used by code_executor_tool)
    # NOTE: code_executor_* fields now in subprocess nested group

    # Subprocess resource limits (POSIX rlimit for tool subprocesses)
    # When enabled, applies memory/CPU/FD limits via preexec_fn.
    # Defaults to False — opt-in to avoid breaking existing workflows.
    # NOTE: subprocess_resource_limits_enabled now in subprocess nested group

    # Usage log semantic sampling (reduces disk I/O for noisy events)
    # NOTE: usage_sampling_enabled now in usage nested group

    usage_content_sample_rate: int = 10  # Emit 1 in N content-chunk events
    usage_dedup_window_seconds: float = 5.0  # Dedup window for progress events

    # Write Approval Mode (safety for autonomous/task mode)
    # Controls when user confirmation is required for file modifications:
    #   - "off": Never require approval (dangerous, testing only)
    #   - "risky_only": Only for HIGH/CRITICAL risk operations (default)
    #   - "all_writes": Require for ALL write operations (recommended for task mode)
    write_approval_mode: str = "risky_only"

    # Headless Mode Settings (for CI/CD and automation)
    # NOTE: These fields are now in headless nested group

    # Unified Embedding Model (Optimized for Memory + Cache Efficiency)
    # Using same model for tool selection AND codebase search provides:
    # - 40% memory reduction (130MB vs 200MB)
    # - Better OS page cache utilization (1 model file instead of 2)
    # - Improved CPU L2/L3 cache hit rates
    # - Simpler management (1 model to download/update)
    #
    # Model: BAAI/bge-small-en-v1.5 (130MB, 384-dim, ~6ms)
    # - MTEB score: 62.2 (vs 58.8 for all-MiniLM-L6-v2)
    # - Excellent for code search (trained on code-related tasks)
    # - CPU-optimized, works great on consumer-grade hardware
    # - Native sentence-transformers support (no API needed)
    # Tool Selection Strategy
    # NOTE: These fields are now in tool_selection and embedding nested groups

    # Codebase Semantic Search (Air-gapped by Default)
    # NOTE: These fields are now in search nested group

    # Semantic Search Quality Improvements (P4.X - Multi-Provider Excellence)
    # NOTE: These fields are now in search nested group

    # RL-based threshold learning per (embedding_model, task_type, tool_context)
    enable_semantic_threshold_rl_learning: bool = False  # Enable automatic threshold learning
    semantic_threshold_overrides: dict = {}  # Format: {"model:task:tool": threshold}

    # Tool call deduplication
    # NOTE: These fields are now in tool_selection nested group

    # UI
    # NOTE: theme, show_token_count, show_cost_metrics, stream_responses, use_emojis now in ui nested group

    # MCP
    use_mcp_tools: bool = False
    mcp_command: Optional[str] = None  # e.g., "python mcp_server.py" or "node mcp-server.js"
    mcp_prefix: str = "mcp"

    # Tool Execution Settings
    # NOTE: These fields are now in tools nested group

    # Models known to support structured tool calls per provider
    # Loaded from model_capabilities.yaml, can be extended in profiles.yaml
    # Computed field that delegates to tools.tool_calling_models for consistency
    @computed_field  # type: ignore[misc]
    @property
    def tool_calling_models(self) -> Dict[str, list[str]]:
        """Return tool_calling_models from tools nested config."""
        if self.tools is None:
            return _load_tool_capable_patterns_from_yaml()
        return self.tools.tool_calling_models

    # Tool Retry Settings
    # NOTE: These fields are now in tools nested group

    # Tool selection fallback
    # NOTE: These fields are now in tool_selection nested group

    # Autonomous Planning Settings
    # NOTE: All planning fields now in agent nested group (enable_planning, planning_min_complexity, planning_show_plan)

    # Tool result caching (opt-in per tool)
    # NOTE: These fields are now in cache nested group

    # Generic runtime cache for non-tool payloads (feature-flagged integration path)
    # NOTE: These fields are now in cache nested group

    # Tool selection result cache for embedding-based selection
    # Caches semantic tool selection results to avoid repeated embedding computation
    # Typical 20-40% latency reduction for conversational agents
    # NOTE: These fields are now in cache nested group

    # Shared HTTP connection pool for network tools (feature-flagged integration path)
    # NOTE: These fields are now in cache nested group

    # Startup/runtime preloading coordinator for warm-path dependencies.
    framework_preload_enabled: bool = (
        True  # Enable preloading by default for 50-70% first-request latency reduction
    )
    framework_preload_parallel: bool = True

    # Strict mode for blocking private attribute fallbacks in framework integration.
    framework_private_fallback_strict_mode: bool = False
    # Strict mode for blocking non-registry protocol fallback probes in framework integration.
    framework_protocol_fallback_strict_mode: bool = False

    # Tool Argument Validation
    # Controls pre-execution JSON Schema validation of tool arguments
    # Options: "strict" (block on errors), "lenient" (warn only), "off" (disable)
    tool_validation_mode: str = "lenient"

    # Context Compaction Settings
    # Controls how conversation history is managed when context grows too large
    # Options: "simple" (keep N recent), "tiered" (prioritize tool results),
    #          "semantic" (use embeddings), "hybrid" (combine tiered + semantic)
    context_compaction_strategy: str = "tiered"
    context_min_messages_to_keep: int = 6  # Minimum messages to retain after compaction
    context_tool_retention_weight: float = 1.5  # Boost for tool result retention
    context_recency_weight: float = 2.0  # Boost for recent messages
    context_semantic_threshold: float = 0.3  # Min similarity for semantic retention

    # ==========================================================================
    # Checkpoint Settings (Time-Travel Debugging)
    # ==========================================================================
    # Controls state checkpointing for conversation replay, forking, and debugging.
    # Inspired by LangGraph's checkpoint system.
    # NOTE: checkpoint_* fields now in checkpoint nested group

    # ==========================================================================
    # HITL Settings (Human-in-the-Loop)
    # ==========================================================================
    # Controls human-in-the-loop workflow interrupts for approval/review/choice.
    # Integrates with SafetyChecker for high-risk action approval.
    # NOTE: hitl_* fields now in hitl nested group

    # ==========================================================================
    # Workflow Definition Cache (P1 Scalability)
    # ==========================================================================
    # NOTE: These fields are now in workflow nested group

    # ==========================================================================
    # StateGraph Copy-on-Write (P2 Scalability)
    # ==========================================================================
    # Enables copy-on-write state management for StateGraph workflows.
    # Delays deep copy of state until the first mutation, reducing overhead
    # for read-heavy workflows where nodes often only read state.
    #
    # Benefits:
    # - Reduced memory allocations for read-only nodes
    # - Lower CPU overhead from avoided deep copies
    # - Better performance for large state objects
    #
    # Trade-offs:
    # - Slightly more complex debugging (state may be shared until mutation)
    # - First mutation incurs full deep copy cost
    # NOTE: stategraph_copy_on_write_enabled now in workflow nested group

    # ==========================================================================
    # Feature Flags (Architecture Rollout)
    # ==========================================================================
    # Feature flags guard optional architecture and optimization behavior that
    # still ships behind rollout controls. The canonical agent-service layer is
    # no longer flag-gated; service bootstrap is unconditional.
    #
    # Enable via environment variables such as:
    #   VICTOR_USE_COMPOSITION_OVER_INHERITANCE=true
    # Or via YAML config: ~/.victor/features.yaml
    #
    # Phase 4 - Vertical Composition:
    #   - Use composition over inheritance for vertical capabilities
    #   - Enables OCP compliance (add capabilities without modifying base)
    #
    # Phase 5 - Tool Registration Strategy:
    #   - Strategy pattern for extensible tool registration
    #   - Enables OCP compliance (add tool types without modifying registry)

    # Phase 4 - Vertical Composition flag
    # NOTE: use_composition_over_inheritance now in feature_flags nested group

    # Phase 5 - Tool Registration Strategy flag
    # NOTE: use_strategy_based_tool_registration now in feature_flags nested group

    # ==========================================================================
    # Smart Routing Settings (Phase 11 - Intelligent Provider Selection)
    # ==========================================================================
    # Automatic provider routing based on health, resources, cost, latency, and performance.
    # Enables local→cloud fallback, cost optimization, and adaptive learning.
    #
    # When enabled:
    # - Automatically selects best provider for each request
    # - Falls back to alternative providers on failures
    # - Learns from provider performance over time (uses rl_provider_stat table)
    # - Respects user's explicit --provider choice (never overrides)
    #
    # Production data verified (2026-04-29): 140 provider stats with 90.5% success rate
    # Enable via: victor chat --routing-profile balanced
    # Disable via: victor chat --disable-smart-routing or VICTOR_SMART_ROUTING_ENABLED=false
    smart_routing_enabled: bool = (
        True  # Master switch for smart routing (ON by default with production data)
    )
    smart_routing_profile: str = (
        "balanced"  # Routing profile (balanced, cost-optimized, performance, local-first)
    )
    smart_routing_fallback_chain: Optional[List[str]] = (
        None  # Custom fallback chain (overrides profile)
    )
    smart_routing_performance_window: int = 100  # Number of requests for learning
    smart_routing_learning_enabled: bool = True  # Enable adaptive learning
    smart_routing_resource_awareness: bool = True  # Enable GPU/API quota detection

    # ==========================================================================
    # Prompt Enrichment Settings (Auto Optimization)
    # ==========================================================================
    # Controls automatic prompt enrichment with contextual information.
    # Enrichment adds relevant context from knowledge graph, web search,
    # conversation history, etc. to improve prompt quality.
    #
    # Trade-off: Higher quality responses vs. latency overhead (~500ms max)
    # Disable for simple tasks or when agents are confident in context.
    prompt_enrichment_enabled: bool = True  # Master toggle for prompt enrichment
    prompt_enrichment_max_tokens: int = 2000  # Max tokens to add via enrichment
    prompt_enrichment_timeout_ms: float = 500.0  # Timeout in milliseconds
    prompt_enrichment_cache_enabled: bool = True  # Cache enrichments for repeated prompts
    prompt_enrichment_cache_ttl: int = 300  # Cache TTL in seconds (5 minutes)
    prompt_enrichment_strategies: List[str] = Field(
        default_factory=lambda: ["knowledge_graph", "conversation", "web_search"],
        description="Enabled enrichment strategies (order matters for priority)",
    )
    # Per-vertical enrichment toggles (when prompt_enrichment_enabled=True)
    prompt_enrichment_coding: bool = True  # Knowledge graph, code snippets
    prompt_enrichment_research: bool = True  # Web search, citations
    prompt_enrichment_devops: bool = True  # Infrastructure context
    prompt_enrichment_data_analysis: bool = True  # Schema context, query patterns

    # Plugin System
    plugin_enabled: bool = True  # Enable plugin system
    # Note: plugin_dirs now uses get_project_paths().global_plugins_dir
    plugin_packages: List[str] = []  # Python packages to load as plugins
    plugin_disabled: List[str] = []  # List of plugin names to disable
    plugin_config: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Plugin-specific configuration (plugin_name -> config dict)",
    )

    # Security scan extensions
    security_dependency_scan: bool = False
    security_iac_scan: bool = False

    # Docker security
    # NOTE: docker_allow_dangerous_operations now in security nested group

    # LMStudio resource guard
    lmstudio_max_vram_gb: Optional[float] = (
        48.0  # Cap model selection to this budget (GB); override via env/config
    )

    # Exploration Loop Settings (prevents endless exploration without output)
    # Higher values = more thorough exploration, slower responses
    # Significantly increased to match Claude Code's unlimited exploration approach
    # Mode multipliers further increase these (PLAN: 10x, EXPLORE: 20x)
    # NOTE: max_exploration_iterations* fields now in pipeline nested group

    # ==========================================================================
    # Recovery & Loop Detection Thresholds
    # ==========================================================================
    # These control when Victor forces completion after detecting stuck behavior.
    # Lower values = faster recovery but may cut off legitimate long operations.
    # Higher values = more patience but may waste tokens on stuck loops.
    # NOTE: recovery_* fields now in pipeline nested group

    # Continuation prompts: How many times to prompt model to continue before forcing
    # These are global defaults - can be overridden per provider/model via RL learning
    # NOTE: max_continuation_prompts_* fields now in pipeline nested group

    # Provider/model-specific continuation prompt overrides (learned via RL)
    # Format: {"provider:model": {"analysis": N, "action": N, "default": N}}
    # Example: {"ollama:qwen3-coder-tools:30b": {"analysis": 8, "action": 6, "default": 4}}
    continuation_prompt_overrides: dict = {}

    # Enable RL-based learning of optimal continuation prompts per provider/model
    # Tracks success rates and adjusts limits automatically
    enable_continuation_rl_learning: bool = True

    # DEPRECATED (default False): the blocking "Did this resolve your task?"
    # prompt after each completed turn. Feedback is now passive — the reward
    # loop records outcomes automatically and explicit feedback lives in
    # /rate. Setting this to True restores the legacy per-turn prompt; the
    # task_completion decision is logged either way.
    enable_rl_feedback_prompt: bool = False

    # Session idle timeout: Maximum seconds of inactivity before forcing completion
    # Timer resets on each provider response or tool execution
    # Set below provider timeout (300s default) to provide graceful completion
    # Can be overridden per profile in profiles.yaml
    # NOTE: session_idle_timeout now in recovery nested group

    # Future: session_time_limit will be separate config for total session duration
    # regardless of activity (for sub-task agents, resource limits, etc.)

    # ==========================================================================
    # Conversation Memory (Multi-turn Context Retention)
    # ==========================================================================
    # NOTE: conversation_memory_enabled, conversation_embeddings_enabled now in conversation nested group
    # NOTE: max_context_tokens now in context nested group

    # ==========================================================================
    # Provider Resilience (Circuit Breaker, Retry, Fallback)
    # ==========================================================================
    # NOTE: resilience_* fields now in resilience nested group

    # Circuit Breaker Settings
    # NOTE: circuit_breaker_* fields now in resilience nested group

    # Retry Settings
    # NOTE: retry_* fields now in resilience nested group

    # ==========================================================================
    # Rate Limiting (Request Throttling)
    # ==========================================================================
    # NOTE: rate_limiting_* fields now in resilience nested group

    # ==========================================================================
    # Streaming Metrics (Performance Monitoring)
    # ==========================================================================
    # NOTE: streaming_metrics_* fields now in analytics nested group

    # ==========================================================================
    # Serialization (Token Optimization for Tool Output)
    # ==========================================================================
    # Controls how tool outputs are serialized for token efficiency.
    # Formats: json, json_minified, toon, csv, markdown_table, reference_encoded
    # TOON (Token-Oriented Object Notation) provides 30-60% savings for tabular data.
    # NOTE: serialization_* fields now in pipeline nested group
    serialization_default_format: Optional[str] = None  # None = auto-select best format
    serialization_min_savings_threshold: float = 0.15  # Min savings to use alternative format

    # ==========================================================================
    # Intelligent Agent Pipeline (RL-based Learning, Quality Scoring)
    # ==========================================================================
    # Controls the intelligent agent features including:
    # - Q-learning based mode transitions (explore -> plan -> build -> review)
    # - Response quality scoring (coherence, completeness, relevance, grounding)
    # - Provider resilience integration (circuit breaker, retries)
    # - Embedding-based prompt optimization
    # NOTE: intelligent_* fields now in pipeline nested group

    # Quality thresholds
    # NOTE: intelligent_*_threshold fields now in pipeline nested group

    # Learning rate for Q-learning (default exploration rate = 0.3, decay = 0.995)
    # NOTE: intelligent_*_rate, *_learning, *_discount_factor now in pipeline nested group

    # Tool-specific exploration boosts (for rebuilding reputation after fixes)
    # Maps tool names to exploration boost multipliers (e.g., {"graph": 5.0})
    # Higher values = more likely to explore this tool despite low Q-value
    # Use after significant tool improvements to accelerate relearning
    # NOTE: tool_exploration_boosts now in tools nested group

    # Serialization settings
    # NOTE: serialization_* fields now in pipeline nested group

    # ==========================================================================
    # Skill Auto-Selection
    # ==========================================================================
    # Embedding-based skill detection in victor chat.
    # When enabled, each user message is matched against skill definitions
    # via cosine similarity. High-confidence matches inject the skill's
    # prompt fragment automatically. Ambiguous matches can use the edge
    # LLM for resolution when USE_EDGE_MODEL is enabled.
    skill_auto_select_enabled: bool = True
    skill_auto_select_high_threshold: float = 0.65  # Above: use skill directly
    skill_auto_select_low_threshold: float = 0.45  # Below: no skill
    skill_auto_select_use_edge_fallback: bool = True  # Edge LLM for ambiguous zone
    skill_auto_select_log_selections: bool = True  # Log which skill was selected

    # ==========================================================================
    # Event System Configuration (Canonical core/events)
    # ==========================================================================
    # Centralized configuration for the unified event system.
    # Replaces legacy EventBus configuration (now in victor.core.events).
    # Can be overridden via environment variables (e.g., VICTOR_EVENT_BACKEND_TYPE=redis)

    # Event Backend Type: in_memory | sqlite | redis | kafka | sqs | rabbitmq
    # - in_memory: In-memory backend (default, for same-process scenarios)
    # - sqlite: SQLite persistent backend (for single-process persistence)
    # - redis: Redis streams backend (for distributed systems)
    # - kafka: Apache Kafka backend (for high-throughput distributed systems)
    # - sqs: AWS SQS backend (for serverless AWS architectures)
    # - rabbitmq: RabbitMQ backend (for traditional message queues)
    #
    # Note: Backend implementations are registered via create_event_backend() factory.
    # External backends (kafka, sqs, rabbitmq, redis) require additional dependencies.
    event_backend_type: str = "in_memory"

    # Event backend lazy initialization
    # When True, backend objects for non-in-memory types are created only on first
    # publish/subscribe/connect call instead of at service construction time.
    # This reduces startup overhead for optional distributed backends.
    event_backend_lazy_init: bool = True

    # Event delivery guarantee: at_most_once | at_least_once | exactly_once
    # - at_most_once: Best effort (may lose events, high performance)
    # - at_least_once: Guaranteed delivery (may duplicate, requires deduplication)
    # - exactly_once: Exactly once delivery (requires idempotency)
    event_delivery_guarantee: str = "at_most_once"

    # Event batching configuration
    event_max_batch_size: int = 100
    event_flush_interval_ms: float = 1000.0
    event_queue_maxsize: int = 10000
    event_queue_overflow_policy: str = "drop_newest"
    event_queue_overflow_block_timeout_ms: float = 50.0
    event_queue_overflow_topic_policies: Dict[str, str] = Field(
        default_factory=lambda: {
            # Critical lifecycle/integration/error events should prefer bounded blocking
            "lifecycle.session.*": "block_with_timeout",
            "vertical.applied": "block_with_timeout",
            "error.*": "block_with_timeout",
            # High-volume telemetry should preserve recency and avoid hard stalls
            "core.events.emit_sync.metrics": "drop_oldest",
            "vertical.extensions.loader.metrics": "drop_oldest",
        }
    )
    event_queue_overflow_topic_block_timeout_ms: Dict[str, float] = Field(
        default_factory=lambda: {
            "lifecycle.session.*": 150.0,
            "vertical.applied": 120.0,
            "error.*": 200.0,
        }
    )

    # Sync emit metrics reporter configuration
    # Emits periodic snapshots for emit_helper delivery counters.
    # Topic defaults to "core.events.emit_sync.metrics".
    # Reporter is disabled by default to avoid background thread overhead
    # unless explicitly enabled.
    event_emit_sync_metrics_enabled: bool = False
    event_emit_sync_metrics_interval_seconds: float = 60.0
    event_emit_sync_metrics_reset_after_emit: bool = False
    event_emit_sync_metrics_topic: str = "core.events.emit_sync.metrics"

    # Extension-loader pressure/reporter defaults (P3 reliability)
    extension_loader_warn_queue_threshold: int = 24
    extension_loader_error_queue_threshold: int = 32
    extension_loader_warn_in_flight_threshold: int = 6
    extension_loader_error_in_flight_threshold: int = 8
    extension_loader_pressure_cooldown_seconds: float = 5.0
    extension_loader_emit_pressure_events: bool = False
    extension_loader_metrics_reporter_enabled: bool = False
    extension_loader_metrics_reporter_interval_seconds: float = 60.0
    extension_loader_metrics_reporter_reset_after_emit: bool = False
    extension_loader_metrics_reporter_topic: str = "vertical.extensions.loader.metrics"

    # Analytics
    analytics_enabled: bool = True
    # Note: analytics_log_file now uses get_project_paths().global_logs_dir / "usage.jsonl"

    # ==========================================================================
    # Code Correction (from VictorSettings merge)
    # ==========================================================================
    code_correction_enabled: bool = Field(
        default=True, description="Enable automatic code correction suggestions"
    )
    code_correction_auto_fix: bool = Field(
        default=True, description="Automatically apply code corrections"
    )
    code_correction_max_iterations: int = Field(
        default=3, ge=1, description="Maximum correction iterations per file"
    )

    # ==========================================================================
    # Auto Commit (from VictorSettings merge)
    # ==========================================================================
    auto_commit_enabled: bool = Field(
        default=False, description="Enable automatic git commits for file changes"
    )

    # ==========================================================================
    # Exploration Loop Extended (from VictorSettings merge)
    # ==========================================================================
    # NOTE: chat_max_iterations and max_consecutive_tool_calls now in recovery nested group

    # ==========================================================================
    # Context Compaction Advanced (from VictorSettings merge)
    # ==========================================================================
    context_proactive_compaction: bool = Field(
        default=True, description="Enable proactive compaction before hitting limit"
    )
    context_proactive_threshold: float = Field(
        default=0.90,
        ge=0.5,
        le=0.99,
        description="Trigger compaction at this fraction of max tokens",
    )
    context_min_messages_after_compact: int = Field(
        default=8, ge=1, description="Minimum messages after compaction"
    )
    context_truncation_strategy: str = Field(
        default="smart", description="Truncation strategy: simple, smart, preserve_code"
    )
    file_structure_threshold: int = Field(
        default=50000,
        ge=1000,
        description="File size threshold for structure-based truncation",
    )

    # ==========================================================================
    # Conversation History (from VictorSettings merge)
    # ==========================================================================
    max_context_chars: Optional[int] = Field(
        default=None,
        ge=1000,
        description="Maximum characters in context (alternative to token-based limit)",
    )
    max_conversation_history: int = Field(
        default=100,
        ge=10,
        description="Maximum messages to retain in conversation history",
    )

    # ==========================================================================
    # Tool Output Limits (from VictorSettings merge)
    # ==========================================================================
    max_tool_output_chars: int = Field(
        default=15000, ge=100, description="Maximum characters in tool output"
    )
    max_tool_output_lines: int = Field(
        default=200, ge=10, description="Maximum lines in tool output"
    )
    tool_result_truncation: bool = Field(
        default=True, description="Enable truncation of large tool results"
    )

    # ==========================================================================
    # Parallel Tool Execution (from VictorSettings merge)
    # ==========================================================================
    parallel_tool_execution: bool = Field(
        default=False,
        description=(
            "Enable concurrent execution of read-only-allowlisted tool calls "
            "within a batch (co-design review item 16). Off by default — the "
            "allowlist gate (registry write-tools complement intersected with "
            "the idempotent-tools set; shell is never eligible) is the "
            "soundness floor, but this still changes execution ordering/timing "
            "for a broad swath of tool calls, so it ships opt-in."
        ),
    )
    max_concurrent_tools: int = Field(
        default=5, ge=1, description="Maximum tools to execute concurrently"
    )

    # ==========================================================================
    # Response Completion (from VictorSettings merge)
    # ==========================================================================
    # NOTE: response_completion_retries now in response nested group
    force_response_on_error: bool = Field(
        default=True, description="Force synthesis even if errors occurred"
    )

    # ==========================================================================
    # Grounding (from VictorSettings merge)
    # ==========================================================================
    grounding_confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for grounding verification",
    )

    # ==========================================================================
    # Provider Resilience Advanced (from VictorSettings merge)
    # ==========================================================================
    provider_health_checks: bool = Field(
        default=True, description="Enable periodic provider health checks"
    )
    provider_auto_fallback: bool = Field(
        default=True,
        description="Automatically fallback to secondary providers on failure",
    )
    fallback_providers: List[str] = Field(
        default_factory=list, description="Ordered list of fallback providers"
    )

    # ==========================================================================
    # Plugin Directories (from VictorSettings merge)
    # ==========================================================================
    plugin_dirs: List[str] = Field(
        default_factory=list, description="Additional directories to scan for plugins"
    )
    disabled_plugins: List[str] = Field(
        default_factory=list,
        description="List of plugin names to disable (renamed from plugin_disabled)",
    )

    # ==========================================================================
    # Tool Cache Storage (from VictorSettings merge)
    # ==========================================================================
    # NOTE: tool_cache_dir now in cache nested group

    # ==========================================================================
    # Subagent Orchestration (from VictorSettings merge)
    # ==========================================================================
    subagent_orchestration_enabled: bool = Field(
        default=True, description="Enable hierarchical subagent task decomposition"
    )

    # ==========================================================================
    # Observability (from VictorSettings merge)
    # ==========================================================================
    # NOTE: enable_observability - not in nested groups, keeping for now

    # ==========================================================================
    # Debug Settings (from VictorSettings merge)
    # ==========================================================================
    debug_logging: bool = Field(default=False, description="Enable verbose debug logging")

    # ==========================================================================
    # System Prompt Policy (from VictorSettings merge)
    # ==========================================================================
    # NOTE: All prompt_policy_* fields now in prompt_policy nested group

    @field_validator("event_queue_overflow_policy")
    @classmethod
    def validate_event_queue_overflow_policy(cls, value: str) -> str:
        """Validate configured in-memory queue overflow policy."""
        normalized = str(value).strip().lower()
        allowed = {"drop_newest", "drop_oldest", "block_with_timeout"}
        if normalized not in allowed:
            allowed_csv = ", ".join(sorted(allowed))
            raise ValueError(
                f"event_queue_overflow_policy must be one of: {allowed_csv}; got '{value}'"
            )
        return normalized

    @field_validator("event_queue_overflow_topic_policies")
    @classmethod
    def validate_event_queue_overflow_topic_policies(
        cls,
        value: Dict[str, str],
    ) -> Dict[str, str]:
        """Validate per-topic overflow policy overrides."""
        if not isinstance(value, dict):
            raise ValueError("event_queue_overflow_topic_policies must be a dict[str, str]")

        allowed = {"drop_newest", "drop_oldest", "block_with_timeout"}
        normalized: Dict[str, str] = {}
        for topic_pattern, policy in value.items():
            pattern = str(topic_pattern).strip()
            if not pattern:
                raise ValueError("event_queue_overflow_topic_policies keys must be non-empty")
            normalized_policy = str(policy).strip().lower()
            if normalized_policy not in allowed:
                allowed_csv = ", ".join(sorted(allowed))
                raise ValueError(
                    "event_queue_overflow_topic_policies values must be one of: "
                    f"{allowed_csv}; got '{policy}' for key '{topic_pattern}'"
                )
            normalized[pattern] = normalized_policy
        return normalized

    @field_validator("event_queue_overflow_topic_block_timeout_ms")
    @classmethod
    def validate_event_queue_overflow_topic_block_timeout_ms(
        cls,
        value: Dict[str, float],
    ) -> Dict[str, float]:
        """Validate per-topic timeout overrides for block-with-timeout policy."""
        if not isinstance(value, dict):
            raise ValueError(
                "event_queue_overflow_topic_block_timeout_ms must be a dict[str, float]"
            )

        normalized: Dict[str, float] = {}
        for topic_pattern, timeout_ms in value.items():
            pattern = str(topic_pattern).strip()
            if not pattern:
                raise ValueError(
                    "event_queue_overflow_topic_block_timeout_ms keys must be non-empty"
                )
            try:
                parsed_timeout = float(timeout_ms)
            except (TypeError, ValueError):
                raise ValueError(
                    "event_queue_overflow_topic_block_timeout_ms values must be numeric"
                ) from None
            if parsed_timeout < 0:
                raise ValueError("event_queue_overflow_topic_block_timeout_ms values must be >= 0")
            normalized[pattern] = parsed_timeout
        return normalized

    @model_validator(mode="after")
    def _sync_nested_groups(self) -> "Settings":
        """Sync flat field values into nested config groups.

        Enables structured access via settings.provider.default_provider
        alongside flat access via settings.default_provider.
        Flat fields remain the source of truth for construction and env vars.

        For backward compatibility, if a nested field is None or was passed
        as a string (legacy behavior), it's populated from the corresponding
        flat fields. A deprecation warning is emitted once to guide migration
        to the nested access pattern (e.g. settings.provider.xxx).
        """
        nested_group_names = set(_NESTED_GROUPS.keys())
        for group_name, model_cls in _NESTED_GROUPS.items():
            # Get current value of the nested field
            nested_obj = getattr(self, group_name)

            # If nested field is None or needs to be synced, create it from flat fields
            if nested_obj is None or not isinstance(nested_obj, model_cls):
                data = {}
                settings_fields = type(self).model_fields
                for field_name in model_cls.model_fields:
                    # Only copy flat Settings fields — skip other nested group names to
                    # prevent type-mismatch when a nested model and a nested group share a
                    # field name (e.g. a nested model's ``feature_flags`` field vs the
                    # ``feature_flags`` group whose type is FeatureFlagSettings).
                    if field_name in settings_fields and field_name not in nested_group_names:
                        data[field_name] = getattr(self, field_name)
                object.__setattr__(self, group_name, model_cls(**data))

        # Flat→nested sync is silent. All source code has been migrated to
        # nested access (settings.provider.X). Flat fields remain as env-var
        # entry points only (VICTOR_DEFAULT_PROVIDER → default_provider → provider.default_provider).
        return self

    @model_validator(mode="after")
    def validate_extension_loader_thresholds(self) -> "Settings":
        """Validate extension-loader pressure threshold relationships."""
        if self.event_queue_maxsize < 1:
            raise ValueError("event_queue_maxsize must be >= 1")
        if self.event_queue_overflow_block_timeout_ms < 0:
            raise ValueError("event_queue_overflow_block_timeout_ms must be >= 0")
        if self.extension_loader_pressure_cooldown_seconds < 0:
            raise ValueError("extension_loader_pressure_cooldown_seconds must be >= 0")
        if self.extension_loader_metrics_reporter_interval_seconds <= 0:
            raise ValueError("extension_loader_metrics_reporter_interval_seconds must be > 0")
        if self.extension_loader_error_queue_threshold < self.extension_loader_warn_queue_threshold:
            raise ValueError(
                "extension_loader_error_queue_threshold must be >= "
                "extension_loader_warn_queue_threshold"
            )
        if (
            self.extension_loader_error_in_flight_threshold
            < self.extension_loader_warn_in_flight_threshold
        ):
            raise ValueError(
                "extension_loader_error_in_flight_threshold must be >= "
                "extension_loader_warn_in_flight_threshold"
            )
        # Cache-related validations (now in nested cache group)
        if (
            self.cache
            and self.cache.generic_result_cache_ttl is not None
            and self.cache.generic_result_cache_ttl < 0
        ):
            raise ValueError("generic_result_cache_ttl must be >= 0")
        if (
            self.cache
            and self.cache.tool_selection_cache_ttl is not None
            and self.cache.tool_selection_cache_ttl < 0
        ):
            raise ValueError("tool_selection_cache_ttl must be >= 0")
        if (
            self.cache
            and self.cache.http_connection_pool_max_connections is not None
            and self.cache.http_connection_pool_max_connections < 1
        ):
            raise ValueError("http_connection_pool_max_connections must be >= 1")
        if (
            self.cache
            and self.cache.http_connection_pool_max_connections_per_host is not None
            and self.cache.http_connection_pool_max_connections_per_host < 1
        ):
            raise ValueError("http_connection_pool_max_connections_per_host must be >= 1")
        if (
            self.cache
            and self.cache.http_connection_pool_connection_timeout is not None
            and self.cache.http_connection_pool_connection_timeout <= 0
        ):
            raise ValueError("http_connection_pool_connection_timeout must be > 0")
        if (
            self.cache
            and self.cache.http_connection_pool_total_timeout is not None
            and self.cache.http_connection_pool_total_timeout <= 0
        ):
            raise ValueError("http_connection_pool_total_timeout must be > 0")
        return self

    @model_validator(mode="after")
    def disable_emojis_in_ci(self) -> "Settings":
        """Automatically disable emoji indicators in CI environments.

        This prevents test failures due to emoji rendering differences in CI.
        Checks the CI environment variable which is set by GitHub Actions, GitLab CI, etc.
        """
        if os.getenv("CI", "false").lower() == "true":
            # Use object.__setattr__ to bypass pydantic validation
            object.__setattr__(self, "use_emojis", False)
        return self

    # ==========================================================================
    # Validators ported from VictorSettings
    # ==========================================================================

    # NOTE: _autogenerate_session_secret validator moved to ServerSettings

    @field_validator("write_approval_mode")
    @classmethod
    def validate_write_approval_mode(cls, v: str) -> str:
        """Validate write approval mode."""
        valid_modes = ["off", "risky_only", "all_writes"]
        if v not in valid_modes:
            raise ValueError(f"Invalid write_approval_mode: {v}. Must be one of {valid_modes}")
        return v

    @field_validator("tool_validation_mode")
    @classmethod
    def validate_tool_validation_mode(cls, v: str) -> str:
        """Validate tool validation mode."""
        valid_modes = ["strict", "lenient", "off"]
        if v not in valid_modes:
            raise ValueError(f"Invalid tool_validation_mode: {v}. Must be one of {valid_modes}")
        return v

    @field_validator("context_compaction_strategy")
    @classmethod
    def validate_context_compaction_strategy(cls, v: str) -> str:
        """Validate context compaction strategy."""
        valid_strategies = ["simple", "tiered", "semantic", "hybrid"]
        if v not in valid_strategies:
            raise ValueError(
                f"Invalid context_compaction_strategy: {v}. " f"Must be one of {valid_strategies}"
            )
        return v

    @model_validator(mode="after")
    def validate_hybrid_search_weights(self) -> "Settings":
        """Validate that hybrid search weights sum to 1.0."""
        if self.search and self.search.enable_hybrid_search:
            total_weight = (
                self.search.hybrid_search_semantic_weight + self.search.hybrid_search_keyword_weight
            )
            if abs(total_weight - 1.0) > 0.01:
                raise ValueError(f"Hybrid search weights must sum to 1.0, got {total_weight}")
        return self

    @model_validator(mode="after")
    def validate_embedding_config(self) -> "Settings":
        """Warn on unknown embedding providers."""
        known_providers = {
            "sentence-transformers",
            "openai",
            "ollama",
            "huggingface",
            "cohere",
        }
        # Check nested groups
        if self.embedding and self.embedding.embedding_provider not in known_providers:
            val = self.embedding.embedding_provider
            if val:
                warnings.warn(
                    f"Unknown embedding_provider='{val}'. Known providers: {sorted(known_providers)}",
                    UserWarning,
                    stacklevel=2,
                )
        if self.search and self.search.codebase_embedding_provider not in known_providers:
            val = self.search.codebase_embedding_provider
            if val:
                warnings.warn(
                    f"Unknown codebase_embedding_provider='{val}'. Known providers: {sorted(known_providers)}",
                    UserWarning,
                    stacklevel=2,
                )
        return self

    # ==========================================================================
    # Loading with Precedence (from VictorSettings merge)
    # ==========================================================================

    @classmethod
    def from_sources(
        cls,
        cli_args: Optional[Dict[str, Any]] = None,
        profile_name: Optional[str] = None,
        config_dir: Optional[Path] = None,
    ) -> "Settings":
        """Load settings with proper precedence.

        Precedence (highest to lowest):
        1. CLI arguments (passed via cli_args)
        2. Environment variables (VICTOR_*)
        3. .env file
        4. ~/.victor/settings.yaml
        5. ~/.victor/profiles.yaml (specific profile)
        6. Default values

        Args:
            cli_args: CLI argument overrides (highest priority)
            profile_name: Active profile name to load from profiles.yaml
            config_dir: Custom config directory (defaults to ~/.victor)

        Returns:
            Settings instance with all sources merged

        Example:
            >>> settings = Settings.from_sources(
            ...     cli_args={"provider": "ollama", "model": "qwen3-coder:30b"},
            ...     profile_name="default"
            ... )
            >>> settings.default_provider
            'ollama'
        """
        # Determine config directory
        if config_dir is None:
            config_dir = Path.home() / ".victor"

        # Start with defaults + env vars + .env file (Pydantic handles these)
        settings_dict: Dict[str, Any] = {}

        # Layer 5: Load profiles.yaml if exists
        profiles_path = config_dir / "profiles.yaml"
        if profiles_path.exists():
            try:
                with open(profiles_path) as f:
                    profiles_data = yaml.safe_load(f) or {}

                # Extract profile if specified
                if profile_name and profile_name in profiles_data.get("profiles", {}):
                    profile_config = profiles_data["profiles"][profile_name]
                    # Accept flat keys — LEGACY_ENV_MAPPINGS routes them to nested groups
                    for key, value in profile_config.items():
                        settings_dict[key] = value
            except Exception as e:
                logger.warning("Failed to load profiles.yaml: %s", e)

        # Layer 4: Load settings.yaml if exists
        settings_path = config_dir / "settings.yaml"
        if settings_path.exists():
            try:
                with open(settings_path) as f:
                    user_settings = yaml.safe_load(f) or {}
                    # Override with user settings — accept flat keys for LEGACY_ENV_MAPPINGS routing
                    for key, value in user_settings.items():
                        settings_dict[key] = value
            except Exception as e:
                logger.warning("Failed to load settings.yaml: %s", e)

        # Layer 3: .env file + Layer 2: Environment variables
        # Pydantic Settings handles these automatically in __init__

        # Create base settings with layers 3-6 (defaults, .env, env vars)
        settings = cls(**settings_dict)

        # Layer 1: Apply CLI overrides (highest priority)
        # Re-create from merged dict so LEGACY_ENV_MAPPINGS routes flat keys to nested groups
        if cli_args:
            cli_overrides = {k: v for k, v in cli_args.items() if v is not None}
            if cli_overrides:
                settings = cls(**{**settings_dict, **cli_overrides})

        return settings

    @staticmethod
    def _estimate_model_vram_gb(model_id: str) -> Optional[float]:
        """Rough VRAM requirement (GB) by model name heuristic."""
        requirements = [
            ("70b", 64.0),
            ("65b", 60.0),
            ("33b", 40.0),
            ("32b", 36.0),
            ("30b", 34.0),
            ("34b", 34.0),
            ("14b", 18.0),
            ("13b", 16.0),
            ("12b", 14.0),
            ("8x7b", 40.0),  # Mixture
            ("8b", 10.0),
            ("7b", 8.0),
            ("6.7b", 8.0),
            ("3b", 6.0),
            ("1.5b", 4.0),
        ]
        m = model_id.lower()
        for key, vram in requirements:
            if key in m:
                return vram
        return None

    @staticmethod
    def _detect_vram_gb() -> Optional[float]:
        """Detect available GPU VRAM in GB (best-effort)."""
        try:
            import subprocess

            # Try NVIDIA GPUs
            cmd = [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ]
            output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True)
            values = [float(x.strip()) for x in output.splitlines() if x.strip()]
            if values:
                return max(values) / 1024.0
        except Exception:
            pass

        # macOS: try system_profiler for VRAM
        try:
            import subprocess

            output = subprocess.check_output(
                ["system_profiler", "SPDisplaysDataType"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            for line in output.splitlines():
                if "VRAM" in line and "Total" in line:
                    # e.g., "      Total VRAM (Dynamic, Max): 4096 MB"
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if (
                            part.replace(",", "").isdigit()
                            and i + 1 < len(parts)
                            and parts[i + 1].upper().startswith("MB")
                        ):
                            return float(part.replace(",", "")) / 1024.0
        except Exception:
            pass

        return None

    @classmethod
    def _choose_default_lmstudio_model(
        cls, urls: list[str], max_vram_gb: Optional[float] = None
    ) -> str:
        """Pick a sane default model from reachable LMStudio servers.

        Preference order favors small coder models for latency, then falls
        back to the first advertised model from the first reachable server.
        """
        preferred_models = [
            "qwen2.5-coder:7b",
            "qwen2.5-coder:14b",
            "qwen2.5-coder:32b",
            "qwen2.5:7b",
            "llama-3.1-8b-instruct",
        ]

        detected_vram = cls._detect_vram_gb()
        max_vram = (
            max_vram_gb if max_vram_gb is not None else getattr(cls, "lmstudio_max_vram_gb", None)
        )
        available_vram = None
        if detected_vram and max_vram:
            available_vram = min(detected_vram, max_vram)
        else:
            available_vram = detected_vram or max_vram

        try:
            import httpx  # Local network call only; safe in airgapped mode
        except Exception:
            return preferred_models[0]

        for url in urls:
            try:
                resp = httpx.get(f"{url}/v1/models", timeout=1.0)
                if resp.status_code != 200:
                    continue
                data = resp.json() or {}
                models = [
                    m.get("id") or m.get("model")
                    for m in data.get("data", [])
                    if isinstance(m, dict)
                ]
                if not models:
                    continue

                # If VRAM is known, choose the most capable coder/instruct model that fits
                if available_vram:
                    candidates = []
                    for m_id in models:
                        if not m_id:
                            continue
                        requirement = cls._estimate_model_vram_gb(m_id or "")
                        if requirement and requirement <= available_vram:
                            # Prefer coder/ instruct models
                            is_coder = "coder" in m_id.lower()
                            candidates.append((requirement, is_coder, m_id))
                    if candidates:
                        # Choose largest VRAM within budget; coder preferred
                        candidates.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
                        return str(candidates[0][2])

                for pref in preferred_models:
                    if pref in models:
                        return pref
                return str(models[0])
            except Exception:
                continue

        return str(preferred_models[0])

    @classmethod
    def get_config_dir(cls) -> Path:
        """Get configuration directory path.

        Returns:
            Path to global config directory (~/.victor)
        """
        config_dir = GLOBAL_VICTOR_DIR
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir

    @classmethod
    def _bundled_default_profiles_text(cls) -> Optional[str]:
        """Return the packaged ``profiles.default.yaml`` contents, if present."""
        try:
            from importlib import resources

            ref = resources.files("victor.config").joinpath("profiles.default.yaml")
            return ref.read_text(encoding="utf-8")
        except Exception:
            return None

    @staticmethod
    def _parse_profiles_yaml(text: str) -> Dict[str, Dict[str, Any]]:
        """Parse a profiles YAML document into raw per-profile dicts."""
        data = yaml.safe_load(text) or {}
        raw = data.get("profiles", {}) or {}
        return {str(name): dict(config or {}) for name, config in raw.items()}

    @classmethod
    def load_profiles(cls) -> Dict[str, ProfileConfig]:
        """Load profiles, layering the user file over bundled defaults.

        The package ships ``victor/config/profiles.default.yaml``. On first run
        (no ``~/.victor/profiles.yaml``) it is seeded there so the CLI works
        out of the box and stays user-editable. At load time bundled defaults
        act as the base layer: a user profile with the same name overrides it
        key-by-key, and bundled profiles added by package upgrades appear
        automatically without touching the user's file.

        Returns:
            Dictionary of profile configurations
        """
        profiles_file = cls.get_config_dir() / "profiles.yaml"
        bundled_text = cls._bundled_default_profiles_text()

        if not profiles_file.exists():
            if bundled_text is not None:
                try:
                    profiles_file.write_text(bundled_text, encoding="utf-8")
                    logger.info("Seeded default profiles at %s", profiles_file)
                except OSError as exc:
                    logger.warning("Could not seed %s: %s", profiles_file, exc)
            else:
                # No bundled resource (unusual install) — legacy LM Studio probe.
                urls = getattr(cls, "lmstudio_base_urls", []) or [
                    "http://localhost:1234",
                ]
                default_model = cls._choose_default_lmstudio_model(
                    urls, max_vram_gb=cls().lmstudio_max_vram_gb
                )
                return {
                    "default": ProfileConfig(
                        provider="lmstudio",
                        model=default_model,
                        temperature=0.6,
                        max_tokens=4096,
                        description=None,
                        tool_selection=None,
                    )
                }

        bundled_raw: Dict[str, Dict[str, Any]] = {}
        if bundled_text is not None:
            try:
                bundled_raw = cls._parse_profiles_yaml(bundled_text)
            except Exception as exc:
                logger.warning("Failed to parse bundled default profiles: %s", exc)

        try:
            user_raw = (
                cls._parse_profiles_yaml(profiles_file.read_text(encoding="utf-8"))
                if profiles_file.exists()
                else {}
            )
        except Exception as e:
            print(f"Warning: Failed to load profiles: {e}")
            return {}

        profiles: Dict[str, ProfileConfig] = {}
        for name in {**bundled_raw, **user_raw}:
            merged = {**bundled_raw.get(name, {}), **user_raw.get(name, {})}
            try:
                profiles[name] = ProfileConfig(**merged)
            except Exception as exc:
                print(f"Warning: Skipping invalid profile '{name}': {exc}")
        return profiles

    @classmethod
    def load_provider_config(cls, provider: str) -> Optional[ProviderConfig]:
        """Load provider-specific configuration.

        Args:
            provider: Provider name

        Returns:
            ProviderConfig or None
        """
        profiles_file = cls.get_config_dir() / "profiles.yaml"

        if not profiles_file.exists():
            return None

        try:
            with open(profiles_file, "r") as f:
                data = yaml.safe_load(f)

            provider_data = data.get("providers", {}).get(provider, {})
            if provider_data:
                # Expand environment variables
                for key, value in provider_data.items():
                    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                        env_var = value[2:-1]
                        provider_data[key] = os.getenv(env_var)

                return ProviderConfig(**provider_data)

        except Exception as e:
            print(f"Warning: Failed to load provider config for {provider}: {e}")

        return None

    @classmethod
    def load_tool_config(cls) -> Dict[str, Any]:
        """Load tool-specific configuration from profiles.yaml (top-level 'tools' key)."""
        profiles_file = cls.get_config_dir() / "profiles.yaml"
        if not profiles_file.exists():
            return {}
        try:
            with open(profiles_file, "r") as f:
                data = yaml.safe_load(f) or {}
            return data.get("tools", {}) or {}
        except Exception as e:
            print(f"Warning: Failed to load tool config: {e}")
            return {}

    def get_provider_settings(
        self,
        provider: str,
        profile_overrides: Optional[Dict[str, Any]] = None,
        account_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get settings for a specific provider.

        Uses ProviderConfigRegistry as the single resolution authority.
        AccountManager (config.yaml) provides credential data only — it never
        bypasses provider-specific strategy logic (endpoint switching, URL probing).

        Merge precedence (lowest to highest):
        1. profiles.yaml providers section
        2. AccountManager credentials (api_key, auth_mode from config.yaml)
        3. Profile overrides (--coding-plan, --auth-mode from CLI)
        4. Provider strategy logic (endpoint switching, URL probing)

        Args:
            provider: Provider name (or alias like 'gemini' for 'google')
            profile_overrides: Runtime overrides (e.g., coding_plan, auth_mode)
            account_name: Credential-identity account to resolve (from the profile's
                ``account`` field). Decouples credential identity from the provider
                transport slug: when set, the named account's own key/endpoint are
                used instead of the provider default, so multiple accounts under one
                provider (e.g. several anthropic-dialect upstreams) never collide on
                a shared credential slot. When None, falls back to the provider
                default account (legacy single-account-per-provider behavior).

        Returns:
            Dictionary of provider settings
        """
        import logging

        logger = logging.getLogger(__name__)

        # Extract credential data from AccountManager (if config.yaml exists)
        account_data = None
        try:
            from victor.config.accounts import get_account_manager

            manager = get_account_manager()
            if manager.config_path.exists():
                # Account-scoped selection: the profile's own account is the unit of
                # credential identity. Fall back to the provider default only when the
                # profile names no account (or names an unknown one), so misconfigured
                # profiles degrade gracefully rather than silently dropping credentials.
                if account_name:
                    account = manager.get_account(name=account_name)
                    if account is None:
                        logger.warning(
                            f"Profile account '{account_name}' not found; "
                            f"falling back to provider '{provider}' default account"
                        )
                        account = manager.get_account(provider=provider)
                else:
                    account = manager.get_account(provider=provider)
                if account:
                    account_data = {}
                    # Extract auth info
                    if account.auth.method == "oauth":
                        account_data["auth_mode"] = "oauth"
                    elif account.auth.method == "api_key":
                        # Try resolving API key from account's configured source
                        resolved = manager.resolve_provider_config(account)
                        if "api_key" in resolved:
                            account_data["api_key"] = resolved["api_key"]
                    # Include endpoint if account specifies one
                    if account.endpoint:
                        account_data["base_url"] = account.endpoint
                    logger.debug(
                        f"AccountManager credentials for '{provider}': "
                        f"keys={list(account_data.keys())}"
                    )
        except Exception as e:
            logger.debug(f"AccountManager lookup skipped: {e}")

        # Always use ProviderConfigRegistry — it handles all provider-specific logic
        from victor.config.provider_config_registry import get_provider_config_registry

        registry = get_provider_config_registry()
        return registry.get_settings(provider, self, profile_overrides, account_data)


_settings_snapshot: Optional["Settings"] = None
_settings_snapshot_lock = threading.Lock()


def load_settings(fresh: bool = False) -> Settings:
    """Load application settings (process-cached snapshot by default).

    ``Settings()`` construction parses .env, scans os.environ twice through
    legacy mapping validators, and validates 155 pydantic fields (~5 ms) —
    previously paid per LLM request (sandhi transport) and per graph
    invocation (co-design review U3-F1 / U2-F3).

    The cached instance is shared across the process: treat it as
    READ-ONLY. Callers that will MUTATE the returned settings (Agent.create,
    VictorClient, config overlays) must pass ``fresh=True`` — that returns a
    new uncached instance without disturbing the snapshot. Tests that change
    os.environ and expect new reads should call ``reset_settings_cache()``
    first.

    Args:
        fresh: Build and return a new instance, bypassing (and not
            updating) the cache — for callers that mutate the result.

    Returns:
        Settings instance (shared snapshot unless ``fresh=True``).
    """
    if fresh:
        return Settings()
    global _settings_snapshot
    if _settings_snapshot is None:
        # Lock-guarded cold start: an unlocked global let N concurrent
        # threads each build their own instance (measured 8/8 distinct),
        # silently breaking the shared-snapshot contract (adversarial-
        # review finding). Settings construction cannot await, so a plain
        # threading lock is safe inside async code.
        with _settings_snapshot_lock:
            if _settings_snapshot is None:
                _settings_snapshot = Settings()
    return _settings_snapshot


def reset_settings_cache() -> None:
    """Drop the cached settings snapshot so the next ``load_settings()``
    rebuilds from current environment/config sources."""
    global _settings_snapshot
    _settings_snapshot = None


# Alias for compatibility with packages/victor-core.
# NOTE: kept as a module attribute (not a def) so tests can monkeypatch the
# symbol — test_sandhi_transport patches victor.config.settings.get_settings.
get_settings = load_settings


def is_first_time_user() -> bool:
    """Detect if this is a first-time user.

    A user is considered a first-time user if:
    1. No onboarding completion marker exists
    2. No unified accounts config (~/.victor/config.yaml) exists
    3. No profiles.yaml exists in ~/.victor/
    4. No API keys are configured in environment
    5. Ollama is not available (local provider check)

    The onboarding completion marker (.onboarding_completed) is authoritative:
    if it exists, the user has completed setup regardless of current provider status.

    Returns:
        True if this appears to be a first-time user
    """
    from pathlib import Path

    # Check if onboarding was already completed
    victor_dir = Path.home() / ".victor"
    onboarding_marker = victor_dir / ".onboarding_completed"

    if onboarding_marker.exists():
        # User has completed onboarding, not first-time
        return False

    # Unified accounts config (written by `victor auth setup` / `victor auth
    # add`) counts as configured even without env keys or profiles.yaml —
    # keys may live only in the system keyring.
    if (victor_dir / "config.yaml").exists():
        return False

    # Check if profiles.yaml exists
    profiles_path = victor_dir / "profiles.yaml"

    if not profiles_path.exists():
        return True

    # Check if any API keys are configured
    # We check environment variables for common cloud providers
    has_keys = bool(
        os.getenv("ANTHROPIC_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("AZURE_API_KEY")
        or os.getenv("COHERE_API_KEY")
        or os.getenv("XAI_API_KEY")
    )

    # If API keys are configured, not first time user (skip Ollama check)
    if has_keys:
        return False

    # Also check if Ollama is available (local provider)
    # Use cached result to avoid 2-second subprocess timeout on every call
    ollama_cache_file = victor_dir / ".ollama_available"

    if ollama_cache_file.exists():
        # Use cached result
        has_ollama = True
    else:
        # Check Ollama availability once and cache result
        try:
            import subprocess

            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                timeout=2,
            )
            has_ollama = result.returncode == 0

            # Cache the result
            if has_ollama:
                try:
                    ollama_cache_file.touch()
                except Exception:
                    pass  # Fail silently if cache can't be written
        except Exception:
            has_ollama = False

    # If no keys and no Ollama, consider it first-time
    return not has_ollama


def validate_default_model(settings: "Settings") -> tuple[bool, str | None]:
    """Check if default model exists in Ollama.

    Args:
        settings: Settings object with provider and model configuration

    Returns:
        Tuple of (is_valid, warning_message)
        - is_valid: True if model exists or provider is not Ollama
        - warning_message: User-friendly message if model not found, None otherwise
    """
    # Only validate for Ollama provider
    if settings.provider.default_provider.lower() != "ollama":
        return True, None

    model = settings.provider.default_model

    try:
        import subprocess

        # Check if Ollama is running and get available models
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode != 0:
            # Ollama not running
            return False, (
                "Ollama is not running. Start it with:\n"
                "  [cyan]ollama serve[/]\n\n"
                f"Then pull the default model:\n"
                f"  [cyan]ollama pull {model}[/]"
            )

        # Parse model list (skip header line)
        models = result.stdout.strip().split("\n")[1:]
        available_models = [line.split()[0] for line in models if line.strip()]

        # Check if default model is available
        if model not in available_models:
            # Model not installed
            return False, (
                f"Default model '[yellow]{model}[/]' is not installed.\n\n"
                f"Available models on your system:\n"
                f"  [dim]{', '.join(available_models[:5])}{'...' if len(available_models) > 5 else ''}[/]\n\n"
                f"Pull the default model:\n"
                f"  [cyan]ollama pull {model}[/]\n\n"
                f"Or use a available model with:\n"
                f"  [cyan]victor chat --model {available_models[0] if available_models else 'qwen2.5-coder:7b'}[/]"
            )

        return True, None

    except FileNotFoundError:
        return False, (
            "Ollama binary not found. Install from https://ollama.com\n\n"
            "After installation, start Ollama and pull the default model:\n"
            f"  [cyan]ollama serve[/]\n"
            f"  [cyan]ollama pull {model}[/]"
        )
    except subprocess.TimeoutExpired:
        return False, (
            "Ollama command timed out. Check if Ollama is running:\n" "  [cyan]ollama serve[/]"
        )
    except Exception:
        # Unexpected error - don't block startup
        return True, None


_warned_graph_writer_compatibility = False


def _get_settings_group_value(
    settings: Any, group_name: str, field_name: str, default: Any = None
) -> Any:
    """Resolve a setting from its nested group, with flat fallback for test doubles."""
    group = getattr(settings, group_name, None)
    if group is not None and hasattr(group, field_name):
        return getattr(group, field_name)
    return getattr(settings, field_name, default)


def _get_search_setting(settings: Any, field_name: str, default: Any = None) -> Any:
    """Resolve a code-search setting from the canonical search group."""
    return _get_settings_group_value(settings, "search", field_name, default)


def _resolve_graph_writer_mode(settings: Any) -> str:
    """Resolve graph writer mode with explicit compatibility-mode warning."""
    global _warned_graph_writer_compatibility

    graph_writer_mode = _get_search_setting(settings, "codebase_graph_writer_mode", "")
    if not graph_writer_mode:
        graph_writer_mode = os.environ.get("VICTOR_CODEBASE_GRAPH_WRITER_MODE", "off")

    mode = str(graph_writer_mode).strip().lower()
    if mode in {"", "off", "false", "0"}:
        return "off"
    if mode == "compatibility":
        if not _warned_graph_writer_compatibility:
            warnings.warn(
                "codebase_graph_writer_mode='compatibility' is deprecated compatibility "
                "support. Victor-coding graph writes should be treated as explicit, "
                "temporary compatibility behavior and are not the primary graph source.",
                UserWarning,
                stacklevel=2,
            )
            _warned_graph_writer_compatibility = True
        return "compatibility"
    if mode in {"on", "true", "1"}:
        warnings.warn(
            "codebase_graph_writer_mode only supports 'off' (default) and "
            "'compatibility' (deprecated). Treating value as 'off'.",
            UserWarning,
            stacklevel=2,
        )
    return "off"

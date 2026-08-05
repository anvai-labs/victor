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

"""Configuration profiles for different user levels.

NOTE: this preset/scaffolding system is NOT what resolves ``victor chat -p``.
The CLI profile flag is resolved by ``Settings.load_profiles()``
(victor/config/settings.py), which layers ``~/.victor/profiles.yaml`` over the
bundled ``victor/config/profiles.default.yaml``. The templates here generate
global-config presets (tool budgets, thresholds) for onboarding.

This module provides predefined configuration profiles that simplify onboarding
by offering sensible defaults for different experience levels:

**Basic Profile** (beginners):
- Local provider (Ollama) for privacy and cost savings
- Conservative tool budgets
- Simplified defaults

**Advanced Profile** (experienced users):
- Choice of local or cloud providers
- Customizable tool settings
- Performance optimizations enabled

**Expert Profile** (power users):
- Full control over all settings
- Custom tool budgets, caching, etc.
- All performance features enabled
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class ProfileLevel(Enum):
    """Profile complexity levels."""

    BASIC = "basic"  # Beginners - simple defaults
    ADVANCED = "advanced"  # Experienced users - customizable
    EXPERT = "expert"  # Power users - full control


@dataclass
class ProfileTemplate:
    """Template for a configuration profile.

    Attributes:
        name: Profile identifier (e.g., "basic", "advanced", "expert")
        display_name: Human-readable name
        description: What this profile is for
        level: Complexity level
        settings: Default settings for this profile
        provider_settings: Provider-specific defaults
    """

    name: str
    display_name: str
    description: str
    level: ProfileLevel
    settings: Dict[str, Any]
    provider_settings: Dict[str, Dict[str, Any]]


# =============================================================================
# Predefined Profiles
# =============================================================================

BASIC_PROFILE = ProfileTemplate(
    name="basic",
    display_name="Basic",
    description=(
        "Simple defaults for beginners. "
        "Uses local models (Ollama) for privacy and cost savings. "
        "Conservative tool budgets for stability."
    ),
    level=ProfileLevel.BASIC,
    settings={
        # Use Ollama by default (local, free)
        "default_provider": "ollama",
        "default_model": "qwen3.5:27b-q4_K_M",  # Fast MoE model with good knowledge
        "default_temperature": 0.6,
        "default_max_tokens": 4096,
        # Conservative tool budget
        "fallback_max_tools": 5,
        # Enable safety features
        "framework_preload_enabled": True,
        "http_connection_pool_enabled": True,
        "tool_selection_cache_enabled": True,
        # Tool selection preset (small model)
        "tool_selection": {
            "model_size_tier": "small",
            "base_threshold": 0.75,
            "base_max_tools": 5,
        },
    },
    provider_settings={
        "ollama": {
            "base_url": "http://localhost:11434",
        },
    },
)

ADVANCED_PROFILE = ProfileTemplate(
    name="advanced",
    display_name="Advanced",
    description=(
        "For experienced users who want customization. "
        "Supports both local and cloud providers with enhanced features. "
        "Higher tool budgets for complex tasks."
    ),
    level=ProfileLevel.ADVANCED,
    settings={
        # Detect available provider or default to Ollama
        "default_provider": "auto",
        "default_model": "auto",
        "default_temperature": 0.6,
        "default_max_tokens": 8192,
        # Higher tool budget for complex tasks
        "fallback_max_tools": 10,
        # All performance optimizations enabled
        "framework_preload_enabled": True,
        "http_connection_pool_enabled": True,
        "tool_selection_cache_enabled": True,
        # Advanced tool selection
        "tool_selection": {
            "model_size_tier": "medium",
            "base_threshold": 0.70,
            "base_max_tools": 10,
        },
        # Loop detection
        "loop_repeat_threshold": 3,
        "max_continuation_prompts": 2,
    },
    provider_settings={
        "ollama": {
            "base_url": "http://localhost:11434",
        },
        "anthropic": {
            # Requires ANTHROPIC_API_KEY env var
            "description": "Claude (Sonnet 4.5) - Best for complex reasoning",
        },
        "openai": {
            # Requires OPENAI_API_KEY env var
            "description": "GPT-4o - Excellent all-purpose model",
        },
    },
)

EXPERT_PROFILE = ProfileTemplate(
    name="expert",
    display_name="Expert",
    description=(
        "Full control for power users. "
        "Configure every aspect: tool budgets, caching, providers, timeouts. "
        "Maximum performance optimizations enabled."
    ),
    level=ProfileLevel.EXPERT,
    settings={
        # Expert chooses their provider
        "default_provider": "auto",
        "default_model": "auto",
        "default_temperature": 0.6,
        "default_max_tokens": 16384,
        # Maximum flexibility
        "fallback_max_tools": 20,
        # All optimizations
        "framework_preload_enabled": True,
        "http_connection_pool_enabled": True,
        "tool_selection_cache_enabled": True,
        "tool_deduplication_enabled": True,
        # Expert tool selection
        "tool_selection": {
            "model_size_tier": "large",
            "base_threshold": 0.65,
            "base_max_tools": 20,
            "adaptive": True,
        },
        # Full control over thresholds
        "loop_repeat_threshold": 4,
        "max_continuation_prompts": 3,
        "quality_threshold": 0.7,
        "grounding_threshold": 0.8,
        "max_tool_calls_per_turn": 15,
        # Timeouts
        "timeout": 120,
        "session_idle_timeout": 3600,
    },
    provider_settings={
        "ollama": {
            "base_url": "http://localhost:11434",
        },
        "lmstudio": {
            "base_url": "http://localhost:1234/v1",
        },
        "anthropic": {
            "description": "Claude Opus 4.6 or Sonnet 4.5",
        },
        "openai": {
            "description": "GPT-4o or o1",
        },
        "google": {
            "description": "Gemini 2.0 Flash/Pro",
        },
    },
)

# Specialized profiles for specific use cases
CODING_PROFILE = ProfileTemplate(
    name="coding",
    display_name="Coding Specialist",
    description=(
        "Optimized for software development tasks. "
        "Uses coding-specialized models with enhanced tool access."
    ),
    level=ProfileLevel.ADVANCED,
    settings={
        "default_provider": "ollama",
        "default_model": "qwen2.5-coder:30b",
        "default_temperature": 0.3,  # Lower temp for code
        "default_max_tokens": 8192,
        "fallback_max_tools": 15,  # More tools for coding
        "tool_selection": {
            "model_size_tier": "large",
            "base_threshold": 0.65,
            "base_max_tools": 15,
        },
        "framework_preload_enabled": True,
        "http_connection_pool_enabled": True,
        "tool_selection_cache_enabled": True,
    },
    provider_settings={
        "ollama": {
            "base_url": "http://localhost:11434",
        },
    },
)

RESEARCH_PROFILE = ProfileTemplate(
    name="research",
    display_name="Research Assistant",
    description=(
        "Optimized for research and analysis tasks. "
        "Higher context window for document processing."
    ),
    level=ProfileLevel.ADVANCED,
    settings={
        "default_provider": "anthropic",  # Cloud for analysis quality
        "default_model": "claude-sonnet-4-5-20250514",
        "default_temperature": 0.8,  # Higher temp for creative thinking
        "default_max_tokens": 16384,  # Large context
        "fallback_max_tools": 10,
        "tool_selection": {
            "model_size_tier": "cloud",
            "base_threshold": 0.70,
            "base_max_tools": 10,
        },
        "framework_preload_enabled": True,
        "http_connection_pool_enabled": True,
        "tool_selection_cache_enabled": True,
    },
    provider_settings={
        "anthropic": {
            "description": "Claude Sonnet 4.5 - Best for analysis",
        },
    },
)

BENCHMARK_PROFILE = ProfileTemplate(
    name="benchmark",
    display_name="Benchmark Testbed",
    description=(
        "Cloud API testbed for running benchmarks (SWE-bench, HumanEval, MBPP). "
        "Uses OAuth authentication to avoid direct API costs. "
        "High tool budget and generous timeouts for evaluation runs."
    ),
    level=ProfileLevel.EXPERT,
    settings={
        "default_provider": "openai",
        "default_model": "gpt-5.4-mini",
        "default_temperature": 0.3,  # Low temp for deterministic benchmark runs
        "default_max_tokens": 16384,
        # High tool budget for benchmark tasks
        "fallback_max_tools": 20,
        # All optimizations
        "framework_preload_enabled": True,
        "http_connection_pool_enabled": True,
        "tool_selection_cache_enabled": True,
        "tool_deduplication_enabled": True,
        # Expert tool selection (cloud tier)
        "tool_selection": {
            "model_size_tier": "cloud",
            "base_threshold": 0.65,
            "base_max_tools": 20,
            "adaptive": True,
        },
        # Generous thresholds for benchmark evaluation
        "loop_repeat_threshold": 4,
        "max_continuation_prompts": 3,
        "max_tool_calls_per_turn": 15,
        "timeout": 300,
        "session_idle_timeout": 7200,
    },
    provider_settings={
        "openai": {
            "auth_mode": "oauth",
            "description": "GPT-5.4 Mini via OAuth (ChatGPT subscription, no API costs)",
        },
    },
)

# All available profiles
PROFILES: Dict[str, ProfileTemplate] = {
    "basic": BASIC_PROFILE,
    "advanced": ADVANCED_PROFILE,
    "expert": EXPERT_PROFILE,
    "coding": CODING_PROFILE,
    "research": RESEARCH_PROFILE,
    "benchmark": BENCHMARK_PROFILE,
}


# =============================================================================
# Profile Management Functions
# =============================================================================


def list_profiles() -> List[ProfileTemplate]:
    """List all available profiles.

    Returns:
        List of all profile templates
    """
    return list(PROFILES.values())


def get_profile(name: str) -> Optional[ProfileTemplate]:
    """Get a profile by name.

    Args:
        name: Profile name (e.g., "basic", "advanced", "expert")

    Returns:
        ProfileTemplate if found, None otherwise
    """
    return PROFILES.get(name)


def get_profiles_by_level(level: ProfileLevel) -> List[ProfileTemplate]:
    """Get all profiles for a given level.

    Args:
        level: Profile level to filter by

    Returns:
        List of profiles at the specified level
    """
    return [p for p in PROFILES.values() if p.level == level]


def get_recommended_profile() -> ProfileTemplate:
    """Get recommended profile based on environment.

    Detects:
    - If Ollama is running → recommend basic
    - If API keys are set → recommend advanced
    - If both available → recommend advanced

    Returns:
        Recommended profile template
    """
    # Check for Ollama
    has_ollama = _check_ollama_available()

    # Check for cloud API keys
    has_cloud_keys = _check_cloud_api_keys()

    if has_cloud_keys:
        return ADVANCED_PROFILE
    elif has_ollama:
        return BASIC_PROFILE
    else:
        return BASIC_PROFILE  # Default to basic, will guide setup


def generate_profile_yaml(
    profile: ProfileTemplate,
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None,
) -> str:
    """Generate profiles.yaml content for a profile template.

    Args:
        profile: Profile template to generate YAML for
        provider_override: Optional provider override
        model_override: Optional model override

    Returns:
        YAML content for profiles.yaml
    """
    import yaml

    # Detect provider/model if set to "auto"
    provider = provider_override or profile.settings.get("default_provider", "ollama")
    model = model_override or profile.settings.get("default_model", "qwen2.5-coder:7b")

    if provider == "auto":
        provider = _detect_provider()

    if model == "auto":
        model = _detect_model_for_provider(provider)

    # Build the YAML structure
    profiles_data = {
        "profiles": {
            "default": {
                "provider": provider,
                "model": model,
                "temperature": profile.settings.get("default_temperature", 0.6),
                "max_tokens": profile.settings.get("default_max_tokens", 4096),
            }
        },
        "settings": {
            "default_provider": provider,
            "fallback_max_tools": profile.settings.get("fallback_max_tools", 10),
            "framework_preload_enabled": profile.settings.get("framework_preload_enabled", True),
            "http_connection_pool_enabled": profile.settings.get(
                "http_connection_pool_enabled", True
            ),
            "tool_selection_cache_enabled": profile.settings.get(
                "tool_selection_cache_enabled", True
            ),
        },
    }

    # Add optional settings if present
    tool_selection = profile.settings.get("tool_selection")
    if tool_selection:
        profiles_data["settings"]["tool_selection"] = tool_selection

    loop_threshold = profile.settings.get("loop_repeat_threshold")
    if loop_threshold:
        profiles_data["settings"]["loop_repeat_threshold"] = loop_threshold

    max_continuation = profile.settings.get("max_continuation_prompts")
    if max_continuation:
        profiles_data["settings"]["max_continuation_prompts"] = max_continuation

    timeout = profile.settings.get("timeout")
    if timeout:
        profiles_data["settings"]["timeout"] = timeout

    session_timeout = profile.settings.get("session_idle_timeout")
    if session_timeout:
        profiles_data["settings"]["session_idle_timeout"] = session_timeout

    return yaml.dump(profiles_data, default_flow_style=False, sort_keys=False)


# =============================================================================
# Detection Helpers
# =============================================================================


def _check_ollama_available() -> bool:
    """Check if Ollama is available and running.

    Returns:
        True if Ollama is running
    """
    try:
        import subprocess

        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            timeout=2,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    except Exception:
        return False


def _check_cloud_api_keys() -> bool:
    """Check if any cloud provider API keys are configured.

    Returns:
        True if at least one cloud API key is set
    """
    return bool(
        os.getenv("ANTHROPIC_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("AZURE_API_KEY")
        or os.getenv("COHERE_API_KEY")
    )


def _detect_provider() -> str:
    """Detect the best available provider.

    Returns:
        Provider name
    """
    if _check_cloud_api_keys():
        # Prefer Anthropic if available, then OpenAI
        if os.getenv("ANTHROPIC_API_KEY"):
            return "anthropic"
        if os.getenv("OPENAI_API_KEY"):
            return "openai"

    # Default to Ollama
    return "ollama"


def _detect_model_for_provider(provider: str) -> str:
    """Detect the best model for a provider.

    Args:
        provider: Provider name

    Returns:
        Model identifier
    """
    model_defaults = {
        "anthropic": "claude-sonnet-4-5-20250514",
        "openai": "gpt-4o",
        "google": "gemini-2.0-flash-exp",
        "ollama": "qwen2.5-coder:7b",
        "lmstudio": "unknown",
    }
    return model_defaults.get(provider, "qwen2.5-coder:7b")


def _resolve_profile_config_dir(config_dir: Optional[Path] = None) -> Path:
    """Resolve the profile config directory through centralized Victor paths."""
    if config_dir is not None:
        return config_dir

    from victor.config.settings import get_project_paths

    return get_project_paths().global_victor_dir


# =============================================================================
# Profile Installation
# =============================================================================


def install_profile(
    profile: ProfileTemplate,
    config_dir: Optional[Path] = None,
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None,
) -> Path:
    """Install a profile by generating profiles.yaml.

    Args:
        profile: Profile template to install
        config_dir: Optional config directory (defaults to ~/.victor)
        provider_override: Optional provider override
        model_override: Optional model override

    Returns:
        Path to the generated profiles.yaml file
    """
    config_dir = _resolve_profile_config_dir(config_dir)

    # Ensure directory exists
    config_dir.mkdir(parents=True, exist_ok=True)

    # Generate YAML content
    yaml_content = generate_profile_yaml(
        profile,
        provider_override=provider_override,
        model_override=model_override,
    )

    # Write to profiles.yaml
    profiles_path = config_dir / "profiles.yaml"
    profiles_path.write_text(yaml_content)

    return profiles_path


def get_current_profile(config_dir: Optional[Path] = None) -> Optional[str]:
    """Detect the current profile from profiles.yaml.

    Args:
        config_dir: Optional config directory (defaults to ~/.victor)

    Returns:
        Profile name if detected, None otherwise
    """
    config_dir = _resolve_profile_config_dir(config_dir)

    profiles_path = config_dir / "profiles.yaml"
    if not profiles_path.exists():
        return None

    try:
        import yaml

        with open(profiles_path) as f:
            data = yaml.safe_load(f)

        # Try to identify profile from settings
        settings = data.get("settings", {})

        # Check for characteristic settings
        max_tools = settings.get("fallback_max_tools", 0)
        preload = settings.get("framework_preload_enabled", False)

        provider = settings.get("default_provider", "")

        if max_tools == 5 and preload:
            return "basic"
        elif max_tools == 10 and preload:
            return "advanced"
        elif max_tools == 20 and preload and provider == "openai":
            return "benchmark"
        elif max_tools == 20 and preload:
            return "expert"

        return "unknown"
    except Exception:
        return None


# =============================================================================
# Profile Manager - Single source of truth for profile operations
# =============================================================================


class ProfileManager:
    """Manager for configuration profile operations.

    Provides a centralized interface for loading and saving profiles,
    eliminating duplication between CLI commands and framework code.
    """

    def __init__(self, config_dir: Optional[Path] = None) -> None:
        """Initialize the profile manager.

        Args:
            config_dir: Optional config directory (defaults to ~/.victor)
        """
        self._config_dir = _resolve_profile_config_dir(config_dir)
        self._profiles_path = self._config_dir / "profiles.yaml"

    @property
    def profiles_path(self) -> Path:
        """Get the path to the profiles.yaml file."""
        return self._profiles_path

    def load_profiles(self) -> Dict[str, Any]:
        """Load profiles from the profiles.yaml file.

        Returns:
            Dictionary containing profiles data, or empty dict if file doesn't exist or is invalid
        """
        if not self._profiles_path.exists():
            return {"profiles": {}}

        try:
            import yaml

            with open(self._profiles_path) as f:
                return yaml.safe_load(f) or {"profiles": {}}
        except (yaml.YAMLError, IOError, OSError):
            return {"profiles": {}}

    def save_profiles(self, data: Dict[str, Any]) -> None:
        """Save profiles to the profiles.yaml file.

        Args:
            data: Dictionary containing profiles data

        Raises:
            IOError: If file cannot be written
        """
        import yaml

        self._config_dir.mkdir(parents=True, exist_ok=True)
        with open(self._profiles_path, "w") as f:
            yaml.safe_dump(data, f, default_flow_style=False)

    def upsert_account_profile(self, account: Any) -> bool:
        """Create or update the chat profile that corresponds to an account.

        Victor still resolves ``victor chat -p`` from profiles.yaml, while
        ``victor auth`` writes accounts to config.yaml. Keep the two stores in
        sync without rewriting the whole profiles file and losing comments.

        Returns:
            True when profiles.yaml was changed, False when it already matched.
        """
        import yaml

        name = getattr(account, "name", None)
        provider = getattr(account, "provider", None)
        model = getattr(account, "model", None)
        if not name or not provider or not model:
            raise ValueError("account must include name, provider, and model")

        auth = getattr(account, "auth", None)
        auth_method = getattr(auth, "method", None)
        auth_source = getattr(auth, "source", None)

        profile: Dict[str, Any] = {
            "provider": provider,
            "model": model,
        }
        temperature = getattr(account, "temperature", None)
        if temperature is not None:
            profile["temperature"] = temperature
        max_tokens = getattr(account, "max_tokens", None)
        if max_tokens is not None:
            profile["max_tokens"] = max_tokens
        if auth_method:
            profile["auth_mode"] = auth_method
        if auth_source in {"codex", "claude-code"}:
            profile["oauth_source"] = auth_source
        endpoint = getattr(account, "endpoint", None)
        if endpoint:
            profile["base_url"] = endpoint
        profile["account"] = name
        profile["description"] = f"{provider} {model} via {auth_source or auth_method or 'account'}"

        self._config_dir.mkdir(parents=True, exist_ok=True)
        if self._profiles_path.exists():
            original = self._profiles_path.read_text()
        else:
            original = "profiles:\n"

        data = yaml.safe_load(original) or {}
        profiles = data.get("profiles")
        if not isinstance(profiles, dict):
            profiles = {}
            data["profiles"] = profiles
        if profiles.get(name) == profile:
            return False
        profiles[name] = profile

        block = _render_profile_block(name, profile)
        updated = _replace_profile_block(original, name, block)
        self._profiles_path.write_text(updated)
        return True

    def get_current_profile_name(self) -> Optional[str]:
        """Detect the current profile from profiles.yaml.

        Returns:
            Profile name if detected, None otherwise
        """
        if not self._profiles_path.exists():
            return None

        try:
            import yaml

            with open(self._profiles_path) as f:
                data = yaml.safe_load(f)

            # Try to identify profile from settings
            settings = data.get("settings", {})

            # Check for characteristic settings
            max_tools = settings.get("fallback_max_tools", 0)
            preload = settings.get("framework_preload_enabled", False)

            provider = settings.get("default_provider", "")

            if max_tools == 5 and preload:
                return "basic"
            elif max_tools == 10 and preload:
                return "advanced"
            elif max_tools == 20 and preload and provider == "openai":
                return "benchmark"
            elif max_tools == 20 and preload:
                return "expert"

            return "unknown"
        except Exception:
            return None

    def has_profiles_file(self) -> bool:
        """Check if the profiles.yaml file exists."""
        return self._profiles_path.exists()

    @classmethod
    def for_config_dir(cls, config_dir: Optional[Path] = None) -> "ProfileManager":
        """Create a ProfileManager for a specific config directory.

        Args:
            config_dir: Optional config directory path

        Returns:
            ProfileManager instance
        """
        return cls(config_dir=config_dir)


def _render_profile_block(name: str, profile: Dict[str, Any]) -> str:
    """Render one profile block in the existing profiles.yaml style."""
    import yaml

    payload = yaml.safe_dump(
        {name: profile},
        default_flow_style=False,
        sort_keys=False,
    )
    return "".join(f"  {line}" if line.strip() else line for line in payload.splitlines(True))


def _replace_profile_block(original: str, name: str, block: str) -> str:
    """Replace or append a profile block under the top-level profiles section."""
    import re

    lines = original.splitlines(True)
    if not lines:
        return "profiles:\n" + block

    profiles_idx = None
    for idx, line in enumerate(lines):
        if re.match(r"^profiles:\s*(?:#.*)?$", line):
            profiles_idx = idx
            break

    if profiles_idx is None:
        suffix = "" if original.endswith("\n") else "\n"
        return original + suffix + "profiles:\n" + block

    section_end = len(lines)
    for idx in range(profiles_idx + 1, len(lines)):
        line = lines[idx]
        if re.match(r"^[A-Za-z0-9_-]+:\s*(?:#.*)?$", line):
            section_end = idx
            break

    # Keep top-level section separator comments with the following section.
    # profiles.yaml is heavily commented, so appending a profile immediately
    # before the next top-level key would otherwise split that section header.
    comment_end = section_end
    while comment_end > profiles_idx + 1 and not lines[comment_end - 1].strip():
        comment_end -= 1
    comment_start = comment_end
    while comment_start > profiles_idx + 1:
        previous = lines[comment_start - 1]
        if previous.startswith("#") or not previous.strip():
            comment_start -= 1
            continue
        break
    if comment_start < comment_end and any(
        line.startswith("#") for line in lines[comment_start:comment_end]
    ):
        section_end = comment_start

    key_pattern = re.compile(rf"^  {re.escape(name)}:\s*(?:#.*)?$")
    start = None
    for idx in range(profiles_idx + 1, section_end):
        if key_pattern.match(lines[idx]):
            start = idx
            break

    if start is not None:
        end = section_end
        for idx in range(start + 1, section_end):
            if re.match(r"^  [^\s#][^:]*:\s*(?:#.*)?$", lines[idx]):
                end = idx
                break
        return "".join(lines[:start] + block.splitlines(True) + lines[end:])

    insert_at = section_end
    if insert_at > profiles_idx + 1 and lines[insert_at - 1].strip():
        block = "\n" + block
    return "".join(lines[:insert_at] + block.splitlines(True) + lines[insert_at:])

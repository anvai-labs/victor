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
Smart Routing Configuration Models.

This module provides configuration dataclasses for smart model routing:
- RoutingProfile: Named profiles with fallback chains and preferences
- SmartRoutingConfig: Runtime configuration for routing behavior

Usage:
    from victor.providers.routing_config import SmartRoutingConfig, RoutingProfile

    config = SmartRoutingConfig(
        enabled=True,
        profile_name="balanced",
    )

    profile = RoutingProfile(
        name="cost-optimized",
        fallback_chains={"default": ["ollama", "lmstudio", "deepseek"]},
        cost_preference="low",
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class RoutingProfile:
    """Routing profile with fallback chains and preferences.

    Attributes:
        name: Profile name (used for profile selection)
        description: Human-readable description
        fallback_chains: Task type -> list of provider names
            Example: {"default": ["ollama", "anthropic"], "coding": ["ollama", "deepseek"]}
        cost_preference: Cost priority (low=cheapest, normal=balanced, high=performance)
        latency_preference: Latency priority (low=fastest, normal=balanced, high=accuracy)
    """

    name: str
    description: str
    fallback_chains: Dict[str, List[str]] = field(default_factory=dict)
    cost_preference: Literal["low", "normal", "high"] = "normal"
    latency_preference: Literal["low", "normal", "high"] = "normal"

    def __post_init__(self):
        """Validate and normalize routing profile."""
        # Ensure default fallback chain exists
        if "default" not in self.fallback_chains:
            self.fallback_chains["default"] = []

        # Normalize provider names to lowercase
        for task_type, providers in self.fallback_chains.items():
            self.fallback_chains[task_type] = [p.lower() for p in providers]

        logger.debug(
            f"Routing profile '{self.name}' initialized with {len(self.fallback_chains)} task types"
        )

    def get_fallback_chain(self, task_type: str = "default") -> List[str]:
        """Get fallback chain for specific task type.

        Args:
            task_type: Task type (default, coding, chat, etc.)

        Returns:
            List of provider names in fallback order
        """
        return self.fallback_chains.get(task_type, self.fallback_chains.get("default", []))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "fallback_chains": self.fallback_chains,
            "cost_preference": self.cost_preference,
            "latency_preference": self.latency_preference,
        }


@dataclass
class SmartRoutingConfig:
    """Configuration for smart routing behavior.

    Attributes:
        enabled: Master switch for smart routing
        profile_name: Name of routing profile to use
        custom_fallback_chain: Custom fallback chain (overrides profile)
        performance_window_size: Number of requests to track for learning
        learning_enabled: Enable adaptive learning from performance
        resource_awareness_enabled: Enable GPU/API quota detection
    """

    enabled: bool = False
    profile_name: str = "balanced"
    custom_fallback_chain: Optional[List[str]] = None
    performance_window_size: int = 100
    learning_enabled: bool = True
    resource_awareness_enabled: bool = True

    def __post_init__(self):
        """Validate smart routing configuration."""
        if self.performance_window_size < 1:
            raise ValueError("performance_window_size must be at least 1")

        if self.custom_fallback_chain:
            self.custom_fallback_chain = [p.lower() for p in self.custom_fallback_chain]

        logger.debug(
            f"SmartRoutingConfig initialized: enabled={self.enabled}, "
            f"profile={self.profile_name}"
        )


def load_routing_profiles(path: Optional[Path] = None) -> Dict[str, RoutingProfile]:
    """Load routing profiles from YAML file.

    Args:
        path: Path to routing_profiles.yaml. If None, uses default path.

    Returns:
        Dict mapping profile name to RoutingProfile

    Raises:
        FileNotFoundError: If profiles file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    if path is None:
        # Default to ~/.victor/routing_profiles.yaml
        victor_dir = Path.home() / ".victor"
        path = victor_dir / "routing_profiles.yaml"

    if not path.exists():
        logger.warning(f"Routing profiles file not found: {path}")
        return get_default_profiles()

    try:
        with open(path) as f:
            data = yaml.safe_load(f)

        profiles: Dict[str, RoutingProfile] = {}
        for profile_data in data.get("profiles", []):
            profile = RoutingProfile(
                name=profile_data["name"],
                description=profile_data.get("description", ""),
                fallback_chains=profile_data.get("fallback_chains", {}),
                cost_preference=profile_data.get("cost_preference", "normal"),
                latency_preference=profile_data.get("latency_preference", "normal"),
            )
            profiles[profile.name] = profile

        logger.info(f"Loaded {len(profiles)} routing profiles from {path}")
        return profiles

    except Exception as e:
        logger.error(f"Failed to load routing profiles from {path}: {e}")
        return get_default_profiles()


def get_default_profiles() -> Dict[str, RoutingProfile]:
    """Get default routing profiles.

    Returns:
        Dict with 4 default profiles: balanced, cost-optimized, performance, local-first
    """
    return {
        "balanced": RoutingProfile(
            name="balanced",
            description="Balance cost and performance",
            fallback_chains={
                "default": ["ollama", "anthropic", "openai"],
                "coding": ["ollama", "deepseek", "anthropic"],
                "chat": ["ollama", "groqcloud", "anthropic"],
            },
            cost_preference="normal",
            latency_preference="normal",
        ),
        "cost-optimized": RoutingProfile(
            name="cost-optimized",
            description="Minimize API costs",
            fallback_chains={
                "default": ["ollama", "lmstudio", "deepseek", "groqcloud"],
            },
            cost_preference="low",
            latency_preference="normal",
        ),
        "performance": RoutingProfile(
            name="performance",
            description="Prioritize speed and quality",
            fallback_chains={
                "default": ["anthropic", "openai", "ollama"],
            },
            cost_preference="high",
            latency_preference="low",
        ),
        "local-first": RoutingProfile(
            name="local-first",
            description="Use local providers when available",
            fallback_chains={
                "default": ["ollama", "lmstudio", "vllm"],
            },
            cost_preference="low",
            latency_preference="normal",
        ),
    }


def save_default_profiles(path: Optional[Path] = None) -> None:
    """Save default routing profiles to YAML file.

    Args:
        path: Path to save routing_profiles.yaml. If None, uses ~/.victor/routing_profiles.yaml
    """
    if path is None:
        victor_dir = Path.home() / ".victor"
        victor_dir.mkdir(parents=True, exist_ok=True)
        path = victor_dir / "routing_profiles.yaml"

    profiles = get_default_profiles()
    data = {
        "profiles": [profile.to_dict() for profile in profiles.values()],
    }

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Saved {len(profiles)} default routing profiles to {path}")

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

"""Framework-level privacy and PII management capability.

This module provides a generic privacy capability that can be reused across
all verticals (RAG, DevOps, Research, DataAnalysis, Coding, Benchmark) for:
- PII (Personally Identifiable Information) detection and anonymization
- Secrets masking (API keys, passwords, tokens)
- Data access auditing
- Privacy policy enforcement

This is a cross-cutting concern promoted from DataAnalysis vertical to the framework.

Design Pattern: Facade + Dependency Injection
- Verticals delegate privacy configuration to this framework provider
- Consistent privacy rules across all verticals
- Verticals can extend with domain-specific privacy rules

Example:
    # In any vertical assistant
    from victor.framework.capabilities.privacy import PrivacyCapabilityProvider

    @classmethod
    def get_capability_provider(cls):
        provider = VerticalCapabilityProvider()
        # Add framework privacy capability
        provider.register_capability(
            "privacy", PrivacyCapabilityProvider()
        )
        return provider
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING

from victor.framework.capabilities.base import (
    BaseCapabilityProvider,
    CapabilityMetadata,
)
from victor.framework.protocols import CapabilityType, OrchestratorCapability
from victor.framework.capability_loader import CapabilityEntry, capability
from victor.framework.capability_config_helpers import (
    load_capability_config,
    store_capability_config,
)

if TYPE_CHECKING:
    from victor.core.protocols import OrchestratorProtocol as AgentOrchestrator

logger = logging.getLogger(__name__)


# =============================================================================
# Capability Config Helpers (P1: Framework CapabilityConfigService Migration)
# =============================================================================


def _store_config(orchestrator: Any, name: str, config: Dict[str, Any]) -> None:
    """Store config in framework service when available, else fallback to orchestrator attr."""
    store_capability_config(
        orchestrator,
        name,
        config,
        require_existing_attr=False,
    )


def _load_config(orchestrator: Any, name: str, defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Load config from framework service when available, else fallback to orchestrator attr."""
    return load_capability_config(orchestrator, name, defaults)


# =============================================================================
# Privacy Configuration Handlers
# =============================================================================


def configure_data_privacy(
    orchestrator: Any,
    *,
    anonymize_pii: bool = True,
    pii_columns: Optional[List[str]] = None,
    hash_identifiers: bool = True,
    log_access: bool = True,
    detect_secrets: bool = True,
    secret_patterns: Optional[List[str]] = None,
) -> None:
    """Configure privacy and PII handling for the orchestrator.

    This framework-level capability provides generic privacy configuration
    that can be used by any vertical (RAG for documents, DevOps for secrets,
    Research for PII filtering, DataAnalysis for data privacy).

    Args:
        orchestrator: Target orchestrator
        anonymize_pii: Whether to anonymize PII columns
        pii_columns: List of column/field names containing PII
        hash_identifiers: Hash identifier columns for privacy
        log_access: Log data access for audit trail
        detect_secrets: Detect and mask secrets (API keys, tokens)
        secret_patterns: Regex patterns for secret detection
    """
    _store_config(
        orchestrator,
        "privacy_config",
        {
            "anonymize_pii": anonymize_pii,
            "pii_columns": pii_columns or [],
            "hash_identifiers": hash_identifiers,
            "log_access": log_access,
            "detect_secrets": detect_secrets,
            "secret_patterns": secret_patterns or _get_default_secret_patterns(),
        },
    )

    logger.info(
        f"Configured data privacy: anonymize={anonymize_pii}, "
        f"secrets_detection={detect_secrets}, audit_logging={log_access}"
    )


def get_privacy_config(orchestrator: Any) -> Dict[str, Any]:
    """Get current privacy configuration.

    Args:
        orchestrator: Target orchestrator

    Returns:
        Privacy configuration dict
    """
    return _load_config(
        orchestrator,
        "privacy_config",
        {
            "anonymize_pii": True,
            "pii_columns": [],
            "hash_identifiers": True,
            "log_access": True,
            "detect_secrets": True,
            "secret_patterns": _get_default_secret_patterns(),
        },
    )


def _get_default_secret_patterns() -> List[str]:
    """Get default regex patterns for secret detection.

    Returns:
        List of regex patterns for common secret types
    """
    return [
        r"sk-[a-zA-Z0-9]{32,}",  # OpenAI/Anthropic API keys
        r"Bearer [a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+",  # JWT tokens
        r"password\s*[:=]\s*\S+",  # password assignments
        r"api[_-]?key\s*[:=]\s*\S+",  # API key assignments
        r"secret\s*[:=]\s*\S+",  # secret assignments
        r"token\s*[:=]\s*\S+",  # token assignments
    ]


def configure_secrets_masking(
    orchestrator: Any,
    *,
    enabled: bool = True,
    replacement: str = "[REDACTED]",
    mask_in_arguments: bool = True,
    mask_in_output: bool = True,
    custom_patterns: Optional[List[str]] = None,
) -> None:
    """Configure secrets masking for tool calls and results.

    Args:
        orchestrator: Target orchestrator
        enabled: Whether secrets masking is enabled
        replacement: Replacement string for masked secrets
        mask_in_arguments: Mask secrets in tool arguments
        mask_in_output: Mask secrets in tool output/results
        custom_patterns: Additional regex patterns for secret detection
    """
    _store_config(
        orchestrator,
        "secrets_masking_config",
        {
            "enabled": enabled,
            "replacement": replacement,
            "mask_in_arguments": mask_in_arguments,
            "mask_in_output": mask_in_output,
            "custom_patterns": custom_patterns or [],
        },
    )

    logger.info(f"Configured secrets masking: enabled={enabled}, replacement={replacement}")


def get_secrets_masking_config(orchestrator: Any) -> Dict[str, Any]:
    """Get current secrets masking configuration.

    Args:
        orchestrator: Target orchestrator

    Returns:
        Secrets masking configuration dict
    """
    return _load_config(
        orchestrator,
        "secrets_masking_config",
        {
            "enabled": True,
            "replacement": "[REDACTED]",
            "mask_in_arguments": True,
            "mask_in_output": True,
            "custom_patterns": [],
        },
    )


def configure_audit_logging(
    orchestrator: Any,
    *,
    enabled: bool = True,
    log_data_access: bool = True,
    log_pii_access: bool = True,
    log_secrets_access: bool = True,
    log_file_path: Optional[str] = None,
) -> None:
    """Configure audit logging for privacy-sensitive operations.

    Args:
        orchestrator: Target orchestrator
        enabled: Whether audit logging is enabled
        log_data_access: Log all data access operations
        log_pii_access: Log PII/sensitive data access
        log_secrets_access: Log secrets masking operations
        log_file_path: Optional file path for audit log
    """
    _store_config(
        orchestrator,
        "audit_logging_config",
        {
            "enabled": enabled,
            "log_data_access": log_data_access,
            "log_pii_access": log_pii_access,
            "log_secrets_access": log_secrets_access,
            "log_file_path": log_file_path,
        },
    )

    logger.info(f"Configured audit logging: enabled={enabled}, data_access={log_data_access}")


def get_audit_logging_config(orchestrator: Any) -> Dict[str, Any]:
    """Get current audit logging configuration.

    Args:
        orchestrator: Target orchestrator

    Returns:
        Audit logging configuration dict
    """
    return _load_config(
        orchestrator,
        "audit_logging_config",
        {
            "enabled": True,
            "log_data_access": True,
            "log_pii_access": True,
            "log_secrets_access": True,
            "log_file_path": None,
        },
    )


# =============================================================================
# Decorated Capability Functions
# =============================================================================


@capability(
    name="framework_privacy",
    capability_type=CapabilityType.SAFETY,
    version="1.0",
    description="Framework-level privacy and PII management",
    getter="get_privacy_config",
)
def privacy_capability(
    anonymize_pii: bool = True,
    detect_secrets: bool = True,
    **kwargs: Any,
) -> Callable:
    """Privacy capability handler."""

    def handler(orchestrator: Any) -> None:
        configure_data_privacy(
            orchestrator,
            anonymize_pii=anonymize_pii,
            detect_secrets=detect_secrets,
            **kwargs,
        )

    return handler


@capability(
    name="framework_secrets_masking",
    capability_type=CapabilityType.SAFETY,
    version="1.0",
    description="Framework-level secrets masking",
    getter="get_secrets_masking_config",
)
def secrets_masking_capability(
    enabled: bool = True,
    replacement: str = "[REDACTED]",
    **kwargs: Any,
) -> Callable:
    """Secrets masking capability handler."""

    def handler(orchestrator: Any) -> None:
        configure_secrets_masking(
            orchestrator,
            enabled=enabled,
            replacement=replacement,
            **kwargs,
        )

    return handler


@capability(
    name="framework_audit_logging",
    capability_type=CapabilityType.SAFETY,
    version="1.0",
    description="Framework-level audit logging for privacy",
    getter="get_audit_logging_config",
)
def audit_logging_capability(
    enabled: bool = True,
    log_data_access: bool = True,
    **kwargs: Any,
) -> Callable:
    """Audit logging capability handler."""

    def handler(orchestrator: Any) -> None:
        configure_audit_logging(
            orchestrator,
            enabled=enabled,
            log_data_access=log_data_access,
            **kwargs,
        )

    return handler


# =============================================================================
# Capability Provider Class
# =============================================================================


class PrivacyCapabilityProvider(BaseCapabilityProvider[Callable[..., None]]):
    """Framework-level privacy capability provider for all verticals.

    This provider offers cross-vertical privacy capabilities:
    - PII detection and anonymization
    - Secrets masking and detection
    - Audit logging for privacy-sensitive operations

    Verticals can delegate privacy configuration to this framework provider
    instead of implementing their own privacy capabilities.

    Example:
        # Use in any vertical
        from victor.framework.capabilities.privacy import PrivacyCapabilityProvider

        provider = PrivacyCapabilityProvider()
        provider.apply_data_privacy(orchestrator, pii_columns=["email", "ssn"])
        provider.apply_secrets_masking(orchestrator)

        # Or use base provider interface
        privacy_cap = provider.get_capability("data_privacy")
        privacy_cap(orchestrator)
    """

    def __init__(self):
        """Initialize the privacy capability provider."""
        self._applied: Set[str] = set()
        # Map capability names to their handler functions
        self._capabilities: Dict[str, Callable[..., None]] = {
            "data_privacy": configure_data_privacy,
            "secrets_masking": configure_secrets_masking,
            "audit_logging": configure_audit_logging,
        }
        # Capability metadata for discovery
        self._metadata: Dict[str, CapabilityMetadata] = {
            "data_privacy": CapabilityMetadata(
                name="data_privacy",
                description="Framework-level privacy and PII management",
                version="1.0",
                tags=["privacy", "pii", "anonymization", "safety", "framework"],
            ),
            "secrets_masking": CapabilityMetadata(
                name="secrets_masking",
                description="Framework-level secrets masking and detection",
                version="1.0",
                dependencies=["data_privacy"],
                tags=["secrets", "masking", "security", "framework"],
            ),
            "audit_logging": CapabilityMetadata(
                name="audit_logging",
                description="Framework-level audit logging for privacy",
                version="1.0",
                dependencies=["data_privacy"],
                tags=["audit", "logging", "compliance", "framework"],
            ),
        }

    def get_capabilities(self) -> Dict[str, Callable[..., None]]:
        """Return all registered capabilities.

        Returns:
            Dictionary mapping capability names to handler functions.
        """
        return self._capabilities.copy()

    def get_capability_metadata(self) -> Dict[str, CapabilityMetadata]:
        """Return metadata for all registered capabilities.

        Returns:
            Dictionary mapping capability names to their metadata.
        """
        return self._metadata.copy()

    def apply_data_privacy(
        self,
        orchestrator: Any,
        **kwargs: Any,
    ) -> None:
        """Apply data privacy capability.

        Args:
            orchestrator: Target orchestrator
            **kwargs: Privacy options
        """
        configure_data_privacy(orchestrator, **kwargs)
        self._applied.add("data_privacy")

    def apply_secrets_masking(
        self,
        orchestrator: Any,
        **kwargs: Any,
    ) -> None:
        """Apply secrets masking capability.

        Args:
            orchestrator: Target orchestrator
            **kwargs: Secrets masking options
        """
        configure_secrets_masking(orchestrator, **kwargs)
        self._applied.add("secrets_masking")

    def apply_audit_logging(
        self,
        orchestrator: Any,
        **kwargs: Any,
    ) -> None:
        """Apply audit logging capability.

        Args:
            orchestrator: Target orchestrator
            **kwargs: Audit logging options
        """
        configure_audit_logging(orchestrator, **kwargs)
        self._applied.add("audit_logging")

    def apply_all(
        self,
        orchestrator: Any,
        **kwargs: Any,
    ) -> None:
        """Apply all privacy capabilities with defaults.

        Args:
            orchestrator: Target orchestrator
            **kwargs: Shared options
        """
        self.apply_data_privacy(orchestrator)
        self.apply_secrets_masking(orchestrator)
        self.apply_audit_logging(orchestrator)

    def get_applied(self) -> Set[str]:
        """Get set of applied capability names.

        Returns:
            Set of applied capability names
        """
        return self._applied.copy()


# =============================================================================
# CAPABILITIES List for CapabilityLoader Discovery
# =============================================================================


CAPABILITIES: List[CapabilityEntry] = [
    CapabilityEntry(
        capability=OrchestratorCapability(
            name="framework_privacy",
            capability_type=CapabilityType.SAFETY,
            version="1.0",
            setter="configure_data_privacy",
            getter="get_privacy_config",
            description="Framework-level privacy and PII management",
        ),
        handler=configure_data_privacy,
        getter_handler=get_privacy_config,
    ),
    CapabilityEntry(
        capability=OrchestratorCapability(
            name="framework_secrets_masking",
            capability_type=CapabilityType.SAFETY,
            version="1.0",
            setter="configure_secrets_masking",
            getter="get_secrets_masking_config",
            description="Framework-level secrets masking",
        ),
        handler=configure_secrets_masking,
        getter_handler=get_secrets_masking_config,
    ),
    CapabilityEntry(
        capability=OrchestratorCapability(
            name="framework_audit_logging",
            capability_type=CapabilityType.SAFETY,
            version="1.0",
            setter="configure_audit_logging",
            getter="get_audit_logging_config",
            description="Framework-level audit logging for privacy",
        ),
        handler=configure_audit_logging,
        getter_handler=get_audit_logging_config,
    ),
]


# =============================================================================
# Convenience Functions
# =============================================================================


def get_framework_privacy_capabilities() -> List[CapabilityEntry]:
    """Get all framework privacy capability entries.

    Returns:
        List of capability entries for loader registration
    """
    return CAPABILITIES.copy()


__all__ = [
    # Handlers
    "configure_data_privacy",
    "configure_secrets_masking",
    "configure_audit_logging",
    # Getters
    "get_privacy_config",
    "get_secrets_masking_config",
    "get_audit_logging_config",
    # Provider class
    "PrivacyCapabilityProvider",
    # Capability list
    "CAPABILITIES",
    # Convenience functions
    "get_framework_privacy_capabilities",
]

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

"""Dynamic capability definitions for the data analysis vertical.

This module provides capability declarations that can be loaded
dynamically by the CapabilityLoader, enabling runtime extension
of the data analysis vertical with custom functionality.

The module follows the CapabilityLoader's discovery patterns:
1. CAPABILITIES list for batch registration
2. @capability decorator for function-based capabilities
3. Capability classes for complex implementations

Example:
    # Register capabilities with loader
    from victor.framework import CapabilityLoader
    loader = CapabilityLoader()
    loader.load_from_module("victor.dataanalysis.capabilities")

    # Or use directly
    from victor_dataanalysis.capabilities import (
        get_data_analysis_capabilities,
        DataAnalysisCapabilityProvider,
    )
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING

from victor_contracts import CapabilityType, OrchestratorCapability
from victor_contracts.capabilities import CapabilityEntry, capability
from victor_contracts.capabilities import (
    load_capability_config,
    store_capability_config,
)
from victor_contracts import BaseCapabilityProvider, CapabilityMetadata

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# =============================================================================
# Capability Config Helpers (P1: Framework CapabilityConfigService Migration)
# =============================================================================


_DATA_QUALITY_DEFAULTS: Dict[str, Any] = {
    "min_completeness": 0.9,
    "max_outlier_ratio": 0.05,
    "require_type_validation": True,
    "handle_missing": "impute",
}
_VISUALIZATION_DEFAULTS: Dict[str, Any] = {
    "backend": "matplotlib",
    "theme": "seaborn-v0_8-whitegrid",
    "figure_size": (10, 6),
    "dpi": 100,
    "save_format": "png",
}
_STATISTICS_DEFAULTS: Dict[str, Any] = {
    "significance_level": 0.05,
    "confidence_interval": 0.95,
    "multiple_testing_correction": "bonferroni",
    "effect_size_threshold": 0.2,
}
_ML_DEFAULTS: Dict[str, Any] = {
    "framework": "sklearn",
    "cv_folds": 5,
    "test_size": 0.2,
    "random_state": 42,
    "hyperparameter_tuning": True,
    "tuning_method": "grid",
}

# =============================================================================
# Capability Handlers
# =============================================================================


def configure_data_quality(
    orchestrator: Any,
    *,
    min_completeness: float = 0.9,
    max_outlier_ratio: float = 0.05,
    require_type_validation: bool = True,
    handle_missing: str = "impute",
) -> None:
    """Configure data quality rules for the orchestrator.

    This capability configures data quality checks and handling
    strategies for data analysis tasks.

    Args:
        orchestrator: Target orchestrator
        min_completeness: Minimum data completeness ratio (0-1)
        max_outlier_ratio: Maximum allowed outlier ratio
        require_type_validation: Require data type validation
        handle_missing: Strategy for missing values: "impute", "drop", "flag"
    """
    store_capability_config(
        orchestrator,
        "data_quality_config",
        {
            "min_completeness": min_completeness,
            "max_outlier_ratio": max_outlier_ratio,
            "require_type_validation": require_type_validation,
            "handle_missing": handle_missing,
        },
    )

    logger.info(
        f"Configured data quality: completeness>={min_completeness:.0%}, "
        f"missing={handle_missing}"
    )


def get_data_quality(orchestrator: Any) -> Dict[str, Any]:
    """Get current data quality configuration.

    Args:
        orchestrator: Target orchestrator

    Returns:
        Data quality configuration dict
    """
    return load_capability_config(
        orchestrator,
        "data_quality_config",
        _DATA_QUALITY_DEFAULTS,
        legacy_service_names=["data_quality"],
    )


def configure_visualization_style(
    orchestrator: Any,
    *,
    default_backend: str = "matplotlib",
    theme: str = "seaborn-v0_8-whitegrid",
    figure_size: tuple = (10, 6),
    dpi: int = 100,
    save_format: str = "png",
) -> None:
    """Configure visualization style for data analysis.

    Args:
        orchestrator: Target orchestrator
        default_backend: Plotting backend (matplotlib, plotly, seaborn)
        theme: Plot theme/style
        figure_size: Default figure size (width, height) in inches
        dpi: Resolution in dots per inch
        save_format: Default save format (png, svg, pdf)
    """
    store_capability_config(
        orchestrator,
        "visualization_config",
        {
            "backend": default_backend,
            "theme": theme,
            "figure_size": figure_size,
            "dpi": dpi,
            "save_format": save_format,
        },
    )

    logger.info(f"Configured visualization: backend={default_backend}, theme={theme}")


def get_visualization_style(orchestrator: Any) -> Dict[str, Any]:
    """Get current visualization style configuration.

    Args:
        orchestrator: Target orchestrator

    Returns:
        Visualization configuration dict
    """
    return load_capability_config(
        orchestrator,
        "visualization_config",
        _VISUALIZATION_DEFAULTS,
        legacy_service_names=["visualization"],
    )


def configure_statistical_analysis(
    orchestrator: Any,
    *,
    significance_level: float = 0.05,
    confidence_interval: float = 0.95,
    multiple_testing_correction: str = "bonferroni",
    effect_size_threshold: float = 0.2,
) -> None:
    """Configure statistical analysis parameters.

    Args:
        orchestrator: Target orchestrator
        significance_level: P-value threshold for significance
        confidence_interval: Confidence interval level
        multiple_testing_correction: Correction method (bonferroni, holm, fdr_bh)
        effect_size_threshold: Minimum effect size for practical significance
    """
    store_capability_config(
        orchestrator,
        "statistics_config",
        {
            "significance_level": significance_level,
            "confidence_interval": confidence_interval,
            "multiple_testing_correction": multiple_testing_correction,
            "effect_size_threshold": effect_size_threshold,
        },
    )

    logger.info(
        f"Configured statistics: alpha={significance_level}, " f"CI={confidence_interval:.0%}"
    )


def get_statistical_config(orchestrator: Any) -> Dict[str, Any]:
    """Get current statistical configuration.

    Args:
        orchestrator: Target orchestrator

    Returns:
        Statistical configuration dict
    """
    return load_capability_config(
        orchestrator,
        "statistics_config",
        _STATISTICS_DEFAULTS,
        legacy_service_names=["statistical_analysis"],
    )


def configure_ml_pipeline(
    orchestrator: Any,
    *,
    default_framework: str = "sklearn",
    cross_validation_folds: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
    enable_hyperparameter_tuning: bool = True,
    tuning_method: str = "grid",
) -> None:
    """Configure machine learning pipeline settings.

    Args:
        orchestrator: Target orchestrator
        default_framework: ML framework (sklearn, xgboost, lightgbm)
        cross_validation_folds: Number of CV folds
        test_size: Test set proportion
        random_state: Random seed for reproducibility
        enable_hyperparameter_tuning: Enable hyperparameter optimization
        tuning_method: Tuning method (grid, random, bayesian)
    """
    store_capability_config(
        orchestrator,
        "ml_config",
        {
            "framework": default_framework,
            "cv_folds": cross_validation_folds,
            "test_size": test_size,
            "random_state": random_state,
            "hyperparameter_tuning": enable_hyperparameter_tuning,
            "tuning_method": tuning_method,
        },
    )

    logger.info(
        f"Configured ML pipeline: framework={default_framework}, "
        f"cv_folds={cross_validation_folds}"
    )


def get_ml_config(orchestrator: Any) -> Dict[str, Any]:
    """Get current ML configuration.

    Args:
        orchestrator: Target orchestrator

    Returns:
        ML configuration dict
    """
    return load_capability_config(
        orchestrator,
        "ml_config",
        _ML_DEFAULTS,
        legacy_service_names=["ml_pipeline"],
    )


def configure_data_privacy(
    orchestrator: Any,
    *,
    anonymize_pii: bool = True,
    pii_columns: Optional[List[str]] = None,
    hash_identifiers: bool = True,
    log_access: bool = True,
) -> None:
    """Configure data privacy settings.

    Delegates to framework PrivacyCapabilityProvider for cross-vertical privacy management.

    Args:
        orchestrator: Target orchestrator
        anonymize_pii: Whether to anonymize PII columns
        pii_columns: List of column names containing PII
        hash_identifiers: Hash identifier columns
        log_access: Log data access for audit trail
    """
    # Delegate to framework privacy capability
    from victor_contracts.capabilities import (
        configure_data_privacy as framework_privacy,
    )

    framework_privacy(
        orchestrator,
        anonymize_pii=anonymize_pii,
        pii_columns=pii_columns,
        hash_identifiers=hash_identifiers,
        log_access=log_access,
    )


def get_privacy_config(orchestrator: Any) -> Dict[str, Any]:
    """Get current privacy configuration.

    Delegates to framework PrivacyCapabilityProvider for cross-vertical privacy management.

    Args:
        orchestrator: Target orchestrator

    Returns:
        Privacy configuration dict
    """
    # Delegate to framework privacy capability
    from victor_contracts.capabilities import (
        get_privacy_config as framework_get_privacy,
    )

    return framework_get_privacy(orchestrator)


# =============================================================================
# Decorated Capability Functions
# =============================================================================


@capability(
    name="data_quality",
    capability_type=CapabilityType.MODE,
    version="1.0",
    description="Data quality rules and validation settings",
)
def data_quality_capability(
    min_completeness: float = 0.9,
    handle_missing: str = "impute",
    **kwargs: Any,
) -> Callable:
    """Data quality capability handler."""

    def handler(orchestrator: Any) -> None:
        configure_data_quality(
            orchestrator,
            min_completeness=min_completeness,
            handle_missing=handle_missing,
            **kwargs,
        )

    return handler


@capability(
    name="visualization_style",
    capability_type=CapabilityType.MODE,
    version="1.0",
    description="Visualization and plotting configuration",
    getter="get_visualization_style",
)
def visualization_style_capability(
    default_backend: str = "matplotlib",
    theme: str = "seaborn-v0_8-whitegrid",
    **kwargs: Any,
) -> Callable:
    """Visualization style capability handler."""

    def handler(orchestrator: Any) -> None:
        configure_visualization_style(
            orchestrator,
            default_backend=default_backend,
            theme=theme,
            **kwargs,
        )

    return handler


@capability(
    name="statistical_analysis",
    capability_type=CapabilityType.MODE,
    version="1.0",
    description="Statistical analysis configuration",
)
def statistical_analysis_capability(
    significance_level: float = 0.05,
    confidence_interval: float = 0.95,
    **kwargs: Any,
) -> Callable:
    """Statistical analysis capability handler."""

    def handler(orchestrator: Any) -> None:
        configure_statistical_analysis(
            orchestrator,
            significance_level=significance_level,
            confidence_interval=confidence_interval,
            **kwargs,
        )

    return handler


@capability(
    name="ml_pipeline",
    capability_type=CapabilityType.TOOL,
    version="1.0",
    description="Machine learning pipeline configuration",
)
def ml_pipeline_capability(
    default_framework: str = "sklearn",
    cross_validation_folds: int = 5,
    **kwargs: Any,
) -> Callable:
    """ML pipeline capability handler."""

    def handler(orchestrator: Any) -> None:
        configure_ml_pipeline(
            orchestrator,
            default_framework=default_framework,
            cross_validation_folds=cross_validation_folds,
            **kwargs,
        )

    return handler


@capability(
    name="data_privacy",
    capability_type=CapabilityType.SAFETY,
    version="1.0",
    description="Data privacy and anonymization settings",
)
def data_privacy_capability(
    anonymize_pii: bool = True,
    **kwargs: Any,
) -> Callable:
    """Data privacy capability handler."""

    def handler(orchestrator: Any) -> None:
        configure_data_privacy(
            orchestrator,
            anonymize_pii=anonymize_pii,
            **kwargs,
        )

    return handler


# =============================================================================
# Capability Provider Class
# =============================================================================


class DataAnalysisCapabilityProvider(BaseCapabilityProvider[Callable[..., None]]):
    """Provider for data analysis-specific capabilities.

    This class provides a structured way to access and apply
    data analysis capabilities to an orchestrator. It inherits from
    BaseCapabilityProvider for consistent capability registration
    and discovery across all verticals.

    Example:
        provider = DataAnalysisCapabilityProvider()

        # List available capabilities
        print(provider.list_capabilities())

        # Apply specific capabilities
        provider.apply_data_quality(orchestrator)
        provider.apply_visualization_style(orchestrator, backend="plotly")

        # Use BaseCapabilityProvider interface
        cap = provider.get_capability("data_quality")
        if cap:
            cap(orchestrator)
    """

    def __init__(self):
        """Initialize the capability provider."""
        self._applied: Set[str] = set()
        # Map capability names to their handler functions
        self._capabilities: Dict[str, Callable[..., None]] = {
            "data_quality": configure_data_quality,
            "visualization_style": configure_visualization_style,
            "statistical_analysis": configure_statistical_analysis,
            "ml_pipeline": configure_ml_pipeline,
            "data_privacy": configure_data_privacy,
        }
        # Capability metadata for discovery
        self._metadata: Dict[str, CapabilityMetadata] = {
            "data_quality": CapabilityMetadata(
                name="data_quality",
                description="Data quality rules and validation settings",
                version="1.0",
                tags=["quality", "validation", "data-cleaning"],
            ),
            "visualization_style": CapabilityMetadata(
                name="visualization_style",
                description="Visualization and plotting configuration",
                version="1.0",
                tags=["visualization", "charts", "plotting"],
            ),
            "statistical_analysis": CapabilityMetadata(
                name="statistical_analysis",
                description="Statistical analysis configuration",
                version="1.0",
                tags=["statistics", "hypothesis-testing", "analysis"],
            ),
            "ml_pipeline": CapabilityMetadata(
                name="ml_pipeline",
                description="Machine learning pipeline configuration",
                version="1.0",
                dependencies=["data_quality"],
                tags=["ml", "machine-learning", "training"],
            ),
            "data_privacy": CapabilityMetadata(
                name="data_privacy",
                description="Data privacy and anonymization settings",
                version="1.0",
                tags=["privacy", "pii", "anonymization", "safety"],
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

    def apply_data_quality(
        self,
        orchestrator: Any,
        **kwargs: Any,
    ) -> None:
        """Apply data quality capability.

        Args:
            orchestrator: Target orchestrator
            **kwargs: Data quality options
        """
        configure_data_quality(orchestrator, **kwargs)
        self._applied.add("data_quality")

    def apply_visualization_style(
        self,
        orchestrator: Any,
        **kwargs: Any,
    ) -> None:
        """Apply visualization style capability.

        Args:
            orchestrator: Target orchestrator
            **kwargs: Visualization options
        """
        configure_visualization_style(orchestrator, **kwargs)
        self._applied.add("visualization_style")

    def apply_statistical_analysis(
        self,
        orchestrator: Any,
        **kwargs: Any,
    ) -> None:
        """Apply statistical analysis capability.

        Args:
            orchestrator: Target orchestrator
            **kwargs: Statistical options
        """
        configure_statistical_analysis(orchestrator, **kwargs)
        self._applied.add("statistical_analysis")

    def apply_ml_pipeline(
        self,
        orchestrator: Any,
        **kwargs: Any,
    ) -> None:
        """Apply ML pipeline capability.

        Args:
            orchestrator: Target orchestrator
            **kwargs: ML options
        """
        configure_ml_pipeline(orchestrator, **kwargs)
        self._applied.add("ml_pipeline")

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

    def apply_all(
        self,
        orchestrator: Any,
        **kwargs: Any,
    ) -> None:
        """Apply all data analysis capabilities with defaults.

        Args:
            orchestrator: Target orchestrator
            **kwargs: Shared options
        """
        self.apply_data_quality(orchestrator)
        self.apply_visualization_style(orchestrator)
        self.apply_statistical_analysis(orchestrator)
        self.apply_ml_pipeline(orchestrator)
        self.apply_data_privacy(orchestrator)

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
            name="dataanalysis_quality",
            capability_type=CapabilityType.MODE,
            version="1.0",
            setter="configure_data_quality",
            getter="get_data_quality",
            description="Data quality rules and validation settings",
        ),
        handler=configure_data_quality,
        getter_handler=get_data_quality,
    ),
    CapabilityEntry(
        capability=OrchestratorCapability(
            name="dataanalysis_visualization",
            capability_type=CapabilityType.MODE,
            version="1.0",
            setter="configure_visualization_style",
            getter="get_visualization_style",
            description="Visualization and plotting configuration",
        ),
        handler=configure_visualization_style,
        getter_handler=get_visualization_style,
    ),
    CapabilityEntry(
        capability=OrchestratorCapability(
            name="dataanalysis_statistics",
            capability_type=CapabilityType.MODE,
            version="1.0",
            setter="configure_statistical_analysis",
            getter="get_statistical_config",
            description="Statistical analysis configuration",
        ),
        handler=configure_statistical_analysis,
        getter_handler=get_statistical_config,
    ),
    CapabilityEntry(
        capability=OrchestratorCapability(
            name="dataanalysis_ml",
            capability_type=CapabilityType.TOOL,
            version="1.0",
            setter="configure_ml_pipeline",
            getter="get_ml_config",
            description="Machine learning pipeline configuration",
        ),
        handler=configure_ml_pipeline,
        getter_handler=get_ml_config,
    ),
    CapabilityEntry(
        capability=OrchestratorCapability(
            name="dataanalysis_privacy",
            capability_type=CapabilityType.SAFETY,
            version="1.0",
            setter="configure_data_privacy",
            getter="get_privacy_config",
            description="Data privacy and anonymization settings",
        ),
        handler=configure_data_privacy,
        getter_handler=get_privacy_config,
    ),
]


# =============================================================================
# Convenience Functions
# =============================================================================


def get_data_analysis_capabilities() -> List[CapabilityEntry]:
    """Get all data analysis capability entries.

    Returns:
        List of capability entries for loader registration
    """
    return CAPABILITIES.copy()


def create_data_analysis_capability_loader() -> Any:
    """Create a CapabilityLoader pre-configured for data analysis vertical.

    Returns:
        CapabilityLoader with data analysis capabilities registered
    """
    from victor_contracts.capabilities import create_runtime_capability_loader

    return create_runtime_capability_loader(
        CAPABILITIES,
        source_module="victor_dataanalysis.capabilities",
    )


# =============================================================================
# SOLID: Centralized Config Storage
# =============================================================================


def get_capability_configs() -> Dict[str, Any]:
    """Get data analysis capability configurations for centralized storage.

    Returns default data analysis configuration for VerticalContext storage.
    This replaces direct orchestrator data_quality/visualization/ml_config assignment.

    Returns:
        Dict with default data analysis capability configurations
    """
    return {
        "data_quality_config": {
            "min_completeness": 0.9,
            "max_outlier_ratio": 0.05,
            "require_type_validation": True,
            "handle_missing": "impute",
        },
        "visualization_config": {
            "backend": "matplotlib",
            "theme": "seaborn-v0_8-whitegrid",
            "figure_size": (10, 6),
            "dpi": 100,
            "save_format": "png",
        },
        "statistics_config": {
            "significance_level": 0.05,
            "confidence_interval": 0.95,
            "multiple_testing_correction": "bonferroni",
            "effect_size_threshold": 0.2,
        },
        "ml_config": {
            "framework": "sklearn",
            "cv_folds": 5,
            "test_size": 0.2,
            "random_state": 42,
            "hyperparameter_tuning": True,
            "tuning_method": "grid",
        },
        "privacy_config": {
            "anonymize_pii": True,
            "pii_columns": [],
            "hash_identifiers": True,
            "log_access": True,
        },
    }


__all__ = [
    # Handlers
    "configure_data_quality",
    "configure_visualization_style",
    "configure_statistical_analysis",
    "configure_ml_pipeline",
    "configure_data_privacy",
    # Getters
    "get_data_quality",
    "get_visualization_style",
    "get_statistical_config",
    "get_ml_config",
    "get_privacy_config",
    # Provider class and base types
    "DataAnalysisCapabilityProvider",
    "CapabilityMetadata",  # Re-exported from framework for convenience
    # Capability list for loader
    "CAPABILITIES",
    # Convenience functions
    "get_data_analysis_capabilities",
    "create_data_analysis_capability_loader",
    # SOLID: Centralized config storage
    "get_capability_configs",
]

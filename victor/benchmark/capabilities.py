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

"""Dynamic capability definitions for the Benchmark vertical.

This module provides capability declarations that can be loaded
dynamically by the CapabilityLoader, enabling runtime extension
of the Benchmark vertical with custom functionality.

The module follows the CapabilityLoader's discovery patterns:
1. CAPABILITIES list for batch registration
2. @capability decorator for function-based capabilities
3. Capability classes for complex implementations

Example:
    # Register capabilities with loader
    from victor.framework import CapabilityLoader
    loader = CapabilityLoader()
    loader.load_from_module("victor.benchmark.capabilities")

    # Or use directly
    from victor.benchmark.capabilities import (
        get_benchmark_capabilities,
        BenchmarkCapabilityProvider,
    )
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING

from victor.framework.protocols import CapabilityType, OrchestratorCapability
from victor.framework.capability_loader import CapabilityEntry, capability
from victor.framework.capability_config_helpers import (
    load_capability_config,
    store_capability_config,
    update_capability_config_section,
)
from victor.framework.capabilities import BaseCapabilityProvider, CapabilityMetadata

if TYPE_CHECKING:
    from victor.core.protocols import OrchestratorProtocol as AgentOrchestrator

logger = logging.getLogger(__name__)


# =============================================================================
# Capability Config Helpers (P1: Framework CapabilityConfigService Migration)
# =============================================================================


_PASSK_DEFAULTS: Dict[str, Any] = {
    "k_value": 10,
    "temperature_range": [0.2, 0.4, 0.6, 0.8],
    "max_samples": 100,
    "enable_early_stopping": True,
    "stop_after_n_success": 5,
}
_BENCHMARK_DEFAULTS: Dict[str, Any] = {
    "swe_bench": {
        "enable_patch_generation": True,
        "require_test_verification": True,
        "max_iterations": 5,
        "timeout_seconds": 300,
        "enable_dry_run": False,
    },
    "passk": dict(_PASSK_DEFAULTS),
}
_METRICS_DEFAULTS: Dict[str, Any] = {
    "track_token_usage": True,
    "track_execution_time": True,
    "track_tool_calls": True,
    "track_test_results": True,
    "enable_detailed_tracing": False,
    "output_format": "json",
}
_TEST_GENERATION_DEFAULTS: Dict[str, Any] = {
    "test_framework": "pytest",
    "coverage_threshold": 0.8,
    "generate_unit_tests": True,
    "generate_integration_tests": False,
    "use_property_based_testing": False,
}
_CODE_QUALITY_DEFAULTS: Dict[str, Any] = {
    "enable_linting": True,
    "linting_tool": "ruff",
    "enable_formatting": True,
    "formatting_tool": "black",
    "enable_type_checking": False,
    "type_checking_tool": "mypy",
    "max_complexity": 10,
}
_PERFORMANCE_DEFAULTS: Dict[str, Any] = {
    "measure_latency": True,
    "measure_throughput": True,
    "benchmark_iterations": 10,
    "warmup_iterations": 2,
    "enable_profiling": False,
    "profile_memory": False,
}


def _load_config(orchestrator: Any, name: str, defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Load config from framework service when available, else fallback to orchestrator attr."""
    return load_capability_config(orchestrator, name, defaults)


def _store_config(orchestrator: Any, name: str, config: Dict[str, Any]) -> None:
    """Store config in framework service when available, else fallback to orchestrator attr."""
    store_capability_config(orchestrator, name, config)


def _store_benchmark_section(orchestrator: Any, section: str, config: Dict[str, Any]) -> None:
    """Store/update benchmark_config section with service-first behavior."""
    update_capability_config_section(
        orchestrator,
        root_name="benchmark_config",
        section_name=section,
        section_config=config,
        root_defaults=_BENCHMARK_DEFAULTS,
    )


# =============================================================================
# Capability Handlers
# =============================================================================


def configure_swe_bench_execution(
    orchestrator: Any,
    *,
    enable_patch_generation: bool = True,
    require_test_verification: bool = True,
    max_iterations: int = 5,
    timeout_seconds: int = 300,
    enable_dry_run: bool = False,
) -> None:
    """Configure SWE-bench task execution settings.

    This capability configures how the benchmark agent handles
    SWE-bench style issue resolution tasks.

    Args:
        orchestrator: Target orchestrator
        enable_patch_generation: Enable automatic patch generation
        require_test_verification: Require tests to pass before completion
        max_iterations: Maximum iterations for fix verification
        timeout_seconds: Timeout per task in seconds
        enable_dry_run: Run in dry-run mode without making changes
    """
    _store_benchmark_section(
        orchestrator,
        "swe_bench",
        {
            "enable_patch_generation": enable_patch_generation,
            "require_test_verification": require_test_verification,
            "max_iterations": max_iterations,
            "timeout_seconds": timeout_seconds,
            "enable_dry_run": enable_dry_run,
        },
    )

    # Configure safety patterns for dry-run mode
    if enable_dry_run:
        # BenchmarkSafetyExtension may not be present in all runtime/test builds.
        try:
            from victor.benchmark.safety import BenchmarkSafetyExtension

            safety = BenchmarkSafetyExtension()
            safety.enable_dry_run_mode()
        except Exception:
            logger.debug(
                "BenchmarkSafetyExtension unavailable; dry-run mode limited to configuration flag",
                exc_info=True,
            )

    logger.info(
        f"Configured SWE-bench execution: patch_gen={enable_patch_generation}, "
        f"test_verify={require_test_verification}"
    )


def configure_passk_evaluation(
    orchestrator: Any,
    *,
    k_value: int = 10,
    temperature_range: Optional[List[float]] = None,
    max_samples: int = 100,
    enable_early_stopping: bool = True,
    stop_after_n_success: int = 5,
) -> None:
    """Configure Pass@k evaluation settings.

    This capability configures pass@k metric evaluation for
    code generation benchmarks like HumanEval and MBPP.

    Args:
        orchestrator: Target orchestrator
        k_value: Number of samples to generate for pass@k calculation
        temperature_range: Temperature values to sample across
        max_samples: Maximum samples per task
        enable_early_stopping: Stop early if all tests pass
        stop_after_n_success: Stop after N successful samples
    """
    effective_temperature_range = (
        list(temperature_range)
        if temperature_range is not None
        else list(_PASSK_DEFAULTS["temperature_range"])
    )

    _store_benchmark_section(
        orchestrator,
        "passk",
        {
            "k_value": k_value,
            "temperature_range": effective_temperature_range,
            "max_samples": max_samples,
            "enable_early_stopping": enable_early_stopping,
            "stop_after_n_success": stop_after_n_success,
        },
    )

    logger.info(f"Configured Pass@k evaluation: k={k_value}, max_samples={max_samples}")


def get_passk_config(orchestrator: Any) -> Dict[str, Any]:
    """Get current pass@k evaluation configuration.

    Args:
        orchestrator: Target orchestrator

    Returns:
        Pass@k configuration dict
    """
    return _load_config(orchestrator, "benchmark_config", _BENCHMARK_DEFAULTS).get(
        "passk",
        dict(_PASSK_DEFAULTS),
    )


def configure_metrics_collection(
    orchestrator: Any,
    *,
    track_token_usage: bool = True,
    track_execution_time: bool = True,
    track_tool_calls: bool = True,
    track_test_results: bool = True,
    enable_detailed_tracing: bool = False,
    output_format: str = "json",
) -> None:
    """Configure metrics collection during benchmark execution.

    Args:
        orchestrator: Target orchestrator
        track_token_usage: Track input/output token usage
        track_execution_time: Track wall-clock execution time
        track_tool_calls: Track all tool calls made
        track_test_results: Track test pass/fail results
        enable_detailed_tracing: Enable detailed execution traces
        output_format: Output format (json, csv, or both)
    """
    _store_config(
        orchestrator,
        "metrics_config",
        {
            "track_token_usage": track_token_usage,
            "track_execution_time": track_execution_time,
            "track_tool_calls": track_tool_calls,
            "track_test_results": track_test_results,
            "enable_detailed_tracing": enable_detailed_tracing,
            "output_format": output_format,
        },
    )

    logger.info(f"Configured metrics collection: format={output_format}")


def get_metrics_config(orchestrator: Any) -> Dict[str, Any]:
    """Get current metrics collection configuration.

    Args:
        orchestrator: Target orchestrator

    Returns:
        Metrics configuration dict
    """
    return _load_config(orchestrator, "metrics_config", _METRICS_DEFAULTS)


def configure_test_generation(
    orchestrator: Any,
    *,
    test_framework: str = "pytest",
    coverage_threshold: float = 0.8,
    generate_unit_tests: bool = True,
    generate_integration_tests: bool = False,
    use_property_based_testing: bool = False,
) -> None:
    """Configure test generation for benchmark tasks.

    Args:
        orchestrator: Target orchestrator
        test_framework: Test framework to use (pytest, unittest, jest)
        coverage_threshold: Minimum code coverage threshold
        generate_unit_tests: Generate unit tests
        generate_integration_tests: Generate integration tests
        use_property_based_testing: Use property-based testing (hypothesis)
    """
    _store_config(
        orchestrator,
        "test_generation_config",
        {
            "test_framework": test_framework,
            "coverage_threshold": coverage_threshold,
            "generate_unit_tests": generate_unit_tests,
            "generate_integration_tests": generate_integration_tests,
            "use_property_based_testing": use_property_based_testing,
        },
    )

    logger.info(
        f"Configured test generation: framework={test_framework}, "
        f"coverage>={coverage_threshold:.0%}"
    )


def configure_code_quality_checks(
    orchestrator: Any,
    *,
    enable_linting: bool = True,
    linting_tool: str = "ruff",
    enable_formatting: bool = True,
    formatting_tool: str = "black",
    enable_type_checking: bool = False,
    type_checking_tool: str = "mypy",
    max_complexity: int = 10,
) -> None:
    """Configure code quality checks for generated code.

    Args:
        orchestrator: Target orchestrator
        enable_linting: Enable linting checks
        linting_tool: Linting tool to use
        enable_formatting: Enable formatting checks
        formatting_tool: Formatting tool to use
        enable_type_checking: Enable type checking
        type_checking_tool: Type checking tool to use
        max_complexity: Maximum cyclomatic complexity
    """
    _store_config(
        orchestrator,
        "code_quality_config",
        {
            "enable_linting": enable_linting,
            "linting_tool": linting_tool,
            "enable_formatting": enable_formatting,
            "formatting_tool": formatting_tool,
            "enable_type_checking": enable_type_checking,
            "type_checking_tool": type_checking_tool,
            "max_complexity": max_complexity,
        },
    )

    logger.info(
        f"Configured code quality checks: linting={enable_linting}, "
        f"formatting={enable_formatting}"
    )


def configure_performance_benchmarking(
    orchestrator: Any,
    *,
    measure_latency: bool = True,
    measure_throughput: bool = True,
    benchmark_iterations: int = 10,
    warmup_iterations: int = 2,
    enable_profiling: bool = False,
    profile_memory: bool = False,
) -> None:
    """Configure performance benchmarking settings.

    Args:
        orchestrator: Target orchestrator
        measure_latency: Measure execution latency
        measure_throughput: Measure tasks per second
        benchmark_iterations: Number of benchmark iterations
        warmup_iterations: Number of warmup iterations
        enable_profiling: Enable CPU profiling
        profile_memory: Enable memory profiling
    """
    _store_config(
        orchestrator,
        "performance_config",
        {
            "measure_latency": measure_latency,
            "measure_throughput": measure_throughput,
            "benchmark_iterations": benchmark_iterations,
            "warmup_iterations": warmup_iterations,
            "enable_profiling": enable_profiling,
            "profile_memory": profile_memory,
        },
    )

    logger.info(f"Configured performance benchmarking: iterations={benchmark_iterations}")


# =============================================================================
# Decorated Capability Functions
# =============================================================================


@capability(
    name="benchmark_swe_bench",
    capability_type=CapabilityType.MODE,
    version="1.0",
    description="SWE-bench task execution configuration",
)
def swe_bench_capability(
    enable_patch_generation: bool = True,
    require_test_verification: bool = True,
    **kwargs: Any,
) -> Callable:
    """SWE-bench execution capability handler."""

    def handler(orchestrator: Any) -> None:
        configure_swe_bench_execution(
            orchestrator,
            enable_patch_generation=enable_patch_generation,
            require_test_verification=require_test_verification,
            **kwargs,
        )

    return handler


@capability(
    name="benchmark_passk",
    capability_type=CapabilityType.MODE,
    version="1.0",
    description="Pass@k evaluation configuration for code generation",
    getter="get_passk_config",
)
def passk_capability(
    k_value: int = 10,
    max_samples: int = 100,
    **kwargs: Any,
) -> Callable:
    """Pass@k evaluation capability handler."""

    def handler(orchestrator: Any) -> None:
        configure_passk_evaluation(
            orchestrator,
            k_value=k_value,
            max_samples=max_samples,
            **kwargs,
        )

    return handler


@capability(
    name="benchmark_metrics",
    capability_type=CapabilityType.TOOL,
    version="1.0",
    description="Metrics collection during benchmark execution",
    getter="get_metrics_config",
)
def metrics_capability(
    track_token_usage: bool = True,
    track_execution_time: bool = True,
    **kwargs: Any,
) -> Callable:
    """Metrics collection capability handler."""

    def handler(orchestrator: Any) -> None:
        configure_metrics_collection(
            orchestrator,
            track_token_usage=track_token_usage,
            track_execution_time=track_execution_time,
            **kwargs,
        )

    return handler


@capability(
    name="benchmark_test_generation",
    capability_type=CapabilityType.TOOL,
    version="1.0",
    description="Test generation configuration for benchmark tasks",
)
def test_generation_capability(
    test_framework: str = "pytest",
    coverage_threshold: float = 0.8,
    **kwargs: Any,
) -> Callable:
    """Test generation capability handler."""

    def handler(orchestrator: Any) -> None:
        configure_test_generation(
            orchestrator,
            test_framework=test_framework,
            coverage_threshold=coverage_threshold,
            **kwargs,
        )

    return handler


@capability(
    name="benchmark_code_quality",
    capability_type=CapabilityType.SAFETY,
    version="1.0",
    description="Code quality checks for generated code",
)
def code_quality_capability(
    enable_linting: bool = True,
    enable_formatting: bool = True,
    **kwargs: Any,
) -> Callable:
    """Code quality capability handler."""

    def handler(orchestrator: Any) -> None:
        configure_code_quality_checks(
            orchestrator,
            enable_linting=enable_linting,
            enable_formatting=enable_formatting,
            **kwargs,
        )

    return handler


@capability(
    name="benchmark_performance",
    capability_type=CapabilityType.TOOL,
    version="1.0",
    description="Performance benchmarking configuration",
)
def performance_capability(
    measure_latency: bool = True,
    benchmark_iterations: int = 10,
    **kwargs: Any,
) -> Callable:
    """Performance benchmarking capability handler."""

    def handler(orchestrator: Any) -> None:
        configure_performance_benchmarking(
            orchestrator,
            measure_latency=measure_latency,
            benchmark_iterations=benchmark_iterations,
            **kwargs,
        )

    return handler


# =============================================================================
# Capability Provider Class
# =============================================================================


class BenchmarkCapabilityProvider(BaseCapabilityProvider[Callable[..., None]]):
    """Provider for Benchmark-specific capabilities.

    This class provides a structured way to access and apply
    Benchmark capabilities to an orchestrator. It inherits from
    BaseCapabilityProvider for consistent capability registration
    and discovery across all verticals.

    Example:
        provider = BenchmarkCapabilityProvider()

        # List available capabilities
        print(provider.list_capabilities())

        # Apply specific capabilities
        provider.apply_swe_bench_execution(orchestrator)
        provider.apply_passk_evaluation(orchestrator, k_value=10)

        # Use BaseCapabilityProvider interface
        cap = provider.get_capability("metrics_collection")
        if cap:
            cap(orchestrator)
    """

    def __init__(self):
        """Initialize the capability provider."""
        self._applied: Set[str] = set()
        # Map capability names to their handler functions
        self._capabilities: Dict[str, Callable[..., None]] = {
            "swe_bench_execution": configure_swe_bench_execution,
            "passk_evaluation": configure_passk_evaluation,
            "metrics_collection": configure_metrics_collection,
            "test_generation": configure_test_generation,
            "code_quality_checks": configure_code_quality_checks,
            "performance_benchmarking": configure_performance_benchmarking,
        }
        # Capability metadata for discovery
        self._metadata: Dict[str, CapabilityMetadata] = {
            "swe_bench_execution": CapabilityMetadata(
                name="swe_bench_execution",
                description="SWE-bench task execution configuration",
                version="1.0",
                tags=["swe-bench", "issue-resolution", "patch-generation"],
            ),
            "passk_evaluation": CapabilityMetadata(
                name="passk_evaluation",
                description="Pass@k evaluation for code generation benchmarks",
                version="1.0",
                dependencies=["metrics_collection"],
                tags=["passk", "human-eval", "mbpp", "code-generation"],
            ),
            "metrics_collection": CapabilityMetadata(
                name="metrics_collection",
                description="Metrics collection during benchmark execution",
                version="1.0",
                tags=["metrics", "observability", "tracking"],
            ),
            "test_generation": CapabilityMetadata(
                name="test_generation",
                description="Test generation configuration for benchmark tasks",
                version="1.0",
                tags=["testing", "test-generation", "coverage"],
            ),
            "code_quality_checks": CapabilityMetadata(
                name="code_quality_checks",
                description="Code quality checks for generated code",
                version="1.0",
                dependencies=["test_generation"],
                tags=["quality", "linting", "formatting", "type-checking"],
            ),
            "performance_benchmarking": CapabilityMetadata(
                name="performance_benchmarking",
                description="Performance benchmarking configuration",
                version="1.0",
                dependencies=["metrics_collection"],
                tags=["performance", "profiling", "latency", "throughput"],
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

    def apply_swe_bench_execution(
        self,
        orchestrator: Any,
        **kwargs: Any,
    ) -> None:
        """Apply SWE-bench execution capability.

        Args:
            orchestrator: Target orchestrator
            **kwargs: SWE-bench options
        """
        configure_swe_bench_execution(orchestrator, **kwargs)
        self._applied.add("swe_bench_execution")

    def apply_passk_evaluation(
        self,
        orchestrator: Any,
        **kwargs: Any,
    ) -> None:
        """Apply pass@k evaluation capability.

        Args:
            orchestrator: Target orchestrator
            **kwargs: Pass@k options
        """
        configure_passk_evaluation(orchestrator, **kwargs)
        self._applied.add("passk_evaluation")

    def apply_metrics_collection(
        self,
        orchestrator: Any,
        **kwargs: Any,
    ) -> None:
        """Apply metrics collection capability.

        Args:
            orchestrator: Target orchestrator
            **kwargs: Metrics options
        """
        configure_metrics_collection(orchestrator, **kwargs)
        self._applied.add("metrics_collection")

    def apply_test_generation(
        self,
        orchestrator: Any,
        **kwargs: Any,
    ) -> None:
        """Apply test generation capability.

        Args:
            orchestrator: Target orchestrator
            **kwargs: Test generation options
        """
        configure_test_generation(orchestrator, **kwargs)
        self._applied.add("test_generation")

    def apply_code_quality_checks(
        self,
        orchestrator: Any,
        **kwargs: Any,
    ) -> None:
        """Apply code quality checks capability.

        Args:
            orchestrator: Target orchestrator
            **kwargs: Code quality options
        """
        configure_code_quality_checks(orchestrator, **kwargs)
        self._applied.add("code_quality_checks")

    def apply_performance_benchmarking(
        self,
        orchestrator: Any,
        **kwargs: Any,
    ) -> None:
        """Apply performance benchmarking capability.

        Args:
            orchestrator: Target orchestrator
            **kwargs: Performance options
        """
        configure_performance_benchmarking(orchestrator, **kwargs)
        self._applied.add("performance_benchmarking")

    def apply_all(
        self,
        orchestrator: Any,
        **kwargs: Any,
    ) -> None:
        """Apply all Benchmark capabilities with defaults.

        Args:
            orchestrator: Target orchestrator
            **kwargs: Shared options
        """
        self.apply_swe_bench_execution(orchestrator)
        self.apply_passk_evaluation(orchestrator)
        self.apply_metrics_collection(orchestrator)
        self.apply_test_generation(orchestrator)
        self.apply_code_quality_checks(orchestrator)
        self.apply_performance_benchmarking(orchestrator)

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
            name="benchmark_swe_bench",
            capability_type=CapabilityType.MODE,
            version="1.0",
            setter="configure_swe_bench_execution",
            description="SWE-bench task execution configuration",
        ),
        handler=configure_swe_bench_execution,
    ),
    CapabilityEntry(
        capability=OrchestratorCapability(
            name="benchmark_passk",
            capability_type=CapabilityType.MODE,
            version="1.0",
            setter="configure_passk_evaluation",
            getter="get_passk_config",
            description="Pass@k evaluation configuration for code generation",
        ),
        handler=configure_passk_evaluation,
        getter_handler=get_passk_config,
    ),
    CapabilityEntry(
        capability=OrchestratorCapability(
            name="benchmark_metrics",
            capability_type=CapabilityType.TOOL,
            version="1.0",
            setter="configure_metrics_collection",
            getter="get_metrics_config",
            description="Metrics collection during benchmark execution",
        ),
        handler=configure_metrics_collection,
        getter_handler=get_metrics_config,
    ),
    CapabilityEntry(
        capability=OrchestratorCapability(
            name="benchmark_test_generation",
            capability_type=CapabilityType.TOOL,
            version="1.0",
            setter="configure_test_generation",
            description="Test generation configuration for benchmark tasks",
        ),
        handler=configure_test_generation,
    ),
    CapabilityEntry(
        capability=OrchestratorCapability(
            name="benchmark_code_quality",
            capability_type=CapabilityType.SAFETY,
            version="1.0",
            setter="configure_code_quality_checks",
            description="Code quality checks for generated code",
        ),
        handler=configure_code_quality_checks,
    ),
    CapabilityEntry(
        capability=OrchestratorCapability(
            name="benchmark_performance",
            capability_type=CapabilityType.TOOL,
            version="1.0",
            setter="configure_performance_benchmarking",
            description="Performance benchmarking configuration",
        ),
        handler=configure_performance_benchmarking,
    ),
]


# =============================================================================
# Convenience Functions
# =============================================================================


def get_benchmark_capabilities() -> List[CapabilityEntry]:
    """Get all Benchmark capability entries.

    Returns:
        List of capability entries for loader registration
    """
    return CAPABILITIES.copy()


def create_benchmark_capability_loader() -> Any:
    """Create a CapabilityLoader pre-configured for Benchmark vertical.

    Returns:
        CapabilityLoader with Benchmark capabilities registered
    """
    from victor.framework import CapabilityLoader

    loader = CapabilityLoader()

    # Register all Benchmark capabilities
    for entry in CAPABILITIES:
        loader._register_capability_internal(
            capability=entry.capability,
            handler=entry.handler,
            getter_handler=entry.getter_handler,
            source_module="victor.benchmark.capabilities",
        )

    return loader


# =============================================================================
# SOLID: Centralized Config Storage
# =============================================================================


def get_capability_configs() -> Dict[str, Any]:
    """Get Benchmark capability configurations for centralized storage.

    Returns default Benchmark configuration for VerticalContext storage.
    This replaces direct orchestrator.benchmark_config assignment.

    Returns:
        Dict with default Benchmark capability configurations
    """
    return {
        "benchmark_config": {
            "swe_bench": {
                "enable_patch_generation": True,
                "require_test_verification": True,
                "max_iterations": 5,
                "timeout_seconds": 300,
                "enable_dry_run": False,
            },
            "passk": {
                "k_value": 10,
                "temperature_range": [0.2, 0.4, 0.6, 0.8],
                "max_samples": 100,
                "enable_early_stopping": True,
                "stop_after_n_success": 5,
            },
        },
        "metrics_config": {
            "track_token_usage": True,
            "track_execution_time": True,
            "track_tool_calls": True,
            "track_test_results": True,
            "enable_detailed_tracing": False,
            "output_format": "json",
        },
        "test_generation_config": {
            "test_framework": "pytest",
            "coverage_threshold": 0.8,
            "generate_unit_tests": True,
            "generate_integration_tests": False,
            "use_property_based_testing": False,
        },
        "code_quality_config": {
            "enable_linting": True,
            "linting_tool": "ruff",
            "enable_formatting": True,
            "formatting_tool": "black",
            "enable_type_checking": False,
            "type_checking_tool": "mypy",
            "max_complexity": 10,
        },
        "performance_config": {
            "measure_latency": True,
            "measure_throughput": True,
            "benchmark_iterations": 10,
            "warmup_iterations": 2,
            "enable_profiling": False,
            "profile_memory": False,
        },
    }


__all__ = [
    # Handlers
    "configure_swe_bench_execution",
    "configure_passk_evaluation",
    "configure_metrics_collection",
    "configure_test_generation",
    "configure_code_quality_checks",
    "configure_performance_benchmarking",
    # Getters
    "get_passk_config",
    "get_metrics_config",
    # Provider class and base types
    "BenchmarkCapabilityProvider",
    "CapabilityMetadata",  # Re-exported from framework for convenience
    # Capability list for loader
    "CAPABILITIES",
    # Convenience functions
    "get_benchmark_capabilities",
    "create_benchmark_capability_loader",
    # SOLID: Centralized config storage
    "get_capability_configs",
]

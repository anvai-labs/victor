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

"""Performance benchmarks for all framework registry operations.

This module provides comprehensive performance benchmarks for:
1. ChainRegistry - chain registration, lookup, discovery, factory invocation
2. PersonaRegistry - persona registration, lookup, discovery, tag filtering
3. CapabilityProvider - capability enumeration, metadata retrieval, apply overhead
4. Middleware - execution overhead, priority sorting, tool filtering
5. WorkflowProvider - lazy loading, instantiation, hook performance

Performance Targets:
- Registration: < 1ms per item
- Lookup: < 0.1ms
- Discovery (1000 items): < 10ms
- Factory invocation: < 0.5ms
- Middleware execution: < 0.1ms per middleware

Usage:
    pytest tests/performance/test_registry_performance.py -v
    pytest tests/performance/test_registry_performance.py --benchmark-only
    pytest tests/performance/test_registry_performance.py -k "chain_registry" -v
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List

import pytest

from victor.framework.capabilities.base import (
    BaseCapabilityProvider,
    CapabilityMetadata,
)
from victor.framework.chain_registry import (
    ChainRegistry,
    ChainMetadata,
    reset_chain_registry,
)
from victor.framework.persona_registry import (
    PersonaRegistry,
    PersonaSpec,
    reset_persona_registry,
)
from victor.framework.middleware import (
    GitSafetyMiddleware,
    LoggingMiddleware,
    MetricsMiddleware,
    SecretMaskingMiddleware,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@dataclass
class _ChainStub:
    """Lightweight chain object for registry benchmarks."""

    name: str


@pytest.fixture(autouse=True)
def reset_registries():
    """Reset all singleton registries before each test."""
    reset_chain_registry()
    reset_persona_registry()
    yield
    reset_chain_registry()
    reset_persona_registry()


def create_mock_chain(n: int) -> Any:
    """Create a mock chain object for testing.

    Args:
        n: Index for the chain

    Returns:
        Mock chain object
    """
    return _ChainStub(name=f"chain_{n}")


def create_persona_spec(n: int, expertise_areas: List[str] | None = None) -> PersonaSpec:
    """Create a test persona spec.

    Args:
        n: Index for the persona
        expertise_areas: Optional list of expertise areas

    Returns:
        PersonaSpec instance
    """
    expertise = expertise_areas or [f"skill_{i % 10}" for i in range(min(n, 10))]
    return PersonaSpec(
        name=f"persona_{n}",
        role=f"Role {n}",
        expertise=expertise,
        communication_style="formal",
        behavioral_traits=["trait_1", "trait_2"],
        tags=[f"tag_{i % 5}" for i in range(min(n, 5))],
    )


class MockCapability:
    """Mock capability for testing."""

    def __init__(self, name: str):
        self.name = name

    def apply(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Mock apply method."""
        return {**context, f"applied_{self.name}": True}


class TestCapabilityProvider(BaseCapabilityProvider[MockCapability]):
    """Test capability provider for benchmarking."""

    def __init__(self, num_capabilities: int = 10):
        self._capabilities = {
            f"capability_{i}": MockCapability(f"capability_{i}") for i in range(num_capabilities)
        }
        self._metadata = {
            f"capability_{i}": CapabilityMetadata(
                name=f"capability_{i}",
                description=f"Test capability {i}",
                version="1.0",
                tags=[f"tag_{j % 5}" for j in range(min(i, 5))],
            )
            for i in range(num_capabilities)
        }

    def get_capabilities(self) -> Dict[str, MockCapability]:
        return self._capabilities

    def get_capability_metadata(self) -> Dict[str, CapabilityMetadata]:
        return self._metadata


# =============================================================================
# ChainRegistry Performance Tests
# =============================================================================


class TestChainRegistryPerformance:
    """Performance benchmarks for ChainRegistry operations.

    Targets:
    - Registration: < 1ms per chain
    - Lookup: < 0.1ms
    - Discovery (1000 chains): < 10ms
    - Factory invocation: < 0.5ms
    - Singleton overhead: < 0.01ms
    """

    def test_chain_registration_speed_10_items(self, benchmark):
        """Benchmark chain registration with 10 items.

        Expected: < 1ms per registration
        """
        registry = ChainRegistry()

        def register_10_chains():
            for i in range(10):
                registry.register(
                    f"chain_{i}",
                    create_mock_chain(i),
                    vertical="test",
                    description=f"Test chain {i}",
                    tags=[f"tag_{j % 5}" for j in range(i % 5)],
                )

        result = benchmark(register_10_chains)

        # Performance assertion: < 1ms per item
        # 10 items should complete in < 10ms
        assert len(registry.list_chains()) == 10

    def test_chain_registration_speed_100_items(self, benchmark):
        """Benchmark chain registration with 100 items.

        Expected: < 1ms per registration
        """
        registry = ChainRegistry()

        def register_100_chains():
            for i in range(100):
                registry.register(
                    f"chain_{i}",
                    create_mock_chain(i),
                    vertical="test",
                    description=f"Test chain {i}",
                    tags=[f"tag_{j % 5}" for j in range(i % 5)],
                )

        result = benchmark(register_100_chains)

        # Performance assertion: < 1ms per item
        # 100 items should complete in < 100ms
        assert len(registry.list_chains()) == 100

    def test_chain_registration_speed_1000_items(self, benchmark):
        """Benchmark chain registration with 1000 items.

        Expected: < 1ms per registration
        """
        registry = ChainRegistry()

        def register_1000_chains():
            for i in range(1000):
                registry.register(
                    f"chain_{i}",
                    create_mock_chain(i),
                    vertical="test",
                    description=f"Test chain {i}",
                    tags=[f"tag_{j % 5}" for j in range(i % 5)],
                )

        result = benchmark(register_1000_chains)

        # Performance assertion: < 1ms per item
        # 1000 items should complete in < 1000ms (1 second)
        assert len(registry.list_chains()) == 1000

    def test_chain_lookup_speed(self, benchmark):
        """Benchmark chain lookup by name.

        Expected: < 0.1ms per lookup
        """
        registry = ChainRegistry()

        # Register 1000 chains
        for i in range(1000):
            registry.register(
                f"chain_{i}",
                create_mock_chain(i),
                vertical="test",
                description=f"Test chain {i}",
            )

        # Benchmark lookup
        def lookup_chain():
            return registry.get("chain_500", vertical="test")

        result = benchmark(lookup_chain)
        assert result is not None

    def test_chain_discovery_by_vertical(self, benchmark):
        """Benchmark chain discovery by vertical.

        Expected: < 10ms for 1000 items
        """
        registry = ChainRegistry()

        # Register chains across multiple verticals
        for i in range(1000):
            vertical = f"vertical_{i % 10}"
            registry.register(
                f"chain_{i}",
                create_mock_chain(i),
                vertical=vertical,
                description=f"Test chain {i}",
            )

        def find_by_vertical():
            return registry.find_by_vertical("vertical_5")

        result = benchmark(find_by_vertical)
        assert len(result) == 100  # 1000 items / 10 verticals

    def test_chain_discovery_by_tag(self, benchmark):
        """Benchmark chain discovery by tag.

        Expected: < 10ms for 1000 items
        """
        registry = ChainRegistry()

        # Register chains with tags
        for i in range(1000):
            registry.register(
                f"chain_{i}",
                create_mock_chain(i),
                vertical="test",
                description=f"Test chain {i}",
                tags=[f"tag_{j % 10}" for j in range(i % 10)],
            )

        def find_by_tag():
            return registry.find_by_tag("tag_5")

        result = benchmark(find_by_tag)
        # Approximately 1/10 of chains should have tag_5
        assert len(result) > 50

    def test_chain_factory_invocation_speed(self, benchmark):
        """Benchmark chain factory invocation.

        Expected: < 0.5ms per invocation
        """
        registry = ChainRegistry()

        # Register factories
        for i in range(100):
            registry.register_factory(
                f"factory_{i}",
                lambda n=i: create_mock_chain(n),
                vertical="test",
                description=f"Factory {i}",
            )

        def invoke_factory():
            return registry.create("factory_50", vertical="test")

        result = benchmark(invoke_factory)
        assert result is not None

    def test_chain_singleton_overhead(self, benchmark):
        """Benchmark singleton pattern overhead.

        Expected: < 0.01ms per access
        """
        # Reset to ensure clean state
        reset_chain_registry()

        def get_singleton():
            from victor.framework.chain_registry import get_chain_registry

            return get_chain_registry()

        result = benchmark(get_singleton)
        assert result is not None

    def test_chain_metadata_retrieval(self, benchmark):
        """Benchmark chain metadata retrieval.

        Expected: < 0.1ms per retrieval
        """
        registry = ChainRegistry()

        # Register chains with metadata
        for i in range(1000):
            registry.register(
                f"chain_{i}",
                create_mock_chain(i),
                vertical="test",
                description=f"Test chain {i}",
                tags=[f"tag_{j % 5}" for j in range(i % 5)],
            )

        def get_metadata():
            return registry.get_metadata("chain_500", vertical="test")

        result = benchmark(get_metadata)
        assert result is not None


# =============================================================================
# PersonaRegistry Performance Tests
# =============================================================================


class TestPersonaRegistryPerformance:
    """Performance benchmarks for PersonaRegistry operations.

    Targets:
    - Registration: < 1ms per persona
    - Lookup: < 0.1ms
    - Discovery (1000 personas): < 10ms
    - Tag filtering: < 5ms for 1000 items
    """

    def test_persona_registration_speed_10_items(self, benchmark):
        """Benchmark persona registration with 10 items.

        Expected: < 1ms per registration
        """
        registry = PersonaRegistry()

        def register_10_personas():
            for i in range(10):
                registry.register(
                    f"persona_{i}",
                    create_persona_spec(i),
                    vertical="test",
                )

        result = benchmark(register_10_personas)
        assert len(registry.list_personas()) == 10

    def test_persona_registration_speed_100_items(self, benchmark):
        """Benchmark persona registration with 100 items.

        Expected: < 1ms per registration
        """
        registry = PersonaRegistry()

        def register_100_personas():
            for i in range(100):
                registry.register(
                    f"persona_{i}",
                    create_persona_spec(i),
                    vertical="test",
                )

        result = benchmark(register_100_personas)
        assert len(registry.list_personas()) == 100

    def test_persona_registration_speed_1000_items(self, benchmark):
        """Benchmark persona registration with 1000 items.

        Expected: < 1ms per registration
        """
        registry = PersonaRegistry()

        def register_1000_personas():
            for i in range(1000):
                registry.register(
                    f"persona_{i}",
                    create_persona_spec(i),
                    vertical="test",
                )

        result = benchmark(register_1000_personas)
        assert len(registry.list_personas()) == 1000

    def test_persona_lookup_speed(self, benchmark):
        """Benchmark persona lookup by name.

        Expected: < 0.1ms per lookup
        """
        registry = PersonaRegistry()

        # Register 1000 personas
        for i in range(1000):
            registry.register(
                f"persona_{i}",
                create_persona_spec(i),
                vertical="test",
            )

        def lookup_persona():
            return registry.get("persona_500", vertical="test")

        result = benchmark(lookup_persona)
        assert result is not None

    def test_persona_discovery_by_expertise(self, benchmark):
        """Benchmark persona discovery by expertise.

        Expected: < 10ms for 1000 items
        """
        registry = PersonaRegistry()

        # Register personas with various expertise
        for i in range(1000):
            expertise = [f"skill_{j % 20}" for j in range(i % 20)]
            registry.register(
                f"persona_{i}",
                create_persona_spec(i, expertise),
                vertical="test",
            )

        def find_by_expertise():
            return registry.find_by_expertise("skill_10")

        result = benchmark(find_by_expertise)
        assert len(result) > 0

    def test_persona_discovery_by_role(self, benchmark):
        """Benchmark persona discovery by role.

        Expected: < 10ms for 1000 items
        """
        registry = PersonaRegistry()

        # Register personas with roles
        for i in range(1000):
            spec = create_persona_spec(i)
            spec.role = f"Developer_{i % 10}"
            registry.register(f"persona_{i}", spec, vertical="test")

        def find_by_role():
            return registry.find_by_role("Developer_5")

        result = benchmark(find_by_role)
        assert len(result) > 0

    def test_persona_tag_filtering(self, benchmark):
        """Benchmark persona tag filtering.

        Expected: < 5ms for 1000 items
        """
        registry = PersonaRegistry()

        # Register personas with tags
        for i in range(1000):
            spec = create_persona_spec(i)
            spec.tags = [f"tag_{j % 10}" for j in range(i % 10)]
            registry.register(f"persona_{i}", spec, vertical="test")

        def find_by_tag():
            return registry.find_by_tag("tag_5")

        result = benchmark(find_by_tag)
        assert len(result) > 0

    def test_persona_multi_tag_filtering(self, benchmark):
        """Benchmark persona multi-tag filtering.

        Expected: < 10ms for 1000 items
        """
        registry = PersonaRegistry()

        # Register personas with multiple tags
        for i in range(1000):
            spec = create_persona_spec(i)
            spec.tags = [f"tag_{j % 10}" for j in range(min(i + 1, 10))]
            registry.register(f"persona_{i}", spec, vertical="test")

        def find_by_tags():
            return registry.find_by_tags(["tag_2", "tag_5"], match_all=True)

        result = benchmark(find_by_tags)
        assert isinstance(result, list)

    def test_persona_singleton_overhead(self, benchmark):
        """Benchmark singleton pattern overhead.

        Expected: < 0.01ms per access
        """
        reset_persona_registry()

        def get_singleton():
            from victor.framework.persona_registry import get_persona_registry

            return get_persona_registry()

        result = benchmark(get_singleton)
        assert result is not None


# =============================================================================
# CapabilityProvider Performance Tests
# =============================================================================


class TestCapabilityProviderPerformance:
    """Performance benchmarks for CapabilityProvider operations.

    Targets:
    - Capability enumeration: < 1ms for 100 capabilities
    - Metadata retrieval: < 0.5ms for 100 capabilities
    - Apply method: < 0.1ms per capability
    - Provider instantiation: < 1ms
    """

    def test_capability_enumeration_speed(self, benchmark):
        """Benchmark capability enumeration.

        Expected: < 1ms for 100 capabilities
        """
        provider = TestCapabilityProvider(num_capabilities=100)

        def enumerate_capabilities():
            return provider.get_capabilities()

        result = benchmark(enumerate_capabilities)
        assert len(result) == 100

    def test_capability_metadata_retrieval_speed(self, benchmark):
        """Benchmark capability metadata retrieval.

        Expected: < 0.5ms for 100 capabilities
        """
        provider = TestCapabilityProvider(num_capabilities=100)

        def get_metadata():
            return provider.get_capability_metadata()

        result = benchmark(get_metadata)
        assert len(result) == 100

    def test_capability_apply_overhead(self, benchmark):
        """Benchmark capability apply method overhead.

        Expected: < 0.1ms per capability
        """
        provider = TestCapabilityProvider(num_capabilities=100)
        test_context = {"test_key": "test_value"}

        def apply_capability():
            caps = provider.get_capabilities()
            return caps["capability_50"].apply(test_context)

        result = benchmark(apply_capability)
        assert "applied_capability_50" in result

    def test_capability_provider_instantiation(self, benchmark):
        """Benchmark capability provider instantiation.

        Expected: < 1ms
        """

        def create_provider():
            return TestCapabilityProvider(num_capabilities=100)

        result = benchmark(create_provider)
        assert result is not None

    def test_capability_has_capability_check(self, benchmark):
        """Benchmark has_capability check.

        Expected: < 0.05ms per check
        """
        provider = TestCapabilityProvider(num_capabilities=100)

        def check_capability():
            return provider.has_capability("capability_50")

        result = benchmark(check_capability)
        assert result is True

    def test_capability_list_capabilities(self, benchmark):
        """Benchmark list_capabilities method.

        Expected: < 0.1ms for 100 capabilities
        """
        provider = TestCapabilityProvider(num_capabilities=100)

        def list_caps():
            return provider.list_capabilities()

        result = benchmark(list_caps)
        assert len(result) == 100


# =============================================================================
# Middleware Performance Tests
# =============================================================================


class TestMiddlewarePerformance:
    """Performance benchmarks for middleware operations.

    Targets:
    - Middleware execution: < 0.1ms per middleware
    - Priority sorting: < 1ms for 10 middleware
    - Tool filtering: < 0.05ms per filter
    """

    @staticmethod
    def _run_ready_coroutine(coro):
        """Drive coroutines that complete without awaiting external work."""
        try:
            coro.send(None)
        except StopIteration as exc:
            return exc.value
        raise AssertionError("Benchmark coroutine suspended; update the benchmark driver")

    def test_logging_middleware_overhead(self, benchmark):
        """Benchmark logging middleware execution overhead.

        Expected: < 0.1ms per call
        """
        middleware = LoggingMiddleware(log_level=0)  # Use high level to reduce I/O

        # Run sync wrapper for benchmark
        def run_sync():
            return self._run_ready_coroutine(
                middleware.before_tool_call("test_tool", {"arg1": "value1"})
            )

        result = benchmark(run_sync)
        assert result is not None

    def test_secret_masking_middleware_overhead(self, benchmark):
        """Benchmark secret masking middleware execution overhead.

        Expected: < 0.1ms per call
        """
        middleware = SecretMaskingMiddleware(replacement="[REDACTED]")

        def run_sync():
            return self._run_ready_coroutine(
                middleware.before_tool_call("test_tool", {"secret": "hidden_value"})
            )

        result = benchmark(run_sync)
        assert result is not None

    def test_metrics_middleware_overhead(self, benchmark):
        """Benchmark metrics middleware execution overhead.

        Expected: < 0.1ms per call
        """
        middleware = MetricsMiddleware(enable_timing=True)

        def run_sync():
            return self._run_ready_coroutine(
                middleware.before_tool_call("test_tool", {"arg1": "value1"})
            )

        result = benchmark(run_sync)
        assert result is not None

    def test_git_safety_middleware_overhead(self, benchmark):
        """Benchmark git safety middleware execution overhead.

        Expected: < 0.1ms per call
        """
        middleware = GitSafetyMiddleware(block_dangerous=True)

        def run_sync():
            return self._run_ready_coroutine(
                middleware.before_tool_call("git", {"command": "git status"})
            )

        result = benchmark(run_sync)
        assert result is not None

    def test_middleware_priority_sorting(self, benchmark):
        """Benchmark middleware priority sorting.

        Expected: < 1ms for 10 middleware
        """
        from victor.core.vertical_types import MiddlewarePriority
        from victor.framework.middleware import LoggingMiddleware, MetricsMiddleware

        middleware_list = [
            LoggingMiddleware(),
            MetricsMiddleware(),
            GitSafetyMiddleware(),
            SecretMaskingMiddleware(),
            LoggingMiddleware(log_level=10),
            MetricsMiddleware(enable_timing=False),
            GitSafetyMiddleware(block_dangerous=False),
            SecretMaskingMiddleware(mask_in_arguments=True),
            LoggingMiddleware(include_results=True),
            MetricsMiddleware(callback=lambda x, y: None),
        ]

        def sort_by_priority():
            return sorted(middleware_list, key=lambda m: m.get_priority().value, reverse=True)

        result = benchmark(sort_by_priority)
        assert len(result) == 10

    def test_middleware_tool_filtering(self, benchmark):
        """Benchmark middleware tool filtering.

        Expected: < 0.05ms per filter
        """
        middleware = GitSafetyMiddleware()

        tools = {"git", "bash", "read_file", "write_file", "execute_bash", "shell"}

        def filter_tools():
            applicable = middleware.get_applicable_tools()
            if applicable is None:
                return tools
            return tools & applicable

        result = benchmark(filter_tools)
        assert len(result) > 0

    def test_middleware_execution_chain(self, benchmark):
        """Benchmark executing multiple middleware in sequence.

        Expected: < 0.5ms for 5 middleware
        """
        middleware_list = [
            LoggingMiddleware(log_level=0),
            SecretMaskingMiddleware(),
            MetricsMiddleware(enable_timing=False),
            GitSafetyMiddleware(),
        ]

        async def execute_chain():
            for mw in middleware_list:
                result = await mw.before_tool_call("test_tool", {"arg1": "value1"})
                if not result.proceed:
                    break
            return True

        def run_sync():
            return self._run_ready_coroutine(execute_chain())

        result = benchmark(run_sync)
        assert result is True


# =============================================================================
# Integration Performance Tests
# =============================================================================


class TestIntegrationPerformance:
    """Integration performance benchmarks for combined operations.

    Targets:
    - Register and lookup: < 1.1ms total (1ms register + 0.1ms lookup)
    - Register and discover: < 11ms total for 1000 items
    - Multi-registry operations: < 20ms for all registries
    """

    def test_register_and_lookup_chain(self, benchmark):
        """Benchmark register followed by lookup.

        Expected: < 1.1ms total
        """
        registry = ChainRegistry()

        def register_and_lookup():
            registry.register("test_chain", create_mock_chain(0), vertical="test")
            return registry.get("test_chain", vertical="test")

        result = benchmark(register_and_lookup)
        assert result is not None

    def test_register_and_lookup_persona(self, benchmark):
        """Benchmark register followed by lookup.

        Expected: < 1.1ms total
        """
        registry = PersonaRegistry()

        def register_and_lookup():
            registry.register("test_persona", create_persona_spec(0), vertical="test")
            return registry.get("test_persona", vertical="test")

        result = benchmark(register_and_lookup)
        assert result is not None

    def test_multi_registry_operations(self, benchmark):
        """Benchmark operations across multiple registries.

        Expected: < 20ms for all operations
        """
        chain_registry = ChainRegistry()
        persona_registry = PersonaRegistry()
        capability_provider = TestCapabilityProvider(num_capabilities=50)

        def multi_registry_ops():
            # Register chains
            for i in range(50):
                chain_registry.register(
                    f"chain_{i}",
                    create_mock_chain(i),
                    vertical="test",
                )

            # Register personas
            for i in range(50):
                persona_registry.register(
                    f"persona_{i}",
                    create_persona_spec(i),
                    vertical="test",
                )

            # Get capabilities
            caps = capability_provider.get_capabilities()

            # Lookup operations
            chain = chain_registry.get("chain_25", vertical="test")
            persona = persona_registry.get("persona_25", vertical="test")
            capability = caps.get("capability_25")

            return chain is not None and persona is not None and capability is not None

        result = benchmark(multi_registry_ops)
        assert result is True

    def test_bulk_registration_performance(self, benchmark):
        """Benchmark bulk registration from vertical.

        Expected: < 10ms for 100 items
        """
        registry = ChainRegistry()

        chains = {f"chain_{i}": create_mock_chain(i) for i in range(100)}

        def bulk_register():
            return registry.register_from_vertical("test_vertical", chains, replace=True)

        result = benchmark(bulk_register)
        assert result == 100

    def test_registry_clear_performance(self, benchmark):
        """Benchmark registry clearing operation.

        Expected: < 1ms for 1000 items
        """
        registry = ChainRegistry()

        # Setup: Register 1000 chains
        for i in range(1000):
            registry.register(
                f"chain_{i}",
                create_mock_chain(i),
                vertical="test",
            )

        def clear_registry():
            registry.clear()

        benchmark(clear_registry)
        assert len(registry.list_chains()) == 0


# =============================================================================
# Performance Assertions Summary
# =============================================================================


class TestPerformanceAssertions:
    """Summary tests with explicit performance assertions.

    These tests measure absolute time and assert that operations
    meet their performance targets.
    """

    def test_chain_registration_meets_target(self):
        """Assert chain registration is < 1ms per item."""
        registry = ChainRegistry()
        num_items = 100

        start = time.perf_counter()
        for i in range(num_items):
            registry.register(
                f"chain_{i}",
                create_mock_chain(i),
                vertical="test",
                description=f"Test chain {i}",
            )
        elapsed = time.perf_counter() - start

        avg_time_ms = (elapsed / num_items) * 1000
        assert (
            avg_time_ms < 1.0
        ), f"Chain registration too slow: {avg_time_ms:.3f}ms per item (target: < 1ms)"

    def test_chain_lookup_meets_target(self):
        """Assert chain lookup is < 0.1ms."""
        registry = ChainRegistry()
        registry.register("test_chain", create_mock_chain(0), vertical="test")

        # Warm-up
        for _ in range(100):
            registry.get("test_chain", vertical="test")

        # Measure
        start = time.perf_counter()
        for _ in range(1000):
            registry.get("test_chain", vertical="test")
        elapsed = time.perf_counter() - start

        avg_time_ms = (elapsed / 1000) * 1000
        assert avg_time_ms < 0.1, f"Chain lookup too slow: {avg_time_ms:.3f}ms (target: < 0.1ms)"

    def test_persona_registration_meets_target(self):
        """Assert persona registration is < 1ms per item."""
        registry = PersonaRegistry()
        num_items = 100

        start = time.perf_counter()
        for i in range(num_items):
            registry.register(
                f"persona_{i}",
                create_persona_spec(i),
                vertical="test",
            )
        elapsed = time.perf_counter() - start

        avg_time_ms = (elapsed / num_items) * 1000
        assert (
            avg_time_ms < 1.0
        ), f"Persona registration too slow: {avg_time_ms:.3f}ms per item (target: < 1ms)"

    def test_capability_enumeration_meets_target(self):
        """Assert capability enumeration is < 1ms for 100 items."""
        provider = TestCapabilityProvider(num_capabilities=100)

        # Warm-up
        for _ in range(10):
            provider.get_capabilities()

        # Measure
        start = time.perf_counter()
        for _ in range(100):
            provider.get_capabilities()
        elapsed = time.perf_counter() - start

        avg_time_ms = (elapsed / 100) * 1000
        assert (
            avg_time_ms < 1.0
        ), f"Capability enumeration too slow: {avg_time_ms:.3f}ms (target: < 1ms)"

    def test_middleware_execution_meets_target(self):
        """Assert middleware execution is < 0.1ms per call."""
        import asyncio

        middleware = LoggingMiddleware(log_level=0)

        async def measure_middleware():
            # Warm-up
            for _ in range(10):
                await middleware.before_tool_call("test_tool", {"arg1": "value1"})

            # Measure
            start = time.perf_counter()
            for _ in range(1000):
                await middleware.before_tool_call("test_tool", {"arg1": "value1"})
            elapsed = time.perf_counter() - start

            avg_time_ms = (elapsed / 1000) * 1000
            assert (
                avg_time_ms < 0.1
            ), f"Middleware execution too slow: {avg_time_ms:.3f}ms (target: < 0.1ms)"

        asyncio.run(measure_middleware())

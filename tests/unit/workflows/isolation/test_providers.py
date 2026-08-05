# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0
"""Tests for SandboxProviderRegistry and sandbox providers."""

from typing import Any, Dict, Optional

import pytest

from victor.workflows.isolation import (
    SandboxResult,
    SandboxProvider,
    SandboxProviderRegistry,
    NoneSandboxProvider,
    ProcessSandboxProvider,
    DockerSandboxProvider,
    IsolationConfig,
    register_builtin_providers,
)


class TestSandboxResult:
    """Test SandboxResult dataclass."""

    def test_success_result(self):
        """Should create successful result."""
        result = SandboxResult(
            success=True,
            output="Hello, World!",
            exit_code=0,
        )
        assert result.success is True
        assert result.output == "Hello, World!"
        assert result.error is None
        assert result.exit_code == 0
        assert result.metadata == {}

    def test_failure_result(self):
        """Should create failure result."""
        result = SandboxResult(
            success=False,
            error="Command failed",
            exit_code=1,
        )
        assert result.success is False
        assert result.output is None
        assert result.error == "Command failed"
        assert result.exit_code == 1

    def test_with_metadata(self):
        """Should store metadata."""
        result = SandboxResult(
            success=True,
            output="result",
            metadata={"duration": 1.5, "memory_mb": 256},
        )
        assert result.metadata["duration"] == 1.5
        assert result.metadata["memory_mb"] == 256


class TestSandboxProvider:
    """Test SandboxProvider abstract base class."""

    def test_is_abstract(self):
        """Should not be instantiable directly."""
        with pytest.raises(TypeError):
            SandboxProvider()

    def test_supports_feature_default(self):
        """Default supports_feature returns False."""
        # Use a concrete implementation to test default method
        provider = NoneSandboxProvider()
        assert provider.supports_feature("unknown_feature") is False

    def test_get_capabilities(self):
        """Should return capabilities dict."""
        provider = NoneSandboxProvider()
        caps = provider.get_capabilities()
        assert "networking" in caps
        assert "filesystem" in caps
        assert "gpu" in caps
        assert "secrets" in caps


class TestNoneSandboxProvider:
    """Test NoneSandboxProvider (direct execution)."""

    def test_sandbox_type(self):
        """Should return 'none' sandbox type."""
        provider = NoneSandboxProvider()
        assert provider.sandbox_type == "none"

    @pytest.mark.asyncio
    async def test_execute(self):
        """Should execute and return result."""
        provider = NoneSandboxProvider()
        config = IsolationConfig(sandbox_type="none")
        result = await provider.execute(
            code="print('hello')",
            context={"key": "value"},
            config=config,
            timeout=10.0,
        )
        assert result.success is True
        assert result.metadata["sandbox_type"] == "none"
        assert result.metadata["executed_inline"] is True

    @pytest.mark.asyncio
    async def test_cleanup(self):
        """Cleanup should be no-op."""
        provider = NoneSandboxProvider()
        await provider.cleanup()  # Should not raise

    def test_supports_networking(self):
        """Should support networking."""
        provider = NoneSandboxProvider()
        assert provider.supports_feature("networking") is True

    def test_supports_filesystem(self):
        """Should support filesystem."""
        provider = NoneSandboxProvider()
        assert provider.supports_feature("filesystem") is True

    def test_no_gpu_support(self):
        """Should not support GPU by default."""
        provider = NoneSandboxProvider()
        assert provider.supports_feature("gpu") is False


class TestProcessSandboxProvider:
    """Test ProcessSandboxProvider."""

    def test_sandbox_type(self):
        """Should return 'process' sandbox type."""
        provider = ProcessSandboxProvider()
        assert provider.sandbox_type == "process"

    @pytest.mark.asyncio
    async def test_execute(self):
        """Should execute in subprocess."""
        provider = ProcessSandboxProvider()
        config = IsolationConfig(sandbox_type="process")
        result = await provider.execute(
            code="print('hello')",
            context={},
            config=config,
        )
        assert result.success is True
        assert result.metadata["sandbox_type"] == "process"

    @pytest.mark.asyncio
    async def test_cleanup(self):
        """Cleanup should be no-op (process cleanup is automatic)."""
        provider = ProcessSandboxProvider()
        await provider.cleanup()  # Should not raise

    def test_supports_networking(self):
        """Should support networking."""
        provider = ProcessSandboxProvider()
        assert provider.supports_feature("networking") is True


class TestDockerSandboxProvider:
    """Test DockerSandboxProvider."""

    def test_sandbox_type(self):
        """Should return 'docker' sandbox type."""
        provider = DockerSandboxProvider()
        assert provider.sandbox_type == "docker"

    @pytest.mark.asyncio
    async def test_execute(self):
        """Should execute in Docker container."""
        provider = DockerSandboxProvider()
        config = IsolationConfig(
            sandbox_type="docker",
            docker_image="python:3.11-slim",
        )
        result = await provider.execute(
            code="print('hello')",
            context={},
            config=config,
        )
        assert result.success is True
        assert result.metadata["sandbox_type"] == "docker"
        assert result.metadata["image"] == "python:3.11-slim"

    @pytest.mark.asyncio
    async def test_cleanup(self):
        """Cleanup should be no-op (container cleanup by executor)."""
        provider = DockerSandboxProvider()
        await provider.cleanup()  # Should not raise

    def test_supports_all_features(self):
        """Docker should support most features."""
        provider = DockerSandboxProvider()
        assert provider.supports_feature("networking") is True
        assert provider.supports_feature("filesystem") is True
        assert provider.supports_feature("gpu") is True
        assert provider.supports_feature("secrets") is True


class TestSandboxProviderRegistry:
    """Test SandboxProviderRegistry singleton."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset registry before each test."""
        SandboxProviderRegistry.reset_instance()
        yield
        SandboxProviderRegistry.reset_instance()

    def test_singleton_pattern(self):
        """Should return same instance."""
        instance1 = SandboxProviderRegistry.get_instance()
        instance2 = SandboxProviderRegistry.get_instance()
        assert instance1 is instance2

    def test_reset_instance(self):
        """Should create new instance after reset."""
        instance1 = SandboxProviderRegistry.get_instance()
        SandboxProviderRegistry.reset_instance()
        instance2 = SandboxProviderRegistry.get_instance()
        assert instance1 is not instance2

    def test_register_provider(self):
        """Should register provider class."""
        registry = SandboxProviderRegistry.get_instance()
        registry.register("none", NoneSandboxProvider)
        assert registry.is_registered("none") is True

    def test_get_provider_creates_instance(self):
        """Should create provider instance on first get."""
        registry = SandboxProviderRegistry.get_instance()
        registry.register("none", NoneSandboxProvider)
        provider = registry.get_provider("none")
        assert provider is not None
        assert isinstance(provider, NoneSandboxProvider)

    def test_get_provider_caches_instance(self):
        """Should return same instance on subsequent calls."""
        registry = SandboxProviderRegistry.get_instance()
        registry.register("none", NoneSandboxProvider)
        provider1 = registry.get_provider("none")
        provider2 = registry.get_provider("none")
        assert provider1 is provider2

    def test_get_provider_unknown(self):
        """Should return None for unknown type."""
        registry = SandboxProviderRegistry.get_instance()
        provider = registry.get_provider("unknown")
        assert provider is None

    def test_list_types(self):
        """Should list registered types."""
        registry = SandboxProviderRegistry.get_instance()
        registry.register("none", NoneSandboxProvider)
        registry.register("process", ProcessSandboxProvider)
        types = registry.list_types()
        assert "none" in types
        assert "process" in types

    def test_is_registered(self):
        """Should check registration status."""
        registry = SandboxProviderRegistry.get_instance()
        assert registry.is_registered("none") is False
        registry.register("none", NoneSandboxProvider)
        assert registry.is_registered("none") is True

    def test_get_all_capabilities(self):
        """Should get capabilities for all providers."""
        registry = SandboxProviderRegistry.get_instance()
        registry.register("none", NoneSandboxProvider)
        registry.register("docker", DockerSandboxProvider)
        caps = registry.get_all_capabilities()
        assert "none" in caps
        assert "docker" in caps
        assert caps["none"]["networking"] is True
        assert caps["docker"]["gpu"] is True

    def test_re_register_clears_cache(self):
        """Re-registering should clear cached instance."""
        registry = SandboxProviderRegistry.get_instance()
        registry.register("none", NoneSandboxProvider)
        provider1 = registry.get_provider("none")
        registry.register("none", NoneSandboxProvider)
        provider2 = registry.get_provider("none")
        assert provider1 is not provider2


class TestCustomSandboxProvider:
    """Test custom sandbox provider registration."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset registry before each test."""
        SandboxProviderRegistry.reset_instance()
        yield
        SandboxProviderRegistry.reset_instance()

    def test_custom_provider(self):
        """Should support custom providers."""

        class KubernetesSandboxProvider(SandboxProvider):
            """Custom Kubernetes provider."""

            @property
            def sandbox_type(self) -> str:
                return "kubernetes"

            async def execute(
                self,
                code: str,
                context: Dict[str, Any],
                config: IsolationConfig,
                timeout: Optional[float] = None,
            ) -> SandboxResult:
                return SandboxResult(
                    success=True,
                    output="executed in pod",
                    metadata={"sandbox_type": "kubernetes"},
                )

            async def cleanup(self) -> None:
                pass

            def supports_feature(self, feature: str) -> bool:
                return feature in ("networking", "gpu", "secrets")

        registry = SandboxProviderRegistry.get_instance()
        registry.register("kubernetes", KubernetesSandboxProvider)

        provider = registry.get_provider("kubernetes")
        assert provider is not None
        assert provider.sandbox_type == "kubernetes"
        assert provider.supports_feature("gpu") is True

    @pytest.mark.asyncio
    async def test_custom_provider_execute(self):
        """Custom provider should execute correctly."""

        class WasmSandboxProvider(SandboxProvider):
            @property
            def sandbox_type(self) -> str:
                return "wasm"

            async def execute(
                self,
                code: str,
                context: Dict[str, Any],
                config: IsolationConfig,
                timeout: Optional[float] = None,
            ) -> SandboxResult:
                return SandboxResult(
                    success=True,
                    output=f"WASM: {code}",
                    metadata={"sandbox_type": "wasm", "runtime": "wasmtime"},
                )

            async def cleanup(self) -> None:
                pass

        registry = SandboxProviderRegistry.get_instance()
        registry.register("wasm", WasmSandboxProvider)

        provider = registry.get_provider("wasm")
        config = IsolationConfig(sandbox_type="none")  # Type doesn't matter for test
        result = await provider.execute(
            code="test_code",
            context={},
            config=config,
        )

        assert result.success is True
        assert "WASM: test_code" in result.output
        assert result.metadata["runtime"] == "wasmtime"


class TestRegisterBuiltinProviders:
    """Test register_builtin_providers function."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset registry before each test."""
        SandboxProviderRegistry.reset_instance()
        yield
        SandboxProviderRegistry.reset_instance()

    def test_registers_all_builtins(self):
        """Should register all built-in providers."""
        register_builtin_providers()
        registry = SandboxProviderRegistry.get_instance()

        assert registry.is_registered("none") is True
        assert registry.is_registered("process") is True
        assert registry.is_registered("docker") is True

    def test_providers_are_correct_types(self):
        """Registered providers should be correct types."""
        register_builtin_providers()
        registry = SandboxProviderRegistry.get_instance()

        assert isinstance(registry.get_provider("none"), NoneSandboxProvider)
        assert isinstance(registry.get_provider("process"), ProcessSandboxProvider)
        assert isinstance(registry.get_provider("docker"), DockerSandboxProvider)

    def test_idempotent(self):
        """Should be safe to call multiple times."""
        register_builtin_providers()
        register_builtin_providers()
        registry = SandboxProviderRegistry.get_instance()
        assert len(registry.list_types()) == 3

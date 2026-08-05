# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0

"""TDD hardening tests for ServiceContainer.

Tests for 6 identified gaps:
1. dispose() TOCTOU race
2. _resolving thread-safety for transient services
3. freeze() after bootstrap
4. validate() for early cycle detection
5. @service decorator dependency injection
6. Async disposal support
"""

import threading
import time

import pytest

from victor.core.container import (
    ServiceContainer,
    ServiceLifetime,
    ServiceNotFoundError,
    ServiceResolutionError,
)

# =========================================================================
# Helper classes
# =========================================================================


class DisposableCounter:
    """Tracks how many times dispose() is called."""

    def __init__(self):
        self.dispose_count = 0
        self._lock = threading.Lock()

    def dispose(self) -> None:
        with self._lock:
            self.dispose_count += 1


class DisposableRaiseOnDouble:
    """Raises on second dispose() call."""

    def __init__(self):
        self._disposed = False

    def dispose(self) -> None:
        if self._disposed:
            raise RuntimeError("Already disposed!")
        self._disposed = True


# =========================================================================
# Step 1: Gap 2 — dispose() TOCTOU Race
# =========================================================================


class TestDisposeRace:
    """Verify dispose() is atomic — no double-disposal under concurrency."""

    def test_concurrent_dispose_services_disposed_exactly_once(self):
        """10 threads call dispose() simultaneously. Service disposed once."""
        container = ServiceContainer()
        counter = DisposableCounter()
        container.register_instance(DisposableCounter, counter)

        barrier = threading.Barrier(10)
        errors = []

        def dispose_thread():
            try:
                barrier.wait(timeout=5)
                container.dispose()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=dispose_thread) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Unexpected errors: {errors}"
        assert counter.dispose_count == 1

    def test_concurrent_dispose_no_exceptions(self):
        """Service that raises on double-dispose should not surface errors."""
        container = ServiceContainer()
        svc = DisposableRaiseOnDouble()
        container.register_instance(DisposableRaiseOnDouble, svc)

        barrier = threading.Barrier(10)
        errors = []

        def dispose_thread():
            try:
                barrier.wait(timeout=5)
                container.dispose()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=dispose_thread) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        # No errors should propagate — dispose should be atomic
        assert not errors

    def test_dispose_idempotent_single_thread(self):
        """Calling dispose() multiple times is safe, service disposed once."""
        container = ServiceContainer()
        counter = DisposableCounter()
        container.register_instance(DisposableCounter, counter)

        container.dispose()
        container.dispose()
        container.dispose()

        assert counter.dispose_count == 1


# =========================================================================
# Step 2: Gap 1 — _resolving Thread-Safety for Transient
# =========================================================================


class TestResolvingThreadSafety:
    """Verify _resolving is per-thread, no false circular errors."""

    def test_concurrent_transient_no_false_circular_error(self):
        """Two threads resolve different transients sharing a dep."""
        container = ServiceContainer()

        class SharedDep:
            pass

        class ServiceA:
            def __init__(self, dep):
                self.dep = dep

        class ServiceB:
            def __init__(self, dep):
                self.dep = dep

        container.register(SharedDep, lambda c: SharedDep(), ServiceLifetime.TRANSIENT)
        container.register(
            ServiceA,
            lambda c: ServiceA(c.get(SharedDep)),
            ServiceLifetime.TRANSIENT,
        )
        container.register(
            ServiceB,
            lambda c: ServiceB(c.get(SharedDep)),
            ServiceLifetime.TRANSIENT,
        )

        errors = []
        barrier = threading.Barrier(2)

        def resolve(svc_type):
            barrier.wait(timeout=5)
            try:
                container.get(svc_type)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=resolve, args=(ServiceA,))
        t2 = threading.Thread(target=resolve, args=(ServiceB,))
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert not errors, f"False circular errors: {errors}"

    def test_concurrent_transient_with_slow_factory(self):
        """Slow factory doesn't block other threads' resolution."""
        container = ServiceContainer()

        class SlowService:
            pass

        class FastService:
            pass

        def slow_factory(c):
            time.sleep(0.05)
            return SlowService()

        container.register(SlowService, slow_factory, ServiceLifetime.TRANSIENT)
        container.register(
            FastService,
            lambda c: FastService(),
            ServiceLifetime.TRANSIENT,
        )

        results = []
        barrier = threading.Barrier(5)

        def resolve(svc_type):
            barrier.wait(timeout=5)
            try:
                results.append(container.get(svc_type))
            except Exception as e:
                results.append(e)

        threads = []
        for i in range(5):
            svc = SlowService if i % 2 == 0 else FastService
            t = threading.Thread(target=resolve, args=(svc,))
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        errors = [r for r in results if isinstance(r, Exception)]
        assert not errors, f"Errors: {errors}"
        assert len(results) == 5

    def test_real_circular_dependency_still_detected(self):
        """Genuine A->B->A cycle is still caught."""
        container = ServiceContainer()

        class A:
            pass

        class B:
            pass

        container.register(A, lambda c: c.get(B), ServiceLifetime.TRANSIENT)
        container.register(B, lambda c: c.get(A), ServiceLifetime.TRANSIENT)

        with pytest.raises(ServiceResolutionError, match="Circular"):
            container.get(A)

    def test_thread_local_resolving_isolation(self):
        """Thread 1's slow resolve doesn't affect thread 2."""
        container = ServiceContainer()

        class SameType:
            pass

        call_count = {"value": 0}
        lock = threading.Lock()

        def counting_factory(c):
            with lock:
                call_count["value"] += 1
            time.sleep(0.03)
            return SameType()

        container.register(SameType, counting_factory, ServiceLifetime.TRANSIENT)

        errors = []
        barrier = threading.Barrier(2)

        def resolve():
            barrier.wait(timeout=5)
            try:
                container.get(SameType)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=resolve)
        t2 = threading.Thread(target=resolve)
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert not errors, f"Errors: {errors}"
        assert call_count["value"] == 2


# =========================================================================
# Step 3: Gap 5 — freeze() After Bootstrap
# =========================================================================


class TestContainerFreeze:
    """Verify container can be frozen to prevent stray registrations."""

    def test_freeze_blocks_register(self):
        from victor.core.container import ContainerFrozenError

        container = ServiceContainer()
        container.freeze()
        with pytest.raises(ContainerFrozenError):
            container.register(str, lambda c: "x")

    def test_freeze_blocks_register_instance(self):
        from victor.core.container import ContainerFrozenError

        container = ServiceContainer()
        container.freeze()
        with pytest.raises(ContainerFrozenError):
            container.register_instance(str, "x")

    def test_freeze_blocks_register_or_replace(self):
        from victor.core.container import ContainerFrozenError

        container = ServiceContainer()
        container.freeze()
        with pytest.raises(ContainerFrozenError):
            container.register_or_replace(str, lambda c: "x")

    def test_freeze_allows_get(self):
        container = ServiceContainer()
        container.register(str, lambda c: "hello")
        container.freeze()
        assert container.get(str) == "hello"

    def test_freeze_allows_create_scope(self):
        container = ServiceContainer()
        container.register(str, lambda c: "scoped", ServiceLifetime.SCOPED)
        container.freeze()
        with container.create_scope() as scope:
            assert scope.get(str) == "scoped"

    def test_freeze_is_idempotent(self):
        container = ServiceContainer()
        container.freeze()
        container.freeze()  # should not raise

    def test_reset_container_clears_frozen_state(self):
        from victor.core.container import (
            get_container,
            reset_container,
            set_container,
        )

        c = ServiceContainer()
        c.register(int, lambda c: 42)
        c.freeze()
        set_container(c)

        reset_container()

        # New global container should be unfrozen
        fresh = get_container()
        fresh.register(int, lambda c: 99)  # should NOT raise
        assert fresh.get(int) == 99

    def test_freeze_blocks_decorator_registration(self):
        from victor.core.container import ContainerFrozenError

        container = ServiceContainer()
        container.freeze()

        with pytest.raises(ContainerFrozenError):

            @container.service(str)
            class MyStr:
                pass


# =========================================================================
# Step 4: Gap 6 — validate() for Early Cycle Detection
# =========================================================================


class TestContainerValidate:
    """Verify validate() catches problems before resolve-time."""

    def test_validate_no_errors_on_healthy_container(self):
        container = ServiceContainer()
        container.register(str, lambda c: "ok")
        container.register(int, lambda c: 42)
        errors = container.validate()
        assert errors == []

    def test_validate_detects_circular_dependency(self):
        container = ServiceContainer()

        class A:
            pass

        class B:
            pass

        container.register(A, lambda c: c.get(B), ServiceLifetime.TRANSIENT)
        container.register(B, lambda c: c.get(A), ServiceLifetime.TRANSIENT)

        errors = container.validate()
        assert len(errors) >= 1
        assert any("Circular" in str(e) for e in errors)

    def test_validate_detects_missing_dependency(self):
        container = ServiceContainer()

        class Needs:
            pass

        class Missing:
            pass

        container.register(Needs, lambda c: c.get(Missing), ServiceLifetime.TRANSIENT)

        errors = container.validate()
        assert len(errors) >= 1
        assert any(isinstance(e, ServiceNotFoundError) for e in errors)

    def test_validate_reports_multiple_errors(self):
        container = ServiceContainer()

        class Bad1:
            pass

        class Bad2:
            pass

        class Missing1:
            pass

        class Missing2:
            pass

        container.register(Bad1, lambda c: c.get(Missing1), ServiceLifetime.TRANSIENT)
        container.register(Bad2, lambda c: c.get(Missing2), ServiceLifetime.TRANSIENT)

        errors = container.validate()
        assert len(errors) == 2

    def test_validate_does_not_cache_singletons(self):
        container = ServiceContainer()
        call_count = {"n": 0}

        class Svc:
            pass

        def factory(c):
            call_count["n"] += 1
            return Svc()

        container.register(Svc, factory, ServiceLifetime.SINGLETON)

        container.validate()
        # Validation should NOT have cached the singleton
        assert call_count["n"] == 1  # validate called factory once

        instance = container.get(Svc)
        # get() should call factory again (not cached from validate)
        # Actually, validate uses _try_resolve which doesn't cache,
        # but the factory internally calls c.get() which does cache.
        # For a top-level singleton, validate's _try_resolve doesn't
        # store the result, so get() creates a new one.
        assert call_count["n"] == 2
        assert isinstance(instance, Svc)

    def test_validate_with_transient_services(self):
        container = ServiceContainer()

        class A:
            pass

        class B:
            pass

        container.register(A, lambda c: c.get(B), ServiceLifetime.TRANSIENT)
        container.register(B, lambda c: c.get(A), ServiceLifetime.TRANSIENT)

        errors = container.validate()
        assert len(errors) >= 1


# =========================================================================
# Step 5: Gap 4 — @service Decorator DI
# =========================================================================


class ILogger:
    """Test interface for logger."""

    pass


class MockLogger(ILogger):
    """Concrete logger for tests."""

    def __init__(self):
        self.name = "mock"


class TestDecoratorDependencyInjection:
    """Verify @service injects annotated constructor params."""

    def test_decorator_service_with_dependency(self):
        container = ServiceContainer()
        container.register(ILogger, lambda c: MockLogger())

        @container.service(str)
        class Greeter:
            def __init__(self, logger: ILogger):
                self.logger = logger

        svc = container.get(str)
        assert isinstance(svc.logger, MockLogger)

    def test_decorator_service_with_no_args_still_works(self):
        container = ServiceContainer()

        @container.service(int)
        class Simple:
            def __init__(self):
                self.value = 42

        assert container.get(int).value == 42

    def test_decorator_service_with_default_args(self):
        container = ServiceContainer()

        @container.service(str)
        class WithDefault:
            def __init__(self, name: str = "default"):
                self.name = name

        # str not registered as a service, so default should be used
        # But str IS a built-in, and we'd need it registered.
        # The decorator should fall back to default when resolution
        # fails and a default exists.
        svc = container.get(str)
        assert svc.name == "default"

    def test_decorator_service_with_mixed_args(self):
        container = ServiceContainer()
        container.register(ILogger, lambda c: MockLogger())

        @container.service(float)
        class Mixed:
            def __init__(self, logger: ILogger, name: str = "default"):
                self.logger = logger
                self.name = name

        svc = container.get(float)
        assert isinstance(svc.logger, MockLogger)
        assert svc.name == "default"

    def test_decorator_unresolvable_required_arg_raises(self):
        container = ServiceContainer()

        class IMissing:
            pass

        @container.service(str)
        class NeedsMissing:
            def __init__(self, dep: IMissing):
                self.dep = dep

        with pytest.raises(ServiceNotFoundError):
            container.get(str)


# =========================================================================
# Step 6: Gap 3 — Async Disposal
# =========================================================================


class AsyncDisposableService:
    """Mock async-disposable service."""

    def __init__(self):
        self.disposed = False

    async def adispose(self) -> None:
        self.disposed = True


class TestAsyncDisposal:
    """Verify async disposal support."""

    async def test_adispose_awaits_async_disposable(self):
        from victor.core.container import AsyncDisposable

        container = ServiceContainer()
        svc = AsyncDisposableService()
        container.register_instance(AsyncDisposableService, svc)

        assert isinstance(svc, AsyncDisposable)
        await container.adispose()
        assert svc.disposed

    async def test_adispose_also_disposes_sync_services(self):
        container = ServiceContainer()
        sync_svc = DisposableCounter()
        async_svc = AsyncDisposableService()

        container.register_instance(DisposableCounter, sync_svc)
        container.register_instance(AsyncDisposableService, async_svc)

        await container.adispose()
        assert sync_svc.dispose_count == 1
        assert async_svc.disposed

    async def test_async_context_manager(self):
        svc = AsyncDisposableService()
        async with ServiceContainer() as container:
            container.register_instance(AsyncDisposableService, svc)
        assert svc.disposed

    async def test_scope_adispose(self):
        container = ServiceContainer()
        container.register(
            AsyncDisposableService,
            lambda c: AsyncDisposableService(),
            ServiceLifetime.SCOPED,
        )

        scope = container.create_scope()
        svc = scope.get(AsyncDisposableService)
        assert not svc.disposed

        await scope.adispose()
        assert svc.disposed

    async def test_scope_async_context_manager(self):
        container = ServiceContainer()
        container.register(
            AsyncDisposableService,
            lambda c: AsyncDisposableService(),
            ServiceLifetime.SCOPED,
        )

        async with container.create_scope() as scope:
            svc = scope.get(AsyncDisposableService)
        assert svc.disposed

    async def test_sync_dispose_still_works(self):
        """Regression: sync dispose still handles sync services."""
        container = ServiceContainer()
        counter = DisposableCounter()
        container.register_instance(DisposableCounter, counter)
        container.dispose()
        assert counter.dispose_count == 1

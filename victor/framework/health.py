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

"""Framework-layer facade for health check system.

This module provides a unified access point to health check components from
the framework layer. It re-exports existing implementations from core modules
without duplicating code, following the Facade Pattern.

Delegated modules:
- victor.core.health: Comprehensive health check framework with composite pattern
- victor.providers.health: Provider-specific health monitoring

Design Pattern: Facade
- Single import point for framework users
- No code duplication - pure re-exports
- Maintains backward compatibility with original modules
- Enables discovery of health capabilities through framework namespace

Example:
    from victor.framework import (
        # Core Health
        HealthChecker,
        HealthStatus,
        ComponentHealth,
        HealthReport,
        create_default_health_checker,

        # Built-in checks
        ProviderHealthCheck,
        ToolHealthCheck,
        CacheHealthCheck,
        MemoryHealthCheck,

        # Provider-specific
        ProviderHealthChecker,
    )

    # Create health checker with multiple checks
    checker = HealthChecker()
    checker.add_check(ProviderHealthCheck("anthropic", provider))
    checker.add_check(ToolHealthCheck("filesystem", read))
    checker.add_check(CacheHealthCheck("redis", cache))
    checker.add_check(MemoryHealthCheck(threshold_mb=1024))

    # Get health report
    report = await checker.check_health()
    print(f"Status: {report.status}")
    for component in report.components:
        print(f"  {component.name}: {component.status}")

    # For Kubernetes probes
    is_ready = await checker.is_ready()
    is_alive = await checker.is_alive()
"""

from __future__ import annotations

# =============================================================================
# Core Health Check System
# From: victor/core/health.py
# =============================================================================
from victor.core.health import (
    BaseHealthCheck,
    CacheHealthCheck,
    CallableHealthCheck,
    ComponentHealth,
    HealthCheckProtocol,
    HealthChecker,
    HealthReport,
    HealthStatus,
    MemoryHealthCheck,
    ProviderHealthCheck,
    ToolHealthCheck,
    create_default_health_checker,
)

# =============================================================================
# Provider-Specific Health Monitoring
# From: victor/providers/health.py
# =============================================================================
from victor.providers.health import (
    ProviderHealthChecker,
    ProviderHealthResult,
    check_provider_health,
)

# Backward-compatible alias
HealthCheckResult = ProviderHealthResult

__all__ = [
    # Core Health Check System
    "BaseHealthCheck",
    "CacheHealthCheck",
    "CallableHealthCheck",
    "ComponentHealth",
    "HealthCheckProtocol",
    "HealthChecker",
    "HealthReport",
    "HealthStatus",
    "MemoryHealthCheck",
    "ProviderHealthCheck",
    "ToolHealthCheck",
    "create_default_health_checker",
    # Provider-Specific Health
    "HealthCheckResult",
    "ProviderHealthChecker",
    "ProviderHealthResult",
    "check_provider_health",
]

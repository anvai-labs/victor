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

"""Victor configuration system.

This module provides a streamlined configuration system for Victor that unifies
provider accounts, authentication, and settings.

Key Components:
- ProviderAccount: Unified account model
- AccountManager: Account management and resolution
- ConfigMigrator: Migration from old configuration format
- ConnectionValidator: Test provider connections

Example:
    from victor.config import get_account_manager

    manager = get_account_manager()
    account = manager.get_account("default")
    config = manager.resolve_provider_config(account)
"""

from importlib import import_module
from importlib.util import find_spec

# New unified configuration system
from victor.config.accounts import (
    ProviderAccount,
    AuthConfig,
    AccountManager,
    VictorConfig,
    ConfigDefaults,
    get_account_manager,
    reset_account_manager,
)

from victor.config.resolution import (
    ProviderResolver,
    ResolutionContext,
    get_provider_resolver,
    resolve_provider_config,
    reset_provider_resolver,
)

from victor.config.migration import (
    ConfigMigrator,
    MigrationResult,
    check_migration_needed,
    run_migration,
    rollback_migration,
)

from victor.config.connection_validation import (
    ConnectionValidator,
    ConnectionValidationResult,
    ConnectionTestResult,
    validate_account_sync,
    validate_account_async,
    ping_provider,
)

from victor.config.timeouts import TimeoutConfig, Timeouts

# Legacy components (for backward compatibility)
from victor.config.settings import Settings, load_settings, get_settings
from victor.config.provider_config_registry import (
    ProviderConfigStrategy,
    ProviderConfigRegistry,
    get_provider_config_registry,
    register_provider_config,
    DefaultProviderConfig,
    DEFAULT_PROVIDER_ENDPOINTS,
)

__all__ = [
    # New unified configuration
    "ProviderAccount",
    "AuthConfig",
    "AccountManager",
    "VictorConfig",
    "ConfigDefaults",
    "get_account_manager",
    "reset_account_manager",
    # Resolution
    "ProviderResolver",
    "ResolutionContext",
    "get_provider_resolver",
    "resolve_provider_config",
    "reset_provider_resolver",
    # Migration
    "ConfigMigrator",
    "MigrationResult",
    "check_migration_needed",
    "run_migration",
    "rollback_migration",
    # Connection validation
    "ConnectionValidator",
    "ConnectionValidationResult",
    "ConnectionTestResult",
    "validate_account_sync",
    "validate_account_async",
    "ping_provider",
    # Timeouts
    "TimeoutConfig",
    "Timeouts",
    # Legacy (backward compatibility)
    "Settings",
    "load_settings",
    "get_settings",
    "ProviderConfigStrategy",
    "ProviderConfigRegistry",
    "get_provider_config_registry",
    "register_provider_config",
    "DefaultProviderConfig",
    "DEFAULT_PROVIDER_ENDPOINTS",
]


def __getattr__(name: str):
    """Lazily expose config submodules on the package object.

    Some test and plugin flows reload ``victor.config`` without clearing its
    descendant submodules from ``sys.modules``. In that state, package-level
    attribute lookup such as ``victor.config.api_keys`` should still resolve to
    the already-imported submodule instead of failing with ``AttributeError``.
    """

    submodule_name = f"{__name__}.{name}"
    if find_spec(submodule_name) is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(submodule_name)
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    """Include lazily exposed submodules in package introspection."""

    return sorted(
        set(globals()) | {"api_keys", "config_loaders", "settings", "timeouts", "validation"}
    )

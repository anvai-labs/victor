# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Framework compatibility re-exports for capability config helper utilities."""

from victor_contracts.capabilities import (
    load_capability_config,
    resolve_capability_config_scope_key,
    resolve_capability_config_service,
    store_capability_config,
    update_capability_config_section,
)

__all__ = [
    "load_capability_config",
    "resolve_capability_config_scope_key",
    "resolve_capability_config_service",
    "store_capability_config",
    "update_capability_config_section",
]

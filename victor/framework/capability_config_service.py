# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Framework compatibility re-exports for capability config storage types."""

from victor_contracts.capabilities import (
    CapabilityConfigMergePolicy,
    CapabilityConfigService,
    DEFAULT_CAPABILITY_CONFIG_SCOPE_KEY,
)

__all__ = [
    "CapabilityConfigMergePolicy",
    "CapabilityConfigService",
    "DEFAULT_CAPABILITY_CONFIG_SCOPE_KEY",
]

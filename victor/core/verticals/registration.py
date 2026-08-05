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

"""Compatibility wrapper around the SDK vertical registration contract.

The canonical definition-layer decorator now lives in ``victor_contracts``.
This module remains for in-repo and incremental migration compatibility while
preserving the extra runtime side effects core expects:

- attach an SDK manifest
- register the decorated class with ``VerticalRegistry`` when available
- register the derived behavior configuration with ``VerticalBehaviorConfigRegistry``
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Optional, Type

from victor_contracts.verticals.registration import (
    ExtensionDependency,
    get_vertical_manifest as sdk_get_vertical_manifest,
    register_vertical as sdk_register_vertical,
)

if TYPE_CHECKING:
    from victor_contracts.verticals.manifest import ExtensionManifest

logger = logging.getLogger(__name__)


def register_vertical(*args, **kwargs) -> Callable[[Type], Type]:
    """Decorate a vertical using the SDK contract and add core runtime registration.

    This is a compatibility wrapper over ``victor_contracts.register_vertical``.
    """

    sdk_decorator = sdk_register_vertical(*args, **kwargs)

    def decorator(cls: Type) -> Type:
        decorated = sdk_decorator(cls)
        manifest = sdk_get_vertical_manifest(decorated)

        try:
            from victor.core.verticals.base import VerticalRegistry

            VerticalRegistry.register(decorated)
            logger.debug(
                "Registered vertical '%s' (%s)",
                getattr(decorated, "name", None),
                decorated,
            )
        except ImportError:
            logger.debug(
                "VerticalRegistry not available - skipping registration for '%s'",
                getattr(decorated, "name", None),
            )

        if manifest is not None:
            try:
                from victor.core.verticals.config_registry import (
                    VerticalBehaviorConfigRegistry,
                )

                behavior_config = VerticalBehaviorConfigRegistry.from_manifest(manifest)
                VerticalBehaviorConfigRegistry.register(manifest.name, behavior_config)
                logger.debug("Registered behavior configuration for '%s'", manifest.name)
            except ImportError:
                logger.debug(
                    "VerticalBehaviorConfigRegistry not available - "
                    "skipping behavior config registration for '%s'",
                    getattr(decorated, "name", None),
                )

        return decorated

    return decorator


def get_vertical_manifest(vertical_class: Type) -> Optional["ExtensionManifest"]:
    """Return the attached SDK manifest for a decorated vertical class."""

    return sdk_get_vertical_manifest(vertical_class)


__all__ = [
    "ExtensionDependency",
    "get_vertical_manifest",
    "register_vertical",
]

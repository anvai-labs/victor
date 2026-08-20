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

"""Post-bootstrap container validation.

Validates that all critical services are registered after bootstrap completes.
Logs warnings for optional missing services and raises for critical ones.
"""

from __future__ import annotations

import logging
from typing import Any, List

logger = logging.getLogger(__name__)


def validate_container(
    container: Any,
    specs: List[Any],
) -> List[str]:
    """Validate all critical service dependencies are registered.

    Args:
        container: ServiceContainer instance (must have ``is_registered()``).
        specs: List of ServiceSpec objects with ``protocol``, ``critical``,
            and ``depends_on`` attributes.

    Returns:
        List of error messages. Empty list means the container is healthy.
    """
    errors: List[str] = []

    for spec in specs:
        protocol = spec.protocol
        name = getattr(protocol, "__name__", str(protocol))

        if not container.is_registered(protocol):
            if getattr(spec, "critical", True):
                errors.append(f"Missing critical service: {name}")
            else:
                logger.debug("Optional service not registered: %s", name)
            continue

        # Check declared dependencies
        for dep in getattr(spec, "depends_on", ()):
            dep_name = getattr(dep, "__name__", str(dep))
            if not container.is_registered(dep):
                errors.append(
                    f"{name} declares dependency on {dep_name} " f"which is not registered"
                )

    return errors


def validate_and_report(
    container: Any,
    specs: List[Any],
    raise_on_critical: bool = True,
) -> None:
    """Validate container and log/raise results.

    Args:
        container: ServiceContainer instance.
        specs: List of ServiceSpec objects.
        raise_on_critical: If True, raises RuntimeError on critical failures.
    """
    errors = validate_container(container, specs)

    if not errors:
        logger.debug("Container validation passed: all services registered")
        return

    for error in errors:
        logger.error("Bootstrap validation: %s", error)

    if raise_on_critical:
        raise RuntimeError(
            f"Bootstrap validation failed with {len(errors)} error(s): " + "; ".join(errors[:3])
        )

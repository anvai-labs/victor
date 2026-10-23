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

"""Concrete helpers that accompany the framework's protocol contracts.

``victor.framework.protocols`` defines the public protocols and re-exports
these names unchanged (same objects). They live in core so the runtime can
raise and use them without importing the framework layer.
"""

from __future__ import annotations

from typing import Any, List, Tuple

__all__ = ["IncompatibleVersionError", "verify_protocol_conformance"]


class IncompatibleVersionError(Exception):
    """Raised when a capability version is incompatible with requirements.

    This exception is raised when invoking a capability that does not meet
    the minimum version requirement specified by the caller.

    Attributes:
        capability_name: Name of the capability
        required_version: Minimum version required
        actual_version: Actual version of the capability
    """

    def __init__(
        self,
        capability_name: str,
        required_version: str,
        actual_version: str,
    ) -> None:
        self.capability_name = capability_name
        self.required_version = required_version
        self.actual_version = actual_version
        super().__init__(
            f"Capability '{capability_name}' version {actual_version} "
            f"is incompatible with required version {required_version}"
        )


def verify_protocol_conformance(obj: Any, protocol: type) -> Tuple[bool, List[str]]:
    """Verify that an object conforms to a protocol.

    Checks that all required methods/properties are present.
    Useful for debugging protocol conformance issues.

    Args:
        obj: Object to check
        protocol: Protocol class to check against

    Returns:
        Tuple of (conforms: bool, missing: List[str])

    Example:
        conforms, missing = verify_protocol_conformance(orch, FrameworkOrchestratorProtocol)
        if not conforms:
            raise TypeError(f"Missing protocol methods: {missing}")
    """
    missing = []

    # Get protocol's __protocol_attrs__ if available (runtime_checkable)
    if hasattr(protocol, "__protocol_attrs__"):
        for attr in protocol.__protocol_attrs__:
            if not hasattr(obj, attr):
                missing.append(attr)
    else:
        # Fall back to checking annotations
        hints = getattr(protocol, "__annotations__", {})
        for attr in hints:
            if not hasattr(obj, attr):
                missing.append(attr)

    return len(missing) == 0, missing

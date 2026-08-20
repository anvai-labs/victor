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

"""TDD tests for Dependency Inversion Principle compliance in coordinators.

Coordinators must not access private attributes on the orchestrator via
hasattr(orch, "_private") or getattr(orch, "_private"). Instead they
should use the CapabilityRegistryMixin API:
  - orch.has_capability("name")
  - orch.get_capability_value("name")
  - orch.invoke_capability("name", ...)

Or use public methods/properties that already exist.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

# =============================================================================
# Regression guard: no hasattr/getattr on private orchestrator attributes
# =============================================================================

# Coordinator files that must be DIP-compliant
COORDINATOR_DIR = Path("victor/agent/coordinators")
COORDINATOR_FILES = [
    COORDINATOR_DIR / "planning_coordinator.py",
    COORDINATOR_DIR / "chat_coordinator.py",
    COORDINATOR_DIR / "sync_chat_coordinator.py",
    COORDINATOR_DIR / "protocol_adapters.py",
    # NOTE: tool_coordinator.py removed - now in victor.agent.services.tool_compat
]

# Pattern: hasattr(something, "_private") or getattr(something, "_private"...)
# This catches both `hasattr(orch, "_foo")` and `getattr(self._orchestrator, "_foo", ...)`
_PRIVATE_ATTR_PATTERN = re.compile(r"""(?:hasattr|getattr)\s*\(\s*[^,]+,\s*['"](_[a-z_]+)['"]\s*""")

# Attributes that are legitimate private-attribute access patterns:
# - _cumulative_token_usage: internal state dict accessed via known pattern
# - _context_manager: accessed for start_background_compaction (has hasattr guard on method)
# - _middleware_chain: registered as capability with attribute= field
ALLOWLISTED_ATTRS = frozenset(
    {
        "_cumulative_token_usage",
        "_context_manager",
        "_middleware_chain",
    }
)


class TestNoDIPViolationsInCoordinators:
    """Regression guard: coordinators must not probe private orchestrator attrs."""

    @pytest.mark.parametrize("filepath", COORDINATOR_FILES, ids=lambda p: p.name)
    def test_no_hasattr_getattr_on_private_attrs(self, filepath: Path):
        """No coordinator should use hasattr/getattr to check private attributes.

        Private attributes (_foo) on the orchestrator should be accessed via
        the CapabilityRegistryMixin API, not via duck-typing.
        """
        if not filepath.exists():
            pytest.skip(f"{filepath} does not exist")

        source = filepath.read_text(encoding="utf-8")
        violations = []

        for lineno, line in enumerate(source.splitlines(), start=1):
            # Skip comments
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue

            match = _PRIVATE_ATTR_PATTERN.search(line)
            if match:
                attr_name = match.group(1)
                if attr_name not in ALLOWLISTED_ATTRS:
                    violations.append(f"  {filepath.name}:{lineno}: {stripped.strip()}")

        assert not violations, (
            "DIP violations found — use CapabilityRegistryMixin API instead "
            "of hasattr/getattr on private attributes:\n" + "\n".join(violations)
        )


# =============================================================================
# Capability registration tests
# =============================================================================


class TestRequiredCapabilitiesRegistered:
    """Verify that capabilities needed by coordinators are registered."""

    def _get_registered_capability_names(self) -> set:
        """Get the set of capability names from the registry source."""
        registry_path = Path("victor/agent/capability_registry.py")
        source = registry_path.read_text(encoding="utf-8")
        # Extract capability names from _register_capability calls
        return set(re.findall(r'name="([^"]+)"', source))

    def test_tool_sequence_tracker_registered(self):
        """tool_sequence_tracker should be registered (used by chat_coordinator)."""
        names = self._get_registered_capability_names()
        assert "tool_sequence_tracker" in names

    def test_enabled_tools_registered(self):
        """enabled_tools should be registered (used by tool_coordinator)."""
        names = self._get_registered_capability_names()
        assert "enabled_tools" in names

    def test_usage_analytics_registered(self):
        """usage_analytics should be registered (used by chat_coordinator)."""
        names = self._get_registered_capability_names()
        assert "usage_analytics" in names

    def test_context_compactor_registered(self):
        """context_compactor should be registered (used by planning_coordinator)."""
        names = self._get_registered_capability_names()
        assert "context_compactor" in names

    def test_current_stream_context_registered(self):
        """current_stream_context should be registered (used by chat_coordinator)."""
        names = self._get_registered_capability_names()
        assert "current_stream_context" in names

    def test_system_prompt_added_registered(self):
        """system_prompt_added should be registered (used by protocol_adapters)."""
        names = self._get_registered_capability_names()
        assert "system_prompt_added" in names


# =============================================================================
# Specific coordinator fix tests
# =============================================================================


# NOTE: TestToolCoordinatorUsesPublicAPI class removed
# tool_coordinator.py has been deleted - functionality moved to victor.agent.services.tool_compat
# The service-level implementation is tested separately in tests/unit/agent/services/

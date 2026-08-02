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

"""Tests for victor.framework.persona_registry module.

These tests verify the PersonaRegistry functionality for cross-vertical
persona discovery and access.
"""

import pytest

from victor.framework.persona_registry import (
    PersonaRegistry,
    PersonaSpec,
    get_persona_registry,
    register_persona_spec,
    get_persona_spec,
)

# =============================================================================
# PersonaSpec Tests
# =============================================================================


class TestPersonaSpec:
    """Tests for PersonaSpec dataclass."""

    def test_persona_spec_basic_attributes(self):
        """PersonaSpec should store basic attributes."""
        spec = PersonaSpec(
            name="security_expert",
            role="Security Analyst",
            expertise=["security", "authentication", "encryption"],
            communication_style="formal",
            behavioral_traits=["thorough", "risk-aware"],
            vertical="coding",
            tags=["security", "enterprise"],
        )

        assert spec.name == "security_expert"
        assert spec.role == "Security Analyst"
        assert spec.expertise == ["security", "authentication", "encryption"]
        assert spec.communication_style == "formal"
        assert spec.behavioral_traits == ["thorough", "risk-aware"]
        assert spec.vertical == "coding"
        assert spec.tags == ["security", "enterprise"]

    def test_persona_spec_full_name_with_vertical(self):
        """PersonaSpec should return full name with vertical namespace."""
        spec = PersonaSpec(name="senior_dev", role="Senior Developer", vertical="coding")
        assert spec.full_name == "coding:senior_dev"

    def test_persona_spec_full_name_without_vertical(self):
        """PersonaSpec should return name as full_name when no vertical."""
        spec = PersonaSpec(name="senior_dev", role="Senior Developer")
        assert spec.full_name == "senior_dev"

    def test_persona_spec_defaults(self):
        """PersonaSpec should have sensible defaults."""
        spec = PersonaSpec(name="simple", role="Simple Role")

        assert spec.name == "simple"
        assert spec.role == "Simple Role"
        assert spec.expertise == []
        assert spec.communication_style == ""
        assert spec.behavioral_traits == []
        assert spec.vertical is None
        assert spec.tags == []

    def test_persona_spec_to_dict(self):
        """PersonaSpec should convert to dictionary."""
        spec = PersonaSpec(
            name="test_persona",
            role="Tester",
            expertise=["testing"],
            communication_style="casual",
            behavioral_traits=["friendly"],
            vertical="coding",
            tags=["test"],
        )

        result = spec.to_dict()

        assert result["name"] == "test_persona"
        assert result["role"] == "Tester"
        assert result["expertise"] == ["testing"]
        assert result["communication_style"] == "casual"
        assert result["behavioral_traits"] == ["friendly"]
        assert result["vertical"] == "coding"
        assert result["tags"] == ["test"]


# =============================================================================
# PersonaRegistry Basic Tests
# =============================================================================


class TestPersonaRegistry:
    """Tests for PersonaRegistry class."""

    def test_registry_register_and_get(self):
        """Registry should register and retrieve personas."""
        registry = PersonaRegistry()
        spec = PersonaSpec(name="test_persona", role="Tester")

        registry.register("test_persona", spec)
        result = registry.get("test_persona")

        assert result is spec

    def test_registry_register_with_vertical(self):
        """Registry should register personas with vertical namespace."""
        registry = PersonaRegistry()
        spec = PersonaSpec(name="senior_dev", role="Senior Developer")

        registry.register("senior_dev", spec, vertical="coding")
        result = registry.get("senior_dev", vertical="coding")

        assert result is spec
        assert result.vertical == "coding"

    def test_registry_get_with_vertical_returns_namespaced(self):
        """Registry should return vertical-namespaced persona when vertical specified."""
        registry = PersonaRegistry()
        global_persona = PersonaSpec(name="reviewer", role="Global Reviewer")
        coding_persona = PersonaSpec(name="reviewer", role="Code Reviewer")

        registry.register("reviewer", global_persona)
        registry.register("reviewer", coding_persona, vertical="coding")

        # Without vertical, should get global
        assert registry.get("reviewer") is global_persona
        # With vertical, should get namespaced
        assert registry.get("reviewer", vertical="coding") is coding_persona

    def test_registry_register_duplicate_skips(self):
        """Registry should skip duplicate registration without replace=True."""
        registry = PersonaRegistry()
        spec1 = PersonaSpec(name="test", role="Role1")
        spec2 = PersonaSpec(name="test", role="Role2")

        registry.register("test", spec1)
        registry.register("test", spec2)  # Should be skipped

        assert registry.get("test") is spec1

    def test_registry_register_duplicate_with_replace(self):
        """Registry should allow replacing with replace=True."""
        registry = PersonaRegistry()
        spec1 = PersonaSpec(name="test", role="Role1")
        spec2 = PersonaSpec(name="test", role="Role2")

        registry.register("test", spec1)
        registry.register("test", spec2, replace=True)

        assert registry.get("test") is spec2

    def test_registry_unregister(self):
        """Registry should unregister personas."""
        registry = PersonaRegistry()
        spec = PersonaSpec(name="test", role="Tester")

        registry.register("test", spec)
        result = registry.unregister("test")

        assert result is True
        assert registry.get("test") is None

    def test_registry_unregister_with_vertical(self):
        """Registry should unregister personas with vertical namespace."""
        registry = PersonaRegistry()
        spec = PersonaSpec(name="dev", role="Developer")

        registry.register("dev", spec, vertical="coding")
        result = registry.unregister("dev", vertical="coding")

        assert result is True
        assert registry.get("dev", vertical="coding") is None

    def test_registry_unregister_nonexistent(self):
        """Registry should return False when unregistering nonexistent persona."""
        registry = PersonaRegistry()
        result = registry.unregister("nonexistent")
        assert result is False


# =============================================================================
# PersonaRegistry Listing Tests
# =============================================================================


class TestPersonaRegistryListing:
    """Tests for PersonaRegistry listing methods."""

    def test_registry_list_personas(self):
        """Registry should list all persona names."""
        registry = PersonaRegistry()
        registry.register("persona1", PersonaSpec(name="p1", role="R1"))
        registry.register("persona2", PersonaSpec(name="p2", role="R2"), vertical="coding")
        registry.register("persona3", PersonaSpec(name="p3", role="R3"), vertical="devops")

        personas = registry.list_personas()

        assert len(personas) == 3
        assert "persona1" in personas
        assert "coding:persona2" in personas
        assert "devops:persona3" in personas

    def test_registry_list_personas_by_vertical(self):
        """Registry should list personas filtered by vertical."""
        registry = PersonaRegistry()
        registry.register("persona1", PersonaSpec(name="p1", role="R1"))
        registry.register("persona2", PersonaSpec(name="p2", role="R2"), vertical="coding")
        registry.register("persona3", PersonaSpec(name="p3", role="R3"), vertical="coding")
        registry.register("persona4", PersonaSpec(name="p4", role="R4"), vertical="devops")

        coding_personas = registry.list_personas(vertical="coding")

        assert len(coding_personas) == 2
        assert "coding:persona2" in coding_personas
        assert "coding:persona3" in coding_personas

    def test_registry_list_specs(self):
        """Registry should list all persona specs."""
        registry = PersonaRegistry()
        spec1 = PersonaSpec(name="persona1", role="Role1")
        spec2 = PersonaSpec(name="persona2", role="Role2")

        registry.register("persona1", spec1)
        registry.register("persona2", spec2, vertical="coding")

        specs = registry.list_specs()

        assert len(specs) == 2
        assert spec1 in specs
        assert spec2 in specs

    def test_registry_list_specs_by_vertical(self):
        """Registry should list specs filtered by vertical."""
        registry = PersonaRegistry()
        spec1 = PersonaSpec(name="persona1", role="Role1")
        spec2 = PersonaSpec(name="persona2", role="Role2")
        spec3 = PersonaSpec(name="persona3", role="Role3")

        registry.register("persona1", spec1)
        registry.register("persona2", spec2, vertical="coding")
        registry.register("persona3", spec3, vertical="devops")

        coding_specs = registry.list_specs(vertical="coding")

        assert len(coding_specs) == 1
        assert coding_specs[0].name == "persona2"


# =============================================================================
# PersonaRegistry Find Tests
# =============================================================================


class TestPersonaRegistryFind:
    """Tests for PersonaRegistry find methods."""

    def test_registry_find_by_vertical(self):
        """Registry should find personas by vertical."""
        registry = PersonaRegistry()
        coding_spec = PersonaSpec(name="dev", role="Developer")
        devops_spec = PersonaSpec(name="ops", role="Operator")

        registry.register("dev", coding_spec, vertical="coding")
        registry.register("ops", devops_spec, vertical="devops")

        coding_personas = registry.find_by_vertical("coding")

        assert len(coding_personas) == 1
        assert coding_personas[0] is coding_spec

    def test_registry_find_by_expertise(self):
        """Registry should find personas by expertise."""
        registry = PersonaRegistry()
        security_expert = PersonaSpec(
            name="security",
            role="Security Expert",
            expertise=["security", "authentication"],
        )
        python_expert = PersonaSpec(
            name="python",
            role="Python Expert",
            expertise=["python", "django"],
        )

        registry.register("security", security_expert)
        registry.register("python", python_expert)

        security_personas = registry.find_by_expertise("security")

        assert len(security_personas) == 1
        assert security_personas[0] is security_expert

    def test_registry_find_by_role(self):
        """Registry should find personas by role (case-insensitive substring)."""
        registry = PersonaRegistry()
        senior_dev = PersonaSpec(name="senior", role="Senior Developer")
        junior_dev = PersonaSpec(name="junior", role="Junior Developer")
        analyst = PersonaSpec(name="analyst", role="Security Analyst")

        registry.register("senior", senior_dev)
        registry.register("junior", junior_dev)
        registry.register("analyst", analyst)

        developers = registry.find_by_role("developer")

        assert len(developers) == 2
        assert senior_dev in developers
        assert junior_dev in developers

    def test_registry_find_by_tag(self):
        """Registry should find personas by tag."""
        registry = PersonaRegistry()
        spec1 = PersonaSpec(name="p1", role="R1", tags=["senior", "coding"])
        spec2 = PersonaSpec(name="p2", role="R2", tags=["junior", "testing"])

        registry.register("p1", spec1)
        registry.register("p2", spec2)

        senior_personas = registry.find_by_tag("senior")

        assert len(senior_personas) == 1
        assert senior_personas[0] is spec1

    def test_registry_find_by_tags_match_any(self):
        """Registry should find personas matching any of multiple tags."""
        registry = PersonaRegistry()
        spec1 = PersonaSpec(name="p1", role="R1", tags=["senior", "coding"])
        spec2 = PersonaSpec(name="p2", role="R2", tags=["junior"])
        spec3 = PersonaSpec(name="p3", role="R3", tags=["coding", "testing"])

        registry.register("p1", spec1)
        registry.register("p2", spec2)
        registry.register("p3", spec3)

        personas = registry.find_by_tags(["coding", "testing"], match_all=False)

        assert len(personas) == 2
        assert spec1 in personas  # Has coding
        assert spec3 in personas  # Has both

    def test_registry_find_by_tags_match_all(self):
        """Registry should find personas matching all tags."""
        registry = PersonaRegistry()
        spec1 = PersonaSpec(name="p1", role="R1", tags=["senior", "coding"])
        spec2 = PersonaSpec(name="p2", role="R2", tags=["senior", "coding", "python"])

        registry.register("p1", spec1)
        registry.register("p2", spec2)

        personas = registry.find_by_tags(["senior", "coding"], match_all=True)

        assert len(personas) == 2  # Both have senior and coding

    def test_registry_replacement_updates_indexes(self):
        """Replacing a persona should remove stale indexed lookup entries."""
        registry = PersonaRegistry()
        original = PersonaSpec(
            name="persona",
            role="Old Role",
            expertise=["legacy"],
            tags=["old"],
        )
        replacement = PersonaSpec(
            name="persona",
            role="New Role",
            expertise=["modern"],
            tags=["new"],
        )

        registry.register("persona", original, vertical="coding")
        registry.register("persona", replacement, vertical="coding", replace=True)

        assert registry.find_by_tag("old") == []
        assert registry.find_by_expertise("legacy") == []
        assert registry.find_by_role("Old Role") == []
        assert registry.find_by_tag("new") == [replacement]
        assert registry.find_by_expertise("modern") == [replacement]
        assert registry.find_by_vertical("coding") == [replacement]

    def test_registry_unregister_clears_index_entries(self):
        """Unregistering a persona should remove all indexed lookup entries."""
        registry = PersonaRegistry()
        spec = PersonaSpec(
            name="persona",
            role="Indexed Role",
            expertise=["python"],
            tags=["review"],
        )

        registry.register("persona", spec, vertical="coding")
        assert registry.unregister("persona", vertical="coding") is True

        assert registry.find_by_tag("review") == []
        assert registry.find_by_expertise("python") == []
        assert registry.find_by_role("Indexed Role") == []
        assert registry.find_by_vertical("coding") == []


# =============================================================================
# PersonaRegistry Bulk Operations Tests
# =============================================================================


class TestPersonaRegistryBulkOperations:
    """Tests for PersonaRegistry bulk operations."""

    def test_registry_clear(self):
        """Registry should clear all personas."""
        registry = PersonaRegistry()
        registry.register("p1", PersonaSpec(name="p1", role="R1"))
        registry.register("p2", PersonaSpec(name="p2", role="R2"))

        registry.clear()

        assert len(registry.list_personas()) == 0

    def test_registry_register_from_vertical(self):
        """Registry should register multiple personas from a vertical."""
        registry = PersonaRegistry()
        personas = {
            "senior_dev": PersonaSpec(name="senior_dev", role="Senior Developer"),
            "junior_dev": PersonaSpec(name="junior_dev", role="Junior Developer"),
        }

        count = registry.register_from_vertical("coding", personas)

        assert count == 2
        assert registry.get("senior_dev", vertical="coding") is not None
        assert registry.get("junior_dev", vertical="coding") is not None


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset the global registry before each test."""
        get_persona_registry().clear()
        yield
        get_persona_registry().clear()

    def test_register_persona_spec_function(self):
        """register_persona_spec should add to global registry."""
        spec = PersonaSpec(name="test", role="Tester")
        register_persona_spec("test", spec)

        result = get_persona_spec("test")
        assert result is spec

    def test_register_persona_spec_with_vertical(self):
        """register_persona_spec should add to global registry with vertical."""
        spec = PersonaSpec(name="dev", role="Developer")
        register_persona_spec("dev", spec, vertical="coding")

        result = get_persona_spec("dev", vertical="coding")
        assert result is spec

    def test_get_persona_spec_function(self):
        """get_persona_spec should retrieve from global registry."""
        spec = PersonaSpec(name="test", role="Tester")
        register_persona_spec("test", spec)

        result = get_persona_spec("test")
        assert result is spec

    def test_get_persona_spec_returns_none_for_missing(self):
        """get_persona_spec should return None for missing persona."""
        result = get_persona_spec("nonexistent")
        assert result is None

    def test_get_persona_registry_singleton(self):
        """get_persona_registry should return the same instance."""
        registry1 = get_persona_registry()
        registry2 = get_persona_registry()
        assert registry1 is registry2


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestPersonaRegistryThreadSafety:
    """Tests for PersonaRegistry thread safety."""

    def test_concurrent_registration(self):
        """Registry should handle concurrent registrations safely."""
        import threading

        registry = PersonaRegistry()
        errors = []

        def register_personas(prefix: str, count: int):
            try:
                for i in range(count):
                    spec = PersonaSpec(name=f"{prefix}_persona_{i}", role=f"Role {i}")
                    registry.register(f"{prefix}_persona_{i}", spec, replace=True)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=register_personas, args=("thread1", 100)),
            threading.Thread(target=register_personas, args=("thread2", 100)),
            threading.Thread(target=register_personas, args=("thread3", 100)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(registry.list_personas()) == 300

    def test_concurrent_read_write(self):
        """Registry should handle concurrent reads and writes safely."""
        import threading

        registry = PersonaRegistry()
        errors = []

        def writer():
            try:
                for i in range(50):
                    spec = PersonaSpec(name=f"persona_{i}", role=f"Role {i}")
                    registry.register(f"persona_{i}", spec, replace=True)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(50):
                    registry.list_personas()
                    registry.get("persona_0")
                    registry.find_by_expertise("python")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

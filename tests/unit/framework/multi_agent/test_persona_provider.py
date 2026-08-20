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

"""Tests for FrameworkPersonaProvider.

These tests verify the persona provider's singleton behavior, versioning,
registration, and retrieval mechanisms.
"""

import pytest
import threading
import time
from typing import Dict

from victor.framework.multi_agent.persona_provider import (
    FrameworkPersonaProvider,
    PersonaMetadata,
    get_persona_provider,
)
from victor.framework.multi_agent.personas import PersonaTraits

# =============================================================================
# Fixture: Reset Provider Singleton
# =============================================================================


@pytest.fixture(autouse=True)
def reset_persona_provider():
    """Reset FrameworkPersonaProvider singleton before/after each test."""
    # Reset before test (thread-safe)
    FrameworkPersonaProvider.reset_instance()
    yield
    # Reset after test (thread-safe)
    FrameworkPersonaProvider.reset_instance()


@pytest.fixture
def sample_persona() -> PersonaTraits:
    """Create a sample persona for testing."""
    from victor.framework.multi_agent.personas import CommunicationStyle, ExpertiseLevel

    return PersonaTraits(
        name="Test Researcher",
        role="researcher",
        description="A test researcher persona",
        communication_style=CommunicationStyle.TECHNICAL,
        expertise_level=ExpertiseLevel.EXPERT,
        strengths=["research", "analysis"],
        custom_traits={"traits": ["thorough", "systematic"]},
    )


# =============================================================================
# Singleton Behavior Tests
# =============================================================================


class TestFrameworkPersonaProviderSingleton:
    """Tests for FrameworkPersonaProvider singleton pattern."""

    def test_returns_same_instance(self):
        """Should return the same instance on multiple calls."""
        provider1 = FrameworkPersonaProvider()
        provider2 = FrameworkPersonaProvider()

        assert provider1 is provider2
        assert id(provider1) == id(provider2)

    def test_get_persona_provider_singleton(self):
        """get_persona_provider() should return singleton instance."""
        provider1 = get_persona_provider()
        provider2 = get_persona_provider()

        assert provider1 is provider2
        assert isinstance(provider1, FrameworkPersonaProvider)

    def test_thread_safe_singleton(self):
        """Singleton creation should be thread-safe."""
        instances = []
        lock = threading.Lock()

        def create_instance():
            provider = FrameworkPersonaProvider()
            with lock:
                instances.append(provider)

        threads = [threading.Thread(target=create_instance) for _ in range(10)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # All instances should be the same
        first_instance = instances[0]
        for instance in instances[1:]:
            assert instance is first_instance

    def test_initialized_only_once(self):
        """__init__ should only execute once despite multiple calls."""
        provider1 = FrameworkPersonaProvider()
        provider2 = FrameworkPersonaProvider()

        # Both should have same internal state
        assert hasattr(provider1, "_initialized")
        assert hasattr(provider2, "_initialized")
        assert provider1._initialized is True
        assert provider2._initialized is True


# =============================================================================
# Initialization Tests
# =============================================================================


class TestFrameworkPersonaProviderInitialization:
    """Tests for FrameworkPersonaProvider initialization."""

    def test_initializes_with_empty_personas(self):
        """Should start with empty persona registry."""
        provider = FrameworkPersonaProvider()

        assert provider._personas == {}
        assert provider._metadata == {}

    def test_initializes_categories(self):
        """Should initialize all categories."""
        provider = FrameworkPersonaProvider()

        expected_categories = {"research", "planning", "execution", "review", "other"}
        assert set(provider._categories.keys()) == expected_categories

        # All categories should start empty
        for category in expected_categories:
            assert provider._categories[category] == set()

    def test_initializes_capability_provider_attributes(self):
        """Should initialize BaseCapabilityProvider attributes."""
        provider = FrameworkPersonaProvider()

        assert hasattr(provider, "_capabilities")
        assert hasattr(provider, "_metadata_full")
        assert provider._capabilities == {}
        assert provider._metadata_full == {}


# =============================================================================
# Persona Registration Tests
# =============================================================================


class TestFrameworkPersonaProviderRegistration:
    """Tests for persona registration."""

    def test_register_basic_persona(self, sample_persona):
        """Should register a basic persona successfully."""
        provider = FrameworkPersonaProvider()

        provider.register_persona(
            name="test_persona",
            version="1.0.0",
            persona=sample_persona,
            category="research",
            description="Test persona",
        )

        assert "test_persona" in provider._personas
        assert "1.0.0" in provider._personas["test_persona"]
        assert provider._personas["test_persona"]["1.0.0"] is sample_persona

    def test_register_persona_with_metadata(self, sample_persona):
        """Should store persona metadata correctly."""
        provider = FrameworkPersonaProvider()

        provider.register_persona(
            name="test_persona",
            version="1.0.0",
            persona=sample_persona,
            category="execution",
            description="Test execution persona",
            tags=["execution", "testing"],
            author="Test Author",
            vertical="coding",
            deprecated=False,
        )

        metadata = provider._metadata["test_persona"]["1.0.0"]
        assert metadata.name == "test_persona"
        assert metadata.version == "1.0.0"
        assert metadata.description == "Test execution persona"
        assert metadata.category == "execution"
        assert metadata.tags == ["execution", "testing"]
        assert metadata.author == "Test Author"
        assert metadata.vertical == "coding"
        assert metadata.deprecated is False

    def test_register_persona_adds_to_category(self, sample_persona):
        """Should add persona to correct category."""
        provider = FrameworkPersonaProvider()

        provider.register_persona(
            name="test_planner",
            version="1.0.0",
            persona=sample_persona,
            category="planning",
            description="Test planner",
        )

        assert "test_planner" in provider._categories["planning"]
        assert "test_planner" not in provider._categories["research"]
        assert "test_planner" not in provider._categories["execution"]

    def test_register_persona_adds_to_capabilities(self, sample_persona):
        """Should add persona to capabilities dict."""
        provider = FrameworkPersonaProvider()

        provider.register_persona(
            name="test_cap",
            version="1.0.0",
            persona=sample_persona,
            category="review",
            description="Test capability",
        )

        assert "test_cap" in provider._capabilities
        assert provider._capabilities["test_cap"] is sample_persona

    def test_register_multiple_versions_same_persona(self, sample_persona):
        """Should register multiple versions of the same persona."""
        from victor.framework.multi_agent.personas import (
            CommunicationStyle,
            ExpertiseLevel,
        )

        provider = FrameworkPersonaProvider()

        provider.register_persona(
            name="multi_version",
            version="1.0.0",
            persona=sample_persona,
            category="research",
            description="Version 1.0.0",
        )

        # Create modified version
        v2_persona = PersonaTraits(
            name="Test Researcher v2",
            role="researcher",
            description="Updated researcher persona",
            communication_style=CommunicationStyle.TECHNICAL,
            expertise_level=ExpertiseLevel.EXPERT,
            strengths=["research", "analysis", "synthesis"],
            custom_traits={"traits": ["thorough", "systematic", "innovative"]},
        )

        provider.register_persona(
            name="multi_version",
            version="2.0.0",
            persona=v2_persona,
            category="research",
            description="Version 2.0.0",
        )

        assert "1.0.0" in provider._personas["multi_version"]
        assert "2.0.0" in provider._personas["multi_version"]
        assert provider._personas["multi_version"]["1.0.0"] is sample_persona
        assert provider._personas["multi_version"]["2.0.0"] is v2_persona


# =============================================================================
# SemVer Validation Tests
# =============================================================================


class TestSemVerValidation:
    """Tests for semantic versioning validation."""

    def test_valid_semver_basic(self, sample_persona):
        """Should accept valid basic SemVer."""
        provider = FrameworkPersonaProvider()

        valid_versions = ["1.0.0", "2.1.3", "0.1.0", "10.20.30"]

        for version in valid_versions:
            provider.register_persona(
                name=f"test_{version.replace('.', '_')}",
                version=version,
                persona=sample_persona,
                category="research",
                description=f"Test {version}",
            )
            assert version in provider._personas[f"test_{version.replace('.', '_')}"]

    def test_valid_semver_with_pre_release(self, sample_persona):
        """Should accept SemVer with pre-release identifiers."""
        provider = FrameworkPersonaProvider()

        valid_versions = [
            "1.0.0-alpha",
            "2.0.0-beta.1",
            "1.0.0-rc.1",
            "3.0.0-alpha.1.beta.2",
        ]

        for version in valid_versions:
            provider.register_persona(
                name=f"test_{version.replace('.', '_').replace('-', '_')}",
                version=version,
                persona=sample_persona,
                category="research",
                description=f"Test {version}",
            )

    def test_valid_semver_with_build_metadata(self, sample_persona):
        """Should accept SemVer with build metadata."""
        provider = FrameworkPersonaProvider()

        valid_versions = ["1.0.0+20130313144700", "1.0.0+build.1"]

        for version in valid_versions:
            provider.register_persona(
                name=f"test_{version.replace('.', '_').replace('+', '_')}",
                version=version,
                persona=sample_persona,
                category="research",
                description=f"Test {version}",
            )

    def test_invalid_semver_raises(self, sample_persona):
        """Should reject invalid SemVer."""
        provider = FrameworkPersonaProvider()

        invalid_versions = [
            "1",  # Missing minor and patch
            "1.0",  # Missing patch
            "1.0.0.0",  # Too many parts
            "01.0.0",  # Leading zero
            "v1.0.0",  # Prefix
            "1.0.",  # Trailing dot
            ".1.0.0",  # Leading dot
            "a.b.c",  # Non-numeric
            "",  # Empty string
        ]

        for version in invalid_versions:
            with pytest.raises(ValueError, match="Invalid SemVer"):
                provider.register_persona(
                    name="invalid",
                    version=version,
                    persona=sample_persona,
                    category="research",
                    description="Invalid version",
                )

    def test_is_valid_semver_static_method(self):
        """_is_valid_semver static method should work correctly."""
        assert FrameworkPersonaProvider._is_valid_semver("1.0.0") is True
        assert FrameworkPersonaProvider._is_valid_semver("2.1.3") is True
        assert FrameworkPersonaProvider._is_valid_semver("1.0.0-alpha") is True
        assert FrameworkPersonaProvider._is_valid_semver("1.0.0+build") is True
        assert FrameworkPersonaProvider._is_valid_semver("1.0") is False
        assert FrameworkPersonaProvider._is_valid_semver("invalid") is False
        assert FrameworkPersonaProvider._is_valid_semver("") is False


# =============================================================================
# SemVer Sorting Tests
# =============================================================================


class TestSemVerSorting:
    """Tests for semantic versioning sorting."""

    def test_semver_key_basic(self):
        """_semver_key should convert versions to comparable tuples."""
        assert FrameworkPersonaProvider._semver_key("1.0.0") == (1, 0, 0)
        assert FrameworkPersonaProvider._semver_key("2.1.3") == (2, 1, 3)
        assert FrameworkPersonaProvider._semver_key("10.20.30") == (10, 20, 30)

    def test_semver_key_ignores_pre_release_and_build(self):
        """_semver_key should strip pre-release and build metadata."""
        assert FrameworkPersonaProvider._semver_key("1.0.0-alpha") == (1, 0, 0)
        assert FrameworkPersonaProvider._semver_key("1.0.0+build") == (1, 0, 0)
        assert FrameworkPersonaProvider._semver_key("1.0.0-alpha+build") == (1, 0, 0)

    def test_get_latest_version_basic(self):
        """_get_latest_version should return the highest version."""
        versions = ["1.0.0", "1.1.0", "2.0.0", "1.0.1"]
        latest = FrameworkPersonaProvider._get_latest_version(versions)

        assert latest == "2.0.0"

    def test_get_latest_version_with_prerelease(self):
        """_get_latest_version should handle pre-release versions."""
        versions = ["1.0.0", "1.0.0-alpha", "1.0.0-beta", "2.0.0-alpha"]
        latest = FrameworkPersonaProvider._get_latest_version(versions)

        # Should strip pre-release and return highest version
        # After stripping: 1.0.0, 1.0.0, 1.0.0, 2.0.0 -> highest is 2.0.0-alpha
        assert latest == "2.0.0-alpha"

    def test_get_latest_version_empty_list(self):
        """_get_latest_version should return '0.0.0' for empty list."""
        latest = FrameworkPersonaProvider._get_latest_version([])
        assert latest == "0.0.0"


# =============================================================================
# Category Validation Tests
# =============================================================================


class TestCategoryValidation:
    """Tests for category validation."""

    def test_valid_categories(self, sample_persona):
        """Should accept all valid categories."""
        provider = FrameworkPersonaProvider()

        valid_categories = ["research", "planning", "execution", "review", "other"]

        for i, category in enumerate(valid_categories):
            provider.register_persona(
                name=f"test_{category}",
                version="1.0.0",
                persona=sample_persona,
                category=category,
                description=f"Test {category}",
            )
            assert f"test_{category}" in provider._categories[category]

    def test_invalid_category_raises(self, sample_persona):
        """Should raise ValueError for invalid category."""
        provider = FrameworkPersonaProvider()

        with pytest.raises(ValueError, match="Invalid category"):
            provider.register_persona(
                name="invalid_category",
                version="1.0.0",
                persona=sample_persona,
                category="invalid_category_name",
                description="Invalid category",
            )


# =============================================================================
# Persona Retrieval Tests
# =============================================================================


class TestFrameworkPersonaProviderRetrieval:
    """Tests for persona retrieval."""

    @pytest.fixture
    def populated_provider(self, sample_persona) -> FrameworkPersonaProvider:
        """Create a provider with registered personas."""
        from victor.framework.multi_agent.personas import (
            CommunicationStyle,
            ExpertiseLevel,
        )

        provider = FrameworkPersonaProvider()

        provider.register_persona(
            name="single_version",
            version="1.0.0",
            persona=sample_persona,
            category="research",
            description="Single version",
        )

        # Register multiple versions
        provider.register_persona(
            name="multi_version",
            version="1.0.0",
            persona=sample_persona,
            category="execution",
            description="Multi v1",
        )

        v2_persona = PersonaTraits(
            name="Multi v2",
            role="executor",
            description="Version 2",
            communication_style=CommunicationStyle.CONCISE,
            expertise_level=ExpertiseLevel.EXPERT,
            strengths=["execution"],
        )
        provider.register_persona(
            name="multi_version",
            version="2.0.0",
            persona=v2_persona,
            category="execution",
            description="Multi v2",
        )

        v3_persona = PersonaTraits(
            name="Multi v3",
            role="executor",
            description="Version 3",
            communication_style=CommunicationStyle.CONCISE,
            expertise_level=ExpertiseLevel.EXPERT,
            strengths=["execution", "testing"],
        )
        provider.register_persona(
            name="multi_version",
            version="1.5.0",
            persona=v3_persona,
            category="execution",
            description="Multi v1.5",
        )

        return provider

    def test_get_persona_exists(self, populated_provider):
        """Should retrieve existing persona."""
        persona = populated_provider.get_persona("single_version")

        assert persona is not None
        assert persona.name == "Test Researcher"
        assert persona.role == "researcher"

    def test_get_persona_not_exists(self, populated_provider):
        """Should return None for non-existent persona."""
        persona = populated_provider.get_persona("nonexistent")
        assert persona is None

    def test_get_persona_with_version(self, populated_provider):
        """Should retrieve specific version."""
        persona = populated_provider.get_persona("multi_version", version="1.0.0")

        assert persona is not None
        assert persona.name == "Test Researcher"

    def test_get_persona_without_version_returns_latest(self, populated_provider):
        """Should return latest version when version not specified."""
        persona = populated_provider.get_persona("multi_version")

        assert persona is not None
        # Latest should be 2.0.0
        assert persona.name == "Multi v2"

    def test_get_persona_metadata(self, populated_provider):
        """Should retrieve persona metadata."""
        metadata = populated_provider.get_persona_metadata("single_version")

        assert metadata is not None
        assert metadata.name == "single_version"
        assert metadata.version == "1.0.0"
        assert metadata.description == "Single version"
        assert metadata.category == "research"

    def test_get_persona_metadata_not_exists(self, populated_provider):
        """Should return None for non-existent persona metadata."""
        metadata = populated_provider.get_persona_metadata("nonexistent")
        assert metadata is None

    def test_get_persona_metadata_latest_version(self, populated_provider):
        """Should return latest version metadata when version not specified."""
        metadata = populated_provider.get_persona_metadata("multi_version")

        assert metadata is not None
        assert metadata.version == "2.0.0"  # Latest version

    def test_get_persona_version(self, populated_provider):
        """Should return latest version string."""
        version = populated_provider.get_persona_version("multi_version")

        assert version == "2.0.0"

    def test_get_persona_version_not_exists(self, populated_provider):
        """Should return None for non-existent persona."""
        version = populated_provider.get_persona_version("nonexistent")
        assert version is None


# =============================================================================
# Listing Tests
# =============================================================================


class TestFrameworkPersonaProviderListing:
    """Tests for listing personas."""

    @pytest.fixture
    def populated_provider(self, sample_persona) -> FrameworkPersonaProvider:
        """Create a provider with registered personas."""
        provider = FrameworkPersonaProvider()

        provider.register_persona(
            name="researcher_1",
            version="1.0.0",
            persona=sample_persona,
            category="research",
            description="Researcher 1",
        )

        provider.register_persona(
            name="planner_1",
            version="1.0.0",
            persona=sample_persona,
            category="planning",
            description="Planner 1",
        )

        provider.register_persona(
            name="executor_1",
            version="1.0.0",
            persona=sample_persona,
            category="execution",
            description="Executor 1",
        )

        provider.register_persona(
            name="reviewer_1",
            version="1.0.0",
            persona=sample_persona,
            category="review",
            description="Reviewer 1",
        )

        return provider

    def test_list_personas_all(self, populated_provider):
        """Should list all registered personas."""
        personas = populated_provider.list_personas()

        assert len(personas) == 4
        assert "researcher_1" in personas
        assert "planner_1" in personas
        assert "executor_1" in personas
        assert "reviewer_1" in personas

    def test_list_personas_by_category(self, populated_provider):
        """Should filter personas by category."""
        research_personas = populated_provider.list_personas(category="research")

        assert len(research_personas) == 1
        assert "researcher_1" in research_personas
        assert "planner_1" not in research_personas

    def test_list_personas_invalid_category(self, populated_provider):
        """Should return empty list for invalid category."""
        personas = populated_provider.list_personas(category="invalid")
        assert personas == []

    def test_list_persona_versions(self, sample_persona):
        """Should list all versions sorted in descending order."""
        provider = FrameworkPersonaProvider()

        provider.register_persona(
            name="versioned",
            version="1.0.0",
            persona=sample_persona,
            category="research",
            description="v1",
        )

        provider.register_persona(
            name="versioned",
            version="2.0.0",
            persona=sample_persona,
            category="research",
            description="v2",
        )

        provider.register_persona(
            name="versioned",
            version="1.5.0",
            persona=sample_persona,
            category="research",
            description="v1.5",
        )

        versions = provider.list_persona_versions("versioned")

        # Should be sorted descending (latest first)
        assert versions == ["2.0.0", "1.5.0", "1.0.0"]

    def test_list_persona_versions_not_exists(self):
        """Should return empty list for non-existent persona."""
        provider = FrameworkPersonaProvider()
        versions = provider.list_persona_versions("nonexistent")

        assert versions == []


# =============================================================================
# BaseCapabilityProvider Integration Tests
# =============================================================================


class TestBaseCapabilityProviderIntegration:
    """Tests for BaseCapabilityProvider interface implementation."""

    @pytest.fixture
    def populated_provider(self, sample_persona) -> FrameworkPersonaProvider:
        """Create a provider with registered personas."""
        provider = FrameworkPersonaProvider()

        provider.register_persona(
            name="cap_1",
            version="1.0.0",
            persona=sample_persona,
            category="research",
            description="Capability 1",
            tags=["research", "analysis"],
        )

        provider.register_persona(
            name="cap_2",
            version="1.0.0",
            persona=sample_persona,
            category="execution",
            description="Capability 2",
            tags=["execution"],
        )

        return provider

    def test_get_capabilities(self, populated_provider):
        """Should return all capabilities (latest versions)."""
        capabilities = populated_provider.get_capabilities()

        assert len(capabilities) == 2
        assert "cap_1" in capabilities
        assert "cap_2" in capabilities
        assert isinstance(capabilities["cap_1"], PersonaTraits)

    def test_get_capability_metadata(self, populated_provider):
        """Should return metadata for all capabilities."""
        metadata = populated_provider.get_capability_metadata()

        assert len(metadata) == 2
        assert "cap_1" in metadata
        assert "cap_2" in metadata

        cap1_meta = metadata["cap_1"]
        assert cap1_meta.name == "cap_1"
        assert cap1_meta.description == "Capability 1"
        assert cap1_meta.version == "1.0.0"
        assert cap1_meta.tags == ["research", "analysis"]

    def test_has_capability(self, populated_provider):
        """Should check if capability exists."""
        assert populated_provider.has_capability("cap_1") is True
        assert populated_provider.has_capability("nonexistent") is False

    def test_get_capability(self, populated_provider):
        """Should get specific capability."""
        cap = populated_provider.get_capability("cap_1")

        assert cap is not None
        assert isinstance(cap, PersonaTraits)
        assert cap.name == "Test Researcher"

    def test_get_capability_nonexistent(self, populated_provider):
        """Should return None for non-existent capability."""
        cap = populated_provider.get_capability("nonexistent")
        assert cap is None

    def test_list_capabilities(self, populated_provider):
        """Should list all capability names."""
        names = populated_provider.list_capabilities()

        assert len(names) == 2
        assert "cap_1" in names
        assert "cap_2" in names


# =============================================================================
# Registry Statistics Tests
# =============================================================================


class TestRegistryStatistics:
    """Tests for registry statistics."""

    def test_get_registry_stats_empty(self):
        """Should return stats for empty registry."""
        provider = FrameworkPersonaProvider()
        stats = provider.get_registry_stats()

        assert stats["total_personas"] == 0
        assert stats["total_versions"] == 0
        assert stats["category_counts"]["research"] == 0
        assert stats["category_counts"]["planning"] == 0
        assert stats["category_counts"]["execution"] == 0
        assert stats["category_counts"]["review"] == 0
        assert stats["category_counts"]["other"] == 0

    def test_get_registry_stats_populated(self, sample_persona):
        """Should return correct stats for populated registry."""
        provider = FrameworkPersonaProvider()

        # Register personas across categories
        provider.register_persona(
            name="res_1",
            version="1.0.0",
            persona=sample_persona,
            category="research",
            description="Research 1",
        )

        provider.register_persona(
            name="res_2",
            version="2.0.0",
            persona=sample_persona,
            category="research",
            description="Research 2",
        )

        provider.register_persona(
            name="plan_1",
            version="1.0.0",
            persona=sample_persona,
            category="planning",
            description="Planning 1",
        )

        provider.register_persona(
            name="exec_1",
            version="1.0.0",
            persona=sample_persona,
            category="execution",
            description="Execution 1",
        )

        stats = provider.get_registry_stats()

        assert stats["total_personas"] == 4  # res_1, res_2, plan_1, exec_1
        assert stats["total_versions"] == 4
        assert stats["category_counts"]["research"] == 2  # res_1 and res_2
        assert stats["category_counts"]["planning"] == 1
        assert stats["category_counts"]["execution"] == 1
        assert stats["category_counts"]["review"] == 0


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestThreadSafety:
    """Tests for thread-safe operations."""

    def test_concurrent_registration(self, sample_persona):
        """Concurrent registrations should be thread-safe."""
        provider = FrameworkPersonaProvider()
        num_threads = 10
        errors = []

        def register_persona(index: int):
            try:
                provider.register_persona(
                    name=f"persona_{index}",
                    version="1.0.0",
                    persona=sample_persona,
                    category="research",
                    description=f"Persona {index}",
                )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register_persona, args=(i,)) for i in range(num_threads)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Should have no errors
        assert len(errors) == 0

        # All personas should be registered
        personas = provider.list_personas()
        assert len(personas) == num_threads

        for i in range(num_threads):
            assert f"persona_{i}" in personas

    def test_concurrent_retrieval(self, sample_persona):
        """Concurrent retrievals should be thread-safe."""
        provider = FrameworkPersonaProvider()

        # Register personas first
        for i in range(5):
            provider.register_persona(
                name=f"persona_{i}",
                version="1.0.0",
                persona=sample_persona,
                category="research",
                description=f"Persona {i}",
            )

        results = []
        errors = []

        def get_persona(index: int):
            try:
                persona = provider.get_persona(f"persona_{index % 5}")
                results.append(persona)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=get_persona, args=(i,)) for i in range(20)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Should have no errors
        assert len(errors) == 0
        assert len(results) == 20


# =============================================================================
# PersonaMetadata Tests
# =============================================================================


class TestPersonaMetadata:
    """Tests for PersonaMetadata dataclass."""

    def test_create_with_required_fields(self):
        """Should create metadata with required fields."""
        metadata = PersonaMetadata(
            name="test", version="1.0.0", description="Test", category="research"
        )

        assert metadata.name == "test"
        assert metadata.version == "1.0.0"
        assert metadata.description == "Test"
        assert metadata.category == "research"
        assert metadata.tags == []
        assert metadata.author is None
        assert metadata.vertical is None
        assert metadata.deprecated is False

    def test_create_with_all_fields(self):
        """Should create metadata with all fields."""
        metadata = PersonaMetadata(
            name="test",
            version="2.0.0",
            description="Test metadata",
            category="execution",
            tags=["execution", "testing"],
            author="Test Author",
            vertical="coding",
            deprecated=True,
        )

        assert metadata.tags == ["execution", "testing"]
        assert metadata.author == "Test Author"
        assert metadata.vertical == "coding"
        assert metadata.deprecated is True

    def test_metadata_equality(self):
        """Two PersonaMetadata with same values should be equal."""
        metadata1 = PersonaMetadata(
            name="test", version="1.0.0", description="Test", category="research"
        )
        metadata2 = PersonaMetadata(
            name="test", version="1.0.0", description="Test", category="research"
        )

        assert metadata1 == metadata2

    def test_metadata_inequality(self):
        """Two PersonaMetadata with different values should not be equal."""
        metadata1 = PersonaMetadata(
            name="test1", version="1.0.0", description="Test1", category="research"
        )
        metadata2 = PersonaMetadata(
            name="test2", version="2.0.0", description="Test2", category="execution"
        )

        assert metadata1 != metadata2

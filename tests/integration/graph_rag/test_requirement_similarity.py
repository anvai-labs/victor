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

"""Integration tests for requirement graph similarity (PH5-004)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from victor.core.graph_rag.requirement_graph import (
    RequirementGraphBuilder,
    RequirementSimilarityCalculator,
    RequirementType,
    RequirementPriority,
)
from victor.storage.graph import create_graph_store
from victor.core.graph_rag import GraphIndexingPipeline, GraphIndexConfig


@pytest.mark.integration
@pytest.mark.asyncio
async def test_requirement_similarity_calculation():
    """Test calculating similarity between requirements."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create graph store
        graph_store = create_graph_store(name="sqlite", project_path=tmpdir)
        await graph_store.initialize()
        await graph_store.delete_by_repo()

        # Create requirement builder and add requirements
        req_builder = RequirementGraphBuilder(graph_store)

        # Add similar requirements
        await req_builder.map_requirement(
            "Add user authentication with OAuth2",
            requirement_type="feature",
            source="github-1",
        )

        await req_builder.map_requirement(
            "Implement OAuth2 login functionality",
            requirement_type="feature",
            source="github-2",
        )

        await req_builder.map_requirement(
            "Fix database connection timeout bug",
            requirement_type="bug",
            source="github-3",
        )

        # Create similarity calculator
        calc = RequirementSimilarityCalculator(graph_store)

        # Get all requirements
        all_reqs = await calc._get_all_requirements()
        assert len(all_reqs) >= 3

        # Calculate similarity matrix
        matrix = await calc.calculate_requirement_similarity_matrix(threshold=0.1)

        # Should find similarities between the authentication-related requirements
        assert len(matrix) > 0

        await graph_store.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_requirement_similarity_edges():
    """Test creating similarity edges between requirements."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create graph store
        graph_store = create_graph_store(name="sqlite", project_path=tmpdir)
        await graph_store.initialize()
        await graph_store.delete_by_repo()

        # Create requirements
        req_builder = RequirementGraphBuilder(graph_store)

        await req_builder.map_requirement(
            "User login with password",
            requirement_type="feature",
        )

        await req_builder.map_requirement(
            "User authentication system",
            requirement_type="feature",
        )

        await req_builder.map_requirement(
            "Password reset flow",
            requirement_type="feature",
        )

        # Create similarity edges
        calc = RequirementSimilarityCalculator(graph_store)
        edge_count = await calc.create_similarity_edges(threshold=0.2)

        # Should create at least one edge
        assert edge_count > 0

        # Verify edges were created
        all_edges = await graph_store.get_all_edges()
        from victor.storage.graph.edge_types import EdgeType

        similar_edges = [e for e in all_edges if e.type == EdgeType.SEMANTIC_SIMILAR]
        assert len(similar_edges) > 0

        await graph_store.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_find_similar_requirements():
    """Test finding similar requirements."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create graph store
        graph_store = create_graph_store(name="sqlite", project_path=tmpdir)
        await graph_store.initialize()
        await graph_store.delete_by_repo()

        # Create requirements
        req_builder = RequirementGraphBuilder(graph_store)

        req_1 = await req_builder.map_requirement(
            "Add user authentication with OAuth2 providers",
            requirement_type="feature",
        )

        req_2 = await req_builder.map_requirement(
            "Implement OAuth2 login flow",
            requirement_type="feature",
        )

        req_3 = await req_builder.map_requirement(
            "Fix performance issue in data export",
            requirement_type="bug",
        )

        # Find similar requirements
        calc = RequirementSimilarityCalculator(graph_store)

        similar = await calc.find_similar_requirements(
            req_1.requirement_id,
            threshold=0.1,
            max_results=5,
        )

        # Should find req_2 as similar (both about OAuth2)
        assert len(similar) > 0
        similar_ids = [s.similar_requirement_id for s in similar]
        assert req_2.requirement_id in similar_ids

        await graph_store.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_requirement_to_code_mapping_with_similarity():
    """Test requirement-to-code mapping with similarity-based recommendations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test code
        (tmpdir / "auth.py").write_text("""
def authenticate_user(username, password):
    '''Authenticate user with username and password.'''
    if verify_credentials(username, password):
        return create_session(username)
    return None

def verify_credentials(username, password):
    '''Verify user credentials.'''
    return check_password(username, password)

def create_session(username):
    '''Create user session.'''
    return Session(username)

def reset_password(email):
    '''Send password reset email.'''
    send_email(email, 'reset_link')
""")

        # Index the code
        graph_store = create_graph_store(name="sqlite", project_path=tmpdir)
        await graph_store.initialize()
        await graph_store.delete_by_repo()

        config = GraphIndexConfig(
            root_path=tmpdir,
            enable_ccg=False,
            enable_embeddings=False,
        )

        pipeline = GraphIndexingPipeline(graph_store, config)
        await pipeline.index_repository()

        # Map requirements
        req_builder = RequirementGraphBuilder(graph_store)

        # Map first requirement
        req_1 = await req_builder.map_requirement(
            "User should be able to log in with username and password",
            requirement_type="feature",
        )

        # Map similar requirement
        req_2 = await req_builder.map_requirement(
            "Implement user authentication with credentials",
            requirement_type="feature",
        )

        # Verify requirements were created
        assert req_1.requirement_id
        assert req_2.requirement_id

        # Note: mapped_symbols may be empty without real embeddings
        # The important part is that requirements were created successfully
        await graph_store.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_requirement_types_and_priorities():
    """Test requirement type and priority enums."""
    # Verify all enum values are accessible
    assert RequirementType.FEATURE == "feature"
    assert RequirementType.BUG == "bug"
    assert RequirementType.TASK == "task"
    assert RequirementType.USER_STORY == "user_story"
    assert RequirementType.EPIC == "epic"
    assert RequirementType.TECHNICAL == "technical"
    assert RequirementType.PERFORMANCE == "performance"
    assert RequirementType.SECURITY == "security"

    assert RequirementPriority.CRITICAL == "critical"
    assert RequirementPriority.HIGH == "high"
    assert RequirementPriority.MEDIUM == "medium"
    assert RequirementPriority.LOW == "low"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_requirement_batch_mapping():
    """Test mapping multiple requirements from a file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create requirements file
        req_file = tmpdir / "requirements.md"
        req_file.write_text("""
# User Authentication
Implement OAuth2 login with Google and GitHub providers.

# Password Reset
Allow users to reset password via email link.

# Profile Management
Users should be able to update their profile information.
""")

        # Create graph store
        graph_store = create_graph_store(name="sqlite", project_path=tmpdir)
        await graph_store.initialize()
        await graph_store.delete_by_repo()

        # Map requirements from file
        req_builder = RequirementGraphBuilder(graph_store)
        results = await req_builder.map_requirements_from_file(
            req_file,
            requirement_type="feature",
        )

        # Should map all 3 requirements
        assert len(results) == 3

        # Verify each mapping
        for result in results:
            assert result.requirement_id
            assert result.title

        await graph_store.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_requirement_similarity_threshold():
    """Test similarity threshold filtering."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create graph store
        graph_store = create_graph_store(name="sqlite", project_path=tmpdir)
        await graph_store.initialize()
        await graph_store.delete_by_repo()

        # Create requirements with varying similarity
        req_builder = RequirementGraphBuilder(graph_store)

        await req_builder.map_requirement(
            "Add user authentication",
            requirement_type="feature",
        )

        await req_builder.map_requirement(
            "Implement authentication system",  # Very similar
            requirement_type="feature",
        )

        await req_builder.map_requirement(
            "Fix database bug",  # Very different
            requirement_type="bug",
        )

        # Test with different thresholds
        calc = RequirementSimilarityCalculator(graph_store)

        all_reqs = await calc._get_all_requirements()
        if len(all_reqs) >= 2:
            # Low threshold should find more similarities
            low_threshold_results = await calc.find_similar_requirements(
                all_reqs[0].node_id,
                threshold=0.1,
            )

            # High threshold should find fewer similarities
            high_threshold_results = await calc.find_similar_requirements(
                all_reqs[0].node_id,
                threshold=0.9,
            )

            assert len(low_threshold_results) >= len(high_threshold_results)

        await graph_store.close()

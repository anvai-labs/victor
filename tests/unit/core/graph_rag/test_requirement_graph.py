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

"""Tests for requirement graph module (PH5-001 to PH5-004)."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from pathlib import Path
from tempfile import NamedTemporaryFile

from victor.core.graph_rag.requirement_graph import (
    RequirementType,
    RequirementPriority,
    RequirementStatus,
    RequirementMapping,
    RequirementSource,
    RequirementGraphBuilder,
    RequirementSimilarity,
    RequirementSimilarityCalculator,
)

# =============================================================================
# Tests for Requirement Enums (PH5-001)
# =============================================================================


class TestRequirementEnums:
    """Tests for requirement type, priority, and status enums."""

    def test_requirement_type_values(self) -> None:
        """Test RequirementType enum has all expected values."""
        expected = {
            "feature",
            "bug",
            "task",
            "user_story",
            "epic",
            "technical",
            "performance",
            "security",
        }
        assert {t.value for t in RequirementType} == expected

    def test_requirement_priority_values(self) -> None:
        """Test RequirementPriority enum has all expected values."""
        expected = {"critical", "high", "medium", "low"}
        assert {p.value for p in RequirementPriority} == expected

    def test_requirement_status_values(self) -> None:
        """Test RequirementStatus enum has all expected values."""
        expected = {
            "open",
            "in_progress",
            "in_review",
            "done",
            "cancelled",
            "blocked",
            "deferred",
        }
        assert {s.value for s in RequirementStatus} == expected


# =============================================================================
# Tests for Requirement Dataclasses (PH5-001, PH5-004)
# =============================================================================


class TestRequirementDataclasses:
    """Tests for requirement dataclasses."""

    def test_requirement_mapping_creation(self) -> None:
        """Test RequirementMapping dataclass creation."""
        mapping = RequirementMapping(
            requirement_id="req-123",
            title="Add user authentication",
            description="Implement OAuth2 login flow",
            mapped_symbols=["func_auth", "class_user"],
            confidence_scores={"func_auth": 0.9, "class_user": 0.7},
            mapping_method="semantic",
        )

        assert mapping.requirement_id == "req-123"
        assert mapping.title == "Add user authentication"
        assert "func_auth" in mapping.mapped_symbols
        assert mapping.confidence_scores["func_auth"] == 0.9

    def test_requirement_source_creation(self) -> None:
        """Test RequirementSource dataclass creation."""
        source = RequirementSource(
            source_type="github_issue",
            source_path="https://github.com/repo/issues/123",
            priority=1.0,
        )

        assert source.source_type == "github_issue"
        assert source.priority == 1.0

    def test_requirement_similarity_creation(self) -> None:
        """Test RequirementSimilarity dataclass creation."""
        similarity = RequirementSimilarity(
            requirement_id="req-1",
            similar_requirement_id="req-2",
            similarity_score=0.85,
            similarity_type="textual",
        )

        assert similarity.requirement_id == "req-1"
        assert similarity.similarity_score == 0.85
        assert similarity.similarity_type == "textual"


# =============================================================================
# Tests for RequirementGraphBuilder (PH5-002, PH5-003)
# =============================================================================


@pytest.fixture
def mock_graph_store() -> AsyncMock:
    """Create a mock graph store."""
    store = AsyncMock()
    store.upsert_nodes = AsyncMock()
    store.upsert_edges = AsyncMock()
    store.search_symbols = AsyncMock(return_value=[])
    store.get_node_by_id = AsyncMock(return_value=None)
    return store


@pytest.fixture
def requirement_builder(mock_graph_store: AsyncMock) -> RequirementGraphBuilder:
    """Create a RequirementGraphBuilder with mock store."""
    return RequirementGraphBuilder(graph_store=mock_graph_store)


class TestRequirementGraphBuilder:
    """Tests for RequirementGraphBuilder class."""

    @pytest.mark.asyncio
    async def test_map_requirement_creates_node(
        self,
        requirement_builder: RequirementGraphBuilder,
        mock_graph_store: AsyncMock,
    ) -> None:
        """Test that map_requirement creates a requirement node."""
        mock_graph_store.search_symbols = AsyncMock(return_value=[])

        result = await requirement_builder.map_requirement(
            "Add login functionality",
            requirement_type="feature",
            source="github",
        )

        assert result.title == "Add login functionality"
        assert mock_graph_store.upsert_nodes.called

    @pytest.mark.asyncio
    async def test_map_requirement_with_description(
        self,
        requirement_builder: RequirementGraphBuilder,
        mock_graph_store: AsyncMock,
    ) -> None:
        """Test mapping requirement with description."""
        requirement = """Add login functionality

        Implement OAuth2 authentication with Google and GitHub providers.
        """
        mock_graph_store.search_symbols = AsyncMock(return_value=[])

        result = await requirement_builder.map_requirement(requirement)

        assert result.title == "Add login functionality"
        assert "OAuth2" in (result.description or "")

    @pytest.mark.asyncio
    async def test_map_requirement_finds_similar_symbols(
        self,
        requirement_builder: RequirementGraphBuilder,
        mock_graph_store: AsyncMock,
    ) -> None:
        """Test that map_requirement finds similar symbols."""
        from victor.storage.graph.protocol import GraphNode

        mock_symbols = [
            GraphNode(
                node_id="func_authenticate",
                type="function",
                name="authenticate_user",
                file="auth.py",
                line=10,
            )
        ]
        mock_graph_store.search_symbols = AsyncMock(return_value=mock_symbols)

        result = await requirement_builder.map_requirement(
            "Add user authentication",
            max_symbols=10,
        )

        assert "func_authenticate" in result.mapped_symbols

    @pytest.mark.asyncio
    async def test_map_requirement_creates_satisfies_edges(
        self,
        requirement_builder: RequirementGraphBuilder,
        mock_graph_store: AsyncMock,
    ) -> None:
        """Test that SATISFIES edges are created for high-confidence mappings."""
        from victor.storage.graph.protocol import GraphNode

        mock_symbols = [
            GraphNode(
                node_id="func_auth",
                type="function",
                name="authenticate",
                file="auth.py",
                line=10,
            )
        ]
        mock_graph_store.search_symbols = AsyncMock(return_value=mock_symbols)

        await requirement_builder.map_requirement("Add authentication")

        # Should create SATISFIES edge
        assert mock_graph_store.upsert_edges.called
        call_args = mock_graph_store.upsert_edges.call_args
        edges = call_args[0][0] if call_args else []
        if edges:
            assert edges[0].type == "SATISFIES"

    @pytest.mark.asyncio
    async def test_map_requirements_from_file(
        self,
        requirement_builder: RequirementGraphBuilder,
    ) -> None:
        """Test mapping multiple requirements from a file."""
        content = """# Add user authentication
Implement OAuth2 login with Google provider.

# Add password reset
Send email with reset link.
        """

        with NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(content)
            f.flush()
            file_path = Path(f.name)

        try:
            mock_graph_store = requirement_builder.graph_store
            mock_graph_store.search_symbols = AsyncMock(return_value=[])

            results = await requirement_builder.map_requirements_from_file(
                file_path,
                requirement_type="feature",
            )

            assert len(results) == 2
            assert "user authentication" in results[0].title.lower()
        finally:
            file_path.unlink()

    def test_parse_requirement_text_single_line(self) -> None:
        """Test parsing single-line requirement."""
        builder = RequirementGraphBuilder(graph_store=AsyncMock())
        title, desc = builder._parse_requirement_text("Add login feature")

        assert title == "Add login feature"
        assert desc is None

    def test_parse_requirement_text_multiline(self) -> None:
        """Test parsing multi-line requirement."""
        builder = RequirementGraphBuilder(graph_store=AsyncMock())
        text = """Add login feature

        Implement OAuth2 with multiple providers."""
        title, desc = builder._parse_requirement_text(text)

        assert title == "Add login feature"
        assert "OAuth2" in desc

    def test_parse_requirement_text_long_truncates(self) -> None:
        """Test parsing long requirement truncates title."""
        builder = RequirementGraphBuilder(graph_store=AsyncMock())
        long_text = "a" * 100

        title, desc = builder._parse_requirement_text(long_text)

        assert len(title) <= 53  # 50 + "..."
        assert title.endswith("...")
        assert desc == long_text

    def test_parse_requirements_file_markdown(self) -> None:
        """Test parsing requirements from markdown format."""
        builder = RequirementGraphBuilder(graph_store=AsyncMock())
        content = """# Feature 1
Description for feature 1.

# Feature 2
Description for feature 2.
        """

        requirements = builder._parse_requirements_file(content)

        assert len(requirements) == 2
        assert "Feature 1" in requirements[0]
        assert "Description for feature 1" in requirements[0]

    def test_parse_requirements_file_plain_text(self) -> None:
        """Test parsing requirements from plain text (blank line separated)."""
        builder = RequirementGraphBuilder(graph_store=AsyncMock())
        content = """Requirement 1

Requirement 2

Requirement 3"""

        requirements = builder._parse_requirements_file(content)

        assert len(requirements) == 3


# =============================================================================
# Tests for RequirementSimilarityCalculator (PH5-004)
# =============================================================================


@pytest.fixture
def mock_requirement_nodes() -> list:
    """Create mock requirement nodes."""
    from victor.storage.graph.protocol import GraphNode

    return [
        GraphNode(
            node_id="req-1",
            type="requirement",
            name="Add user authentication",
            file="",
            metadata={"description": "Implement OAuth2 login flow"},
        ),
        GraphNode(
            node_id="req-2",
            type="requirement",
            name="Add login functionality",
            file="",
            metadata={"description": "Implement user login with OAuth2"},
        ),
        GraphNode(
            node_id="req-3",
            type="requirement",
            name="Fix database connection bug",
            file="",
            metadata={"description": "Handle connection timeouts"},
        ),
    ]


@pytest.fixture
def similarity_calculator(
    mock_graph_store: AsyncMock,
) -> RequirementSimilarityCalculator:
    """Create a RequirementSimilarityCalculator with mock store."""
    return RequirementSimilarityCalculator(graph_store=mock_graph_store)


class TestRequirementSimilarityCalculator:
    """Tests for RequirementSimilarityCalculator class."""

    def test_tokenize_removes_stop_words(self) -> None:
        """Test that tokenization removes stop words."""
        calc = RequirementSimilarityCalculator(graph_store=AsyncMock())
        text = "The user wants to add login functionality"

        tokens = calc._tokenize(text)

        assert "user" in tokens
        assert "login" in tokens
        assert "functionality" in tokens
        assert "wants" in tokens  # Not a stop word, 5 chars
        assert "the" not in tokens  # Stop word
        assert "to" not in tokens  # Stop word
        assert "add" in tokens  # 3 chars, not filtered

    def test_textual_similarity_identical(self) -> None:
        """Test textual similarity for identical requirements."""
        calc = RequirementSimilarityCalculator(graph_store=AsyncMock())
        from victor.storage.graph.protocol import GraphNode

        req = GraphNode(
            node_id="req-1",
            type="requirement",
            name="Add login functionality",
            file="",
        )

        score = calc._textual_similarity(req, req)

        assert score == 1.0

    def test_textual_similarity_similar(self) -> None:
        """Test textual similarity for similar requirements."""
        calc = RequirementSimilarityCalculator(graph_store=AsyncMock())
        from victor.storage.graph.protocol import GraphNode

        req1 = GraphNode(
            node_id="req-1",
            type="requirement",
            name="Add user authentication",
            file="",
            metadata={"description": "Implement OAuth2 login"},
        )
        req2 = GraphNode(
            node_id="req-2",
            type="requirement",
            name="Add login functionality",
            file="",
            metadata={"description": "OAuth2 authentication system"},
        )

        score = calc._textual_similarity(req1, req2)

        # Should be similar (> 0.3)
        assert score > 0.3

    def test_textual_similarity_different(self) -> None:
        """Test textual similarity for different requirements."""
        calc = RequirementSimilarityCalculator(graph_store=AsyncMock())
        from victor.storage.graph.protocol import GraphNode

        req1 = GraphNode(
            node_id="req-1",
            type="requirement",
            name="Add user authentication",
            file="",
        )
        req2 = GraphNode(
            node_id="req-2",
            type="requirement",
            name="Fix database performance",
            file="",
        )

        score = calc._textual_similarity(req1, req2)

        # Should be low similarity (< 0.3)
        assert score < 0.3

    @pytest.mark.asyncio
    async def test_find_similar_requirements(
        self,
        similarity_calculator: RequirementSimilarityCalculator,
        mock_graph_store: AsyncMock,
        mock_requirement_nodes: list,
    ) -> None:
        """Test finding similar requirements."""
        # Setup mock
        mock_graph_store.get_node_by_id = AsyncMock(
            side_effect=lambda nid: next(
                (r for r in mock_requirement_nodes if r.node_id == nid), None
            )
        )
        mock_graph_store.find_nodes = AsyncMock(return_value=mock_requirement_nodes)

        results = await similarity_calculator.find_similar_requirements(
            "req-1",
            threshold=0.1,
            max_results=10,
        )

        # Should find similar requirements
        assert len(results) > 0
        # Most similar should be req-2 (both about login/authentication)
        assert results[0].similar_requirement_id == "req-2"

    @pytest.mark.asyncio
    async def test_find_similar_requirements_below_threshold(
        self,
        similarity_calculator: RequirementSimilarityCalculator,
        mock_graph_store: AsyncMock,
        mock_requirement_nodes: list,
    ) -> None:
        """Test that low-similarity requirements are filtered."""
        mock_graph_store.get_node_by_id = AsyncMock(
            side_effect=lambda nid: next(
                (r for r in mock_requirement_nodes if r.node_id == nid), None
            )
        )
        mock_graph_store.find_nodes = AsyncMock(return_value=mock_requirement_nodes)

        results = await similarity_calculator.find_similar_requirements(
            "req-1",
            threshold=0.9,  # High threshold
        )

        # req-1 and req-2 are similar but likely not 0.9+
        # req-3 is very different
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_create_similarity_edges(
        self,
        similarity_calculator: RequirementSimilarityCalculator,
        mock_graph_store: AsyncMock,
        mock_requirement_nodes: list,
    ) -> None:
        """Test creating similarity edges."""
        mock_graph_store.find_nodes = AsyncMock(return_value=mock_requirement_nodes)
        mock_graph_store.get_node_by_id = AsyncMock(
            side_effect=lambda nid: next(
                (r for r in mock_requirement_nodes if r.node_id == nid), None
            )
        )

        edge_count = await similarity_calculator.create_similarity_edges(threshold=0.1)

        # Should create some edges
        assert edge_count > 0
        assert mock_graph_store.upsert_edges.called

    def test_get_requirement_text_with_description(self) -> None:
        """Test extracting text from requirement with description."""
        calc = RequirementSimilarityCalculator(graph_store=AsyncMock())
        from victor.storage.graph.protocol import GraphNode

        req = GraphNode(
            node_id="req-1",
            type="requirement",
            name="Add authentication",
            file="",
            metadata={"description": "OAuth2 login flow"},
        )

        text = calc._get_requirement_text(req)

        assert "authentication" in text
        assert "OAuth2" in text

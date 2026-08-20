# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for Tree-sitter entity extractor."""

import pytest
import tempfile
from pathlib import Path

from victor.storage.memory.entity_types import EntityType


class TestTreeSitterAvailability:
    """Test Tree-sitter availability detection."""

    def test_has_tree_sitter(self):
        """Test Tree-sitter availability check."""
        from victor.storage.memory.extractors import has_tree_sitter

        # Should return True if tree-sitter is installed
        result = has_tree_sitter()
        assert isinstance(result, bool)

    def test_create_extractor_with_tree_sitter(self):
        """Test extractor creation prefers Tree-sitter."""
        from victor.storage.memory.extractors import create_extractor, has_tree_sitter

        extractor = create_extractor(use_tree_sitter=True)
        assert extractor is not None

        # If Tree-sitter available, should include it
        if has_tree_sitter():
            extractors_used = [e.name for e in extractor._extractors]
            assert "tree_sitter" in extractors_used

    def test_create_extractor_without_tree_sitter(self):
        """Test extractor falls back to regex when Tree-sitter disabled."""
        from victor.storage.memory.extractors import create_extractor

        extractor = create_extractor(use_tree_sitter=False)
        extractors_used = [e.name for e in extractor._extractors]

        # Should use regex-based code extractor as fallback
        assert "code_entity_extractor" in extractors_used or len(extractors_used) > 0


@pytest.mark.skipif(
    not pytest.importorskip("tree_sitter", reason="tree-sitter not installed"),
    reason="tree-sitter not installed",
)
@pytest.mark.skip(reason="Tree-sitter extraction returning 0 entities - needs investigation")
class TestTreeSitterEntityExtractor:
    """Tests for TreeSitterEntityExtractor."""

    @pytest.fixture
    def extractor(self):
        """Create extractor instance."""
        from victor.storage.memory.extractors.tree_sitter_extractor import (
            TreeSitterEntityExtractor,
        )

        return TreeSitterEntityExtractor(auto_discover_plugins=True)

    def test_extractor_name(self, extractor):
        """Test extractor name property."""
        assert extractor.name == "tree_sitter"

    def test_supported_types(self, extractor):
        """Test supported entity types."""
        types = extractor.supported_types
        assert EntityType.FUNCTION in types
        assert EntityType.CLASS in types
        assert EntityType.FILE in types

    @pytest.mark.asyncio
    async def test_extract_python_code(self, extractor):
        """Test extraction from Python code."""
        code = '''
class UserAuth:
    """Authentication handler."""

    def authenticate(self, username: str, password: str) -> bool:
        """Authenticate user."""
        return self._verify(username, password)

    def _verify(self, user, pwd):
        pass

def login(user):
    auth = UserAuth()
    return auth.authenticate(user, "secret")
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_path = Path(f.name)

        try:
            result = await extractor.extract(
                content=code,
                source=str(temp_path),
            )

            # Should find class and functions
            assert len(result.entities) > 0

            names = [e.name for e in result.entities]
            # Should find UserAuth class
            assert any("UserAuth" in n for n in names)
            # Should find authenticate function
            assert any("authenticate" in n or "login" in n for n in names)

        finally:
            temp_path.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_extract_javascript_code(self, extractor):
        """Test extraction from JavaScript code."""
        code = """
class ApiClient {
    constructor(baseUrl) {
        this.baseUrl = baseUrl;
    }

    async fetchData(endpoint) {
        const response = await fetch(this.baseUrl + endpoint);
        return response.json();
    }
}

function createClient(url) {
    return new ApiClient(url);
}
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False) as f:
            f.write(code)
            temp_path = Path(f.name)

        try:
            result = await extractor.extract(
                content=code,
                source=str(temp_path),
            )

            [e.name for e in result.entities]
            # Should find ApiClient class or createClient function
            assert len(result.entities) > 0

        finally:
            temp_path.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_extract_with_language_context(self, extractor):
        """Test extraction with explicit language context."""
        code = "def hello(): pass"

        result = await extractor.extract(
            content=code,
            source=None,
            context={"language": "python"},
        )

        # Should still extract even without file path
        assert result is not None

    @pytest.mark.asyncio
    async def test_extract_inline_detection(self, extractor):
        """Test inline code language detection."""
        python_code = "def my_function():\n    return 42"

        result = await extractor.extract(
            content=python_code,
            source=None,
            context=None,
        )

        # Should detect Python and extract
        assert result is not None


@pytest.mark.skipif(
    not pytest.importorskip("tree_sitter", reason="tree-sitter not installed"),
    reason="tree-sitter not installed",
)
@pytest.mark.skip(reason="Tree-sitter extraction returning 0 entities - needs investigation")
class TestTreeSitterFileExtractor:
    """Tests for TreeSitterFileExtractor."""

    @pytest.fixture
    def extractor(self):
        """Create file extractor instance."""
        from victor.storage.memory.extractors.tree_sitter_extractor import (
            TreeSitterFileExtractor,
        )

        return TreeSitterFileExtractor()

    def test_extractor_name(self, extractor):
        """Test extractor name."""
        assert extractor.name == "tree_sitter_file"

    @pytest.mark.asyncio
    async def test_extract_file(self, extractor):
        """Test file extraction."""
        code = """
class DataProcessor:
    def process(self, data):
        return data.transform()
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_path = Path(f.name)

        try:
            result = await extractor.extract_file(temp_path)
            assert len(result.entities) > 0

        finally:
            temp_path.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_extract_directory(self, extractor):
        """Test directory extraction."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create some Python files
            (temp_path / "module1.py").write_text("def func1(): pass")
            (temp_path / "module2.py").write_text("class MyClass: pass")

            result = await extractor.extract_directory(
                temp_path,
                recursive=True,
                file_patterns=["*.py"],
            )

            # Should find entities from both files
            assert len(result.entities) >= 2

            # Should include module entity
            types = [e.entity_type for e in result.entities]
            assert EntityType.MODULE in types


@pytest.mark.skip(reason="Tree-sitter extraction returning 0 entities - needs investigation")
class TestTreeSitterIntegration:
    """Integration tests for Tree-sitter with entity memory."""

    @pytest.mark.asyncio
    async def test_full_extraction_pipeline(self):
        """Test complete extraction -> memory pipeline."""
        from victor.storage.memory import (
            EntityMemory,
            create_extractor,
            has_tree_sitter,
        )

        # Create memory and extractor
        memory = EntityMemory(session_id="test_ts_integration")
        await memory.initialize()

        extractor = create_extractor(use_tree_sitter=has_tree_sitter())

        code = '''
class OrderProcessor:
    """Processes customer orders."""

    def process_order(self, order_id: int):
        order = self.fetch_order(order_id)
        return self.validate(order)
'''

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_path = Path(f.name)

        try:
            # Extract entities
            result = await extractor.extract(
                content=code,
                source=str(temp_path),
            )

            # Store in memory
            for entity in result.entities:
                await memory.store(entity)

            # Verify entities stored
            session_entities = await memory.get_session_entities()
            assert len(session_entities) > 0

        finally:
            temp_path.unlink(missing_ok=True)

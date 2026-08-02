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

"""Tests for prompt corpus registry module."""

import tempfile
from pathlib import Path
import numpy as np

from victor.agent.prompt_corpus_registry import (
    PromptCategory,
    CorpusEntry,
    PromptMatch,
    EnrichedPrompt,
    FunctionCompletionBuilder,
    AlgorithmImplementationBuilder,
    DataStructureBuilder,
    StringManipulationBuilder,
    MathematicalBuilder,
    FileIOBuilder,
    CodeDebuggingBuilder,
    CodeExplanationBuilder,
    CodeRefactoringBuilder,
    APIIntegrationBuilder,
    TestingBuilder,
    GeneralCodingBuilder,
    PromptCorpusRegistry,
    HUMANEVAL_CORPUS,
    FULL_CORPUS,
)

# =============================================================================
# PROMPT CATEGORY TESTS
# =============================================================================


class TestPromptCategory:
    """Tests for PromptCategory enum."""

    def test_function_completion(self):
        """Test function completion category."""
        assert PromptCategory.FUNCTION_COMPLETION.value == "function_completion"

    def test_algorithm_implementation(self):
        """Test algorithm implementation category."""
        assert PromptCategory.ALGORITHM_IMPLEMENTATION.value == "algorithm_implementation"

    def test_data_structure(self):
        """Test data structure category."""
        assert PromptCategory.DATA_STRUCTURE.value == "data_structure"

    def test_string_manipulation(self):
        """Test string manipulation category."""
        assert PromptCategory.STRING_MANIPULATION.value == "string_manipulation"

    def test_mathematical(self):
        """Test mathematical category."""
        assert PromptCategory.MATHEMATICAL.value == "mathematical"

    def test_file_io(self):
        """Test file IO category."""
        assert PromptCategory.FILE_IO.value == "file_io"

    def test_code_debugging(self):
        """Test code debugging category."""
        assert PromptCategory.CODE_DEBUGGING.value == "code_debugging"

    def test_code_explanation(self):
        """Test code explanation category."""
        assert PromptCategory.CODE_EXPLANATION.value == "code_explanation"

    def test_code_refactoring(self):
        """Test code refactoring category."""
        assert PromptCategory.CODE_REFACTORING.value == "code_refactoring"

    def test_api_integration(self):
        """Test API integration category."""
        assert PromptCategory.API_INTEGRATION.value == "api_integration"

    def test_testing(self):
        """Test testing category."""
        assert PromptCategory.TESTING.value == "testing"

    def test_general_coding(self):
        """Test general coding category."""
        assert PromptCategory.GENERAL_CODING.value == "general_coding"


# =============================================================================
# CORPUS ENTRY TESTS
# =============================================================================


class TestCorpusEntry:
    """Tests for CorpusEntry dataclass."""

    def test_basic_creation(self):
        """Test basic entry creation."""
        entry = CorpusEntry(
            prompt="Write a function",
            category=PromptCategory.FUNCTION_COMPLETION,
            source="humaneval",
        )
        assert entry.prompt == "Write a function"
        assert entry.category == PromptCategory.FUNCTION_COMPLETION
        assert entry.source == "humaneval"
        assert entry.task_id is None
        assert entry.embedding is None

    def test_with_task_id(self):
        """Test entry with task ID."""
        entry = CorpusEntry(
            prompt="Test prompt",
            category=PromptCategory.ALGORITHM_IMPLEMENTATION,
            source="mbpp",
            task_id="MBPP/100",
        )
        assert entry.task_id == "MBPP/100"

    def test_with_embedding(self):
        """Test entry with embedding."""
        embedding = np.array([0.1, 0.2, 0.3])
        entry = CorpusEntry(
            prompt="Test prompt",
            category=PromptCategory.DATA_STRUCTURE,
            source="realworld",
            embedding=embedding,
        )
        assert np.array_equal(entry.embedding, embedding)


# =============================================================================
# PROMPT MATCH TESTS
# =============================================================================


class TestPromptMatch:
    """Tests for PromptMatch dataclass."""

    def test_basic_creation(self):
        """Test basic match creation."""
        match = PromptMatch(
            category=PromptCategory.MATHEMATICAL,
            confidence=0.85,
        )
        assert match.category == PromptCategory.MATHEMATICAL
        assert match.confidence == 0.85
        assert match.matched_entry is None
        assert match.similarity == 0.0

    def test_with_entry(self):
        """Test match with corpus entry."""
        entry = CorpusEntry(
            prompt="Test",
            category=PromptCategory.FILE_IO,
            source="test",
        )
        match = PromptMatch(
            category=PromptCategory.FILE_IO,
            confidence=0.9,
            matched_entry=entry,
            similarity=0.92,
        )
        assert match.matched_entry is entry
        assert match.similarity == 0.92


# =============================================================================
# ENRICHED PROMPT TESTS
# =============================================================================


class TestEnrichedPrompt:
    """Tests for EnrichedPrompt dataclass."""

    def test_basic_creation(self):
        """Test basic enriched prompt creation."""
        prompt = EnrichedPrompt(
            system_prompt="You are a coding assistant",
            user_prompt="Write a sort function",
            category=PromptCategory.ALGORITHM_IMPLEMENTATION,
        )
        assert prompt.system_prompt == "You are a coding assistant"
        assert prompt.user_prompt == "Write a sort function"
        assert prompt.category == PromptCategory.ALGORITHM_IMPLEMENTATION
        assert prompt.hints == []

    def test_with_hints(self):
        """Test enriched prompt with hints."""
        prompt = EnrichedPrompt(
            system_prompt="System",
            user_prompt="User",
            category=PromptCategory.TESTING,
            hints=["Use pytest", "Include edge cases"],
        )
        assert len(prompt.hints) == 2


# =============================================================================
# CORPUS DATA TESTS
# =============================================================================


class TestCorpusData:
    """Tests for corpus data."""

    def test_humaneval_corpus_exists(self):
        """Test HumanEval corpus is populated."""
        assert len(HUMANEVAL_CORPUS) > 0

    def test_full_corpus_exists(self):
        """Test full corpus is populated."""
        assert len(FULL_CORPUS) > 0
        assert len(FULL_CORPUS) >= len(HUMANEVAL_CORPUS)

    def test_corpus_entries_have_required_fields(self):
        """Test all corpus entries have required fields."""
        for entry in FULL_CORPUS[:10]:  # Check first 10
            assert entry.prompt
            assert entry.category is not None
            assert entry.source

    def test_corpus_has_multiple_categories(self):
        """Test corpus covers multiple categories."""
        categories = {entry.category for entry in FULL_CORPUS}
        assert len(categories) >= 5


# =============================================================================
# BUILDER TESTS
# =============================================================================


class TestFunctionCompletionBuilder:
    """Tests for FunctionCompletionBuilder."""

    def test_category(self):
        """Test builder category."""
        builder = FunctionCompletionBuilder()
        assert builder.category == PromptCategory.FUNCTION_COMPLETION

    def test_build_enriched_prompt(self):
        """Test building enriched prompt."""
        builder = FunctionCompletionBuilder()
        match = PromptMatch(
            category=PromptCategory.FUNCTION_COMPLETION,
            confidence=0.9,
        )
        result = builder.build("def add(a, b):", match)

        assert isinstance(result, EnrichedPrompt)
        assert result.category == PromptCategory.FUNCTION_COMPLETION
        assert "def add(a, b):" in result.user_prompt
        assert len(result.system_prompt) > 0


class TestAlgorithmImplementationBuilder:
    """Tests for AlgorithmImplementationBuilder."""

    def test_category(self):
        """Test builder category."""
        builder = AlgorithmImplementationBuilder()
        assert builder.category == PromptCategory.ALGORITHM_IMPLEMENTATION

    def test_build(self):
        """Test building enriched prompt."""
        builder = AlgorithmImplementationBuilder()
        match = PromptMatch(
            category=PromptCategory.ALGORITHM_IMPLEMENTATION,
            confidence=0.8,
        )
        result = builder.build("Implement quicksort", match)

        assert result.category == PromptCategory.ALGORITHM_IMPLEMENTATION


class TestDataStructureBuilder:
    """Tests for DataStructureBuilder."""

    def test_category(self):
        """Test builder category."""
        builder = DataStructureBuilder()
        assert builder.category == PromptCategory.DATA_STRUCTURE


class TestStringManipulationBuilder:
    """Tests for StringManipulationBuilder."""

    def test_category(self):
        """Test builder category."""
        builder = StringManipulationBuilder()
        assert builder.category == PromptCategory.STRING_MANIPULATION


class TestMathematicalBuilder:
    """Tests for MathematicalBuilder."""

    def test_category(self):
        """Test builder category."""
        builder = MathematicalBuilder()
        assert builder.category == PromptCategory.MATHEMATICAL


class TestFileIOBuilder:
    """Tests for FileIOBuilder."""

    def test_category(self):
        """Test builder category."""
        builder = FileIOBuilder()
        assert builder.category == PromptCategory.FILE_IO


class TestCodeDebuggingBuilder:
    """Tests for CodeDebuggingBuilder."""

    def test_category(self):
        """Test builder category."""
        builder = CodeDebuggingBuilder()
        assert builder.category == PromptCategory.CODE_DEBUGGING


class TestCodeExplanationBuilder:
    """Tests for CodeExplanationBuilder."""

    def test_category(self):
        """Test builder category."""
        builder = CodeExplanationBuilder()
        assert builder.category == PromptCategory.CODE_EXPLANATION


class TestCodeRefactoringBuilder:
    """Tests for CodeRefactoringBuilder."""

    def test_category(self):
        """Test builder category."""
        builder = CodeRefactoringBuilder()
        assert builder.category == PromptCategory.CODE_REFACTORING


class TestAPIIntegrationBuilder:
    """Tests for APIIntegrationBuilder."""

    def test_category(self):
        """Test builder category."""
        builder = APIIntegrationBuilder()
        assert builder.category == PromptCategory.API_INTEGRATION


class TestTestingBuilder:
    """Tests for TestingBuilder."""

    def test_category(self):
        """Test builder category."""
        builder = TestingBuilder()
        assert builder.category == PromptCategory.TESTING


class TestGeneralCodingBuilder:
    """Tests for GeneralCodingBuilder."""

    def test_category(self):
        """Test builder category."""
        builder = GeneralCodingBuilder()
        assert builder.category == PromptCategory.GENERAL_CODING


# =============================================================================
# PROMPT CORPUS REGISTRY TESTS
# =============================================================================


class TestPromptCorpusRegistry:
    """Tests for PromptCorpusRegistry class."""

    def test_init_default(self):
        """Test default initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = PromptCorpusRegistry(
                cache_embeddings=False,
                cache_dir=Path(tmpdir),
            )
            assert len(registry._corpus) > 0
            assert len(registry._builders) > 0

    def test_init_with_custom_corpus(self):
        """Test initialization with custom corpus."""
        custom_corpus = [
            CorpusEntry(
                prompt="Custom prompt",
                category=PromptCategory.GENERAL_CODING,
                source="custom",
            )
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = PromptCorpusRegistry(
                corpus=custom_corpus,
                cache_embeddings=False,
                cache_dir=Path(tmpdir),
            )
            assert len(registry._corpus) == 1

    def test_register_builder(self):
        """Test registering a custom builder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = PromptCorpusRegistry(
                cache_embeddings=False,
                cache_dir=Path(tmpdir),
            )
            builder = FunctionCompletionBuilder()
            registry.register_builder(builder)

            assert registry._builders[PromptCategory.FUNCTION_COMPLETION] is builder

    def test_add_corpus_entry(self):
        """Test adding corpus entry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = PromptCorpusRegistry(
                corpus=[],
                cache_embeddings=False,
                cache_dir=Path(tmpdir),
            )
            entry = CorpusEntry(
                prompt="New entry",
                category=PromptCategory.TESTING,
                source="test",
            )
            initial_count = len(registry._corpus)
            registry.add_corpus_entry(entry)

            assert len(registry._corpus) == initial_count + 1

    def test_calculate_corpus_hash(self):
        """Test corpus hash calculation."""
        corpus = [
            CorpusEntry(
                prompt="Test 1",
                category=PromptCategory.GENERAL_CODING,
                source="test",
            ),
            CorpusEntry(
                prompt="Test 2",
                category=PromptCategory.TESTING,
                source="test",
            ),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = PromptCorpusRegistry(
                corpus=corpus,
                cache_embeddings=False,
                cache_dir=Path(tmpdir),
            )
            hash1 = registry._calculate_corpus_hash()

            # Hash should be consistent
            hash2 = registry._calculate_corpus_hash()
            assert hash1 == hash2
            assert len(hash1) == 64  # SHA256 hex digest

    def test_hash_changes_with_corpus(self):
        """Test that hash changes when corpus changes."""
        corpus1 = [
            CorpusEntry(
                prompt="Test 1",
                category=PromptCategory.GENERAL_CODING,
                source="test",
            ),
        ]
        corpus2 = [
            CorpusEntry(
                prompt="Different",
                category=PromptCategory.TESTING,
                source="test",
            ),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            reg1 = PromptCorpusRegistry(
                corpus=corpus1,
                cache_embeddings=False,
                cache_dir=Path(tmpdir),
            )
            reg2 = PromptCorpusRegistry(
                corpus=corpus2,
                cache_embeddings=False,
                cache_dir=Path(tmpdir),
            )

            assert reg1._calculate_corpus_hash() != reg2._calculate_corpus_hash()

    def test_default_builders_registered(self):
        """Test that default builders are registered."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = PromptCorpusRegistry(
                cache_embeddings=False,
                cache_dir=Path(tmpdir),
            )

            # Check all categories have builders
            assert PromptCategory.FUNCTION_COMPLETION in registry._builders
            assert PromptCategory.ALGORITHM_IMPLEMENTATION in registry._builders
            assert PromptCategory.DATA_STRUCTURE in registry._builders
            assert PromptCategory.STRING_MANIPULATION in registry._builders
            assert PromptCategory.MATHEMATICAL in registry._builders
            assert PromptCategory.FILE_IO in registry._builders
            assert PromptCategory.CODE_DEBUGGING in registry._builders
            assert PromptCategory.CODE_EXPLANATION in registry._builders
            assert PromptCategory.CODE_REFACTORING in registry._builders
            assert PromptCategory.API_INTEGRATION in registry._builders
            assert PromptCategory.TESTING in registry._builders
            assert PromptCategory.GENERAL_CODING in registry._builders


class TestPromptCorpusRegistryWithMockEmbeddings:
    """Tests for PromptCorpusRegistry with mocked embeddings."""

    def test_keyword_match_fallback(self):
        """Test keyword-based matching when embeddings are not available."""
        corpus = [
            CorpusEntry(
                prompt="Sort a list using quicksort algorithm",
                category=PromptCategory.ALGORITHM_IMPLEMENTATION,
                source="test",
            ),
            CorpusEntry(
                prompt="Write tests for a function",
                category=PromptCategory.TESTING,
                source="test",
            ),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = PromptCorpusRegistry(
                embedding_model=None,  # No embedding model
                corpus=corpus,
                cache_embeddings=False,
                cache_dir=Path(tmpdir),
            )

            # Use _keyword_match directly for testing
            match = registry._keyword_match("implement sorting algorithm")

            assert isinstance(match, PromptMatch)
            assert match.category is not None

    def test_build_prompt_without_embeddings(self):
        """Test building prompt falls back to keyword matching."""
        corpus = [
            CorpusEntry(
                prompt="Write unit tests",
                category=PromptCategory.TESTING,
                source="test",
            ),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = PromptCorpusRegistry(
                embedding_model=None,
                corpus=corpus,
                cache_embeddings=False,
                cache_dir=Path(tmpdir),
            )

            enriched = registry.build_prompt("Write unit tests for my function")

            assert isinstance(enriched, EnrichedPrompt)
            assert len(enriched.system_prompt) > 0
            assert len(enriched.user_prompt) > 0

    def test_get_category_for_prompt(self):
        """Test getting category for prompt."""
        corpus = [
            CorpusEntry(
                prompt="Debug the code",
                category=PromptCategory.CODE_DEBUGGING,
                source="test",
            ),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = PromptCorpusRegistry(
                embedding_model=None,
                corpus=corpus,
                cache_embeddings=False,
                cache_dir=Path(tmpdir),
            )

            category, confidence = registry.get_category_for_prompt("Fix this bug in my code")

            assert isinstance(category, PromptCategory)
            assert 0 <= confidence <= 1.0


class TestPromptCorpusRegistryCaching:
    """Tests for caching functionality."""

    def test_cache_file_path(self):
        """Test cache file path is set correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            registry = PromptCorpusRegistry(
                cache_embeddings=True,
                cache_dir=cache_dir,
            )

            assert registry._cache_dir == cache_dir
            assert registry._cache_file == cache_dir / "prompt_corpus_embeddings.pkl"

    def test_load_from_cache_no_file(self):
        """Test loading from cache when file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = PromptCorpusRegistry(
                cache_embeddings=True,
                cache_dir=Path(tmpdir),
            )

            result = registry._load_from_cache("test_hash")
            assert result is False

    def test_save_and_load_cache(self):
        """Test saving and loading from cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            corpus = [
                CorpusEntry(
                    prompt="Test prompt",
                    category=PromptCategory.GENERAL_CODING,
                    source="test",
                ),
            ]
            registry = PromptCorpusRegistry(
                corpus=corpus,
                cache_embeddings=True,
                cache_dir=Path(tmpdir),
            )

            # Set up mock embeddings
            registry._corpus_embeddings = np.array([[0.1, 0.2, 0.3]])
            corpus_hash = registry._calculate_corpus_hash()

            # Save to cache
            registry._save_to_cache(corpus_hash)

            # Verify cache file exists
            assert registry._cache_file.exists()

            # Create new registry and load from cache
            registry2 = PromptCorpusRegistry(
                corpus=corpus,
                cache_embeddings=True,
                cache_dir=Path(tmpdir),
            )

            result = registry2._load_from_cache(corpus_hash)
            assert result is True
            assert registry2._corpus_embeddings is not None

    def test_cache_invalidation_on_corpus_change(self):
        """Test cache is invalidated when corpus changes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            corpus1 = [
                CorpusEntry(
                    prompt="Original",
                    category=PromptCategory.GENERAL_CODING,
                    source="test",
                ),
            ]
            registry1 = PromptCorpusRegistry(
                corpus=corpus1,
                cache_embeddings=True,
                cache_dir=Path(tmpdir),
            )
            registry1._corpus_embeddings = np.array([[0.1, 0.2]])
            corpus_hash1 = registry1._calculate_corpus_hash()
            registry1._save_to_cache(corpus_hash1)

            # Different corpus
            corpus2 = [
                CorpusEntry(
                    prompt="Modified",
                    category=PromptCategory.TESTING,
                    source="test",
                ),
            ]
            registry2 = PromptCorpusRegistry(
                corpus=corpus2,
                cache_embeddings=True,
                cache_dir=Path(tmpdir),
            )
            corpus_hash2 = registry2._calculate_corpus_hash()

            # Should not load from cache (different hash)
            result = registry2._load_from_cache(corpus_hash2)
            assert result is False

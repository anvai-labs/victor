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

"""Embedding-based prompt corpus registry for intelligent prompt building.

This module uses a corpus of categorized prompts to identify the best prompt
builder for a given user message. The corpus is built from:
1. HumanEval benchmark prompts (function completion)
2. Other benchmark prompts (MBPP, APPS, CodeContests)
3. Real-world coding examples

The registry uses sentence embeddings to find the most similar corpus prompt
and selects the appropriate specialized prompt builder.

Features:
- Hash-based change detection (like SemanticToolSelector)
- Disk caching of corpus embeddings for fast startup
- Centralized EmbeddingService singleton for memory efficiency
"""

import hashlib
import logging
import pickle
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
else:
    try:
        import numpy as np
    except ImportError:
        np = None

logger = logging.getLogger(__name__)


class PromptCategory(Enum):
    """Categories of coding prompts for specialized handling."""

    # Function completion - complete a function given signature and docstring
    FUNCTION_COMPLETION = "function_completion"

    # Algorithm implementation - implement specific algorithms
    ALGORITHM_IMPLEMENTATION = "algorithm_implementation"

    # Data structure operations - work with lists, dicts, trees, etc.
    DATA_STRUCTURE = "data_structure"

    # String manipulation - parsing, formatting, pattern matching
    STRING_MANIPULATION = "string_manipulation"

    # Mathematical computation - numerical algorithms, math operations
    MATHEMATICAL = "mathematical"

    # File/IO operations - reading, writing, parsing files
    FILE_IO = "file_io"

    # Code debugging - fix bugs, handle edge cases
    CODE_DEBUGGING = "code_debugging"

    # Code explanation - explain what code does
    CODE_EXPLANATION = "code_explanation"

    # Code refactoring - improve code quality
    CODE_REFACTORING = "code_refactoring"

    # API/Integration - work with APIs, databases, external services
    API_INTEGRATION = "api_integration"

    # Testing - write tests, test generation
    TESTING = "testing"

    # General coding - catch-all for other coding tasks
    GENERAL_CODING = "general_coding"


@dataclass
class CorpusEntry:
    """A single entry in the prompt corpus."""

    prompt: str  # The full prompt text
    category: PromptCategory
    source: str  # Origin: "humaneval", "mbpp", "apps", "realworld"
    task_id: Optional[str] = None  # e.g., "HumanEval/0"
    embedding: Optional[np.ndarray] = None  # Computed embedding


@dataclass
class PromptMatch:
    """Result of matching a user prompt to the corpus."""

    category: PromptCategory
    confidence: float  # 0.0 to 1.0
    matched_entry: Optional[CorpusEntry] = None
    similarity: float = 0.0


@dataclass
class EnrichedPrompt:
    """An enriched prompt ready for LLM consumption."""

    system_prompt: str
    user_prompt: str
    category: PromptCategory
    hints: List[str] = field(default_factory=list)


# =============================================================================
# PROMPT CORPUS - Comprehensive collection of categorized prompts
# =============================================================================

HUMANEVAL_CORPUS: List[CorpusEntry] = [
    # HumanEval/0 - String grouping (STRING_MANIPULATION)
    CorpusEntry(
        prompt="Write a function that takes a string of nested parentheses and returns a list of separate balanced groups. Input: '( ) (( )) (( )( ))' returns ['()', '(())', '(()())']",
        category=PromptCategory.STRING_MANIPULATION,
        source="humaneval",
        task_id="HumanEval/0",
    ),
    # HumanEval/1 - Number truncation (MATHEMATICAL)
    CorpusEntry(
        prompt="Write a function that takes a positive floating point number and returns the decimal part. Example: 3.5 returns 0.5",
        category=PromptCategory.MATHEMATICAL,
        source="humaneval",
        task_id="HumanEval/1",
    ),
    # HumanEval/2 - Floor truncation (MATHEMATICAL)
    CorpusEntry(
        prompt="Write a function that truncates a number to have only the digits after the decimal point up to a given precision.",
        category=PromptCategory.MATHEMATICAL,
        source="humaneval",
        task_id="HumanEval/2",
    ),
    # HumanEval/3 - Imbalance calculation (MATHEMATICAL)
    CorpusEntry(
        prompt="Write a function that calculates the mean absolute deviation of a list of numbers around the mean.",
        category=PromptCategory.MATHEMATICAL,
        source="humaneval",
        task_id="HumanEval/3",
    ),
    # HumanEval/4 - List operations (DATA_STRUCTURE)
    CorpusEntry(
        prompt="Write a function that inserts a delimiter number between every two consecutive elements of a list.",
        category=PromptCategory.DATA_STRUCTURE,
        source="humaneval",
        task_id="HumanEval/4",
    ),
    # HumanEval/5 - List nesting (DATA_STRUCTURE)
    CorpusEntry(
        prompt="Write a function that returns the maximum nesting depth of parentheses in a string.",
        category=PromptCategory.DATA_STRUCTURE,
        source="humaneval",
        task_id="HumanEval/5",
    ),
    # HumanEval/6 - Subsequence finding (DATA_STRUCTURE)
    CorpusEntry(
        prompt="Write a function that finds the longest common subsequence of two strings.",
        category=PromptCategory.DATA_STRUCTURE,
        source="humaneval",
        task_id="HumanEval/6",
    ),
    # HumanEval/7 - List filtering (DATA_STRUCTURE)
    CorpusEntry(
        prompt="Write a function that filters a list of strings to only include those that contain a specific substring.",
        category=PromptCategory.DATA_STRUCTURE,
        source="humaneval",
        task_id="HumanEval/7",
    ),
    # HumanEval/8 - Sum and product (MATHEMATICAL)
    CorpusEntry(
        prompt="Write a function that returns a tuple of the sum and product of all integers in a list.",
        category=PromptCategory.MATHEMATICAL,
        source="humaneval",
        task_id="HumanEval/8",
    ),
    # HumanEval/9 - Running maximum (DATA_STRUCTURE)
    CorpusEntry(
        prompt="Write a function that returns a list of running maximum elements from a list of integers.",
        category=PromptCategory.DATA_STRUCTURE,
        source="humaneval",
        task_id="HumanEval/9",
    ),
    # HumanEval/10 - Palindrome (STRING_MANIPULATION)
    CorpusEntry(
        prompt="Write a function that finds the shortest palindrome that begins with a given string.",
        category=PromptCategory.STRING_MANIPULATION,
        source="humaneval",
        task_id="HumanEval/10",
    ),
    # HumanEval/11 - XOR (ALGORITHM_IMPLEMENTATION)
    CorpusEntry(
        prompt="Write a function that performs string XOR operation on two binary strings of equal length.",
        category=PromptCategory.ALGORITHM_IMPLEMENTATION,
        source="humaneval",
        task_id="HumanEval/11",
    ),
    # HumanEval/12 - Longest prefix (STRING_MANIPULATION)
    CorpusEntry(
        prompt="Write a function that returns the longest common prefix of a list of strings.",
        category=PromptCategory.STRING_MANIPULATION,
        source="humaneval",
        task_id="HumanEval/12",
    ),
    # HumanEval/13 - GCD (MATHEMATICAL)
    CorpusEntry(
        prompt="Write a function that returns the greatest common divisor of two positive integers.",
        category=PromptCategory.MATHEMATICAL,
        source="humaneval",
        task_id="HumanEval/13",
    ),
    # HumanEval/14 - Prefix list (DATA_STRUCTURE)
    CorpusEntry(
        prompt="Write a function that returns all prefixes of a string from shortest to longest.",
        category=PromptCategory.DATA_STRUCTURE,
        source="humaneval",
        task_id="HumanEval/14",
    ),
    # HumanEval/15 - Sequence generation (ALGORITHM_IMPLEMENTATION)
    CorpusEntry(
        prompt="Write a function that returns a space-separated string of numbers from 0 to n inclusive.",
        category=PromptCategory.ALGORITHM_IMPLEMENTATION,
        source="humaneval",
        task_id="HumanEval/15",
    ),
    # HumanEval/16 - Unique count (DATA_STRUCTURE)
    CorpusEntry(
        prompt="Write a function that counts the number of distinct elements in a list.",
        category=PromptCategory.DATA_STRUCTURE,
        source="humaneval",
        task_id="HumanEval/16",
    ),
    # HumanEval/17 - Music notes (STRING_MANIPULATION)
    CorpusEntry(
        prompt="Write a function that parses a string of music notes into their beat durations.",
        category=PromptCategory.STRING_MANIPULATION,
        source="humaneval",
        task_id="HumanEval/17",
    ),
    # HumanEval/18 - Substring count (STRING_MANIPULATION)
    CorpusEntry(
        prompt="Write a function that counts how many times a substring appears in a string including overlaps.",
        category=PromptCategory.STRING_MANIPULATION,
        source="humaneval",
        task_id="HumanEval/18",
    ),
    # HumanEval/19 - Number to string (STRING_MANIPULATION)
    CorpusEntry(
        prompt="Write a function that sorts a list of integers and returns them as a space-separated string.",
        category=PromptCategory.STRING_MANIPULATION,
        source="humaneval",
        task_id="HumanEval/19",
    ),
    # HumanEval/20 - Closest elements (DATA_STRUCTURE)
    CorpusEntry(
        prompt="Write a function that finds the two closest numbers in a list.",
        category=PromptCategory.DATA_STRUCTURE,
        source="humaneval",
        task_id="HumanEval/20",
    ),
    # HumanEval/21 - Rescale (MATHEMATICAL)
    CorpusEntry(
        prompt="Write a function that rescales a list of numbers to the range [0, 1].",
        category=PromptCategory.MATHEMATICAL,
        source="humaneval",
        task_id="HumanEval/21",
    ),
    # HumanEval/22 - Filter integers (DATA_STRUCTURE)
    CorpusEntry(
        prompt="Write a function that filters a list to keep only the integer values.",
        category=PromptCategory.DATA_STRUCTURE,
        source="humaneval",
        task_id="HumanEval/22",
    ),
    # HumanEval/23 - String length (STRING_MANIPULATION)
    CorpusEntry(
        prompt="Write a function that returns the length of a string.",
        category=PromptCategory.STRING_MANIPULATION,
        source="humaneval",
        task_id="HumanEval/23",
    ),
    # HumanEval/24 - Largest divisor (MATHEMATICAL)
    CorpusEntry(
        prompt="Write a function that returns the largest divisor of n that is smaller than n.",
        category=PromptCategory.MATHEMATICAL,
        source="humaneval",
        task_id="HumanEval/24",
    ),
    # HumanEval/25 - Factorize (MATHEMATICAL)
    CorpusEntry(
        prompt="Write a function that returns the prime factorization of an integer as a list.",
        category=PromptCategory.MATHEMATICAL,
        source="humaneval",
        task_id="HumanEval/25",
    ),
    # HumanEval/26 - Remove duplicates (DATA_STRUCTURE)
    CorpusEntry(
        prompt="Write a function that removes duplicate elements from a list while preserving order.",
        category=PromptCategory.DATA_STRUCTURE,
        source="humaneval",
        task_id="HumanEval/26",
    ),
    # HumanEval/27 - Flip case (STRING_MANIPULATION)
    CorpusEntry(
        prompt="Write a function that flips the case of each character in a string.",
        category=PromptCategory.STRING_MANIPULATION,
        source="humaneval",
        task_id="HumanEval/27",
    ),
    # HumanEval/28 - Concatenate (STRING_MANIPULATION)
    CorpusEntry(
        prompt="Write a function that concatenates a list of strings into a single string.",
        category=PromptCategory.STRING_MANIPULATION,
        source="humaneval",
        task_id="HumanEval/28",
    ),
    # HumanEval/29 - Filter prefix (DATA_STRUCTURE)
    CorpusEntry(
        prompt="Write a function that filters a list of strings to keep only those starting with a given prefix.",
        category=PromptCategory.DATA_STRUCTURE,
        source="humaneval",
        task_id="HumanEval/29",
    ),
    # HumanEval/30 - Positive only (DATA_STRUCTURE)
    CorpusEntry(
        prompt="Write a function that returns only the positive numbers from a list.",
        category=PromptCategory.DATA_STRUCTURE,
        source="humaneval",
        task_id="HumanEval/30",
    ),
]

# MBPP-style prompts
MBPP_CORPUS: List[CorpusEntry] = [
    CorpusEntry(
        prompt="Write a function to find the nth Fibonacci number.",
        category=PromptCategory.ALGORITHM_IMPLEMENTATION,
        source="mbpp",
        task_id="MBPP/1",
    ),
    CorpusEntry(
        prompt="Write a function to check if a number is prime.",
        category=PromptCategory.MATHEMATICAL,
        source="mbpp",
        task_id="MBPP/2",
    ),
    CorpusEntry(
        prompt="Write a function to find the factorial of a number using recursion.",
        category=PromptCategory.ALGORITHM_IMPLEMENTATION,
        source="mbpp",
        task_id="MBPP/3",
    ),
    CorpusEntry(
        prompt="Write a function to reverse a string without using built-in functions.",
        category=PromptCategory.STRING_MANIPULATION,
        source="mbpp",
        task_id="MBPP/4",
    ),
    CorpusEntry(
        prompt="Write a function to merge two sorted lists into a single sorted list.",
        category=PromptCategory.ALGORITHM_IMPLEMENTATION,
        source="mbpp",
        task_id="MBPP/5",
    ),
    CorpusEntry(
        prompt="Write a function to find the second largest element in a list.",
        category=PromptCategory.DATA_STRUCTURE,
        source="mbpp",
        task_id="MBPP/6",
    ),
    CorpusEntry(
        prompt="Write a function to check if two strings are anagrams of each other.",
        category=PromptCategory.STRING_MANIPULATION,
        source="mbpp",
        task_id="MBPP/7",
    ),
    CorpusEntry(
        prompt="Write a function to find the intersection of two lists.",
        category=PromptCategory.DATA_STRUCTURE,
        source="mbpp",
        task_id="MBPP/8",
    ),
    CorpusEntry(
        prompt="Write a function to calculate the power of a number without using the ** operator.",
        category=PromptCategory.MATHEMATICAL,
        source="mbpp",
        task_id="MBPP/9",
    ),
    CorpusEntry(
        prompt="Write a function to flatten a nested list into a single-level list.",
        category=PromptCategory.DATA_STRUCTURE,
        source="mbpp",
        task_id="MBPP/10",
    ),
    CorpusEntry(
        prompt="Write a function to rotate a list by k positions to the right.",
        category=PromptCategory.DATA_STRUCTURE,
        source="mbpp",
        task_id="MBPP/11",
    ),
    CorpusEntry(
        prompt="Write a function to find all permutations of a string.",
        category=PromptCategory.ALGORITHM_IMPLEMENTATION,
        source="mbpp",
        task_id="MBPP/12",
    ),
    CorpusEntry(
        prompt="Write a function to check if a string contains balanced brackets.",
        category=PromptCategory.STRING_MANIPULATION,
        source="mbpp",
        task_id="MBPP/13",
    ),
    CorpusEntry(
        prompt="Write a function to convert a Roman numeral to an integer.",
        category=PromptCategory.STRING_MANIPULATION,
        source="mbpp",
        task_id="MBPP/14",
    ),
    CorpusEntry(
        prompt="Write a function to find the longest increasing subsequence in a list.",
        category=PromptCategory.ALGORITHM_IMPLEMENTATION,
        source="mbpp",
        task_id="MBPP/15",
    ),
]

# Real-world coding prompts
REALWORLD_CORPUS: List[CorpusEntry] = [
    # File I/O
    CorpusEntry(
        prompt="Write a function to read a CSV file and return a list of dictionaries.",
        category=PromptCategory.FILE_IO,
        source="realworld",
    ),
    CorpusEntry(
        prompt="Write a function to parse a JSON configuration file and validate required fields.",
        category=PromptCategory.FILE_IO,
        source="realworld",
    ),
    CorpusEntry(
        prompt="Write a function to read a log file and extract error messages with timestamps.",
        category=PromptCategory.FILE_IO,
        source="realworld",
    ),
    # API Integration
    CorpusEntry(
        prompt="Write a function to make an HTTP GET request and parse the JSON response.",
        category=PromptCategory.API_INTEGRATION,
        source="realworld",
    ),
    CorpusEntry(
        prompt="Write a function to authenticate with an OAuth2 API and refresh tokens.",
        category=PromptCategory.API_INTEGRATION,
        source="realworld",
    ),
    CorpusEntry(
        prompt="Write a function to connect to a PostgreSQL database and execute a query.",
        category=PromptCategory.API_INTEGRATION,
        source="realworld",
    ),
    # Testing
    CorpusEntry(
        prompt="Write unit tests for a function that validates email addresses.",
        category=PromptCategory.TESTING,
        source="realworld",
    ),
    CorpusEntry(
        prompt="Write a pytest fixture that sets up a mock database connection.",
        category=PromptCategory.TESTING,
        source="realworld",
    ),
    CorpusEntry(
        prompt="Write integration tests for a REST API endpoint.",
        category=PromptCategory.TESTING,
        source="realworld",
    ),
    # Code Debugging
    CorpusEntry(
        prompt="Fix the bug in this function that causes an IndexError when the list is empty.",
        category=PromptCategory.CODE_DEBUGGING,
        source="realworld",
    ),
    CorpusEntry(
        prompt="Debug this async function that is causing a race condition.",
        category=PromptCategory.CODE_DEBUGGING,
        source="realworld",
    ),
    CorpusEntry(
        prompt="Fix the memory leak in this function that processes large files.",
        category=PromptCategory.CODE_DEBUGGING,
        source="realworld",
    ),
    # Code Explanation
    CorpusEntry(
        prompt="Explain what this recursive function does and trace through an example.",
        category=PromptCategory.CODE_EXPLANATION,
        source="realworld",
    ),
    CorpusEntry(
        prompt="Explain the time and space complexity of this algorithm.",
        category=PromptCategory.CODE_EXPLANATION,
        source="realworld",
    ),
    CorpusEntry(
        prompt="Explain how this decorator pattern works in Python.",
        category=PromptCategory.CODE_EXPLANATION,
        source="realworld",
    ),
    # Code Refactoring
    CorpusEntry(
        prompt="Refactor this function to use list comprehension instead of loops.",
        category=PromptCategory.CODE_REFACTORING,
        source="realworld",
    ),
    CorpusEntry(
        prompt="Refactor this class to follow the Single Responsibility Principle.",
        category=PromptCategory.CODE_REFACTORING,
        source="realworld",
    ),
    CorpusEntry(
        prompt="Refactor this code to handle errors with proper exception handling.",
        category=PromptCategory.CODE_REFACTORING,
        source="realworld",
    ),
    # General Coding
    CorpusEntry(
        prompt="Write a Python script to automate sending emails with attachments.",
        category=PromptCategory.GENERAL_CODING,
        source="realworld",
    ),
    CorpusEntry(
        prompt="Create a command-line tool that converts between file formats.",
        category=PromptCategory.GENERAL_CODING,
        source="realworld",
    ),
    CorpusEntry(
        prompt="Write a class that implements a thread-safe singleton pattern.",
        category=PromptCategory.GENERAL_CODING,
        source="realworld",
    ),
]

# Combine all corpus entries (legacy - kept for backwards compatibility)
_LEGACY_CORPUS: List[CorpusEntry] = HUMANEVAL_CORPUS + MBPP_CORPUS + REALWORLD_CORPUS


def _build_extended_corpus() -> List[CorpusEntry]:
    """Build extended corpus from prompt_corpus_data module.

    Returns the full 1000+ entry corpus with realistic distribution:
    - Function completion (25%)
    - Code debugging (20%)
    - Code explanation (12%)
    - Code refactoring (10%)
    - Testing (8%)
    - Algorithm implementation (7%)
    - API integration (6%)
    - Data structure (5%)
    - File I/O (3%)
    - String manipulation (2%)
    - Mathematical (1%)
    - General coding (1%)
    """
    try:
        from victor.agent.prompt_corpus_data import build_complete_corpus

        corpus_data = build_complete_corpus()
        entries = []
        for prompt, category, source in corpus_data:
            entries.append(
                CorpusEntry(
                    prompt=prompt,
                    category=category,
                    source=source,
                )
            )
        logger.info(f"Loaded extended corpus with {len(entries)} entries")
        return entries
    except ImportError:
        logger.warning("prompt_corpus_data not available, using legacy corpus")
        return _LEGACY_CORPUS


# Full corpus - uses extended 1000+ entries if available
FULL_CORPUS: List[CorpusEntry] = _build_extended_corpus()


# =============================================================================
# PROMPT BUILDERS - Category-specific prompt builders
# =============================================================================

# Simplified builder system using a mapping approach
PROMPT_TEMPLATES: Dict[PromptCategory, Dict[str, Any]] = {
    PromptCategory.FUNCTION_COMPLETION: {
        "system": """You are an expert programmer completing Python functions.

TASK: Complete the function implementation based on the signature and docstring.

RULES:
1. Implement the function body to match the docstring specification exactly.
2. Handle all edge cases mentioned in the docstring.
3. Return only the complete function implementation.
4. Do NOT include test code, examples, or explanations.
5. Ensure the code is syntactically correct and follows Python best practices.

OUTPUT FORMAT:
- Return ONLY the Python code for the function.
- Include proper indentation.
- Do not wrap in markdown code blocks unless explicitly asked.""",
        "hints": ["complete_function", "no_exploration_needed"],
    },
    PromptCategory.ALGORITHM_IMPLEMENTATION: {
        "system": """You are an algorithm expert implementing efficient solutions.

TASK: Implement the requested algorithm in Python.

APPROACH:
1. Analyze the problem requirements and constraints.
2. Choose an appropriate algorithm with optimal time/space complexity.
3. Implement the solution with clear variable names.
4. Handle edge cases (empty input, single element, etc.).

OUTPUT FORMAT:
- Return the complete Python function implementation.
- Include type hints for parameters and return type.
- Add a brief complexity comment (# O(n) time, O(1) space).""",
        "hints": ["algorithm", "complexity_aware"],
    },
    PromptCategory.DATA_STRUCTURE: {
        "system": """You are a data structure specialist.

TASK: Implement the requested data structure operation.

GUIDELINES:
1. Use appropriate Python data structures (list, dict, set, deque, etc.).
2. Consider mutability and side effects.
3. Handle empty collections gracefully.
4. Prefer built-in methods when they provide the cleanest solution.

OUTPUT FORMAT:
- Return the complete function implementation.
- Use clear parameter names that indicate expected types.
- Include handling for edge cases.""",
        "hints": ["data_structure", "edge_cases"],
    },
    PromptCategory.STRING_MANIPULATION: {
        "system": """You are a string processing expert.

TASK: Implement the requested string manipulation.

GUIDELINES:
1. Handle Unicode and special characters correctly.
2. Consider empty strings and whitespace.
3. Use Python's string methods effectively.
4. For pattern matching, consider regex if appropriate.

OUTPUT FORMAT:
- Return the complete function implementation.
- Handle edge cases (empty string, single character, etc.).
- Use efficient string operations (avoid += in loops).""",
        "hints": ["string_ops", "unicode_aware"],
    },
    PromptCategory.MATHEMATICAL: {
        "system": """You are a mathematical computing expert.

TASK: Implement the requested mathematical operation.

GUIDELINES:
1. Handle numerical precision carefully (floats vs integers).
2. Consider edge cases (zero, negative, very large numbers).
3. Use appropriate Python math operations.
4. Avoid integer overflow issues.

OUTPUT FORMAT:
- Return the complete function implementation.
- Include type hints (int, float, List[int], etc.).
- Handle division by zero and other mathematical edge cases.""",
        "hints": ["numerical", "precision_aware"],
    },
    PromptCategory.FILE_IO: {
        "system": """You are a file handling expert.

TASK: Implement the requested file I/O operation.

GUIDELINES:
1. Use context managers (with statement) for file handling.
2. Handle file not found and permission errors gracefully.
3. Support both text and binary modes as appropriate.
4. Consider encoding for text files (UTF-8 default).

OUTPUT FORMAT:
- Return the complete function implementation.
- Include proper error handling with try/except.
- Close resources properly (or use context managers).""",
        "hints": ["file_io", "resource_management"],
    },
    PromptCategory.CODE_DEBUGGING: {
        "system": """You are a debugging expert.

TASK: Identify and fix the bug in the provided code.

APPROACH:
1. Analyze the code to understand intended behavior.
2. Identify the root cause of the bug.
3. Provide the corrected implementation.
4. Explain what was wrong and how you fixed it.

OUTPUT FORMAT:
- First, briefly explain the bug.
- Then provide the corrected code.
- Highlight the specific lines that changed.""",
        "hints": ["debugging", "explanation_needed"],
    },
    PromptCategory.CODE_EXPLANATION: {
        "system": """You are a code explanation expert.

TASK: Explain the provided code clearly and thoroughly.

APPROACH:
1. Start with a high-level overview of what the code does.
2. Walk through the logic step by step.
3. Explain any complex or non-obvious parts.
4. If applicable, trace through an example.

OUTPUT FORMAT:
- Provide a clear, structured explanation.
- Use examples to illustrate key points.
- Mention time/space complexity if relevant.""",
        "hints": ["explanation", "educational"],
    },
    PromptCategory.CODE_REFACTORING: {
        "system": """You are a code refactoring expert.

TASK: Refactor the provided code to improve its quality.

PRINCIPLES:
1. Improve readability and maintainability.
2. Follow Python best practices (PEP 8, Pythonic idioms).
3. Reduce complexity where possible.
4. Preserve the original functionality.

OUTPUT FORMAT:
- Provide the refactored code.
- Briefly explain the improvements made.
- Ensure tests would still pass (same behavior).""",
        "hints": ["refactoring", "preserve_behavior"],
    },
    PromptCategory.API_INTEGRATION: {
        "system": """You are an API integration expert.

TASK: Implement the requested API or integration code.

GUIDELINES:
1. Use appropriate libraries (requests, aiohttp, etc.).
2. Handle network errors and timeouts.
3. Implement proper authentication if needed.
4. Parse responses correctly.

OUTPUT FORMAT:
- Provide complete, runnable code.
- Include error handling for network issues.
- Use async if the context requires it.""",
        "hints": ["api", "error_handling"],
    },
    PromptCategory.TESTING: {
        "system": """You are a testing expert.

TASK: Write tests for the specified code or functionality.

GUIDELINES:
1. Use pytest as the testing framework.
2. Cover normal cases, edge cases, and error cases.
3. Use descriptive test names.
4. Include fixtures and mocks where appropriate.

OUTPUT FORMAT:
- Provide complete, runnable test code.
- Include necessary imports.
- Organize tests logically.""",
        "hints": ["testing", "pytest"],
    },
    PromptCategory.GENERAL_CODING: {
        "system": """You are an expert programmer.

TASK: Implement the requested functionality.

GUIDELINES:
1. Write clean, readable, maintainable code.
2. Follow Python best practices.
3. Handle edge cases appropriately.
4. Include type hints where helpful.

OUTPUT FORMAT:
- Provide complete, working code.
- Include necessary imports.
- Add brief comments for complex logic.""",
        "hints": ["general"],
    },
}


# =============================================================================
# PROMPT BUILDER CLASSES
# =============================================================================


class PromptTemplateBuilder:
    """Base class for category-specific prompt template builders."""

    category: PromptCategory = PromptCategory.GENERAL_CODING

    def build(self, user_prompt: str, match: PromptMatch) -> EnrichedPrompt:
        """Build an enriched prompt.

        Args:
            user_prompt: The user's prompt
            match: The match result from corpus

        Returns:
            EnrichedPrompt with system and user prompts
        """
        template = PROMPT_TEMPLATES.get(
            self.category, PROMPT_TEMPLATES[PromptCategory.GENERAL_CODING]
        )
        return EnrichedPrompt(
            system_prompt=template["system"],
            user_prompt=user_prompt,
            category=self.category,
            hints=template["hints"],
        )


class FunctionCompletionBuilder(PromptTemplateBuilder):
    """Builder for function completion prompts."""

    category = PromptCategory.FUNCTION_COMPLETION


class AlgorithmImplementationBuilder(PromptTemplateBuilder):
    """Builder for algorithm implementation prompts."""

    category = PromptCategory.ALGORITHM_IMPLEMENTATION


class DataStructureBuilder(PromptTemplateBuilder):
    """Builder for data structure prompts."""

    category = PromptCategory.DATA_STRUCTURE


class StringManipulationBuilder(PromptTemplateBuilder):
    """Builder for string manipulation prompts."""

    category = PromptCategory.STRING_MANIPULATION


class MathematicalBuilder(PromptTemplateBuilder):
    """Builder for mathematical prompts."""

    category = PromptCategory.MATHEMATICAL


class FileIOBuilder(PromptTemplateBuilder):
    """Builder for file I/O prompts."""

    category = PromptCategory.FILE_IO


class CodeDebuggingBuilder(PromptTemplateBuilder):
    """Builder for code debugging prompts."""

    category = PromptCategory.CODE_DEBUGGING


class CodeExplanationBuilder(PromptTemplateBuilder):
    """Builder for code explanation prompts."""

    category = PromptCategory.CODE_EXPLANATION


class CodeRefactoringBuilder(PromptTemplateBuilder):
    """Builder for code refactoring prompts."""

    category = PromptCategory.CODE_REFACTORING


class APIIntegrationBuilder(PromptTemplateBuilder):
    """Builder for API integration prompts."""

    category = PromptCategory.API_INTEGRATION


class TestingBuilder(PromptTemplateBuilder):
    """Builder for testing prompts."""

    category = PromptCategory.TESTING


class GeneralCodingBuilder(PromptTemplateBuilder):
    """Builder for general coding prompts."""

    category = PromptCategory.GENERAL_CODING


# =============================================================================
# PROMPT CORPUS REGISTRY - Main class for embedding-based prompt matching
# =============================================================================


class PromptCorpusRegistry:
    """Registry for embedding-based prompt matching and builder selection.

    Uses sentence embeddings to match user prompts to the corpus and
    select the appropriate specialized prompt builder.

    Features:
    - Hash-based change detection for corpus entries (like SemanticToolSelector)
    - Disk caching of corpus embeddings for fast startup
    - Centralized EmbeddingService singleton for memory efficiency
    """

    def __init__(
        self,
        embedding_model: Optional[Any] = None,
        corpus: Optional[List[CorpusEntry]] = None,
        cache_embeddings: bool = True,
        cache_dir: Optional[Path] = None,
    ):
        """Initialize the registry.

        Args:
            embedding_model: EmbeddingService instance (or auto-fetched singleton)
            corpus: Custom corpus (defaults to FULL_CORPUS)
            cache_embeddings: Whether to cache embeddings to disk (default: True)
            cache_dir: Directory for embedding cache (default: ~/.victor/embeddings/)
        """
        self._embedding_model = embedding_model
        self._corpus = corpus or FULL_CORPUS.copy()
        self._corpus_embeddings: Optional[Any] = None  # np.ndarray when available
        self._cache_embeddings = cache_embeddings

        # Cache directory and file path
        if cache_dir is None:
            try:
                from victor.config.settings import get_project_paths

                cache_dir = get_project_paths().global_embeddings_dir
            except ImportError:
                try:
                    from victor.config.secure_paths import get_victor_dir

                    cache_dir = get_victor_dir() / "embeddings"
                except ImportError:
                    cache_dir = Path.home() / ".victor" / "embeddings"
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_file = self._cache_dir / "prompt_corpus_embeddings.pkl"

        # Hash of corpus for change detection
        self._corpus_hash: Optional[str] = None

        # Initialize default builders
        self._builders: Dict[PromptCategory, PromptTemplateBuilder] = {
            PromptCategory.FUNCTION_COMPLETION: FunctionCompletionBuilder(),
            PromptCategory.ALGORITHM_IMPLEMENTATION: AlgorithmImplementationBuilder(),
            PromptCategory.DATA_STRUCTURE: DataStructureBuilder(),
            PromptCategory.STRING_MANIPULATION: StringManipulationBuilder(),
            PromptCategory.MATHEMATICAL: MathematicalBuilder(),
            PromptCategory.FILE_IO: FileIOBuilder(),
            PromptCategory.CODE_DEBUGGING: CodeDebuggingBuilder(),
            PromptCategory.CODE_EXPLANATION: CodeExplanationBuilder(),
            PromptCategory.CODE_REFACTORING: CodeRefactoringBuilder(),
            PromptCategory.API_INTEGRATION: APIIntegrationBuilder(),
            PromptCategory.TESTING: TestingBuilder(),
            PromptCategory.GENERAL_CODING: GeneralCodingBuilder(),
        }

    def register_builder(self, builder: PromptTemplateBuilder) -> None:
        """Register a custom builder for a category.

        Args:
            builder: The builder to register
        """
        self._builders[builder.category] = builder

    def add_corpus_entry(self, entry: CorpusEntry) -> None:
        """Add an entry to the corpus.

        Args:
            entry: The corpus entry to add
        """
        self._corpus.append(entry)
        self._corpus_embeddings = None  # Invalidate in-memory cache
        self._corpus_hash = None  # Force hash recalculation

    def _calculate_corpus_hash(self) -> str:
        """Calculate hash of all corpus entries to detect changes.

        Similar to SemanticToolSelector._calculate_tools_hash(), this creates
        a deterministic hash from all corpus entries to detect when the corpus
        changes and embeddings need to be recomputed.

        Returns:
            SHA256 hash of corpus entries
        """
        # Create deterministic string from all corpus entries
        corpus_strings = []
        for entry in sorted(self._corpus, key=lambda e: e.prompt):
            corpus_string = f"{entry.prompt}:{entry.category.value}:{entry.source}"
            corpus_strings.append(corpus_string)

        combined = "|".join(corpus_strings)
        return hashlib.sha256(combined.encode()).hexdigest()

    def _load_from_cache(self, corpus_hash: str) -> bool:
        """Load embeddings from pickle cache if valid.

        Args:
            corpus_hash: Current hash of corpus entries

        Returns:
            True if loaded successfully, False otherwise
        """
        if not self._cache_file.exists():
            return False

        try:
            with open(self._cache_file, "rb") as f:
                cache_data = pickle.load(f)

            # Verify cache is for same corpus
            if cache_data.get("corpus_hash") != corpus_hash:
                logger.info("Corpus entries changed, cache invalidated")
                return False

            # Verify cache has embeddings
            embeddings = cache_data.get("embeddings")
            if embeddings is None:
                logger.info("Cache missing embeddings")
                return False

            # Verify embedding count matches corpus count
            if len(embeddings) != len(self._corpus):
                logger.info(f"Cache size mismatch: {len(embeddings)} != {len(self._corpus)}")
                return False

            # Load embeddings
            self._corpus_embeddings = embeddings
            self._corpus_hash = corpus_hash

            # Also restore embeddings to individual entries
            for i, entry in enumerate(self._corpus):
                entry.embedding = embeddings[i]

            return True

        except Exception as e:
            logger.warning(f"Failed to load corpus embedding cache: {e}")
            return False

    def _save_to_cache(self, corpus_hash: str) -> None:
        """Save embeddings to pickle cache.

        Args:
            corpus_hash: Hash of corpus entries
        """
        if not self._cache_embeddings:
            return

        try:
            # Get model name for cache metadata
            model_name = "unknown"
            service = self._get_embedding_service()
            if service is not None and hasattr(service, "model_name"):
                model_name = service.model_name

            cache_data = {
                "corpus_hash": corpus_hash,
                "embedding_model": model_name,
                "embeddings": self._corpus_embeddings,
                "corpus_size": len(self._corpus),
            }

            with open(self._cache_file, "wb") as f:
                pickle.dump(cache_data, f)

            cache_size = self._cache_file.stat().st_size / 1024  # KB
            logger.info(
                f"Saved corpus embedding cache to {self._cache_file} "
                f"({cache_size:.1f} KB, {len(self._corpus)} entries)"
            )

        except Exception as e:
            logger.warning(f"Failed to save corpus embedding cache: {e}")

    def _get_embedding_service(self) -> Any:
        """Get the centralized EmbeddingService singleton.

        Uses the shared EmbeddingService for memory efficiency - same model
        instance is used across SemanticToolSelector, IntentClassifier, and
        other components (saves ~80MB per model).
        """
        if self._embedding_model is None:
            try:
                from victor.storage.embeddings.service import get_embedding_service

                self._embedding_model = get_embedding_service()
                logger.info(
                    f"Using shared EmbeddingService singleton "
                    f"(model: {self._embedding_model.model_name})"
                )
            except ImportError:
                logger.warning("EmbeddingService not available, using keyword fallback")
                return None
        return self._embedding_model

    def _compute_corpus_embeddings(self) -> Optional[Any]:
        """Compute embeddings for all corpus entries.

        Uses hash-based change detection and disk caching for fast startup:
        1. Calculate hash of corpus entries
        2. Try to load from disk cache if hash matches
        3. If cache miss, compute embeddings and save to cache

        Uses the centralized EmbeddingService for consistent embeddings
        across all Victor components.
        """
        if self._corpus_embeddings is not None:
            return self._corpus_embeddings

        corpus_hash = self._calculate_corpus_hash()

        if self._cache_embeddings and self._load_from_cache(corpus_hash):
            logger.info(f"Loaded corpus embeddings from cache for {len(self._corpus)} entries")
            return self._corpus_embeddings

        service = self._get_embedding_service()
        if service is None:
            return None

        logger.info(f"Computing corpus embeddings for {len(self._corpus)} entries")
        prompts = [entry.prompt for entry in self._corpus]
        self._corpus_embeddings = service.embed_batch_sync(prompts)
        self._corpus_hash = corpus_hash

        for i, entry in enumerate(self._corpus):
            entry.embedding = self._corpus_embeddings[i]

        self._save_to_cache(corpus_hash)
        return self._corpus_embeddings

    def match(self, user_prompt: str) -> PromptMatch:
        """Match a user prompt to the corpus.

        Args:
            user_prompt: The user's prompt

        Returns:
            PromptMatch with category and confidence
        """
        # Try embedding-based matching first
        corpus_embeddings = self._compute_corpus_embeddings()
        if corpus_embeddings is not None:
            return self._embedding_match(user_prompt, corpus_embeddings)

        # Fall back to keyword matching
        return self._keyword_match(user_prompt)

    def _embedding_match(self, user_prompt: str, corpus_embeddings: Any) -> PromptMatch:
        """Match using sentence embeddings."""
        service = self._get_embedding_service()
        query_embedding = service.embed_text_sync(user_prompt)
        similarities = service.cosine_similarity_matrix(query_embedding, corpus_embeddings)

        best_idx = int(np.argmax(similarities))
        best_similarity = float(similarities[best_idx])
        best_entry = self._corpus[best_idx]
        confidence = min(1.0, max(0.0, (best_similarity - 0.2) / 0.6))

        return PromptMatch(
            category=best_entry.category,
            confidence=confidence,
            matched_entry=best_entry,
            similarity=best_similarity,
        )

    def _keyword_match(self, user_prompt: str) -> PromptMatch:
        """Fallback keyword-based matching.

        Args:
            user_prompt: The user's prompt

        Returns:
            PromptMatch result
        """
        prompt_lower = user_prompt.lower()

        # Category keywords
        keyword_map: Dict[PromptCategory, List[str]] = {
            PromptCategory.FUNCTION_COMPLETION: [
                "complete",
                "implement the function",
                "fill in",
                "def ",
                '"""',
            ],
            PromptCategory.ALGORITHM_IMPLEMENTATION: [
                "algorithm",
                "sort",
                "search",
                "binary",
                "dynamic programming",
                "recursion",
            ],
            PromptCategory.DATA_STRUCTURE: [
                "list",
                "dict",
                "dictionary",
                "array",
                "tree",
                "graph",
                "queue",
                "stack",
            ],
            PromptCategory.STRING_MANIPULATION: [
                "string",
                "parse",
                "format",
                "regex",
                "pattern",
                "substring",
            ],
            PromptCategory.MATHEMATICAL: [
                "calculate",
                "compute",
                "sum",
                "product",
                "factorial",
                "prime",
                "number",
            ],
            PromptCategory.FILE_IO: [
                "file",
                "read",
                "write",
                "csv",
                "json",
                "parse file",
            ],
            PromptCategory.CODE_DEBUGGING: ["fix", "bug", "debug", "error", "issue"],
            PromptCategory.CODE_EXPLANATION: [
                "explain",
                "what does",
                "how does",
                "trace",
            ],
            PromptCategory.CODE_REFACTORING: [
                "refactor",
                "improve",
                "clean up",
                "optimize",
            ],
            PromptCategory.API_INTEGRATION: [
                "api",
                "http",
                "request",
                "database",
                "connect",
            ],
            PromptCategory.TESTING: ["test", "unittest", "pytest", "mock", "fixture"],
        }

        # Score each category
        scores: Dict[PromptCategory, int] = {}
        for category, keywords in keyword_map.items():
            score = sum(1 for kw in keywords if kw in prompt_lower)
            if score > 0:
                scores[category] = score

        if not scores:
            return PromptMatch(
                category=PromptCategory.GENERAL_CODING,
                confidence=0.3,
            )

        # Get best category
        best_category = max(scores, key=lambda k: scores[k])
        best_score = scores[best_category]
        confidence = min(1.0, best_score / 3.0 + 0.3)

        return PromptMatch(
            category=best_category,
            confidence=confidence,
        )

    def build_prompt(self, user_prompt: str) -> EnrichedPrompt:
        """Build an enriched prompt for the given user input.

        Args:
            user_prompt: The user's prompt

        Returns:
            EnrichedPrompt with system and user prompts
        """
        match = self.match(user_prompt)
        template = PROMPT_TEMPLATES.get(
            match.category, PROMPT_TEMPLATES[PromptCategory.GENERAL_CODING]
        )

        return EnrichedPrompt(
            system_prompt=template["system"],
            user_prompt=user_prompt,
            category=match.category,
            hints=template["hints"],
        )

    def get_category_for_prompt(self, user_prompt: str) -> Tuple[PromptCategory, float]:
        """Get the category for a prompt (convenience method).

        Args:
            user_prompt: The user's prompt

        Returns:
            Tuple of (category, confidence)
        """
        match = self.match(user_prompt)
        return match.category, match.confidence


# Backward-compatible alias for older imports. New code should use
# PromptTemplateBuilder to avoid confusion with the canonical prompt builder.
PromptBuilder = PromptTemplateBuilder


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Global registry instance (lazy initialization)
_global_registry: Optional[PromptCorpusRegistry] = None


def get_registry() -> PromptCorpusRegistry:
    """Get the global prompt corpus registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = PromptCorpusRegistry()
    return _global_registry


def match_prompt(user_prompt: str) -> PromptMatch:
    """Match a user prompt to the corpus (convenience function).

    Args:
        user_prompt: The user's prompt

    Returns:
        PromptMatch result
    """
    return get_registry().match(user_prompt)


def build_enriched_prompt(user_prompt: str) -> EnrichedPrompt:
    """Build an enriched prompt (convenience function).

    Args:
        user_prompt: The user's prompt

    Returns:
        EnrichedPrompt with system and user prompts
    """
    return get_registry().build_prompt(user_prompt)


def get_prompt_category(user_prompt: str) -> Tuple[PromptCategory, float]:
    """Get the category for a prompt (convenience function).

    Args:
        user_prompt: The user's prompt

    Returns:
        Tuple of (category, confidence)
    """
    return get_registry().get_category_for_prompt(user_prompt)

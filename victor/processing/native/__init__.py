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

"""Native extensions for high-performance operations.

This module provides Python bindings to Rust implementations of
performance-critical operations. If the Rust extension is not available,
it falls back to pure Python implementations.

Usage:
    from victor.processing.native import (
        rolling_hash_blocks,
        batch_cosine_similarity,
        repair_json,
        compute_signature,
    )

    # Functions automatically use Rust when available
    blocks = rolling_hash_blocks(content, min_block_length=50)
    similarities = batch_cosine_similarity(query, corpus)

Performance:
    - Deduplication: ~10-50x faster with Rust
    - Cosine similarity: ~2-5x faster with SIMD
    - JSON repair: ~5-10x faster with streaming parser
    - Signature hashing: ~10x faster with xxHash3
"""

# Base: native availability check
from victor.processing.native._base import (
    is_native_available,
    get_native_version,
)  # noqa: F401

# Deduplication & hashing
from victor.processing.native.deduplication import (  # noqa: F401
    normalize_block,
    rolling_hash_blocks,
    find_duplicate_blocks,
    compute_signature,
    compute_batch_signatures,
    signature_similarity,
)

# Similarity
from victor.processing.native.similarity import (  # noqa: F401
    cosine_similarity,
    batch_cosine_similarity,
    top_k_similar,
    batch_normalize_vectors,
    batch_cosine_similarity_normalized,
    top_k_similar_normalized,
)

# JSON repair & type coercion
from victor.processing.native.json_repair import (  # noqa: F401
    repair_json,
    extract_json_objects,
    coerce_string_type,
)

# Streaming filter & thinking detection
from victor.processing.native.streaming import (  # noqa: F401
    StreamingFilter,
    StreamingChunkResult,
    StreamingChunkResultFallback,
    strip_thinking_tokens,
    contains_thinking_tokens,
    find_thinking_tokens,
    extract_thinking_content,
    ThinkingDetector,
    PatternAnalysis,
    detect_circular_phrases,
    count_circular_patterns,
    find_circular_patterns,
)

# Task classifier
from victor.processing.native.classifier import (  # noqa: F401
    NativeTaskClassifier,
    NativeClassificationResult,
    NativeTaskType,
    classify_task_native,
    has_action_keywords,
    has_analysis_keywords,
    has_generation_keywords,
    has_negation,
    find_all_keywords,
)

# Document chunking
from victor.processing.native.chunking import (  # noqa: F401
    chunk_by_sentences,
    chunk_by_chars,
    chunk_by_paragraphs,
    detect_doc_type,
    count_tokens_approx,
    count_tokens,
    count_tokens_batch,
)

# Secret detection & pattern matching
from victor.processing.native.secrets import (  # noqa: F401
    SecretMatch,
    SecretMatchFallback,
    scan_secrets,
    has_secrets,
    get_secret_types,
    mask_secrets,
    list_secret_patterns,
    scan_secrets_summary,
    PatternMatcher,
    PatternMatch,
    PatternMatchFallback,
    PatternMatcherFallback,
    contains_any_pattern,
    find_all_patterns,
    count_pattern_matches,
    get_matched_pattern_indices,
    batch_contains_any,
    weighted_pattern_score,
)

# Tool call extraction & response sanitization
from victor.processing.native.tool_extraction import (  # noqa: F401
    extract_file_path,
    extract_code_blocks,
    extract_shell_commands,
    extract_tool_call,
    batch_extract_file_paths,
    ExtractedToolCallResult,
    sanitize_response_fast,
    is_garbage_content_fast,
    detect_leakage_patterns,
    strip_markup_fast,
    validate_tool_name,
)

# Tokenizer
from victor.processing.native.tokenizer import (  # noqa: F401
    count_tokens,
    count_tokens_fast,
    count_tokens_batch,
)

# Context fitter
from victor.processing.native.context_fitter import (  # noqa: F401
    fit_context,
    truncate_message,
    FitResult,
)

# Accelerator, stdlib, YAML, protocol dispatch
from victor.processing.native.accelerator import (  # noqa: F401
    # Stdlib detection
    is_stdlib_module,
    # YAML parsing
    parse_yaml,
    parse_yaml_with_env,
    parse_yaml_file,
    parse_yaml_file_with_env,
    validate_yaml,
    extract_workflow_names,
    # Protocol-based dispatch
    get_symbol_extractor,
    get_argument_normalizer,
    get_similarity_computer,
    get_text_chunker,
    get_ast_indexer,
    get_default_symbol_extractor,
    get_default_argument_normalizer,
    get_default_similarity_computer,
    get_default_text_chunker,
    get_default_ast_indexer,
    reset_protocol_singletons,
    # Token counting
    get_token_counter,
    get_context_fitter,
    get_default_token_counter,
    get_default_context_fitter,
    # Content hashing
    get_content_hasher,
    get_default_content_hasher_fuzzy,
    get_default_content_hasher_exact,
    # Accelerator priority system
    AcceleratorPreference,
    AcceleratorBenchmark,
    ACCELERATOR_BENCHMARKS,
    set_accelerator_preference,
    get_preferred_backend,
    get_all_benchmarks,
    calibrate_operation,
)

__all__ = [
    # Status
    "is_native_available",
    "get_native_version",
    # Deduplication
    "normalize_block",
    "rolling_hash_blocks",
    "find_duplicate_blocks",
    # Similarity
    "cosine_similarity",
    "batch_cosine_similarity",
    "top_k_similar",
    "batch_normalize_vectors",
    "batch_cosine_similarity_normalized",
    "top_k_similar_normalized",
    # JSON repair
    "repair_json",
    "extract_json_objects",
    # Hashing
    "compute_signature",
    "compute_batch_signatures",
    "signature_similarity",
    # Streaming filter
    "StreamingFilter",
    "StreamingChunkResult",
    "strip_thinking_tokens",
    "contains_thinking_tokens",
    "find_thinking_tokens",
    "extract_thinking_content",
    # Task classifier
    "NativeTaskClassifier",
    "NativeClassificationResult",
    "NativeTaskType",
    "classify_task_native",
    "has_action_keywords",
    "has_analysis_keywords",
    "has_generation_keywords",
    "has_negation",
    "find_all_keywords",
    # Thinking detector
    "ThinkingDetector",
    "PatternAnalysis",
    "detect_circular_phrases",
    "count_circular_patterns",
    "find_circular_patterns",
    # Document chunking
    "chunk_by_sentences",
    "chunk_by_chars",
    "chunk_by_paragraphs",
    "detect_doc_type",
    "count_tokens_approx",
    "count_tokens",
    "count_tokens_batch",
    # Secret detection
    "SecretMatch",
    "scan_secrets",
    "has_secrets",
    "get_secret_types",
    "mask_secrets",
    "list_secret_patterns",
    "scan_secrets_summary",
    # Pattern matching
    "PatternMatcher",
    "PatternMatch",
    "contains_any_pattern",
    "find_all_patterns",
    "count_pattern_matches",
    "get_matched_pattern_indices",
    "batch_contains_any",
    "weighted_pattern_score",
    # Tool call extraction
    "extract_file_path",
    "extract_code_blocks",
    "extract_shell_commands",
    "extract_tool_call",
    "batch_extract_file_paths",
    "ExtractedToolCallResult",
    # Response sanitization
    "sanitize_response_fast",
    "is_garbage_content_fast",
    "detect_leakage_patterns",
    "strip_markup_fast",
    "validate_tool_name",
    # Type coercion
    "coerce_string_type",
    # Stdlib detection
    "is_stdlib_module",
    # YAML parsing
    "parse_yaml",
    "parse_yaml_with_env",
    "parse_yaml_file",
    "parse_yaml_file_with_env",
    "validate_yaml",
    "extract_workflow_names",
    # Protocol-based dispatch
    "get_symbol_extractor",
    "get_argument_normalizer",
    "get_similarity_computer",
    "get_text_chunker",
    "get_ast_indexer",
    "get_default_symbol_extractor",
    "get_default_argument_normalizer",
    "get_default_similarity_computer",
    "get_default_text_chunker",
    "get_default_ast_indexer",
    "reset_protocol_singletons",
    # Tokenizer
    "count_tokens",
    "count_tokens_fast",
    "count_tokens_batch",
    # Context fitter
    "fit_context",
    "truncate_message",
    "FitResult",
    # Token counter / context fitter dispatch
    "get_token_counter",
    "get_context_fitter",
    "get_default_token_counter",
    "get_default_context_fitter",
    # Content hashing
    "get_content_hasher",
    "get_default_content_hasher_fuzzy",
    "get_default_content_hasher_exact",
    # Accelerator priority system
    "AcceleratorPreference",
    "AcceleratorBenchmark",
    "ACCELERATOR_BENCHMARKS",
    "set_accelerator_preference",
    "get_preferred_backend",
    "get_all_benchmarks",
    "calibrate_operation",
]

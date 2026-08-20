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

"""Accelerator priority system, stdlib detection, YAML parsing, and protocol dispatch."""

import logging
import os
import re
import sys
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from victor.processing.native._base import _NATIVE_AVAILABLE, _native

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from victor.native.protocols import (
        ArgumentNormalizerProtocol,
        AstIndexerProtocol,
        ContentHasherProtocol,
        ContextFitterProtocol,
        SimilarityComputerProtocol,
        SymbolExtractorProtocol,
        TextChunkerProtocol,
        TokenCounterProtocol,
    )


# =============================================================================
# STDLIB DETECTION (v0.4.0 - Indexer Hot Path)
# =============================================================================

# Python stdlib modules for fallback lookup
_PYTHON_STDLIB_MODULES = frozenset(
    {
        # Core builtins and language
        "abc",
        "asyncio",
        "builtins",
        "collections",
        "contextlib",
        "copy",
        "dataclasses",
        "enum",
        "functools",
        "gc",
        "inspect",
        "io",
        "itertools",
        "operator",
        "sys",
        "types",
        "typing",
        "typing_extensions",
        "weakref",
        # File/OS operations
        "os",
        "pathlib",
        "shutil",
        "stat",
        "tempfile",
        "glob",
        "fnmatch",
        # Data formats
        "json",
        "csv",
        "xml",
        "html",
        "pickle",
        "base64",
        "codecs",
        "struct",
        # Text processing
        "re",
        "string",
        "textwrap",
        "unicodedata",
        "difflib",
        # Date/Time
        "datetime",
        "time",
        "calendar",
        "zoneinfo",
        # Math/Numbers
        "math",
        "decimal",
        "fractions",
        "random",
        "statistics",
        "cmath",
        # Networking
        "socket",
        "ssl",
        "http",
        "urllib",
        "email",
        "ftplib",
        "smtplib",
        # Concurrent
        "threading",
        "multiprocessing",
        "concurrent",
        "queue",
        "subprocess",
        "signal",
        "selectors",
        # Testing/Debug
        "unittest",
        "doctest",
        "pdb",
        "traceback",
        "logging",
        "warnings",
        # Crypto/Hashing
        "hashlib",
        "hmac",
        "secrets",
        # Archive/Compression
        "zipfile",
        "tarfile",
        "gzip",
        "bz2",
        "lzma",
        "zlib",
        # Other common stdlib
        "argparse",
        "configparser",
        "getopt",
        "pprint",
        "shelve",
        "sqlite3",
        "atexit",
        "sched",
        "heapq",
        "bisect",
        "array",
        "cProfile",
        "profile",
        "timeit",
        "trace",
        "ast",
        "dis",
        "code",
        "codeop",
        "compileall",
        "py_compile",
        "importlib",
        "pkgutil",
        "modulefinder",
        "runpy",
        "venv",
        "site",
        "sysconfig",
        "platform",
        "ctypes",
        "mmap",
        "uuid",
        "ipaddress",
        "locale",
        "gettext",
    }
)
_ALLOWED_THIRD_PARTY_MODULES = frozenset(
    {
        "typing_extensions",
        "numpy",
        "pandas",
        "requests",
        "aiohttp",
        "httpx",
        "pydantic",
        "pytest",
        "mock",
    }
)
_FAST_STDLIB_MODULES = frozenset(
    set(sys.stdlib_module_names) | set(_PYTHON_STDLIB_MODULES) | set(_ALLOWED_THIRD_PARTY_MODULES)
)


def is_stdlib_module(module_name: str) -> bool:
    """Check if a module is part of Python's standard library.

    This path is intentionally pure Python. For tiny module-name lookups, the
    Python<->Rust boundary costs more than the actual set membership test.
    We use Python's built-in stdlib table plus the repo's allowed third-party
    modules to keep the behavior broad without paying FFI overhead.

    Args:
        module_name: Full module name (e.g., "os.path", "collections.abc")

    Returns:
        True if the module is a stdlib module
    """
    if not module_name:
        return False

    top_level = module_name.partition(".")[0]
    return top_level in _FAST_STDLIB_MODULES


# =============================================================================
# YAML PARSING (v0.4.0 - Workflow Acceleration)
# =============================================================================


def parse_yaml(yaml_content: str) -> Any:
    """Parse YAML string to Python object.

    Uses Rust serde_yaml for ~5-20x speedup on large workflow files.
    Falls back to PyYAML's safe_load.

    Args:
        yaml_content: Raw YAML string to parse

    Returns:
        Parsed Python object (dict, list, or scalar)
    """
    if _NATIVE_AVAILABLE and hasattr(_native, "parse_yaml"):
        return _native.parse_yaml(yaml_content)

    import yaml

    return yaml.safe_load(yaml_content)


def parse_yaml_with_env(yaml_content: str) -> Any:
    """Parse YAML with environment variable interpolation.

    Supports:
    - $env.VAR_NAME - Simple env var reference
    - ${VAR_NAME:-default} - Shell-style with optional default

    Uses Rust implementation when available for ~5-20x speedup.

    Args:
        yaml_content: Raw YAML string to parse

    Returns:
        Parsed Python object with env vars interpolated
    """
    if _NATIVE_AVAILABLE and hasattr(_native, "parse_yaml_with_env"):
        return _native.parse_yaml_with_env(yaml_content)

    # Python fallback with env var interpolation
    import yaml

    def interpolate_env_vars(value: Any) -> Any:
        if isinstance(value, str):
            # Handle $env.VAR_NAME
            result = re.sub(
                r"\$env\.([A-Za-z_][A-Za-z0-9_]*)",
                lambda m: os.environ.get(m.group(1), f"$env.{m.group(1)}"),
                value,
            )
            # Handle ${VAR:-default}
            result = re.sub(
                r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}",
                lambda m: os.environ.get(m.group(1), m.group(2) or ""),
                result,
            )
            return result
        elif isinstance(value, dict):
            return {k: interpolate_env_vars(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [interpolate_env_vars(item) for item in value]
        return value

    data = yaml.safe_load(yaml_content)
    return interpolate_env_vars(data)


def parse_yaml_file(file_path: str) -> Any:
    """Parse YAML file directly.

    Args:
        file_path: Path to YAML file

    Returns:
        Parsed Python object
    """
    if _NATIVE_AVAILABLE and hasattr(_native, "parse_yaml_file"):
        return _native.parse_yaml_file(file_path)

    import yaml

    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def parse_yaml_file_with_env(file_path: str) -> Any:
    """Parse YAML file with environment variable interpolation.

    Args:
        file_path: Path to YAML file

    Returns:
        Parsed Python object with env vars interpolated
    """
    if _NATIVE_AVAILABLE and hasattr(_native, "parse_yaml_file_with_env"):
        return _native.parse_yaml_file_with_env(file_path)

    with open(file_path, "r") as f:
        return parse_yaml_with_env(f.read())


def validate_yaml(yaml_content: str) -> bool:
    """Validate YAML syntax without full parsing.

    Args:
        yaml_content: Raw YAML string

    Returns:
        True if valid, False if invalid
    """
    if _NATIVE_AVAILABLE and hasattr(_native, "validate_yaml"):
        return _native.validate_yaml(yaml_content)

    import yaml

    try:
        yaml.safe_load(yaml_content)
        return True
    except yaml.YAMLError:
        return False


def extract_workflow_names(yaml_content: str) -> List[str]:
    """Extract workflow names from YAML content.

    Fast scan to find workflow names without full parsing.

    Args:
        yaml_content: Raw YAML string

    Returns:
        List of workflow names found
    """
    if _NATIVE_AVAILABLE and hasattr(_native, "extract_workflow_names"):
        return _native.extract_workflow_names(yaml_content)

    import yaml

    data = yaml.safe_load(yaml_content)
    if not isinstance(data, dict):
        return []

    names = []
    workflows = data.get("workflows", data)
    for key, val in workflows.items():
        if isinstance(val, dict) and "nodes" in val:
            names.append(key)

    return names


# =============================================================================
# ACCELERATOR PRIORITY SYSTEM (Benchmark-Based)
# =============================================================================


class AcceleratorPreference(str, Enum):
    """Backend preference for native accelerators."""

    RUST = "rust"  # Force Rust (fail if unavailable)
    PYTHON = "python"  # Force Python
    AUTO = "auto"  # Use benchmark-based default


@dataclass(frozen=True)
class AcceleratorBenchmark:
    """Benchmark data for an accelerator operation."""

    name: str
    rust_ms: float
    python_ms: float
    preferred: str  # "rust" or "python"
    notes: str = ""

    @property
    def speedup(self) -> float:
        """Speedup ratio (Python time / Rust time)."""
        return self.python_ms / self.rust_ms if self.rust_ms > 0 else 0.0


# Benchmark-based defaults (measured on Apple M-series, 2025-01)
ACCELERATOR_BENCHMARKS: Dict[str, AcceleratorBenchmark] = {
    # Text processing - Rust wins
    "normalize_block": AcceleratorBenchmark(
        "normalize_block", 0.21, 1.37, "rust", "Whitespace/punctuation normalization"
    ),
    "chunk_with_overlap": AcceleratorBenchmark(
        "chunk_with_overlap", 0.03, 0.14, "rust", "Line-aware text chunking"
    ),
    "content_hashing": AcceleratorBenchmark(
        "content_hashing",
        0.71,
        1.76,
        "rust",
        "Hash with normalization (SHA-256 dominates)",
    ),
    "count_lines": AcceleratorBenchmark(
        "count_lines", 0.004, 0.009, "rust", "SIMD-optimized line counting"
    ),
    "type_coercion": AcceleratorBenchmark(
        "type_coercion", 0.19, 0.40, "rust", "String to bool/int/float coercion"
    ),
    # Python wins - NumPy/BLAS or simple operations
    "stdlib_detection": AcceleratorBenchmark(
        "stdlib_detection", 0.12, 0.10, "python", "frozenset O(1) lookup is optimal"
    ),
    "json_repair": AcceleratorBenchmark(
        "json_repair", 0.06, 0.04, "python", "Simple string ops faster for small inputs"
    ),
    "batch_similarity": AcceleratorBenchmark(
        "batch_similarity", 0.18, 0.03, "python", "NumPy+BLAS has hardware SIMD"
    ),
    "similarity_matrix": AcceleratorBenchmark(
        "similarity_matrix", 0.20, 0.05, "python", "NumPy matmul is highly optimized"
    ),
}

# User-configurable overrides (can be set at runtime)
_accelerator_overrides: Dict[str, str] = {}


def set_accelerator_preference(operation: str, preference: str) -> None:
    """Override the default backend for an operation.

    Args:
        operation: Operation name (e.g., "normalize_block", "batch_similarity")
        preference: "rust", "python", or "auto" (reset to benchmark default)

    Example:
        # Force Python for all similarity operations
        set_accelerator_preference("batch_similarity", "python")

        # Reset to benchmark-based default
        set_accelerator_preference("batch_similarity", "auto")
    """
    if preference == "auto":
        _accelerator_overrides.pop(operation, None)
    else:
        _accelerator_overrides[operation] = preference


def get_preferred_backend(operation: str) -> str:
    """Get the optimal backend for an operation.

    Returns "rust" or "python" based on:
    1. User override (if set via set_accelerator_preference)
    2. Adaptive — observed per-backend EWMA latency, once enough samples exist
       (closes the measurement→dispatch loop; self-corrects ops the static
       benchmark mis-ranks for THIS machine). Gated by
       ``observability.native_adaptive_dispatch_enabled``.
    3. Benchmark data (static prior; hardware-aware cold start)
    4. Default to "rust" if native available, else "python"

    Args:
        operation: Operation name

    Returns:
        "rust" or "python"
    """
    # 1. User override always wins.
    if operation in _accelerator_overrides:
        return _accelerator_overrides[operation]

    # 2. Adaptive: pick the empirically faster backend from observed latency.
    if _NATIVE_AVAILABLE:
        adaptive = _adaptive_preference(operation)
        if adaptive is not None:
            return adaptive

    # 3. Static benchmark prior (cold start; encodes hardware-aware knowledge
    #    like "Python wins batch_similarity via NumPy+BLAS").
    if operation in ACCELERATOR_BENCHMARKS:
        benchmark = ACCELERATOR_BENCHMARKS[operation]
        preferred = benchmark.preferred
        # Only use Rust if it's available
        if preferred == "rust" and not _NATIVE_AVAILABLE:
            return "python"
        return preferred

    # 4. Default: use Rust if available
    return "rust" if _NATIVE_AVAILABLE else "python"


# -----------------------------------------------------------------------------
# Adaptive dispatch helpers (Gap 3).
# -----------------------------------------------------------------------------

_ADAPTIVE_SYNCED = False


def _get_adaptive_settings() -> Dict[str, Any]:
    """Read adaptive-dispatch settings, falling back to safe defaults."""
    defaults = {"enabled": True, "min_samples": 20, "alpha": 0.3}
    try:
        from victor.config.settings import get_settings

        s = get_settings().observability
        return {
            "enabled": bool(getattr(s, "native_adaptive_dispatch_enabled", True)),
            "min_samples": int(getattr(s, "native_adaptive_min_samples", 20)),
            "alpha": float(getattr(s, "native_adaptive_ewma_alpha", 0.3)),
        }
    except Exception:
        return defaults


def _sync_adaptive_tunables() -> None:
    """Push adaptive settings into the metrics module once (EWMA alpha/min samples)."""
    global _ADAPTIVE_SYNCED
    if _ADAPTIVE_SYNCED:
        return
    _ADAPTIVE_SYNCED = True
    try:
        from victor.native.observability import (
            set_adaptive_ewma_alpha,
            set_adaptive_min_samples,
        )

        s = _get_adaptive_settings()
        set_adaptive_ewma_alpha(s["alpha"])
        set_adaptive_min_samples(s["min_samples"])
    except Exception:
        logger.debug("adaptive tunable sync failed", exc_info=True)


def _adaptive_preference(operation: str) -> Optional[str]:
    """Return an adaptive backend hint from observed EWMA, or None to defer."""
    try:
        s = _get_adaptive_settings()
        if not s["enabled"]:
            return None
        _sync_adaptive_tunables()
        from victor.native.observability import get_operation_stats

        stats = get_operation_stats(operation)
        if stats is None:
            return None
        return stats.preferred_backend(min_samples=s["min_samples"])
    except Exception:
        return None


def calibrate_operation(
    operation: str,
    rust_fn: Optional[Callable[..., Any]] = None,
    python_fn: Optional[Callable[..., Any]] = None,
    inputs: Optional[List[Any]] = None,
    rounds: int = 25,
) -> Dict[str, float]:
    """Seed the adaptive EWMA for ``operation`` by timing both backends.

    Optional warmup: run each backend over ``inputs`` (repeated to ``rounds``
    total calls) and record the per-call latencies as known-backend samples, so
    adaptive dispatch has data immediately on cold start instead of waiting for
    organic traffic. No-op (returns zeros) if a backend or inputs are missing.

    Args:
        operation: Operation name (must match the name used at dispatch time).
        rust_fn: Callable running the Rust path. Takes one ``inputs`` element.
        python_fn: Callable running the Python fallback path.
        inputs: Sample inputs to time against.
        rounds: Approximate total calls per backend.

    Returns:
        ``{"rust_ms_avg": float, "python_ms_avg": float}`` of measured averages.
    """
    import time as _time

    from victor.native.observability import NativeMetrics

    metrics = NativeMetrics.get_instance()
    result: Dict[str, float] = {"rust_ms_avg": 0.0, "python_ms_avg": 0.0}
    if not inputs:
        return result

    for backend_name, fn in (("rust", rust_fn), ("python", python_fn)):
        if fn is None:
            continue
        times: List[float] = []
        work = inputs * max(1, rounds // max(1, len(inputs)))
        for inp in work:
            t0 = _time.perf_counter()
            try:
                fn(inp)
            except Exception:
                continue
            times.append((_time.perf_counter() - t0) * 1000.0)
        for ms in times:
            metrics.record_call(operation, ms, used_rust=(backend_name == "rust"))
        if times:
            result[f"{backend_name}_ms_avg"] = sum(times) / len(times)
    return result


def get_all_benchmarks() -> Dict[str, Dict[str, Any]]:
    """Get all benchmark data for display/debugging.

    Returns:
        Dict mapping operation names to benchmark info
    """
    return {
        name: {
            "rust_ms": b.rust_ms,
            "python_ms": b.python_ms,
            "speedup": f"{b.speedup:.1f}x",
            "preferred": b.preferred,
            "override": _accelerator_overrides.get(name),
            "effective": get_preferred_backend(name),
            "notes": b.notes,
        }
        for name, b in ACCELERATOR_BENCHMARKS.items()
    }


# =============================================================================
# PROTOCOL-BASED DISPATCH (DEPRECATED - Unused)
# =============================================================================
#
# NOTE: The following protocol-based dispatch functions are currently unused
# throughout the codebase. All callers use the flat function dispatch from
# victor.processing.native modules (similarity.py, chunking.py, etc.) instead.
#
# These functions are kept for potential future use (dependency injection,
# testing) but add maintenance burden. Consider removing in v0.9.0 if still
# unused, or alternatively migrate all callers to use this layer consistently.
#
# Current state (2025-04):
# - get_symbol_extractor: Unused (no callers)
# - get_argument_normalizer: Unused (no callers)
# - get_similarity_computer: Unused (no callers)
# - get_text_chunker: Unused (no callers)
# - get_token_counter: Unused (no callers)
# - get_context_fitter: Unused (no callers)
#
# Flat dispatch (Layer A) is the recommended approach for all new code.
# =============================================================================


def get_symbol_extractor(backend: Optional[str] = None) -> "SymbolExtractorProtocol":
    """Get a symbol extractor implementation.

    Args:
        backend: Explicit backend choice ("rust" or "python").
                 If None, uses Rust if available, otherwise Python.

    Returns:
        SymbolExtractorProtocol implementation
    """
    from victor.native.protocols import SymbolExtractorProtocol

    if backend == "rust" or (backend is None and _NATIVE_AVAILABLE):
        try:
            from victor.native.rust.symbol_extractor import RustSymbolExtractor

            return RustSymbolExtractor()
        except ImportError:
            pass

    from victor.native.python.symbol_extractor import PythonSymbolExtractor

    return PythonSymbolExtractor()


def get_argument_normalizer(
    backend: Optional[str] = None,
) -> "ArgumentNormalizerProtocol":
    """Get an argument normalizer implementation.

    Args:
        backend: Explicit backend choice ("rust" or "python").
                 If None, uses Rust if available, otherwise Python.

    Returns:
        ArgumentNormalizerProtocol implementation
    """
    from victor.native.protocols import ArgumentNormalizerProtocol

    if backend == "rust" or (backend is None and _NATIVE_AVAILABLE):
        try:
            from victor.native.rust.arg_normalizer import RustArgumentNormalizer

            return RustArgumentNormalizer()
        except ImportError:
            pass

    from victor.native.python.arg_normalizer import PythonArgumentNormalizer

    return PythonArgumentNormalizer()


def get_similarity_computer(
    backend: Optional[str] = None,
) -> "SimilarityComputerProtocol":
    """Get a similarity computer implementation.

    Note: Benchmark data shows NumPy+BLAS (Python) is ~6x faster than Rust FFI
    for batch similarity operations.

    Args:
        backend: Explicit backend choice ("rust" or "python").
                 If None, uses benchmark-based preference (Python for similarity).

    Returns:
        SimilarityComputerProtocol implementation
    """
    from victor.native.protocols import SimilarityComputerProtocol

    effective_backend = backend or get_preferred_backend("batch_similarity")

    if effective_backend == "rust" and _NATIVE_AVAILABLE:
        try:
            from victor.native.rust.similarity import RustSimilarityComputer

            return RustSimilarityComputer()
        except ImportError:
            pass

    from victor.native.python.similarity import PythonSimilarityComputer

    return PythonSimilarityComputer()


def get_text_chunker(backend: Optional[str] = None) -> "TextChunkerProtocol":
    """Get a text chunker implementation.

    Note: Benchmark data shows Rust is ~5x faster for text chunking.

    Args:
        backend: Explicit backend choice ("rust" or "python").
                 If None, uses benchmark-based preference (Rust for chunking).

    Returns:
        TextChunkerProtocol implementation
    """
    from victor.native.protocols import TextChunkerProtocol

    effective_backend = backend or get_preferred_backend("chunk_with_overlap")

    if effective_backend == "rust" and _NATIVE_AVAILABLE:
        try:
            from victor.native.rust.chunker import RustTextChunker

            return RustTextChunker()
        except ImportError:
            pass

    from victor.native.python.chunker import PythonTextChunker

    return PythonTextChunker()


def get_ast_indexer(backend: Optional[str] = None) -> "AstIndexerProtocol":
    """Get an AST indexer implementation.

    Args:
        backend: Explicit backend choice ("rust" or "python").
                 If None, uses Rust if available, otherwise Python.

    Returns:
        AstIndexerProtocol implementation
    """
    from victor.native.protocols import AstIndexerProtocol

    if backend == "rust" or (backend is None and _NATIVE_AVAILABLE):
        try:
            from victor.native.rust.ast_indexer import RustAstIndexer

            return RustAstIndexer()
        except ImportError:
            pass

    from victor.native.python.ast_indexer import PythonAstIndexer

    return PythonAstIndexer()


# Convenience singletons for common use cases
_symbol_extractor_instance: Optional["SymbolExtractorProtocol"] = None
_argument_normalizer_instance: Optional["ArgumentNormalizerProtocol"] = None
_similarity_computer_instance: Optional["SimilarityComputerProtocol"] = None
_text_chunker_instance: Optional["TextChunkerProtocol"] = None
_ast_indexer_instance: Optional["AstIndexerProtocol"] = None
_token_counter_instance: Optional["TokenCounterProtocol"] = None
_context_fitter_instance: Optional["ContextFitterProtocol"] = None


def get_default_symbol_extractor() -> "SymbolExtractorProtocol":
    """Get the default symbol extractor singleton."""
    global _symbol_extractor_instance
    if _symbol_extractor_instance is None:
        _symbol_extractor_instance = get_symbol_extractor()
    return _symbol_extractor_instance


def get_default_argument_normalizer() -> "ArgumentNormalizerProtocol":
    """Get the default argument normalizer singleton."""
    global _argument_normalizer_instance
    if _argument_normalizer_instance is None:
        _argument_normalizer_instance = get_argument_normalizer()
    return _argument_normalizer_instance


def get_default_similarity_computer() -> "SimilarityComputerProtocol":
    """Get the default similarity computer singleton."""
    global _similarity_computer_instance
    if _similarity_computer_instance is None:
        _similarity_computer_instance = get_similarity_computer()
    return _similarity_computer_instance


def get_default_text_chunker() -> "TextChunkerProtocol":
    """Get the default text chunker singleton."""
    global _text_chunker_instance
    if _text_chunker_instance is None:
        _text_chunker_instance = get_text_chunker()
    return _text_chunker_instance


def get_default_ast_indexer() -> "AstIndexerProtocol":
    """Get the default AST indexer singleton."""
    global _ast_indexer_instance
    if _ast_indexer_instance is None:
        _ast_indexer_instance = get_ast_indexer()
    return _ast_indexer_instance


def get_default_token_counter() -> "TokenCounterProtocol":
    """Get the default token counter singleton."""
    global _token_counter_instance
    if _token_counter_instance is None:
        _token_counter_instance = get_token_counter()
    return _token_counter_instance


def get_default_context_fitter() -> "ContextFitterProtocol":
    """Get the default context fitter singleton."""
    global _context_fitter_instance
    if _context_fitter_instance is None:
        _context_fitter_instance = get_context_fitter()
    return _context_fitter_instance


def get_token_counter(backend: Optional[str] = None) -> "TokenCounterProtocol":
    """Get a token counter implementation.

    Args:
        backend: Explicit backend choice ("rust" or "python").
                 If None, uses Rust if available, otherwise Python.

    Returns:
        TokenCounterProtocol implementation
    """
    effective_backend = backend or get_preferred_backend("token_counting")

    if effective_backend == "rust" and _NATIVE_AVAILABLE:
        try:
            from victor.native.rust.tokenizer import RustTokenCounter

            return RustTokenCounter()
        except ImportError:
            pass

    from victor.native.python.tokenizer import PythonTokenCounter

    return PythonTokenCounter()


def get_context_fitter(backend: Optional[str] = None) -> "ContextFitterProtocol":
    """Get a context fitter implementation.

    Args:
        backend: Explicit backend choice ("rust" or "python").
                 If None, uses Rust if available, otherwise Python.

    Returns:
        ContextFitterProtocol implementation
    """
    effective_backend = backend or get_preferred_backend("context_fitting")

    if effective_backend == "rust" and _NATIVE_AVAILABLE:
        try:
            from victor.native.rust.context_fitter import RustContextFitter

            return RustContextFitter()
        except ImportError:
            pass

    from victor.native.python.context_fitter import PythonContextFitter

    return PythonContextFitter()


def get_content_hasher(
    normalize_whitespace: bool = True,
    case_insensitive: bool = False,
    hash_length: int = 16,
    remove_punctuation: bool = False,
) -> "ContentHasherProtocol":
    """Get a content hasher implementation with configured normalization.

    Args:
        normalize_whitespace: Collapse multiple whitespace to single space
        case_insensitive: Convert to lowercase before hashing
        hash_length: Number of hex chars to return (1-64)
        remove_punctuation: Remove trailing punctuation

    Returns:
        ContentHasherProtocol implementation
    """
    from victor.core.utils.content_hasher import ContentHasher

    return ContentHasher(
        normalize_whitespace=normalize_whitespace,
        case_insensitive=case_insensitive,
        hash_length=hash_length,
        remove_punctuation=remove_punctuation,
    )


# Content hasher preset singletons
_content_hasher_fuzzy: Optional["ContentHasherProtocol"] = None
_content_hasher_exact: Optional["ContentHasherProtocol"] = None


def get_default_content_hasher_fuzzy() -> "ContentHasherProtocol":
    """Get the default fuzzy content hasher singleton."""
    global _content_hasher_fuzzy
    if _content_hasher_fuzzy is None:
        from victor.core.utils.content_hasher import HasherPresets

        _content_hasher_fuzzy = HasherPresets.text_fuzzy()
    return _content_hasher_fuzzy


def get_default_content_hasher_exact() -> "ContentHasherProtocol":
    """Get the default exact content hasher singleton."""
    global _content_hasher_exact
    if _content_hasher_exact is None:
        from victor.core.utils.content_hasher import HasherPresets

        _content_hasher_exact = HasherPresets.exact_match()
    return _content_hasher_exact


def reset_protocol_singletons() -> None:
    """Reset all protocol singletons.

    Useful for testing to ensure clean state between tests.
    """
    global _symbol_extractor_instance, _argument_normalizer_instance
    global _similarity_computer_instance, _text_chunker_instance
    global _ast_indexer_instance, _content_hasher_fuzzy, _content_hasher_exact
    global _token_counter_instance, _context_fitter_instance
    _symbol_extractor_instance = None
    _argument_normalizer_instance = None
    _similarity_computer_instance = None
    _text_chunker_instance = None
    _ast_indexer_instance = None
    _content_hasher_fuzzy = None
    _content_hasher_exact = None
    _token_counter_instance = None
    _context_fitter_instance = None

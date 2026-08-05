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

"""Performance plugin architecture for victor-coding.

This module enables registering and using performance-improved backend
implementations (typically Rust-based via PyO3) with automatic fallback
to pure Python implementations.

Main Components:
- PerformanceBackendRegistry: Registry for backend implementations
- BackendFactory: Factory for creating backend instances with fallback
- Protocol definitions: Interfaces for chunker, regex

NOTE: Tree-sitter symbol extraction uses Python's tree-sitter bindings directly,
not Rust. Python's bindings to the C library are already efficient.

Usage:
    from victor_coding.performance import BackendFactory

    # Create chunker (tries native first, falls back to Python)
    chunker = BackendFactory.create_chunker(config)

    # Check if native backends are available
    if BackendFactory.has_native_chunker():
        logger.info("Using native chunker")
"""

from victor_coding.performance.factory import BackendFactory, PythonRegexProcessor
from victor_coding.performance.protocols import (
    BackendCapabilities,
    ChunkInfo,
    ExtractedData,
    FastChunkerProtocol,
    FastRegexProcessorProtocol,
    IndexedFileData,
    get_backend_capabilities,
)
from victor_coding.performance.registry import (
    BackendConfig,
    BackendRegistration,
    PerformanceBackendRegistry,
    auto_register_native_backends,
)
from victor_coding.performance.settings import (
    PerformanceSettings,
    clear_performance_settings_cache,
    get_performance_settings,
)

__all__ = [
    # Factory
    "BackendFactory",
    "PythonRegexProcessor",
    # Protocols
    "FastChunkerProtocol",
    "FastRegexProcessorProtocol",
    # Data types
    "IndexedFileData",
    "ExtractedData",
    "ChunkInfo",
    "BackendCapabilities",
    # Registry
    "PerformanceBackendRegistry",
    "BackendRegistration",
    "BackendConfig",
    "auto_register_native_backends",
    "get_backend_capabilities",
    # Settings
    "PerformanceSettings",
    "get_performance_settings",
    "clear_performance_settings_cache",
]

# Auto-register native backends on import
auto_register_native_backends()

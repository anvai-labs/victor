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

"""Rust implementation wrappers.

These wrappers provide protocol-compliant interfaces around the Rust
native functions, adding observability hooks and maintaining compatibility
with the Python fallback implementations.
"""

from victor.native.rust.arg_normalizer import RustArgumentNormalizer
from victor.native.rust.ast_indexer import RustAstIndexer
from victor.native.rust.chunker import RustTextChunker
from victor.native.rust.context_fitter import RustContextFitter
from victor.native.rust.similarity import RustSimilarityComputer
from victor.native.rust.tokenizer import RustTokenCounter

__all__ = [
    "RustArgumentNormalizer",
    "RustAstIndexer",
    "RustContextFitter",
    "RustSimilarityComputer",
    "RustTextChunker",
    "RustTokenCounter",
]

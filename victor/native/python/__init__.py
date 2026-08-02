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

"""Pure Python fallback implementations for native accelerations.

These implementations serve as:
1. Reference implementations for testing
2. Fallbacks when Rust extensions are not compiled
3. Documentation of expected behavior

All implementations follow the protocols defined in victor.native.protocols.
"""

from victor.native.python.symbol_extractor import PythonSymbolExtractor
from victor.native.python.arg_normalizer import PythonArgumentNormalizer
from victor.native.python.similarity import PythonSimilarityComputer
from victor.native.python.chunker import PythonTextChunker
from victor.native.python.ast_indexer import PythonAstIndexer
from victor.native.python.context_fitter import PythonContextFitter
from victor.native.python.tokenizer import PythonTokenCounter

__all__ = [
    "PythonSymbolExtractor",
    "PythonArgumentNormalizer",
    "PythonSimilarityComputer",
    "PythonTextChunker",
    "PythonAstIndexer",
    "PythonContextFitter",
    "PythonTokenCounter",
]

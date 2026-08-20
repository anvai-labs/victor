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

"""Pure Python token counter implementation.

Provides token counting using tiktoken when available,
falling back to word-based estimation.
"""

from __future__ import annotations

from typing import List, Optional

from victor.native.observability import InstrumentedAccelerator

# Try to use tiktoken for accurate counting
try:
    import tiktoken

    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False

# Cached tiktoken encoder
_tiktoken_encoder = None


def _get_tiktoken_encoder():
    """Get or create cached tiktoken encoder."""
    global _tiktoken_encoder
    if _tiktoken_encoder is None and _TIKTOKEN_AVAILABLE:
        _tiktoken_encoder = tiktoken.get_encoding("cl100k_base")
    return _tiktoken_encoder


class PythonTokenCounter(InstrumentedAccelerator):
    """Pure Python implementation of TokenCounterProtocol.

    Uses tiktoken when available for accurate BPE token counting.
    Falls back to word-based estimation otherwise.
    """

    def __init__(self) -> None:
        super().__init__(backend="python")
        self._version = "1.0.0"
        self._use_tiktoken = _TIKTOKEN_AVAILABLE

    def get_version(self) -> Optional[str]:
        return self._version

    def count_tokens(self, text: str) -> int:
        """Count tokens using exact BPE tokenization.

        Uses tiktoken when available, otherwise word-based estimation.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        with self._timed_call("token_counting"):
            encoder = _get_tiktoken_encoder()
            if encoder is not None:
                return len(encoder.encode(text))
            return len(text.split()) * 13 // 10

    def count_tokens_fast(self, text: str) -> int:
        """Count tokens using fast approximate method.

        Uses word-based estimation for maximum speed.

        Args:
            text: Text to count tokens for

        Returns:
            Approximate number of tokens
        """
        with self._timed_call("token_counting_fast"):
            return len(text.split()) * 13 // 10

    def count_tokens_batch(self, texts: List[str]) -> List[int]:
        """Count tokens for multiple texts in batch.

        Args:
            texts: List of texts to count tokens for

        Returns:
            List of token counts, one per input text
        """
        with self._timed_call("token_counting_batch"):
            encoder = _get_tiktoken_encoder()
            if encoder is not None:
                return [len(encoder.encode(text)) for text in texts]
            return [len(text.split()) * 13 // 10 for text in texts]

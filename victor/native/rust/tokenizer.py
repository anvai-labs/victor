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

"""Rust token counter wrapper.

Provides a protocol-compliant wrapper around the Rust BPE tokenizer.
The wrapper delegates to victor_native functions while maintaining
the TokenCounterProtocol interface.

Performance characteristics:
- count_tokens: 2-5x faster (Rust BPE implementation)
- count_tokens_batch: crosses the FFI boundary ONCE via the native
  ``count_tokens_fast_batch`` (rayon-parallel), instead of looping the
  single-text entry point from Python (which crossed once per text).
  Falls back to the per-element loop when the batch symbol is absent
  (older native builds), per the graceful-degradation mandate.
"""

from __future__ import annotations

from typing import List, Optional

import victor_native

from victor.native.observability import InstrumentedAccelerator


class RustTokenCounter(InstrumentedAccelerator):
    """Rust implementation of TokenCounterProtocol.

    Wraps the high-performance Rust BPE tokenizer with
    protocol-compliant interface.

    Performance characteristics:
    - count_tokens: 2-5x faster than tiktoken
    - count_tokens_fast: Optimized for speed over accuracy
    - count_tokens_batch: Amortized FFI overhead for batch operations
    """

    def __init__(self) -> None:
        super().__init__(backend="rust")
        self._version = victor_native.__version__

    def get_version(self) -> Optional[str]:
        return self._version

    def count_tokens(self, text: str) -> int:
        """Count tokens using exact BPE tokenization.

        Delegates to Rust BPE implementation.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        with self._timed_call("token_counting"):
            return victor_native.count_tokens_fast(text)

    def count_tokens_fast(self, text: str) -> int:
        """Count tokens using fast approximate method.

        Delegates to Rust native counting.

        Args:
            text: Text to count tokens for

        Returns:
            Approximate number of tokens
        """
        with self._timed_call("token_counting_fast"):
            return victor_native.count_tokens_fast(text)

    def count_tokens_batch(self, texts: List[str]) -> List[int]:
        """Count tokens for multiple texts in batch.

        Uses the native ``count_tokens_fast_batch`` so the FFI boundary is
        crossed once for the whole batch (rayon-parallel internally), rather
        than once per text. Falls back to the per-element loop when the batch
        symbol is unavailable (older native build / version skew).

        Args:
            texts: List of texts to count tokens for

        Returns:
            List of token counts, one per input text
        """
        with self._timed_call("token_counting_batch"):
            batch_fn = getattr(victor_native, "count_tokens_fast_batch", None)
            if batch_fn is not None:
                return list(batch_fn(list(texts)))
            return [victor_native.count_tokens_fast(text) for text in texts]

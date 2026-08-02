# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

pytest.importorskip("victor_coding.codebase.symbol_resolver")

from victor_coding.codebase.symbol_resolver import SymbolResolver


def test_symbol_resolver_prefers_other_file():
    resolver = SymbolResolver()
    resolver.ingest(
        [
            "symbol:foo.py:helper",
            "symbol:bar.py:helper",
            "symbol:bar.py:Class.method",
        ]
    )

    # Prefer match not in the same file
    resolved = resolver.resolve("helper", preferred_file="foo.py")
    assert resolved == "symbol:bar.py:helper"

    # Short-name heuristic for dotted names
    resolved_short = resolver.resolve("method")
    assert resolved_short == "symbol:bar.py:Class.method"


def test_symbol_resolver_clear():
    resolver = SymbolResolver()
    resolver.ingest(["symbol:foo.py:helper"])
    resolver.clear()
    assert resolver.resolve("helper") is None

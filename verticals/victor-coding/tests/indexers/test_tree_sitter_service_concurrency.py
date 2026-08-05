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

"""Concurrency contract: parsers are per-thread, languages are shared.

These tests are the key safety net against regressing the global-mutable
parser-cache pattern that lived in tree_sitter_manager.py prior to TSA-1.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

pytest.importorskip("victor_coding.codebase.tree_sitter_service")

from victor_coding.codebase.tree_sitter_service import (
    TreeSitterService,
    _reset_for_tests,
    get_tree_sitter_service,
)


@pytest.fixture(autouse=True)
def _isolate_service():
    _reset_for_tests()
    yield
    _reset_for_tests()


def _require_python_grammar():
    svc = get_tree_sitter_service()
    if not svc.supports_language("python"):
        pytest.skip("tree-sitter-python not installed")


class TestParserPerThread:
    """``get_parser`` returns a distinct Parser per OS thread."""

    def test_parsers_are_unique_per_thread(self):
        _require_python_grammar()
        svc = get_tree_sitter_service()
        worker_count = 8
        # Barrier forces every worker to be a distinct OS thread by making
        # them rendezvous before returning. Without it ThreadPoolExecutor.map
        # may reuse a single thread if the work returns fast enough.
        barrier = threading.Barrier(worker_count)

        def grab_parser(_: int) -> int:
            parser = svc.get_parser("python")
            assert parser is not None
            barrier.wait(timeout=5)
            return id(parser)

        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            ids = list(pool.map(grab_parser, range(worker_count)))

        # Each worker thread must own its own Parser instance. If a parser is
        # shared across two threads we'd see two ids collide.
        assert len(set(ids)) == worker_count

    def test_parsers_reused_within_same_thread(self):
        _require_python_grammar()
        svc = get_tree_sitter_service()

        def grab_two() -> tuple[int, int]:
            a = svc.get_parser("python")
            b = svc.get_parser("python")
            assert a is not None and b is not None
            return id(a), id(b)

        with ThreadPoolExecutor(max_workers=1) as pool:
            first_id, second_id = pool.submit(grab_two).result()

        assert first_id == second_id


class TestLanguageSharedAcrossThreads:
    """``get_language`` returns the same Language across threads."""

    def test_language_is_shared(self):
        _require_python_grammar()
        svc = get_tree_sitter_service()

        def grab_language(_: int) -> int:
            lang = svc.get_language("python")
            assert lang is not None
            return id(lang)

        with ThreadPoolExecutor(max_workers=4) as pool:
            ids = list(pool.map(grab_language, range(4)))

        assert len(set(ids)) == 1


class TestConcurrentParse:
    """Parsing the same content from many threads succeeds and yields trees."""

    def test_parallel_parses_complete(self):
        _require_python_grammar()
        svc = get_tree_sitter_service()
        source = b"def foo():\n    return 1\n"

        def parse_once(_: int) -> bool:
            parsed = svc.parse(source, "python")
            return parsed is not None and parsed.root_node is not None

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(parse_once, range(32)))

        assert all(results)


def test_service_constructor_isolated_caches():
    """A freshly-constructed service does not share caches with the singleton."""
    singleton = get_tree_sitter_service()
    fresh = TreeSitterService()
    assert fresh is not singleton
    assert fresh._languages is not singleton._languages
    assert fresh._queries is not singleton._queries

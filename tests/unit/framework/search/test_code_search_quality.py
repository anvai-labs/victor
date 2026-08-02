# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Tests for shared code search quality helpers."""

from victor.framework.search import (
    enrich_code_search_results,
    rerank_code_search_results,
)


def test_enrich_code_search_results_resolves_line_window_for_opaque_payload(
    tmp_path,
) -> None:
    """Opaque semantic payloads should be replaced with grounded file snippets."""
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    target = source_dir / "parser.py"
    target.write_text(
        "import json\n\n"
        "def parse_json(data):\n"
        "    if not data:\n"
        "        return {}\n"
        "    return json.loads(data)\n",
        encoding="utf-8",
    )

    results = [
        {
            "file_path": "src/parser.py",
            "line_number": 3,
            "content": "symbol:src/parser.py:parse_json",
            "metadata": {
                "unified_id": "symbol:src/parser.py:parse_json",
                "end_line": 6,
            },
        }
    ]

    enriched, metadata = enrich_code_search_results(results, root_path=tmp_path)

    assert metadata["chunking_strategy"] == "symbol_only"
    assert metadata["snippet_strategy"] == "line_window"
    assert metadata["snippet_enriched_hits"] == 1
    assert "def parse_json(data):" in enriched[0]["snippet"]
    assert "return json.loads(data)" in enriched[0]["content"]
    assert enriched[0]["metadata"]["snippet_source"] == "line_window"
    assert enriched[0]["metadata"]["content_source"] == "line_window"


def test_rerank_code_search_results_prefers_implementation_hits_and_diversity() -> None:
    """Utility reranking should demote tests and repeated duplicate snippets."""
    results = [
        {
            "file_path": "tests/test_parser.py",
            "symbol_name": "test_parse_json",
            "snippet": "def test_parse_json(): assert parse_json('{}') == {}",
            "combined_score": 0.92,
        },
        {
            "file_path": "victor/parser.py",
            "symbol_name": "parse_json",
            "snippet": "def parse_json(data): return json.loads(data)",
            "combined_score": 0.89,
        },
        {
            "file_path": "victor/parser.py",
            "symbol_name": "parse_json_or_none",
            "snippet": "def parse_json(data): return json.loads(data)",
            "combined_score": 0.88,
        },
    ]

    reranked, metadata = rerank_code_search_results(results, query="where is parse_json defined")

    assert reranked[0]["file_path"] == "victor/parser.py"
    assert reranked[-1]["symbol_name"] == "parse_json_or_none"
    assert metadata["strategy"] == "bounded_code_utility"
    assert metadata["file_diversity"] == 2
    assert metadata["repeated_file_hits"] == 1
    assert metadata["duplicate_snippet_hits"] == 1
    assert metadata["implementation_hits"] == 2
    assert metadata["identifier_hits"] >= 2

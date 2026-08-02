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

"""Unified ``rag`` command tool — bash-style knowledge-base surface.

Parses ``rag ingest|search|query|list|delete|stats`` and delegates to the
victor-rag ``BaseTool`` classes. Unlike the ``@tool``-function verticals, the
rag tools are classes with an ``execute()`` method, so delegation instantiates
the resolved class and awaits ``execute(**kwargs)``. There is no shell
equivalent, so when victor-rag is absent the tool returns a graceful message.
Advertised only in the RAG vertical (``vertical_tools.yaml``).

Example commands:
    rag ingest --path docs/guide.md
    rag search "how does auth work"
    rag query "What is the login flow?" --no-synthesize
    rag list
    rag delete doc-42
    rag stats
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, Optional, Tuple

from victor.tools.base import AccessMode, DangerLevel, ExecutionCategory, Priority
from victor.tools.decorators import tool
from victor.tools.unified._vertical_resolver import resolve_vertical_callable
from victor.tools.unified.parser import split_command


class UnifiedRagParser(argparse.ArgumentParser):
    def error(self, message):  # type: ignore[override]
        self.print_usage(sys.stderr)
        raise ValueError(f"Argument parsing error: {message}")


def create_rag_parser() -> UnifiedRagParser:
    parser = UnifiedRagParser(
        prog="rag",
        description="Unified RAG knowledge-base operations.",
        exit_on_error=False,
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="The operation to perform")

    ingest = subparsers.add_parser("ingest", help="Ingest a file/url/content")
    ingest.add_argument("--path", default=None)
    ingest.add_argument("--url", default=None)
    ingest.add_argument("--content", default=None)
    ingest.add_argument("--type", default="text", help="text|markdown|code|pdf|html")
    ingest.add_argument("--doc-id", dest="doc_id", default=None)
    ingest.add_argument("--recursive", action="store_true")
    ingest.add_argument("--pattern", default="*")

    search = subparsers.add_parser("search", help="Search for relevant chunks")
    search.add_argument("query")
    search.add_argument("--k", default=10, type=int)

    query = subparsers.add_parser("query", help="Retrieve context / synthesize an answer")
    query.add_argument("question")
    query.add_argument("--k", default=5, type=int)
    query.add_argument(
        "--no-synthesize",
        dest="synthesize",
        action="store_false",
        help="Context only, no LLM",
    )
    query.add_argument("--provider", default=None)
    query.add_argument("--model", default=None)

    subparsers.add_parser("list", help="List documents")
    delete = subparsers.add_parser("delete", help="Delete a document")
    delete.add_argument("doc_id")
    subparsers.add_parser("stats", help="Knowledge-base statistics")

    return parser


# subcommand -> (entry-point key, fallback module, fallback class attr)
_RAG_TARGETS: Dict[str, Tuple[str, str, str]] = {
    "ingest": ("rag_ingest", "victor_rag.tools.ingest", "RAGIngestTool"),
    "search": ("rag_search", "victor_rag.tools.search", "RAGSearchTool"),
    "query": ("rag_query", "victor_rag.tools.query", "RAGQueryTool"),
    "list": ("rag_list", "victor_rag.tools.management", "RAGListTool"),
    "delete": ("rag_delete", "victor_rag.tools.management", "RAGDeleteTool"),
    "stats": ("rag_stats", "victor_rag.tools.management", "RAGStatsTool"),
}


def _kwargs_for(sub: str, parsed: argparse.Namespace) -> Dict[str, Any]:
    if sub == "ingest":
        return {
            "path": parsed.path,
            "url": parsed.url,
            "content": parsed.content,
            "doc_type": parsed.type,
            "doc_id": parsed.doc_id,
            "recursive": parsed.recursive,
            "pattern": parsed.pattern,
        }
    if sub == "search":
        return {"query": parsed.query, "k": parsed.k}
    if sub == "query":
        return {
            "question": parsed.question,
            "k": parsed.k,
            "synthesize": parsed.synthesize,
            "provider": parsed.provider,
            "model": parsed.model,
        }
    if sub == "delete":
        return {"doc_id": parsed.doc_id}
    return {}


def _format_tool_result(result: Any) -> str:
    """Normalize a BaseTool ToolResult (or dict/str) into a display string."""
    success = getattr(result, "success", None)
    if success is None and isinstance(result, dict):
        success = result.get("success")
    if success is False:
        err = getattr(result, "error", None) or (
            result.get("error") if isinstance(result, dict) else None
        )
        return f"### ❌ ERROR\n{err or 'rag operation failed'}"
    output = getattr(result, "output", None)
    if output is None and isinstance(result, dict):
        output = result.get("output")
    return str(output) if output is not None else str(result)


@tool(
    name="rag",
    category="rag",
    access_mode=AccessMode.MIXED,
    danger_level=DangerLevel.LOW,
    execution_category=ExecutionCategory.MIXED,
    priority=Priority.MEDIUM,
    keywords=[
        "rag",
        "knowledge base",
        "embedding",
        "vector",
        "retrieval",
        "semantic search",
    ],
    task_types=["search", "analysis", "action"],
)
async def rag_tool(cmd: str) -> str:
    """RAG domain (bash-style): ingest, search, query, list, delete, stats.
    Delegates to victor-rag. e.g. rag search "x" · rag ingest --path f.md · rag list.
    """
    parser = create_rag_parser()
    try:
        args_list = split_command(cmd)
        if args_list and args_list[0] == "rag":
            args_list = args_list[1:]
        parsed = parser.parse_args(args_list)
    except ValueError as e:
        return f"### ❌ ERROR\n{e}"
    except Exception as e:
        return f"### ❌ ERROR\nUnexpected error parsing command: {e}"

    if not parsed.subcommand:
        return (
            "### ❌ ERROR\nNo rag subcommand given. Use: rag ingest|search|query|list|delete|stats"
        )

    key, fallback_module, fallback_attr = _RAG_TARGETS[parsed.subcommand]
    cls, _src = resolve_vertical_callable(
        key, fallback_module=fallback_module, fallback_attr=fallback_attr
    )
    if cls is None:
        return (
            "### ❌ ERROR\nRAG operations require the victor-rag package, which is not installed. "
            "Install victor-rag to ingest/search/query a knowledge base."
        )

    try:
        instance = cls()
        result = await instance.execute(**_kwargs_for(parsed.subcommand, parsed))
    except Exception as e:
        return f"### ❌ ERROR\nrag {parsed.subcommand} failed: {e}"
    return _format_tool_result(result)


__all__ = ["rag_tool", "create_rag_parser"]

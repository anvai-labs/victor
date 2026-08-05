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

"""Tool advertisement folding policy.

Folded tools remain registered and executable, but are not advertised as
separate enabled schemas by default. Their common use cases are represented by
the target tool's description to reduce tool count and schema/token overhead.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Optional


@dataclass(frozen=True)
class ToolFold:
    """A tool that should be represented by another advertised tool."""

    target: str
    hint: str


FOLDED_TOOLS: dict[str, ToolFold] = {
    # Shell folds
    "database": ToolFold(
        target="db",
        hint="Use db query/tables/schema/describe for SQL and database inspection.",
    ),
    "dependency": ToolFold(
        target="shell",
        hint="Use shell with pip, pipdeptree, uv, poetry, npm, cargo, or equivalent package CLIs.",
    ),
    "test": ToolFold(
        target="shell",
        hint="Use shell with pytest, npm test, cargo test, go test, make test, or project test CLIs.",
    ),
    "rag_ingest": ToolFold(
        target="rag",
        hint="Use rag ingest --path/--url/--content to ingest into the knowledge base.",
    ),
    "rag_query": ToolFold(
        target="rag", hint='Use rag query "<question>" to query the knowledge base.'
    ),
    "rag_search": ToolFold(
        target="rag", hint='Use rag search "<query>" to search the knowledge base.'
    ),
    "rag_list": ToolFold(target="rag", hint="Use rag list to list knowledge-base documents."),
    "rag_delete": ToolFold(
        target="rag",
        hint="Use rag delete <doc_id> to remove a knowledge-base document.",
    ),
    "rag_stats": ToolFold(target="rag", hint="Use rag stats for knowledge-base statistics."),
    "rename": ToolFold(
        target="shell",
        hint="Use shell with standard tools (sed, fastmod) or edit for renaming symbols.",
    ),
    "inline": ToolFold(
        target="shell", hint="Use shell with standard tools (sed) or edit for inlining."
    ),
    "extract": ToolFold(
        target="shell", hint="Use shell with standard tools or edit for extraction."
    ),
    "organize_imports": ToolFold(
        target="shell",
        hint="Use shell with standard tools (isort, ruff) for organizing imports.",
    ),
    "scaffold": ToolFold(
        target="shell",
        hint="Use shell to scaffold projects via npx, cookiecutter, cargo new, or equivalent CLI tools.",
    ),
    "sandbox": ToolFold(
        target="shell",
        hint="Use shell with the --sandbox flag to execute isolated containerized commands.",
    ),
    # Integration folds
    "jira": ToolFold(
        target="integrations",
        hint="Use integrations tool to interact with Jira for ticket management.",
    ),
    "slack": ToolFold(target="integrations", hint="Use integrations tool to message Slack."),
    "teams": ToolFold(
        target="integrations", hint="Use integrations tool to message Microsoft Teams."
    ),
    # MCP folds
    "mcp": ToolFold(
        target="mcp_bridge",
        hint="Use mcp_bridge to access Model Context Protocol specific tools.",
    ),
    # Unified command-domain folds: granular primitives previously represented
    # by the fs/web/code bash-style surfaces. The fs domain has been REMOVED —
    # read/write/edit/ls/find are now first-class tools with named parameters,
    # exposed directly to the LLM (no subcommand parsing).
    "ls": ToolFold(target="shell", hint="Use shell with ls to list directories."),
    "find": ToolFold(target="shell", hint="Use shell with find to locate files."),
    "web_search": ToolFold(target="web", hint="Use web search <query> for web search."),
    "web_fetch": ToolFold(target="web", hint="Use web fetch <url> to fetch a URL as markdown."),
    # `search` is a deprecated back-compat shim. It stays folded (hidden from the
    # advertised schema) but gets an EMPTY hint so no deprecation text leaks into
    # `code`'s description — the LLM can't act on a deprecation, so telling it is
    # pure token noise. The DeprecationWarning in search_tool.py is for logs/devs.
    "search": ToolFold(target="code", hint=""),
}


def tool_folding_enabled() -> bool:
    """Return whether default advertisement folding is enabled."""
    raw = os.getenv("VICTOR_TOOL_FOLDING", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def get_fold(tool_name: str) -> Optional[ToolFold]:
    """Return fold metadata for a tool name, if it is folded."""
    if not tool_folding_enabled():
        return None
    return FOLDED_TOOLS.get(str(tool_name or "").strip())


def is_folded_tool(tool_name: str) -> bool:
    """Return whether ``tool_name`` is folded out of default advertisements."""
    return get_fold(tool_name) is not None


def should_advertise_tool(tool_name: str, *, include_folded: bool = False) -> bool:
    """Return whether a tool should be listed in default enabled tool surfaces."""
    return include_folded or not is_folded_tool(tool_name)


def folded_tool_names_for_target(target: str) -> list[str]:
    """Return folded tool names represented by ``target``."""
    if not tool_folding_enabled():
        return []
    return sorted(name for name, fold in FOLDED_TOOLS.items() if fold.target == target)


def folded_tool_hints_for_target(target: str) -> list[str]:
    """Return human-readable fold hints for a target tool description."""
    if not tool_folding_enabled():
        return []
    return [
        f"{name}: {fold.hint}"
        for name, fold in sorted(FOLDED_TOOLS.items())
        if fold.target == target and fold.hint
    ]

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

"""Research vertical compute handlers.

Domain-specific handlers for research workflows:
- web_scraper: Structured web content extraction
- citation_formatter: Reference formatting

Usage:
    from victor_research import handlers
    handlers.register_handlers()

    # In YAML workflow:
    - id: scrape_page
      type: compute
      handler: web_scraper
      inputs:
        url: $ctx.target_url
        selectors:
          title: h1
          content: article
      output: scraped_data
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional

from victor_contracts.workflows import (
    ComputeHandlerRegistrar,
    ExecutorNodeStatus,
    NodeResult,
    register_compute_handlers,
)

if TYPE_CHECKING:
    from victor_contracts.verticals.protocols.tools import (
        ToolRegistryProtocol as ToolRegistry,
    )
    from victor_contracts.workflows import ComputeNodeProtocol as ComputeNode
    from victor_contracts.workflows import WorkflowContextProtocol as WorkflowContext

logger = logging.getLogger(__name__)


@dataclass
class WebScraperHandler:
    """Structured web content extraction.

    Fetches and parses web content.

    Example YAML:
        - id: scrape_page
          type: compute
          handler: web_scraper
          inputs:
            url: $ctx.target_url
            selectors:
              title: h1
              content: article
          output: scraped_data
    """

    async def __call__(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> "NodeResult":
        start_time = time.time()

        url = node.input_mapping.get("url", "")
        if isinstance(url, str) and url.startswith("$ctx."):
            url = context.get(url[5:]) or url

        selectors = node.input_mapping.get("selectors", {})

        try:
            result = await tool_registry.execute(
                "web_fetch",
                url=url,
                selectors=selectors,
            )

            output = {
                "url": url,
                "success": result.success,
                "data": result.output if result.success else None,
                "error": result.error if not result.success else None,
            }

            output_key = node.output_key or node.id
            context.set(output_key, output)

            return NodeResult(
                node_id=node.id,
                status=(
                    ExecutorNodeStatus.COMPLETED if result.success else ExecutorNodeStatus.FAILED
                ),
                output=output,
                duration_seconds=time.time() - start_time,
                tool_calls_used=1,
            )
        except Exception as e:
            return NodeResult(
                node_id=node.id,
                status=ExecutorNodeStatus.FAILED,
                error=str(e),
                duration_seconds=time.time() - start_time,
            )


@dataclass
class CitationFormatterHandler:
    """Format references and citations.

    Converts references to standard citation formats.

    Example YAML:
        - id: format_refs
          type: compute
          handler: citation_formatter
          inputs:
            references: $ctx.raw_references
            style: apa
          output: formatted_citations
    """

    async def __call__(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> "NodeResult":
        start_time = time.time()

        refs_key = node.input_mapping.get("references")
        references = context.get(refs_key) if refs_key else []
        style = node.input_mapping.get("style", "apa")

        try:
            ref_list = references if isinstance(references, list) else [references]
            formatted = [self._format_citation(ref, style) for ref in ref_list]

            output = {
                "style": style,
                "count": len(formatted),
                "citations": formatted,
            }

            output_key = node.output_key or node.id
            context.set(output_key, output)

            return NodeResult(
                node_id=node.id,
                status=ExecutorNodeStatus.COMPLETED,
                output=output,
                duration_seconds=time.time() - start_time,
            )
        except Exception as e:
            return NodeResult(
                node_id=node.id,
                status=ExecutorNodeStatus.FAILED,
                error=str(e),
                duration_seconds=time.time() - start_time,
            )

    def _format_citation(self, ref: Dict[str, Any], style: str) -> str:
        """Format a single citation."""
        if not isinstance(ref, dict):
            return str(ref)

        authors = ref.get("authors", ["Unknown"])
        year = ref.get("year", "n.d.")
        title = ref.get("title", "Untitled")
        source = ref.get("source", "")

        if style == "apa":
            author_str = ", ".join(authors[:3])
            if len(authors) > 3:
                author_str += " et al."
            return f"{author_str} ({year}). {title}. {source}"
        elif style == "mla":
            author_str = ", ".join(authors[:3])
            return f'{author_str}. "{title}." {source}, {year}.'
        elif style == "chicago":
            author_str = ", ".join(authors)
            return f"{author_str}. {title}. {source}, {year}."
        else:
            return f"{authors[0] if authors else 'Unknown'} ({year}). {title}"


HANDLERS = {
    "web_scraper": WebScraperHandler(),
    "citation_formatter": CitationFormatterHandler(),
}


def register_handlers(
    registrar: Optional[ComputeHandlerRegistrar] = None,
):
    """Register Research handlers through an explicit host-side registrar."""
    registered = register_compute_handlers(registrar, HANDLERS)
    if registrar is not None:
        for name in HANDLERS:
            logger.debug(f"Registered Research handler: {name}")
    return registered


__all__ = [
    "WebScraperHandler",
    "CitationFormatterHandler",
    "HANDLERS",
    "register_handlers",
]

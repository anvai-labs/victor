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

"""MIPROv2-inspired few-shot prompt adaptation.

This is not a full implementation of the original MIPROv2 optimizer with
proposal search and Bayesian optimization. In the current framework shape,
it serves as a query-aware few-shot retriever that mines successful traces,
prefers stronger examples, enforces light diversity, and bounds the injected
few-shot payload size.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_FEW_SHOT_TOOL_ALIASES = {
    "code_search": "search",
    "semantic_code_search": "search",
    "grep": "search",
    "find": "search",
    "run_tests": "code",
}


def _normalize_few_shot_tools(tools: List[str]) -> List[str]:
    """Render mined traces with currently advertised grouped tool names."""
    normalized: List[str] = []
    seen = set()
    for tool in tools:
        name = _FEW_SHOT_TOOL_ALIASES.get(tool, tool)
        if not name or name in seen:
            continue
        seen.add(name)
        normalized.append(name)
    return normalized


class MIPROv2Strategy:
    """MIPROv2-inspired few-shot demonstration mining strategy.

    Uses embedding-based KNN selection to find the most relevant
    traces for a given query, producing input-aware few-shot examples.

    Attributes:
        _max_examples: Maximum number of few-shot examples to include.
    """

    def __init__(
        self,
        max_examples: int = 3,
        min_completion_score: float = 0.7,
        example_diversity: bool = True,
        max_example_chars: int = 400,
    ):
        """Initialize the MIPROv2 strategy.

        Args:
            max_examples: Maximum number of few-shot examples to produce.
            min_completion_score: Minimum trace score required for inclusion.
            example_diversity: Whether to deduplicate near-identical traces.
            max_example_chars: Maximum size of the formatted example block.
        """
        self._max_examples = max_examples
        self._min_completion_score = min_completion_score
        self._example_diversity = example_diversity
        self._max_example_chars = max_example_chars

    def select_similar_traces(self, traces: List[Any], query: str, top_k: int = 3) -> List[Any]:
        """KNNFewShot: select traces most similar to the current query.

        Embeds trace descriptions and the query, then returns the top-k
        most similar traces by cosine similarity. Falls back to returning
        the first top_k traces if embedding is unavailable.

        Args:
            traces: List of trace objects with task_type, tool_call_details,
                and tool_failures attributes.
            query: The current user query to match against.
            top_k: Number of traces to return.

        Returns:
            List of selected traces, length <= top_k.
        """
        if not traces:
            return []

        embedded = self._embed_traces(traces)
        if not embedded:
            # Fallback: return first top_k traces (score-based ordering
            # is the caller's responsibility)
            return traces[:top_k]

        try:
            from victor.storage.embeddings.service import EmbeddingService
            import numpy as np

            svc = EmbeddingService.get_instance()
            query_emb = svc.embed_text_sync(query)
            corpus_embs = np.array([e for _, e in embedded])
            similarities = EmbeddingService.cosine_similarity_matrix(
                np.array(query_emb), corpus_embs
            )
            top_indices = np.argsort(similarities)[::-1][:top_k]
            return [embedded[i][0] for i in top_indices]
        except Exception:
            logger.debug("KNNFewShot: embedding similarity failed, falling back to top-k")
            return traces[:top_k]

    def _embed_traces(self, traces: List[Any]) -> List[Tuple[Any, Any]]:
        """Embed trace descriptions for KNN search.

        Builds a text description for each trace from its task type,
        tool names, and failure keys, then batch-embeds them.

        Args:
            traces: List of trace objects.

        Returns:
            List of (trace, embedding) tuples, or empty list on failure.
        """
        try:
            from victor.storage.embeddings.service import EmbeddingService

            svc = EmbeddingService.get_instance()
        except Exception:
            return []

        descriptions = []
        for trace in traces:
            task = getattr(trace, "task_type", "default")
            tools = _normalize_few_shot_tools(
                [getattr(d, "tool_name", "") for d in getattr(trace, "tool_call_details", [])[:5]]
            )
            failures = list(getattr(trace, "tool_failures", {}).keys())
            desc = f"{task}: {' '.join(tools)} failures={','.join(failures)}"
            descriptions.append(desc)

        try:
            embeddings = svc.embed_batch_sync(descriptions)
            return list(zip(traces, embeddings))
        except Exception:
            return []

    def reflect(
        self,
        traces: List[Any],
        section_name: str,
        current_text: str,
        **kwargs: Any,
    ) -> str:
        """Reflect on traces to produce updated prompt section text.

        Filters to successful traces, optionally narrows them using
        KNN selection when a query is provided, and formats them
        into few-shot example text.

        Args:
            traces: List of trace objects with success and completion_score.
            section_name: Name of the prompt section being optimized.
            current_text: Current text of the prompt section.
            **kwargs: Optional keyword arguments. Supports:
                - query (str): Current user query for KNN selection.

        Returns:
            Updated prompt section text with few-shot examples.
        """
        query = kwargs.get("query")

        # Filter to successful traces
        successful = [
            t
            for t in traces
            if getattr(t, "success", False)
            and getattr(t, "completion_score", 0) >= self._min_completion_score
        ]

        if not successful:
            return current_text

        successful.sort(key=lambda trace: getattr(trace, "completion_score", 0.0), reverse=True)

        # Use KNN selection when query is available and we have more
        # traces than needed
        if query and len(successful) > self._max_examples:
            successful = self.select_similar_traces(successful, query, self._max_examples)

        if self._example_diversity:
            successful = self._dedupe_traces(successful)

        # Limit to max_examples
        selected = successful[: self._max_examples]

        # Format few-shot examples from selected traces
        examples = []
        sparse_examples = 0
        for i, trace in enumerate(selected, 1):
            task = getattr(trace, "task_type", "unknown")
            tools = [
                getattr(d, "tool_name", "unknown")
                for d in getattr(trace, "tool_call_details", [])[:5]
            ]
            tools = _normalize_few_shot_tools(tools)
            score = getattr(trace, "completion_score", 0)
            failures = list(getattr(trace, "tool_failures", {}).keys())[:3]
            if tools:
                descriptor = f"tools=[{', '.join(tools)}]"
            else:
                sparse_examples += 1
                provider = getattr(trace, "provider", "unknown")
                model = getattr(trace, "model", "unknown")
                tool_calls = getattr(trace, "tool_calls", 0)
                descriptor = f"provider={provider}, model={model}, tool_calls={tool_calls}"
            if failures:
                descriptor = f"{descriptor}, failures=[{', '.join(failures)}]"
            examples.append(f"Example {i} ({task}, score={score:.1f}): {descriptor}")

        if not examples:
            return current_text

        logger.info(
            "MIPROv2 few-shot mining: successful=%d selected=%d sparse=%d query_aware=%s",
            len(successful),
            len(selected),
            sparse_examples,
            bool(query),
        )

        example_text = "\n".join(examples)
        if len(example_text) > self._max_example_chars:
            max_len = max(self._max_example_chars - 3, 0)
            example_text = f"{example_text[:max_len].rstrip()}..."
        if current_text.strip():
            return f"{current_text}\n\n--- Few-shot demonstrations ---\n{example_text}"
        return f"--- Few-shot demonstrations ---\n{example_text}"

    def mutate(self, current_text: str, reflection: str, section_name: str) -> str:
        """Return the mined few-shot block as the updated section text."""
        del section_name
        return reflection or current_text

    def _dedupe_traces(self, traces: List[Any]) -> List[Any]:
        """Drop near-identical examples so few-shot blocks stay varied."""
        unique: List[Any] = []
        seen = set()
        for trace in traces:
            signature = self._trace_signature(trace)
            if signature in seen:
                continue
            seen.add(signature)
            unique.append(trace)
        return unique

    @staticmethod
    def _trace_signature(trace: Any) -> str:
        """Build a lightweight trace signature for diversity filtering."""
        task = getattr(trace, "task_type", "unknown")
        tools = [
            getattr(detail, "tool_name", "unknown")
            for detail in getattr(trace, "tool_call_details", [])[:5]
        ]
        tools = _normalize_few_shot_tools(tools)
        return f"{task}:{','.join(tools)}"

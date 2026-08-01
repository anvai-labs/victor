# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Readable and token-efficient JSON schema for LLM task planning.

This module provides a Pydantic schema for LLM-generated task plans that balances:
- **Readability**: Clear, self-documenting field names for LLM reliability
- **Token Efficiency**: Still ~40% savings vs verbose JSON through list format
- **Type Safety**: Full Pydantic validation
- **LLM-Friendly**: Structure optimized for LLM generation
- **Context-Aware Tool Selection**: Step-based tool filtering for 50-80% token savings

Key Design Decision: Readable keywords over abbreviations
- Better for LLM reliability (clear structure)
- Easier to debug and maintain
- Still achieves token savings through list format
- Self-documenting for developers

Example Readable JSON (120 tokens vs 180 for verbose JSON):
    {"name":"Add auth","complexity":"moderate","desc":"OAuth2 login",
     "steps":[[1,"research","Analyze patterns","overview"],
              [2,"feature","Create module","write,test"]]}

Usage:
    from victor.agent.planning import TaskPlan, generate_compact_plan

    # LLM generates readable compact JSON
    json_str = '{"name":"auth","complexity":"simple","desc":"Fix bug","steps":[[1,"analyze","Find bug","grep"]]}'

    # Validate and expand
    task_plan = TaskPlan.model_validate_json(json_str)
    execution_plan = task_plan.to_execution_plan()

    # Get tools for a specific step
    tools = task_plan.get_contextual_tools(tool_selector, step_index=0)
"""

from __future__ import annotations

import json
import logging
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, ValidationError

from victor.agent.recovery.empty_response import (
    diagnose_empty_response,
    next_retry_parameters,
)
from victor.agent.planning.base import (
    ExecutionPlan,
    PlanStep,
    StepStatus,
    StepType,
)

logger = logging.getLogger(__name__)

# Import unified response parser from framework
try:
    from victor.processing.response_parser import (
        extract_content_from_response,
        extract_json_from_response,
    )

    # Create aliases for backward compatibility
    extract_llm_response_content = extract_content_from_response
    extract_json_from_llm_response = extract_json_from_response
    _UNIFIED_PARSER_AVAILABLE = True
except ImportError:
    logger.warning("Unified response parser not available, using local fallback")
    _UNIFIED_PARSER_AVAILABLE = False

    # Fallback implementations if unified parser not available
    def extract_llm_response_content(
        response: Union[str, Dict[str, Any], Any],
    ) -> Optional[str]:
        """Fallback content extraction."""
        if response is None:
            return None
        if isinstance(response, str):
            return response
        if isinstance(response, dict) and "content" in response:
            content = response["content"]
            return content if isinstance(content, str) else None
        if hasattr(response, "content"):
            content = response.content
            return content if isinstance(content, str) else None
        return None

    def extract_json_from_llm_response(
        response: Union[str, Dict[str, Any], Any],
    ) -> Optional[str]:
        """Fallback JSON extraction."""
        content = extract_llm_response_content(response)
        if not content:
            return None
        try:
            json.loads(content.strip())
            return content.strip()
        except json.JSONDecodeError:
            return None


class TaskComplexity(str, Enum):
    """Task complexity levels for planning."""

    SIMPLE = "simple"  # Auto mode, 2-3 steps, <30 min
    MODERATE = "moderate"  # Plan-mode, 3-5 steps, 30min-2hr
    COMPLEX = "complex"  # Plan-mode, 5-8 steps, >2hr


# ---------------------------------------------------------------------------
# Exec-type inference helpers (module-level, domain-agnostic)
# ---------------------------------------------------------------------------
# Scoring strategy: weighted combination of regex (primary) + embedding similarity
# (secondary).  Regex weight=0.7, embedding weight=0.3.  Archetype embeddings are
# pre-computed once at session start via precompute_plan_inference_embeddings().
# In environments where embeddings are not available the regex path runs alone.

_COND_PATTERNS: List[re.Pattern] = [
    re.compile(r"^\s*route\s*:", re.I),
    re.compile(r"\bdetermine\s+if\b", re.I),
    re.compile(r"\bif\s+this\s+is\b.{0,60}\bor\b", re.I),
    re.compile(r"\bbranch\b.{0,30}\bon\b", re.I),
    re.compile(r"(?:multi|multiple).{0,30}(?:vs|versus|or).{0,30}single", re.I),
    re.compile(r"single.{0,30}(?:vs|versus|or).{0,30}(?:multi|multiple)", re.I),
    re.compile(r"\bchoose\s+between\b", re.I),
    re.compile(r"\bselect\b.{0,30}\bstrategy\b", re.I),
]

_LOOP_PATTERNS: List[re.Pattern] = [
    re.compile(r"\bloop\s+over\b", re.I),
    re.compile(r"\bfor\s+each\b", re.I),
    re.compile(r"\biterate\s+over\b", re.I),
    re.compile(r"\breview\s+each\b", re.I),
    re.compile(r"\bper\s+\w+\s+(?:do|perform|review|analyze)\b", re.I),
]

_APPROVAL_PATTERNS: List[re.Pattern] = [
    re.compile(r"\bpresent\b.{0,50}\bto\s+(?:the\s+)?user\b", re.I),
    re.compile(r"\bshow\b.{0,50}\bfor\s+(?:user\s+)?(?:review|approval)\b", re.I),
    re.compile(
        r"\b(?:review|approve|confirm)\b.{0,50}\bbefore\s+(?:begin|continu|proceed)",
        re.I,
    ),
    re.compile(r"\bfor\s+user\s+(?:review|approval|confirmation)\b", re.I),
    re.compile(r"\buser\s+(?:review|approval)\s+before\b", re.I),
]

# Verbs that indicate a step is "producing" a collection for later steps.
# Substring-matched (case-insensitive), so prefix variants also match.
_PRODUCER_VERBS = frozenset(
    [
        "inventory",
        "inventori",
        "list",
        "enumerate",
        "discover",
        "identify",
        "collect",
        "gather",
        "catalog",
        "find",
        "scan",
        "map",
        "read all",
        "parse all",
        "extract all",
        "detect all",
        "locate all",
        "audit",
        "retrieve",
        "fetch",
        "query",
        "examine",
        "inspect",
        "survey",
    ]
)

# Archetype phrases for semantic similarity scoring.
# These are static strings — their embeddings can be pre-computed once.
_EXEC_ARCHETYPES: Dict[str, List[str]] = {
    "conditional": [
        "determine if this is option A or option B",
        "route to the correct path based on a condition",
        "branch to different steps based on what was discovered",
        "choose between two strategies based on context",
    ],
    "loop": [
        "for each item in the collection do the work",
        "iterate over every element and process it",
        "loop over each member and perform analysis",
        "review each item in the list one by one",
    ],
    "approval": [
        "present results to the user for review",
        "show findings and ask for user approval",
        "pause execution and wait for user confirmation",
        "user checkpoint before continuing",
    ],
    "producer": [
        "list all items in a collection",
        "discover all elements in the system",
        "find and enumerate all entries",
        "collect all members from source",
        "retrieve all records from storage",
    ],
}

# Pre-computed archetype embeddings: category -> (n_archetypes, dim) numpy array.
# Populated by precompute_plan_inference_embeddings().
_ARCHETYPE_VECS: Dict[str, Any] = {}

# Scoring weights: regex is the primary signal; embedding is a secondary boost.
_REGEX_WEIGHT: float = 0.7
_EMBED_WEIGHT: float = 0.3


def precompute_plan_inference_embeddings() -> bool:
    """Pre-compute archetype embeddings for plan exec-type and producer inference.

    Call this at session start (e.g. inside PlanningRuntimeService.__init__ or
    after EmbeddingService is initialized) so that the inference hot path never
    needs to compute embeddings from scratch.

    Returns True when pre-computation succeeded; False when the service is
    unavailable or not yet initialized.
    """
    global _ARCHETYPE_VECS
    try:
        from victor.storage.embeddings.service import EmbeddingService

        instance = EmbeddingService._instance
        if instance is None or not getattr(instance, "_initialized", False):
            return False

        for category, phrases in _EXEC_ARCHETYPES.items():
            _ARCHETYPE_VECS[category] = instance.embed_batch_sync(phrases)

        logger.debug(
            "Plan inference: pre-computed %d archetype categories (%d phrases)",
            len(_ARCHETYPE_VECS),
            sum(len(p) for p in _EXEC_ARCHETYPES.values()),
        )
        return True
    except Exception:
        return False


def _cosine_max(desc_vec: Any, arch_vecs: Any) -> float:
    """Return the max cosine similarity between desc_vec and each row of arch_vecs."""
    try:
        import numpy as np

        d = desc_vec / (np.linalg.norm(desc_vec) + 1e-8)
        best = 0.0
        for av in arch_vecs:
            av_norm = av / (np.linalg.norm(av) + 1e-8)
            best = max(best, float(np.dot(d, av_norm)))
        return best
    except Exception:
        return 0.0


def _embed_desc(desc: str) -> Any:
    """Embed desc using EmbeddingService (sync).  Returns None if unavailable."""
    try:
        from victor.storage.embeddings.service import EmbeddingService

        instance = EmbeddingService._instance
        if instance is None or not getattr(instance, "_initialized", False):
            return None
        return instance.embed_text_sync(desc, use_cache=True)
    except Exception:
        return None


def _infer_exec_type(desc: str) -> Optional[str]:
    """Return inferred exec type using weighted regex + embedding scoring.

    Scores each candidate type:
      score(type) = regex_match * 0.7 + max_cosine_similarity * 0.3

    The highest-scoring type is returned when score >= 0.5.
    Falls back to pure regex when embeddings are unavailable.
    """
    desc_lower = desc.lower()
    if re.search(
        r"\b(synthesize|summarize|compile|write)\b.{0,80}\b(report|findings)\b",
        desc_lower,
    ):
        return None

    pattern_map = {
        "conditional": _COND_PATTERNS,
        "loop": _LOOP_PATTERNS,
        "approval": _APPROVAL_PATTERNS,
    }
    desc_vec = _embed_desc(desc) if _ARCHETYPE_VECS else None

    best_type: Optional[str] = None
    best_score = 0.0

    for exec_type, patterns in pattern_map.items():
        regex_hit = any(p.search(desc) for p in patterns)
        regex_score = _REGEX_WEIGHT if regex_hit else 0.0

        embed_score = 0.0
        if desc_vec is not None and exec_type in _ARCHETYPE_VECS:
            embed_score = _EMBED_WEIGHT * _cosine_max(desc_vec, _ARCHETYPE_VECS[exec_type])

        score = regex_score + embed_score
        if score > best_score:
            best_score = score
            best_type = exec_type

    # Require ≥ 0.5 to avoid false positives (0.5 means regex hit alone is enough).
    return best_type if best_score >= 0.5 else None


def _infer_loop_over_key(desc: str) -> Optional[str]:
    """Extract a snake_case plural key from a loop step's description.

    Generic across ecosystems — recognises "workspace member", "package", "module",
    "component", "target", or "crate" wording and maps them to the canonical
    review_targets key.  Verticals can register additional vocabulary if needed.
    """
    desc_lower = desc.lower()
    if re.search(
        r"\beach\s+(?:review\s+)?target\b|\bper-target\b|\btarget-by-target\b",
        desc_lower,
    ):
        return "review_targets"
    if re.search(
        r"\beach\s+workspace\s+(?:member|crate|target|package|module)\b",
        desc_lower,
    ):
        return "review_targets"
    if re.search(
        r"\beach\s+(?:crate|module|package|component)\b|"
        r"\bper-(?:crate|module|package|component)\b|"
        r"\b(?:crate|module|package|component)-by-(?:crate|module|package|component)\b",
        desc_lower,
    ):
        return "review_targets"

    # Capture the noun phrase after loop/iterate/review each.
    # Terminator: whitespace+verb, comma, period, colon, or end-of-string.
    _TERM = r"(?:\s+(?:performing|do\b|perform|review|analyze|check|scan)|[,.:;]|$)"
    candidates = [
        re.compile(rf"loop\s+over\s+(?:each\s+)?([\w][\w\s]{{1,30}}?){_TERM}", re.I),
        re.compile(rf"for\s+each\s+([\w][\w\s]{{1,30}}?){_TERM}", re.I),
        re.compile(
            r"iterate\s+over\s+(?:each\s+)?([\w][\w\s]{1,20}?)(?:\s+and|\s*,|\s*$)",
            re.I,
        ),
        re.compile(r"review\s+each\s+([\w][\w\s]{1,20}?)(?:\s+and|\s*,|\s*$)", re.I),
    ]
    for pat in candidates:
        m = pat.search(desc)
        if m:
            noun = m.group(1).strip().rstrip(".,;:")
            words = noun.lower().split()[:3]  # at most 3 words
            if not words:
                continue
            key = "_".join(words)
            if not key.endswith("s"):
                key += "s"
            return key
    return None


def _best_matching_key(candidate: str, known_keys: List[str]) -> Optional[str]:
    """Return the known key whose words overlap most with *candidate*."""
    cand_words = {w.rstrip("s") for w in candidate.replace("_", " ").lower().split()}
    best: Optional[str] = None
    best_score = 0
    for key in known_keys:
        key_words = {w.rstrip("s") for w in key.replace("_", " ").lower().split()}
        score = len(cand_words & key_words)
        if score > best_score:
            best_score = score
            best = key
    return best if best_score > 0 else None


def _infer_condition_key(desc: str, known_keys: List[str]) -> Optional[str]:
    """Return a plan_state key that the condition should test, or None."""
    desc_lower = desc.lower()
    for key in known_keys:
        key_readable = key.replace("_", " ").lower()
        if key_readable in desc_lower or key.lower() in desc_lower:
            return key
    # Looser: any significant word from a known key appears in description
    for key in known_keys:
        parts = [p for p in key.replace("_", " ").split() if len(p) > 4]
        if any(p in desc_lower for p in parts):
            return key
    return None


def _infer_condition_subject_key(desc: str) -> Optional[str]:
    """Infer what collection a conditional routing step is routing on.

    Examples:
        "Route: determine if rust/ is a multi-crate workspace or single crate"
        → "workspaces"  (then matched against upstream step that inventories crates)

        "Route: if this is multi-module or single module"
        → "modules"

        "Determine if multiple services or single service"
        → "services"
    """
    candidates = [
        # "multi-X workspace or single crate" → group(1) = "workspace"
        re.compile(
            r"(?:multi[- ]?\w+\s+)([\w]+(?:\s+\w+)?)\s+or\s+single",
            re.I,
        ),
        # "determine if ... is a multi-crate/multi-module X" → group(1) = noun after multi-*
        re.compile(
            r"determine\s+if\b.{0,60}?multi[- ]?\w+\s+([\w]+)\b",
            re.I,
        ),
        # "if this is X or Y" (where X contains a noun) → the Y (binary choice subject)
        re.compile(
            r"if\s+(?:this|it|the)\s+is\s+(?:a\s+)?(?:multi[- ]?\w+\s+)?([\w]+(?:\s+\w+)?)\s+or\b",
            re.I,
        ),
        # "route ... workspace/module/service/package" → group(1) = noun
        re.compile(
            r"\broute\b.{0,60}?\b(workspace|crate|module|service|package|component|repository|repo)s?\b",
            re.I,
        ),
        # "multi-crate" or "multi-module" prefix directly
        re.compile(
            r"\bmulti[- ]?(crate|module|workspace|package|service|component)s?\b",
            re.I,
        ),
    ]
    for pat in candidates:
        m = pat.search(desc)
        if m:
            noun = m.group(m.lastindex).strip().lower() if m.lastindex else ""
            if not noun or len(noun) < 3:
                continue
            words = [w for w in noun.split() if len(w) > 2][:2]
            if not words:
                continue
            key = "_".join(words)
            if not key.endswith("s"):
                key += "s"
            return key
    return None


def _infer_branches(step_id: str, all_ids: List[str]) -> Optional[Dict[str, List[str]]]:
    """Derive true/false branch IDs from sibling IDs with 'a'/'b' suffixes."""
    num_match = re.match(r"^(\d+)$", str(step_id))
    if not num_match:
        return None
    base = num_match.group(1)
    for try_base in (base, str(int(base) + 1)):
        sa = next(
            (sid for sid in all_ids if re.fullmatch(rf"{re.escape(try_base)}a", sid, re.I)),
            None,
        )
        sb = next(
            (sid for sid in all_ids if re.fullmatch(rf"{re.escape(try_base)}b", sid, re.I)),
            None,
        )
        if sa and sb:
            return {"true": [sa], "false": [sb]}
    return None


def _step_likely_produces(desc: str, key: str) -> bool:
    """Heuristic: does this step's description suggest it produces *key*?

    Scoring: regex_weight * verb_hit + embed_weight * cosine_similarity >= 0.5
    The noun check is a required pre-filter (no noun match → False immediately).
    """
    desc_lower = desc.lower()
    key_parts = [p for p in key.replace("_", " ").split() if len(p) > 3]
    has_noun = any(p.rstrip("s") in desc_lower for p in key_parts)
    if not has_noun:
        return False

    regex_score = _REGEX_WEIGHT if any(v in desc_lower for v in _PRODUCER_VERBS) else 0.0

    embed_score = 0.0
    if _ARCHETYPE_VECS and "producer" in _ARCHETYPE_VECS:
        desc_vec = _embed_desc(desc)
        if desc_vec is not None:
            embed_score = _EMBED_WEIGHT * _cosine_max(desc_vec, _ARCHETYPE_VECS["producer"])

    return (regex_score + embed_score) >= 0.5


def _infer_produces_key_from_desc(step: Dict[str, Any]) -> Optional[str]:
    """Infer common plan_state producer keys from a step description.

    This is intentionally limited to structural planning artifacts that the
    planner itself depends on.  Without these keys, later loop/synthesis steps
    can run too early or fail to use deterministic inventory helpers.
    """
    desc = str(step.get("desc", step.get("description", "")) or "").lower()
    step_type = str(step.get("type", "") or "").lower()
    tools_raw = step.get("tools", [])
    tools = (
        {str(t).strip().lower() for t in tools_raw if str(t).strip()}
        if isinstance(tools_raw, list)
        else {t.strip().lower() for t in str(tools_raw).split(",") if t.strip()}
    )

    if re.search(r"^\s*route\b|\bpresent\b.{0,80}\bto\s+(?:the\s+)?user\b", desc):
        return None

    if re.search(r"\b(file\s+inventory|module\s+tree|src/|tests/|benches/|examples/)\b", desc):
        return "target_file_inventory"
    if re.search(
        r"\b(review\s+targets?|workspace\s+members?|target\s+director|package\s+inventory|"
        r"module\s+inventory|component\s+inventory)\b",
        desc,
    ) and re.search(
        r"\b(extract|parse|inventory|enumerate|list|discover|map)\b",
        desc,
    ):
        return "review_targets"
    if "checklist" in desc and re.search(r"\b(create|build|generate|write|draft)\b", desc):
        return "review_checklist"
    if (
        step_type in {"doc", "documentation"}
        or "write" in tools
        or re.search(r"\b(synthesize|summarize|compile)\b", desc)
    ) and re.search(r"\b(final\s+report|prioritized\s+report|consolidated\s+report)\b", desc):
        return "final_report"
    if re.search(r"\bcross[- ](?:crate|target|module|package|component)\b", desc) and re.search(
        r"\b(analysis|analyze|review|findings|issues|patterns)\b",
        desc,
    ):
        return "cross_target_findings"
    if re.search(r"\bdependenc(?:y|ies)\b", desc) and re.search(
        r"\b(analysis|analyze|review|audit|identify|map|extract|findings)\b",
        desc,
    ):
        return "dependency_findings"
    if re.search(r"\bsingle[- ](?:crate|target|module|package|component)\b", desc) and re.search(
        r"\b(analysis|analyze|review|audit|findings)\b",
        desc,
    ):
        return "findings_single_target"
    if re.search(r"\b(performance|hotspot|allocation-heavy|high-frequency|blocking)\b", desc):
        if re.search(
            r"\b(analysis|analyze|identify|find|detect|check|rankings?|findings?)\b",
            desc,
        ):
            return "performance_hotspot_findings"
    if re.search(
        r"\b(per[- ](?:crate|target|module|package|component)|"
        r"each\s+(?:workspace\s+(?:member|crate|target)|crate|target|module|package|component)|"
        r"(?:crate|target|module|package|component)-by-(?:crate|target|module|package|component))\b",
        desc,
    ):
        if re.search(r"\b(review|analysis|analyze|audit|findings|record)\b", desc):
            return "per_target_findings"
    return None


def _iter_json_candidates(
    primary_json: Optional[str], response_content: Optional[str]
) -> List[str]:
    """Return JSON candidates ordered from most likely to least likely plan object.

    The generic JSON extractor is intentionally broad and may return a nested step array
    before the surrounding plan object. Plan generation needs schema-aware selection, so
    include the extracted candidate, the full response when valid JSON, and every balanced
    JSON object/array found in the response text.
    """
    candidates: List[str] = []
    seen: set[str] = set()

    def add(candidate: Optional[str]) -> None:
        if not candidate:
            return
        text = candidate.strip()
        if text and text not in seen:
            seen.add(text)
            candidates.append(text)

    add(primary_json)

    content = (response_content or "").strip()
    if not content:
        return candidates
    add(content)

    decoder = json.JSONDecoder()
    for index, char in enumerate(content):
        if char not in "{[":
            continue
        try:
            _, end = decoder.raw_decode(content[index:])
        except json.JSONDecodeError:
            continue
        add(content[index : index + end])

    # Prefer full plan-shaped objects over nested arrays or unrelated JSON snippets.
    def score(candidate: str) -> tuple[int, int]:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            return (0, len(candidate))
        if isinstance(data, dict):
            required = {"name", "complexity", "desc", "steps"}
            if required.issubset(data):
                return (3, len(candidate))
            if "steps" in data:
                return (2, len(candidate))
            return (1, len(candidate))
        return (0, len(candidate))

    candidates.sort(key=score, reverse=True)
    return candidates


def _validate_readable_plan_from_candidates(
    primary_json: Optional[str],
    response_content: Optional[str],
) -> ReadableTaskPlan:
    """Validate the first JSON candidate that satisfies ReadableTaskPlan."""
    last_error: Optional[Exception] = None
    for candidate in _iter_json_candidates(primary_json, response_content):
        try:
            return ReadableTaskPlan.model_validate_json(candidate)
        except ValidationError as exc:
            last_error = exc
            logger.debug("Plan JSON candidate rejected: %s; candidate=%.120s", exc, candidate)
    if last_error is not None:
        raise last_error
    raise ValueError(
        "No valid JSON found in plan response. "
        f"Response: {response_content[:500] if response_content else 'empty'}"
    )


class ReadableTaskPlan(BaseModel):
    """Readable and token-efficient task plan schema for LLM generation.

    This schema uses READABLE field names (not cryptic abbreviations) while
    maintaining token efficiency through list-based step representation.

    Token Efficiency Strategy:
    - Readable field names: name, desc, steps (not n, d, s)
    - List format for steps: [id, type, desc, tools, deps]
    - Achieves ~40% token savings while remaining fully readable

    Field Name Mapping:
    - name: task name (short, clear)
    - complexity: simple|moderate|complex (not c)
    - desc: description (not d)
    - steps: list of step data (not s)
    - duration: estimated duration (not e, optional)
    - approval: requires approval (not a, optional)

    Step Data Format:
    - List: [id, type, description, tools, dependencies]
    - Types: research, feature, bugfix, refactor, test, review, deploy, analyze, doc

    Example:
        {
          "name": "Add authentication",
          "complexity": "moderate",
          "desc": "Implement OAuth2 login",
          "steps": [
            [1, "research", "Analyze patterns", "overview"],
            [2, "feature", "Create module", "write,test"]
          ],
          "duration": "30min"
        }
    """

    # Readable field names (not cryptic abbreviations)
    name: str = Field(..., description="Task name (short, clear)")
    complexity: TaskComplexity = Field(..., description="Complexity level")
    desc: str = Field(..., description="Task description")
    steps: List[Union[List, Dict[str, Any]]] = Field(
        ...,
        description=(
            "Steps: [[id, type, desc, tools, deps, exec], ...] or "
            "[{id, type, desc, tools, deps, exec, node, exit}, ...]"
        ),
    )
    duration: Optional[str] = Field(None, description="Estimated duration (e.g., '30min', '2hr')")
    approval: bool = Field(False, description="Requires user approval")

    @field_validator("steps")
    @classmethod
    def validate_steps(
        cls, v: List[Union[List, Dict[str, Any]]]
    ) -> List[Union[List, Dict[str, Any]]]:
        """Validate step data format — accepts both list tuples and rich dicts."""
        for i, step_data in enumerate(v, 1):
            if isinstance(step_data, dict):
                has_desc = "desc" in step_data or "description" in step_data
                if "id" not in step_data or "type" not in step_data or not has_desc:
                    raise ValueError(
                        f"Step {i}: dict must have 'id', 'type', and 'desc'/'description' keys, "
                        f"got {list(step_data.keys())}"
                    )
            elif isinstance(step_data, list):
                if len(step_data) < 3:
                    raise ValueError(
                        f"Step {i}: must be list with at least [id, type, desc], got {step_data}"
                    )
                if not isinstance(step_data[0], (int, str)):
                    raise ValueError(f"Step {i}: id must be int or str, got {type(step_data[0])}")
            else:
                raise ValueError(f"Step {i}: must be list or dict, got {type(step_data)}")
        return v

    @classmethod
    def _enrich_step_dicts(
        cls, steps: List[Union[List, Dict[str, Any]]]
    ) -> List[Union[List, Dict[str, Any]]]:
        """Infer exec types and data-flow keys when the LLM omits them.

        This is a best-effort inference pass on the raw step list.  Explicitly
        set fields are **never** overwritten.  The inference is generic — no
        domain or language keywords are assumed.

        Pass 1 — exec type: pattern-match description to conditional/loop/approval.
        Pass 2 — produces:  find upstream steps that "produce" a collection that
                            loop/conditional nodes will consume.
        Pass 3 — loop_over / condition_on: resolve to the best matching produces key.
        Pass 4 — branches:  derive true/false step IDs from numeric+alpha sibling IDs.
        """
        result: List[Union[List, Dict[str, Any]]] = [
            dict(s) if isinstance(s, dict) else list(s) for s in steps
        ]
        all_ids = [
            str(s.get("id", "") if isinstance(s, dict) else (s[0] if s else "")) for s in result
        ]

        # --- Pass 1: infer exec types ---
        for step in result:
            if not isinstance(step, dict):
                continue
            if step.get("exec") or step.get("execution"):
                continue
            desc = str(step.get("desc", step.get("description", "")))
            inferred = _infer_exec_type(desc)
            if inferred:
                step["exec"] = inferred
                logger.info(
                    "Step %s: inferred exec=%s from description: %.80r",
                    step.get("id"),
                    inferred,
                    desc,
                )

        # --- Pass 1.5: infer common producer keys ---
        for step in result:
            if not isinstance(step, dict):
                continue
            if step.get("produces"):
                continue
            inferred_produces = _infer_produces_key_from_desc(step)
            if inferred_produces:
                step["produces"] = inferred_produces
                logger.info(
                    "Step %s: inferred produces=%s from description",
                    step.get("id"),
                    inferred_produces,
                )

        # --- Pass 2: infer produces on steps that feed downstream consumers ---
        # First collect keys that consumer steps need (loop_over / condition_on).
        needed_keys: List[str] = []
        for step in result:
            if not isinstance(step, dict):
                continue
            exec_type = str(step.get("exec", "")).lower()
            if exec_type not in ("loop", "conditional"):
                continue
            desc = str(step.get("desc", step.get("description", "")))
            if exec_type == "loop" and not step.get("loop_over"):
                key = _infer_loop_over_key(desc)
                if key and key not in needed_keys:
                    needed_keys.append(key)
            elif exec_type == "conditional" and not step.get("condition_on"):
                # Infer what collection the conditional is routing on, then
                # back-populate 'produces' on an upstream inventory step so the
                # collection is available in plan_state when the conditional fires.
                key = _infer_condition_subject_key(desc)
                if key and key not in needed_keys:
                    needed_keys.append(key)
                    logger.debug(
                        "Step %s: conditional needs collection key '%s'",
                        step.get("id"),
                        key,
                    )

        # Back-populate 'produces' on the first upstream step whose description
        # suggests it inventories/lists the needed collection.
        current_produces: List[str] = [
            str(s.get("produces", "")) for s in result if isinstance(s, dict) and s.get("produces")
        ]
        for key in needed_keys:
            if key in current_produces:
                continue
            for step in result:
                if not isinstance(step, dict) or step.get("produces"):
                    continue
                # Loop steps are consumers, not producers — skip them in back-populate.
                if str(step.get("exec", "")).lower() == "loop":
                    continue
                desc = str(step.get("desc", step.get("description", "")))
                if _step_likely_produces(desc, key):
                    step["produces"] = key
                    current_produces.append(key)
                    logger.info(
                        "Step %s: inferred produces='%s' (feeds downstream consumer)",
                        step.get("id"),
                        key,
                    )
                    break

        # --- Pass 3: resolve loop_over and condition_on ---
        for step in result:
            if not isinstance(step, dict):
                continue
            exec_type = str(step.get("exec", "")).lower()
            desc = str(step.get("desc", step.get("description", "")))

            if exec_type == "loop" and not step.get("loop_over"):
                raw_key = _infer_loop_over_key(desc)
                if raw_key:
                    # Prefer an exact match in known produces, fall back to raw
                    aligned = _best_matching_key(raw_key, current_produces) or raw_key
                    step["loop_over"] = aligned
                    logger.debug("Step %s: inferred loop_over=%s", step.get("id"), aligned)

            elif exec_type == "conditional":
                if not step.get("condition_on"):
                    key = _infer_condition_key(desc, current_produces)
                    if key:
                        step["condition_on"] = key
                        logger.debug("Step %s: inferred condition_on=%s", step.get("id"), key)
                if not step.get("condition"):
                    # Only default to "multiple" when the description implies a
                    # quantity comparison (multi vs single, more than one, etc.).
                    # Otherwise leave unset so _parse_step_dict uses "non_empty".
                    desc_lower = desc.lower()
                    if re.search(
                        r"\bmulti(?:ple)?\b|\bmore\s+than\s+one\b|\bseveral\b",
                        desc_lower,
                    ):
                        step["condition"] = "multiple"
                if not step.get("produces"):
                    # Store the boolean result so later steps can read it
                    cond_key = str(step.get("condition_on", ""))
                    if cond_key:
                        step["produces"] = f"is_{cond_key}_multiple"

        # --- Pass 4: infer branches for conditional steps ---
        for step in result:
            if not isinstance(step, dict):
                continue
            if str(step.get("exec", "")).lower() != "conditional":
                continue
            if step.get("branches"):
                continue
            step_id = str(step.get("id", ""))
            branches = _infer_branches(step_id, all_ids)
            if branches:
                step["branches"] = branches
                logger.info(
                    "Step %s: inferred branches=%s from sibling step IDs",
                    step_id,
                    branches,
                )

        # --- Pass 5: infer data-flow inputs (which plan_state keys each step consumes) ---
        # Build a positional index so we can limit inference to UPSTREAM keys only.
        step_positions: Dict[str, int] = {
            str(s.get("id", "")): idx for idx, s in enumerate(result) if isinstance(s, dict)
        }
        # produces_key -> position of the step that produces it
        produces_position: Dict[str, int] = {}
        for s in result:
            if not isinstance(s, dict):
                continue
            key = str(s.get("produces", ""))
            sid = str(s.get("id", ""))
            if key and sid in step_positions:
                produces_position[key] = step_positions[sid]

        for idx, step in enumerate(result):
            if not isinstance(step, dict):
                continue
            # Only infer when the step doesn't already declare inputs.
            if step.get("inputs") or step.get("consumes"):
                continue
            own_produces = str(step.get("produces", ""))
            desc_lower = str(step.get("desc", step.get("description", ""))).lower()
            inferred_inputs: List[str] = []
            for key, prod_pos in produces_position.items():
                # Never infer own output, and never infer from a downstream step.
                if key == own_produces or prod_pos >= idx:
                    continue
                # Match: full key phrase (underscores → spaces) or any long word of the key.
                words = key.replace("_", " ")
                generic_parts = {
                    "finding",
                    "findings",
                    "result",
                    "results",
                    "review",
                    "audit",
                    "crate",
                    "crates",
                }
                parts = [
                    w for w in key.split("_") if len(w) >= 4 and w.rstrip("s") not in generic_parts
                ]
                if words in desc_lower or any(p in desc_lower for p in parts):
                    inferred_inputs.append(key)
            if inferred_inputs:
                step["inputs"] = inferred_inputs
                logger.debug(
                    "Step %s: inferred inputs=%s",
                    step.get("id"),
                    inferred_inputs,
                )

        # --- Pass 6: minimize depends_on using data-flow edges to enable parallelism ---
        # When a step declares inputs, its only *required* deps are the steps that produce
        # those inputs.  LLM plans often chain every step sequentially even when multiple
        # steps only need the same upstream output — those siblings can run in parallel if
        # their deps list is trimmed to the true data dependencies.
        #
        # Safety rule A: keep explicit deps that point at steps that produce NO named output
        # (approval/checkpoint/conditional gates).  Those gates are intentional control-flow
        # synchronisation points that must be respected regardless of data dependencies.
        #
        # Safety rule B (critical): ALWAYS keep deps on control-gate exec types even when
        # the gate happens to produce a named output (e.g. a conditional that sets
        # `is_multi_crate`).  Without this guard, Pass 6 would see the conditional as a
        # "data producer" for `is_multi_crate`, note that downstream branches declare no
        # input of `is_multi_crate`, and remove the conditional from their deps — causing
        # both branches to run in parallel BEFORE the conditional fires.
        _CONTROL_GATE_EXEC: frozenset = frozenset(
            {"conditional", "approval", "checkpoint", "loop", "compute"}
        )

        produces_map: Dict[str, str] = {}  # key → step_id
        for s in result:
            if not isinstance(s, dict):
                continue
            ctx = s.get("context") or {}
            # Accept produces at top level or nested under context (both forms occur).
            key = str(s.get("produces", "") or ctx.get("produces", "") or "")
            sid = str(s.get("id", ""))
            if key and sid:
                produces_map[key] = sid

        # Map step_id → exec type so Rule B can check gate types quickly.
        step_exec_type: Dict[str, str] = {
            str(s.get("id", "")): str(s.get("exec", s.get("execution", s.get("type", "")))).lower()
            for s in result
            if isinstance(s, dict)
        }

        # Set of step IDs that produce a named output (data producers).
        data_producing_step_ids: set = set(produces_map.values())

        # --- Pass 5.5: enforce declared data-flow dependencies ---
        # If a step declares inputs=["foo"], it cannot safely run until the step
        # that produces "foo" has completed.  Earlier passes infer inputs for
        # many valid plan shapes, but LLMs often omit matching deps.  Add those
        # deps generically for any upstream producer.
        deps_by_id: Dict[str, set[str]] = {
            str(s.get("id", "")): {str(d) for d in (s.get("deps") or s.get("depends_on") or [])}
            for s in result
            if isinstance(s, dict)
        }

        def _is_transitively_satisfied(producer_id: str, current_deps: set[str]) -> bool:
            stack = list(current_deps)
            seen: set[str] = set()
            while stack:
                dep = stack.pop()
                if dep == producer_id:
                    return True
                if dep in seen:
                    continue
                seen.add(dep)
                stack.extend(deps_by_id.get(dep, set()))
            return False

        for step in result:
            if not isinstance(step, dict):
                continue
            step_id = str(step.get("id", ""))
            current_pos = step_positions.get(step_id, 10_000)
            declared_inputs = [
                str(k).strip()
                for k in (step.get("inputs") or step.get("consumes") or [])
                if str(k).strip()
            ]
            if not declared_inputs:
                continue
            deps_raw = step.get("deps") or step.get("depends_on") or []
            current_set = {str(d) for d in deps_raw}
            data_deps = {
                produces_map[key]
                for key in declared_inputs
                if key in produces_map
                and produces_map[key] != step_id
                and step_positions.get(produces_map[key], 10_000) < current_pos
            }
            missing = sorted(
                dep
                for dep in data_deps
                if dep not in current_set and not _is_transitively_satisfied(dep, current_set)
            )
            if missing:
                new_deps = sorted(current_set | set(missing))
                deps_field = "deps" if "deps" in step else "depends_on"
                step[deps_field] = new_deps
                if "deps" in step and "depends_on" in step:
                    step["depends_on"] = new_deps
                deps_by_id[step_id] = set(new_deps)
                logger.info(
                    "Step %s: enforced data-flow deps %s for inputs=%s",
                    step_id,
                    missing,
                    declared_inputs,
                )

        for step in result:
            if not isinstance(step, dict):
                continue
            declared_inputs = [
                str(k).strip()
                for k in (step.get("inputs") or step.get("consumes") or [])
                if str(k).strip()
            ]
            if not declared_inputs:
                continue  # No data-flow info — leave deps unchanged.

            current_deps = list(step.get("deps") or step.get("depends_on") or [])
            if not current_deps:
                continue  # Already has no deps; nothing to minimise.

            current_dep_ids: set = {str(d) for d in current_deps}

            # Compute the minimal set of deps using ALL declared-input producers
            # (unrestricted — may include step IDs not in current_deps).
            # The length guard below ensures we never grow the dep set: if the
            # candidate minimal_deps is longer than current_deps, the step is left
            # unchanged.  This allows one-for-one replacement of redundant sequential
            # intermediates (e.g. deps=['2']→['1'] when step 1 directly produces the
            # needed input and step 2 was only an intermediate) while preventing dep
            # graph growth (e.g. deps=['4'] must not become ['1','2','4'] just because
            # steps 1 and 2 also produce some of the declared inputs).
            data_deps: set = {produces_map[key] for key in declared_inputs if key in produces_map}
            # Retain any dep that is either:
            #   (A) not a data producer at all — pure gate node
            #   (B) a control-gate exec type — even if it also produces a named output
            control_deps: set = {
                str(d)
                for d in current_deps
                if str(d) not in data_producing_step_ids
                or step_exec_type.get(str(d), "") in _CONTROL_GATE_EXEC
            }
            minimal_deps = sorted(data_deps | control_deps)

            # Guard: only trim when the result is no larger than the current dep set.
            # This prevents Pass 6 from adding producer step IDs that were never
            # explicit deps (which would widen the graph and defeat the purpose).
            if len(minimal_deps) <= len(current_deps) and set(minimal_deps) != current_dep_ids:
                removed = [d for d in current_deps if str(d) not in set(minimal_deps)]
                logger.info(
                    "Step %s: trimmed deps %s → %s (removed redundant sequential deps %s; "
                    "enables parallel dispatch with siblings)",
                    step.get("id"),
                    [str(d) for d in current_deps],
                    minimal_deps,
                    removed,
                )
                step["deps"] = minimal_deps
                if "depends_on" in step:
                    step["depends_on"] = minimal_deps

        # --- Pass 6.5: enforce data-flow deps for conditional key consumers ---
        # A conditional step that evaluates plan_state[condition_on] MUST NOT run
        # until the step that produces condition_on has completed.  Pass 6 is
        # trim-only and cannot add missing deps; this pass fills that gap.
        #
        # Race condition this prevents: if the LLM writes an empty deps list for
        # a conditional step, it can fire in the same parallel batch as the step
        # that produces condition_on, evaluating against an empty plan_state and
        # routing to the wrong branch.
        for step in result:
            if not isinstance(step, dict):
                continue
            if step_exec_type.get(str(step.get("id", "")), "") != "conditional":
                continue
            cond_key = str(step.get("condition_on", "") or "")
            if not cond_key or cond_key not in produces_map:
                continue
            producer_id = produces_map[cond_key]
            deps_raw = step.get("deps") or step.get("depends_on") or []
            current_set = {str(d) for d in deps_raw}
            if producer_id not in current_set:
                new_deps = sorted(current_set | {producer_id})
                deps_field = "deps" if "deps" in step else "depends_on"
                step[deps_field] = new_deps
                if "deps" in step and "depends_on" in step:
                    step["depends_on"] = new_deps
                logger.info(
                    "Step %s: enforced dep on '%s' (produces condition_on='%s')",
                    step.get("id"),
                    producer_id,
                    cond_key,
                )

        # --- Pass 6.6: anchor approval/review steps to their predecessor ---
        # An approval or review step with no deps fires in the very first parallel
        # batch, auto-approving before any work has been done.  The LLM often omits
        # the final-step dep (e.g. "step 10 review" with no dep on "step 9 report").
        # For any approval/review/checkpoint step that has no explicit deps, add a
        # dep on the step that immediately precedes it in the plan list.
        _APPROVAL_EXECS = frozenset({"approval", "checkpoint", "review"})
        dict_positions = [i for i, s in enumerate(result) if isinstance(s, dict)]
        for pos_idx, pos in enumerate(dict_positions):
            step = result[pos]
            exec_t = step_exec_type.get(str(step.get("id", "")), "")
            step_type_s = str(step.get("type", "")).lower()
            is_approval = exec_t in _APPROVAL_EXECS or step_type_s in ("review",)
            if not is_approval:
                continue
            deps_raw = step.get("deps") or step.get("depends_on") or []
            if deps_raw:
                continue  # Already has explicit deps — don't override
            # Find the nearest preceding dict step that is not itself an approval
            prev_id = None
            for prev_pos in reversed(dict_positions[:pos_idx]):
                prev = result[prev_pos]
                prev_exec = step_exec_type.get(str(prev.get("id", "")), "")
                prev_type = str(prev.get("type", "")).lower()
                if prev_exec not in _APPROVAL_EXECS and prev_type not in ("review",):
                    prev_id = str(prev.get("id", ""))
                    break
            if prev_id:
                deps_field = "deps" if "deps" in step else "depends_on"
                step[deps_field] = [prev_id]
                if "deps" in step and "depends_on" in step:
                    step["depends_on"] = [prev_id]
                logger.info(
                    "Step %s: anchored approval/review to preceding step '%s'",
                    step.get("id"),
                    prev_id,
                )

        # --- Pass 7: normalize phantom branch-base deps ---
        # The LLM sometimes writes deps=["7"] when the plan only has "7a"/"7b" as
        # the actual branch steps.  "7" never enters the satisfied-set so downstream
        # steps block forever.  Replace each such phantom dep with the existing branch
        # variants.  Only replace when branch variants exist; leave unrecognized deps
        # untouched (the is_ready() fallback in base.py handles any remaining phantoms
        # at runtime without permanent BLOCKED state).
        all_valid_ids: set = {str(s.get("id", "")) for s in result if isinstance(s, dict)}
        for step in result:
            if not isinstance(step, dict):
                continue
            deps_raw = step.get("deps") or step.get("depends_on") or []
            if not deps_raw:
                continue
            normalized: list = []
            changed = False
            for d in deps_raw:
                ds = str(d)
                if ds in all_valid_ids:
                    normalized.append(ds)
                    continue
                # Phantom dep: look for branch variants (e.g. "7" → ["7a","7b"])
                branch_variants = sorted(
                    sid
                    for sid in all_valid_ids
                    if re.fullmatch(rf"{re.escape(ds)}[a-z]", sid, re.I)
                )
                if branch_variants:
                    normalized.extend(branch_variants)
                    logger.info(
                        "Step %s: replaced phantom dep '%s' with branch variants %s",
                        step.get("id"),
                        ds,
                        branch_variants,
                    )
                    changed = True
                else:
                    # No branch variants — keep the dep as-is.
                    # is_ready() treats non-existent step IDs as satisfied at runtime.
                    normalized.append(ds)
            if changed:
                step["deps"] = normalized
                if "depends_on" in step:
                    step["depends_on"] = normalized

        # --- Pass 8: enforce synthesis deps on all findings-producing steps ---
        # LLMs routinely under-specify deps for synthesis/report steps, causing the
        # report to be generated before per-crate analysis and cross-crate analysis
        # finish.  This pass detects synthesis steps (produces a key ending in
        # "_report" or containing "final") and adds deps on every step that produces
        # a key ending in "_findings", "_results", "_audit", or "_review".
        _SYNTHESIS_KEY_RE = re.compile(r"_report$|^final_|^consolidated_|^summary_", re.IGNORECASE)
        _FINDINGS_KEY_RE = re.compile(
            r"(?:^findings_|_findings$|_results$|_audit$|_review$)",
            re.IGNORECASE,
        )

        # Collect step IDs that produce findings/analysis keys.
        findings_step_ids: set = {
            step_id for key, step_id in produces_map.items() if _FINDINGS_KEY_RE.search(key)
        }
        findings_keys_by_step_id: Dict[str, str] = {
            step_id: key for key, step_id in produces_map.items() if _FINDINGS_KEY_RE.search(key)
        }

        if findings_step_ids:
            for step in result:
                if not isinstance(step, dict):
                    continue
                ctx = step.get("context") or {}
                produces_key = str(ctx.get("produces", "") or "") or str(
                    step.get("produces", "") or ""
                )
                desc = str(step.get("desc", step.get("description", "")) or "")
                step_type = str(step.get("type", "") or "").lower()
                tools_raw = step.get("tools", [])
                tools = (
                    [str(t).strip() for t in tools_raw if str(t).strip()]
                    if isinstance(tools_raw, list)
                    else [t.strip() for t in str(tools_raw).split(",") if t.strip()]
                )
                is_synthesis_step = bool(
                    (produces_key and _SYNTHESIS_KEY_RE.search(produces_key))
                    or (
                        step_type in {"doc", "documentation", "review"}
                        and re.search(
                            r"\b(synthesize|summarize|compile|report|write up|consolidate|"
                            r"findings summary|per-crate findings)\b",
                            desc,
                            re.IGNORECASE,
                        )
                    )
                    or (
                        "write" in tools
                        and re.search(
                            r"\b(final|prioritized|consolidated|summary|report|findings)\b",
                            desc,
                            re.IGNORECASE,
                        )
                    )
                )
                if not is_synthesis_step:
                    is_synthesis_step = bool(
                        produces_key
                        in {
                            "cross_target_findings",
                            "cross_crate_findings",  # legacy
                            "cross_module_findings",
                            "cross_package_findings",
                            "cross_component_findings",
                        }
                        or re.search(
                            r"\bcross[- ](?:target|crate|module|package|component)\b",
                            desc,
                            re.IGNORECASE,
                        )
                    )
                if not is_synthesis_step:
                    continue
                deps_raw = step.get("deps") or step.get("depends_on") or []
                current_set = {str(d) for d in deps_raw}
                current_id = str(step.get("id", ""))
                current_pos = step_positions.get(current_id, 10_000)
                missing = sorted(
                    fid
                    for fid in findings_step_ids
                    if fid != current_id
                    and fid not in current_set
                    and step_positions.get(fid, -1) < current_pos
                )
                if missing:
                    new_deps = sorted(current_set | set(missing))
                    deps_field = "deps" if "deps" in step else "depends_on"
                    step[deps_field] = new_deps
                    if "deps" in step and "depends_on" in step:
                        step["depends_on"] = new_deps
                    current_inputs = [
                        str(item).strip()
                        for item in (step.get("inputs") or step.get("consumes") or [])
                        if str(item).strip()
                    ]
                    for dep_id in missing:
                        key = findings_keys_by_step_id.get(dep_id)
                        if key and key not in current_inputs:
                            current_inputs.append(key)
                    if current_inputs:
                        step["inputs"] = current_inputs
                    logger.info(
                        "Step %s: added findings deps %s to synthesis step (produces='%s')",
                        step.get("id"),
                        missing,
                        produces_key,
                    )

        return result

    def to_execution_plan(self) -> ExecutionPlan:
        """Convert readable task plan to full ExecutionPlan."""
        import uuid

        # Enrich steps with inferred exec types and data-flow keys when the LLM
        # did not annotate them (e.g. smaller models that ignore the exec field).
        enriched_steps = self._enrich_step_dicts(self.steps)

        steps = []
        for step_data in enriched_steps:
            plan_step = self._parse_step_data(step_data)
            steps.append(plan_step)

        # Generate a unique plan ID
        plan_id = f"plan_{uuid.uuid4().hex[:8]}"

        # Build full execution plan
        return ExecutionPlan(
            id=plan_id,
            goal=self.desc,
            steps=steps,
            metadata={
                "task_name": self.name,
                "complexity": self.complexity.value,
                "estimated_duration": self.duration,
                "requires_approval": self.approval,
            },
        )

    def _parse_step_data(self, step_data: Union[List, Dict[str, Any]]) -> PlanStep:
        """Parse step data (list tuple or rich dict) into PlanStep."""
        if isinstance(step_data, dict):
            return self._parse_step_dict(step_data)
        return self._parse_step_list(step_data)

    def _parse_step_dict(self, step_data: Dict[str, Any]) -> PlanStep:
        """Parse rich dict step — supports all execution node fields."""
        step_id = str(step_data["id"])
        step_type_str = str(step_data.get("type", "analyze"))
        description = str(step_data.get("desc", step_data.get("description", "")))
        step_type = self._map_step_type(step_type_str)

        tools_raw = step_data.get("tools", "")
        if isinstance(tools_raw, list):
            tools = [str(t).strip() for t in tools_raw if str(t).strip()]
        elif isinstance(tools_raw, str):
            tools = [t.strip() for t in tools_raw.split(",") if t.strip()]
        else:
            tools = []

        deps_raw = step_data.get("deps", step_data.get("depends_on", []))
        # Strip self-referential deps: a step depending on its own ID can never be satisfied.
        dependencies = (
            [str(d) for d in deps_raw if str(d) != step_id] if isinstance(deps_raw, list) else []
        )

        execution = str(step_data.get("exec", step_data.get("execution", ""))).lower()
        if not execution and step_type == StepType.REVIEW:
            execution = _infer_exec_type(description) or ""
        node = str(step_data.get("node", ""))
        exit_criteria = list(step_data.get("exit", step_data.get("exit_criteria", [])))
        requires_approval = (
            step_type == StepType.DEPLOYMENT or step_type == StepType.PLANNING or self.approval
        )

        loop_over = str(step_data.get("loop_over", ""))
        produces = str(step_data.get("produces", ""))
        inputs_raw = step_data.get("inputs", step_data.get("consumes", []))
        inputs = (
            [str(k).strip() for k in inputs_raw if str(k).strip()]
            if isinstance(inputs_raw, list)
            else []
        )
        condition_on = str(step_data.get("condition_on", ""))
        condition = str(step_data.get("condition", "non_empty"))
        branches_raw = step_data.get("branches", {})
        branches: Dict[str, Any] = (
            {str(k): [str(v) for v in vals] for k, vals in branches_raw.items()}
            if isinstance(branches_raw, dict)
            else {}
        )

        ctx: Dict[str, Any] = {}
        if tools:
            ctx["tools"] = tools
        if node:
            ctx["node"] = node
        if execution:
            ctx["execution"] = execution
        if loop_over:
            ctx["loop_over"] = loop_over
        if produces:
            ctx["produces"] = produces
        items_raw = step_data.get("items", [])
        if items_raw:
            ctx["items"] = list(items_raw)
        if condition_on:
            ctx["condition_on"] = condition_on
            ctx["condition"] = condition
        if branches:
            ctx["branches"] = branches

        # Budget defaults by step class:
        # - loop: 15 (per-iteration budget; independent of iteration count)
        # - doc/synthesis: 8 (read findings + write report)
        # - all others: 10
        #
        # The planner explicitly requests more via "tool_calls": N when a step
        # needs a larger budget (e.g. deep per-target review).  The framework
        # no longer auto-bumps based on description heuristics — that heuristic
        # was rust-specific (it required Arc/Rc/ownership keywords) and any
        # generic equivalent ended up over-matching ordinary loop steps.
        _is_doc = step_type_str.lower() in ("doc", "document", "documentation")
        if execution == "loop":
            _default_budget = 15
        elif _is_doc:
            _default_budget = 8
        else:
            _default_budget = 10
        return PlanStep(
            id=step_id,
            description=description,
            step_type=step_type,
            depends_on=dependencies,
            inputs=inputs,
            estimated_tool_calls=int(step_data.get("tool_calls", _default_budget)),
            requires_approval=requires_approval,
            sub_agent_role=self._get_sub_agent_role(step_type),
            allowed_tools=tools,
            context=ctx,
            execution=execution,
            exit_criteria=[str(c) for c in exit_criteria],
        )

    def _parse_step_list(self, step_data: List) -> PlanStep:
        """Parse compact list step: [id, type, desc, tools, deps, exec?]."""
        step_id = str(step_data[0])
        step_type_str = step_data[1]
        description = step_data[2]

        # Map readable step type to full StepType
        step_type = self._map_step_type(step_type_str)

        # Parse tools and dependencies. Tool hints are preserved on PlanStep so
        # execution adapters can constrain sub-agents without dropping required tools.
        tools = []
        dependencies = []

        if len(step_data) > 3:
            # Fourth element can be tools (string) or dependencies (list)
            fourth = step_data[3]
            if isinstance(fourth, list):
                # Strip self-referential deps at parse time
                dependencies = [str(d) for d in fourth if str(d) != step_id]
            elif isinstance(fourth, str):
                tools = [tool.strip() for tool in fourth.split(",") if tool.strip()]

        if len(step_data) > 4:
            # Fifth element is dependencies if fourth was tools
            deps = step_data[4]
            if isinstance(deps, list):
                # Strip self-referential deps at parse time
                dependencies = [str(d) for d in deps if str(d) != step_id]

        # Optional 6th element: explicit execution node type
        execution = ""
        if len(step_data) > 5:
            execution = str(step_data[5]).lower()
        if not execution and step_type == StepType.REVIEW:
            execution = _infer_exec_type(description) or ""

        # Check if deployment or high-risk step
        requires_approval = (
            step_type == StepType.DEPLOYMENT or step_type == StepType.PLANNING or self.approval
        )

        ctx: Dict[str, Any] = {}
        if tools:
            ctx["tools"] = tools
        if execution:
            ctx["execution"] = execution

        return PlanStep(
            id=step_id,
            description=description,
            step_type=step_type,
            depends_on=dependencies,
            estimated_tool_calls=10,
            requires_approval=requires_approval,
            sub_agent_role=self._get_sub_agent_role(step_type),
            allowed_tools=tools,
            context=ctx,
            execution=execution,
        )

    def _map_step_type(self, step_type_str: str) -> StepType:
        """Map readable step type to StepType enum.

        Uses TaskTypeRegistry.to_planning_step_type() as the SINGLE SOURCE OF TRUTH.
        This ensures consistency across all systems and eliminates duplicate mappings.

        For the canonical alias registry, see victor/framework/task_types.py
        """
        from victor.framework.task_types import TaskTypeRegistry

        # Use the canonical registry - SINGLE SOURCE OF TRUTH
        registry = TaskTypeRegistry.get_instance()
        result = registry.to_planning_step_type(step_type_str)

        # Default to IMPLEMENTATION if no mapping found (backward compatibility)
        return result if result is not None else StepType.IMPLEMENTATION

    def _get_sub_agent_role(self, step_type: StepType) -> Optional[str]:
        """Map step type to sub-agent role."""
        role_map = {
            StepType.RESEARCH: "researcher",
            StepType.PLANNING: "planner",
            StepType.IMPLEMENTATION: "executor",
            StepType.TESTING: "tester",
            StepType.REVIEW: "reviewer",
            StepType.DEPLOYMENT: "deployer",
        }
        return role_map.get(step_type)

    def to_yaml(self) -> str:
        """Convert to YAML format for storage/human editing."""
        import yaml

        plan = self.to_execution_plan()

        # Build YAML structure
        yaml_data = {
            "workflows": {
                self.name: {
                    "description": self.desc,
                    "metadata": {
                        "complexity": self.complexity.value,
                        "estimated_duration": self.duration,
                        "requires_approval": self.approval,
                    },
                    "nodes": [],
                }
            }
        }

        # Convert steps to nodes
        for step in plan.steps:
            node = {
                "id": step.id,
                "type": "agent",
                "role": step.sub_agent_role or "executor",
                "goal": step.description,
                "description": step.description,
                "tool_budget": step.estimated_tool_calls or 10,
            }

            if step.depends_on:
                node["depends_on"] = step.depends_on

            if step.requires_approval:
                node["requires_approval"] = True

            yaml_data["workflows"][self.name]["nodes"].append(node)

        return yaml.safe_dump(yaml_data, sort_keys=False)

    def to_markdown(self) -> str:
        """Convert to markdown for display."""
        plan = self.to_execution_plan()
        lines = [
            f"# {self.name}",
            "",
            f"**Description**: {self.desc}",
            f"**Complexity**: {self.complexity.value}",
            f"**Estimated**: {self.duration or 'Unknown'}",
            f"**Approval**: {'Required' if self.approval else 'Not required'}",
            "",
            "## Steps",
            "",
        ]

        for step in plan.steps:
            status_icon = "⏳"
            lines.append(f"{status_icon} **Step {step.id}**: {step.description}")
            lines.append(f"   - Type: {step.step_type.value}")
            if step.depends_on:
                lines.append(f"   - Depends on: {', '.join(step.depends_on)}")
            lines.append("")

        return "\n".join(lines)

    @classmethod
    def from_execution_plan(cls, plan: ExecutionPlan) -> "ReadableTaskPlan":
        """Create readable task plan from full ExecutionPlan."""
        steps_data = []

        for step in plan.steps:
            # Map StepType to readable type string
            type_str = cls._step_type_to_readable(step.step_type)

            # Build step data list
            step_list = [int(step.id), type_str, step.description]

            # Add tools if available
            if step.allowed_tools:
                tools_str = ",".join(step.allowed_tools)
                step_list.append(tools_str)

            # Add dependencies
            if step.depends_on:
                if not step.allowed_tools:
                    step_list.append([])  # Placeholder for tools
                step_list.append([int(d) for d in step.depends_on])
            elif step.allowed_tools:
                step_list.append([])  # Empty deps if no dependencies

            steps_data.append(step_list)

        metadata = plan.metadata or {}

        return cls(
            name=metadata.get("task_name", plan.goal[:50]),
            complexity=TaskComplexity(metadata.get("complexity", "moderate")),
            desc=plan.goal,  # ExecutionPlan uses goal, not description
            steps=steps_data,
            duration=metadata.get("estimated_duration"),
            approval=metadata.get("requires_approval", False),
        )

    @staticmethod
    def _step_type_to_readable(step_type: StepType) -> str:
        """Convert StepType enum to readable string."""
        return {
            StepType.RESEARCH: "research",
            StepType.PLANNING: "planning",
            StepType.IMPLEMENTATION: "feature",
            StepType.TESTING: "test",
            StepType.REVIEW: "review",
            StepType.DEPLOYMENT: "deploy",
        }.get(step_type, "feature")

    def get_contextual_tools(
        self,
        tool_selector,
        step_index: int,
        conversation_stage=None,
    ) -> List:
        """Get context-appropriate tools for a specific plan step.

        This method uses StepAwareToolSelector to return only the tools
        that are relevant for this step type, complexity, and stage.
        This provides 50-80% token savings compared to exposing all tools.

        Args:
            tool_selector: ToolSelector instance for tool registry access
            step_index: Index of step in plan (0-based)
            conversation_stage: Optional conversation stage

        Returns:
            List of ToolDefinition objects for this step

        Example:
            from victor.agent.planning import ReadableTaskPlan
            from victor.agent.tool_selection import ToolSelector

            plan = ReadableTaskPlan(...)
            tool_selector = ToolSelector(...)

            # Get tools for first step
            tools = plan.get_contextual_tools(tool_selector, step_index=0)
            print(f"Tools for step 0: {[t.name for t in tools]}")
        """
        from victor.agent.planning.tool_selection import StepAwareToolSelector

        if step_index >= len(self.steps):
            return []

        step_data = self.steps[step_index]
        if isinstance(step_data, dict):
            step_type = str(step_data.get("type", ""))
            step_description = str(step_data.get("desc", step_data.get("description", "")))
        else:
            step_type = step_data[1]  # [id, type, desc, tools, deps]
            step_description = step_data[2]

        # Create step-aware selector
        selector = StepAwareToolSelector(
            tool_selector=tool_selector,
        )

        # Get tools for this step
        return selector.get_tools_for_step(
            step_type=step_type,
            complexity=self.complexity,
            step_description=step_description,
            conversation_stage=conversation_stage,
        )

    def to_json(self, verbose: bool = False) -> str:
        """Convert to JSON string.

        Args:
            verbose: If False, use compact list representation (default)
                    If True, use expanded object representation

        Returns:
            JSON string
        """
        if verbose:
            # Expanded to verbose JSON
            plan = self.to_execution_plan()
            return json.dumps(
                {
                    "name": self.name,
                    "complexity": self.complexity.value,
                    "description": self.desc,
                    "steps": [
                        {
                            "id": step.id,
                            "type": step.step_type.value,
                            "description": step.description,
                            "depends_on": step.depends_on,
                            "estimated_tool_calls": step.estimated_tool_calls,
                            "requires_approval": step.requires_approval,
                        }
                        for step in plan.steps
                    ],
                    "estimated_duration": self.duration,
                    "requires_approval": self.approval,
                },
                indent=2,
            )
        else:
            # Use compact schema (this is what LLM generates)
            return self.model_dump_json(exclude_none=True)

    @classmethod
    def get_llm_prompt(cls) -> str:
        """Get optimized prompt for LLM to generate readable task plans.

        Uses readable keywords for LLM reliability while maintaining
        token efficiency through list-based format.
        """
        return """You are a task planning assistant. Create a task plan in JSON format.

{
  "name": "short task name",
  "complexity": "simple|moderate|complex",
  "desc": "task description",
  "steps": [ ... ],
  "duration": "estimated time (optional)",
  "approval": false
}

─── STEP TYPES (canonical names + common aliases) ────────────────────────────
Canonical types: research, planning, feature, bugfix, refactor, test, review, deploy, analyze, doc
• research    — investigate, analyze, review, doc, documentation, explore, inspect
• planning    — plan, design, architect, strategy, conceptual
• feature     — create, implement, new, add, write, generate, scaffold
• bugfix      — edit, fix, debug, patch, refactor, modify, change, restructure
• refactor    — restructure, reorganize, cleanup, improve_structure, reorganize_code
• test        — testing, verify, validate, check, unit_test
• review      — audit, examine, assess, evaluate, code_review, comprehensive_review
• deploy      — deployment, action, execute, run, release
• analyze     — analysis, investigation, assess, evaluate, check, understand
• doc         — documentation, write_docs, create_docs, document

NOTE: You may use ANY of the aliases shown. They all map to canonical types internally.
For example: "bugfix", "fix", "patch" → feature (implementation step type)
           "refactor", "restructure" → feature (implementation step type)
           "investigate", "inspect" → research (analysis step type)

─── TOOLS ────────────────────────────────────────────────────────────────────
read, write, grep, git, shell, test, code_search, overview, scaffold

─── EXECUTION SHAPES (exec field) ───────────────────────────────────────────
compute     — deterministic function, NO model call (build checklists, format output)
tool        — tool calls only, NO model reasoning (read manifest, ls files, parse config)
agent       — single model-backed worker (DEFAULT when exec is omitted)
team        — parallel or hierarchical multi-agent formation
loop        — iterate over a collection one item at a time with exit criteria
conditional — branch-route downstream steps based on a plan_state condition (no model call)
approval    — user checkpoint; plan pauses for user review before continuing

─── COMPACT FORMAT (simple steps) ───────────────────────────────────────────
[id, type, description, "tool1,tool2", [dep_id1], "exec"]

─── DICT FORMAT (required for loop / conditional / compute / approval) ───────
{
  "id": N, "type": "...", "desc": "...",
  "tools":  [...],          ← allowed tools
  "deps":   [...],          ← control-flow: step IDs that must COMPLETE before this runs
  "inputs": [...],          ← data-flow: plan_state KEYS this step reads from prior steps
  "exec":   "...",
  "produces": "KEY",        ← stores this step's list output in plan_state as KEY
  "tool_calls": 15,         ← optional: override default tool budget (default=10)
  "exit":   [...]           ← completion criteria
}

─── DATA FLOW: deps vs inputs ────────────────────────────────────────────────
deps    control WHEN a step runs (execution order).
inputs  declare WHAT data a step needs from plan_state (data routing).

The runtime injects plan_state[key] into the sub-agent's task for each key
listed in inputs. Without inputs, the sub-agent has no knowledge of prior
results and must re-read files from scratch — wasteful and error-prone.

RULES:
• Every step that reads a prior step's named output MUST declare inputs.
• inputs values must match produces keys from earlier steps exactly.
• A step that produces a key must NOT list that same key in its own inputs.
• deps + inputs together fully specify the dependency graph.

PARALLELISM (critical for performance):
• The runtime dispatches ALL steps whose deps are satisfied simultaneously.
• deps must be MINIMAL: only list the step IDs whose produces values appear in
  your inputs. Do NOT chain steps sequentially if they only share the same parent.
• WRONG: step 4 deps=[3], step 5 deps=[4]  — forces 5 to wait for 4 needlessly.
• RIGHT:  step 4 deps=[2], step 5 deps=[2]  — both run in parallel after step 2.
• Multiple analysis steps that all need the same inventory should share the same
  deps entry, not form a chain. The runtime runs all ready steps at once.

─── PARALLEL SIBLINGS (same deps → run concurrently) ────────────────────────
Steps 3, 4, 5 all need only review_targets (from step 2) — run in parallel:
  {"id": 3, "deps": [2], "inputs": ["review_targets"], "produces": "findings_A"},
  {"id": 4, "deps": [2], "inputs": ["review_targets"], "produces": "findings_B"},
  {"id": 5, "deps": [2], "inputs": ["review_targets"], "produces": "findings_C"},
  {"id": 6, "deps": [3, 4, 5], "inputs": ["findings_A", "findings_B", "findings_C"]}
NOT: {"id": 3, "deps": [2]}, {"id": 4, "deps": [3]}, {"id": 5, "deps": [4]} ← serial!

─── LOOP NODE ────────────────────────────────────────────────────────────────
{"id": N, "exec": "loop",
 "loop_over": "review_targets",     ← plan_state key containing the item list
 "inputs":    ["review_targets"],   ← declare it as input so it is injected
 "exit": ["all targets reviewed"]}

─── CONDITIONAL NODE ─────────────────────────────────────────────────────────
{"id": N, "exec": "conditional",
 "condition_on": "review_targets",
 "condition":    "multiple",          ← "non_empty"|"multiple"|"single"|"empty"
 "produces":     "has_multiple_targets",
 "inputs":       ["review_targets"],
 "branches": {"true": ["6a"], "false": ["6b"]}}

─── COMPUTE NODE ─────────────────────────────────────────────────────────────
{"id": N, "exec": "compute",
 "node":    "my_node",                ← registered compute node name
 "produces": "checklist",
 "exit":    ["checklist has 10+ items"]}

─── APPROVAL NODE ────────────────────────────────────────────────────────────
{"id": N, "type": "review", "desc": "Review plan before executing fixes",
 "exec": "approval"}

─── TOOL BUDGET HINTS ────────────────────────────────────────────────────────
Defaults (no override needed unless you want more/less):
• Loop step:             15  per-iteration (each item gets its own budget)
• Synthesis / doc step:  8   (read findings + write; default raised from 5)
• All other steps:       10

Override with "tool_calls": N when the default is insufficient:
• Per-target deep analysis: "tool_calls": 20  (reads many files per target)
• Cross-target / large inventory: "tool_calls": 15
• Shallow inventory step: "tool_calls": 5  (just ls + one read)

─── EXAMPLES ─────────────────────────────────────────────────────────────────

Simple:
{
  "name": "Fix login bug",
  "complexity": "simple",
  "desc": "Fix login bug",
  "steps": [
    [1, "analyze", "Find the bug", "grep,code_search", [], "tool"],
    [2, "feature", "Fix the bug", "read,write", [1]]
  ]
}

Complex (multi-target codebase review — generic data-flow graph, language-agnostic):
{
  "name": "Codebase review (workspace-by-workspace)",
  "complexity": "complex",
  "desc": "Review a multi-target codebase one target at a time, then synthesize",
  "steps": [
    [1, "analyze", "Read root project/manifest file to identify the build system and workspace layout", "read", [], "tool"],
    {"id": 2, "type": "analyze",
     "desc": "Inventory all review targets (workspace members, packages, modules, or top-level directories) from the manifest and filesystem",
     "tools": ["shell", "read"], "deps": [1], "exec": "tool",
     "produces": "review_targets",
     "exit": ["list of target directories returned"]},
    {"id": 3, "type": "analyze",
     "desc": "Map source file inventory for each review target (src/, tests/, benches/, examples/ layouts)",
     "tools": ["shell", "read"], "deps": [2], "exec": "tool",
     "inputs": ["review_targets"],
     "produces": "target_file_inventory",
     "tool_calls": 15,
     "exit": ["file tree per target captured"]},
    {"id": 4, "type": "doc",
     "desc": "Create a domain-appropriate review checklist covering the categories the user asked about (concurrency, performance, error handling, etc.). Categories MUST be supplied verbatim in the step description so the checklist generator can extract them.",
     "tools": [], "deps": [3], "exec": "compute",
     "node": "_checklist_artifact",
     "produces": "review_checklist",
     "exit": ["checklist covers every requested category"]},
    [5, "review", "Present checklist to user for approval before analysis", "", [4], "approval"],
    {"id": 6, "type": "analyze",
     "desc": "Route: multi-target workspace vs single target — select analysis strategy",
     "deps": [5], "exec": "conditional",
     "condition_on": "review_targets", "condition": "multiple",
     "inputs": ["review_targets"],
     "produces": "has_multiple_targets",
     "branches": {"true": ["7a"], "false": ["7b"]}},
    {"id": "7a", "type": "analyze",
     "desc": "Review each review target against the checklist — read sources, record file:line evidence and concrete findings per target. Do not stop at file listings.",
     "tools": ["read", "grep", "code_search"], "deps": [6], "exec": "loop",
     "loop_over": "review_targets",
     "inputs": ["review_targets", "review_checklist"],
     "tool_calls": 20,
     "produces": "per_target_findings",
     "exit": ["every target reviewed", "findings recorded per target with file:line evidence"]},
    {"id": "7b", "type": "analyze",
     "desc": "Review the single target against the checklist — read sources file by file, record file:line evidence and concrete findings",
     "tools": ["read", "grep", "code_search"], "deps": [6],
     "inputs": ["target_file_inventory", "review_checklist"],
     "tool_calls": 20,
     "produces": "per_target_findings"},
    {"id": 8, "type": "analyze",
     "desc": "Cross-target analysis: shared patterns, redundant work across targets, boundary inconsistencies, global optimization opportunities. Cite file:line evidence.",
     "tools": ["read", "grep", "code_search"], "deps": ["7a", "7b"],
     "inputs": ["review_targets", "per_target_findings"],
     "tool_calls": 15,
     "produces": "cross_target_findings"},
    {"id": 9, "type": "doc",
     "desc": "Synthesize all findings into a prioritized report: per-target summaries, cross-target themes, ranked recommendations with effort/impact, specific code locations",
     "tools": ["write"], "deps": [8],
     "inputs": ["review_checklist", "per_target_findings", "cross_target_findings", "review_targets"],
     "produces": "final_report",
     "exit": ["report written to file", "all targets covered", "recommendations ranked"]},
    [10, "review", "Present consolidated report to user for feedback", "", [9]]
  ],
  "duration": "4-6hr"
}

Notes for the planner:
• Target vocabulary above ("review_targets", "per_target_findings", "cross_target_findings") is generic.
  Use it for any codebase review regardless of language.
• Step 4 uses node="_checklist_artifact" — a generic compute node. Put the requested
  category list verbatim in the step description so it can be extracted.
• For language-specific deterministic scanners, the vertical/plugin must register the
  compute node and the plan must reference it explicitly via "node": "<registered_name>".
  Do not invent node names that have not been registered.

Please generate the task plan as valid JSON. Do not include markdown code blocks."""

    @classmethod
    def get_complexity_prompt(cls) -> str:
        """Get prompt for classifying task complexity."""
        return """Classify the task complexity:

Task: {user_request}

Consider:
- SIMPLE: Single file, well-defined scope, <30 minutes, 2-3 steps
- MODERATE: Multiple files, some uncertainty, 30min-2 hours, 3-5 steps
- COMPLEX: Multiple components, high uncertainty, >2 hours, 5-8 steps

Respond with ONLY valid JSON:
{
  "complexity": "simple|moderate|complex",
  "reason": "brief explanation"
}"""


class TaskPlannerContext:
    """Session context manager for task planning.

    Manages task plans within a conversation session, allowing plans
    to be referenced, updated, and executed across multiple turns.
    """

    def __init__(self):
        self.current_plan: Optional[ExecutionPlan] = None
        self.plans_history: List[ExecutionPlan] = []
        self.approved_plans: List[str] = []

    def set_plan(self, plan: ExecutionPlan) -> None:
        """Set the current active plan."""
        self.current_plan = plan
        logger.info(f"Set current plan: {plan.goal}")

    def approve_plan(self) -> None:
        """Mark current plan as approved."""
        if self.current_plan:
            plan_id = id(self.current_plan)
            self.approved_plans.append(str(plan_id))
            logger.info(f"Approved plan: {self.current_plan.goal}")

    def archive_plan(self) -> None:
        """Archive current plan to history."""
        if self.current_plan:
            self.plans_history.append(self.current_plan)
            self.current_plan = None
            logger.info("Archived current plan")

    def get_plan_summary(self) -> Dict[str, Any]:
        """Get summary of all plans in context."""
        return {
            "current_plan": self.current_plan.goal if self.current_plan else None,
            "total_plans": len(self.plans_history) + (1 if self.current_plan else 0),
            "approved_plans": len(self.approved_plans),
            "history": [
                {"goal": plan.goal, "steps": len(plan.steps)}
                for plan in self.plans_history[-5:]  # Last 5 plans
            ],
        }

    def to_context_dict(self) -> Dict[str, Any]:
        """Export context for inclusion in LLM prompts."""
        summary = self.get_plan_summary()

        context = {
            "task_planner": {
                "active": self.current_plan is not None,
                "total_plans": summary["total_plans"],
                "approved_count": summary["approved_plans"],
            }
        }

        if self.current_plan:
            plan_dict = self._plan_to_dict(self.current_plan)
            context["task_planner"]["current_plan"] = plan_dict

        if self.plans_history:
            context["task_planner"]["recent_plans"] = summary["history"]

        return context

    def _plan_to_dict(self, plan: ExecutionPlan) -> Dict[str, Any]:
        """Convert plan to dictionary for context."""
        return {
            "goal": plan.goal,
            "steps": [
                {
                    "id": step.id,
                    "description": step.description,
                    "type": step.step_type.value,
                    "status": step.status.value if step.status else "pending",
                }
                for step in plan.steps
            ],
            "metadata": plan.metadata,
        }

    @classmethod
    def from_context_dict(cls, context: Dict[str, Any]) -> "TaskPlannerContext":
        """Restore context from dictionary (e.g., from session storage)."""
        ctx = cls()

        if "task_planner" in context:
            tp_data = context["task_planner"]
            # Reconstruct plans from context if needed
            # This is a simplified version - full restoration would need more logic
            if tp_data.get("current_plan"):
                # Would need to reconstruct ExecutionPlan from dict
                pass

        return ctx


# Helper functions for workflow integration


#: Plan generation used to send a bare user message with no system prompt. Naming
#: the job and the required output shape up front costs a few tokens and makes the
#: JSON-shaped response substantially more likely.
_PLAN_GENERATION_SYSTEM_PROMPT = (
    "You are a planning assistant. Produce a task plan as a single JSON object "
    "matching the schema in the user message. Respond with JSON only — no prose, "
    "no markdown fences, no commentary before or after."
)


class _EmptyPlanResponse(ValueError):
    """The model returned no content at all for a plan request.

    Distinct from "returned content we could not parse as a plan": an empty
    response is deterministic and reproduces on an identical retry, so it calls
    for parameter escalation rather than another identical attempt. Subclasses
    ValueError so existing callers catching ValueError are unaffected.
    """


def _empty_response_diagnostics(plan_response: Any) -> Dict[str, Any]:
    """Best-effort extraction of empty-response signals from a provider reply.

    Providers differ in shape (dict vs object) and in where they report reasoning
    output, so every lookup is defensive: a missing signal yields an undiagnosed
    result rather than an exception.
    """
    diagnostics: Dict[str, Any] = {}
    if plan_response is None:
        return diagnostics

    def _get(source: Any, key: str) -> Any:
        if isinstance(source, dict):
            return source.get(key)
        return getattr(source, key, None)

    for key in ("reasoning_content", "reasoning", "thinking"):
        value = _get(plan_response, key)
        if isinstance(value, str) and value:
            diagnostics["reasoning_chars"] = len(value)
            break

    if "reasoning_chars" not in diagnostics:
        usage = _get(plan_response, "usage") or {}
        reasoning_tokens = _get(usage, "reasoning_tokens")
        if isinstance(reasoning_tokens, int) and reasoning_tokens > 0:
            # Approximate chars from tokens purely to signal "reasoning happened";
            # only the truthiness of this value drives the diagnosis.
            diagnostics["reasoning_chars"] = reasoning_tokens * 4

    for key in ("stop_reason", "finish_reason"):
        value = _get(plan_response, key)
        if value:
            diagnostics["stop_reason"] = value
            break

    return diagnostics


def _provider_supports_reasoning_effort(provider: Any, model: Optional[str]) -> bool:
    """Whether ``reasoning_effort`` may be forwarded for this provider/model.

    Sending it where it is not accepted converts a recoverable empty response
    into a hard 400, so this defaults to False whenever it cannot be confirmed.
    """
    checker = getattr(provider, "supports_reasoning_effort", None)
    if not callable(checker):
        return False
    try:
        return bool(checker(model))
    except Exception:
        logger.debug("supports_reasoning_effort check failed", exc_info=True)
        return False


async def generate_task_plan(
    provider,
    user_request: str,
    complexity: Optional[TaskComplexity] = None,
    model: Optional[str] = None,
    max_retries: int = 2,
    conversation_context: Optional[str] = None,
) -> ReadableTaskPlan:
    """Generate a readable task plan using LLM.

    This function uses the framework's response parsing utilities for robust
    JSON extraction and validation. Includes retry logic for reliability.

    Args:
        provider: LLM provider instance
        user_request: Natural language task description
        complexity: Optional pre-classified complexity level
        model: Optional model identifier (if None, will try to get from provider)
        max_retries: Maximum number of retries for plan generation (default: 2)
        conversation_context: Optional prior conversation summary to ground the plan in
            specific findings rather than generating a generic template.

    Returns:
        Validated ReadableTaskPlan

    Raises:
        ValueError: If model identifier cannot be determined
        ValidationError: If LLM response cannot be validated as ReadableTaskPlan
    """
    from victor.providers.base import Message

    # Get model identifier
    if not model:
        # Try to get from orchestrator if available
        if hasattr(provider, "model") and provider.model:
            model = provider.model
        elif hasattr(provider, "_provider") and hasattr(provider._provider, "model"):
            model = provider._provider.model
        else:
            raise ValueError("Model identifier must be provided or available from provider")

    # Classify complexity if not provided
    task_complexity = complexity
    if not task_complexity:
        complexity_prompt = ReadableTaskPlan.get_complexity_prompt()
        complexity_prompt = complexity_prompt.replace("{user_request}", user_request)

        logger.debug("Classifying task complexity...")
        complexity_response = await provider.chat(
            messages=[Message(role="user", content=complexity_prompt)],
            model=model,
            temperature=0.1,
            max_tokens=200,
        )

        # Extract JSON using framework utilities
        complexity_json = extract_json_from_llm_response(complexity_response)
        if not complexity_json:
            raise ValueError(
                "Failed to extract JSON from complexity classification response. "
                f"Response: {extract_llm_response_content(complexity_response)[:200]}"
            )

        complexity_data = json.loads(complexity_json)
        task_complexity = TaskComplexity(complexity_data["complexity"])
        logger.debug(f"Classified complexity as: {task_complexity.value}")

    # Generate task plan with retries
    plan_prompt = ReadableTaskPlan.get_llm_prompt()
    if conversation_context:
        # Inject prior analysis so the plan is grounded in actual findings, not generic steps.
        plan_prompt = (
            f"{plan_prompt}\n\n"
            f"PRIOR ANALYSIS (ground the plan in these specific findings):\n"
            f"{conversation_context}\n\n"
            f"Task: {user_request}"
        )
    else:
        plan_prompt = f"{plan_prompt}\n\nTask: {user_request}"

    logger.debug(f"Planning prompt length: {len(plan_prompt)} chars")
    logger.debug(f"Planning prompt preview: {plan_prompt[:500]}...")

    # Base parameters for the first attempt. Retries escalate from here — see
    # victor.agent.recovery.empty_response. Retrying an empty response with
    # identical parameters reproduces it deterministically, which is how session
    # sandhi-cdfbc589 spent ~100s on three identical 35s failures.
    # Some models (like qwen2.5-coder) need room to generate structured JSON.
    base_plan_max_tokens = 3000
    plan_max_tokens = base_plan_max_tokens
    plan_temperature = 0.2  # Lower temp for consistent structure
    plan_reasoning_effort: Optional[str] = None
    supports_reasoning_effort = _provider_supports_reasoning_effort(provider, model)

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                logger.info(f"Retry {attempt}/{max_retries} for plan generation")

            request_kwargs: Dict[str, Any] = {}
            if plan_reasoning_effort is not None:
                request_kwargs["reasoning_effort"] = plan_reasoning_effort

            plan_response = await provider.chat(
                messages=[
                    Message(role="system", content=_PLAN_GENERATION_SYSTEM_PROMPT),
                    Message(role="user", content=plan_prompt),
                ],
                model=model,
                temperature=plan_temperature,
                max_tokens=plan_max_tokens,
                **request_kwargs,
            )

            # Log raw response for debugging
            logger.debug(f"Raw plan response type: {type(plan_response)}")

            # Log response details before extraction
            if isinstance(plan_response, dict):
                logger.debug(f"Response dict keys: {list(plan_response.keys())}")
                for key in plan_response.keys():
                    value = plan_response[key]
                    if isinstance(value, str):
                        logger.debug(f"Response dict[{key}] length: {len(value)}")
                        logger.debug(f"Response dict[{key}] preview: {value[:200]}")
                    else:
                        logger.debug(f"Response dict[{key}] type: {type(value)}")
            elif hasattr(plan_response, "__dict__"):
                logger.debug(f"Response object attributes: {list(plan_response.__dict__.keys())}")

            response_content = extract_llm_response_content(plan_response)
            logger.info(
                f"Plan attempt {attempt + 1}: response length={len(response_content) if response_content else 0}, "
                f"response_preview={response_content[:200] if response_content else 'empty'}..."
            )

            # Extract JSON using framework utilities. Distinguish "the model
            # returned nothing" from "the model returned prose we could not
            # parse" — they have different causes and different remedies, but
            # both used to surface as `Response: empty`.
            plan_json = extract_json_from_llm_response(plan_response)
            if not plan_json:
                if not response_content:
                    raise _EmptyPlanResponse(
                        "Model returned no content for the plan request "
                        f"(max_tokens={plan_max_tokens}, temperature={plan_temperature})"
                    )
                raise ValueError(
                    f"No valid JSON found in plan response. " f"Response: {response_content[:500]}"
                )

            logger.debug(f"Extracted JSON: {plan_json[:200]}...")

            # Validate and return. Use schema-aware candidate selection because the
            # generic extractor may return a nested step array before the outer plan.
            plan = _validate_readable_plan_from_candidates(plan_json, response_content)
            logger.info(
                f"Generated plan: {plan.name} with {len(plan.steps)} steps, "
                f"complexity={plan.complexity.value}"
            )
            return plan

        except (ValidationError, ValueError, json.JSONDecodeError) as e:
            last_error = e
            log = logger.warning if attempt >= max_retries else logger.debug
            log(
                "Plan generation attempt %s/%s failed: %s. Response was: %.200s",
                attempt + 1,
                max_retries + 1,
                e,
                response_content if response_content else "empty",
            )

            # An empty response is deterministic: repeating the same request
            # reproduces it. Escalate the parameters before the next attempt.
            if isinstance(e, _EmptyPlanResponse) and attempt < max_retries:
                diagnosis = diagnose_empty_response(_empty_response_diagnostics(plan_response))
                retry = next_retry_parameters(
                    attempt=attempt,
                    base_max_tokens=plan_max_tokens,
                    diagnosis=diagnosis,
                    supports_reasoning_effort=supports_reasoning_effort,
                )
                if retry.escalated:
                    logger.info("Plan generation: %s", retry.reason)
                    plan_max_tokens = retry.max_tokens
                    if retry.temperature is not None:
                        plan_temperature = retry.temperature
                    plan_reasoning_effort = retry.reasoning_effort
                else:
                    logger.warning(
                        "Plan generation: %s — further retries would repeat the "
                        "same request, stopping early",
                        retry.reason,
                    )
                    break

    # All retries exhausted
    raise ValueError(
        f"Failed to generate valid task plan after {max_retries + 1} attempts. "
        f"Last error: {last_error}"
    ) from last_error


def plan_to_workflow_yaml(plan: ReadableTaskPlan) -> str:
    """Convert readable task plan to YAML workflow format.

    This converts the task plan into a YAML workflow that can be
    executed by the WorkflowEngine.

    Args:
        plan: ReadableTaskPlan to convert

    Returns:
        YAML workflow string
    """
    return plan.to_yaml()


def plan_to_session_context(
    plan: ReadableTaskPlan,
    session_id: str,
    context_store=None,
) -> Dict[str, Any]:
    """Add plan to session context for persistence.

    This allows the task plan to be referenced in future conversation turns
    and persisted across sessions.

    Args:
        plan: ReadableTaskPlan to add to context
        session_id: Session identifier
        context_store: Optional context storage backend

    Returns:
        Updated context dictionary
    """
    context = {
        "session_id": session_id,
        "task_plan": {
            "name": plan.name,
            "complexity": plan.complexity.value,
            "description": plan.desc,
            "estimated_duration": plan.duration,
            "requires_approval": plan.approval,
            "steps": [
                {
                    "id": step.get("id") if isinstance(step, dict) else step[0],
                    "type": step.get("type") if isinstance(step, dict) else step[1],
                    "description": (
                        step.get("desc", step.get("description", ""))
                        if isinstance(step, dict)
                        else (step[2] if len(step) > 2 else "")
                    ),
                }
                for step in plan.steps
            ],
        },
        "created_at": plan.model_dump_json(
            include={"name", "complexity", "desc", "duration", "approval"}
        ),
    }

    # Store in context backend if provided
    if context_store:
        context_store.set(session_id, "task_plan", context["task_plan"])

    return context


# Legacy aliases for backward compatibility
CompactTaskPlan = ReadableTaskPlan
CompactStepType = None  # Removed, using readable strings instead

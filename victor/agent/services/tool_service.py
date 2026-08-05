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

"""Tool service implementation.

Extracts tool operations from the AgentOrchestrator into
a focused, single-responsibility service following SOLID principles.

This service handles:
- Tool selection based on context
- Tool execution with budgeting
- Tool usage tracking
"""

from __future__ import annotations

import asyncio
import ast
from victor.core.json_utils import json_dumps, json_loads
import logging
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

from victor.agent.services.tool_batch_executor import execute_tool_call_batch
from victor.agent.services.tool_budget_runtime import BudgetManager, ToolBudgetRuntime
from victor.agent.services.tool_access_policy import ToolAccessPolicy
from victor.agent.services.tool_error_display import print_tool_error_once
from victor.agent.services.tool_usage_stats import ToolUsageStats
from victor.tools.unified.parser import classify_tool_outcome
from victor.tools.core_tool_aliases import (
    canonicalize_core_tool_name,
    normalize_model_tool_name,
)

if TYPE_CHECKING:
    from victor.agent.services.protocols.tool_service import ToolSelectionContext

logger = logging.getLogger(__name__)


@dataclass
class ToolResultContext:
    """State-passed context for post-processing tool pipeline results.

    This is the canonical service-layer carrier for mutable state and callbacks
    needed by ``ToolService.process_tool_results``.
    """

    executed_tools: List[str]
    observed_files: Set[str]
    failed_tool_signatures: Set[str]
    shown_tool_errors: Set[str]
    continuation_prompts: int = 0
    asking_input_prompts: int = 0
    tool_calls_used: int = 0
    record_tool_execution: Optional[Any] = None
    conversation_state: Optional[Any] = None
    unified_tracker: Optional[Any] = None
    usage_logger: Optional[Any] = None
    add_message: Optional[Any] = None
    format_tool_output: Optional[Any] = None
    console: Optional[Any] = None
    presentation: Optional[Any] = None
    stream_context: Optional[Any] = None
    task_type: str = "unknown"


def normalize_tool_result_arguments(arguments: Any) -> Dict[str, Any]:
    """Normalize tool arguments for formatting/pruning decisions."""
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            parsed = json_loads(arguments)
        except Exception:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def _extract_result_items(output: Any, limit: int = 50) -> Optional[List[Dict[str, Any]]]:
    """Compact structured hits from a search-tool output for novelty/fulfillment accounting.

    Returns a small ``[{path, qualified_name, score}]`` list when ``output`` is a dict with a
    ``results`` list (the code_search/grep shape); otherwise ``None``. The formatted result
    strings carry no structured hits, so this is the only place they can be captured. Bounded
    to ``limit`` items to keep the per-turn payload small.
    """
    if not isinstance(output, dict):
        return None
    results = output.get("results")
    if not isinstance(results, list) or not results:
        return None
    items: List[Dict[str, Any]] = []
    for entry in results[:limit]:
        if not isinstance(entry, dict):
            continue
        path = entry.get("path") or entry.get("file_path")
        if not path:
            continue
        items.append(
            {
                "path": str(path),
                "qualified_name": str(
                    entry.get("qualified_name") or entry.get("symbol_type") or ""
                ),
                "score": entry.get("score") or entry.get("combined_score"),
            }
        )
    return items or None


def _extract_tool_diagnostics(output: Any) -> List[str]:
    """Extract concise model-visible diagnostics from structured tool metadata."""
    if not isinstance(output, dict):
        return []

    diagnostics: List[str] = []
    fallback = output.get("fallback")
    if isinstance(fallback, str) and fallback:
        diagnostics.append(f"fallback={fallback}")

    metadata = output.get("metadata")
    if isinstance(metadata, dict):
        for key in ("fallback_guidance", "warning", "diagnostic"):
            value = metadata.get(key)
            if isinstance(value, str) and value and value not in diagnostics:
                diagnostics.append(value)

    return diagnostics


def _append_tool_diagnostics(content: str, diagnostics: List[str]) -> str:
    """Append structured diagnostics to the provider-visible tool output."""
    if not diagnostics:
        return content
    diagnostic_lines = "\n".join(f"- {diagnostic}" for diagnostic in diagnostics)
    return f"{content}\n\nTool diagnostics:\n{diagnostic_lines}"


def _persist_tool_result_message(
    ctx: ToolResultContext,
    *,
    tool_name: str,
    content: str,
    tool_call_id: Optional[str],
    persist_synchronously: bool = False,
) -> bool:
    """Persist a tool result message for provider-visible tool-call pairing."""
    if not ctx.add_message:
        return False

    add_kwargs: Dict[str, Any] = {
        "name": tool_name,
        "tool_call_id": tool_call_id,
    }
    if persist_synchronously:
        add_kwargs["persist_synchronously"] = True
    try:
        ctx.add_message("tool", content, **add_kwargs)
        return True
    except TypeError:
        if not persist_synchronously:
            logger.exception(
                "Failed to persist tool result message for %s (tool_call_id=%s)",
                tool_name,
                tool_call_id,
            )
            return False
        try:
            ctx.add_message(
                "tool",
                content,
                name=tool_name,
                tool_call_id=tool_call_id,
            )
            return True
        except Exception:
            logger.exception(
                "Failed to persist tool result message for %s (tool_call_id=%s)",
                tool_name,
                tool_call_id,
            )
            return False
    except Exception:
        logger.exception(
            "Failed to persist tool result message for %s (tool_call_id=%s)",
            tool_name,
            tool_call_id,
        )
        return False


_PREVIEW_ONLY_TOOL_RESULT_FIELDS = frozenset(
    {
        "formatted_output",
        "formatted_summary",
        "formatted_results",
        "contains_markup",
    }
)


def _strip_preview_only_tool_fields(output: Any) -> Any:
    """Remove preview-only fields from tool results before LLM injection.

    Rich preview payloads improve CLI UX, but they duplicate structured data
    and add token cost when sent back to the model. Keep them for display paths
    while excluding them from the LLM-facing serialization.
    """
    if isinstance(output, dict):
        stripped = {
            key: _strip_preview_only_tool_fields(value)
            for key, value in output.items()
            if key not in _PREVIEW_ONLY_TOOL_RESULT_FIELDS
        }
        # Preserve original content if preview fields were the only payload.
        return stripped or output
    if isinstance(output, list):
        return [_strip_preview_only_tool_fields(value) for value in output]
    return output


def format_and_prune_tool_output(
    tool_name: str,
    arguments: Any,
    output: Any,
    *,
    task_type: str = "unknown",
    formatter: Optional[Any] = None,
) -> tuple[str, str, str, bool, Optional[Any]]:
    """Format tool output for display and LLM injection.

    CRITICAL: LLM receives FULL output for accuracy. Pruning only affects user preview.
    """
    normalized_args = normalize_tool_result_arguments(arguments)
    llm_safe_output = _strip_preview_only_tool_fields(output)
    llm_output_changed = llm_safe_output != output

    if formatter is not None:
        formatted_output = formatter(tool_name, normalized_args, output)
        llm_output = (
            formatter(tool_name, normalized_args, llm_safe_output)
            if llm_output_changed
            else formatted_output
        )
    else:
        from victor.agent.tool_output_formatter import (
            format_tool_output as default_formatter,
        )
        from victor.agent.tool_output_formatter import ToolOutputFormatterConfig

        # CRITICAL: Use very high max_output_chars for LLM input (no truncation)
        # This ensures LLM receives complete tool output for accurate decisions
        llm_formatter_config = ToolOutputFormatterConfig(
            max_output_chars=1_000_000,  # 1MB - effectively no truncation for LLM
            file_structure_threshold=30000,
            min_savings_threshold=0.15,
            max_classes_shown=15,
            max_functions_shown=20,
            sample_lines_start=20,
            sample_lines_end=15,
        )

        # Format with NO truncation for LLM input
        formatted_output = default_formatter(
            tool_name,
            normalized_args,
            output,
            config=llm_formatter_config,
        )
        llm_output = (
            default_formatter(
                tool_name,
                normalized_args,
                llm_safe_output,
                config=llm_formatter_config,
            )
            if llm_output_changed
            else formatted_output
        )

    if not isinstance(formatted_output, str):
        formatted_output = str(formatted_output)
    if not isinstance(llm_output, str):
        llm_output = str(llm_output)

    # Generate preview for user display only (does not affect LLM)
    preview_output = formatted_output
    was_pruned = False
    pruning_info = None

    from victor.config.tool_settings import get_tool_settings
    from victor.tools.output_pruner import get_output_pruner

    tool_settings = get_tool_settings()
    if tool_settings.tool_output_preview_enabled and tool_settings.tool_output_pruning_enabled:
        pruner = get_output_pruner()
        # Prune only for user preview, NOT for LLM input
        preview_output, pruning_info = pruner.prune(
            tool_output=formatted_output,
            task_type=task_type,
            tool_name=tool_name,
            context={
                "formatted_output": True,
                "safe_only": tool_settings.tool_output_pruning_safe_only,
                "tool_args": normalized_args,
            },
        )
        was_pruned = pruning_info.was_pruned

    if not isinstance(preview_output, str):
        preview_output = str(preview_output)

    # Return:
    # - preview_output: user-facing preview payload (may be pruned)
    # - llm_output: full output sent to the model
    # - formatted_output: full display output retained for expansion/debug
    # - was_pruned / pruning_info: preview pruning metadata
    return preview_output, llm_output, formatted_output, was_pruned, pruning_info


def process_tool_results_with_context(
    pipeline_result: Any,
    ctx: ToolResultContext,
) -> List[Dict[str, Any]]:
    """Canonical service-layer implementation for tool result post-processing."""
    results: List[Dict[str, Any]] = []

    for call_result in pipeline_result.results:
        tool_name = getattr(call_result, "tool_name", "unknown") or "unknown"
        normalized_args = normalize_tool_result_arguments(getattr(call_result, "arguments", {}))
        elapsed_ms = float(getattr(call_result, "execution_time_ms", 0.0) or 0.0)
        skipped = bool(getattr(call_result, "skipped", False))
        skip_reason = getattr(call_result, "skip_reason", None)
        outcome_kind = getattr(call_result, "outcome_kind", None)
        block_source = getattr(call_result, "block_source", None)
        retryable = getattr(call_result, "retryable", None)
        user_message = getattr(call_result, "user_message", None)
        tool_call_id = getattr(call_result, "tool_call_id", None)

        try:
            # Invalid tools without tool_call_id do not need a provider-visible response.
            if skipped and not tool_call_id:
                continue

            output = call_result.result
            success = call_result.success
            error_msg = call_result.error

            ctx.executed_tools.append(tool_name)
            if tool_name == "read" and "path" in normalized_args:
                ctx.observed_files.add(str(normalized_args.get("path")))

            if ctx.stream_context is not None:
                ctx.stream_context.reset_activity_timer()

            if success:
                if ctx.continuation_prompts > 0:
                    ctx.continuation_prompts = 0
                if ctx.asking_input_prompts > 0:
                    ctx.asking_input_prompts = 0

            error_type = type(error_msg).__name__ if error_msg and not success else None
            if ctx.record_tool_execution:
                ctx.record_tool_execution(tool_name, success, elapsed_ms, error_type=error_type)
            if ctx.conversation_state:
                ctx.conversation_state.record_tool_execution(tool_name, normalized_args)

            result_dict: Dict[str, Any] = {"success": success}
            if output is not None:
                result_dict["result"] = output
            if ctx.unified_tracker:
                ctx.unified_tracker.update_from_tool_call(tool_name, normalized_args, result_dict)

            follow_up_suggestions = None
            tool_diagnostics = _extract_tool_diagnostics(output)
            semantic_success = success
            if isinstance(output, dict):
                metadata = output.get("metadata")
                if isinstance(metadata, dict):
                    suggestions = metadata.get("follow_up_suggestions")
                    if isinstance(suggestions, list) and suggestions:
                        follow_up_suggestions = suggestions

            if success and isinstance(output, dict) and output.get("success") is False:
                semantic_success = False
                error_msg = output.get("error", "Operation returned success=False")

            if semantic_success:
                error_display = None
            else:
                error_display = user_message or error_msg or skip_reason or "Unknown error"

            # Telemetry truth: the EVENT reflects `### ❌` string errors; the
            # LLM-visible path (semantic_success branches) stays untouched.
            marker_kind, error_detail = classify_tool_outcome(output)
            telemetry_success = semantic_success and marker_kind != "tool_error"

            if ctx.usage_logger and hasattr(ctx.usage_logger, "set_duration_context"):
                ctx.usage_logger.set_duration_context(elapsed_ms)
            if ctx.usage_logger:
                ctx.usage_logger.log_event(
                    "tool_result",
                    {
                        "tool_name": tool_name,
                        "success": telemetry_success,
                        "skipped": skipped,
                        "outcome_kind": outcome_kind or marker_kind,
                        "block_source": block_source,
                        "retryable": retryable,
                        "result": output,
                        "error": error_display or error_detail,
                        "error_detail": error_detail,
                    },
                )

            if semantic_success:
                preview_output, llm_output, full_output, was_pruned, pruning_info = (
                    format_and_prune_tool_output(
                        tool_name=tool_name,
                        arguments=normalized_args,
                        output=output,
                        task_type=ctx.task_type,
                        formatter=ctx.format_tool_output,
                    )
                )
                content = _append_tool_diagnostics(llm_output, tool_diagnostics)
                _persist_tool_result_message(
                    ctx,
                    tool_name=tool_name,
                    content=content,
                    tool_call_id=tool_call_id,
                )
                results.append(
                    {
                        "name": tool_name,
                        "success": True,
                        "elapsed": elapsed_ms / 1000,
                        "args": normalized_args,
                        "result": preview_output,
                        "full_result": full_output,
                        # Compact structured hits (path/symbol) for downstream novelty +
                        # fulfillment accounting — the formatted result/full_result are strings,
                        # so the raw search hits are only recoverable here from `output`.
                        "result_items": _extract_result_items(output),
                        "follow_up_suggestions": follow_up_suggestions,
                        "tool_diagnostics": tool_diagnostics,
                        "was_pruned": was_pruned,
                        "pruning_info": pruning_info,
                        "tool_call_id": tool_call_id,
                        "content": content,
                        "skipped": skipped,
                        "outcome_kind": outcome_kind,
                        "block_source": block_source,
                        "retryable": retryable,
                        "user_message": user_message,
                    }
                )
                continue

            if not skipped:
                sig = f"{tool_name}:{hash(str(sorted(normalized_args.items())))}"
                ctx.failed_tool_signatures.add(sig)

            print_tool_error_once(
                ctx, tool_name, error_display, skipped=skipped, elapsed_ms=elapsed_ms
            )

            if isinstance(output, dict):
                error_output = dict(output)
            else:
                error_output = {"error": error_display}
            if skipped:
                error_output["skipped"] = True
                if skip_reason:
                    error_output["skip_reason"] = skip_reason
            if outcome_kind:
                error_output["outcome_kind"] = outcome_kind
            if block_source:
                error_output["block_source"] = block_source
            if retryable is not None:
                error_output["retryable"] = retryable
            if user_message:
                error_output["user_message"] = user_message
            if tool_diagnostics:
                error_output["tool_diagnostics"] = tool_diagnostics
            if ctx.format_tool_output:
                formatted_error = ctx.format_tool_output(tool_name, normalized_args, error_output)
            else:
                formatted_error, _, _, _, _ = format_and_prune_tool_output(
                    tool_name=tool_name,
                    arguments=normalized_args,
                    output=error_output,
                    task_type=ctx.task_type,
                )
            _persist_tool_result_message(
                ctx,
                tool_name=tool_name,
                content=formatted_error,
                tool_call_id=tool_call_id,
            )
            results.append(
                {
                    "name": tool_name,
                    "success": False,
                    "elapsed": elapsed_ms / 1000,
                    "error": error_display,
                    "result": formatted_error,
                    "full_result": formatted_error,
                    "follow_up_suggestions": follow_up_suggestions,
                    "tool_diagnostics": tool_diagnostics,
                    "was_pruned": False,
                    "tool_call_id": tool_call_id,
                    "content": formatted_error,
                    "skipped": skipped,
                    "outcome_kind": outcome_kind,
                    "block_source": block_source,
                    "retryable": retryable,
                    "user_message": user_message,
                }
            )
        except Exception as exc:
            logger.exception(
                "Failed to post-process tool result for %s (tool_call_id=%s)",
                tool_name,
                tool_call_id,
            )
            fallback_error = (
                f"Tool result post-processing failed for '{tool_name}': {type(exc).__name__}"
            )
            fallback_content = (
                f"Tool result unavailable for '{tool_name}'. Victor failed to post-process this "
                "tool call, so treat it as failed and continue with the available context."
            )
            _persist_tool_result_message(
                ctx,
                tool_name=tool_name,
                content=fallback_content,
                tool_call_id=tool_call_id,
                persist_synchronously=True,
            )
            results.append(
                {
                    "name": tool_name,
                    "success": False,
                    "elapsed": elapsed_ms / 1000,
                    "error": fallback_error,
                    "result": fallback_content,
                    "full_result": fallback_content,
                    "follow_up_suggestions": None,
                    "was_pruned": False,
                    "tool_call_id": tool_call_id,
                    "content": fallback_content,
                    "skipped": True,
                    "outcome_kind": outcome_kind or "tool_result_processing_failed",
                    "block_source": block_source or "tool_result_processing",
                    "retryable": False if retryable is None else retryable,
                    "user_message": fallback_content,
                }
            )

    return results


class ToolServiceConfig:
    """Configuration for ToolService.

    Attributes:
        default_max_tools: Default maximum tools per selection
        default_tool_budget: Default tool budget per session
        enable_parallel_execution: Enable parallel tool execution
        enable_caching: Enable tool result caching
        cache_ttl: Cache TTL in seconds
    """

    def __init__(
        self,
        default_max_tools: int = 10,
        default_tool_budget: int = 100,
        enable_parallel_execution: bool = True,
        enable_caching: bool = True,
        cache_ttl: int = 600,
        retry_enabled: bool = True,
        max_retry_attempts: int = 3,
        retry_base_delay: float = 1.0,
        retry_max_delay: float = 10.0,
    ):
        self.default_max_tools = default_max_tools
        self.default_tool_budget = default_tool_budget
        self.enable_parallel_execution = enable_parallel_execution
        self.enable_caching = enable_caching
        self.cache_ttl = cache_ttl
        self.retry_enabled = retry_enabled
        self.max_retry_attempts = max_retry_attempts
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay


class ToolService:
    """[CANONICAL] Service for managing tool operations.

    The target implementation for tool operations following the
    state-passed architectural pattern.

    This service follows SOLID principles:
    - SRP: Only handles tool operations
    - OCP: Extensible through strategy pattern
    - LSP: Implements ToolServiceProtocol
    - ISP: Focused interface
    - DIP: Depends on abstractions

    Example:
        config = ToolServiceConfig()
        service = ToolService(
            config=config,
            tool_selector=selector,
            tool_executor=executor,
            tool_registrar=registrar,
        )

        tools = await service.select_tools(context)
        result = await service.execute_tool("read", {"path": "file.txt"})
    """

    # Stable tool ordering is both a cache primitive and a model-salience prior.
    # Keep high-value coding-agent tools first; unknown tools fall back to schema
    # level and name for deterministic extension behavior.
    #
    # Tiered by usage frequency in a coding-agent session. Tier 0 is the core
    # read/edit loop that almost every turn touches; later tiers are progressively
    # less frequent. Ranking every session tool here (not just the core loop)
    # guarantees priority-based ordering instead of alphabetical fallback — see
    # ``sort_tools_for_kv_stability`` where unranked tools get ``unknown_rank``.
    STABLE_TOOL_ORDER = (
        # Tier 0 — core read/edit loop (highest frequency, exact order matters)
        "read",
        "edit",
        "write",
        "shell",
        "ls",
        "code_search",
        # Tier 1 — navigation / lookup
        "find",
        "refs",
        "symbol",
        "graph",
        "project_overview",
        # Tier 2 — version control + test workflow
        "git",
        "test",
        "pr",
        # Tier 3 — refactoring (single-file AST ops)
        "rename",
        "extract",
        "patch",
        "inline",
        "organize_imports",
        # Tier 4 — analysis / quality
        "metrics",
        "scan",
        "docs_coverage",
        "analysis_checkpoint",
        "lsp",
        # Tier 5 — web / external integrations
        "web_search",
        "web_fetch",
        "http",
        "jira",
        # Tier 6 — infra / data / generation (lower frequency)
        "docker",
        "cicd",
        "pipeline",
        "iac",
        "audit",
        "sandbox",
        "mcp",
        "db",
        "deps",
        "notebook_edit",
        "docs",
        "scaffold",
        "cache",
        "batch",
        "workflow",
        "merge",
    )
    STABLE_TOOL_ORDER_RANK = {name: rank for rank, name in enumerate(STABLE_TOOL_ORDER)}

    def __init__(
        self,
        config: ToolServiceConfig,
        tool_selector: Any,
        tool_executor: Any,
        tool_registrar: Any,
    ):
        """Initialize the tool service.

        Args:
            config: Service configuration
            tool_selector: Tool selection component
            tool_executor: Tool execution component
            tool_registrar: Tool registration component
        """
        self._config = config
        self._selector = tool_selector
        self._executor = tool_executor
        self._registrar = tool_registrar
        self._logger = logging.getLogger(f"{__name__}.{id(self)}")
        self._budget_manager = BudgetManager(config.default_tool_budget)
        self._budget_runtime = ToolBudgetRuntime(
            self._budget_manager,
            lambda: self._tool_pipeline,
        )
        self._usage_tracker = ToolUsageStats()
        self._tool_access_policy = ToolAccessPolicy(
            get_registrar=lambda: self._registrar,
            get_selector=lambda: self._selector,
            logger=self._logger,
        )
        self._mode_controller: Optional[Any] = None
        self._tool_planner: Optional[Any] = None
        self._argument_normalizer: Optional[Any] = None
        self._tool_pipeline: Optional[Any] = None
        self._tool_cache: Optional[Any] = None
        self._retry_executor: Optional[Any] = None
        from victor.tools.tool_call_parser import ToolCallParser
        from victor.tools.tool_call_validator import ToolCallValidator

        self._tool_call_parser: Optional[Any] = ToolCallParser()
        self._tool_call_validator: Optional[Any] = ToolCallValidator()
        self._last_tool_call_parse_diagnostics: Optional[Dict[str, Any]] = None

    def _hydrate_tools_for_context(self, context: Any) -> None:
        """Hydrate demand tools before selection when startup is lazy."""
        ensure_for_query = getattr(self._registrar, "ensure_tools_for_query", None)
        if not callable(ensure_for_query):
            return

        fragments: List[str] = []
        for attr in ("user_message", "message", "prompt", "query", "task_type"):
            try:
                value = getattr(context, attr, None)
            except Exception:
                value = None
            if isinstance(value, str) and value:
                fragments.append(value)

        metadata = getattr(context, "metadata", None)
        if isinstance(metadata, dict):
            for key in ("user_message", "message", "prompt", "query", "task_type"):
                value = metadata.get(key)
                if isinstance(value, str) and value:
                    fragments.append(value)

        text = "\n".join(fragments)
        if text:
            try:
                ensure_for_query(text)
            except Exception:
                self._logger.debug("Demand tool hydration failed", exc_info=True)

    def _hydrate_tool_if_missing(self, tool_name: str) -> None:
        """Hydrate one lazily registered tool if the registrar supports it."""
        ensure_tool = getattr(self._registrar, "ensure_tool_registered", None)
        if callable(ensure_tool):
            try:
                ensure_tool(tool_name)
            except Exception:
                self._logger.debug("Tool hydration failed for %s", tool_name, exc_info=True)

    @property
    def _enabled_tools(self) -> Optional[set[str]]:
        """Compatibility view over tool-access policy state."""
        return self._tool_access_policy.enabled_tools

    @_enabled_tools.setter
    def _enabled_tools(self, tools: Optional[set[str]]) -> None:
        self._tool_access_policy.enabled_tools = set(tools) if tools else None

    @property
    def _usage_stats(self) -> Dict[str, int]:
        """Compatibility view over usage-stat state."""
        if not hasattr(self, "_usage_tracker") or self._usage_tracker is None:
            self._usage_tracker = ToolUsageStats()
        return self._usage_tracker.counts

    @_usage_stats.setter
    def _usage_stats(self, value: Dict[str, int]) -> None:
        if not hasattr(self, "_usage_tracker") or self._usage_tracker is None:
            self._usage_tracker = ToolUsageStats()
        self._usage_tracker.counts = value

    @staticmethod
    def _count_tool_calls(tool_calls: Optional[Any]) -> int:
        """Count tool calls across list/dict/None inputs."""
        if isinstance(tool_calls, list):
            return len(tool_calls)
        if isinstance(tool_calls, dict):
            return 1
        return 0

    def _record_tool_call_parse_diagnostics(
        self,
        *,
        tool_adapter: Any,
        parse_method: str,
        adapter_parse_invoked: bool,
        raw_tool_call_count: int,
        extracted_tool_call_count: int,
        final_tool_call_count: int,
        confidence: float,
        warnings: Optional[List[str]] = None,
    ) -> None:
        """Persist and log tool-call parsing diagnostics for observability."""
        warning_list = list(warnings or [])
        provider_name = getattr(tool_adapter, "provider_name", None) or "unknown"
        model_name = getattr(tool_adapter, "model", None) or "unknown"
        diagnostics = {
            "provider_name": provider_name,
            "model_name": model_name,
            "parse_method": parse_method or "none",
            "adapter_parse_invoked": adapter_parse_invoked,
            "raw_tool_call_count": raw_tool_call_count,
            "extracted_tool_call_count": extracted_tool_call_count,
            "final_tool_call_count": final_tool_call_count,
            "confidence": confidence,
            "warning_count": len(warning_list),
            "warnings": warning_list,
            "fallback_used": parse_method not in {"none", "native", "native_passthrough"},
            "native_passthrough": parse_method == "native_passthrough",
        }
        self._last_tool_call_parse_diagnostics = diagnostics

        if final_tool_call_count > 0 or raw_tool_call_count > 0 or warning_list:
            self._logger.info(
                "tool_call_parse provider=%s model=%s parse_method=%s "
                "adapter_parse_invoked=%s raw_tool_call_count=%d "
                "extracted_tool_call_count=%d final_tool_call_count=%d "
                "warning_count=%d fallback_used=%s",
                provider_name,
                model_name,
                diagnostics["parse_method"],
                adapter_parse_invoked,
                raw_tool_call_count,
                extracted_tool_call_count,
                final_tool_call_count,
                diagnostics["warning_count"],
                diagnostics["fallback_used"],
            )
        else:
            self._logger.debug(
                "tool_call_parse provider=%s model=%s parse_method=%s raw_tool_call_count=%d",
                provider_name,
                model_name,
                diagnostics["parse_method"],
                raw_tool_call_count,
            )

    def get_last_tool_call_parse_diagnostics(self) -> Optional[Dict[str, Any]]:
        """Return the most recent tool-call parse diagnostics snapshot."""
        if self._last_tool_call_parse_diagnostics is None:
            return None
        return dict(self._last_tool_call_parse_diagnostics)

    def _get_budget_limit(self) -> int:
        """Return the active tool budget ceiling."""
        return self._budget_runtime.get_limit()

    def _get_budget_used(self) -> int:
        """Return the active count of tool calls already spent this turn."""
        return self._budget_runtime.get_used()

    def _sync_budget_manager_from_runtime(self) -> None:
        """Mirror the active runtime budget into the local compatibility manager."""
        self._budget_runtime.sync_from_runtime()

    def _set_budget_limit(self, budget: int) -> None:
        """Set the active tool budget ceiling across bound runtime owners."""
        self._budget_runtime.set_limit(budget)

    def _consume_budget(self, amount: int = 1) -> None:
        """Record tool usage against the active runtime budget."""
        self._budget_runtime.consume(amount)

    async def select_tools(
        self,
        context: "ToolSelectionContext",
        max_tools: int = 10,
    ) -> List[str]:
        """Select tools based on context.

        Uses the tool selector to analyze the context and select
        the most relevant tools for the task.

        Args:
            context: Tool selection context
            max_tools: Maximum number of tools to select

        Returns:
            List of selected tool names, ordered by relevance

        Raises:
            ToolSelectionError: If tool selection fails
        """
        self._logger.debug(f"Selecting tools (max={max_tools})")

        try:
            self._hydrate_tools_for_context(context)
            # Use selector to choose tools
            selected = await self._selector.select(context, max_tools)

            self._logger.debug(f"Selected {len(selected)} tools: {selected}")
            return selected

        except Exception as e:
            self._logger.error(f"Tool selection failed: {str(e)[:500]}")
            # Return empty list on failure for resilience
            return []

    async def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Any:
        """Execute a single tool with validation and budgeting.

        Validates arguments, checks budget, executes the tool,
        and tracks usage.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments

        Returns:
            ToolResult with execution outcome

        Raises:
            ToolBudgetExceededError: If tool budget is exhausted
            ToolNotFoundError: If tool is not registered
            ToolValidationError: If arguments are invalid
            ToolExecutionError: If tool execution fails
        """
        self._logger.debug(f"Executing tool: {tool_name}")
        self._hydrate_tool_if_missing(tool_name)

        # Check budget
        if self.is_budget_exhausted():
            self._logger.warning("Tool budget exhausted")
            raise ToolBudgetExceededError(
                f"Tool budget exhausted ({self.budget_used} / {self.budget})"
            )

        # Check cache if enabled
        if self._config.enable_caching:
            cached = await self._get_cached_result(tool_name, arguments)
            if cached is not None:
                self._logger.debug(f"Using cached result for {tool_name}")
                return cached

        # Execute tool
        try:
            result = await self._executor.execute(tool_name, arguments)

            # Track usage
            self._consume_budget()
            self._track_tool_usage(tool_name, success=True)

            # Cache result if enabled and successful
            if self._config.enable_caching and result.success:
                await self._cache_result(tool_name, arguments, result)

            return result

        except Exception as e:
            self._logger.error(f"Tool execution failed: {tool_name}: {str(e)[:500]}")
            self._track_tool_usage(tool_name, success=False)
            raise

    async def execute_tools_parallel(
        self,
        tool_calls: List[tuple[str, Dict[str, Any]]],
        max_parallel: int = 5,
    ) -> AsyncIterator[Any]:
        """Execute multiple tools in parallel.

        Executes independent tools concurrently for improved performance.

        Args:
            tool_calls: List of (tool_name, arguments) tuples
            max_parallel: Maximum number of concurrent executions

        Yields:
            ToolResult objects as they complete

        Raises:
            ToolExecutionError: If any critical tool execution fails
        """
        if not self._config.enable_parallel_execution:
            # Execute sequentially
            for tool_name, arguments in tool_calls:
                yield await self.execute_tool(tool_name, arguments)
            return

        self._logger.debug(f"Executing {len(tool_calls)} tools in parallel")

        # Create semaphore for parallelism limit
        semaphore = asyncio.Semaphore(max_parallel)

        async def execute_with_limit(tool_name: str, arguments: Dict[str, Any]):
            async with semaphore:
                return await self.execute_tool(tool_name, arguments)

        # Execute all tools in parallel
        tasks = [execute_with_limit(name, args) for name, args in tool_calls]

        for task in asyncio.as_completed(tasks):
            result = await task
            yield result

    def get_tool_budget(self) -> int:
        """Get the remaining executable tool budget for the current turn.

        Returns:
            Number of remaining tool calls allowed
        """
        return self.get_remaining_budget()

    def set_tool_budget(self, budget: int) -> None:
        """Set the tool budget limit.

        Args:
            budget: Maximum number of tool calls allowed

        Raises:
            ValueError: If budget is negative
        """
        old_max = self.budget
        self._set_budget_limit(budget)
        self._logger.info(f"Tool budget updated: {old_max} -> {budget}")

    def get_tool_usage_stats(self) -> Dict[str, Any]:
        """Get tool usage statistics.

        Returns:
            Dictionary with usage statistics
        """
        return self._usage_tracker.snapshot(
            budget_remaining=self.get_tool_budget(),
            budget_used=self.budget_used,
        )

    def start_new_turn(self) -> None:
        """Reset per-turn budget while preserving cumulative usage stats.

        Interactive chat sessions should replenish executable tool budget for
        each new user prompt, but cross-turn analytics should remain intact.
        """
        self._budget_runtime.start_new_turn()
        self._logger.debug("Tool turn budget reset")

    def reset_tool_budget(self) -> None:
        """Fully reset tool budget and usage stats to initial state.

        Useful for starting a fresh session, test isolation, or explicit
        operator-driven recovery where cumulative usage analytics should be
        cleared along with the budget.
        """
        self.start_new_turn()
        self._usage_tracker.clear()
        self._logger.info("Tool budget reset")

    def process_tool_results(
        self,
        pipeline_result: Any,
        ctx: ToolResultContext,
    ) -> List[Dict[str, Any]]:
        """Post-process tool pipeline results using the canonical service path."""
        return process_tool_results_with_context(pipeline_result, ctx)

    def bind_runtime_components(
        self,
        *,
        tool_registry: Optional[Any] = None,
        tool_pipeline: Optional[Any] = None,
        tool_cache: Optional[Any] = None,
        mode_controller: Optional[Any] = None,
        tool_planner: Optional[Any] = None,
        argument_normalizer: Optional[Any] = None,
        retry_executor: Optional[Any] = None,
        tool_call_parser: Optional[Any] = None,
    ) -> None:
        """Bind live runtime collaborators after bootstrap.

        The service bootstrap path creates the canonical ToolService before
        the full orchestrator runtime is assembled. This hook upgrades the
        service with the real runtime collaborators once they exist.
        """
        if tool_registry is not None:
            self._registrar = tool_registry
        if tool_pipeline is not None:
            self._tool_pipeline = tool_pipeline
            self._sync_budget_manager_from_runtime()
        if tool_cache is not None:
            self._tool_cache = tool_cache
        if mode_controller is not None:
            self._mode_controller = mode_controller
        if tool_planner is not None:
            self._tool_planner = tool_planner
        if argument_normalizer is not None:
            self._argument_normalizer = argument_normalizer
        if retry_executor is not None:
            self._retry_executor = retry_executor
        elif self._tool_pipeline is not None and self._retry_executor is None:
            from victor.agent.services.tool_retry import ToolRetryExecutor

            self._retry_executor = ToolRetryExecutor(
                self._config,
                self._tool_pipeline,
                self._tool_cache,
            )
        if tool_call_parser is not None:
            self._tool_call_parser = tool_call_parser

    # ==========================================================================
    # Budget Properties
    # ==========================================================================

    @property
    def budget(self) -> int:
        """Get the current tool budget limit.

        Returns:
            Maximum number of tool calls allowed
        """
        return self._get_budget_limit()

    @budget.setter
    def budget(self, value: int) -> None:
        """Set the tool budget limit.

        Args:
            value: Maximum number of tool calls allowed
        """
        self.set_tool_budget(value)

    @property
    def budget_used(self) -> int:
        """Get the number of tool calls used.

        Returns:
            Number of tool calls made so far
        """
        return self._get_budget_used()

    @property
    def execution_count(self) -> int:
        """Get the number of tool executions.

        Returns:
            Number of tool calls executed
        """
        return self.budget_used

    def set_budget_multiplier(self, multiplier: float) -> None:
        """Set a budget multiplier for scaling the budget.

        Adjusts the budget limit by multiplying the current budget
        by the given multiplier. Useful for scaling operations up or down.

        Args:
            multiplier: Multiplier to apply to current budget
                        (e.g., 2.0 doubles the budget)

        Raises:
            ValueError: If multiplier is negative

        Example:
            service.set_budget_multiplier(2.0)  # Double the budget
            service.set_budget_multiplier(0.5)  # Halve the budget
        """
        if multiplier < 0:
            raise ValueError(f"Budget multiplier must be non-negative: {multiplier}")

        current_budget = self.budget
        new_budget = int(current_budget * multiplier)
        self.budget = new_budget
        self._logger.info(f"Budget multiplied by {multiplier}: {current_budget} -> {new_budget}")

    def consume_budget(self, amount: int = 1) -> None:
        """Consume budget for tool calls.

        Records the specified number of tool calls against the budget.

        Args:
            amount: Number of tool calls to record (default: 1)

        Raises:
            BudgetExhaustedError: If budget is insufficient

        Example:
            # Record 3 tool calls
            service.consume_budget(3)

            # Check if budget exhausted
            if service.is_budget_exhausted():
                # Handle budget exhaustion
                pass
        """
        if amount < 0:
            raise ValueError(f"Cannot consume negative budget: {amount}")

        remaining = self.get_remaining_budget()
        if amount > remaining:
            from victor.core.errors import BudgetExhaustedError

            raise BudgetExhaustedError(
                f"Insufficient budget: need {amount}, have {remaining}",
                budget=self.budget,
                used=self.budget_used,
            )

        self._consume_budget(amount)

    def on_tool_complete(
        self,
        result: Any,
        metrics_collector: Optional[Any] = None,
        *,
        read_files_session: Optional[Set[str]] = None,
        required_files: Optional[List[str]] = None,
        required_outputs: Optional[List[str]] = None,
        nudge_sent_flag: Optional[List[bool]] = None,
        add_message: Optional[Callable[[str, str], None]] = None,
        observability: Optional[Any] = None,
        iteration_count: int = 0,
        tool_name: Optional[str] = None,
        elapsed: float = 0.0,
        session_id: Optional[str] = None,
    ) -> None:
        """Handle tool completion through the canonical service path.

        When ``metrics_collector`` is provided this uses the existing shared
        ``ToolObservabilityHandler`` implementation, which preserves file-read
        nudges, tool completion events, and UI preview behavior. When only the
        lightweight hook arguments are provided, it degrades to the narrower
        observability callback for backward compatibility.
        """
        if metrics_collector is not None:
            from victor.agent.services.tool_observability import (
                ToolObservabilityHandler,
            )

            ToolObservabilityHandler(self).on_tool_complete(
                result=result,
                metrics_collector=metrics_collector,
                read_files_session=read_files_session,
                required_files=required_files,
                required_outputs=required_outputs,
                nudge_sent_flag=nudge_sent_flag,
                add_message=add_message,
                observability=observability,
                iteration_count=iteration_count,
            )
            return

        effective_observability = observability or getattr(self, "_observability", None)
        effective_tool_name = tool_name or getattr(result, "tool_name", "")
        if effective_observability is not None and hasattr(
            effective_observability,
            "on_tool_complete",
        ):
            try:
                effective_observability.on_tool_complete(
                    effective_tool_name,
                    result,
                    elapsed,
                    session_id,
                )
            except Exception:
                logger.debug("on_tool_complete observability hook failed", exc_info=True)

    def is_healthy(self) -> bool:
        """Check if the tool service is healthy.

        Returns:
            True if the service is healthy
        """
        return (
            self._selector is not None
            and self._executor is not None
            and not self.is_budget_exhausted()
        )

    # ==========================================================================
    # Tool Enable/Disable Management
    # ==========================================================================

    def set_enabled_tools(self, tools: set[str]) -> None:
        """Set which tools are enabled for this session.

        When tools are explicitly set, only those tools can be used.
        Setting to None or empty set enables all available tools.

        Args:
            tools: Set of tool names to enable, or None for all tools

        Example:
            service.set_enabled_tools({"read", "write", "search"})
            # Only these tools can be used

            service.set_enabled_tools(set())
            # All tools enabled (default)
        """
        self._tool_access_policy.set_enabled_tools(tools)

    def get_enabled_tools(self) -> set[str]:
        """Get currently enabled tool names.

        Returns:
            Set of enabled tool names. If None or empty set returned,
            all tools are considered enabled.

        Example:
            enabled = service.get_enabled_tools()
            if "shell" in enabled:
                # Shell tool is available
        """
        if self._enabled_tools is not None:
            return self._enabled_tools.copy()
        return self.get_available_tools()

    def is_tool_enabled(self, tool_name: str) -> bool:
        """Check if a specific tool is enabled.

        Args:
            tool_name: Name of the tool to check

        Returns:
            True if the tool is enabled, False otherwise

        Example:
            if service.is_tool_enabled("shell"):
                # Shell tool can be used
        """
        return self._tool_access_policy.is_tool_enabled(tool_name)

    # ==========================================================================
    # Tool Alias Resolution
    # ==========================================================================

    def resolve_tool_alias(self, tool_name: str) -> str:
        """Resolve tool alias to canonical name.

        Handles shell variants and other tool aliases by:
        1. Converting to canonical name using tool_names registry
        2. Checking if shell is enabled
        3. Returning the canonical name for non-shell tools

        Args:
            tool_name: Original tool name (may be alias)

        Returns:
            Canonical tool name that should be used

        Example:
            canonical = service.resolve_tool_alias("bash")
            # Returns "shell" if enabled

            canonical = service.resolve_tool_alias("execute_bash")
            # Returns "shell" if enabled
        """
        return self._tool_access_policy.resolve_tool_alias(tool_name)

    # ==========================================================================
    # Tool Statistics and Budget Queries
    # ==========================================================================

    def get_tool_call_count(self, tool_name: str) -> int:
        """Get the number of times a specific tool was called.

        Args:
            tool_name: Name of the tool

        Returns:
            Number of successful calls (errors counted separately)

        Example:
            count = service.get_tool_call_count("read")
            # Returns: 10
        """
        return self._usage_tracker.get_tool_call_count(tool_name)

    def get_tool_error_count(self, tool_name: str) -> int:
        """Get the number of errors for a specific tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Number of error occurrences

        Example:
            errors = service.get_tool_error_count("shell")
            # Returns: 2
        """
        return self._usage_tracker.get_tool_error_count(tool_name)

    def get_remaining_budget(self) -> int:
        """Get remaining tool budget.

        Returns the number of tool calls remaining before budget exhaustion.

        Returns:
            Remaining budget count

        Example:
            remaining = service.get_remaining_budget()
            # Returns: 85 (out of 100)
        """
        return self._budget_runtime.get_remaining()

    def is_budget_exhausted(self) -> bool:
        """Check if tool budget is exhausted.

        Returns:
            True if no more tool calls allowed, False otherwise

        Example:
            if service.is_budget_exhausted():
                # Stop tool execution
        """
        return self._budget_runtime.is_exhausted()

    def get_budget_info(self) -> Dict[str, int]:
        """Get detailed budget information.

        Returns:
            Dictionary with budget details (max, used, remaining)

        Example:
            info = service.get_budget_info()
            # {"max": 100, "used": 15, "remaining": 85}
        """
        return self._budget_runtime.get_info()

    def get_available_tools(self) -> set[str]:
        """Get all available tools from the registrar.

        Returns a set of tool names that are registered and available.
        Returns empty set if registrar not available.

        Returns:
            Set of available tool names

        Example:
            tools = service.get_available_tools()
            # {"read", "write", "search", "shell", ...}
        """
        return self._tool_access_policy.get_available_tools()

    def build_tool_access_context(self) -> Any:
        """Build the canonical access-control context for tool gating."""
        from victor.agent.protocols import ToolAccessContext

        return ToolAccessContext(
            session_enabled_tools=self._enabled_tools,
            current_mode=(self._mode_controller.config.name if self._mode_controller else None),
        )

    def validate_tool_call(
        self,
        tool_call: Any,
        sanitizer: Any,
        is_tool_enabled_fn: Optional[Callable[[str], bool]] = None,
    ) -> Any:
        """Validate a tool call's structure, name, and enabled status."""
        from victor.agent.services.tool_contracts import ToolCallValidation

        _is_enabled = is_tool_enabled_fn or self.is_tool_enabled
        if not isinstance(tool_call, dict):
            return ToolCallValidation(
                valid=False,
                skip_reason="Skipping malformed tool call payload",
            )

        tool_name = normalize_model_tool_name(str(tool_call.get("name", "")).strip())
        if not tool_name:
            return ToolCallValidation(
                valid=False,
                skip_reason="Skipping tool call with no name",
            )

        canonical = self.resolve_tool_alias(tool_name)
        available_tools = self.get_available_tools()
        enabled = _is_enabled(canonical)

        if not sanitizer.is_valid_tool_name(tool_name) and not (
            enabled and canonical in available_tools
        ):
            return ToolCallValidation(
                valid=False,
                original_name=tool_name,
                skip_reason=f"Skipping invalid/hallucinated tool name: {tool_name}",
                error_result={
                    "tool_name": tool_name,
                    "success": False,
                    "result": None,
                    "error": (
                        f"Invalid tool name '{tool_name}'. This tool does not exist. "
                        "Use only tools from the provided tool list. "
                        "Check for typos or hallucinated tool names."
                    ),
                },
            )

        if not enabled:
            return ToolCallValidation(
                valid=False,
                original_name=tool_name,
                canonical_name=canonical,
                skip_reason=(
                    f"Skipping unknown or disabled tool: {tool_name} (resolved: {canonical})"
                ),
                error_result={
                    "tool_name": tool_name,
                    "success": False,
                    "result": None,
                    "error": (
                        f"Tool '{tool_name}' is not available. It may be disabled, not "
                        "registered, or not included in the current tool selection. "
                        "Use only the tools listed in your available tools."
                    ),
                },
            )

        validator = self._tool_call_validator
        if validator is not None and canonical in validator._tool_schemas:
            schema_result = validator.validate(canonical, tool_call.get("arguments", {}))
            if not schema_result.valid:
                self._logger.warning("Schema validation: %s: %s", canonical, schema_result.errors)

        return ToolCallValidation(
            valid=True,
            original_name=tool_name,
            canonical_name=canonical,
        )

    def normalize_arguments_full(
        self,
        tool_name: str,
        original_name: str,
        raw_args: Any,
        argument_normalizer: Any,
        tool_adapter: Any,
        failed_signatures: Optional[Set[Tuple[str, str]]] = None,
    ) -> Any:
        """Normalize tool arguments through parsing, repair, adapter defaults, and dedup."""
        from victor.agent.services.tool_contracts import NormalizedArgs
        from victor.agent.orchestrator_utils import infer_git_operation

        _failed = failed_signatures if failed_signatures is not None else set()

        # Coerce via the single shared ladder (json/ast/native-repair/{value:...}),
        # then delegate to the INJECTED normalizer's normalize_arguments — preserving
        # the dependency-injection seam (callers may pass a custom/mock normalizer).
        from victor.agent.argument_normalizer import ArgumentNormalizer

        if isinstance(raw_args, str):
            tool_args = ArgumentNormalizer.coerce_arg_string(raw_args)
        elif raw_args is None:
            tool_args = {}
        else:
            tool_args = raw_args
        normalized_args, strategy = argument_normalizer.normalize_arguments(tool_args, tool_name)
        normalized_args = tool_adapter.normalize_arguments(normalized_args, tool_name)
        normalized_args = infer_git_operation(original_name, tool_name, normalized_args)

        try:
            signature = (
                tool_name,
                json_dumps(normalized_args, sort_keys=True, default=str),
            )
        except Exception:
            signature = (tool_name, str(normalized_args))

        return NormalizedArgs(
            args=normalized_args,
            strategy=strategy,
            signature=signature,
            is_repeated_failure=signature in _failed,
        )

    # ==========================================================================
    # Tool Call Validation
    # ==========================================================================

    def _validate_tool_call(
        self,
        tool_call: Dict[str, Any],
        available_tools: Optional[set[str]] = None,
    ) -> tuple[bool, Optional[str]]:
        """Internal: Validate a single tool call.

        Checks if the tool call is valid by verifying:
        - Tool name is present
        - Tool is available/enabled
        - Required fields are present

        Args:
            tool_call: Tool call dictionary with 'name' and 'arguments'
            available_tools: Set of available tool names (uses get_available_tools if None)

        Returns:
            Tuple of (is_valid, error_message)
        """

        def _normalize_tools(tool_names: set[str]) -> set[str]:
            return {canonicalize_core_tool_name(name) for name in tool_names}

        # Check required fields
        if not isinstance(tool_call, dict):
            return False, "Tool call must be a dictionary"

        tool_name = normalize_model_tool_name(tool_call.get("name") or tool_call.get("tool"))
        if not tool_name:
            return False, "Tool call missing 'name' field"

        canonical_name = canonicalize_core_tool_name(self.resolve_tool_alias(str(tool_name)))
        self._hydrate_tool_if_missing(canonical_name)

        # Check if tool is available
        if available_tools is None:
            available_tools = self.get_enabled_tools()

        normalized_available = _normalize_tools(available_tools)

        if canonical_name not in normalized_available:
            registered_tools = _normalize_tools(self.get_available_tools())
            available_preview = ", ".join(sorted(normalized_available)[:8]) or "(none)"

            if canonical_name not in registered_tools:
                return (
                    False,
                    f"Tool '{tool_name}' is not available because it is not registered. "
                    f"Available tools: {available_preview}",
                )

            if self._mode_controller is not None:
                mode_name = self._mode_controller.current_mode.value.upper()
                sandbox_dir = getattr(self._mode_controller.config, "sandbox_dir", None)
                if (
                    canonical_name in {"edit", "write"}
                    and not getattr(self._mode_controller.config, "allow_all_tools", False)
                    and getattr(self._mode_controller.config, "allow_sandbox_edits", False)
                    and sandbox_dir
                ):
                    return (
                        False,
                        f"Tool '{tool_name}' is not available because it resolves to "
                        f"'{canonical_name}', which is limited to {sandbox_dir} in {mode_name} "
                        f"mode. Available tools: {available_preview}",
                    )
                if not self._mode_controller.is_tool_allowed(canonical_name):
                    return (
                        False,
                        f"Tool '{tool_name}' is not available because it resolves to "
                        f"'{canonical_name}', which is not allowed in {mode_name} mode. "
                        f"Available tools: {available_preview}",
                    )

            if self._enabled_tools is not None and canonical_name not in _normalize_tools(
                self._enabled_tools
            ):
                return (
                    False,
                    f"Tool '{tool_name}' is not available because it resolves to "
                    f"'{canonical_name}', which is not enabled for this session. "
                    f"Available tools: {available_preview}",
                )

            return (
                False,
                f"Tool '{tool_name}' is not available because it resolves to "
                f"'{canonical_name}', which is not enabled in the current runtime state. "
                f"Available tools: {available_preview}",
            )

        # Check arguments field
        arguments = tool_call.get("arguments", {})
        if not isinstance(arguments, dict):
            return False, "Tool arguments must be a dictionary"

        return True, None

    def validate_tool_calls(
        self,
        tool_calls: Union[Dict[str, Any], List[Dict[str, Any]]],
        available_tools: Optional[set[str]] = None,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Validate tool calls (single or multiple).

        This is the canonical method for tool validation. It automatically handles
        both single tool calls and batches of tool calls.

        Splits tool calls into valid and invalid groups. Invalid calls include
        a '_validation_error' field with the error message.

        Args:
            tool_calls: A single tool call dict or a list of tool call dicts.
                Each dict should have 'name' and 'arguments' fields.
            available_tools: Set of available tool names (uses get_available_tools if None)

        Returns:
            Tuple of (valid_calls, invalid_calls). Both are lists, even for single input.

        Examples:
            # Validate a single tool call
            valid, invalid = service.validate_tool_calls(
                {"name": "code_search", "arguments": {"query": "foo"}}
            )
            # valid = [{"name": "code_search", ...}]
            # invalid = []

            # Validate multiple tool calls
            valid, invalid = service.validate_tool_calls([
                {"name": "code_search", "arguments": {"query": "foo"}},
                {"name": "invalid_tool", "arguments": {}}
            ])
            # valid = [{"name": "code_search", ...}]
            # invalid = [{"name": "invalid_tool", ..., "_validation_error": "..."}]
        """
        # Normalize single tool call to list
        if isinstance(tool_calls, dict):
            tool_calls = [tool_calls]
        elif not isinstance(tool_calls, list):
            return [], [{"_validation_error": "tool_calls must be a dict or list"}]

        valid_calls = []
        invalid_calls = []

        for call in tool_calls:
            is_valid, error = self._validate_tool_call(call, available_tools)
            if is_valid:
                valid_calls.append(call)
            else:
                # Add error message to invalid call
                invalid_call = dict(call)
                invalid_call["_validation_error"] = error
                invalid_calls.append(invalid_call)

        return valid_calls, invalid_calls

    def filter_hallucinated_tools(
        self,
        tool_calls: List[Dict[str, Any]],
        available_tools: Optional[set[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Filter out hallucinated (non-existent) tool calls.

        Removes tool calls for tools that don't exist in the available tool set.
        This is useful when LLMs invent tool names.

        Args:
            tool_calls: List of tool call dictionaries
            available_tools: Set of available tool names (uses get_available_tools if None)

        Returns:
            List of valid tool calls (hallucinated ones removed)

        Example:
            filtered = service.filter_hallucinated_tools(tool_calls)
            print(f"Filtered from {len(tool_calls)} to {len(filtered)} calls")
        """
        if available_tools is None:
            available_tools = self.get_available_tools()

        valid_calls = []
        for call in tool_calls:
            tool_name = call.get("name") or call.get("tool")
            if tool_name and tool_name in available_tools:
                valid_calls.append(call)
            else:
                self._logger.debug(f"Filtered hallucinated tool: {tool_name}")

        return valid_calls

    def check_tool_arguments(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        schema: Optional[Dict[str, Any]] = None,
    ) -> tuple[bool, Optional[str]]:
        """Check if tool arguments are valid.

        Performs basic validation of tool arguments:
        - Checks for None values
        - Checks for empty strings
        - Validates argument types if schema available

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments dictionary

        Returns:
            Tuple of (is_valid, error_message)

        Example:
            is_valid, error = service.check_tool_arguments(
                "code_search", {"query": "test"}
            )
        """
        if not isinstance(arguments, dict):
            return False, "Arguments must be a dictionary"

        # Check for None values in arguments
        for key, value in arguments.items():
            if value is None:
                return False, f"Argument '{key}' is None"

            # Check for empty strings (often indicates missing required param)
            if isinstance(value, str) and not value.strip():
                return False, f"Argument '{key}' is empty"

        # Schema-aware validation can be layered in later. For now we keep
        # validation intentionally conservative to avoid rejecting valid calls.
        _ = tool_name
        _ = schema

        return True, None

    def get_tool_schema(
        self,
        tool_name: str,
    ) -> Optional[Dict[str, Any]]:
        """Get the schema for a tool.

        Returns the JSON schema for a tool's arguments if available.
        Useful for validation and documentation.

        Args:
            tool_name: Name of the tool

        Returns:
            Tool schema dictionary or None if not available

        Example:
            schema = service.get_tool_schema("code_search")
            if schema:
                print(f"Required: {schema.get('required', [])}")
        """
        if not self._registrar:
            return None

        try:
            self._hydrate_tool_if_missing(tool_name)
            tool = self._registrar.get_tool(tool_name)
            if tool and hasattr(tool, "get_schema"):
                return tool.get_schema()
        except Exception as e:
            self._logger.warning(f"Failed to get schema for {tool_name}: {str(e)[:500]}")

        return None

    def parse_tool_calls(
        self,
        response_content: str,
        tool_call_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Parse tool calls from response content.

        Extracts tool calls from LLM response text, handling various
        formats (XML tags, JSON blocks, etc.).

        Args:
            response_content: Response text containing tool calls
            tool_call_id: Optional tool call ID for association

        Returns:
            List of parsed tool call dictionaries

        Example:
            tool_calls = service.parse_tool_calls(
                '<function_calls>...</function_calls>'
            )
        """
        import re
        import json

        tool_calls = []

        # Try XML format: <function_calls><invoke name="tool">...</invoke></function_calls>
        xml_pattern = r'<invoke\s+name="([^"]+)">(.*?)</invoke>'
        for match in re.finditer(xml_pattern, response_content, re.DOTALL):
            tool_name = match.group(1)
            params_xml = match.group(2)

            # Extract parameters
            params = {}
            param_pattern = r'<parameter\s+name="([^"]+)">(.*?)</parameter>'
            for param_match in re.finditer(param_pattern, params_xml, re.DOTALL):
                param_name = param_match.group(1)
                param_value = param_match.group(2).strip()
                params[param_name] = param_value

            tool_call = {"name": tool_name, "arguments": params}
            if tool_call_id:
                tool_call["id"] = tool_call_id
            tool_calls.append(tool_call)

        # Try JSON format: [{"name": "tool", "arguments": {...}}]
        if not tool_calls:
            try:
                json_match = re.search(r"\[.*\]", response_content, re.DOTALL)
                if json_match:
                    json_data = json_loads(json_match.group(0))
                    if isinstance(json_data, list):
                        for item in json_data:
                            if isinstance(item, dict) and "name" in item:
                                tool_call = {
                                    "name": item.get("name"),
                                    "arguments": item.get("arguments", item.get("parameters", {})),
                                }
                                if tool_call_id:
                                    tool_call["id"] = tool_call_id
                                tool_calls.append(tool_call)
            except ValueError:
                pass

        return tool_calls

    # Harmony/chat-template role and channel tokens that some providers (notably
    # gpt-oss via Ollama) leak as a tool-call name when the response format is not
    # parsed correctly. These are never real tool names.
    _CHANNEL_ARTIFACT_NAMES = frozenset(
        {"assistant", "user", "system", "developer", "tool", "analysis", "commentary", "final"}
    )

    def _all_tool_calls_are_channel_artifacts(
        self, tool_calls: Optional[List[Dict[str, Any]]]
    ) -> bool:
        """True if every tool call's name is a harmony/role channel artifact."""
        names = [
            str(tc.get("name", "")).strip().lower()
            for tc in (tool_calls or [])
            if isinstance(tc, dict)
        ]
        return bool(names) and all(name in self._CHANNEL_ARTIFACT_NAMES for name in names)

    def parse_and_validate_tool_calls(
        self,
        tool_calls: Optional[List[Dict[str, Any]]],
        full_content: str,
        tool_adapter: Any,
    ) -> Tuple[Optional[List[Dict[str, Any]]], str]:
        """Parse, validate, normalize, and filter tool calls from provider output."""
        raw_tool_call_count = self._count_tool_calls(tool_calls)
        parse_method = "native_passthrough" if raw_tool_call_count else "none"
        adapter_parse_invoked = False
        extracted_tool_call_count = raw_tool_call_count
        parse_confidence = 1.0
        parse_warnings: List[str] = []

        # gpt-oss / harmony artifact: a provider can surface the channel role token
        # ("assistant") as the function name instead of the real tool. Such names are
        # never valid tools — drop them so content fallback parsing can recover the
        # actual call below, instead of passing a bogus call straight through.
        if tool_calls and self._all_tool_calls_are_channel_artifacts(tool_calls):
            artifact_names = [tc.get("name") for tc in tool_calls if isinstance(tc, dict)]
            self._logger.warning(
                "Discarding %d native tool call(s) with channel-artifact name(s) %s; "
                "falling back to content parsing",
                raw_tool_call_count,
                artifact_names,
            )
            tool_calls = None
            raw_tool_call_count = 0
            parse_method = "none"
            extracted_tool_call_count = 0

        if not tool_calls and full_content:
            self._logger.debug(
                "No native tool_calls, attempting fallback parsing on content len=%s",
                len(full_content),
            )
            adapter_parse_invoked = True
            parse_result = tool_adapter.parse_tool_calls(full_content, tool_calls)
            parse_method = getattr(parse_result, "parse_method", "none") or "none"
            parse_confidence = getattr(parse_result, "confidence", 1.0)
            parse_warnings = list(getattr(parse_result, "warnings", []) or [])
            extracted_tool_call_count = self._count_tool_calls(parse_result.tool_calls)
            for warning in parse_result.warnings:
                self._logger.warning(f"Tool call parse warning: {warning}")
            if parse_result.tool_calls:
                tool_calls = [tc.to_dict() for tc in parse_result.tool_calls]
                self._logger.debug(
                    "Fallback parser found %s tool calls: %s",
                    len(tool_calls),
                    [tc.get("name") for tc in tool_calls],
                )
                full_content = parse_result.remaining_content
            else:
                self._logger.debug("Fallback parser found no tool calls")

        if tool_calls:
            normalized_tool_calls = [tc for tc in tool_calls if isinstance(tc, dict)]
            if len(normalized_tool_calls) != len(tool_calls):
                self._logger.warning(f"Dropped non-dict tool_calls: {tool_calls}")
            tool_calls = normalized_tool_calls or None
            self._logger.debug("After normalization: %s tool_calls", len(tool_calls or []))

        if tool_calls:
            valid_tool_calls = []
            invalid_count = 0
            for tc in tool_calls:
                original_name = tc.get("name", "")
                name = normalize_model_tool_name(original_name)
                if name != original_name:
                    tc["name"] = name
                resolved_name = self.resolve_tool_alias(name)
                if resolved_name != name:
                    tc["name"] = resolved_name
                    name = resolved_name
                is_enabled = self.is_tool_enabled(name)
                self._logger.debug("Tool '%s' enabled=%s", name, is_enabled)
                if is_enabled:
                    valid_tool_calls.append(tc)
                else:
                    invalid_count += 1
                    tc["_invalid"] = True
                    tc["_error"] = f"Unknown tool '{name}'. Use one of the available tools."
                    valid_tool_calls.append(tc)
                    self._logger.debug("Marked invalid tool for error response: %s", name)
            if invalid_count:
                self._logger.warning(
                    "Marked %s invalid tool calls for error responses", invalid_count
                )
            tool_calls = valid_tool_calls or None
            self._logger.debug("After filtering: %s valid tool_calls", len(tool_calls or []))

        parser = self._tool_call_parser
        if parser is None:
            from victor.tools.tool_call_parser import ToolCallParser

            parser = ToolCallParser()
            self._tool_call_parser = parser

        if tool_calls:
            from victor.agent.argument_normalizer import ArgumentNormalizer

            for tc in tool_calls:
                ArgumentNormalizer.normalize_arguments_shape(tc)
                parsed_args = tc.get("arguments")
                if isinstance(parsed_args, dict):
                    tc["arguments"] = parser.normalize_args(tc.get("name", ""), parsed_args)

        self._record_tool_call_parse_diagnostics(
            tool_adapter=tool_adapter,
            parse_method=parse_method,
            adapter_parse_invoked=adapter_parse_invoked,
            raw_tool_call_count=raw_tool_call_count,
            extracted_tool_call_count=extracted_tool_call_count,
            final_tool_call_count=self._count_tool_calls(tool_calls),
            confidence=parse_confidence,
            warnings=parse_warnings,
        )
        return tool_calls, full_content

    def normalize_tool_call(
        self,
        tool_call: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Normalize a tool call to have consistent structure.

        Ensures tool call has consistent structure:
        - Has 'name' field (not 'tool')
        - Has 'arguments' field (not 'parameters')
        - Removes None values from arguments

        Args:
            tool_call: Raw tool call dictionary

        Returns:
            Normalized tool call dictionary

        Example:
            normalized = service.normalize_tool_call({
                "tool": "code_search",
                "parameters": {"query": "test"}
            })
            # {"name": "code_search", "arguments": {"query": "test"}}
        """
        normalized = {}

        # Normalize tool name field
        tool_name = tool_call.get("name") or tool_call.get("tool")
        if tool_name:
            normalized["name"] = tool_name

        # Normalize arguments field
        arguments = tool_call.get("arguments") or tool_call.get("parameters", {})
        if isinstance(arguments, dict):
            # Remove None values
            normalized["arguments"] = {k: v for k, v in arguments.items() if v is not None}

        # Preserve any other fields
        for key, value in tool_call.items():
            if key not in ("name", "tool", "arguments", "parameters"):
                normalized[key] = value

        return normalized

    # ==========================================================================
    # Mode Controller and Tool Planner
    # ==========================================================================

    def set_mode_controller(self, mode_controller: Any) -> None:
        """Set the mode controller for tool access control.

        The mode controller can restrict tool access based on the current
        mode (e.g., safe mode, research mode, etc.).

        Args:
            mode_controller: Mode controller instance

        Example:
            service.set_mode_controller(mode_controller)
        """
        self._mode_controller = mode_controller
        self._logger.debug(f"Mode controller set: {type(mode_controller).__name__}")

    def set_tool_planner(self, tool_planner: Any) -> None:
        """Set the tool planner for goal-based tool selection.

        The tool planner analyzes tasks and infers goals to guide
        intelligent tool selection.

        Args:
            tool_planner: Tool planner instance

        Example:
            service.set_tool_planner(tool_planner)
        """
        self._tool_planner = tool_planner
        self._logger.debug(f"Tool planner set: {type(tool_planner).__name__}")

    def _pre_filter_tool_calls(
        self, tool_calls: List[Dict[str, Any]]
    ) -> tuple[List[Dict[str, Any]], List[str]]:
        """Pre-filter tool calls, removing hallucinated tool names.

        Filters out tool calls that reference tools that don't exist
        in the registry. This prevents execution failures from
        hallucinated tool names.

        Args:
            tool_calls: Raw tool calls from the model

        Returns:
            Tuple of (valid_calls, filtered_names)

        Example:
            valid, filtered = service._pre_filter_tool_calls(tool_calls)
            # valid: List of tool calls with known tool names
            # filtered: List of hallucinated tool names that were filtered
        """
        # Get known tool names
        known = self._get_known_tool_names()
        if not known:
            # No filtering if we can't determine known tools
            return tool_calls, []

        valid = []
        filtered_names = []

        for call in tool_calls:
            name = call.get("name", "")
            if name in known:
                valid.append(call)
            else:
                filtered_names.append(name)
                self._logger.debug(f"Filtered hallucinated tool: {name}")

        if filtered_names:
            self._logger.warning(
                f"Filtered {len(filtered_names)} hallucinated tool(s): {filtered_names}"
            )

        return valid, filtered_names

    def _get_known_tool_names(self) -> set[str]:
        """Get set of known tool names from registrar.

        Returns:
            Set of known tool names, or empty set if registrar unavailable
        """
        if self._registrar is None:
            return set()

        try:
            # Try to get all registered tools
            if hasattr(self._registrar, "get_all_tools"):
                all_tools = self._registrar.get_all_tools()
                return {tool.name for tool in all_tools if hasattr(tool, "name")}
            if hasattr(self._registrar, "get_tool_names"):
                return set(self._registrar.get_tool_names())
            elif hasattr(self._registrar, "list_tools"):
                tool_names = set()
                for tool in self._registrar.list_tools():
                    if isinstance(tool, str):
                        tool_names.add(tool)
                    elif hasattr(tool, "name"):
                        tool_names.add(tool.name)
                return tool_names
            else:
                return set()
        except Exception as e:
            self._logger.debug(f"Failed to get known tool names: {e}")
            return set()

    def set_argument_normalizer(self, normalizer: Any) -> None:
        """Set the argument normalizer for tool arguments.

        Args:
            normalizer: Argument normalizer instance

        Example:
            service.set_argument_normalizer(normalizer)
        """
        self._argument_normalizer = normalizer
        self._logger.debug(f"Argument normalizer set: {type(normalizer).__name__}")

    def normalize_tool_arguments(
        self,
        tool_args: Dict[str, Any],
        tool_name: str,
    ) -> tuple[Dict[str, Any], str]:
        """Normalize tool arguments to handle malformed JSON.

        Args:
            tool_args: Raw arguments from tool call
            tool_name: Name of the tool being called

        Returns:
            Tuple of (normalized_args, strategy_used)

        Example:
            args, strategy = service.normalize_tool_arguments(
                {"query": "test"}, "code_search"
            )
        """
        if not self._argument_normalizer:
            return tool_args, "direct"

        try:
            # Try to use the normalizer
            result = self._argument_normalizer.normalize_arguments(tool_args, tool_name)
            if isinstance(result, tuple):
                return result
            return result, "normalized"
        except Exception as e:
            self._logger.debug(f"Argument normalization failed: {e}")
            return tool_args, "direct"

    async def execute_tool_with_retry(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        context: Dict[str, Any],
        tool_executor: Optional[Callable[..., Awaitable[Any]]] = None,
        cache: Optional[Any] = None,
        on_success: Optional[Callable[[str, Dict[str, Any], Any], None]] = None,
        retry_config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[Any], bool, Optional[str]]:
        """Execute a tool with retry logic and exponential backoff."""
        if self._retry_executor is not None:
            return await self._retry_executor.execute_tool_with_retry(
                tool_name=tool_name,
                tool_args=tool_args,
                context=context,
                tool_executor=tool_executor,
                cache=cache,
                on_success=on_success,
                retry_config=retry_config,
            )

        try:
            effective_executor = tool_executor
            if effective_executor is None and hasattr(self._executor, "execute"):

                async def _default_execute(
                    name: str,
                    arguments: Dict[str, Any],
                    _context: Dict[str, Any],
                ) -> Any:
                    return await self._executor.execute(name, arguments)

                effective_executor = _default_execute

            if effective_executor is None:
                raise RuntimeError("Tool retry execution requires a bound executor")

            result = await effective_executor(tool_name, tool_args, context)
            success = bool(getattr(result, "success", False))
            error = None if success else str(getattr(result, "error", "Unknown error"))
            if success and on_success is not None:
                on_success(tool_name, tool_args, result)
            return result, success, error
        except Exception as exc:
            return None, False, str(exc)

    # ==========================================================================
    # Tool Selection Convenience Methods
    # ==========================================================================

    def can_select_tools(self) -> bool:
        """Check if tool selection is available.

        Returns True if a tool selector is configured and ready.

        Returns:
            True if tool selection is available

        Example:
            if service.can_select_tools():
                tools = await service.select_tools(context)
        """
        return self._selector is not None

    def get_selection_config(self) -> Dict[str, Any]:
        """Get tool selection configuration.

        Returns the current configuration for tool selection
        including max tools, thresholds, etc.

        Returns:
            Dictionary with selection configuration

        Example:
            config = service.get_selection_config()
            print(f"Max tools: {config['max_tools']}")
        """
        return {
            "max_tools": self._config.default_max_tools,
            "has_selector": self._selector is not None,
            "selector_type": type(self._selector).__name__ if self._selector else None,
        }

    async def select_tools_sync(
        self,
        message: str,
        task_type: str = "unknown",
        max_tools: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Select tools with simplified synchronous-style interface.

        Convenience wrapper around select_tools that provides a simpler
        interface for basic tool selection.

        Args:
            message: User message/query
            task_type: Task type (e.g., "edit", "analyze", "debug")
            max_tools: Maximum tools to select (uses config default if None)
            context: Additional context for selection

        Returns:
            List of selected tool names

        Example:
            tools = await service.select_tools_sync(
                message="Search for foo function",
                task_type="analyze",
                max_tools=5
            )
        """
        if not self._selector:
            self._logger.warning("No tool selector configured")
            return []

        # Build context dict
        selection_context: "ToolSelectionContext" = {
            "message": message,
            "task_type": task_type,
            "max_tools": max_tools or self._config.default_max_tools,
        }

        if context:
            selection_context.update(context)

        # Call the async select_tools method
        return await self.select_tools(
            selection_context, max_tools or self._config.default_max_tools
        )

    def estimate_tools_needed(
        self,
        message: str,
        task_type: str = "unknown",
    ) -> int:
        """Estimate number of tools needed for a task.

        Provides a quick heuristic estimate without full selection.
        Useful for budget planning and early filtering.

        Args:
            message: User message/query
            task_type: Task type

        Returns:
            Estimated number of tools needed

        Example:
            estimated = service.estimate_tools_needed(message, "edit")
            if estimated > remaining_budget:
                # Handle budget constraint
        """
        # Simple heuristic based on message length and task type
        word_count = len(message.split())

        # Base estimate on task type
        base_estimate = {
            "edit": 3,
            "analyze": 4,
            "debug": 5,
            "search": 2,
            "refactor": 4,
            "test": 3,
            "unknown": 3,
        }.get(task_type, 3)

        # Adjust based on message complexity
        if word_count > 50:
            base_estimate += 1
        if word_count > 100:
            base_estimate += 1

        # Cap at configured max
        return min(base_estimate, self._config.default_max_tools)

    def get_recommended_tools(
        self,
        task_type: str = "unknown",
    ) -> List[str]:
        """Get recommended tools for a task type.

        Returns a list of commonly used tools for a given task type.
        This is a static recommendation, not based on message content.

        Args:
            task_type: Task type (e.g., "edit", "analyze", "debug")

        Returns:
            List of recommended tool names

        Example:
            tools = service.get_recommended_tools("analyze")
            # ["code_search", "file_read", "grep", ...]
        """
        # Static recommendations based on task type
        recommendations = {
            "edit": ["code_search", "file_read", "file_write", "edit"],
            "analyze": ["code_search", "file_read", "grep", "syntax_check"],
            "debug": ["code_search", "file_read", "grep", "test_runner"],
            "search": ["code_search", "grep", "file_search"],
            "refactor": ["code_search", "file_read", "edit", "syntax_check"],
            "test": ["test_runner", "code_search", "file_read"],
            "build": ["shell", "file_read"],
            "deploy": ["shell", "file_read", "file_write"],
            "unknown": ["code_search", "file_read"],
        }

        # Filter to only available tools
        all_tools = self.get_available_tools()
        recommended = recommendations.get(task_type, recommendations["unknown"])

        return [tool for tool in recommended if tool in all_tools]

    # ==========================================================================
    # Tool Execution Convenience Methods
    # ==========================================================================

    def can_execute_tools(self) -> bool:
        """Check if tool execution is available.

        Returns True if a tool executor is configured and ready.

        Returns:
            True if tool execution is available

        Example:
            if service.can_execute_tools():
                result = await service.execute_tool_call(call)
        """
        return self._executor is not None

    async def execute_tool_call(
        self,
        tool_call: Dict[str, Any],
        validate: bool = True,
        check_budget: bool = True,
    ) -> Dict[str, Any]:
        """Execute a single tool call with validation and budgeting.

        Convenience wrapper that validates, checks budget, and executes
        a single tool call.

        Args:
            tool_call: Tool call dictionary with 'name' and 'arguments'
            validate: Whether to validate tool call before execution
            check_budget: Whether to check budget before execution

        Returns:
            Execution result dictionary with 'result', 'error', 'success' fields

        Example:
            result = await service.execute_tool_call(
                {"name": "code_search", "arguments": {"query": "foo"}}
            )
            if result["success"]:
                print(result["result"])
        """
        result = {
            "tool": tool_call.get("name", "unknown"),
            "success": False,
            "result": None,
            "error": None,
        }

        try:
            # Normalize tool call
            normalized = self.normalize_tool_call(tool_call)
            tool_name = normalized.get("name")
            arguments = normalized.get("arguments", {})

            if not tool_name:
                result["error"] = "Tool name missing"
                return result

            # Validate if requested
            if validate:
                is_valid, error = self._validate_tool_call(normalized)
                if not is_valid:
                    result["error"] = error
                    return result

                # Check arguments
                args_valid, args_error = self.check_tool_arguments(tool_name, arguments)
                if not args_valid:
                    result["error"] = args_error
                    return result

            # Check budget if requested
            if check_budget:
                if self.is_budget_exhausted():
                    result["error"] = "Tool budget exhausted"
                    return result

            # Execute the tool
            if not self._executor:
                result["error"] = "No tool executor configured"
                return result

            # Call the executor's execute_tool method
            execution_result = await self._executor.execute_tool(
                tool_name=tool_name,
                arguments=arguments,
            )

            # Process result
            if execution_result:
                result["success"] = True
                result["result"] = execution_result

                # Track usage
                self._budget_manager.record_usage(1)
                self._track_tool_usage(tool_name, success=True)
            else:
                result["error"] = "Execution returned no result"
                self._track_tool_usage(tool_name, success=False)

        except Exception as e:
            result["error"] = str(e)
            self._track_tool_usage(result["tool"], success=False)

        return result

    async def execute_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
        validate: bool = True,
        check_budget: bool = True,
        parallel: bool = True,
    ) -> List[Dict[str, Any]]:
        """Execute multiple tool calls with validation and budgeting.

        Convenience wrapper that executes multiple tool calls with
        optional validation, budget checking, and parallel execution.

        Args:
            tool_calls: List of tool call dictionaries
            validate: Whether to validate tool calls before execution
            check_budget: Whether to check budget before execution
            parallel: Whether to execute tools in parallel (if budget allows)

        Returns:
            List of execution result dictionaries

        Example:
            results = await service.execute_tool_calls([
                {"name": "code_search", "arguments": {"query": "foo"}},
                {"name": "file_read", "arguments": {"path": "/tmp/test"}},
            ])
            for result in results:
                if result["success"]:
                    print(f"{result['tool']}: {result['result']}")
        """
        return await execute_tool_call_batch(
            self,
            tool_calls,
            validate=validate,
            check_budget=check_budget,
            parallel=parallel,
        )

    async def execute_tool_with_validation(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        normalize: bool = True,
    ) -> Dict[str, Any]:
        """Execute a tool with comprehensive validation.

        High-level execution method that validates the tool name,
        normalizes arguments, checks budget, and executes.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments
            normalize: Whether to normalize arguments

        Returns:
            Execution result dictionary

        Example:
            result = await service.execute_tool_with_validation(
                "code_search",
                {"query": "foo", "path": "/tmp"}
            )
        """
        # Build tool call
        tool_call = {"name": tool_name, "arguments": arguments}

        # Execute with full validation
        return await self.execute_tool_call(
            tool_call,
            validate=True,
            check_budget=True,
        )

    def get_execution_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get summary of tool execution results.

        Analyzes a list of execution results and provides summary
        statistics including success rate, errors, etc.

        Args:
            results: List of execution result dictionaries

        Returns:
            Summary dictionary with statistics

        Example:
            results = await service.execute_tool_calls(calls)
            summary = service.get_execution_summary(results)
            print(f"Success: {summary['success_count']}/{summary['total_count']}")
        """
        total = len(results)
        successful = sum(1 for r in results if r.get("success", False))
        failed = total - successful

        errors = {}
        for result in results:
            error = result.get("error")
            if error:
                errors[error] = errors.get(error, 0) + 1

        return {
            "total_count": total,
            "success_count": successful,
            "failed_count": failed,
            "success_rate": successful / total if total > 0 else 0.0,
            "errors": errors,
            "has_errors": failed > 0,
        }

    async def safe_execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        default: Any = None,
    ) -> Any:
        """Execute a tool safely with error handling.

        Executes a tool and returns a default value on error,
        never raises exceptions.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments
            default: Default value to return on error

        Returns:
            Tool result or default value on error

        Example:
            result = await service.safe_execute_tool(
                "code_search",
                {"query": "foo"},
                default=[]
            )
        """
        try:
            result = await self.execute_tool_with_validation(tool_name, arguments)
            if result.get("success"):
                return result.get("result", default)
            return default
        except Exception:
            return default

    # ==========================================================================
    # KV-Prefix Strategy Methods (extracted from AgentOrchestrator)
    # ==========================================================================

    def sort_tools_for_kv_stability(self, tools, *, kv_optimization_enabled: bool = True) -> list:
        """Sort tools by stable semantic rank, schema level, then name.

        Produces deterministic ordering so the stable prefix bytes remain unchanged
        across turns. The primary key is a numeric coding-agent tool rank rather
        than alphabetical order, because serialized tool order is also a model
        salience prior. Session-local caching of the sorted list is the caller's
        responsibility (ToolService is a container singleton and must not store
        per-session state).
        """
        if tools is None:
            return None
        if not kv_optimization_enabled:
            return tools
        level_order = {"full": 0, "compact": 1, "stub": 2, None: 2}
        unknown_rank = len(self.STABLE_TOOL_ORDER_RANK)
        return sorted(
            tools,
            key=lambda t: (
                self.STABLE_TOOL_ORDER_RANK.get(t.name, unknown_rank),
                level_order.get(getattr(t, "schema_level", None), 2),
                t.name,
            ),
        )

    def estimate_tool_tokens(self, tool, *, provider_category: Optional[str] = None) -> int:
        """Estimate token cost for a tool at its current schema level.

        Falls back to a name-length heuristic when the tool's schema cannot be
        generated (e.g. missing ``to_schema`` implementation).
        """
        from victor.config.tool_tiers import get_provider_tool_tier, get_tool_tier

        try:
            tier = (
                get_provider_tool_tier(tool.name, provider_category)
                if provider_category
                else get_tool_tier(tool.name)
            )
            schema = tool.to_schema(tier)
            return len(str(schema)) // 4
        except Exception:
            return len(tool.name) + 50

    def apply_kv_tool_strategy(
        self,
        tools,
        *,
        kv_optimization_enabled: bool,
        provider: Any,
        model: str,
        session_semantic_tools: Optional[list] = None,
        kv_tool_strategy: str = "context_aware",
    ) -> list:
        """Apply the configured KV tool-selection strategy.

        All session-scoped mutable state (session_semantic_tools) is passed by
        value; ToolService never stores per-session state.

        Strategies:
          ``session_stable`` / ``additive`` — grow-only session tool set: keep the
                               cached set (KV-cache prefix) and ADD any newly-selected
                               tools, so a mid-session intent shift ("now write the fix")
                               can still surface edit/write instead of being locked out.
                               A turn that selects nothing new returns the cached set
                               unchanged (full cache hit). Caller persists the result.
          ``per_turn``       — return tools unchanged (fresh each turn).
          ``context_aware``  — economy-first context-window-aware selection
                               (default).
        """
        if tools is None:
            return None
        if not tools:
            return tools

        if kv_tool_strategy == "per_turn":
            return tools

        if kv_tool_strategy in ("session_stable", "additive"):
            return self._merge_session_tools(tools, session_semantic_tools)

        return self.apply_context_aware_strategy(
            tools,
            provider=provider,
            model=model,
            session_semantic_tools=session_semantic_tools,
        )

    @staticmethod
    def _merge_session_tools(tools: list, cached: Optional[list]) -> list:
        """Additive session-stable merge (tool-supply P4): grow-only tool set.

        Returns the cached set plus any newly-selected tools not already in it. When
        nothing new is selected, returns the cached object unchanged so the downstream
        name-sort yields a byte-identical prefix (KV cache hit). The set only grows —
        it never shrinks or drops a tool the model may have started using — so a later
        turn can surface new tools without re-locking the prefix from scratch.
        """
        if not cached:
            return tools
        cached_names = {t.name for t in cached}
        additions = [t for t in tools if t.name not in cached_names]
        if not additions:
            return cached
        return list(cached) + additions

    def apply_context_aware_strategy(
        self,
        tools,
        *,
        provider: Any,
        model: str,
        session_semantic_tools: Optional[list] = None,
    ) -> list:
        """Economy-first, context-window-aware tool selection.

        Decision tree (priority order):
        1. Resolve one provider-economics profile for cache/token tradeoffs.
        2. Demote/drop tools that exceed the profile's tool-schema budget.
        3. Apply additive session stability when the profile asks for it.
        4. For hard token-constrained profiles, semantic-select within budget.

        Does NOT emit tool-strategy events — that responsibility stays with the
        orchestrator shim to preserve ``AgentMetricsService`` ownership.
        """
        from victor.config.tool_tiers import get_provider_category
        from victor.config.tool_tiers import resolve_tool_supply_profile

        context_window = self._get_tool_context_window(provider, model)
        fallback_max_tools = self._fallback_max_full_tools()
        profile = resolve_tool_supply_profile(
            provider,
            context_window,
            fallback_max_tools=fallback_max_tools,
        )
        provider_category = get_provider_category(context_window)
        tool_tokens = sum(
            self.estimate_tool_tokens(t, provider_category=provider_category) for t in tools
        )
        max_tool_tokens = profile.budget_tokens or int(context_window * 0.25)

        if tool_tokens > max_tool_tokens:
            self._logger.warning(
                f"Tool tokens ({tool_tokens}) exceed 25% of context window ({context_window}). "
                "Demoting low-priority tools."
            )
            tools = self._demote_tools_to_fit_budget(
                tools, max_tool_tokens, context_window, provider_category
            )

        if profile.cap_mode == "none" or self._should_session_lock_tools(provider, context_window):
            if profile.session_lock == "additive":
                return self._merge_session_tools(tools, session_semantic_tools)
            return tools

        tools = self.semantic_select_tools(
            tools, max_tool_tokens, provider_category=provider_category
        )
        if profile.session_lock == "additive":
            return self._merge_session_tools(tools, session_semantic_tools)
        return tools

    def _fallback_max_full_tools(self) -> int:
        """Return configured full-schema head size for profile resolution."""
        try:
            tools_settings = getattr(self._settings, "tools", None)
            budget = getattr(tools_settings, "budget", None)
            if budget and int(budget) > 0:
                return int(budget)
        except Exception:
            pass
        return 8

    def semantic_select_tools(
        self, tools, max_tokens: int, *, provider_category: Optional[str] = None
    ) -> list:
        """Select tools by semantic relevance within a token budget.

        CRITICAL-priority tools are always included first.  Remaining tools are
        added in declaration order until the budget is 90 % consumed.
        """
        from victor.tools.enums import Priority

        core_tools = [
            t for t in tools if hasattr(t, "priority") and t.priority == Priority.CRITICAL
        ]
        core_tokens = sum(
            self.estimate_tool_tokens(t, provider_category=provider_category) for t in core_tools
        )

        if core_tokens > max_tokens:
            result, used = [], 0
            for tool in core_tools:
                cost = self.estimate_tool_tokens(tool)
                if used + cost <= max_tokens:
                    result.append(tool)
                    used += cost
            return result

        selected = core_tools.copy()
        for tool in (t for t in tools if t not in core_tools):
            cost = self.estimate_tool_tokens(tool, provider_category=provider_category)
            if core_tokens + cost <= max_tokens:
                selected.append(tool)
                core_tokens += cost
                if core_tokens >= max_tokens * 0.9:
                    break

        return selected

    # ------------------------------------------------------------------
    # Internal helpers for KV strategy methods
    # ------------------------------------------------------------------

    def _get_tool_context_window(self, provider: Any, model: str) -> int:
        if hasattr(provider, "context_window"):
            return provider.context_window(model)
        self._logger.warning(
            f"Provider {getattr(provider, 'name', '?')} has no context_window(); using 8192"
        )
        return 8192

    def _should_session_lock_tools(self, provider: Any, context_window: int) -> bool:
        if hasattr(provider, "supports_prompt_caching") and provider.supports_prompt_caching():
            return True
        return context_window >= 32000

    def _demote_tools_to_fit_budget(
        self,
        tools: list,
        max_tokens: int,
        context_window: int,
        provider_category: Optional[str] = None,
    ) -> list:
        from victor.tools.enums import Priority, SchemaLevel

        sorted_tools = sorted(
            tools,
            key=lambda t: (t.priority.value if hasattr(t, "priority") else 99, t.name),
        )
        result, used = [], 0
        for tool in sorted_tools:
            cost = self.estimate_tool_tokens(tool, provider_category=provider_category)
            if used + cost <= max_tokens:
                result.append(tool)
                used += cost
            elif hasattr(tool, "priority") and tool.priority == Priority.CRITICAL:
                try:
                    original = getattr(tool, "_schema_level", None)
                    tool._schema_level = SchemaLevel.STUB
                    stub_cost = self.estimate_tool_tokens(tool)
                    tool._schema_level = original
                    if used + stub_cost <= max_tokens:
                        result.append(tool)
                        used += stub_cost
                        self._logger.debug(f"Demoted critical tool {tool.name} to STUB")
                    else:
                        self._logger.warning(
                            f"Critical tool {tool.name} exceeds budget even as STUB; dropping"
                        )
                except Exception as exc:
                    self._logger.warning(f"Error demoting tool {tool.name}: {exc}")
        self._logger.info(
            f"Demoted tools to fit context window: {len(tools)} → {len(result)} tools "
            f"({used} tokens, budget: {max_tokens}, context: {context_window})"
        )
        return result

    # ==========================================================================
    # Private Methods
    # ==========================================================================

    def _track_tool_usage(self, tool_name: str, success: bool) -> None:
        """Track tool usage for statistics.

        Args:
            tool_name: Name of the tool
            success: Whether execution was successful
        """
        self._usage_tracker.record(tool_name, success=success)

    async def _get_cached_result(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Optional[Any]:
        """Get cached tool result if available.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments

        Returns:
            Cached result if available and valid, None otherwise
        """
        # This would integrate with the caching service
        # For now, return None to indicate no cache
        return None

    async def _cache_result(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Any,
    ) -> None:
        """Cache a tool result.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments
            result: Result to cache
        """
        # This would integrate with the caching service
        # For now, just log
        self._logger.debug(f"Caching result for {tool_name}")


class ToolBudgetExceededError(Exception):
    """Raised when tool budget is exhausted."""

    pass

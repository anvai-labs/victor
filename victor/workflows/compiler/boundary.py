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

"""Workflow compiler boundary primitives.

This module introduces the parser / validator / compiler split used by the
DI-facing workflow compiler APIs. It provides both a native StateGraph backend
and the legacy compiler adapter so the outer compilation pipeline stays
explicit, injectable, and testable during migration.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

from victor.workflows.definition import (
    ConditionNode,
    ParallelNode,
    WorkflowDefinition,
    WorkflowNodeType,
)
from victor.workflows.runtime_types import GraphNodeResult, WorkflowState

if TYPE_CHECKING:
    from victor.workflows.compiler_protocols import CompiledGraphProtocol

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkflowCompilationRequest:
    """Normalized request for compiling a workflow source."""

    source: str
    workflow_name: Optional[str] = None
    validate: bool = True


@dataclass(frozen=True)
class ParsedWorkflowDefinition:
    """A parsed workflow definition plus source metadata."""

    request: WorkflowCompilationRequest
    workflow: WorkflowDefinition
    source_path: Optional[Path] = None

    @property
    def workflow_name(self) -> str:
        """Return the normalized workflow name."""
        return self.workflow.name


def _looks_like_yaml_content(source: str) -> bool:
    stripped = source.lstrip()
    return stripped.startswith(("workflows:", "-", "{"))


def create_condition_router(node: ConditionNode) -> Callable[[WorkflowState], str]:
    """Build a condition router shared across compiler compatibility layers."""

    def route(state: WorkflowState) -> str:
        try:
            branch = node.condition(dict(state))
            if branch in node.branches:
                return branch
            if "default" in node.branches:
                return "default"

            logger.warning(
                "Condition node '%s' returned '%s' without a matching branch",
                node.id,
                branch,
            )
        except Exception as exc:
            logger.error(
                "Condition evaluation failed for node '%s': %s",
                node.id,
                exc,
                exc_info=True,
            )
            if "default" in node.branches:
                return "default"
        return "__END__"

    return route


class WorkflowParser:
    """Parse file-or-string workflow sources into normalized definitions."""

    def __init__(self, yaml_loader: Any):
        self._yaml_loader = yaml_loader

    def parse(self, request: WorkflowCompilationRequest) -> ParsedWorkflowDefinition:
        """Parse and normalize the requested workflow source."""
        loaded = self._yaml_loader.load(request.source, workflow_name=request.workflow_name)
        workflow = self._normalize_loaded_workflow(loaded, request.workflow_name)
        return ParsedWorkflowDefinition(
            request=request,
            workflow=workflow,
            source_path=self._resolve_source_path(request.source),
        )

    def _normalize_loaded_workflow(
        self,
        loaded: Any,
        workflow_name: Optional[str],
    ) -> WorkflowDefinition:
        if isinstance(loaded, WorkflowDefinition):
            return loaded

        if isinstance(loaded, dict):
            if workflow_name:
                try:
                    workflow = loaded[workflow_name]
                except KeyError as exc:  # pragma: no cover - loader should prevent this
                    raise ValueError(f"Workflow '{workflow_name}' not found") from exc
                if isinstance(workflow, WorkflowDefinition):
                    return workflow

            definitions = {
                name: workflow
                for name, workflow in loaded.items()
                if isinstance(workflow, WorkflowDefinition)
            }
            if len(definitions) == 1:
                return next(iter(definitions.values()))
            if len(definitions) > 1:
                raise ValueError(
                    "Multiple workflows found. Please specify workflow_name. "
                    f"Available: {list(definitions.keys())}"
                )

        raise TypeError(f"Unsupported workflow load result: {type(loaded).__name__}")

    def _resolve_source_path(self, source: str) -> Optional[Path]:
        if _looks_like_yaml_content(source):
            return None

        candidate = Path(source)
        if candidate.exists():
            return candidate.resolve()
        return None


class WorkflowDefinitionValidator:
    """Validate parsed workflow definitions before graph compilation."""

    def __init__(self, validator: Any = None):
        self._validator = validator

    def validate(self, parsed: ParsedWorkflowDefinition) -> ParsedWorkflowDefinition:
        """Validate the normalized workflow definition and return it unchanged."""
        errors = list(parsed.workflow.validate())
        if errors:
            raise ValueError(
                f"Invalid workflow '{parsed.workflow.name}': {'; '.join(str(error) for error in errors)}"
            )

        errors.extend(self._normalize_validator_errors(parsed.workflow))

        if errors:
            raise ValueError(
                f"Invalid workflow '{parsed.workflow.name}': {'; '.join(str(error) for error in errors)}"
            )

        return parsed

    def _normalize_validator_errors(self, workflow: WorkflowDefinition) -> list[str]:
        if self._validator is None:
            return []

        result = self._validator.validate(workflow)
        if result is None:
            return []
        if isinstance(result, bool):
            return [] if result else ["Workflow validator rejected definition"]
        if isinstance(result, (list, tuple, set)):
            return [str(error) for error in result if error]

        is_valid = getattr(result, "is_valid", True)
        errors = getattr(result, "errors", None)
        if errors:
            return [str(error) for error in errors if error]
        if is_valid is False:
            return ["Workflow validator rejected definition"]
        return []


class NativeWorkflowGraphCompiler:
    """Compiler backend that builds StateGraph directly from WorkflowDefinition.

    This replaces the previous facade-to-legacy-compiler dependency while still
    preserving the established node semantics by routing execution through the
    injected NodeExecutorFactoryProtocol.
    """

    def __init__(
        self,
        node_executor_factory: Any,
        *,
        checkpointer_factory: Optional[Callable[[], Any]] = None,
        enable_checkpointing: bool = True,
        interrupt_on_hitl: bool = True,
    ):
        self._node_executor_factory = node_executor_factory
        self._checkpointer_factory = checkpointer_factory or self._create_checkpointer
        self._enable_checkpointing = enable_checkpointing
        self._interrupt_on_hitl = interrupt_on_hitl

    def compile(self, parsed: ParsedWorkflowDefinition) -> "CompiledGraphProtocol":
        """Compile a parsed workflow definition into an executable graph."""
        from victor.framework.graph import END, StateGraph

        workflow = parsed.workflow
        graph = StateGraph(WorkflowState)
        parallel_children = self._collect_parallel_children(workflow)

        for node_id, node in workflow.nodes.items():
            if node_id in parallel_children:
                continue

            if isinstance(node, ParallelNode):
                self._add_parallel_node_group(graph, workflow, node)
                continue

            graph.add_node(node_id, self._node_executor_factory.create_executor(node))

        for node_id, node in workflow.nodes.items():
            if node_id in parallel_children:
                continue

            if isinstance(node, ConditionNode):
                graph.add_conditional_edge(
                    node_id, self._create_condition_router(node), node.branches
                )
                continue

            if isinstance(node, ParallelNode):
                for next_node_id in node.next_nodes:
                    if next_node_id in workflow.nodes or next_node_id == END:
                        graph.add_edge(node_id, next_node_id)
                continue

            for next_node_id in node.next_nodes:
                if next_node_id in parallel_children:
                    continue
                if next_node_id in workflow.nodes or next_node_id == END:
                    graph.add_edge(node_id, next_node_id)

        entry_point = workflow.start_node or next(iter(workflow.nodes.keys()), None)
        if not entry_point:
            raise ValueError("WorkflowDefinition has no start_node")

        graph.set_entry_point(entry_point)

        interrupt_before = []
        if self._interrupt_on_hitl:
            interrupt_before = [
                node_id
                for node_id, node in workflow.nodes.items()
                if node.node_type == WorkflowNodeType.HITL
            ]

        logger.debug(
            "Compiling parsed workflow '%s' via native StateGraph backend",
            parsed.workflow_name,
        )
        return graph.compile(
            checkpointer=self._build_checkpointer(),
            max_iterations=workflow.max_iterations,
            timeout=workflow.max_execution_timeout_seconds,
            interrupt_before=interrupt_before,
        )

    def _collect_parallel_children(self, workflow: WorkflowDefinition) -> set[str]:
        parallel_children: set[str] = set()
        for node in workflow.nodes.values():
            if isinstance(node, ParallelNode):
                parallel_children.update(node.parallel_nodes)
        return parallel_children

    def _add_parallel_node_group(
        self,
        graph: Any,
        workflow: WorkflowDefinition,
        parallel_node: ParallelNode,
    ) -> None:
        child_nodes = [
            workflow.nodes[child_id]
            for child_id in parallel_node.parallel_nodes
            if child_id in workflow.nodes
        ]
        child_executors = [
            (child_node, self._node_executor_factory.create_executor(child_node))
            for child_node in child_nodes
        ]

        async def execute_parallel_group(state: WorkflowState) -> WorkflowState:
            current_state = dict(state)
            start_time = time.time()
            parallel_results = dict(current_state.get("_parallel_results", {}))
            node_results = dict(current_state.get("_node_results", {}))

            tasks = [executor(copy.deepcopy(current_state)) for _, executor in child_executors]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for (child_node, _), result in zip(child_executors, results):
                if isinstance(result, Exception):
                    parallel_results[child_node.id] = {
                        "success": False,
                        "error": str(result),
                    }
                    continue

                for key, value in result.items():
                    if not key.startswith("_"):
                        current_state[key] = value
                parallel_results[child_node.id] = {
                    "success": True,
                    "output": result.get(getattr(child_node, "output_key", None) or child_node.id),
                }

            current_state["_parallel_results"] = parallel_results
            node_results[parallel_node.id] = GraphNodeResult(
                node_id=parallel_node.id,
                success=all(result.get("success", False) for result in parallel_results.values()),
                output=parallel_results,
                duration_seconds=time.time() - start_time,
            )
            current_state["_node_results"] = node_results
            return current_state

        graph.add_node(parallel_node.id, execute_parallel_group)

    def _create_condition_router(self, node: ConditionNode) -> Callable[[WorkflowState], str]:
        return create_condition_router(node)

    def _build_checkpointer(self) -> Any:
        if not self._enable_checkpointing:
            return None
        return self._checkpointer_factory()

    def _create_checkpointer(self) -> Any:
        from victor.framework.graph import MemoryCheckpointer

        return MemoryCheckpointer()


class LegacyWorkflowGraphCompiler:
    """Compiler adapter that still targets the legacy graph compiler backend."""

    def __init__(
        self,
        compiler_factory: Optional[Callable[[], Any]] = None,
    ):
        self._compiler_factory = compiler_factory or self._create_legacy_compiler

    def compile(self, parsed: ParsedWorkflowDefinition) -> "CompiledGraphProtocol":
        """Compile a parsed workflow definition into an executable graph."""
        compiler = self._compiler_factory()
        logger.debug(
            "Compiling parsed workflow '%s' via legacy graph backend",
            parsed.workflow_name,
        )
        return compiler.compile(parsed.workflow)

    def _create_legacy_compiler(self) -> Any:
        from victor.workflows.yaml_to_graph_compiler import YAMLToStateGraphCompiler

        return YAMLToStateGraphCompiler(
            orchestrator=None,
            orchestrators=None,
        )


class LegacyWorkflowDslCompiler:
    """Compiler adapter for WorkflowGraph DSL via the legacy graph compiler backend."""

    def __init__(
        self,
        *,
        runner_registry: Optional[Any] = None,
        validate_before_compile: bool = True,
        preserve_state_type: bool = False,
        emitter: Optional[Any] = None,
        enable_observability: bool = False,
        compiler_factory: Optional[Callable[[], Any]] = None,
    ):
        self._runner_registry = runner_registry
        self._validate_before_compile = validate_before_compile
        self._preserve_state_type = preserve_state_type
        self._emitter = emitter
        self._enable_observability = enable_observability
        self._compiler_factory = compiler_factory or self._create_legacy_compiler
        self._compiler: Optional[Any] = None

    def compile(self, graph: Any, name: Optional[str] = None) -> "CompiledGraphProtocol":
        """Compile a WorkflowGraph DSL object through the legacy backend."""
        compiler = self._get_compiler()
        return compiler.compile(graph, name)

    def _get_compiler(self) -> Any:
        if self._compiler is None:
            self._compiler = self._compiler_factory()
        return self._compiler

    def _create_legacy_compiler(self) -> Any:
        from victor.workflows.graph_compiler import (
            CompilerConfig,
            WorkflowGraphCompiler,
        )

        return WorkflowGraphCompiler(
            CompilerConfig(
                use_node_runners=self._runner_registry is not None,
                runner_registry=self._runner_registry,
                validate_before_compile=self._validate_before_compile,
                preserve_state_type=self._preserve_state_type,
                emitter=self._emitter,
                enable_observability=self._enable_observability,
            )
        )


__all__ = [
    "create_condition_router",
    "LegacyWorkflowDslCompiler",
    "LegacyWorkflowGraphCompiler",
    "NativeWorkflowGraphCompiler",
    "ParsedWorkflowDefinition",
    "WorkflowCompilationRequest",
    "WorkflowDefinitionValidator",
    "WorkflowParser",
]

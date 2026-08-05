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

"""Concrete compiler wrapper for DI registration using canonical implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from victor.workflows.compiler.boundary import (
    NativeWorkflowGraphCompiler,
    WorkflowCompilationRequest,
    WorkflowDefinitionValidator,
    WorkflowParser,
)

if TYPE_CHECKING:
    from victor.workflows.compiler_protocols import (
        CompiledGraphProtocol,
        NodeExecutorFactoryProtocol,
    )


class WorkflowCompilerImpl:
    """Concrete compiler wrapper using canonical compilation backend.

    This provides a DI-compatible wrapper around the canonical compilation
    primitives from ``victor.workflows.compiler.boundary``.

    For new code, prefer using ``UnifiedWorkflowCompiler`` directly from
    ``victor.workflows.unified_compiler`` which provides caching,
    observability, and multi-source compilation.
    """

    def __init__(
        self,
        yaml_loader: Any,
        validator: Any,
        node_factory: "NodeExecutorFactoryProtocol",
    ) -> None:
        """Initialize the compiler wrapper.

        Args:
            yaml_loader: YAMLWorkflowLoader instance
            validator: WorkflowValidator instance
            node_factory: NodeExecutorFactoryProtocol instance
        """
        self._yaml_loader = yaml_loader
        self._validator = validator
        self._node_executor_factory = node_factory
        self._workflow_parser = WorkflowParser(yaml_loader)
        self._workflow_definition_validator = WorkflowDefinitionValidator(validator)
        self._graph_compiler = NativeWorkflowGraphCompiler(node_executor_factory=node_factory)

    def compile(
        self,
        source: str,
        *,
        workflow_name: str | None = None,
        validate: bool = True,
    ) -> "CompiledGraphProtocol":
        """Compile a workflow from YAML source.

        Args:
            source: YAML file path or YAML string content
            workflow_name: Name of workflow to compile (for multi-workflow files)
            validate: Whether to validate workflow definition before compilation

        Returns:
            CompiledGraphProtocol: Executable compiled graph

        Raises:
            ValueError: If source is invalid or validation fails
            FileNotFoundError: If YAML file doesn't exist
        """
        request = WorkflowCompilationRequest(
            source=source,
            workflow_name=workflow_name,
            validate=validate,
        )
        parsed = self._workflow_parser.parse(request)

        if validate:
            parsed = self._workflow_definition_validator.validate(parsed)

        return self._graph_compiler.compile(parsed)


__all__ = [
    "WorkflowCompilerImpl",
]

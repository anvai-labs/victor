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

"""Workflow compiler protocols package.

This package provides ISP-compliant minimal protocols for the workflow system.
Import from this module for protocol-based dependency injection.

Example:
    from victor.workflows.compiler_protocols import (
        WorkflowCompilerProtocol,
        CompiledGraphProtocol,
        NodeExecutorFactoryProtocol,
    )

    def execute_workflow(
        compiler: WorkflowCompilerProtocol,
        yaml_path: str,
    ) -> ExecutionResultProtocol:
        compiled = compiler.compile(yaml_path)
        return await compiled.invoke({"input": "data"})
"""

from victor.workflows.compiler_protocols.compiler_protocols import (
    CompiledGraphProtocol,
    ExecutionContextProtocol,
    ExecutionEventProtocol,
    ExecutionResultProtocol,
    NodeExecutorFactoryProtocol,
    NodeExecutorProtocol,
    WorkflowCompilerProtocol,
)

__all__ = [
    "WorkflowCompilerProtocol",
    "CompiledGraphProtocol",
    "ExecutionResultProtocol",
    "ExecutionEventProtocol",
    "NodeExecutorFactoryProtocol",
    "NodeExecutorProtocol",
    "ExecutionContextProtocol",
]

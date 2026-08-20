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

"""Workflow compiler package.

Provides concrete implementations of the workflow compiler
that can be registered in the DI container.

.. note::
    The deprecated ``WorkflowCompiler`` has been removed.
    Use ``victor.workflows.unified_compiler.UnifiedWorkflowCompiler`` instead.
"""

from victor.workflows.compiler.workflow_compiler_impl import WorkflowCompilerImpl
from victor.workflows.compiler.boundary import (
    LegacyWorkflowDslCompiler,
    LegacyWorkflowGraphCompiler,
    NativeWorkflowGraphCompiler,
    ParsedWorkflowDefinition,
    WorkflowCompilationRequest,
    WorkflowDefinitionValidator,
    WorkflowParser,
)

__all__ = [
    "LegacyWorkflowDslCompiler",
    "LegacyWorkflowGraphCompiler",
    "NativeWorkflowGraphCompiler",
    "ParsedWorkflowDefinition",
    "WorkflowCompilerImpl",
    "WorkflowCompilationRequest",
    "WorkflowDefinitionValidator",
    "WorkflowParser",
]

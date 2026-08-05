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

"""Workflow compiler plugins.

This package provides plugin infrastructure for third-party verticals
to extend Victor's workflow compilation capabilities.

Purpose:
    Allow third-party packages to provide custom workflow compilers
    without modifying Victor's core code.

Example (Third-party package):
    # victor-security package
    from victor.workflows.plugins import WorkflowCompilerPlugin
    from victor.workflows.compiler_registry import register_compiler

    class SecurityCompilerPlugin(WorkflowCompilerPlugin):
        def compile(self, source, *, workflow_name=None, validate=True):
            # Security-specific compilation
            return self._compile_security_workflow(source)

    # Register plugin
    register_compiler("security", SecurityCompilerPlugin)

Usage:
    No built-in plugins are registered by default. Third-party packages
    should register their plugins during initialization.

    Built-in verticals use UnifiedWorkflowCompiler directly for YAML workflows.
"""

import logging

logger = logging.getLogger(__name__)


def register_builtin_plugins() -> None:
    """Register built-in workflow compiler plugins.

    This is a no-op function provided for compatibility with the bootstrap
    process. There are currently no built-in plugins - all workflow
    compilation is handled by UnifiedWorkflowCompiler directly.

    Third-party packages should register their plugins through their
    own initialization code.
    """
    pass


__all__ = []

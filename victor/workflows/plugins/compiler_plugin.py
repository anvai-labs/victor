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

"""Base protocol for workflow compiler plugins.

This module defines the protocol that all workflow compiler plugins
must implement. Plugins provide extensible workflow compilation
from various sources (YAML, JSON, database, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class WorkflowCompilerPlugin(ABC):
    """Base class for workflow compiler plugins.

    All compiler plugins must implement this protocol. This provides
    a consistent interface for compiling workflows from various sources.

    Similar to SQLAlchemy's Dialect class.

    Plugins are registered by file extension (.yaml, .json, etc.) and
    optional URI schemes for remote locations (s3, http, etc.).

    Example:
        class YamlCompilerPlugin(WorkflowCompilerPlugin):
            def compile(self, source, **kwargs):
                # Load and compile YAML
                ...
    """

    @abstractmethod
    def compile(
        self,
        source: str,
        *,
        workflow_name: Optional[str] = None,
        validate: bool = True,
    ) -> Any:
        """Compile workflow from source.

        Args:
            source: Workflow source (file path, string, URI, etc.)
            workflow_name: Name of workflow to compile (for multi-workflow files)
            validate: Whether to validate before compilation

        Returns:
            CompiledGraphProtocol instance

        Raises:
            ValueError: If source is invalid or validation fails
            FileNotFoundError: If source file doesn't exist

        Example:
            plugin = YamlCompilerPlugin()
            compiled = plugin.compile("workflow.yaml")
            result = await compiled.invoke({"input": "data"})
        """
        ...

    def validate_source(self, source: str) -> bool:
        """Validate source before compilation.

        Args:
            source: Workflow source to validate

        Returns:
            True if valid, False otherwise

        Note:
            Default implementation always returns True.
            Override for custom validation logic.
        """
        return True

    def get_cache_key(self, source: str) -> str:
        """Generate cache key for source.

        Args:
            source: Workflow source

        Returns:
            Cache key string

        Note:
            Default implementation returns source as-is.
            Override for custom cache key generation.
        """
        return source


__all__ = [
    "WorkflowCompilerPlugin",
]

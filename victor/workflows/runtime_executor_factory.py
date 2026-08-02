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

"""Canonical construction seam for compatibility workflow executors.

The legacy DAG executors remain public compatibility surfaces. Internal
framework owners should not construct them directly from scattered import
sites; they should go through this module so the remaining migration seam is
centralized and can be swapped later in one place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from victor.framework.coordinators.protocols import (
        IStreamingExecutor,
        IWorkflowExecutor,
    )


def create_legacy_workflow_executor(*args: Any, **kwargs: Any) -> "IWorkflowExecutor":
    """Create the compatibility workflow executor through a single seam."""
    from victor.workflows.unified_executor import CompiledWorkflowExecutor

    return CompiledWorkflowExecutor(*args, **kwargs)


def create_legacy_streaming_workflow_executor(*args: Any, **kwargs: Any) -> "IStreamingExecutor":
    """Create the compatibility streaming workflow executor through a single seam."""
    from victor.workflows.streaming_executor import StreamingWorkflowExecutor

    return StreamingWorkflowExecutor(*args, **kwargs)


__all__ = [
    "create_legacy_streaming_workflow_executor",
    "create_legacy_workflow_executor",
]

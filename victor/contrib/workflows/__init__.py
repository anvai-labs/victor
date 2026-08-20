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

"""Workflow escape hatch base classes for Victor verticals.

Provides base classes and utilities for implementing YAML workflow
support in verticals. Verticals can inherit from BaseWorkflowProvider
to get common workflow functionality while adding vertical-specific
workflows.

Usage:
    from victor.contrib.workflows import BaseWorkflowProvider

    class MyVerticalWorkflowProvider(BaseWorkflowProvider):
        def get_vertical_name(self) -> str:
            return \"myvertical\"

        def get_workflow_directories(self) -> List[str]:
            return [
                \"/usr/local/lib/victor-workflows/common\",
                \"~/.victor/workflows/myvertical\",
            ]
"""

from victor.contrib.workflows.base_provider import BaseWorkflowProvider
from victor.contrib.workflows.workflow_loader import WorkflowLoaderMixin

__all__ = [
    "BaseWorkflowProvider",
    "WorkflowLoaderMixin",
]

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

from unittest.mock import patch

from victor.framework.coordinators.yaml_coordinator import YAMLWorkflowCoordinator


def test_get_executor_uses_runtime_executor_factory() -> None:
    coordinator = YAMLWorkflowCoordinator()
    sentinel = object()

    with patch(
        "victor.framework.coordinators.yaml_coordinator.create_legacy_workflow_executor",
        return_value=sentinel,
    ) as factory:
        assert coordinator._get_executor() is sentinel
        assert coordinator._get_executor() is sentinel

    factory.assert_called_once_with()


def test_get_streaming_executor_uses_runtime_executor_factory() -> None:
    coordinator = YAMLWorkflowCoordinator()
    sentinel = object()

    with patch(
        "victor.framework.coordinators.yaml_coordinator.create_legacy_streaming_workflow_executor",
        return_value=sentinel,
    ) as factory:
        assert coordinator._get_streaming_executor() is sentinel
        assert coordinator._get_streaming_executor() is sentinel

    factory.assert_called_once_with()

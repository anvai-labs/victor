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

"""Tests for canonical A/B testing storage path resolution."""

from types import SimpleNamespace
from unittest.mock import patch

from victor.experiments.ab_testing.experiment import ABTestManager
from victor.experiments.ab_testing.metrics import MetricsCollector


def test_ab_test_manager_uses_global_victor_dir_by_default(tmp_path):
    """A/B manager should resolve its default storage through centralized Victor paths."""
    global_dir = tmp_path / ".victor"

    with patch(
        "victor.experiments.ab_testing.paths.get_project_paths",
        return_value=SimpleNamespace(global_victor_dir=global_dir),
    ):
        manager = ABTestManager()

    assert manager.storage_path == global_dir / "ab_tests.db"
    assert manager.storage_path.exists()


def test_metrics_collector_uses_global_victor_dir_by_default(tmp_path):
    """Metrics collector should resolve its default storage through centralized Victor paths."""
    global_dir = tmp_path / ".victor"

    with patch(
        "victor.experiments.ab_testing.paths.get_project_paths",
        return_value=SimpleNamespace(global_victor_dir=global_dir),
    ):
        collector = MetricsCollector()

    assert collector.storage_path == global_dir / "ab_tests.db"

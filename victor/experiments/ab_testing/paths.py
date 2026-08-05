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

"""Canonical storage-path helpers for A/B testing."""

from pathlib import Path

from victor.config.settings import get_project_paths


def get_default_ab_test_db_path() -> Path:
    """Resolve the canonical global A/B testing database path."""
    return get_project_paths().global_victor_dir / "ab_tests.db"

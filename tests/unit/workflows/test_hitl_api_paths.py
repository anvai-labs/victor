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

"""Tests for HITL API default storage path resolution."""

from types import SimpleNamespace
from unittest.mock import patch

from victor.workflows.hitl_api import SQLiteHITLStore


def test_sqlite_hitl_store_uses_global_victor_dir_by_default(tmp_path):
    """SQLite HITL storage should resolve its default database via centralized paths."""
    global_dir = tmp_path / ".victor"

    with patch(
        "victor.workflows.hitl_api.get_project_paths",
        return_value=SimpleNamespace(global_victor_dir=global_dir),
    ):
        store = SQLiteHITLStore()

    assert store.db_path == str(global_dir / "hitl.db")
    assert (global_dir / "hitl.db").exists()

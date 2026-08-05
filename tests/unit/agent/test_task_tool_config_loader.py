# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Tests for canonical task-tool config loading."""

from textwrap import dedent

from victor.agent.task_tool_config_loader import TaskToolConfigLoader
from victor.agent.unified_task_tracker import (
    Milestone,
    TrackerTaskType,
    UnifiedTaskConfigLoader,
    UnifiedTaskTracker,
)


def test_task_tool_config_loader_canonicalizes_legacy_yaml(tmp_path):
    """Legacy tool names in YAML should normalize to the canonical surface."""
    config_path = tmp_path / "task_tool_config.yaml"
    config_path.write_text(dedent("""
            task_types:
              edit:
                required_tools: [edit_files, read_file]
                stage_tools:
                  executing: [edit_files, read_file, execute_bash]
                force_action_hints:
                  after_target_read: "Use edit_files now."
            """))

    loader = TaskToolConfigLoader(str(config_path))

    edit_config = loader.get_task_config("edit")
    assert edit_config["required_tools"] == ["edit", "read"]
    assert edit_config["stage_tools"]["executing"] == ["edit", "read", "shell"]


def test_unified_task_tracker_accepts_legacy_aliases_for_progress_tracking():
    """Alias tool names should still update canonical tracker state correctly."""
    tracker = UnifiedTaskTracker()
    tracker._progress.target_files.add("src/app.py")

    tracker.record_tool_call("read_file", {"path": "src/app.py"})
    assert "src/app.py" in tracker._progress.files_read

    tracker.record_tool_call(
        "edit_files", {"ops": [{"path": "src/app.py", "old_str": "a", "new_str": "b"}]}
    )
    assert "src/app.py" in tracker._progress.files_modified

    tracker.record_tool_call("execute_bash", {"cmd": "pytest -q"})
    assert Milestone.CHANGE_VERIFIED in tracker._progress.milestones


def test_unified_task_config_loader_returns_canonical_required_tools():
    """Unified task config should expose canonical names to the tracker."""
    loader = UnifiedTaskConfigLoader()
    config = loader.get_task_config(TrackerTaskType.EDIT)

    assert "edit" in config.required_tools
    assert "read" in config.required_tools

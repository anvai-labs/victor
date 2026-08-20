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

"""Tests for session-scoped task management."""

import pytest

from victor.tools.task_management_tool import Task, TaskStatus, TaskStore


class TestTaskStatus:
    def test_values(self):
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.IN_PROGRESS.value == "in_progress"
        assert TaskStatus.COMPLETED.value == "completed"


class TestTask:
    def test_to_dict(self):
        task = Task(
            id="1",
            subject="Test",
            description="Desc",
            status=TaskStatus.PENDING,
            created_at=100.0,
            updated_at=100.0,
        )
        d = task.to_dict()
        assert d["id"] == "1"
        assert d["subject"] == "Test"
        assert d["status"] == "pending"

    def test_from_dict(self):
        data = {
            "id": "1",
            "subject": "Test",
            "description": "Desc",
            "status": "in_progress",
            "created_at": 100.0,
            "updated_at": 100.0,
            "metadata": {},
        }
        task = Task.from_dict(data)
        assert task.id == "1"
        assert task.status == TaskStatus.IN_PROGRESS

    def test_roundtrip(self):
        task = Task(
            id="5",
            subject="RT",
            description="roundtrip",
            status=TaskStatus.COMPLETED,
            created_at=1.0,
            updated_at=2.0,
            metadata={"key": "value"},
        )
        restored = Task.from_dict(task.to_dict())
        assert restored.id == task.id
        assert restored.subject == task.subject
        assert restored.status == task.status
        assert restored.metadata == task.metadata


class TestTaskStore:
    def test_create(self):
        store = TaskStore()
        task = store.create("Test task", "Description")
        assert task.subject == "Test task"
        assert task.description == "Description"
        assert task.status == TaskStatus.PENDING
        assert task.id == "1"
        assert task.created_at > 0

    def test_get(self):
        store = TaskStore()
        t = store.create("Task", "Desc")
        found = store.get(t.id)
        assert found is not None
        assert found.subject == "Task"

    def test_get_nonexistent(self):
        store = TaskStore()
        assert store.get("999") is None

    def test_update_status(self):
        store = TaskStore()
        t = store.create("Task", "Desc")
        updated = store.update(t.id, status=TaskStatus.IN_PROGRESS)
        assert updated.status == TaskStatus.IN_PROGRESS
        assert updated.updated_at >= t.created_at

    def test_update_subject(self):
        store = TaskStore()
        t = store.create("Old", "Desc")
        updated = store.update(t.id, subject="New")
        assert updated.subject == "New"

    def test_update_description(self):
        store = TaskStore()
        t = store.create("Task", "Old desc")
        updated = store.update(t.id, description="New desc")
        assert updated.description == "New desc"

    def test_update_metadata(self):
        store = TaskStore()
        t = store.create("Task", "Desc", metadata={"a": 1})
        updated = store.update(t.id, metadata={"b": 2})
        assert updated.metadata == {"a": 1, "b": 2}

    def test_update_nonexistent_raises(self):
        store = TaskStore()
        with pytest.raises(KeyError):
            store.update("999", status=TaskStatus.COMPLETED)

    def test_list_all(self):
        store = TaskStore()
        store.create("A", "a")
        store.create("B", "b")
        assert len(store.list_tasks()) == 2

    def test_list_by_status(self):
        store = TaskStore()
        store.create("A", "a")
        t = store.create("B", "b")
        store.update(t.id, status=TaskStatus.COMPLETED)
        pending = store.list_tasks(status=TaskStatus.PENDING)
        assert len(pending) == 1
        assert pending[0].subject == "A"

    def test_list_empty(self):
        store = TaskStore()
        assert store.list_tasks() == []

    def test_list_ordered_by_creation(self):
        store = TaskStore()
        store.create("First", "first")
        store.create("Second", "second")
        store.create("Third", "third")
        tasks = store.list_tasks()
        assert tasks[0].subject == "First"
        assert tasks[2].subject == "Third"

    def test_delete(self):
        store = TaskStore()
        t = store.create("Task", "Desc")
        assert store.delete(t.id)
        assert store.get(t.id) is None

    def test_delete_nonexistent(self):
        store = TaskStore()
        assert not store.delete("nonexistent")

    def test_incremental_ids(self):
        store = TaskStore()
        t1 = store.create("A", "a")
        t2 = store.create("B", "b")
        t3 = store.create("C", "c")
        assert int(t2.id) > int(t1.id)
        assert int(t3.id) > int(t2.id)

    def test_create_with_metadata(self):
        store = TaskStore()
        task = store.create("Task", "Desc", metadata={"priority": "high"})
        assert task.metadata == {"priority": "high"}

    def test_persistence_save_and_load(self, tmp_path):
        path = tmp_path / "tasks.json"
        store1 = TaskStore(persist_path=path)
        store1.create("Persist me", "desc")
        store1.create("Also persist", "desc2")

        store2 = TaskStore(persist_path=path)
        tasks = store2.list_tasks()
        assert len(tasks) == 2
        assert tasks[0].subject == "Persist me"
        assert tasks[1].subject == "Also persist"

    def test_persistence_preserves_ids(self, tmp_path):
        path = tmp_path / "tasks.json"
        store1 = TaskStore(persist_path=path)
        store1.create("A", "a")
        store1.create("B", "b")

        store2 = TaskStore(persist_path=path)
        # Next ID should continue from where store1 left off
        t3 = store2.create("C", "c")
        assert int(t3.id) > 2

    def test_persistence_empty_file(self, tmp_path):
        path = tmp_path / "tasks.json"
        # No file exists yet, should start empty
        store = TaskStore(persist_path=path)
        assert store.list_tasks() == []

    def test_persistence_status_preserved(self, tmp_path):
        path = tmp_path / "tasks.json"
        store1 = TaskStore(persist_path=path)
        t = store1.create("Task", "desc")
        store1.update(t.id, status=TaskStatus.COMPLETED)

        store2 = TaskStore(persist_path=path)
        loaded = store2.get(t.id)
        assert loaded is not None
        assert loaded.status == TaskStatus.COMPLETED

    def test_delete_persisted(self, tmp_path):
        path = tmp_path / "tasks.json"
        store1 = TaskStore(persist_path=path)
        t = store1.create("Temp", "desc")
        store1.delete(t.id)

        store2 = TaskStore(persist_path=path)
        assert store2.get(t.id) is None
        assert store2.list_tasks() == []

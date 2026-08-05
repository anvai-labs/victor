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

"""Tests for SWE-bench dataset loader."""

import json
import pytest
import tempfile
import hashlib
from unittest.mock import AsyncMock, MagicMock
from pathlib import Path

from victor.evaluation.swe_bench_loader import (
    SWEBenchConfig,
    SWEBenchInstance,
    SWEBenchLoader,
    SWEBenchWorkspaceManager,
    load_swe_bench_tasks,
    get_swe_bench_repos,
)
from victor.evaluation.protocol import BenchmarkType


class TestSWEBenchConfig:
    """Tests for SWEBenchConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SWEBenchConfig()
        assert config.repos is None
        assert config.max_tasks is None
        assert config.instance_ids is None
        assert config.exclude_ids is None
        assert config.shuffle is False
        assert config.seed == 42
        assert config.clone_timeout == 300
        assert config.shallow_clone is True
        assert config.use_cache is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = SWEBenchConfig(
            repos=["django/django"],
            max_tasks=10,
            instance_ids=["django__django-11583"],
            shuffle=True,
            seed=123,
            clone_timeout=600,
        )
        assert config.repos == ["django/django"]
        assert config.max_tasks == 10
        assert config.instance_ids == ["django__django-11583"]
        assert config.shuffle is True
        assert config.seed == 123
        assert config.clone_timeout == 600


class TestSWEBenchInstance:
    """Tests for SWEBenchInstance dataclass."""

    @pytest.fixture
    def sample_jsonl_data(self):
        """Sample JSONL data matching SWE-bench format."""
        return {
            "instance_id": "django__django-11583",
            "repo": "django/django",
            "base_commit": "abc123def456",
            "problem_statement": "Fix issue with QuerySet filtering",
            "hints_text": "Check the filter implementation",
            "patch": "--- a/django/db/models/query.py\n+++ b/django/db/models/query.py\n@@ -1 +1 @@\n-old\n+new",
            "test_patch": "def test_filter(): pass",
            "FAIL_TO_PASS": ["test_filter"],
            "PASS_TO_PASS": ["test_other"],
            "created_at": "2023-01-01T00:00:00Z",
            "version": "3.2",
        }

    def test_from_jsonl_line(self, sample_jsonl_data):
        """Test creating instance from JSONL data."""
        instance = SWEBenchInstance.from_jsonl_line(sample_jsonl_data)

        assert instance.instance_id == "django__django-11583"
        assert instance.repo == "django/django"
        assert instance.base_commit == "abc123def456"
        assert instance.problem_statement == "Fix issue with QuerySet filtering"
        assert instance.hints_text == "Check the filter implementation"
        assert "django/db/models/query.py" in instance.patch
        assert instance.test_patch == "def test_filter(): pass"
        assert instance.fail_to_pass == ["test_filter"]
        assert instance.pass_to_pass == ["test_other"]
        assert instance.version == "3.2"

    def test_to_benchmark_task(self, sample_jsonl_data):
        """Test converting instance to BenchmarkTask."""
        instance = SWEBenchInstance.from_jsonl_line(sample_jsonl_data)
        task = instance.to_benchmark_task()

        assert task.task_id == "django__django-11583"
        assert task.benchmark == BenchmarkType.SWE_BENCH
        assert task.language == "python"
        assert task.prompt == "Fix issue with QuerySet filtering"
        assert task.repo == "https://github.com/django/django.git"
        assert task.base_commit == "abc123def456"
        assert "Check the filter implementation" in task.hints
        assert task.patch == instance.patch
        assert task.test_code == "def test_filter(): pass"

    def test_estimate_difficulty_easy(self):
        """Test difficulty estimation for small patches."""
        data = {
            "instance_id": "test/1",
            "repo": "test/repo",
            "base_commit": "abc",
            "problem_statement": "Test",
            "hints_text": "",
            "patch": "--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new",
            "test_patch": "",
            "FAIL_TO_PASS": [],
            "PASS_TO_PASS": [],
            "created_at": "",
        }
        instance = SWEBenchInstance.from_jsonl_line(data)
        task = instance.to_benchmark_task()
        assert task.difficulty == "easy"

    def test_estimate_difficulty_hard(self):
        """Test difficulty estimation for large patches."""
        large_patch = "\n".join([f"line {i}" for i in range(150)])
        data = {
            "instance_id": "test/1",
            "repo": "test/repo",
            "base_commit": "abc",
            "problem_statement": "Test",
            "hints_text": "",
            "patch": large_patch,
            "test_patch": "",
            "FAIL_TO_PASS": [],
            "PASS_TO_PASS": [],
            "created_at": "",
        }
        instance = SWEBenchInstance.from_jsonl_line(data)
        task = instance.to_benchmark_task()
        assert task.difficulty == "hard"

    def test_get_category_django(self):
        """Test category detection for Django."""
        data = {
            "instance_id": "django__django-11583",
            "repo": "django/django",
            "base_commit": "abc",
            "problem_statement": "Test",
            "hints_text": "",
            "patch": "",
            "test_patch": "",
            "FAIL_TO_PASS": [],
            "PASS_TO_PASS": [],
            "created_at": "",
        }
        instance = SWEBenchInstance.from_jsonl_line(data)
        task = instance.to_benchmark_task()
        assert task.category == "web_framework"

    def test_get_category_pandas(self):
        """Test category detection for pandas."""
        data = {
            "instance_id": "pandas-dev__pandas-1234",
            "repo": "pandas-dev/pandas",
            "base_commit": "abc",
            "problem_statement": "Test",
            "hints_text": "",
            "patch": "",
            "test_patch": "",
            "FAIL_TO_PASS": [],
            "PASS_TO_PASS": [],
            "created_at": "",
        }
        instance = SWEBenchInstance.from_jsonl_line(data)
        task = instance.to_benchmark_task()
        assert task.category == "data_science"


class TestSWEBenchLoader:
    """Tests for SWEBenchLoader class."""

    @pytest.fixture
    def sample_jsonl_file(self):
        """Create a temporary JSONL file with sample data."""
        data = [
            {
                "instance_id": "django__django-11583",
                "repo": "django/django",
                "base_commit": "abc123",
                "problem_statement": "Fix QuerySet issue",
                "hints_text": "",
                "patch": "patch1",
                "test_patch": "",
                "FAIL_TO_PASS": [],
                "PASS_TO_PASS": [],
                "created_at": "",
            },
            {
                "instance_id": "psf__requests-1234",
                "repo": "psf/requests",
                "base_commit": "def456",
                "problem_statement": "Fix HTTP issue",
                "hints_text": "",
                "patch": "patch2",
                "test_patch": "",
                "FAIL_TO_PASS": [],
                "PASS_TO_PASS": [],
                "created_at": "",
            },
            {
                "instance_id": "flask__flask-5678",
                "repo": "pallets/flask",
                "base_commit": "ghi789",
                "problem_statement": "Fix route issue",
                "hints_text": "",
                "patch": "patch3",
                "test_patch": "",
                "FAIL_TO_PASS": [],
                "PASS_TO_PASS": [],
                "created_at": "",
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
            return Path(f.name)

    def test_load_from_file(self, sample_jsonl_file):
        """Test loading tasks from JSONL file."""
        loader = SWEBenchLoader()
        tasks = loader.load_from_file(sample_jsonl_file)

        assert len(tasks) == 3
        assert tasks[0].task_id == "django__django-11583"
        assert tasks[1].task_id == "psf__requests-1234"
        assert tasks[2].task_id == "flask__flask-5678"

    def test_load_with_max_tasks(self, sample_jsonl_file):
        """Test loading with max_tasks limit."""
        config = SWEBenchConfig(max_tasks=2)
        loader = SWEBenchLoader(config)
        tasks = loader.load_from_file(sample_jsonl_file)

        assert len(tasks) == 2

    def test_load_with_repo_filter(self, sample_jsonl_file):
        """Test loading with repo filter."""
        config = SWEBenchConfig(repos=["django/django"])
        loader = SWEBenchLoader(config)
        tasks = loader.load_from_file(sample_jsonl_file)

        assert len(tasks) == 1
        assert tasks[0].task_id == "django__django-11583"

    def test_load_with_instance_ids(self, sample_jsonl_file):
        """Test loading specific instance IDs."""
        config = SWEBenchConfig(instance_ids=["django__django-11583", "flask__flask-5678"])
        loader = SWEBenchLoader(config)
        tasks = loader.load_from_file(sample_jsonl_file)

        assert len(tasks) == 2
        task_ids = {t.task_id for t in tasks}
        assert "django__django-11583" in task_ids
        assert "flask__flask-5678" in task_ids

    def test_load_with_exclude_ids(self, sample_jsonl_file):
        """Test loading with excluded instance IDs."""
        config = SWEBenchConfig(exclude_ids=["psf__requests-1234"])
        loader = SWEBenchLoader(config)
        tasks = loader.load_from_file(sample_jsonl_file)

        assert len(tasks) == 2
        task_ids = {t.task_id for t in tasks}
        assert "psf__requests-1234" not in task_ids

    def test_load_instances_from_file(self, sample_jsonl_file):
        """Test loading raw instances from file."""
        loader = SWEBenchLoader()
        instances = loader.load_instances_from_file(sample_jsonl_file)

        assert len(instances) == 3
        assert isinstance(instances[0], SWEBenchInstance)
        assert instances[0].instance_id == "django__django-11583"

    def test_load_file_not_found(self):
        """Test loading from non-existent file."""
        loader = SWEBenchLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_from_file("/nonexistent/path.jsonl")

    def test_export_to_jsonl(self, sample_jsonl_file):
        """Test exporting tasks to JSONL."""
        loader = SWEBenchLoader()
        tasks = loader.load_from_file(sample_jsonl_file)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "exported.jsonl"
            loader.export_to_jsonl(tasks, output_path)

            assert output_path.exists()

            # Verify content
            with open(output_path) as f:
                lines = f.readlines()
                assert len(lines) == 3

                first_line = json.loads(lines[0])
                assert first_line["instance_id"] == "django__django-11583"


class TestSWEBenchWorkspaceManager:
    """Tests for SWEBenchWorkspaceManager class."""

    @pytest.mark.asyncio
    async def test_setup_repo_with_indexes_passes_graph_writer_settings(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """setup_repo_with_indexes should forward graph settings to CodebaseIndexFactory.create."""
        from types import SimpleNamespace
        from victor.evaluation.swe_bench_loader import SWEBenchWorkspaceManager

        graph_path = tmp_path / "graph.sqlite"
        workspace_base = tmp_path / "workspaces"
        cache_dir = tmp_path / "cache"

        manager = SWEBenchWorkspaceManager(
            workspace_base=workspace_base,
            cache_dir=cache_dir,
        )

        task = SWEBenchInstance.from_jsonl_line(
            {
                "instance_id": "django__django-123",
                "repo": "https://github.com/django/django",
                "base_commit": "abc123",
                "problem_statement": "Fix something",
                "hints_text": "",
                "patch": "",
                "test_patch": "",
                "FAIL_TO_PASS": [],
                "PASS_TO_PASS": [],
                "created_at": "",
            }
        ).to_benchmark_task()

        cache_path = cache_dir / hashlib.md5(task.repo.encode()).hexdigest()[:16]
        (cache_path / ".victor").mkdir(parents=True, exist_ok=True)

        settings = SimpleNamespace(
            codebase_graph_writer_mode="off",
            codebase_graph_store="sqlite",
            codebase_graph_path=str(graph_path),
        )

        mock_indexer = AsyncMock()
        mock_indexer.index_codebase = AsyncMock()
        mock_factory = MagicMock()
        mock_factory.create.return_value = mock_indexer
        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_factory

        monkeypatch.setattr("victor.config.settings.load_settings", lambda: settings)
        monkeypatch.setattr(
            "victor.evaluation.swe_bench_loader._resolve_graph_writer_mode",
            lambda _: "off",
        )
        monkeypatch.setattr(
            "victor.core.capability_registry.CapabilityRegistry.get_instance",
            lambda: mock_registry,
        )
        monkeypatch.setattr(
            "victor.context.project_context.ProjectContext.auto_generate",
            lambda *_args, **_kwargs: None,
        )

        result = await manager.setup_repo_with_indexes(task, force_reindex=True)

        assert result == cache_path
        mock_factory.create.assert_called_once_with(
            root_path=str(cache_path),
            use_embeddings=True,
            graph_writer_mode="off",
            graph_store_name="sqlite",
            graph_path=graph_path,
        )
        mock_indexer.index_codebase.assert_awaited_once()

    def test_init_creates_directories(self):
        """Test that initialization creates necessary directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_base = Path(tmpdir) / "workspaces"
            cache_dir = Path(tmpdir) / "cache"

            SWEBenchWorkspaceManager(
                workspace_base=workspace_base,
                cache_dir=cache_dir,
            )

            assert workspace_base.exists()
            assert cache_dir.exists()

    def test_cleanup_all_workspaces(self):
        """Test cleaning up all workspaces."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_base = Path(tmpdir) / "workspaces"
            workspace_base.mkdir()

            # Create some dummy workspaces
            (workspace_base / "task_1").mkdir()
            (workspace_base / "task_2").mkdir()

            manager = SWEBenchWorkspaceManager(workspace_base=workspace_base)
            manager.cleanup_all_workspaces()

            # Should have cleaned up and recreated empty directory
            assert workspace_base.exists()
            assert len(list(workspace_base.iterdir())) == 0


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_get_swe_bench_repos(self):
        """Test getting list of supported repos."""
        repos = get_swe_bench_repos()

        assert len(repos) > 0
        assert "django/django" in repos
        assert "psf/requests" in repos
        assert "pallets/flask" in repos

    def test_load_swe_bench_tasks(self):
        """Test convenience function for loading tasks."""
        data = [
            {
                "instance_id": "test/1",
                "repo": "test/repo",
                "base_commit": "abc",
                "problem_statement": "Test issue",
                "hints_text": "",
                "patch": "",
                "test_patch": "",
                "FAIL_TO_PASS": [],
                "PASS_TO_PASS": [],
                "created_at": "",
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
            file_path = Path(f.name)

        try:
            tasks = load_swe_bench_tasks(file_path)
            assert len(tasks) == 1
            assert tasks[0].task_id == "test/1"
        finally:
            file_path.unlink()

    def test_load_swe_bench_tasks_with_filters(self):
        """Test convenience function with filters."""
        data = [
            {
                "instance_id": "django/1",
                "repo": "django/django",
                "base_commit": "abc",
                "problem_statement": "Test",
                "hints_text": "",
                "patch": "",
                "test_patch": "",
                "FAIL_TO_PASS": [],
                "PASS_TO_PASS": [],
                "created_at": "",
            },
            {
                "instance_id": "flask/1",
                "repo": "pallets/flask",
                "base_commit": "def",
                "problem_statement": "Test",
                "hints_text": "",
                "patch": "",
                "test_patch": "",
                "FAIL_TO_PASS": [],
                "PASS_TO_PASS": [],
                "created_at": "",
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
            file_path = Path(f.name)

        try:
            tasks = load_swe_bench_tasks(
                file_path,
                max_tasks=1,
                repos=["django/django"],
            )
            assert len(tasks) == 1
            assert tasks[0].task_id == "django/1"
        finally:
            file_path.unlink()

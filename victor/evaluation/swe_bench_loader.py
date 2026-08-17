from __future__ import annotations

# Copyright 2025 Vijaykumar Singh
# SPDX-License-Identifier: Apache-2.0
"""SWE-bench dataset loader for agentic benchmark evaluation.

This module provides utilities for loading and processing the SWE-bench dataset,
which contains real GitHub issues with their corresponding patches.

SWE-bench Format:
Each task in SWE-bench has:
- instance_id: Unique identifier (e.g., "django__django-11583")
- repo: GitHub repository (e.g., "django/django")
- base_commit: The commit hash before the fix
- problem_statement: The issue description/problem
- hints_text: Optional hints for solving
- created_at: Timestamp
- patch: The gold standard patch that fixes the issue
- test_patch: Test code to verify the fix
- FAIL_TO_PASS: Tests that should pass after the fix
- PASS_TO_PASS: Tests that should still pass

Supported Variants:
- SWE-bench-full: Complete dataset (~2000 tasks)
- SWE-bench-lite: Curated subset (~300 tasks)
- SWE-bench-verified: Human-verified subset

Usage:
    from victor.evaluation.swe_bench_loader import (
        SWEBenchLoader,
        SWEBenchConfig,
        load_swe_bench_tasks,
    )

    # Load from local JSONL file
    loader = SWEBenchLoader()
    tasks = loader.load_from_file("swe-bench-lite.jsonl")

    # Load with filtering
    config = SWEBenchConfig(
        repos=["django/django", "psf/requests"],
        max_tasks=50,
    )
    tasks = loader.load_from_file("swe-bench.jsonl", config)

    # Download and load official dataset
    tasks = await loader.load_from_huggingface("princeton-nlp/SWE-bench_Lite")
"""

import asyncio
import hashlib
import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

from victor.evaluation.protocol import BenchmarkTask, BenchmarkType
from victor.config.settings import _resolve_graph_writer_mode

logger = logging.getLogger(__name__)


@dataclass
class SWEBenchConfig:
    """Configuration for SWE-bench dataset loading."""

    # Filtering options
    repos: Optional[list[str]] = None  # Filter by repository
    max_tasks: Optional[int] = None  # Limit number of tasks
    instance_ids: Optional[list[str]] = None  # Specific instances to load
    exclude_ids: Optional[list[str]] = None  # Instances to exclude

    # Task selection
    shuffle: bool = False  # Randomize task order
    seed: int = 42  # Random seed for reproducibility

    # Workspace settings
    workspace_base: Path = field(default_factory=lambda: Path("/tmp/swe_bench_workspaces"))
    clone_timeout: int = 300  # Timeout for git clone in seconds
    shallow_clone: bool = True  # Use shallow clones for speed

    # Cache settings
    cache_dir: Optional[Path] = None  # Cache directory for cloned repos
    use_cache: bool = True  # Whether to use cached repos


def _normalize_test_list(value: Any) -> list[str]:
    """Normalize a SWE-bench FAIL_TO_PASS / PASS_TO_PASS field to a list.

    The dataset (HuggingFace + JSONL) stores these as a **JSON-stringified**
    list (e.g. ``'["tests/test_a.py::test_a", "tests/test_b.py"]'``), not a
    native list. Storing the raw string broke the test command: ``list(str)``
    explodes it into single characters, and ``cmd.extend(chars)`` produced a
    malformed pytest invocation (one-char args) → 0 tests collected on every
    task. This always returns a proper ``list[str]`` of test node IDs.
    """
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return []


@dataclass
class SWEBenchInstance:
    """A single SWE-bench task instance with raw data."""

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    hints_text: str
    patch: str
    test_patch: str
    fail_to_pass: list[str]
    pass_to_pass: list[str]
    created_at: str
    version: str = ""
    environment_setup_commit: str = ""

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_jsonl_line(cls, data: dict[str, Any]) -> "SWEBenchInstance":
        """Create instance from a JSONL line."""
        return cls(
            instance_id=data.get("instance_id", ""),
            repo=data.get("repo", ""),
            base_commit=data.get("base_commit", ""),
            problem_statement=data.get("problem_statement", ""),
            hints_text=data.get("hints_text", ""),
            patch=data.get("patch", ""),
            test_patch=data.get("test_patch", ""),
            fail_to_pass=_normalize_test_list(data.get("FAIL_TO_PASS", [])),
            pass_to_pass=_normalize_test_list(data.get("PASS_TO_PASS", [])),
            created_at=data.get("created_at", ""),
            version=data.get("version", ""),
            environment_setup_commit=data.get("environment_setup_commit", ""),
            metadata={
                k: v
                for k, v in data.items()
                if k
                not in {
                    "instance_id",
                    "repo",
                    "base_commit",
                    "problem_statement",
                    "hints_text",
                    "patch",
                    "test_patch",
                    "FAIL_TO_PASS",
                    "PASS_TO_PASS",
                    "created_at",
                    "version",
                    "environment_setup_commit",
                }
            },
        )

    def to_benchmark_task(self) -> BenchmarkTask:
        """Convert to BenchmarkTask for use with AgenticBenchmarkRunner."""
        # Parse repo for difficulty estimation
        repo_difficulty = self._estimate_difficulty()

        # Build hints list
        hints = []
        if self.hints_text:
            hints.append(self.hints_text)

        return BenchmarkTask(
            task_id=self.instance_id,
            benchmark=BenchmarkType.SWE_BENCH,
            description=f"Fix issue in {self.repo}: {self.problem_statement[:100]}...",
            language="python",  # Most SWE-bench tasks are Python
            prompt=self.problem_statement,
            context_code="",  # Loaded from cloned repo
            test_code=self.test_patch,
            fail_to_pass=list(self.fail_to_pass),  # FAIL_TO_PASS test node IDs
            repo=f"https://github.com/{self.repo}.git",
            base_commit=self.base_commit,
            issue_text=self.problem_statement,
            hints=hints,
            solution=None,  # We don't provide gold solution
            patch=self.patch,
            difficulty=repo_difficulty,
            category=self._get_category(),
            tags=self._get_tags(),
            timeout_seconds=900,  # SWE-bench tasks need more time (15 min)
            # complexity_override removed - classifier now detects GitHub issue format
            # and automatically promotes to ACTION complexity (budget=50, timeout=600s)
        )

    def _estimate_difficulty(self) -> str:
        """Estimate task difficulty based on patch size and repo."""
        # Large patches are typically harder
        patch_lines = len(self.patch.split("\n"))
        if patch_lines > 100:
            return "hard"
        elif patch_lines > 30:
            return "medium"
        return "easy"

    def _get_category(self) -> str:
        """Determine category from repo name."""
        repo_lower = self.repo.lower()
        if "django" in repo_lower:
            return "web_framework"
        elif "flask" in repo_lower:
            return "web_framework"
        elif "requests" in repo_lower:
            return "networking"
        elif "pandas" in repo_lower or "numpy" in repo_lower:
            return "data_science"
        elif "scikit" in repo_lower:
            return "machine_learning"
        elif "matplotlib" in repo_lower or "seaborn" in repo_lower:
            return "visualization"
        elif "pytest" in repo_lower or "unittest" in repo_lower:
            return "testing"
        return "general"

    def _get_tags(self) -> list[str]:
        """Extract tags from instance."""
        tags = [self.repo.split("/")[-1]]  # Add repo name as tag
        if self.version:
            tags.append(f"version:{self.version}")
        return tags


class SWEBenchLoader:
    """Loader for SWE-bench dataset files."""

    def __init__(self, config: Optional[SWEBenchConfig] = None):
        """Initialize the loader.

        Args:
            config: Configuration for loading and filtering
        """
        self.config = config or SWEBenchConfig()

    def load_from_file(
        self,
        file_path: Path | str,
        config: Optional[SWEBenchConfig] = None,
    ) -> list[BenchmarkTask]:
        """Load tasks from a JSONL file.

        Args:
            file_path: Path to the JSONL file
            config: Optional config override

        Returns:
            List of BenchmarkTask objects
        """
        cfg = config or self.config
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"SWE-bench file not found: {file_path}")

        tasks = []
        for instance in self._iter_instances(file_path, cfg):
            tasks.append(instance.to_benchmark_task())

        logger.info(f"Loaded {len(tasks)} tasks from {file_path}")
        return tasks

    def load_instances_from_file(
        self,
        file_path: Path | str,
        config: Optional[SWEBenchConfig] = None,
    ) -> list[SWEBenchInstance]:
        """Load raw SWE-bench instances from a JSONL file.

        Args:
            file_path: Path to the JSONL file
            config: Optional config override

        Returns:
            List of SWEBenchInstance objects with raw data
        """
        cfg = config or self.config
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"SWE-bench file not found: {file_path}")

        instances = list(self._iter_instances(file_path, cfg))
        logger.info(f"Loaded {len(instances)} instances from {file_path}")
        return instances

    def _iter_instances(
        self,
        file_path: Path,
        config: SWEBenchConfig,
    ) -> Iterator[SWEBenchInstance]:
        """Iterate over instances from file with filtering."""
        import random

        instances = []
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    instance = SWEBenchInstance.from_jsonl_line(data)

                    # Apply filters
                    if not self._passes_filters(instance, config):
                        continue

                    instances.append(instance)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line: {e}")
                    continue

        # Shuffle if requested
        if config.shuffle:
            random.seed(config.seed)
            random.shuffle(instances)

        # Apply max_tasks limit
        if config.max_tasks:
            instances = instances[: config.max_tasks]

        yield from instances

    def _passes_filters(
        self,
        instance: SWEBenchInstance,
        config: SWEBenchConfig,
    ) -> bool:
        """Check if instance passes all filters."""
        # Filter by repo
        if config.repos:
            if instance.repo not in config.repos:
                return False

        # Filter by specific instance IDs
        if config.instance_ids:
            if instance.instance_id not in config.instance_ids:
                return False

        # Exclude specific IDs
        if config.exclude_ids:
            if instance.instance_id in config.exclude_ids:
                return False

        return True

    async def load_from_huggingface(
        self,
        dataset_name: str = "princeton-nlp/SWE-bench_Lite",
        split: str = "test",
        config: Optional[SWEBenchConfig] = None,
    ) -> list[BenchmarkTask]:
        """Load tasks from HuggingFace dataset.

        Args:
            dataset_name: HuggingFace dataset name
            split: Dataset split (test, dev, train)
            config: Optional config override

        Returns:
            List of BenchmarkTask objects
        """
        cfg = config or self.config

        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "datasets package required for HuggingFace loading. "
                "Install with: pip install datasets"
            )

        logger.info(f"Loading {dataset_name} ({split} split) from HuggingFace...")

        # Load dataset
        dataset = load_dataset(dataset_name, split=split)

        tasks = []
        for i, item in enumerate(dataset):
            # Apply max_tasks limit
            if cfg.max_tasks and i >= cfg.max_tasks:
                break

            instance = SWEBenchInstance.from_jsonl_line(dict(item))

            if not self._passes_filters(instance, cfg):
                continue

            tasks.append(instance.to_benchmark_task())

        logger.info(f"Loaded {len(tasks)} tasks from HuggingFace")
        return tasks

    def export_to_jsonl(
        self,
        tasks: list[BenchmarkTask],
        output_path: Path | str,
    ) -> None:
        """Export tasks to JSONL format for sharing.

        Args:
            tasks: Tasks to export
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for task in tasks:
                data = {
                    "instance_id": task.task_id,
                    "repo": task.repo,
                    "base_commit": task.base_commit,
                    "problem_statement": task.prompt,
                    "hints_text": task.hints[0] if task.hints else "",
                    "patch": task.patch or "",
                    "test_patch": task.test_code,
                    "FAIL_TO_PASS": [],
                    "PASS_TO_PASS": [],
                    "difficulty": task.difficulty,
                    "category": task.category,
                }
                f.write(json.dumps(data) + "\n")

        logger.info(f"Exported {len(tasks)} tasks to {output_path}")


class SWEBenchWorkspaceManager:
    """Manages workspaces for SWE-bench task execution."""

    def __init__(
        self,
        workspace_base: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
        shallow_clone: bool = True,
        clone_timeout: int = 300,
    ):
        """Initialize workspace manager.

        Args:
            workspace_base: Base directory for task workspaces
            cache_dir: Directory for caching cloned repositories
            shallow_clone: Use shallow clones for speed
            clone_timeout: Timeout for git operations
        """
        self.workspace_base = workspace_base or Path("/tmp/swe_bench_workspaces")
        if cache_dir is None:
            try:
                from victor.config.secure_paths import get_victor_dir

                cache_dir = get_victor_dir() / "swe_bench_cache"
            except ImportError:
                cache_dir = Path.home() / ".victor" / "swe_bench_cache"
        self.cache_dir = cache_dir
        self.shallow_clone = shallow_clone
        self.clone_timeout = clone_timeout

        self.workspace_base.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    async def setup_workspace(
        self,
        task: BenchmarkTask,
        use_cache: bool = True,
    ) -> Path:
        """Set up a workspace for a SWE-bench task.

        Args:
            task: The benchmark task
            use_cache: Whether to use cached repository

        Returns:
            Path to the prepared workspace
        """
        # Create unique workspace directory
        workspace_id = hashlib.md5(f"{task.task_id}_{task.base_commit}".encode()).hexdigest()[:12]
        workspace_dir = self.workspace_base / f"task_{workspace_id}"

        if workspace_dir.exists():
            shutil.rmtree(workspace_dir)
        workspace_dir.mkdir(parents=True)

        # Clone or copy from cache
        repo_dir = workspace_dir / "repo"
        if use_cache:
            cached_repo = await self._get_cached_repo(task)
            if cached_repo:
                await self._copy_from_cache(cached_repo, repo_dir)
            else:
                await self._clone_repo(task, repo_dir)
                await self._cache_repo(repo_dir, task)
        else:
            await self._clone_repo(task, repo_dir)

        # Checkout to base commit
        await self._checkout_commit(repo_dir, task.base_commit)

        # Write test file if provided
        if task.test_code:
            test_file = workspace_dir / "test_verification.py"
            test_file.write_text(task.test_code)

        # Write task info
        task_info = workspace_dir / "task_info.json"
        task_info.write_text(
            json.dumps(
                {
                    "task_id": task.task_id,
                    "repo": task.repo,
                    "base_commit": task.base_commit,
                    "description": task.description,
                },
                indent=2,
            )
        )

        logger.info(f"Workspace ready at {workspace_dir}")
        return workspace_dir

    async def _clone_repo(self, task: BenchmarkTask, target_dir: Path) -> None:
        """Clone repository to target directory."""
        if not task.repo:
            raise ValueError(f"Task {task.task_id} has no repository URL")

        cmd = ["git", "clone"]
        if self.shallow_clone:
            cmd.extend(["--depth", "100"])  # Need some history for checkout
        cmd.extend([task.repo, str(target_dir)])

        logger.info(f"Cloning {task.repo}...")
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.clone_timeout,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"Git clone failed: {stderr.decode()}")
        except asyncio.TimeoutError:
            proc.kill()
            raise RuntimeError(f"Git clone timed out after {self.clone_timeout}s")

    async def _checkout_commit(self, repo_dir: Path, commit: str) -> None:
        """Checkout specific commit in repository."""
        if not commit:
            return

        proc = await asyncio.create_subprocess_exec(
            "git",
            "checkout",
            commit,
            cwd=repo_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            # Try fetching the commit first (for shallow clones)
            fetch_proc = await asyncio.create_subprocess_exec(
                "git",
                "fetch",
                "--depth",
                "1",
                "origin",
                commit,
                cwd=repo_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await fetch_proc.communicate()

            # Try checkout again
            proc2 = await asyncio.create_subprocess_exec(
                "git",
                "checkout",
                commit,
                cwd=repo_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc2.communicate()
            if proc2.returncode != 0:
                logger.warning(f"Could not checkout {commit}: {stderr.decode()}")

    async def _get_cached_repo(self, task: BenchmarkTask) -> Optional[Path]:
        """Get cached repository path if available."""
        if not task.repo:
            return None

        # Create cache key from repo URL
        repo_hash = hashlib.md5(task.repo.encode()).hexdigest()[:16]
        cached_path = self.cache_dir / repo_hash

        if cached_path.exists():
            logger.debug(f"Using cached repo at {cached_path}")
            return cached_path
        return None

    async def _cache_repo(self, repo_dir: Path, task: BenchmarkTask) -> None:
        """Cache a cloned repository."""
        if not task.repo:
            return

        repo_hash = hashlib.md5(task.repo.encode()).hexdigest()[:16]
        cache_path = self.cache_dir / repo_hash

        if cache_path.exists():
            return  # Already cached

        try:
            shutil.copytree(repo_dir, cache_path)
            logger.debug(f"Cached repo to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to cache repo: {e}")

    async def _copy_from_cache(self, cached_repo: Path, target_dir: Path) -> None:
        """Copy repository from cache."""
        shutil.copytree(cached_repo, target_dir)

    async def setup_repo_with_indexes(
        self,
        task: BenchmarkTask,
        force_reindex: bool = False,
    ) -> Path:
        """Clone repo and build indexes (graph, embeddings, project.db).

        This is Phase 1 of the two-phase benchmark approach:
        - Clone repo to cache (if not already)
        - Build code indexes in repo's .victor/ directory
        - Return path to indexed repo

        Args:
            task: The benchmark task
            force_reindex: Force rebuild of indexes even if they exist

        Returns:
            Path to the cached repo with indexes
        """
        if not task.repo:
            raise ValueError(f"Task {task.task_id} has no repository URL")

        repo_hash = hashlib.md5(task.repo.encode()).hexdigest()[:16]
        cache_path = self.cache_dir / repo_hash

        # Clone if not cached
        if not cache_path.exists():
            logger.info(f"Cloning {task.repo} to cache...")
            await self._clone_repo(task, cache_path)

        # Check if already indexed
        victor_dir = cache_path / ".victor"
        index_marker = victor_dir / "indexed_at"
        embeddings_dir = victor_dir / "embeddings"

        if not force_reindex and index_marker.exists():
            logger.info(f"Repo already indexed: {cache_path}")
            return cache_path

        # Check if embeddings exist on disk even without marker
        # (e.g., marker was deleted but embeddings persist)
        if not force_reindex and embeddings_dir.exists() and any(embeddings_dir.iterdir()):
            logger.info(f"Embeddings exist at {embeddings_dir}, restoring index marker")
            victor_dir.mkdir(parents=True, exist_ok=True)
            from datetime import datetime, timezone

            index_marker.write_text(datetime.now(timezone.utc).isoformat())
            return cache_path

        # Build indexes
        logger.info(f"Building indexes for {task.repo}...")
        victor_dir.mkdir(parents=True, exist_ok=True)
        from victor.config.settings import load_settings

        settings = load_settings()
        graph_writer_mode = _resolve_graph_writer_mode(settings)
        # Honor the per-repo backend marker (.victor/graph_backend) with the
        # global setting as fallback — this path previously read only the
        # global setting, so the marker (whose whole purpose is flipping a
        # single repo without changing global settings) was silently ignored
        # by benchmark/eval indexing.
        from victor.storage.graph.registry import resolve_graph_backend

        graph_store_name = resolve_graph_backend(
            cache_path,
            default=getattr(settings, "codebase_graph_store", "sqlite"),
        )
        graph_path = getattr(settings, "codebase_graph_path", None)

        try:
            # Run indexer on the repo with embeddings enabled for semantic search
            from victor.core.capability_registry import CapabilityRegistry
            from victor.framework.vertical_protocols import CodebaseIndexFactoryProtocol

            factory = CapabilityRegistry.get_instance().get(CodebaseIndexFactoryProtocol)
            if factory is None:
                raise ImportError("Codebase indexing not available")
            indexer = factory.create(
                root_path=str(cache_path),
                use_embeddings=True,
                graph_writer_mode=str(graph_writer_mode),
                graph_store_name=graph_store_name,
                graph_path=Path(graph_path) if graph_path else None,
            )
            await indexer.index_codebase()

            # Auto-generate init.md for project context
            from victor.context.project_context import ProjectContext

            init_path = ProjectContext.auto_generate(str(cache_path))
            if init_path:
                logger.info(f"Auto-generated project context: {init_path}")

            # Mark as indexed
            from datetime import datetime, timezone

            index_marker.write_text(datetime.now(timezone.utc).isoformat())

            logger.info(f"Indexed {task.repo} successfully")
        except Exception as e:
            logger.warning(f"Indexing failed (will work without): {e}")

        return cache_path

    def is_repo_indexed(self, task: BenchmarkTask) -> bool:
        """Check if repo has been indexed."""
        if not task.repo:
            return False

        repo_hash = hashlib.md5(task.repo.encode()).hexdigest()[:16]
        cache_path = self.cache_dir / repo_hash
        index_marker = cache_path / ".victor" / "indexed_at"

        return index_marker.exists()

    def get_cached_repo_path(self, task: BenchmarkTask) -> Optional[Path]:
        """Get path to cached repo (for execution phase)."""
        if not task.repo:
            return None

        repo_hash = hashlib.md5(task.repo.encode()).hexdigest()[:16]
        cache_path = self.cache_dir / repo_hash

        if cache_path.exists():
            return cache_path
        return None

    async def cleanup_workspace(self, workspace_dir: Path) -> None:
        """Clean up a task workspace."""
        try:
            if workspace_dir.exists():
                shutil.rmtree(workspace_dir)
        except Exception as e:
            logger.warning(f"Failed to cleanup workspace: {e}")

    def cleanup_all_workspaces(self) -> None:
        """Clean up all task workspaces."""
        try:
            if self.workspace_base.exists():
                shutil.rmtree(self.workspace_base)
                self.workspace_base.mkdir(parents=True)
            logger.info("Cleaned up all workspaces")
        except Exception as e:
            logger.warning(f"Failed to cleanup workspaces: {e}")


# Convenience functions


def load_swe_bench_tasks(
    file_path: Path | str,
    max_tasks: Optional[int] = None,
    repos: Optional[list[str]] = None,
) -> list[BenchmarkTask]:
    """Load SWE-bench tasks from a JSONL file.

    Args:
        file_path: Path to JSONL file
        max_tasks: Maximum number of tasks to load
        repos: Filter by repository names

    Returns:
        List of BenchmarkTask objects
    """
    config = SWEBenchConfig(
        max_tasks=max_tasks,
        repos=repos,
    )
    loader = SWEBenchLoader(config)
    return loader.load_from_file(file_path)


async def setup_swe_bench_workspace(
    task: BenchmarkTask,
    workspace_base: Optional[Path] = None,
) -> Path:
    """Set up workspace for a SWE-bench task.

    Args:
        task: The benchmark task
        workspace_base: Base directory for workspaces

    Returns:
        Path to prepared workspace
    """
    manager = SWEBenchWorkspaceManager(workspace_base=workspace_base)
    return await manager.setup_workspace(task)


def get_swe_bench_repos() -> list[str]:
    """Get list of supported SWE-bench repositories."""
    return [
        "django/django",
        "psf/requests",
        "pallets/flask",
        "pydata/xarray",
        "pytest-dev/pytest",
        "scikit-learn/scikit-learn",
        "sphinx-doc/sphinx",
        "sympy/sympy",
        "pylint-dev/pylint",
        "matplotlib/matplotlib",
        "astropy/astropy",
        "mwaskom/seaborn",
        "pydata/pandas",
    ]

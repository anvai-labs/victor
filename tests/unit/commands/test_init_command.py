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

"""Tests for the init command."""

from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from rich.console import Console

import victor.ui.commands.init as init_module
from victor.ui.commands.init_content import (
    count_architecture_patterns,
    ensure_architecture_evidence_section,
    ensure_architecture_patterns_section,
    ensure_quality_baseline_section,
)


def test_init_ccg_reports_project_graph_stats_without_name_error(tmp_path, monkeypatch):
    """CCG init should report graph stats from the project graph store."""
    output = StringIO()
    test_console = Console(file=output, force_terminal=False, color_system=None)
    monkeypatch.setattr(init_module, "console", test_console)

    project_paths = SimpleNamespace(
        global_victor_dir=tmp_path / "global-victor",
        project_context_file=tmp_path / "init.md",
        project_victor_dir=tmp_path / ".victor",
    )
    monkeypatch.setattr(init_module, "get_project_paths", lambda: project_paths)
    monkeypatch.setattr(
        init_module,
        "get_database",
        lambda: SimpleNamespace(db_path=project_paths.global_victor_dir / "victor.db"),
    )
    monkeypatch.setattr(
        init_module,
        "get_project_database",
        lambda: SimpleNamespace(db_path=project_paths.project_victor_dir / "project.db"),
    )
    monkeypatch.setattr(init_module, "latest_mtime", lambda _root: 0.0)
    monkeypatch.setattr(init_module, "_generate_init_content", lambda **_kwargs: "# init\n")
    # _gather_graph_context is a sync function — use MagicMock, not AsyncMock
    monkeypatch.setattr(init_module, "_gather_graph_context", MagicMock(return_value=None))

    fake_graph_store = SimpleNamespace(stats=AsyncMock(return_value={"nodes": 12, "edges": 34}))
    create_graph_store = MagicMock(return_value=fake_graph_store)

    fake_graph_rag = ModuleType("victor.core.graph_rag")
    fake_graph_rag.GraphIndexConfig = lambda **kwargs: SimpleNamespace(**kwargs)

    class FakePipeline:
        def __init__(self, graph_store, config) -> None:
            self.graph_store = graph_store
            self.config = config

        async def index_repository(
            self, root_path=None, progress_callback=None, status_callback=None
        ):
            return SimpleNamespace(
                files_processed=1,
                files_deleted=0,
                files_unchanged=2,
                nodes_created=3,
                edges_created=4,
            )

    fake_graph_rag.GraphIndexingPipeline = FakePipeline
    # Prevent real run_indexing_with_lock from being imported so the simple
    # pipeline.index_repository() path is used (no file-lock acquisition in tests).
    fake_graph_rag_indexing = ModuleType("victor.core.graph_rag.indexing")
    fake_graph_rag_indexing.run_indexing_with_lock = None

    with patch.dict(
        sys.modules,
        {
            "victor.core.graph_rag": fake_graph_rag,
            "victor.core.graph_rag.indexing": fake_graph_rag_indexing,
        },
    ):
        with patch("victor.storage.graph.create_graph_store", create_graph_store):
            with patch("victor.ui.commands.utils.setup_logging"):
                init_module.init(
                    ctx=SimpleNamespace(invoked_subcommand=None),
                    update=False,
                    force=False,
                    learn=False,
                    index=False,
                    deep=False,
                    agentic=False,
                    quick=False,
                    ccg=True,
                    symlinks=False,
                    config_only=False,
                    interactive=False,
                    local=False,
                    airgapped=False,
                    wizard=False,
                    provider=None,
                    model=None,
                    log_level=None,
                )

    # "auto", not "sqlite": init must resolve the per-repo
    # <project>/.victor/graph_backend marker the read paths already honor.
    # Pinning a concrete backend here is what let init and the query tools
    # disagree about which store holds the index.
    create_graph_store.assert_called_once_with("auto", project_path=Path.cwd())
    fake_graph_store.stats.assert_awaited_once()

    rendered = output.getvalue()
    # The index-updated branch reports "Graph index updated" + a "N nodes, M
    # edges in database" line (the legacy "CCG index updated" / "N total nodes"
    # wording was deliberately retired — see init.py).
    assert "Graph index updated" in rendered
    assert "12 nodes, 34 edges in database" in rendered
    assert "CCG indexing skipped" not in rendered


def test_init_ccg_lock_timeout_uses_existing_graph_snapshot(tmp_path, monkeypatch):
    """Lock timeout should degrade to current graph stats instead of a hard-looking skip."""
    output = StringIO()
    test_console = Console(file=output, force_terminal=False, color_system=None)
    monkeypatch.setattr(init_module, "console", test_console)

    project_paths = SimpleNamespace(
        global_victor_dir=tmp_path / "global-victor",
        project_context_file=tmp_path / "init.md",
        project_victor_dir=tmp_path / ".victor",
    )
    monkeypatch.setattr(init_module, "get_project_paths", lambda: project_paths)
    monkeypatch.setattr(
        init_module,
        "get_database",
        lambda: SimpleNamespace(db_path=project_paths.global_victor_dir / "victor.db"),
    )
    monkeypatch.setattr(
        init_module,
        "get_project_database",
        lambda: SimpleNamespace(db_path=project_paths.project_victor_dir / "project.db"),
    )
    monkeypatch.setattr(init_module, "latest_mtime", lambda _root: 0.0)
    monkeypatch.setattr(init_module, "_generate_init_content", lambda **_kwargs: "# init\n")
    monkeypatch.setattr(init_module, "_gather_graph_context", AsyncMock(return_value=None))

    fake_graph_store = SimpleNamespace(stats=AsyncMock(return_value={"nodes": 120, "edges": 340}))
    create_graph_store = MagicMock(return_value=fake_graph_store)

    fake_graph_rag = ModuleType("victor.core.graph_rag")
    fake_graph_rag.GraphIndexConfig = lambda **kwargs: SimpleNamespace(**kwargs)
    fake_graph_rag_indexing = ModuleType("victor.core.graph_rag.indexing")

    class FakePipeline:
        def __init__(self, graph_store, config) -> None:
            self.graph_store = graph_store
            self.config = config

        async def index_repository(self):
            raise AssertionError("index_repository should not be called after lock timeout")

    fake_graph_rag.GraphIndexingPipeline = FakePipeline

    async def _raise_timeout(*_args, **_kwargs):
        raise TimeoutError("Failed to acquire index lock for /tmp/project after 5 seconds")

    fake_graph_rag_indexing.run_indexing_with_lock = _raise_timeout

    with patch.dict(
        sys.modules,
        {
            "victor.core.graph_rag": fake_graph_rag,
            "victor.core.graph_rag.indexing": fake_graph_rag_indexing,
        },
    ):
        with patch("victor.storage.graph.create_graph_store", create_graph_store):
            with patch("victor.ui.commands.utils.setup_logging"):
                init_module.init(
                    ctx=SimpleNamespace(invoked_subcommand=None),
                    update=False,
                    force=False,
                    learn=False,
                    index=False,
                    deep=False,
                    agentic=False,
                    quick=False,
                    ccg=True,
                    symlinks=False,
                    config_only=False,
                    interactive=False,
                    local=False,
                    airgapped=False,
                    wizard=False,
                    provider=None,
                    model=None,
                    log_level=None,
                )

    rendered = output.getvalue()
    assert "CCG refresh deferred" in rendered
    assert "Using existing graph snapshot (120 total nodes, 340 total edges)" in rendered


async def test_generate_init_content_async_closes_temporary_agent(monkeypatch):
    """Enhanced init should close the temporary synthesis agent."""
    mock_agent = SimpleNamespace(close=AsyncMock())
    mock_generator = AsyncMock(return_value="# generated\n")

    monkeypatch.setattr(init_module, "_create_init_agent", AsyncMock(return_value=mock_agent))
    monkeypatch.setattr(
        init_module,
        "load_codebase_analyzer_attr",
        lambda name: mock_generator if name == "generate_enhanced_init_md" else None,
    )

    result = await init_module._generate_init_content_async(
        mode="enhanced",
        use_llm=True,
        include_conversations=False,
        provider="zai-coding",
        model=None,
    )

    assert result == "# generated\n"
    mock_agent.close.assert_awaited_once()
    assert mock_generator.await_count == 1


async def test_create_init_agent_uses_lightweight_profile_provider():
    """CLI init should not build a full orchestrator just to synthesize init.md."""
    fake_provider = SimpleNamespace(close=AsyncMock(), name="zai")
    mock_create = MagicMock(return_value=fake_provider)
    profile = SimpleNamespace(
        provider="zai",
        model="glm-5.1",
        temperature=0.7,
        max_tokens=8192,
        __pydantic_extra__={"coding_plan": True},
    )
    mock_settings = SimpleNamespace(
        default_provider=None,
        default_model=None,
        provider=SimpleNamespace(default_provider=None, default_model=None),
        load_profiles=lambda: {"zai-coding": profile},
        get_provider_settings=lambda provider_name, overrides, account_name=None: {
            "base_url": "https://api.z.ai/api/coding/paas/v4/",
            "coding_plan": overrides.get("coding_plan", False),
            "timeout": 120,
        },
    )
    with patch("victor.config.settings.load_settings", return_value=mock_settings):
        with patch("victor.providers.registry.ProviderRegistry.create", mock_create):
            agent = await init_module._create_init_agent("zai-coding")

    assert agent.provider is fake_provider
    assert agent.provider_name == "zai"
    assert agent.model == "glm-5.1"
    assert agent.max_tokens == 8192
    mock_create.assert_called_once_with(
        "zai",
        base_url="https://api.z.ai/api/coding/paas/v4/",
        coding_plan=True,
        # Cloud providers get the 300s init-synthesis budget (_init_provider_timeout);
        # local providers get 600s. The old flat 120s constant was retired.
        timeout=300,
        max_retries=0,
    )


def test_count_architecture_patterns_reads_markdown_section() -> None:
    content = """# init.md

## Architecture Patterns

- **Facade**: One public entry point
- **Registry**: Pluggable extensions

## Development Commands
"""

    assert count_architecture_patterns(content) == 2


def test_ensure_architecture_patterns_section_adds_graph_fallback() -> None:
    content = """# init.md

## Development Commands

```bash
make test
```
"""
    graph_context = {
        "has_ccg": True,
        "stats": {"ccg_edges": 42},
        "patterns": {
            "protocol": 5,
            "inheritance": 7,
        },
    }

    enhanced = ensure_architecture_patterns_section(content, graph_context)

    assert "## Architecture Patterns" in enhanced
    assert "Registry/plugin extensibility" not in enhanced
    assert "Protocol/interface contracts" in enhanced
    assert count_architecture_patterns(enhanced) >= 3


def test_ensure_architecture_evidence_section_adds_graph_backed_summary() -> None:
    content = """# init.md

## Architecture Patterns

- **Facade**: One entry point
"""
    graph_context = {
        "has_ccg": True,
        "stats": {
            "total_nodes": 1200,
            "total_edges": 3400,
            "ccg_edges": 240,
        },
        "patterns": {
            "protocol": 6,
            "inheritance": 9,
        },
        "complexity": {
            "avg_branching": 2.83,
        },
    }

    enhanced = ensure_architecture_evidence_section(content, graph_context)

    assert "## Architecture Evidence" in enhanced
    assert "Graph scale" in enhanced
    # REGISTERS/DECORATES evidence lines are gone with the synthetic edges that
    # fed them — they were fabricated, not observed. Real relationship evidence
    # (IMPLEMENTS from language handlers, INHERITS) still renders.
    assert "Registry/plugin evidence" not in enhanced
    assert "Decorator evidence" not in enhanced
    assert "Protocol/interface evidence" in enhanced
    assert "Protocol/interface evidence" in enhanced
    assert "Statement-level flow evidence" in enhanced


def test_ensure_quality_baseline_section_adds_repository_working_agreements() -> None:
    content = """# init.md

## Development Commands

```bash
make test
```
"""

    enhanced = ensure_quality_baseline_section(content)

    assert "## Repository Working Agreements" in enhanced
    assert "Preserve user work in git" in enhanced
    assert "Match local naming and style" in enhanced
    assert "Validate close to the change" in enhanced
    assert enhanced.index("## Repository Working Agreements") < enhanced.index(
        "## Development Commands"
    )


def test_ensure_quality_baseline_section_extends_existing_guidelines() -> None:
    content = """# init.md

## Repository Guidelines

- **Custom rule**: Keep this project-specific rule.
"""

    enhanced = ensure_quality_baseline_section(content)

    assert "Custom rule" in enhanced
    assert "Preserve user work in git" in enhanced
    assert enhanced.count("Preserve user work in git") == 1


def test_init_reports_architecture_patterns_from_markdown_section(tmp_path, monkeypatch):
    output = StringIO()
    test_console = Console(file=output, force_terminal=False, color_system=None)
    monkeypatch.setattr(init_module, "console", test_console)

    project_paths = SimpleNamespace(
        global_victor_dir=tmp_path / "global-victor",
        project_context_file=tmp_path / "init.md",
        project_victor_dir=tmp_path / ".victor",
    )
    monkeypatch.setattr(init_module, "get_project_paths", lambda: project_paths)
    monkeypatch.setattr(
        init_module,
        "get_database",
        lambda: SimpleNamespace(db_path=project_paths.global_victor_dir / "victor.db"),
    )
    monkeypatch.setattr(
        init_module,
        "get_project_database",
        lambda: SimpleNamespace(db_path=project_paths.project_victor_dir / "project.db"),
    )
    monkeypatch.setattr(init_module, "latest_mtime", lambda _root: 0.0)

    generated = """# init.md

## Architecture Patterns

- **Facade**: One entry point
- **Registry**: Shared extension lookup
"""
    monkeypatch.setattr(init_module, "_generate_init_content", lambda **_kwargs: generated)

    with patch("victor.ui.commands.utils.setup_logging"):
        init_module.init(
            ctx=SimpleNamespace(invoked_subcommand=None),
            update=False,
            force=False,
            learn=False,
            index=False,
            deep=False,
            agentic=False,
            quick=True,
            ccg=True,
            symlinks=False,
            config_only=False,
            interactive=False,
            local=False,
            airgapped=False,
            wizard=False,
            provider=None,
            model=None,
            log_level=None,
        )

    rendered = output.getvalue()
    assert "Found 2 architecture patterns" in rendered
    written = project_paths.project_context_file.read_text(encoding="utf-8")
    assert "## Repository Working Agreements" in written
    assert "Preserve user work in git" in written


def test_init_enriches_content_with_architecture_evidence(tmp_path, monkeypatch):
    output = StringIO()
    test_console = Console(file=output, force_terminal=False, color_system=None)
    monkeypatch.setattr(init_module, "console", test_console)

    project_paths = SimpleNamespace(
        global_victor_dir=tmp_path / "global-victor",
        project_context_file=tmp_path / "init.md",
        project_victor_dir=tmp_path / ".victor",
    )
    monkeypatch.setattr(init_module, "get_project_paths", lambda: project_paths)
    monkeypatch.setattr(
        init_module,
        "get_database",
        lambda: SimpleNamespace(db_path=project_paths.global_victor_dir / "victor.db"),
    )
    monkeypatch.setattr(
        init_module,
        "get_project_database",
        lambda: SimpleNamespace(db_path=project_paths.project_victor_dir / "project.db"),
    )
    monkeypatch.setattr(init_module, "latest_mtime", lambda _root: 0.0)
    monkeypatch.setattr(
        init_module,
        "_generate_init_content",
        lambda **_kwargs: "# init.md\n\n## Architecture Patterns\n\n- **Facade**: One entry point\n",
    )
    # _gather_graph_context is a sync function — use MagicMock, not AsyncMock
    monkeypatch.setattr(
        init_module,
        "_gather_graph_context",
        MagicMock(
            return_value={
                "has_ccg": True,
                "stats": {
                    "total_nodes": 100,
                    "total_edges": 250,
                    "ccg_edges": 50,
                },
                "patterns": {
                    "protocol": 3,
                    "inheritance": 4,
                },
                "complexity": {"avg_branching": 2.5},
            }
        ),
    )

    with patch("victor.ui.commands.utils.setup_logging"):
        init_module.init(
            ctx=SimpleNamespace(invoked_subcommand=None),
            update=False,
            force=False,
            learn=False,
            index=False,
            deep=True,
            agentic=False,
            quick=False,
            ccg=False,
            symlinks=False,
            config_only=False,
            interactive=False,
            local=False,
            airgapped=False,
            wizard=False,
            provider=None,
            model=None,
            log_level=None,
        )

    written = project_paths.project_context_file.read_text(encoding="utf-8")
    assert "## Architecture Evidence" in written
    assert "Registry/plugin evidence" not in written

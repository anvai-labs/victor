"""WorkspaceContextBuilder and ProjectContextDiscovery (public prompt-builder API)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from victor.agent.prompting.builder import (
    ContextFile,
    ProjectContext,
    ProjectContextDiscovery,
    PromptBudget,
    WorkspaceContextBuilder,
    _classify_scope,
    _run_git,
    _truncate,
)

BOUNDARY = "# --- DYNAMIC CONTEXT BELOW ---"


def _context(tmp_path: Path, **overrides) -> ProjectContext:
    values = {"cwd": tmp_path, "current_date": "2026-10-22"}
    values.update(overrides)
    return ProjectContext(**values)


class TestTruncate:
    def test_short_text_is_unchanged(self):
        assert _truncate("abc", 3) == "abc"

    def test_long_text_is_cut_with_a_labelled_notice(self):
        assert _truncate("abcdef", 3, "notes") == "abc\n... [notes truncated to 3 chars]"


class TestWorkspaceContextBuilder:
    def test_sections_are_assembled_in_documented_order(self, tmp_path):
        ctx = _context(
            tmp_path,
            is_git_repo=True,
            branch_name="develop",
            git_status="## develop\n M a.py",
            git_diff_summary=" a.py | 2 +-",
            instruction_files=[
                ContextFile(path="AGENTS.md", content="Be terse.", scope="workspace")
            ],
        )
        prompt = (
            WorkspaceContextBuilder()
            .with_output_style("concise", "Answer briefly.")
            .with_base_prompt("You are a coding assistant.")
            .with_tools(["git", "filesystem"])
            .with_os_info("Linux", "6.18")
            .with_model_info("model-x")
            .with_project_context(ctx)
            .append_section("# Extra\nfooter")
            .build()
        )
        markers = [
            "# Output Style: concise",
            "You are a coding assistant.",
            "Tools available: filesystem, git",  # sorted
            "# Environment",
            BOUNDARY,
            "# Project Instructions",
            "# Git Context",
            "# Extra",
        ]
        positions = [prompt.index(m) for m in markers]
        assert positions == sorted(positions)
        for line in (
            "Model: model-x",
            f"Working directory: {tmp_path}",
            "Current date: 2026-10-22",
            "Platform: Linux 6.18",
            "Git repository: yes",
            "Git branch: develop",
            "## AGENTS.md (scope: workspace)\nBe terse.",
            "## Git Status\n```\n## develop\n M a.py\n```",
            "## Git Diff Summary\n```\n a.py | 2 +-\n```",
        ):
            assert line in prompt

    def test_empty_builder_emits_only_the_boundary(self):
        assert WorkspaceContextBuilder().build().strip() == BOUNDARY

    def test_output_style_needs_both_name_and_prompt(self):
        assert "Output Style" not in WorkspaceContextBuilder().with_output_style("x", "").build()

    def test_non_git_context_omits_git_lines_and_sections(self, tmp_path):
        prompt = WorkspaceContextBuilder().with_project_context(_context(tmp_path)).build()
        assert "Working directory" in prompt
        assert "Git repository" not in prompt and "# Git Context" not in prompt
        assert "# Project Instructions" not in prompt

    def test_per_file_and_total_instruction_budgets(self, tmp_path):
        files = [ContextFile(path=f"f{i}.md", content="x" * 50, scope="user") for i in range(4)]
        budget = PromptBudget(max_per_file_chars=30, max_total_instruction_chars=70)
        prompt = (
            WorkspaceContextBuilder(budget)
            .with_project_context(_context(tmp_path, instruction_files=files))
            .build()
        )
        assert "[f0.md truncated to 30 chars]" in prompt
        assert "instruction budget exhausted, remaining files skipped" in prompt
        assert "## f3.md" not in prompt

    def test_git_output_is_budget_truncated(self, tmp_path):
        budget = PromptBudget(max_git_status_chars=5, max_git_diff_chars=4)
        ctx = _context(tmp_path, git_status="s" * 20, git_diff_summary="d" * 20)
        prompt = WorkspaceContextBuilder(budget).with_project_context(ctx).build()
        assert "[git status truncated to 5 chars]" in prompt
        assert "[git diff truncated to 4 chars]" in prompt


class TestClassifyScope:
    def test_scopes(self, tmp_path):
        repo = tmp_path / "repo"
        (repo / ".git").mkdir(parents=True)
        work = repo / "pkg"
        work.mkdir()
        assert _classify_scope(work, work) == "workspace"
        assert _classify_scope(repo, work) == "project"
        assert _classify_scope(tmp_path, work) == "user"


class TestRunGit:
    async def test_nonzero_exit_returns_none(self, tmp_path):
        assert await _run_git(["git", "rev-parse", "--abbrev-ref", "HEAD"], tmp_path) is None

    async def test_spawn_failure_returns_none(self, tmp_path):
        assert await _run_git(["git", "status"], tmp_path / "missing") is None


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
class TestProjectContextDiscovery:
    async def test_git_workspace_is_discovered(self, tmp_path):
        def git(*args):
            subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

        git("init", "-q", "-b", "trunk")
        git("config", "user.email", "t@example.com")
        git("config", "user.name", "t")
        (tmp_path / "AGENTS.md").write_text("Repo rules.", encoding="utf-8")
        git("add", "AGENTS.md")
        git("commit", "-q", "-m", "init")
        (tmp_path / "AGENTS.md").write_text("Repo rules, edited.", encoding="utf-8")

        ctx = await ProjectContextDiscovery.discover(tmp_path, current_date="2026-10-22")

        assert ctx.is_git_repo and ctx.branch_name == "trunk"
        assert ctx.current_date == "2026-10-22"
        assert "AGENTS.md" in (ctx.git_status or "")
        assert "AGENTS.md" in (ctx.git_diff_summary or "")
        mine = [f for f in ctx.instruction_files if Path(f.path).parent == tmp_path]
        assert [(Path(f.path).name, f.content, f.scope) for f in mine] == [
            ("AGENTS.md", "Repo rules, edited.", "workspace")
        ]

    async def test_non_git_directory_skips_git_probes(self, tmp_path):
        ctx = await ProjectContextDiscovery.discover(tmp_path)
        assert not ctx.is_git_repo
        assert ctx.branch_name is ctx.git_status is ctx.git_diff_summary is None
        assert len(ctx.current_date) == 10  # ISO date default

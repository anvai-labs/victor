# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for ProjectContext.auto_generate and project context loading."""

import pytest
from pathlib import Path

from victor.context.project_context import ProjectContext


class TestAutoGenerate:
    """Tests for ProjectContext.auto_generate classmethod."""

    def test_creates_init_md_from_readme(self, tmp_path):
        """Auto-generate creates init.md with README summary."""
        (tmp_path / "README.md").write_text(
            "# My Project\n\nA Python library for data processing.\n\n## Install\n"
        )
        (tmp_path / "src").mkdir()
        (tmp_path / "tests").mkdir()
        (tmp_path / "pyproject.toml").write_text("[project]\nname='my-project'\n")

        result = ProjectContext.auto_generate(str(tmp_path))

        assert result is not None
        assert result.exists()
        content = result.read_text()
        assert "A Python library for data processing" in content
        assert "src/" in content
        assert "tests/" in content
        assert "Python (pyproject.toml)" in content

    def test_skips_if_init_md_exists(self, tmp_path):
        """Auto-generate is idempotent — skips existing init.md."""
        victor_dir = tmp_path / ".victor"
        victor_dir.mkdir(exist_ok=True)
        init_file = victor_dir / "init.md"
        init_file.write_text("# Existing context\n")

        result = ProjectContext.auto_generate(str(tmp_path))

        assert result is None
        assert init_file.read_text() == "# Existing context\n"

    def test_creates_victor_directory(self, tmp_path):
        """Auto-generate creates .victor/ directory if missing."""
        (tmp_path / "README.md").write_text("# Test\n\nSome project.\n")

        result = ProjectContext.auto_generate(str(tmp_path))

        assert result is not None
        assert (tmp_path / ".victor").is_dir()

    def test_compact_output_under_500_tokens(self, tmp_path):
        """Generated init.md should be compact (~500 tokens)."""
        (tmp_path / "README.md").write_text("# Big Project\n\n" + "Description. " * 100 + "\n")
        for d in ["src", "lib", "tests", "docs", "scripts"]:
            (tmp_path / d).mkdir()

        result = ProjectContext.auto_generate(str(tmp_path))

        content = result.read_text()
        # Rough token estimate: ~4 chars per token
        assert len(content) < 3000, f"Too large: {len(content)} chars"

    def test_skips_hidden_and_venv_dirs(self, tmp_path):
        """Hidden dirs and venv are excluded from listing."""
        for d in [".git", "__pycache__", "venv", "env", "node_modules", "src"]:
            (tmp_path / d).mkdir()

        result = ProjectContext.auto_generate(str(tmp_path))

        content = result.read_text()
        # Check the structure section only (avoid matching project name)
        struct_idx = content.find("## Structure")
        if struct_idx >= 0:
            struct_section = content[struct_idx:]
            assert "venv/" not in struct_section
            assert "node_modules/" not in struct_section
            assert ".git/" not in struct_section
            assert "src/" in struct_section

    def test_no_readme_still_works(self, tmp_path):
        """Works without README — just lists structure."""
        (tmp_path / "lib").mkdir()
        (tmp_path / "Cargo.toml").write_text("[package]\nname='test'\n")

        result = ProjectContext.auto_generate(str(tmp_path))

        assert result is not None
        content = result.read_text()
        assert "lib/" in content
        assert "Rust" in content

    def test_loaded_by_project_context(self, tmp_path):
        """Auto-generated init.md is loadable by ProjectContext."""
        (tmp_path / "README.md").write_text("# Test\n\nA test project.\n")

        ProjectContext.auto_generate(str(tmp_path))

        ctx = ProjectContext(root_path=str(tmp_path))
        assert ctx.load() is True
        assert "test project" in ctx.content.lower()

    def test_loads_agents_and_claude_files_when_victor_init_missing(self, tmp_path):
        """Compatibility instruction files should load in deterministic scope order."""
        (tmp_path / "AGENTS.md").write_text("# Repo Rules\nUse AGENTS first.\n")
        nested = tmp_path / "pkg"
        nested.mkdir()
        (nested / "CLAUDE.md").write_text("# Nested Rules\nUse CLAUDE here.\n")

        ctx = ProjectContext(root_path=str(nested))

        assert ctx.load() is True
        assert ctx.context_file == nested / "CLAUDE.md"
        assert "use claude here" in ctx.content.lower()
        assert "use agents first" in ctx.content.lower()
        assert [item.scope for item in ctx.instruction_files] == ["workspace", "user"]

    def test_prefers_victor_native_context_in_same_directory(self, tmp_path):
        """Victor-native instructions should be the canonical primary file."""
        victor_dir = tmp_path / ".victor"
        victor_dir.mkdir(exist_ok=True)
        init_file = victor_dir / "init.md"
        init_file.write_text("# Victor\n\n## Architecture\nCanonical runtime.\n")
        (tmp_path / "AGENTS.md").write_text("# Agents\nCompatibility instructions.\n")

        ctx = ProjectContext(root_path=str(tmp_path))

        assert ctx.load() is True
        assert ctx.context_file == init_file
        assert ctx.get_section("architecture") == "Canonical runtime."

    def test_system_prompt_addition_explains_instruction_import_behavior(self, tmp_path):
        """Prompt injection should make import order and loaded files explicit."""
        (tmp_path / "AGENTS.md").write_text("# Repo Rules\nUse AGENTS.\n")
        ctx = ProjectContext(root_path=str(tmp_path))

        assert ctx.load() is True
        prompt = ctx.get_system_prompt_addition()

        assert "walking upward from the current working directory" in prompt
        assert "Victor-native files load before" in prompt
        assert "AGENTS.md" in prompt

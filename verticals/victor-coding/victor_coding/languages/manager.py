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

"""Language Manager for high-level language operations.

Provides a unified interface for working with multiple languages,
abstracting away plugin details.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from victor_coding.languages.base import (
    LanguageCapabilities,
    LanguageConfig,
)
from victor_coding.languages.registry import LanguageRegistry, get_language_registry

logger = logging.getLogger(__name__)


class LanguageManager:
    """High-level manager for language operations.

    Provides convenient methods for common operations like
    running tests, formatting, linting across any supported language.
    """

    def __init__(
        self,
        registry: Optional[LanguageRegistry] = None,
        auto_discover: bool = True,
    ):
        """Initialize language manager.

        Args:
            registry: Custom registry (uses global if None)
            auto_discover: Whether to auto-discover plugins
        """
        self._registry = registry or get_language_registry()

        if auto_discover:
            self._registry.discover_plugins()

    # Detection

    def detect_language(self, path: Path) -> Optional[str]:
        """Detect language from file path.

        Args:
            path: File path

        Returns:
            Language name or None
        """
        return self._registry.detect_language(path)

    def detect_project_languages(self, project_root: Path) -> Dict[str, int]:
        """Detect languages used in a project.

        Args:
            project_root: Project root directory

        Returns:
            Dict mapping language -> file count
        """
        counts: Dict[str, int] = {}

        for path in project_root.rglob("*"):
            if path.is_file():
                lang = self._registry.detect_language(path)
                if lang:
                    counts[lang] = counts.get(lang, 0) + 1

        return dict(sorted(counts.items(), key=lambda x: -x[1]))

    # Configuration

    def get_config(self, language: str) -> LanguageConfig:
        """Get configuration for a language.

        Args:
            language: Language name

        Returns:
            Language configuration
        """
        plugin = self._registry.get(language)
        return plugin.config

    def get_capabilities(self, language: str) -> LanguageCapabilities:
        """Get capabilities for a language.

        Args:
            language: Language name

        Returns:
            Language capabilities
        """
        plugin = self._registry.get(language)
        return plugin.capabilities

    # Testing

    async def run_tests(
        self,
        project_root: Path,
        language: Optional[str] = None,
        path: Optional[str] = None,
        coverage: bool = False,
        parallel: bool = False,
    ) -> Tuple[bool, str]:
        """Run tests for a project.

        Args:
            project_root: Project root directory
            language: Language (auto-detected if None)
            path: Specific test path (runs all if None)
            coverage: Whether to collect coverage
            parallel: Whether to run in parallel

        Returns:
            Tuple of (success, output)
        """
        # Auto-detect language if needed
        if language is None:
            languages = self.detect_project_languages(project_root)
            if not languages:
                return False, "No languages detected in project"
            language = next(iter(languages.keys()))

        # Get test runner
        plugin = self._registry.get(language)
        runner = plugin.get_test_runner(project_root)

        if runner is None:
            return False, f"No test runner available for {language}"

        # Build command
        cmd = list(runner.command)

        if coverage and runner.coverage_args:
            cmd.extend(runner.coverage_args)

        if parallel and runner.parallel_args:
            cmd.extend(runner.parallel_args)

        if path:
            cmd.append(path)

        # Run tests
        return await self._run_command(cmd, project_root)

    async def discover_tests(self, project_root: Path, language: Optional[str] = None) -> List[str]:
        """Discover tests in a project.

        Args:
            project_root: Project root directory
            language: Language (auto-detected if None)

        Returns:
            List of test file paths
        """
        if language is None:
            languages = self.detect_project_languages(project_root)
            if not languages:
                return []
            language = next(iter(languages.keys()))

        plugin = self._registry.get(language)
        runner = plugin.get_test_runner(project_root)

        if runner is None:
            return []

        # Find test files matching pattern
        tests = []
        for path in project_root.rglob(runner.file_pattern):
            tests.append(str(path.relative_to(project_root)))

        return sorted(tests)

    # Formatting

    async def format_code(
        self,
        project_root: Path,
        language: Optional[str] = None,
        check_only: bool = False,
        paths: Optional[List[str]] = None,
    ) -> Tuple[bool, str]:
        """Format code in a project.

        Args:
            project_root: Project root directory
            language: Language (auto-detected if None)
            check_only: Only check, don't modify
            paths: Specific paths to format (all if None)

        Returns:
            Tuple of (success, output)
        """
        if language is None:
            languages = self.detect_project_languages(project_root)
            if not languages:
                return False, "No languages detected"
            language = next(iter(languages.keys()))

        plugin = self._registry.get(language)
        formatter = plugin.get_formatter(project_root)

        if formatter is None:
            return False, f"No formatter available for {language}"

        # Build command
        cmd = list(formatter.command)

        if check_only and formatter.check_args:
            cmd.extend(formatter.check_args)

        if paths:
            cmd.extend(paths)

        return await self._run_command(cmd, project_root)

    # Linting

    async def lint_code(
        self,
        project_root: Path,
        language: Optional[str] = None,
        fix: bool = False,
        paths: Optional[List[str]] = None,
    ) -> Tuple[bool, str]:
        """Lint code in a project.

        Args:
            project_root: Project root directory
            language: Language (auto-detected if None)
            fix: Whether to auto-fix issues
            paths: Specific paths to lint (all if None)

        Returns:
            Tuple of (success, output)
        """
        if language is None:
            languages = self.detect_project_languages(project_root)
            if not languages:
                return False, "No languages detected"
            language = next(iter(languages.keys()))

        plugin = self._registry.get(language)
        linter = plugin.get_linter(project_root)

        if linter is None:
            return False, f"No linter available for {language}"

        # Build command
        cmd = list(linter.command)

        if fix and linter.fix_args:
            cmd.extend(linter.fix_args)

        if paths:
            cmd.extend(paths)

        return await self._run_command(cmd, project_root)

    # Building

    async def build_project(
        self,
        project_root: Path,
        language: Optional[str] = None,
        release: bool = False,
    ) -> Tuple[bool, str]:
        """Build a project.

        Args:
            project_root: Project root directory
            language: Language (auto-detected if None)
            release: Build in release mode

        Returns:
            Tuple of (success, output)
        """
        if language is None:
            languages = self.detect_project_languages(project_root)
            if not languages:
                return False, "No languages detected"
            language = next(iter(languages.keys()))

        plugin = self._registry.get(language)
        build = plugin.get_build_system(project_root)

        if build is None:
            return False, f"No build system available for {language}"

        # Build command
        cmd = list(build.build_command)

        if release and build.release_args:
            cmd.extend(build.release_args)
        elif not release and build.debug_args:
            cmd.extend(build.debug_args)

        return await self._run_command(cmd, project_root)

    async def run_project(
        self,
        project_root: Path,
        language: Optional[str] = None,
        args: Optional[List[str]] = None,
    ) -> Tuple[bool, str]:
        """Run a project.

        Args:
            project_root: Project root directory
            language: Language (auto-detected if None)
            args: Arguments to pass to the program

        Returns:
            Tuple of (success, output)
        """
        if language is None:
            languages = self.detect_project_languages(project_root)
            if not languages:
                return False, "No languages detected"
            language = next(iter(languages.keys()))

        plugin = self._registry.get(language)
        build = plugin.get_build_system(project_root)

        if build is None or not build.run_command:
            return False, f"No run command available for {language}"

        # Build command
        cmd = list(build.run_command)

        if args:
            cmd.extend(args)

        return await self._run_command(cmd, project_root)

    # Utilities

    def list_languages(self) -> List[str]:
        """List all supported languages."""
        return self._registry.list_languages()

    async def _run_command(self, cmd: List[str], cwd: Path, timeout: int = 300) -> Tuple[bool, str]:
        """Run a command and return result.

        Args:
            cmd: Command to run
            cwd: Working directory
            timeout: Timeout in seconds

        Returns:
            Tuple of (success, output)
        """
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )

            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=timeout)

            output = stdout.decode("utf-8", errors="replace")
            success = process.returncode == 0

            return success, output

        except asyncio.TimeoutError:
            return False, f"Command timed out after {timeout}s"
        except FileNotFoundError:
            return False, f"Command not found: {cmd[0]}"
        except Exception as e:
            return False, f"Error running command: {e}"

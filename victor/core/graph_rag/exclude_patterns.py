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

"""Universal build artifact exclusion patterns for code indexing.

This module provides language-agnostic exclusion patterns to ensure only
source code is indexed, not build artifacts, dependencies, or generated files.

Three strategies are used:
1. Comprehensive static list of known build directories across all major languages
2. Parse .gitignore and respect those patterns (project-specific exclusions)
3. Config file detection (optional) - add excludes based on detected languages

Usage:
    from victor.core.graph_rag.exclude_patterns import get_exclusion_patterns

    patterns = get_exclusion_patterns(
        root_path=Path("/path/to/repo"),
        respect_gitignore=True,
        detect_languages=True,
    )
"""

from __future__ import annotations

import fnmatch
import logging
from pathlib import Path
from typing import List, Optional, Set

logger = logging.getLogger(__name__)


# Comprehensive list of build artifacts across all major languages
# Format: Glob patterns compatible with fnmatch
UNIVERSAL_EXCLUDE_PATTERNS = [
    # Version control
    "**/.git/**",
    "**/.svn/**",
    "**/.hg/**",
    # Python
    "**/__pycache__/**",
    "**/*.pyc",
    "**/*.pyo",
    "**/*.pyd",
    "**/.pytest_cache/**",
    "**/.mypy_cache/**",
    "**/.ruff_cache/**",
    "**/.tox/**",
    "**/.eggs/**",
    "**/*.egg-info/**",
    "**/dist/**",
    "**/build/**",
    # venv* / .venv* cover suffixed environments too (.venv-vllm, venv310, …)
    # — observed 65k site-packages files indexed because only the exact
    # ".venv"/"venv" names were excluded. A source dir starting with "venv"
    # is effectively unheard of; a virtualenv is never source.
    "**/.venv*/**",
    "**/venv*/**",
    "**/.virtualenv/**",
    "**/.env/**",
    "**/pip-wheel-metadata/**",
    # Node.js / JavaScript / TypeScript
    "**/node_modules/**",
    "**/jspm_packages/**",
    "**/.yarn/**",
    "**/.yarn/cache/**",
    "**/.yarn/unplugged/**",
    "**/.yarn/build-state.yml",
    "**/.yarn/install-state.gz",
    "**/dist/**",
    "**/out/**",
    "**/build/**",
    "**/.next/**",
    "**/.nuxt/**",
    "**/.webpack/**",
    "**/.rollup.cache/**",
    "**/.vite/**",
    "**/.cache/**",
    "**/parcel-bundle/**",
    "**/.tsbuildinfo",
    # Rust
    "**/target/**",
    "**/Cargo.lock",
    # Go
    "**/bin/**",
    "**/pkg/**",
    # Java / JVM
    "**/target/**",  # Maven
    "**/build/**",  # Gradle
    "**/bin/**",  # IntelliJ/Eclipse
    "**/out/**",  # IntelliJ
    "**/.gradle/**",
    "**/.idea/**",
    "**/.settings/**",
    "**/.project",
    "**/.classpath",
    # Scala
    "**/target/**",  # sbt
    "**/project/**",
    "**/.bloop/**",
    # Kotlin
    "**/build/**",  # Gradle
    "**/target/**",  # Maven
    "**/.kotlin/**",
    # C/C++
    "**/build/**",
    "**/cmake-build-*/**",
    "**/out/**",
    "**/Release/**",
    "**/Debug/**",
    "**/CMakeFiles/**",
    "**/CMakeCache.txt",
    "**/cmake_install.cmake",
    "**/Makefile**",
    # Vendored source trees (Chromium/Bazel/CMake conventions). Kept to the
    # unambiguous name — `external/` is NOT excluded globally because some
    # projects keep first-party integration code there; CMake projects get
    # it via LANGUAGE_CONFIG_MAP instead.
    "**/third_party/**",
    # Ruby
    "**/vendor/bundle/**",
    "**/.bundle/**",
    "**/tmp/**",
    "**/log/**",
    # PHP
    "**/vendor/**",
    "**/storage/app/**",  # Laravel (narrowed from **/storage/** — that glob matched any
    "**/storage/framework/**",  # `storage/` dir, e.g. a Rust `src/storage/` source tree)
    "**/storage/logs/**",
    "**/storage/debugbar/**",
    "**/bootstrap/cache/**",
    # Swift / Objective-C
    "**/DerivedData/**",
    "**/build/**",
    "**/*.build/**",
    "**/*.xcodeproj/**",
    "**/*.xcworkspace/**",
    "**/.swiftpm/**",
    # Dart / Flutter
    "**/build/**",
    "**/.dart_tool/**",
    "**/flutter_build/**",
    # Elixir
    "**/_build/**",
    "**/deps/**",
    "**/.elixir_ls/**",
    # Erlang
    "**/ebin/**",
    "**/deps/**",
    # Haskell
    "**/dist/**",
    "**/dist-newstyle/**",
    "**/.cabal/**",
    "**/.stack-work/**",
    # Lua
    "**/.luarocks/**",
    # R
    "**/packrat/**",
    # General test artifacts
    "**/coverage/**",
    "**/htmlcov/**",
    "**/.coverage",
    "**/.coverage/**",
    "**/.coverage.*",
    "**/.nyc_output/**",
    "**/test/fixtures/**",
    "**/spec/fixtures/**",
    "**/tests/fixtures/**",
    # IDE files
    "**/.idea/**",
    "**/.vscode/**",
    "**/.vs/**",
    "**/*.swp",
    "**/*.swo",
    "**/*.swn",
    "**/.DS_Store",
    "**/Thumbs.db",
    "**/*.sublime-project",
    "**/*.sublime-workspace",
    # OS files
    "**/.DS_Store/**",
    "**/._*",
    # Documentation builds
    "**/site/**",
    "**/_site/**",
    "**/docs/_build/**",
    "**/.doctrees/**",
    # Dependency lock files (large, not source code)
    "**/package-lock.json",
    "**/yarn.lock",
    "**/pnpm-lock.yaml",
    "**/Poetry.lock",
    "**/Gemfile.lock",
    "**/composer.lock",
    "**/Cargo.lock",
    # Minified assets
    "**/*.min.js",
    "**/*.min.css",
    "**/*.min.html",
    # Generated files
    "**/*.generated.*",
    "**/*.gen.*",
    "**/.gencache/**",
    # Victor-specific
    "**/.victor/**",
    # Claude Code metadata and ephemeral worktrees (never source code)
    "**/.claude/**",
    # VS Code extension test fixtures
    "**/.vscode-test/**",
    # Docker
    "**/.dockerignore",
    # CI/CD
    "**/.github/workflows/*/dist/**",  # Action builds
]


# Language detection via config files
LANGUAGE_CONFIG_MAP = {
    "Cargo.toml": ["target/"],  # Rust
    "package.json": ["node_modules/", "dist/", ".next/", ".nuxt/"],  # Node.js
    "go.mod": [],  # Go (builds to current dir or GOPATH)
    "requirements.txt": ["__pycache__/", "build/", "dist/", "*.egg-info/"],  # Python
    "pyproject.toml": ["__pycache__/", "build/", "dist/", "*.egg-info/"],  # Python
    "setup.py": ["build/", "dist/", "*.egg-info/"],  # Python
    "pom.xml": ["target/"],  # Maven (Java)
    "build.gradle": ["build/"],  # Gradle (Java/Kotlin)
    "build.gradle.kts": ["build/"],  # Gradle Kotlin
    "Gemfile": ["vendor/bundle/"],  # Ruby
    "composer.json": ["vendor/"],  # PHP
    "pubspec.yaml": ["build/", ".dart_tool/"],  # Dart
    "mix.exs": ["_build/", "deps/"],  # Elixir
    # C/C++: out-of-source builds are conventionally build-<flavor>/ siblings
    # (build-cuda, build-debug, …); external/ is the common vendoring dir for
    # CMake FetchContent/submodule trees (e.g. external/llama.cpp).
    "CMakeLists.txt": ["build/", "build-*/", "build_*/", "cmake-build-*/", "external/"],
    "Makefile": ["build/", "out/"],  # C/C++
    "setup.go": [],  # Go
    "Package.swift": [".build/", "build/"],  # Swift
    "Cartfile": [],  # Swift (Carthage)
    "Podfile": ["Pods/"],  # Swift (CocoaPods)
    "project.clj": ["target/"],  # Clojure
    "shard.yml": ["lib/"],  # Crystal
    "rebar.config": ["_build/", "deps/"],  # Erlang
    "*.cabal": ["dist-newstyle/", ".cabal/"],  # Haskell
    "stack.yaml": [".stack-work/"],  # Haskell
}


def parse_gitignore(root_path: Path) -> List[str]:
    """Parse .gitignore file and extract exclusion patterns.

    Args:
        root_path: Root directory of the repository

    Returns:
        List of glob patterns from .gitignore
    """
    gitignore_path = root_path / ".gitignore"
    if not gitignore_path.exists():
        return []

    patterns = []
    try:
        with open(gitignore_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue

                # Skip negation patterns (lines starting with !)
                if line.startswith("!"):
                    continue

                # Convert gitignore pattern to glob pattern
                # gitignore patterns are relative to .gitignore location
                # We need to add **/ prefix for recursive matching
                if not line.startswith("**/"):
                    pattern = f"**/{line}"
                else:
                    pattern = line

                # Handle directory patterns (ending with /).
                # Strip the trailing slash before appending /** to avoid
                # producing double-slash patterns like "**/.claude//**".
                if line.endswith("/"):
                    pattern = f"{pattern.rstrip('/')}/**"

                patterns.append(pattern)

        logger.debug(f"Parsed {len(patterns)} patterns from .gitignore")
    except Exception as e:
        logger.warning(f"Failed to parse .gitignore: {e}")

    return patterns


def detect_language_excludes(root_path: Path) -> List[str]:
    """Detect languages in the project and add appropriate excludes.

    Args:
        root_path: Root directory of the repository

    Returns:
        List of glob patterns for detected language build artifacts
    """
    patterns = []

    # Check for language config files
    for config_file, exclude_dirs in LANGUAGE_CONFIG_MAP.items():
        # Handle wildcards in config file names
        if "*" in config_file:
            # Check if any file matches the pattern
            matching_files = list(root_path.rglob(config_file))
            if matching_files:
                logger.debug(f"Detected language from {config_file}")
                patterns.extend(exclude_dirs)
        else:
            config_path = root_path / config_file
            if config_path.exists():
                logger.debug(f"Detected language from {config_file}")
                patterns.extend(exclude_dirs)

    # Convert directory patterns to glob patterns
    glob_patterns = []
    for pattern in patterns:
        if not pattern.startswith("**/"):
            glob_patterns.append(f"**/{pattern}")
        else:
            glob_patterns.append(pattern)

    # Add recursive glob for directory patterns
    glob_patterns.extend([f"{p}/**" for p in patterns if not p.endswith("**")])

    return list(set(glob_patterns))  # Deduplicate


def get_exclusion_patterns(
    root_path: Path,
    respect_gitignore: bool = True,
    detect_languages: bool = True,
    custom_patterns: Optional[List[str]] = None,
) -> List[str]:
    """Get comprehensive exclusion patterns for code indexing.

    This function combines three strategies:
    1. Universal static patterns (all major languages)
    2. Project-specific patterns from .gitignore
    3. Language-specific patterns based on detected config files

    Args:
        root_path: Root directory of the repository
        respect_gitignore: Whether to parse and include .gitignore patterns
        detect_languages: Whether to detect languages and add their build dirs
        custom_patterns: Additional custom patterns to exclude

    Returns:
        List of glob patterns for exclusion
    """
    patterns = []

    # 1. Add universal patterns
    patterns.extend(UNIVERSAL_EXCLUDE_PATTERNS)

    # 2. Parse .gitignore
    if respect_gitignore:
        gitignore_patterns = parse_gitignore(root_path)
        patterns.extend(gitignore_patterns)

    # 3. Detect languages
    if detect_languages:
        language_patterns = detect_language_excludes(root_path)
        patterns.extend(language_patterns)

    # 4. Add custom patterns
    if custom_patterns:
        patterns.extend(custom_patterns)

    # Deduplicate while preserving order
    seen = set()
    unique_patterns = []
    for pattern in patterns:
        if pattern not in seen:
            seen.add(pattern)
            unique_patterns.append(pattern)

    logger.info(f"Generated {len(unique_patterns)} exclusion patterns")
    return unique_patterns


def is_path_excluded(path: Path, root_path: Path, patterns: List[str]) -> bool:
    """Check if a path matches any exclusion pattern.

    Args:
        path: Path to check (can be absolute or relative)
        root_path: Root directory for resolving relative paths
        patterns: List of glob patterns to match against

    Returns:
        True if path should be excluded, False otherwise
    """
    # Convert to relative path if needed
    try:
        if path.is_absolute():
            rel_path = path.relative_to(root_path)
        else:
            rel_path = path
    except ValueError:
        # Path is not under root_path, treat as excluded
        return True

    path_str = str(rel_path)

    # Check against all patterns
    for pattern in patterns:
        # Remove the **/ prefix for matching if present
        clean_pattern = pattern.replace("**/", "")

        # Check if path matches pattern
        if fnmatch.fnmatch(path_str, pattern):
            return True

        # Check if any part of the path matches the pattern
        # This handles patterns like "**/target/**" matching "foo/target/bar"
        path_parts = path_str.split("/")
        for i in range(len(path_parts)):
            subpath = "/".join(path_parts[i:])
            if fnmatch.fnmatch(subpath, clean_pattern) or fnmatch.fnmatch(subpath, pattern):
                return True

        # Also check if path starts with pattern (for directory patterns)
        if path_str.startswith(clean_pattern.rstrip("/")):
            return True

    return False


__all__ = [
    "UNIVERSAL_EXCLUDE_PATTERNS",
    "LANGUAGE_CONFIG_MAP",
    "get_exclusion_patterns",
    "is_path_excluded",
    "parse_gitignore",
    "detect_language_excludes",
]

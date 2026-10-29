"""Language-specific manifest discovery for planning and deterministic tool steps.

Pre-registered handlers (framework utilities, file-system structural knowledge):
  rust        — Cargo.toml workspace manifests
  python      — pyproject.toml, setup.py/cfg, requirements.txt, Pipfile
  javascript  — package.json, lockfiles, pnpm/yarn/npm workspace roots
  typescript  — package.json + tsconfig*.json
  go          — go.mod
  java        — pom.xml, build.gradle / settings.gradle variants
  kotlin      — settings.gradle.kts, build.gradle.kts
  ruby        — Gemfile, *.gemspec
  php         — composer.json

Verticals call ``register_manifest_handler(language, handler)`` to add more
languages or to replace a default implementation with a richer one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, Iterable, List, Protocol, Sequence, Tuple

from victor.agent.planning.repository_profile import EXCLUDED_DIRS


@dataclass(frozen=True)
class ManifestSelection:
    """Manifest paths selected for a language-aware planning step."""

    language: str
    paths: List[str]
    explicit: bool = False


class LanguageManifestHandler(Protocol):
    """Protocol for language-specific manifest discovery."""

    language: str

    def discover(self, root: Path, *, max_files: int = 5000) -> List[str]:
        """Return relevant manifest paths relative to ``root``."""

    def select_for_step(self, text: str, root: Path) -> ManifestSelection:
        """Return manifest paths relevant to a deterministic plan step."""


def _safe_relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _is_excluded(path: Path, root: Path) -> bool:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        parts = path.parts
    return any(part in EXCLUDED_DIRS for part in parts)


def _dedupe(paths: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(path for path in paths if path))


class RustManifestHandler:
    """Discover and select Cargo manifests for Rust codebases."""

    language = "rust"
    manifest_name = "Cargo.toml"
    _cargo_path_pattern = re.compile(r"(?<![\w.-])(?P<path>(?:[\w.-]+/)*Cargo\.toml)\b", re.I)
    _explicit_read_cargo_pattern = re.compile(
        r"\bread\s+(?:the\s+)?(?P<path>(?:root\s+)?(?:[\w.-]+/)*Cargo\.toml)\b",
        re.I,
    )

    def discover(self, root: Path, *, max_files: int = 5000) -> List[str]:
        """Return Cargo.toml files ordered by workspace relevance."""
        root = root.expanduser().resolve()
        if not root.exists():
            return []

        manifests: List[str] = []
        inspected = 0
        for path in root.rglob(self.manifest_name):
            if inspected >= max_files:
                break
            inspected += 1
            if _is_excluded(path, root) or not path.is_file():
                continue
            manifests.append(_safe_relative(path, root))

        return sorted(
            _dedupe(manifests),
            key=lambda path: (
                0 if path == self.manifest_name else 1,
                path.count("/"),
                path,
            ),
        )

    def select_for_step(self, text: str, root: Path) -> ManifestSelection:
        """Select explicit Cargo paths first, otherwise discovered manifests."""
        explicit_paths = self._extract_explicit_paths(text)
        if explicit_paths:
            resolved_paths = self._resolve_missing_bare_root_manifest(explicit_paths, root)
            return ManifestSelection(self.language, resolved_paths, explicit=True)
        return ManifestSelection(self.language, self.discover(root), explicit=False)

    def _resolve_missing_bare_root_manifest(self, paths: List[str], root: Path) -> List[str]:
        """Use discovered manifests when a bare/root Cargo.toml is not present."""
        root = root.expanduser().resolve()
        discovered = self.discover(root)
        resolved: List[str] = []
        for path in paths:
            if path == self.manifest_name and not (root / path).is_file() and discovered:
                resolved.extend(discovered)
            else:
                resolved.append(path)
        return _dedupe(resolved)

    def _extract_explicit_paths(self, text: str) -> List[str]:
        """Extract Cargo.toml paths mentioned in a step while preserving root intent."""
        paths: List[str] = []
        for match in self._explicit_read_cargo_pattern.finditer(text):
            path = self._normalize_manifest_path(match.group("path"))
            if self._is_bare_plural_manifest_reference(path, text, match.end()):
                continue
            paths.append(path)
        for match in self._cargo_path_pattern.finditer(text):
            path = self._normalize_manifest_path(match.group("path"))
            if self._is_bare_plural_manifest_reference(path, text, match.end()):
                continue
            paths.append(path)
        return _dedupe(paths)

    @staticmethod
    def _normalize_manifest_path(path: str) -> str:
        normalized = " ".join(path.strip().split())
        if normalized.lower() == "root cargo.toml":
            return "Cargo.toml"
        if normalized.startswith("./"):
            normalized = normalized[2:]
        return normalized

    @staticmethod
    def _is_bare_plural_manifest_reference(path: str, text: str, end_index: int) -> bool:
        """Return whether ``Cargo.toml files`` refers to discovery, not one path."""
        if path != "Cargo.toml":
            return False
        return bool(re.match(r"\s+files\b", text[end_index:], flags=re.I))


class SimpleManifestHandler:
    """Generic manifest handler for languages with fixed manifest filenames.

    Discovers manifests by filename and extracts explicit paths mentioned in
    step text.  Subclass to override ``language`` and ``manifest_names``.
    """

    language: str = ""
    manifest_names: Tuple[str, ...] = ()

    def discover(self, root: Path, *, max_files: int = 5000) -> List[str]:
        root = root.expanduser().resolve()
        if not root.exists():
            return []
        found: List[str] = []
        inspected = 0
        for name in self.manifest_names:
            if "*" in name:
                pattern_root = root
                glob_fn = pattern_root.rglob
            else:
                glob_fn = root.rglob  # type: ignore[assignment]
            for path in glob_fn(name):
                if inspected >= max_files:
                    break
                inspected += 1
                if _is_excluded(path, root) or not path.is_file():
                    continue
                found.append(_safe_relative(path, root))
        return sorted(
            _dedupe(found),
            key=lambda p: (p.count("/"), p),
        )

    def select_for_step(self, text: str, root: Path) -> ManifestSelection:
        explicit = self._extract_explicit(text)
        if explicit:
            return ManifestSelection(self.language, explicit, explicit=True)
        return ManifestSelection(self.language, self.discover(root), explicit=False)

    def _extract_explicit(self, text: str) -> List[str]:
        """Return manifest paths explicitly mentioned in ``text``."""
        found: List[str] = []
        for name in self.manifest_names:
            if "*" in name:
                continue
            escaped = re.escape(name)
            for match in re.finditer(
                rf"(?:^|[\s\(\"'])(?P<path>(?:[\w./-]+/)*{escaped})\b",
                text,
                flags=re.IGNORECASE | re.MULTILINE,
            ):
                found.append(match.group("path").strip())
        return _dedupe(found)


class PythonManifestHandler(SimpleManifestHandler):
    language = "python"
    manifest_names = (
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "requirements.txt",
        "Pipfile",
    )


class JavaScriptManifestHandler(SimpleManifestHandler):
    language = "javascript"
    manifest_names = (
        "package.json",
        "pnpm-lock.yaml",
        "yarn.lock",
        "package-lock.json",
    )


class TypeScriptManifestHandler(SimpleManifestHandler):
    language = "typescript"
    manifest_names = ("package.json", "tsconfig.json", "tsconfig*.json")


class GoManifestHandler(SimpleManifestHandler):
    language = "go"
    manifest_names = ("go.mod",)


class JavaManifestHandler(SimpleManifestHandler):
    language = "java"
    manifest_names = (
        "pom.xml",
        "build.gradle",
        "settings.gradle",
        "build.gradle.kts",
        "settings.gradle.kts",
    )


class KotlinManifestHandler(SimpleManifestHandler):
    language = "kotlin"
    manifest_names = ("settings.gradle.kts", "build.gradle.kts")


class RubyManifestHandler(SimpleManifestHandler):
    language = "ruby"
    manifest_names = ("Gemfile", "*.gemspec")


class PhpManifestHandler(SimpleManifestHandler):
    language = "php"
    manifest_names = ("composer.json",)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# The framework registers its own file-system utilities by default.
# All handlers here deal with manifest file structure — they are not domain
# expertise.  Verticals call register_manifest_handler() to add more languages
# or to replace a default with a richer implementation.
_HANDLERS: Dict[str, LanguageManifestHandler] = {
    "rust": RustManifestHandler(),
    "python": PythonManifestHandler(),
    "javascript": JavaScriptManifestHandler(),
    "typescript": TypeScriptManifestHandler(),
    "go": GoManifestHandler(),
    "java": JavaManifestHandler(),
    "kotlin": KotlinManifestHandler(),
    "ruby": RubyManifestHandler(),
    "php": PhpManifestHandler(),
}


def register_manifest_handler(language: str, handler: LanguageManifestHandler) -> None:
    """Register or replace a language manifest handler.

    The framework pre-registers handlers for: rust, python, javascript,
    typescript, go, java, kotlin, ruby, php.  Call this inside
    ``VictorPlugin.register(context)`` to add a new language or to replace
    a default with a richer implementation.

    Example (in victor_coding/plugin.py)::

        from victor.agent.planning.language_manifests import (
            SimpleManifestHandler, register_manifest_handler,
        )

        class SwiftManifestHandler(SimpleManifestHandler):
            language = "swift"
            manifest_names = ("Package.swift",)

        class CodingPlugin(VictorPlugin):
            @classmethod
            def register(cls, context):
                register_manifest_handler("swift", SwiftManifestHandler())
    """
    _HANDLERS[language.lower()] = handler


def get_manifest_handler(language: str) -> LanguageManifestHandler | None:
    """Return the registered manifest handler for ``language`` if available."""
    return _HANDLERS.get(language.lower())


def select_language_manifests(
    language: str,
    text: str,
    *,
    root: Path | None = None,
) -> ManifestSelection:
    """Select manifest paths for a language-aware plan step."""
    handler = get_manifest_handler(language)
    if handler is None:
        return ManifestSelection(language, [])
    return handler.select_for_step(text, root or Path.cwd())


def discover_language_manifests(
    language: str,
    *,
    root: Path | None = None,
    max_files: int = 5000,
) -> Sequence[str]:
    """Discover manifests for ``language`` under ``root``."""
    handler = get_manifest_handler(language)
    if handler is None:
        return []
    return handler.discover(root or Path.cwd(), max_files=max_files)

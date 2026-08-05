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

"""Per-language import resolution strategies for Graph RAG indexing.

This module is the SEAM between the language-agnostic indexing pipeline
and language-specific import semantics, mirroring the registry pattern in
``language_handlers.py``:

- ``_resolve_imports()`` in ``indexing.py`` buffers raw import strings
  (extracted by the TreeSitterAnalysisProtocol provider) and drives the
  generic resolve loop.
- Each language contributes a :class:`LanguageImportResolver` strategy that
  knows how to (a) parse one raw import statement into module-name
  candidates and (b) resolve a candidate to a project file.
- Strategies are discovered through :class:`ImportResolverRegistry`.
  Resolver instances are created per resolution run, so a strategy may
  cache project-layout state (e.g. the Rust workspace crate map) on
  ``self`` without staleness across runs.

Python, Rust, JS/TS, C/C++, and Java/Scala are implemented today. Go needs
its own strategy (module paths from go.mod) — register it here rather than
adding language branches to ``indexing.py``.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path, PurePosixPath
from typing import Callable, Dict, List, Optional, Protocol, Set, Tuple

logger = logging.getLogger(__name__)


class LanguageImportResolver(Protocol):
    """Strategy protocol: parse and resolve one language's imports.

    ``parse`` turns a raw import statement into zero or more self-contained
    module-name candidates; ``resolve`` maps one candidate to a project
    file (or None for stdlib/third-party targets). Candidates must be
    self-contained strings — the pipeline caches ``resolve`` results keyed
    by (candidate, language), so a candidate may not depend on which source
    file emitted it.
    """

    def parse(self, raw: str, src_file: str, root_path: Path) -> List[str]:
        """Parse one raw import statement into module-name candidates."""
        ...

    def resolve(self, module: str, root_path: Path) -> Optional[Path]:
        """Resolve a candidate to a project file, or None if external."""
        ...


class ImportResolverRegistry:
    """Registry of per-language import resolver factories.

    ``create`` returns a fresh strategy instance so per-run caches (e.g.
    the Rust workspace crate map) start clean each indexing run.
    """

    _factories: Dict[str, Callable[[], LanguageImportResolver]] = {}

    @classmethod
    def register(cls, language: str, factory: Callable[[], LanguageImportResolver]) -> None:
        cls._factories[language.lower()] = factory
        logger.debug("Registered import resolver for language: %s", language)

    @classmethod
    def create(cls, language: str) -> Optional[LanguageImportResolver]:
        factory = cls._factories.get(language.lower())
        return factory() if factory is not None else None

    @classmethod
    def supported_languages(cls) -> List[str]:
        return sorted(cls._factories)


class PythonImportResolver:
    """Resolve Python ``import``/``from`` statements to project modules."""

    def parse(self, raw: str, src_file: str, root_path: Path) -> List[str]:
        """Extract candidate dotted module names from a Python import.

        Handles: ``import a``, ``import a as x``, ``import a, b``,
        ``import a.b.c``, ``from a.b import x``, ``from a.b import x, y``,
        ``from .rel import x`` (resolved against ``src_file``'s package),
        and dotted relative imports (``from ..pkg import x``).

        ``from X import Y`` is genuinely ambiguous: ``Y`` could be a symbol
        defined in ``X/__init__.py`` *or* a sibling module ``X/Y.py``. We
        emit both candidates (``X`` and ``X.Y``) and let ``resolve`` decide
        which one actually exists. The downstream resolver dedupes edges by
        (src, dst) pair so emitting both is cheap and prevents losing
        legitimate submodule imports.
        """
        text = raw.strip()
        if text.startswith("import "):
            tail = text[len("import ") :].strip()
            modules: List[str] = []
            for piece in tail.split(","):
                name = piece.strip().split(" as ")[0].strip()
                if name:
                    modules.append(name)
            return modules
        if text.startswith("from "):
            tail = text[len("from ") :].strip()
            if " import " not in tail:
                return []
            module_part, names_part = tail.split(" import ", 1)
            module_part = module_part.strip()
            # Compute the base module path (absolute or resolved-relative).
            base: Optional[str]
            if module_part.startswith("."):
                dots = 0
                while dots < len(module_part) and module_part[dots] == ".":
                    dots += 1
                rest = module_part[dots:]
                try:
                    rel = Path(src_file).resolve().relative_to(root_path.resolve())
                except ValueError:
                    return []
                pkg_parts = list(rel.parts[:-1])
                for _ in range(dots - 1):
                    if not pkg_parts:
                        return []
                    pkg_parts.pop()
                if rest:
                    base = ".".join([*pkg_parts, rest])
                else:
                    base = ".".join(pkg_parts) if pkg_parts else None
            else:
                base = module_part

            candidates: List[str] = []
            if base:
                candidates.append(base)
            # Also emit base.<name> for each imported name so submodule
            # imports (``from pkg import submodule``) get an edge to the
            # submodule itself, not just the parent package. Strip aliases
            # and wildcard form.
            for piece in names_part.split(","):
                name = piece.strip().split(" as ")[0].strip().rstrip(")").lstrip("(")
                if not name or name == "*":
                    continue
                if base:
                    candidates.append(f"{base}.{name}")
                else:
                    candidates.append(name)
            return candidates
        return []

    def resolve(self, module: str, root_path: Path) -> Optional[Path]:
        """Resolve a dotted module name to a project file path.

        Prefers ``foo/bar/baz.py``, falls back to
        ``foo/bar/baz/__init__.py``. Returns None for stdlib/third-party
        modules (anything not under ``root_path``).
        """
        if not module:
            return None
        parts = module.split(".")
        file_candidate = root_path.joinpath(*parts).with_suffix(".py")
        if file_candidate.is_file():
            return file_candidate
        init_candidate = root_path.joinpath(*parts) / "__init__.py"
        if init_candidate.is_file():
            return init_candidate
        return None


class RustImportResolver:
    """Resolve Rust ``use`` declarations to project modules.

    Candidate encoding: ``<src-dir-relpath>`` (crate root) or
    ``<src-dir-relpath>::seg1::seg2`` where the prefix is the anchor
    directory relative to the project root — self-contained so the
    pipeline can cache resolutions without re-deriving crate layout.

    Instances cache the workspace crate map and Cargo.toml walk-ups, so
    create one instance per resolution run (``ImportResolverRegistry
    .create`` does exactly that).
    """

    #: First segments of ``use`` paths that can never be project crates.
    _BUILTIN_CRATES = frozenset({"std", "core", "alloc", "proc_macro", "test"})

    _PACKAGE_NAME_RE = re.compile(r'(?ms)^\[package\][^\[]*?^\s*name\s*=\s*"([^"]+)"')

    def __init__(self) -> None:
        # Workspace crate map (underscored crate name → src dir), built on
        # first external-crate lookup.
        self._crate_src_map: Optional[Dict[str, Path]] = None
        # Cargo.toml walk-up memo (source dir → crate src dir).
        self._cargo_src_cache: Dict[str, Optional[Path]] = {}

    def parse(self, raw: str, src_file: str, root_path: Path) -> List[str]:
        """Extract candidate module names from a Rust ``use`` declaration.

        For each ``use`` path we emit a candidate for *every module prefix*
        below the anchor (``use crate::a::b::X`` → ``a`` and ``a::b``):
        Rust name resolution traverses each of those modules, and the
        prefix set is what makes module-level fan-in (Martin Ca) match
        real use-statement counts. For external workspace crates the crate
        root itself is also emitted (``use other_crate::x`` is a dependency
        on ``other_crate/src/lib.rs``); for ``crate::``/``self::``/
        ``super::`` the anchor is the importer's own crate, so the anchor
        root is deliberately *not* emitted.

        Trailing segments that are clearly symbols (leading uppercase, glob
        ``*``) are dropped; a lowercase final segment that turns out to be
        a function simply fails resolution downstream, mirroring how the
        Python strategy over-emits ``base.name`` candidates.
        """
        try:
            src_path = Path(src_file).resolve()
            src_path.relative_to(root_path.resolve())
        except (OSError, ValueError):
            return []

        text = raw.strip().removesuffix(";").strip()
        # Strip visibility qualifier: pub, pub(crate), pub(in some::path).
        if text.startswith("pub"):
            rest = text[3:].lstrip()
            if rest.startswith("("):
                close = rest.find(")")
                if close == -1:
                    return []
                rest = rest[close + 1 :].lstrip()
            text = rest
        if not text.startswith("use"):
            return []
        text = text[3:].strip()

        candidates: List[str] = []
        seen: Set[str] = set()
        for path in self._expand_use_tree(text):
            segments = [seg.strip().removeprefix("r#") for seg in path.split("::")]
            segments = [seg for seg in segments if seg and seg != "*"]
            if not segments:
                continue

            anchor: Optional[Path] = None
            include_anchor_root = False
            first = segments[0]
            if first == "crate":
                anchor = self._crate_src_dir(src_path, root_path)
                segments = segments[1:]
            elif first in ("self", "super"):
                anchor = self._module_dir(src_path)
                while segments and segments[0] == "super":
                    anchor = anchor.parent
                    segments = segments[1:]
                if segments and segments[0] == "self":
                    segments = segments[1:]
            elif first in self._BUILTIN_CRATES:
                continue
            else:
                # External crate: dependency on a sibling workspace crate.
                anchor = self._workspace_crate_srcs(root_path).get(first)
                segments = segments[1:]
                include_anchor_root = True

            if anchor is None:
                continue
            try:
                anchor_rel = anchor.resolve().relative_to(root_path.resolve()).as_posix()
            except (OSError, ValueError):
                continue  # super:: walked out of the project tree

            # Module-shaped prefixes only: stop at the first segment that
            # can't be a module path component (uppercase = type/const,
            # anything non-identifier = macro noise).
            prefixes: List[str] = []
            for seg in segments:
                if not seg.isidentifier() or seg[0].isupper():
                    break
                prefixes.append(seg)

            if include_anchor_root and anchor_rel not in seen:
                seen.add(anchor_rel)
                candidates.append(anchor_rel)
            for depth in range(1, len(prefixes) + 1):
                name = anchor_rel + "::" + "::".join(prefixes[:depth])
                if name not in seen:
                    seen.add(name)
                    candidates.append(name)
        return candidates

    def resolve(self, module: str, root_path: Path) -> Optional[Path]:
        """Resolve a ``<src-dir>::seg…`` candidate to a project file.

        Prefers ``seg….rs``, falls back to ``seg…/mod.rs``; a bare
        ``<src-dir>`` resolves to the crate root
        (``lib.rs``/``main.rs``/``mod.rs``).
        """
        if not module:
            return None
        dir_part, sep, seg_part = module.partition("::")
        base = root_path / dir_part if dir_part not in ("", ".") else root_path
        segments = [seg for seg in seg_part.split("::") if seg] if sep else []
        # Reject anything that could escape the project tree.
        if ".." in Path(dir_part).parts or any(not seg.isidentifier() for seg in segments):
            return None
        if not base.is_dir():
            return None
        if not segments:
            for name in ("lib.rs", "main.rs", "mod.rs"):
                candidate = base / name
                if candidate.is_file():
                    return candidate
            return None
        file_candidate = base.joinpath(*segments[:-1]) / f"{segments[-1]}.rs"
        if file_candidate.is_file():
            return file_candidate
        mod_candidate = base.joinpath(*segments) / "mod.rs"
        if mod_candidate.is_file():
            return mod_candidate
        return None

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    @classmethod
    def _expand_use_tree(cls, text: str) -> List[str]:
        """Expand a Rust use-tree into plain ``::`` paths.

        Handles brace groups (nested), ``self`` group members (bind to the
        group prefix), aliases (``as x`` stripped), and leading ``::``.
        """
        text = text.strip()
        brace = text.find("{")
        if brace == -1:
            path = text.split(" as ")[0].strip().removeprefix("::").strip()
            return [path] if path else []
        prefix = text[:brace].strip().removesuffix("::").removeprefix("::").strip()
        close = text.rfind("}")
        if close < brace:
            return []
        inner = text[brace + 1 : close]

        # Split group members on top-level commas.
        parts: List[str] = []
        depth = 0
        current: List[str] = []
        for ch in inner:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            elif ch == "," and depth == 0:
                parts.append("".join(current))
                current = []
                continue
            current.append(ch)
        parts.append("".join(current))

        out: List[str] = []
        for part in parts:
            for sub in cls._expand_use_tree(part):
                if sub == "self":
                    if prefix:
                        out.append(prefix)
                elif prefix:
                    out.append(f"{prefix}::{sub}")
                else:
                    out.append(sub)
        return out

    def _crate_src_dir(self, src_path: Path, root_path: Path) -> Optional[Path]:
        """Find the crate src/ dir that ``crate::`` refers to for a file.

        Walks up to the nearest ``Cargo.toml``. Files outside that package's
        ``src/`` (tests/, benches/, examples/) are their own crate roots, so
        ``crate::`` there does *not* point at ``src/`` — return None rather
        than emit false edges into the library crate.
        """
        parent_str = str(src_path.parent)
        if parent_str in self._cargo_src_cache:
            return self._cargo_src_cache[parent_str]
        result: Optional[Path] = None
        d = src_path.parent
        root_resolved = root_path.resolve()
        while True:
            if (d / "Cargo.toml").is_file():
                src_dir = d / "src"
                if src_dir == src_path.parent or src_dir in src_path.parents:
                    result = src_dir
                break
            if d == root_resolved or d == d.parent:
                break
            d = d.parent
        self._cargo_src_cache[parent_str] = result
        return result

    @staticmethod
    def _module_dir(src_path: Path) -> Path:
        """Directory holding a Rust file's child modules (``self::`` anchor).

        ``src/foo/bar.rs`` owns ``src/foo/bar/``; root/section files
        (``lib.rs``, ``main.rs``, ``mod.rs``) own their containing dir.
        """
        if src_path.name in ("lib.rs", "main.rs", "mod.rs"):
            return src_path.parent
        return src_path.parent / src_path.stem

    def _workspace_crate_srcs(self, root_path: Path) -> Dict[str, Path]:
        """Map underscored crate names to their src/ dirs across the workspace.

        Scans for ``Cargo.toml`` files (skipping ``target/`` and hidden
        dirs) and pulls ``[package] name``. Built once per instance, i.e.
        once per resolution run.
        """
        if self._crate_src_map is not None:
            return self._crate_src_map
        crate_map: Dict[str, Path] = {}
        try:
            manifests = list(root_path.rglob("Cargo.toml"))
        except OSError:
            manifests = []
        for manifest in manifests:
            rel_parts = manifest.relative_to(root_path).parts
            if any(part == "target" or part.startswith(".") for part in rel_parts):
                continue
            try:
                match = self._PACKAGE_NAME_RE.search(
                    manifest.read_text(encoding="utf-8", errors="replace")
                )
            except OSError:
                continue
            if not match:
                continue  # workspace-only manifest
            src_dir = manifest.parent / "src"
            if src_dir.is_dir():
                crate_map[match.group(1).replace("-", "_")] = src_dir
        self._crate_src_map = crate_map
        return crate_map


class JsTsImportResolver:
    """Resolve JavaScript/TypeScript import specifiers to project modules.

    Parses the specifier out of ESM imports (``import x from './y'``),
    side-effect imports (``import './y'``), re-exports
    (``export * from './y'``), and ``require('./y')`` / dynamic
    ``import('./y')`` should those raw strings arrive. Specifier classes:

    - Relative (``./x``, ``../x``): resolved against the importing file.
    - tsconfig path aliases (``@/components/Button``): mapped through the
      nearest ``tsconfig.json``/``jsconfig.json`` ``compilerOptions.paths``
      + ``baseUrl`` (walk-up, memoized; ``extends`` chains are not
      followed). Bare specifiers also probe under ``baseUrl`` when set.
    - Bare package names (``react``, ``@scope/pkg``): external — skipped.

    Candidate encoding: root-relative POSIX path without extension
    (``vscode-victor/src/utils/graph``); ``resolve`` applies Node/TS
    extension and ``index.*`` probing, including the ESM ``./x.js`` →
    ``x.ts`` rewrite.

    Instances memoize tsconfig lookups; create one per resolution run
    (``ImportResolverRegistry.create`` does exactly that).
    """

    _FROM_RE = re.compile(r"""\bfrom\s+['"]([^'"]+)['"]""")
    _SIDE_EFFECT_RE = re.compile(r"""^\s*import\s+['"]([^'"]+)['"]""")
    _CALL_RE = re.compile(r"""\b(?:require|import)\s*\(\s*['"]([^'"]+)['"]""")

    #: Extensions probed during resolution, TypeScript first so ``./x`` in a
    #: mixed tree binds to ``x.ts`` over a stale compiled ``x.js``.
    _EXTENSIONS = (".ts", ".tsx", ".mts", ".cts", ".js", ".jsx", ".mjs", ".cjs")
    #: Compiled-JS suffixes that TS sources may reference (``./x.js`` → x.ts).
    _JS_TO_TS = {
        ".js": (".ts", ".tsx"),
        ".mjs": (".mts",),
        ".cjs": (".cts",),
        ".jsx": (".tsx",),
    }

    def __init__(self) -> None:
        # Source dir → (baseUrl dir, paths mapping) from the nearest
        # tsconfig.json/jsconfig.json, or None when none exists.
        self._tsconfig_cache: Dict[str, Optional[Tuple[Path, Dict[str, List[str]]]]] = {}

    def parse(self, raw: str, src_file: str, root_path: Path) -> List[str]:
        try:
            src_path = Path(src_file).resolve()
            src_path.relative_to(root_path.resolve())
        except (OSError, ValueError):
            return []

        specs: List[str] = []
        for pattern in (self._FROM_RE, self._SIDE_EFFECT_RE, self._CALL_RE):
            specs.extend(pattern.findall(raw))

        candidates: List[str] = []
        seen: Set[str] = set()
        for spec in specs:
            # Bundler suffixes (Vite ``?raw``, webpack loaders) aren't part
            # of the module path.
            spec = spec.split("?")[0].split("#")[0].strip()
            if not spec:
                continue
            for candidate in self._spec_to_candidates(spec, src_path, root_path):
                if candidate not in seen:
                    seen.add(candidate)
                    candidates.append(candidate)
        return candidates

    def resolve(self, module: str, root_path: Path) -> Optional[Path]:
        if not module:
            return None
        if ".." in PurePosixPath(module).parts:
            return None
        base = root_path / module
        suffix = base.suffix
        # Specifier carried a source extension and the file exists as-is.
        if suffix in self._EXTENSIONS and base.is_file():
            return base
        # ESM-style ``./x.js`` written from TypeScript: the on-disk source
        # is ``x.ts``.
        for ts_ext in self._JS_TO_TS.get(suffix, ()):
            ts_candidate = base.with_suffix(ts_ext)
            if ts_candidate.is_file():
                return ts_candidate
        # Extensionless specifier: probe extensions, then directory index.
        for ext in self._EXTENSIONS:
            candidate = base.parent / f"{base.name}{ext}"
            if candidate.is_file():
                return candidate
        for ext in self._EXTENSIONS:
            candidate = base / f"index{ext}"
            if candidate.is_file():
                return candidate
        return None

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _spec_to_candidates(self, spec: str, src_path: Path, root_path: Path) -> List[str]:
        """Map one specifier to root-relative candidate paths."""
        root_resolved = root_path.resolve()
        if spec.startswith("."):
            target = Path(os.path.normpath(src_path.parent / PurePosixPath(spec)))
            try:
                return [target.relative_to(root_resolved).as_posix()]
            except ValueError:
                return []  # walked out of the project tree

        # Non-relative: tsconfig path alias, then baseUrl probing. A bare
        # specifier with no tsconfig mapping is an external package.
        config = self._nearest_tsconfig(src_path.parent, root_resolved)
        if config is None:
            return []
        base_dir, paths = config
        out: List[str] = []
        for pattern, targets in self._matching_alias_patterns(spec, paths):
            star_match = self._star_capture(pattern, spec)
            for target in targets:
                mapped = target.replace("*", star_match) if star_match is not None else target
                candidate = Path(os.path.normpath(base_dir / PurePosixPath(mapped)))
                try:
                    out.append(candidate.relative_to(root_resolved).as_posix())
                except ValueError:
                    continue
        if not out:
            # baseUrl makes bare paths root-anchored (``import 'src/utils'``).
            candidate = Path(os.path.normpath(base_dir / PurePosixPath(spec)))
            try:
                out.append(candidate.relative_to(root_resolved).as_posix())
            except ValueError:
                pass
        return out

    @staticmethod
    def _matching_alias_patterns(
        spec: str, paths: Dict[str, List[str]]
    ) -> List[Tuple[str, List[str]]]:
        """tsconfig ``paths`` patterns matching ``spec``, longest prefix first."""
        matches: List[Tuple[str, List[str]]] = []
        for pattern, targets in paths.items():
            if "*" in pattern:
                prefix, _, tail = pattern.partition("*")
                if spec.startswith(prefix) and spec.endswith(tail):
                    matches.append((pattern, targets))
            elif pattern == spec:
                matches.append((pattern, targets))
        matches.sort(key=lambda item: len(item[0]), reverse=True)
        return matches

    @staticmethod
    def _star_capture(pattern: str, spec: str) -> Optional[str]:
        """The substring the ``*`` wildcard captured, or None for exact patterns."""
        if "*" not in pattern:
            return None
        prefix, _, tail = pattern.partition("*")
        return spec[len(prefix) : len(spec) - len(tail)] if tail else spec[len(prefix) :]

    def _nearest_tsconfig(
        self, start_dir: Path, root_resolved: Path
    ) -> Optional[Tuple[Path, Dict[str, List[str]]]]:
        """Find and parse the nearest tsconfig.json/jsconfig.json walking up.

        Returns (baseUrl dir, paths mapping); ``extends`` chains are not
        followed — only the local ``compilerOptions`` are read.
        """
        key = str(start_dir)
        if key in self._tsconfig_cache:
            return self._tsconfig_cache[key]
        result: Optional[Tuple[Path, Dict[str, List[str]]]] = None
        d = start_dir
        while True:
            for name in ("tsconfig.json", "jsconfig.json"):
                config_path = d / name
                if config_path.is_file():
                    result = self._parse_tsconfig(config_path)
                    break
            if result is not None or d == root_resolved or d == d.parent:
                break
            d = d.parent
        self._tsconfig_cache[key] = result
        return result

    @staticmethod
    def _parse_tsconfig(config_path: Path) -> Optional[Tuple[Path, Dict[str, List[str]]]]:
        """Extract (baseUrl dir, paths) from a JSONC tsconfig, or None."""
        import json

        try:
            text = config_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        # Tolerate JSONC: strip comments and trailing commas.
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
        text = re.sub(r"^\s*//.*$", "", text, flags=re.M)
        text = re.sub(r",\s*([}\]])", r"\1", text)
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return None
        options = data.get("compilerOptions", {}) if isinstance(data, dict) else {}
        if not isinstance(options, dict):
            return None
        base_url = options.get("baseUrl")
        paths_raw = options.get("paths")
        if base_url is None and not paths_raw:
            return None
        base_dir = config_path.parent / (base_url or ".")
        paths: Dict[str, List[str]] = {}
        if isinstance(paths_raw, dict):
            for pattern, targets in paths_raw.items():
                if isinstance(pattern, str) and isinstance(targets, list):
                    paths[pattern] = [t for t in targets if isinstance(t, str)]
        return (base_dir, paths)


class CppImportResolver:
    """Resolve C/C++ ``#include`` directives to project headers.

    Include search paths (-I flags) live in build systems we don't parse,
    so resolution is filesystem-driven instead:

    1. Quoted includes try the including file's directory first (the
       standard preprocessor behavior).
    2. Any include (quoted or angle) is tried as a root-relative path
       (``#include "server/logging/logger.h"`` with ``-I <root>`` is the
       dominant project convention).
    3. Otherwise the include path is suffix-matched against a
       once-per-run index of every project header; a UNIQUE match resolves,
       an ambiguous one is dropped — a wrong edge is worse than a missing
       one (this session's recurring lesson).

    System headers (``<vector>``, ``<sys/types.h>``) match nothing in the
    project index and fall out as external. Candidates are emitted as
    root-relative paths, so ``resolve`` is a bare existence check.
    """

    _INCLUDE_RE = re.compile(r'#\s*include\s*["<]([^">]+)[">]')
    _QUOTED_RE = re.compile(r'#\s*include\s*"')

    #: Files an #include can meaningfully target (headers plus the odd
    #: unity-build source include).
    _INCLUDABLE_SUFFIXES = frozenset(
        {
            ".h",
            ".hpp",
            ".hh",
            ".hxx",
            ".inl",
            ".inc",
            ".ipp",
            ".tcc",
            ".cuh",
            ".c",
            ".cc",
            ".cpp",
            ".cxx",
            ".cu",
        }
    )
    #: Directory names never worth scanning for project headers.
    _PRUNE_DIR_NAMES = frozenset(
        {"node_modules", "target", "third_party", "external", "__pycache__", "dist", "out"}
    )
    _PRUNE_DIR_PREFIXES = (".", "venv", "build", "cmake-build")

    def __init__(self) -> None:
        # basename → root-relative posix paths of project headers, built on
        # first suffix-match lookup (once per resolution run).
        self._header_index: Optional[Dict[str, List[str]]] = None

    def parse(self, raw: str, src_file: str, root_path: Path) -> List[str]:
        match = self._INCLUDE_RE.search(raw)
        if not match:
            return []
        include = match.group(1).strip().replace("\\", "/")
        if not include or include.startswith("/"):
            return []
        is_quoted = self._QUOTED_RE.search(raw) is not None
        root_resolved = root_path.resolve()

        try:
            src_path = Path(src_file).resolve()
            src_path.relative_to(root_resolved)
        except (OSError, ValueError):
            return []

        # 1. Quoted: relative to the including file's directory.
        if is_quoted:
            candidate = Path(os.path.normpath(src_path.parent / PurePosixPath(include)))
            if candidate.is_file():
                try:
                    return [candidate.relative_to(root_resolved).as_posix()]
                except ValueError:
                    return []  # resolved above the project root

        # 2. Root-relative as written (-I <root> convention).
        if ".." not in PurePosixPath(include).parts:
            candidate = root_resolved / include
            if candidate.is_file():
                return [include]

        # 3. Unique suffix match against the project header index.
        index = self._project_header_index(root_resolved)
        basename = include.rsplit("/", 1)[-1]
        matches = [
            path
            for path in index.get(basename, [])
            if path == include or path.endswith("/" + include)
        ]
        if len(matches) == 1:
            return matches
        return []

    def resolve(self, module: str, root_path: Path) -> Optional[Path]:
        if not module or ".." in PurePosixPath(module).parts:
            return None
        candidate = root_path / module
        return candidate if candidate.is_file() else None

    def _project_header_index(self, root_resolved: Path) -> Dict[str, List[str]]:
        """Index every includable project file by basename, pruning junk dirs."""
        if self._header_index is not None:
            return self._header_index
        index: Dict[str, List[str]] = {}
        for dirpath, dirnames, filenames in os.walk(root_resolved):
            dirnames[:] = [
                d
                for d in dirnames
                if d not in self._PRUNE_DIR_NAMES and not d.startswith(self._PRUNE_DIR_PREFIXES)
            ]
            for name in filenames:
                if Path(name).suffix.lower() not in self._INCLUDABLE_SUFFIXES:
                    continue
                rel = (Path(dirpath) / name).relative_to(root_resolved).as_posix()
                index.setdefault(name, []).append(rel)
        self._header_index = index
        return index


class JvmImportResolver:
    """Shared machinery for JVM languages: dotted FQN → source file.

    Source roots (src/main/java, src/test/scala, multi-module Maven
    <module>/src/main/java, bare src/) live in build files we don't parse,
    so the FQN's path form (``com/foo/bar/Baz.<ext>``) is suffix-matched
    against a once-per-run index of project sources — the same
    configuration-free approach as the C++ resolver. A UNIQUE match
    resolves; ambiguity emits nothing.

    Nested types and static members (``com.foo.Bar.Inner``,
    ``import static com.foo.Bar.method``) resolve by progressively
    dropping trailing segments until a file matches. External packages
    (``java.util.List``, library FQNs) match nothing and fall out.
    """

    #: Source suffixes indexed for suffix matching (subclass-specific).
    _SOURCE_SUFFIXES: frozenset = frozenset()

    _PRUNE_DIR_NAMES = frozenset(
        {"node_modules", "target", "third_party", "external", "__pycache__", "dist", "out", "bin"}
    )
    _PRUNE_DIR_PREFIXES = (".", "venv", "build", "cmake-build")

    def __init__(self) -> None:
        # filename → root-relative posix paths, built on first lookup.
        self._source_index: Optional[Dict[str, List[str]]] = None

    def resolve(self, module: str, root_path: Path) -> Optional[Path]:
        if not module or ".." in PurePosixPath(module).parts:
            return None
        candidate = root_path / module
        return candidate if candidate.is_file() else None

    def _resolve_dotted(self, segments: List[str], root_resolved: Path) -> Optional[str]:
        """Unique suffix match for a dotted FQN, dropping trailing segments
        for nested types / static members. Requires at least one package
        segment so bare class names don't bind on filename alone."""
        index = self._source_index_for(root_resolved)
        parts = list(segments)
        while len(parts) >= 2:
            rel_stem = "/".join(parts)
            matches = [
                path
                for suffix in self._SOURCE_SUFFIXES
                for path in index.get(parts[-1] + suffix, [])
                if path == rel_stem + suffix or path.endswith("/" + rel_stem + suffix)
            ]
            if len(matches) == 1:
                return matches[0]
            if matches:
                return None  # ambiguous — a wrong edge is worse than none
            parts.pop()
        return None

    def _source_index_for(self, root_resolved: Path) -> Dict[str, List[str]]:
        if self._source_index is not None:
            return self._source_index
        index: Dict[str, List[str]] = {}
        for dirpath, dirnames, filenames in os.walk(root_resolved):
            dirnames[:] = [
                d
                for d in dirnames
                if d not in self._PRUNE_DIR_NAMES and not d.startswith(self._PRUNE_DIR_PREFIXES)
            ]
            for name in filenames:
                if Path(name).suffix.lower() in self._SOURCE_SUFFIXES:
                    rel = (Path(dirpath) / name).relative_to(root_resolved).as_posix()
                    index.setdefault(name, []).append(rel)
        self._source_index = index
        return index

    @staticmethod
    def _valid_segments(segments: List[str]) -> bool:
        return bool(segments) and all(seg.isidentifier() for seg in segments)


class JavaImportResolver(JvmImportResolver):
    """Resolve Java ``import`` declarations to project sources.

    Handles ``import com.foo.Baz;``, ``import static com.foo.Bar.method;``
    (static member drops to the declaring class), and nested types.
    Wildcard imports (``import com.foo.*;``) are skipped — they name a
    package, not a type, and fanning out to every file in the package
    would re-introduce the multiplicity this metric series removed.
    """

    _SOURCE_SUFFIXES = frozenset({".java"})

    def parse(self, raw: str, src_file: str, root_path: Path) -> List[str]:
        text = raw.strip().removesuffix(";").strip()
        if not text.startswith("import"):
            return []
        text = text[len("import") :].strip()
        if text.startswith("static "):
            text = text[len("static ") :].strip()
        if not text or text.endswith(".*"):
            return []
        segments = [seg.strip() for seg in text.split(".")]
        if not self._valid_segments(segments):
            return []
        resolved = self._resolve_dotted(segments, root_path.resolve())
        return [resolved] if resolved else []


class ScalaImportResolver(JvmImportResolver):
    """Resolve Scala ``import`` clauses to project sources.

    Handles multi-imports (``import a.B, c.D``), selector groups with
    renames (``import a.{B, C => D}`` / Scala 3 ``C as D``), and drops
    wildcards (``a._``, Scala 3 ``a.*``) and hidden members (``B => _``).
    Scala files may define types whose names differ from the filename;
    those simply miss the suffix match and stay unresolved (best effort,
    conventional layouts resolve). The index includes ``.java`` too —
    mixed sbt/Maven projects import Java classes from the same repo.
    """

    _SOURCE_SUFFIXES = frozenset({".scala", ".java"})

    def parse(self, raw: str, src_file: str, root_path: Path) -> List[str]:
        text = raw.strip().removesuffix(";").strip()
        if not text.startswith("import"):
            return []
        text = text[len("import") :].strip()
        if not text:
            return []
        root_resolved = root_path.resolve()

        candidates: List[str] = []
        seen: Set[str] = set()
        for clause in self._split_top_level_commas(text):
            for dotted in self._expand_clause(clause.strip()):
                segments = [seg.strip() for seg in dotted.split(".")]
                if not self._valid_segments(segments):
                    continue
                resolved = self._resolve_dotted(segments, root_resolved)
                if resolved and resolved not in seen:
                    seen.add(resolved)
                    candidates.append(resolved)
        return candidates

    @staticmethod
    def _split_top_level_commas(text: str) -> List[str]:
        parts: List[str] = []
        depth = 0
        current: List[str] = []
        for ch in text:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            elif ch == "," and depth == 0:
                parts.append("".join(current))
                current = []
                continue
            current.append(ch)
        parts.append("".join(current))
        return parts

    @classmethod
    def _expand_clause(cls, clause: str) -> List[str]:
        """One import clause → dotted names (selector groups expanded)."""
        brace = clause.find("{")
        if brace == -1:
            path = clause.strip()
            if path.endswith("._") or path.endswith(".*"):
                path = path[:-2]
            return [path] if path else []
        prefix = clause[:brace].strip().removesuffix(".")
        close = clause.rfind("}")
        if close < brace or not prefix:
            return []
        out: List[str] = []
        for selector in clause[brace + 1 : close].split(","):
            # `B => D` / Scala 3 `B as D`: the real member is the LHS;
            # `B => _` hides a member; bare `_`/`*` are wildcards.
            parts = re.split(r"=>|\bas\b", selector)
            name = parts[0].strip()
            if not name or name in ("_", "*"):
                continue
            if len(parts) > 1 and parts[1].strip() == "_":
                continue  # hidden member, not an import
            out.append(f"{prefix}.{name}")
        return out


ImportResolverRegistry.register("python", PythonImportResolver)
ImportResolverRegistry.register("rust", RustImportResolver)
ImportResolverRegistry.register("javascript", JsTsImportResolver)
ImportResolverRegistry.register("typescript", JsTsImportResolver)
ImportResolverRegistry.register("jsx", JsTsImportResolver)
ImportResolverRegistry.register("tsx", JsTsImportResolver)
ImportResolverRegistry.register("c", CppImportResolver)
ImportResolverRegistry.register("cpp", CppImportResolver)
ImportResolverRegistry.register("java", JavaImportResolver)
ImportResolverRegistry.register("scala", ScalaImportResolver)


__all__ = [
    "CppImportResolver",
    "ImportResolverRegistry",
    "JavaImportResolver",
    "JsTsImportResolver",
    "JvmImportResolver",
    "LanguageImportResolver",
    "PythonImportResolver",
    "RustImportResolver",
    "ScalaImportResolver",
]

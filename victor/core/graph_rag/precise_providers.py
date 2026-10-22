"""ADR-032 tier 3 precise-index providers (contract: ADR-032 rev 1.5).

Each precise indexer is built on its language's reference toolchain and
drags that runtime along (scip-python → Node/Pyright, scip-typescript →
Node/tsserver, rust-analyzer → native Rust). The provider contract makes
runtimes a provider detail:

- ``detect()`` — locate the toolchain binary (never installs, never raises);
- ``dump()`` — one offline subprocess dump with the provider's argv/env/cwd
  shape (victor owns the clock; the crash fence is the process boundary);
- ``symbol_parts()`` — the PRODUCER's symbol grammar. Grammars are producer
  properties, not language properties (verified the hard way: scip-python
  0.6.x emits package-qualified descriptors that match no documentation),
  so each producer parses its own symbol strings here.

All three current producers emit SCIP, so the protobuf reader stays shared
(``precise_index._read_index``); a future non-SCIP producer (LSIF adapter,
in-process Jedi) implements its own read against the same normalized parts.

Symbol-grammar facts below are pinned empirically against real dumps
(scip-python 0.6.6, scip-typescript 0.5.1, rust-analyzer 1.99); the unit
tests reproduce the captured shapes.

Tier-3.5 payloads (contract: "ride from day one"): ``symbol_parts`` carries
``syntax_kind`` when the occurrence provides it, and the join may consume
name/scope/kind today while later tiers read the rest without re-dumping.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol

logger = logging.getLogger(__name__)

# Node-based producers hold the whole program in memory; Node's default
# ~4 GB heap aborts (V8 OOM) around ~2,200 files. Injected when the
# environment does not set NODE_OPTIONS (shared by every Node producer).
NODE_HEAP_OPTIONS = "--max-old-space-size=16384"


class PreciseIndexProvider(Protocol):
    """Rev 1.5 contract. See the module docstring and ADR-032."""

    @property
    def name(self) -> str: ...

    @property
    def languages(self) -> frozenset[str]: ...

    def detect(self, tool_hint: str = "auto") -> Optional[str]:
        """Resolved tool path, or None when the toolchain is absent."""
        ...

    def dump(
        self,
        tool: str,
        root: Path,
        output_path: Path,
        *,
        timeout_seconds: int,
    ) -> Optional[Path]:
        """One offline dump; the artifact path, or None on any failure."""
        ...

    def symbol_parts(self, symbol: str) -> Optional[dict[str, Any]]:
        """Producer grammar: symbol string -> normalized parts or None."""
        ...


def _run_dump(
    command: list[str],
    *,
    cwd: Path,
    output_path: Path,
    timeout_seconds: int,
    node_heap: bool,
) -> Optional[Path]:
    """Shared SUBPROCESS transport: bounded, captured, never raises."""

    env = dict(os.environ)
    if node_heap:
        env.setdefault("NODE_OPTIONS", NODE_HEAP_OPTIONS)
    try:
        result = subprocess.run(  # noqa: S603 - fixed argv from detect()
            command,
            capture_output=True,
            text=True,
            cwd=str(cwd),
            timeout=timeout_seconds,
            check=False,
            env=env,
        )
    except subprocess.TimeoutExpired:
        logger.warning(
            "precise-index dump timed out after %ds (raise precise_index_timeout_seconds "
            "for large corpora)",
            timeout_seconds,
        )
        return None
    except OSError as exc:
        logger.warning("precise-index dump failed (%s): %s", type(exc).__name__, exc)
        return None
    if result.returncode != 0 or not output_path.exists():
        logger.warning(
            "precise-index dump exited %d: %s",
            result.returncode,
            (result.stderr or "").strip()[:400],
        )
        return None
    return output_path


def _split_descriptor(symbol: str, prefix: str, tokens_before_path: int) -> Optional[str]:
    """Strip ``<scheme tokens...>`` and return the descriptor path.

    Producers qualify symbols with leading whitespace tokens (package name,
    version, ...). None of the current descriptors contain spaces, but a
    fixed-token split tolerates them better than rsplit when a future
    producer adds a token.
    """

    if not symbol.startswith(prefix):
        return None
    parts = symbol.split(" ", tokens_before_path)
    if len(parts) <= tokens_before_path or not parts[-1]:
        return None
    return parts[-1]


def _strip_backticks(name: str) -> str:
    return name.removeprefix("`").removesuffix("`")


class ScipPythonProvider:
    """npm ``scip-python`` (Pyright) — python."""

    name = "scip-python"
    languages = frozenset({"python"})
    # Measured ~500s on the victor monorepo (6,707 files) under the heap
    # hint; generous headroom for bigger corpora.
    default_timeout_seconds = 1200

    def detect(self, tool_hint: str = "auto") -> Optional[str]:
        if tool_hint not in ("auto", self.name) and Path(tool_hint).exists():
            return tool_hint
        return shutil.which(self.name)

    def dump(
        self,
        tool: str,
        root: Path,
        output_path: Path,
        *,
        timeout_seconds: int,
    ) -> Optional[Path]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return _run_dump(
            [tool, "index", "--project-name", "victor-harvest", "--output", str(output_path)],
            cwd=root,
            output_path=output_path,
            timeout_seconds=timeout_seconds,
            node_heap=True,
        )

    def symbol_parts(self, symbol: str) -> Optional[dict[str, Any]]:
        # Grammar (verified, scip-python 0.6.6): ``scip-python python <pkg>
        # <ver> \`dotted.module\`/Class#method().`` — backticked dotted module,
        # '#' class members, '()' callables, ':' module inits, '(...)' params.
        path = _split_descriptor(symbol, "scip-python python ", 4)
        if path is None:
            return None
        if "/" not in path and "#" not in path:
            return None
        if path.endswith(":") or path.endswith(")"):
            return None
        segments = path.rstrip(".").split("#")
        head = [c for c in segments[0].split("/") if c]
        if not head or (len(segments) == 1 and len(head) < 2):
            return None
        if len(segments) == 1:
            if not head[-1].endswith("()"):
                return None  # module-level assignment, not a symbol node
            scope, name = [], head[-1].removesuffix("()")
        elif len(segments) == 2 and not segments[1]:
            scope, name = [], _strip_backticks(head[-1])
        else:
            if not segments[-1].endswith("()"):
                return None  # attribute/field: no codegraph node
            scope = [_strip_backticks(head[-1])]
            scope += [_strip_backticks(c) for seg in segments[1:-1] for c in seg.split("/") if c]
            name = _strip_backticks(segments[-1].removesuffix("()"))
        if not name:
            return None
        return {"file": "", "path": ".".join([*scope, name])}


class ScipTypescriptProvider:
    """npm ``scip-typescript`` (tsserver) — TypeScript/JavaScript."""

    name = "scip-typescript"
    languages = frozenset({"typescript"})
    # Measured ~3s on vscode-victor (89 docs).
    default_timeout_seconds = 600

    def detect(self, tool_hint: str = "auto") -> Optional[str]:
        if tool_hint not in ("auto", self.name) and Path(tool_hint).exists():
            return tool_hint
        return shutil.which(self.name)

    def tsconfig_projects(self, root: Path) -> list[Path]:
        """Projects to index: root's tsconfig when present, plus any
        first-level subdirectory's (monorepo layouts like victor's
        vscode-victor/). Deterministic; no recursion."""

        found = []
        if (root / "tsconfig.json").exists():
            found.append(root)
        for candidate in sorted(root.glob("*/tsconfig.json")):
            if candidate.parent not in found:
                found.append(candidate.parent)
        return found

    def dump(
        self,
        tool: str,
        root: Path,
        output_path: Path,
        *,
        timeout_seconds: int,
    ) -> Optional[Path]:
        projects = self.tsconfig_projects(root)
        if not projects:
            logger.info("precise-index: no tsconfig.json at %s or one level down; skipping", root)
            return None
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return _run_dump(
            [
                tool,
                "index",
                "--cwd",
                str(root),
                "--output",
                str(output_path),
                *(str(p) for p in projects),
            ],
            cwd=root,
            output_path=output_path,
            timeout_seconds=timeout_seconds,
            node_heap=True,
        )

    def symbol_parts(self, symbol: str) -> Optional[dict[str, Any]]:
        # Grammar (verified, scip-typescript 0.5.1): ``scip-typescript npm
        # <pkg> <ver> src/\`file.ts\`/Class#method().`` — the head is the
        # BACKTICKED FILE PATH (not a dotted module); members follow.
        path = _split_descriptor(symbol, "scip-typescript npm ", 4)
        if path is None:
            return None
        # File prefix: everything through the closing backtick of the file
        # name, plus its trailing '/'. Symbols without a backticked file
        # head are not joinable here.
        close = path.find("`", path.find("`") + 1)
        if close == -1:
            return None
        members = path[close + 1 :].lstrip("/")
        if not members or members.endswith(":") or members.endswith(")"):
            return None
        segments = members.rstrip(".").split("#")
        head = [c for c in segments[0].split("/") if c]
        if not head:
            return None
        if len(segments) == 1:
            if not head[-1].endswith("()"):
                return None  # top-level const, not a symbol node
            scope, name = [], _strip_backticks(head[-1].removesuffix("()"))
        elif len(segments) == 2 and not segments[1]:
            scope, name = [], _strip_backticks(head[-1])
        else:
            if not segments[-1].endswith("()"):
                return None  # property/field: no codegraph node
            scope = [_strip_backticks(head[-1])]
            scope += [_strip_backticks(c) for seg in segments[1:-1] for c in seg.split("/") if c]
            name = _strip_backticks(segments[-1].removesuffix("()"))
        if not name:
            return None
        # codegraph names the synthesized constructor member literally.
        if name == "<constructor>":
            name = "constructor"
        return {"file": "", "path": ".".join([*scope, name])}


class RustAnalyzerProvider:
    """``rust-analyzer scip`` — Rust (native; needs cargo metadata)."""

    name = "rust-analyzer"
    languages = frozenset({"rust"})
    # Measured ~13s on the victor rust workspace (30 docs, 10 crates).
    default_timeout_seconds = 120

    def detect(self, tool_hint: str = "auto") -> Optional[str]:
        if tool_hint not in ("auto", self.name) and Path(tool_hint).exists():
            return tool_hint
        return shutil.which(self.name)

    def crate_root(self, root: Path) -> Optional[Path]:
        """Workspace discovery: root's Cargo.toml, else a first-level
        subdirectory's (monorepo layouts like victor's rust/)."""

        if (root / "Cargo.toml").exists():
            return root
        for candidate in sorted(root.glob("*/Cargo.toml")):
            return candidate.parent
        return None

    def dump(
        self,
        tool: str,
        root: Path,
        output_path: Path,
        *,
        timeout_seconds: int,
    ) -> Optional[Path]:
        import os

        crate_root = self.crate_root(root)
        if crate_root is None:
            logger.info("precise-index: no Cargo.toml at %s or one level down; skipping", root)
            return None
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # rust-analyzer scip takes the project PATH and ALWAYS writes
        # ``index.scip`` to the process CWD (no --output flag): run in the
        # artifact directory, validate the FIXED name, then move it to the
        # caller's path. Validating output_path directly would fail — the
        # tool never writes there (caught by a real pipeline validation;
        # the argv-level tests could not see it).
        produced = output_path.parent / "index.scip"
        result = _run_dump(
            [tool, "scip", str(crate_root)],
            cwd=output_path.parent,
            output_path=produced,
            timeout_seconds=timeout_seconds,
            node_heap=False,
        )
        if result is None:
            return None
        if produced != output_path:
            os.replace(produced, output_path)
        return output_path

    def symbol_parts(self, symbol: str) -> Optional[dict[str, Any]]:
        # Grammar (verified, rust-analyzer 1.99): ``rust-analyzer cargo
        # <crate> <ver> <descriptor>`` where the descriptor is module/ runs
        # (``crate/``, ``extra/``), ``Struct#`` types, ``Struct#field.``
        # fields, and ``impl#[T]method().`` impl members. Top-level items
        # carry NO module prefix (``helper().``) — unlike python, a bare
        # callable name IS joinable. Macro calls end ``!``; external crates
        # (alloc/core/std) fail the out-of-corpus check in the join.
        path = _split_descriptor(symbol, "rust-analyzer cargo ", 4)
        if path is None:
            return None
        if path.endswith(":") or path.endswith(")") or path.endswith("!"):
            return None
        head, sep, tail = path.rstrip(".").partition("#")
        head_parts = [c for c in head.split("/") if c]
        if not sep:
            # No '#': module path then a callable name (modules dropped —
            # codegraph scopes contain types/functions, not modules).
            if not head_parts or not head_parts[-1].endswith("()"):
                return None  # module decl or non-callable
            return {
                "file": "",
                "path": _strip_backticks(head_parts[-1].removesuffix("()")),
            }
        if not tail:
            # Struct/enum class form ``module/Struct#``: the trailing head
            # component is the type name; module components drop.
            if not head_parts:
                return None
            return {"file": "", "path": _strip_backticks(head_parts[-1])}
        if head_parts and head_parts[-1] == "impl":
            # impl#[Type]member(). — the impl'd type(s) lead the tail in
            # [...] groups; each becomes a scope component.
            scope: list[str] = []
            rest = tail
            while rest.startswith("["):
                end = rest.find("]")
                if end == -1:
                    return None
                scope.append(_strip_backticks(rest[1:end]))
                rest = rest[end + 1 :]
            if not rest.endswith("()"):
                return None  # impl const/field: no codegraph node
            name = _strip_backticks(rest.removesuffix("()"))
            if not name or not scope:
                return None
            return {"file": "", "path": ".".join([*scope, name])}
        # Type#member(). — the trailing head component (the type) is the
        # scope; module components before it drop, matching the class form
        # and codegraph scopes (types/functions only, never modules).
        if not head_parts or not tail.endswith("()"):
            return None
        scope = _strip_backticks(head_parts[-1])
        name = _strip_backticks(tail.removesuffix("()"))
        if not name or not scope:
            return None
        return {"file": "", "path": f"{scope}.{name}"}


@dataclass(frozen=True)
class _Registered:
    provider: PreciseIndexProvider
    extensions: frozenset[str]


# The language registry (contract: keyed by language, one flag). File
# extension → language; tsx/jsx/js all route to the typescript producer.
_REGISTRY: tuple[_Registered, ...] = (
    _Registered(ScipPythonProvider(), frozenset({".py"})),
    _Registered(ScipTypescriptProvider(), frozenset({".ts", ".tsx", ".js", ".jsx", ".mjs"})),
    _Registered(RustAnalyzerProvider(), frozenset({".rs"})),
)


def providers_for_files(files: list[str]) -> dict[str, PreciseIndexProvider]:
    """Providers worth running for a set of persisted node file paths,
    keyed by provider name — one dump per language actually present."""

    wanted: dict[str, PreciseIndexProvider] = {}
    for registered in _REGISTRY:
        for file in files:
            if Path(file).suffix.lower() in registered.extensions:
                wanted[registered.provider.name] = registered.provider
                break
    return wanted


def all_providers() -> tuple[PreciseIndexProvider, ...]:
    return tuple(registered.provider for registered in _REGISTRY)


def extensions_for_provider(provider_name: str) -> frozenset[str]:
    """The file extensions a provider's cache slice covers."""

    for registered in _REGISTRY:
        if registered.provider.name == provider_name:
            return registered.extensions
    return frozenset()

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

All current producers emit SCIP, so the protobuf reader stays shared
(``precise_index._read_index``); a future non-SCIP producer (LSIF adapter,
in-process Jedi) implements its own read against the same normalized parts.

Symbol-grammar facts below are pinned empirically against real dumps
(scip-python 0.6.6, scip-typescript 0.5.1, rust-analyzer 1.99, scip-clang
0.4.1, scip-java 0.13.2, scip-ruby 0.5.1, scip-go 0.2.9); the unit tests
reproduce the captured shapes.

Tier-3.5 payloads (contract: "ride from day one"): ``symbol_parts`` carries
``syntax_kind`` when the occurrence provides it, and the join may consume
name/scope/kind today while later tiers read the rest without re-dumping.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol, Union

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
    ) -> Union[Path, DumpSkipped, None]:
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


def _descriptor_after_tokens(symbol: str, scheme: str, tokens: int) -> Optional[str]:
    """Split off the first ``tokens`` space-separated tokens, verifying the
    scheme — for producers whose package/version tokens vary per project
    (scip-java, scip-ruby) and so have no fixed prefix string."""

    parts = symbol.split(" ", tokens)
    if len(parts) <= tokens or parts[0] != scheme or not parts[-1]:
        return None
    return parts[-1]


# scip-clang suffixes callables with a 16-hex SIGNATURE hash — not an
# identity: ``helper(int)`` and ``overload(int)`` share one. Stripped before
# the join; identity rides name+scope like the kind-dormant producers.
_CXX_SIGNATURE_HASH = re.compile(r"\([0-9a-f]{16}\)$")


@dataclass(frozen=True)
class DumpSkipped:
    """A provider refusing to dump because its PROJECT MARKER is absent —
    the honest taxonomy's third state, distinct from ``None`` (toolchain or
    dump failure). Conflating them labeled a Homebrew formula's repo
    ``dump-failed`` when the truth was "no Gemfile setup" (rev 1.9 fallout)."""

    reason: str


def _detect_tool(name: str, tool_hint: str) -> Optional[str]:
    """Shared detection: an existing-path hint overrides, else PATH lookup."""
    if tool_hint not in ("auto", name) and Path(tool_hint).exists():
        return tool_hint
    return shutil.which(name)


def _sharp_chain_path(body, scope_component, callable_name) -> Optional[str]:
    """Parse a ``#``-chained descriptor body — the shared spine of the
    cxx/java/go producers (kept out of the providers so the duplication
    ratchet holds).

    A body ending ``#`` is a TYPE CHAIN (every '#' segment is a type; the
    '/'-joined prefixes are namespaces/packages and drop per
    ``scope_component``). Otherwise the last segment must parse as a
    callable through ``callable_name`` (None ⇒ field/non-callable ⇒ no
    codegraph node) and the earlier segments form the scope; a single
    segment is a free function.
    """

    segments = body.split("#")
    if body.endswith("#"):
        types = [t for t in (scope_component(seg) for seg in segments[:-1]) if t]
        return ".".join(types) if types else None
    if len(segments) == 1:
        return callable_name(scope_component(segments[0]))
    scope = [scope_component(seg) for seg in segments[:-1]]
    if not scope or not all(scope):
        return None
    name = callable_name(segments[-1])
    if name is None:
        return None
    return ".".join([*scope, name])


def _dump_fixed_name(
    command: list[str],
    *,
    cwd: Path,
    produced: Path,
    output_path: Path,
    timeout_seconds: int,
) -> Optional[Path]:
    """Shared fixed-name dump contract: the tool writes ``produced`` in its
    CWD (no --output flag); validate that exact name, then move it to the
    artifact path (the rust-analyzer contract)."""

    result = _run_dump(
        command,
        cwd=cwd,
        output_path=produced,
        timeout_seconds=timeout_seconds,
        node_heap=False,
    )
    if result is None:
        return None
    if produced != output_path:
        os.replace(produced, output_path)
    return output_path


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
        environment_json: Optional[Path] = None,
    ) -> Union[Path, DumpSkipped, None]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        argv = [tool, "index", "--project-name", "victor-harvest", "--output", str(output_path)]
        if environment_json is not None:
            # ADR-032 harvest constraint: skip Pyright's whole-environment
            # enumeration (the measured 2,980 s / ~130 GB-read hot spot on
            # shared conda envs) by handing it exactly the distributions
            # the codebase imports. The file is schema-validated by its
            # builder; a failed build omits this flag (unconstrained run).
            argv += ["--environment", str(environment_json)]
        return _run_dump(
            argv,
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
    ) -> Union[Path, DumpSkipped, None]:
        projects = self.tsconfig_projects(root)
        if not projects:
            logger.info("precise-index: no tsconfig.json at %s or one level down; skipping", root)
            return DumpSkipped("no-tsconfig")
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
    ) -> Union[Path, DumpSkipped, None]:
        import os

        crate_root = self.crate_root(root)
        if crate_root is None:
            logger.info("precise-index: no Cargo.toml at %s or one level down; skipping", root)
            return DumpSkipped("no-cargo-toml")
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


class ScipClangProvider:
    """``scip-clang`` — C/C++ (native; needs a compilation database)."""

    name = "scip-clang"
    languages = frozenset({"c", "cpp"})
    # Measured 0.1s on the 5-TU capture project; real template-heavy TUs are
    # slower and the tool parallelizes by default, so typescript-class budget.
    default_timeout_seconds = 600

    def detect(self, tool_hint: str = "auto") -> Optional[str]:
        return _detect_tool(self.name, tool_hint)

    def compilation_database(self, root: Path) -> Optional[Path]:
        """Compile-database discovery: root, root/build, or one level down
        (out-of-source cmake layouts). Never generates one — building is the
        user's build system's job, not a dump side effect."""

        for candidate in (
            root / "compile_commands.json",
            root / "build" / "compile_commands.json",
            *sorted(root.glob("*/compile_commands.json")),
            *sorted(root.glob("*/build/compile_commands.json")),
        ):
            if candidate.exists():
                return candidate
        return None

    def dump(
        self,
        tool: str,
        root: Path,
        output_path: Path,
        *,
        timeout_seconds: int,
    ) -> Union[Path, DumpSkipped, None]:
        compdb = self.compilation_database(root)
        if compdb is None:
            logger.info(
                "precise-index: no compile_commands.json at %s or one level down "
                "(generate with cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON or bear); skipping",
                root,
            )
            return DumpSkipped("no-compile-database")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # scip-clang filters documents to project_root = CWD — running from
        # the artifact directory silently produces an EMPTY index (caught by
        # real-pipeline validation; argv mocks cannot see it). cwd is root.
        return _run_dump(
            [
                tool,
                "--compdb-path",
                str(compdb),
                "--index-output-path",
                str(output_path),
                "--no-progress-report",
            ],
            cwd=root,
            output_path=output_path,
            timeout_seconds=timeout_seconds,
            node_heap=False,
        )

    @staticmethod
    def _callable_name(segment: str) -> Optional[str]:
        match = _CXX_SIGNATURE_HASH.search(segment)
        if match is None:
            return None  # field, or a hash-less callable variant: refuse to guess
        return _strip_backticks(segment[: match.start()]) or None

    def symbol_parts(self, symbol: str) -> Optional[dict[str, Any]]:
        # Grammar (verified, scip-clang 0.4.1): ``cxx . . $ <descriptor>`` —
        # EMPTY package/version tokens emitted as literal '.', then '$'.
        # Descriptors: namespaces ``ns1/ns2/``, types ``ns/Type#`` (nested
        # types chain '#': ``Outer#Inner#``; C structs share the shape),
        # callables ``name(<hash>).`` / ``Type#member(<hash>).``, fields
        # ``Type#field.``, macros ``<loc>`!``, file pseudo-symbols
        # `` `<file>/path`/ ``. Templates are transparent (``Box#``); the
        # parenthesized suffix is a signature hash (see _CXX_SIGNATURE_HASH).
        # Namespace components (the '/'-joined prefixes) drop — codegraph
        # scopes carry types, never namespaces (the rust module rule).
        path = _split_descriptor(symbol, "cxx . . $ ", 4)
        if path is None:
            return None
        if path.endswith("!") or path.endswith("`/"):
            return None  # macros; file pseudo-symbols
        body = path.removesuffix(".")
        if not body or body.endswith("/"):
            return None  # namespaces
        parsed = _sharp_chain_path(
            body,
            scope_component=lambda seg: _strip_backticks(seg.split("/")[-1]) or None,
            callable_name=self._callable_name,
        )
        return {"file": "", "path": parsed} if parsed else None


class ScipJavaProvider:
    """``scip-java`` — Java on Gradle/Maven, or explicitly configured Bazel.

    Current 0.13.x releases support Java 17/21/25, not Scala/sbt or Mill.
    Bazel requires the absolute indexer path for its compiler aspect.
    """

    name = "scip-java"
    languages = frozenset({"java"})
    # JVM + build-tool startup measured ~12s on the 2-file capture project;
    # real JVM builds are minutes-class → pyright-class budget.
    default_timeout_seconds = 1200

    _AUTOMATIC_BUILD_MARKERS = (
        "pom.xml",
        "build.gradle",
        "build.gradle.kts",
        "settings.gradle",
        "settings.gradle.kts",
        "gradlew",
    )
    _BAZEL_BUILD_MARKERS = (
        "WORKSPACE",
        "WORKSPACE.bazel",
        "MODULE.bazel",
    )
    _BUILD_MARKERS = _AUTOMATIC_BUILD_MARKERS + _BAZEL_BUILD_MARKERS

    def detect(self, tool_hint: str = "auto") -> Optional[str]:
        return _detect_tool(self.name, tool_hint)

    def dump(
        self,
        tool: str,
        root: Path,
        output_path: Path,
        *,
        timeout_seconds: int,
    ) -> Union[Path, DumpSkipped, None]:
        if not any((root / marker).exists() for marker in self._BUILD_MARKERS):
            logger.info(
                "precise-index: no supported JVM build file (Maven/Gradle/Bazel) at %s; skipping",
                root,
            )
            return DumpSkipped("no-jvm-build")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        command = [tool, "index", f"--output={output_path}"]
        if any((root / marker).exists() for marker in self._BAZEL_BUILD_MARKERS):
            # Modern Bazel roots can have MODULE.bazel without WORKSPACE.
            # Supply its required aspect input even in mixed-build roots; the
            # indexer's Gradle/Maven integrations ignore this Bazel-only option.
            command.append(f"--bazel-scip-java-binary={Path(tool).resolve()}")
        return _run_dump(
            command,
            cwd=root,
            output_path=output_path,
            timeout_seconds=timeout_seconds,
            node_heap=False,
        )

    @staticmethod
    def _callable_name(segment: str) -> Optional[str]:
        match = re.search(r"\((?:\+[1-9][0-9]*)?\)$", segment)
        if match is None:
            return None  # fields / interface-style signatures: no codegraph node
        # +N is the producer's overload ordinal, not a signature or arity.
        # Actual definition positions disambiguate overloads in the join.
        name = _strip_backticks(segment[: match.start()])
        if not name:
            return None
        if name == "<init>":
            # codegraph names the synthesized constructor member literally
            # (same mapping as scip-typescript).
            return "constructor"
        return name

    def symbol_parts(self, symbol: str) -> Optional[dict[str, Any]]:
        # Grammar (verified, scip-java 0.13.2, maven): ``scip-java maven
        # <module-coordinate> <version> com/pkg/Class#method().`` — the
        # leading tokens VARY per project (no fixed prefix), package
        # components drop like namespaces, kinds ARE populated (the kind map
        # handles them), and constructors emit the ``<init>`` literal.
        path = _descriptor_after_tokens(symbol, "scip-java", 4)
        if path is None:
            return None
        body = path.removesuffix(".")
        if not body:
            return None
        parsed = _sharp_chain_path(
            body,
            scope_component=lambda seg: _strip_backticks(seg.split("/")[-1]) or None,
            callable_name=self._callable_name,
        )
        return {"file": "", "path": parsed} if parsed else None


class ScipRubyProvider:
    """``scip-ruby`` (Sorbet) — Ruby.

    The gem's wrapper refuses direct invocation: the toolchain only runs as
    ``bundle exec scip-ruby .`` inside a project whose Gemfile includes the
    scip-ruby gem. ``detect()`` therefore requires BOTH bundler and the gem
    binary on PATH; a project without that Gemfile setup degrades to a
    named dump failure rather than a toolchain-absent lie.
    """

    name = "scip-ruby"
    languages = frozenset({"ruby"})
    # Sorbet on large Rails corpora is minutes-class → pyright-class budget.
    default_timeout_seconds = 900

    def detect(self, tool_hint: str = "auto") -> Optional[str]:
        if tool_hint not in ("auto", self.name) and Path(tool_hint).exists():
            return tool_hint  # override = path to the bundler to run through
        bundle = shutil.which("bundle")
        if bundle and shutil.which("scip-ruby"):
            return bundle
        return None

    def dump(
        self,
        tool: str,
        root: Path,
        output_path: Path,
        *,
        timeout_seconds: int,
    ) -> Union[Path, DumpSkipped, None]:
        if not (root / "Gemfile").exists():
            logger.info(
                "precise-index: no Gemfile at %s; skipping "
                "(add gem 'scip-ruby' to the Gemfile and bundle install)",
                root,
            )
            return DumpSkipped("no-gemfile")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # No --output flag exists: index.scip lands in the CWD (the repo
        # root) — the shared fixed-name contract moves it to the artifact.
        return _dump_fixed_name(
            [tool, "exec", "scip-ruby", "."],
            cwd=root,
            produced=root / "index.scip",
            output_path=output_path,
            timeout_seconds=timeout_seconds,
        )

    @staticmethod
    def _scope_component(segment: str) -> Optional[str]:
        component = _strip_backticks(segment.split("/")[-1])
        if component.startswith("<Class:") and component.endswith(">"):
            # Singleton-class scope for `def self.x` methods — the metaclass
            # literal maps to the class itself, matching how codegraph scopes
            # class methods.
            return component[len("<Class:") : -1]
        return component or None

    @staticmethod
    def _callable_name(segment: str) -> Optional[str]:
        segment = _strip_backticks(segment)
        if not segment.endswith("()"):
            return None  # ivars/fields: no codegraph node
        name = segment.removesuffix("()")
        return name or None

    def symbol_parts(self, symbol: str) -> Optional[dict[str, Any]]:
        # Grammar (verified, scip-ruby 0.5.1): ``scip-ruby gem <name> <ver>
        # Module#Class#method().`` — module/class nesting is KEPT as scope
        # (ruby modules are real lexical nesting, unlike cxx/java
        # namespaces); singleton methods ride `` `<Class:Toy>` `` in any
        # position; ivars `` `@name` `` are fields; kinds are dormant.
        path = _descriptor_after_tokens(symbol, "scip-ruby", 4)
        if path is None:
            return None
        body = path.removesuffix(".")
        if not body:
            return None
        parsed = _sharp_chain_path(
            body,
            scope_component=self._scope_component,
            callable_name=self._callable_name,
        )
        return {"file": "", "path": parsed} if parsed else None


class ScipGoProvider:
    """``scip-go`` — Go (native, gopls-based; needs go.mod)."""

    name = "scip-go"
    languages = frozenset({"go"})
    # Measured ~1s on the 2-package capture project; gopls-class tooling
    # stays fast on large corpora — typescript-class budget for headroom.
    default_timeout_seconds = 600

    def detect(self, tool_hint: str = "auto") -> Optional[str]:
        return _detect_tool(self.name, tool_hint)

    def module_root(self, root: Path) -> Optional[Path]:
        """Module discovery: root's go.mod, else a first-level
        subdirectory's (multi-module layouts)."""

        if (root / "go.mod").exists():
            return root
        for candidate in sorted(root.glob("*/go.mod")):
            return candidate.parent
        return None

    def dump(
        self,
        tool: str,
        root: Path,
        output_path: Path,
        *,
        timeout_seconds: int,
    ) -> Union[Path, DumpSkipped, None]:
        module_root = self.module_root(root)
        if module_root is None:
            logger.info("precise-index: no go.mod at %s or one level down; skipping", root)
            return DumpSkipped("no-go-module")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # No --output flag: index.scip lands in the CWD — run from the
        # module root (its package resolution) under the shared fixed-name
        # contract.
        return _dump_fixed_name(
            [tool, "index"],
            cwd=module_root,
            produced=module_root / "index.scip",
            output_path=output_path,
            timeout_seconds=timeout_seconds,
        )

    @staticmethod
    def _callable_name(segment: str) -> Optional[str]:
        segment = _strip_backticks(segment)
        if not segment.endswith("()"):
            return None  # interface method signatures / fields: non-callable
        name = segment.removesuffix("()")
        return name or None

    def symbol_parts(self, symbol: str) -> Optional[dict[str, Any]]:
        # Grammar (verified, scip-go 0.2.9): ``scip-go gomod <module> . `pkg/import/path`/Sym#member().``
        # — FOUR leading tokens (version is the literal '.'), the head is
        # the BACKTICKED IMPORT PATH whose components drop (codegraph scopes
        # are receiver/file-local, never import paths); kinds ARE populated
        # (Struct/Interface/Method/Function/Field); pointer receivers share
        # the value-receiver shape (``Box#Set().``); interface method
        # SIGNATURES are non-callable (``Speaker#Speak.``) and stay
        # unjoinable like fields.
        path = _descriptor_after_tokens(symbol, "scip-go", 4)
        if path is None:
            return None
        if path.endswith("`/"):
            return None  # package pseudo-symbols
        body = path.removesuffix(".")
        if not body or body.endswith("/"):
            return None
        parsed = _sharp_chain_path(
            body,
            scope_component=lambda seg: _strip_backticks(seg.split("/")[-1]) or None,
            callable_name=self._callable_name,
        )
        return {"file": "", "path": parsed} if parsed else None


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
    _Registered(
        ScipClangProvider(),
        frozenset({".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh", ".hxx"}),
    ),
    _Registered(ScipJavaProvider(), frozenset({".java"})),
    _Registered(ScipRubyProvider(), frozenset({".rb", ".rbi"})),
    _Registered(ScipGoProvider(), frozenset({".go"})),
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

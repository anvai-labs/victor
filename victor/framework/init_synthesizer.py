"""Init.md synthesis via Agent framework.

Generates project context files using the full AgentOrchestrator
infrastructure (logging, events, GEPA traces, tool execution).

Two modes:
- synthesize(base_content, agent=ctx.agent) — reuses existing orchestrator (slash command)
- synthesize(base_content, provider="ollama") — creates fresh Agent (CLI)
- synthesize_with_tools(agent=...) — fallback when victor-coding unavailable
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator

logger = logging.getLogger(__name__)

INIT_PROVIDER_TIMEOUT_SECONDS = 120
INIT_PROVIDER_MAX_RETRIES = 0

# Local providers run inference on the box; a large (~20K-char) synthesis prompt
# to a big local model (e.g. a 31B) needs a generous budget or it times out and
# silently drops to the template. Cloud providers respond faster.
_LOCAL_INIT_PROVIDERS = frozenset(
    {"ollama", "lmstudio", "llamacpp", "llama_cpp", "llama-cpp", "mlx", "vllm"}
)


def _init_provider_timeout(provider_name: str, configured: Optional[float] = None) -> int:
    """Resolve the init-synthesis request timeout in seconds.

    Honors ``VICTOR_INIT_SYNTH_TIMEOUT`` as an explicit override. Otherwise local
    providers get a larger budget than cloud providers, and any timeout already
    configured for the provider is never reduced.
    """
    override = os.environ.get("VICTOR_INIT_SYNTH_TIMEOUT")
    if override:
        try:
            return max(1, int(float(override)))
        except ValueError:
            logger.debug("[init] Ignoring invalid VICTOR_INIT_SYNTH_TIMEOUT=%r", override)

    base = 600 if (provider_name or "").lower() in _LOCAL_INIT_PROVIDERS else 300
    if configured is not None:
        try:
            return max(int(float(configured)), base)
        except (TypeError, ValueError):
            return base
    return base


def _classify_init_failure(error: Exception) -> str:
    """Return a short, log-friendly category for an init-synthesis failure.

    Distinguishes timeout vs connection vs empty-output vs other so the template
    fallback is explainable rather than a generic warning.
    """
    message = str(error).lower()
    if "timed out" in message or "timeout" in message:
        return "timeout"
    if "connection" in message or "unavailable" in message or isinstance(error, OSError):
        return "connection"
    if "empty" in message:
        return "empty_output"
    if "rate" in message and "limit" in message:
        return "rate_limit"
    return type(error).__name__


_ZAI_PROVIDER_ALIASES = frozenset(
    {"zai", "zhipu", "zhipuai", "zai-coding", "zai-coding-plan", "glm-coding"}
)
_ZAI_CODING_PROVIDER_ALIASES = frozenset({"zai-coding", "zai-coding-plan", "glm-coding"})


@dataclass(frozen=True)
class InitProviderBootstrap:
    """Profile-aware provider bootstrap settings for init synthesis."""

    provider_name: str
    request_model: Optional[str]
    provider_init_kwargs: dict[str, Any]
    temperature: float
    max_tokens: int


# Frame: fixed structure that wraps the evolvable RULES section.
# GEPA evolves SYNTHESIS_RULES only; the frame is never mutated.
_SYNTHESIS_FRAME_BEFORE = """You are writing an init.md file — a compact system-prompt context
file that an AI coding assistant reads at the start of every conversation about this codebase.
Think of it like a CLAUDE.md or Cursor rules file: the goal is to give an AI assistant
enough context to navigate the codebase confidently without reading every file.

Below is raw auto-generated data about the project (symbol graph + project docs).
Your job is to SYNTHESIZE it into a compact, high-signal init.md (100–150 lines, <2500 tokens).

First: detect the primary language(s) and project type from the raw data. Then apply
language-appropriate patterns. An AI reading this file should immediately understand
the architecture, key design patterns, where to find important code, and how pieces fit
together. Do NOT re-list the raw data — INTERPRET and SYNTHESIZE it.

{rules}

RAW DATA:

```markdown
{base_content}
```

Return ONLY the final init.md markdown. No preamble, no explanation."""

# The evolvable rules section — this is what GEPA v2 optimizes.
# Evolved Gen 2 version from GEPA v2 (1500 chars cap, g_e_p_a_service strategy)
# Key enhancement: "analyze the specific project files" emphasis before writing
SYNTHESIS_RULES = """RULES:
- Before writing each section, analyze the specific project files and metadata carefully; tailor content precisely. Do NOT use generic templates or phrases.
- Verify tool argument types and output structure before each tool call. Treat tool outputs as structured data and confirm fields exist before using them.
- Use targeted graph/code_search queries with specific filters or limits; avoid broad scans when a narrower query will answer the question.
- For large files or directories, use pagination or incremental reads with offsets, limits, or search selectors instead of rereading the full artifact.
- Always verify file or directory existence with ls() before read() or shell operations.
- Write project-specific content only. No generic advice.
- Sections in order: Project Overview, System Flow, Package Layout, Key Entry Points,
  Architecture Patterns, Development Commands, Dependencies, Configuration, Codebase Scale.

- Project Overview: 3–4 sentences: what the project does, primary language/framework,
  key capabilities, intended users. Include version/release info if present.

- System Flow: one arrow-chain line showing end-to-end data/control flow. Examples:
  - Web service:    "Request → Router → Handler → Service → Repository → DB"
  - CLI tool:       "User → CLI Parser → Command → Core Logic → Output"
  - Library/SDK:    "User Code → Public API → Core Engine → Backends/Adapters"
  - Agent/AI:       "User → Orchestrator → Provider → Tools → Storage"
  - Compiler/tool:  "Source → Lexer → Parser → AST → Codegen → Output"

- Package Layout: table (Path | Type | Description). Language conventions:
  - Go:        cmd/ (binaries), pkg/ (libraries), internal/ (private), api/ (protos)
  - Rust:      src/ (lib root), each workspace crate, bin/ for binaries
  - Node/TS:   src/, packages/ (monorepo), lib/; omit dist/ and node_modules/
  - Python:    top-level package dir, scripts/, config files
  - Java/Kotlin: src/main/java, modules/ (multi-module), resources/
  Only include directories that actually exist. Omit build artifacts and test subfolders.

- Key Entry Points: table (Component | Type | Path:line | Description).
  Pick 10–15 most architecturally important: entry points, core abstractions, base
  classes/interfaces/traits, registries, facades, public API surfaces.
  Use real line numbers from raw data. Exclude alphabetically sorted utility classes.

- Architecture Patterns: MOST IMPORTANT SECTION. 5–8 bullet points. Cite concrete names.
  Detect and report patterns appropriate to the language:

  Python patterns to look for:
  * Inheritance backbone — most-subclassed base classes with subclass count
  * Hub classes — highest-connectivity types (many imports/references)
  * Facade/Orchestrator — large coordinator classes delegating to sub-components
  * Plugin/Extension system — how plugins register (entry_points, subclassing, decorators)
  * Async-first design — asyncio coroutine boundaries, sync/async adapter layers
  * Feature flag / settings gating — config-driven behavior variants

  Go patterns to look for:
  * Interface segregation — key interfaces and their main implementors
  * Constructor injection — how dependencies are wired (wire/dig/fx or manual)
  * Package cohesion — cmd/ vs internal/ vs pkg/ boundary discipline
  * Error handling — sentinel errors, wrapped errors (%w), custom error types
  * Concurrency — goroutine/channel patterns, sync.Mutex, errgroup usage

  TypeScript/Node patterns to look for:
  * Module system — ESM vs CJS, barrel exports (index.ts), path aliases
  * DI framework — NestJS decorators, inversify, tsyringe, or manual factories
  * Type hierarchy — discriminated unions, branded types, generic constraints
  * Async patterns — Promise, async/await, RxJS observables, event emitters
  * Monorepo structure — workspaces (nx/turborepo/pnpm), shared packages

  Rust patterns to look for:
  * Trait system — key traits and their main implementors (analogous to interfaces)
  * Ownership strategy — Arc<Mutex<>>, RwLock, channels for shared state
  * Error handling — thiserror/anyhow, custom Error enums, Result propagation
  * Crate architecture — workspace layout, public API surface via lib.rs pub use
  * Async runtime — tokio/async-std task spawning, Pin/Future patterns

  Java/Kotlin patterns to look for:
  * DI framework — Spring beans, Guice modules, Dagger components
  * Layered architecture — Controller → Service → Repository boundary
  * Domain model — key entities, value objects, aggregates (DDD markers)
  * Exception hierarchy — checked vs unchecked, custom exception classes

  Universal patterns (apply to any language):
  * Coupling hotspots — files/modules with highest fan-in (note as "many callers")
  * Extensibility mechanism — the primary way the system is extended by users/plugins
  * Configuration pattern — how config is loaded and propagated through the system

- Development Commands: essential commands in a code block (build, test, run, lint).
  Language defaults if not found in raw data:
  - Python:  pip install -e ".[dev]", pytest, ruff/black/mypy
  - Go:      go build ./..., go test ./..., go vet ./..., golangci-lint run
  - Node/TS: npm install, npm run build, npm test, npm run lint
  - Rust:    cargo build, cargo test, cargo clippy, cargo fmt
  - Java:    ./mvnw package, ./mvnw test  OR  ./gradlew build, ./gradlew test
  Include environment setup (venv activate, nvm use, etc.) if present in raw data.

- Dependencies: one line — package manager, total count, top 6–8 package names.
  Source from whichever manifest is present: pyproject.toml, package.json, go.mod,
  Cargo.toml, pom.xml, build.gradle.

- Configuration: 2–3 lines on how config works: sources, override order, key classes/files.

- Codebase Scale: one line — LOC, file count, symbol/node count and graph edges if
  available, test coverage if known.

- OMIT: Analyzer Coverage, Performance Hints, Embeddings & Chunking, raw PageRank
  numbers, graph node IDs, Learned from Conversations, Common Topics, FAQ, indexer
  internals, sections about the analysis tooling itself.
- End with: "Run `/init --update` to refresh after code changes.\""""


def _build_synthesis_prompt(
    base_content: str,
    rules: str = SYNTHESIS_RULES,
    graph_context: Optional[dict] = None,
) -> str:
    """Assemble the full synthesis prompt from frame + rules + data + graph context."""
    # Format graph context if available
    graph_section = ""
    if graph_context:
        graph_section = _format_graph_context_for_prompt(graph_context)

    # Inject graph context between rules and base content
    data_section = base_content
    if graph_section:
        data_section = f"{graph_section}\n\n---\n\n{base_content}"

    return _SYNTHESIS_FRAME_BEFORE.format(rules=rules, base_content=data_section)


def _trim_to_first_markdown_heading(content: str) -> Optional[str]:
    """Drop any preamble text before the first markdown heading.

    The init.md synthesis prompt asks the model to "return ONLY the
    init.md markdown content (start with ``# init.md``) … no preamble".
    Some models still leak a single intro sentence ("Writing the init.md
    now."). This helper returns the substring starting at the first
    ``# `` / ``## `` line. Returns ``None`` if no heading is found, in
    which case the caller should keep the original content unchanged.
    """
    if not content:
        return None
    lines = content.splitlines(keepends=True)
    for idx, line in enumerate(lines):
        if line.lstrip().startswith("#"):
            stripped = line.lstrip()
            # require an actual heading: ``# `` or ``## `` etc.
            if len(stripped) > 1 and stripped[1:].lstrip().startswith("#") is False:
                # ``# x`` — first char after the # must be space or end
                after_hash = stripped[1:]
                if after_hash.startswith(" ") or after_hash.startswith("\t"):
                    return "".join(lines[idx:])
            elif stripped.startswith("##"):
                return "".join(lines[idx:])
    return None


# Signal phrases that indicate the model appended a self-reflective
# postamble about tool availability instead of just returning the doc.
# Matched case-insensitively against the trailing prose block only —
# never against content inside the document body.
_META_COMMENTARY_SIGNALS = (
    "no write",
    "no edit",
    "no file-write",
    "write tool",
    "edit tool",
    "write/edit",
    "file-write",
    "registered toolset",
    "registered tool set",
    "tool access",
    "tool with file-write",
    "tool with write",
    "cannot be persisted",
    "cannot be saved",
    "cannot write to",
    "lack the ability",
    "lack write",
    "i don't have",
    "i do not have",
    "i was unable",
    "needs to be saved",
    "save this output",
    "save the file",
    "save this file",
    "ready to be written",
)


# Signal phrases that indicate the model emitted a chat-style recap
# INSTEAD of the actual init.md document (typically because the agentic
# loop ran extra iterations after the deliverable was already produced
# in an earlier turn).
_LEADING_META_COMMENTARY_SIGNALS = (
    "already generated",
    "already wrote",
    "already written",
    "already created",
    "previous turn",
    "earlier turn",
    "task is complete",
    "task complete",
    "i have generated",
    "i've generated",
    "i have created",
    "i've created",
    "i have written",
    "i've written",
    "verified through tool calls",
    "all claims verified",
    "claims verified",
    "here is the init",
    "here's the init",
    "below is the init",
    "writing the init.md",
    "let me write",
    "i'll write",
    "i will write",
    "i'll generate",
    "i will generate",
    "now i'll",
    "now i will",
    "generated `.victor/init.md`",
    "generated .victor/init.md",
)


def _is_numbered_list_item(stripped: str) -> bool:
    """Return True if ``stripped`` starts with ``N. `` / ``NN. `` / ``NNN. ``."""
    dot_idx = stripped.find(". ")
    if dot_idx < 1 or dot_idx > 3:
        return False
    return stripped[:dot_idx].isdigit()


def _strip_leading_meta_commentary(content: str) -> str:
    """Trim a leading self-reflective preamble before the document body.

    Observed leak (2026-05-28): the agentic synthesis loop ran extra
    iterations after the model had already produced the real init.md in
    an earlier turn. The final assistant message — which is what gets
    persisted — was a chat-style recap like::

        The init.md document was already generated and output in my
        previous turn. The task is complete.

        Generated `.victor/init.md` with all claims verified through
        tool calls:

        1. **Cargo.toml** — verified version `0.2.0`, ...
        2. **Makefile** — verified all build/test/benchmark commands
        ...

    Because the recap has no markdown headings,
    ``_trim_to_first_markdown_heading`` can't find a trim point and the
    recap gets persisted as the saved file.

    Strategy: walk forward over leading paragraphs (separated by blank
    lines). Drop a paragraph if it is plain prose AND contains any
    leading-meta signal phrase. After dropping >=1 prose paragraph,
    also drop a directly-following numbered list where every item
    matches a "verification" pattern. Stops at the first markdown
    heading or non-meta content paragraph.

    Conservative on purpose: only trims LEADING content, never anything
    after real document structure has started.
    """
    if not content:
        return content
    lines = content.splitlines()
    trimmed_any = False
    while True:
        # Drop leading blank lines.
        while lines and not lines[0].strip():
            lines.pop(0)
        if not lines:
            break
        # If we hit a markdown heading, the document body has started.
        first_stripped = lines[0].lstrip()
        if first_stripped.startswith("#"):
            break
        # Find the end of the first paragraph (next blank line, or EOF).
        para_end = len(lines)
        for idx in range(len(lines)):
            if not lines[idx].strip():
                para_end = idx
                break
        paragraph_lines = lines[:para_end]
        non_blank_lines = [ln for ln in paragraph_lines if ln.strip()]
        if not non_blank_lines:
            break
        # Case 1: a numbered list (e.g. "1. **Cargo.toml** — verified ...").
        # Only treated as recap when we've already stripped a prose
        # paragraph above it AND every item looks like a verification
        # entry. Otherwise it's real content (numbered list of facts).
        if all(_is_numbered_list_item(ln.lstrip()) for ln in non_blank_lines):
            if trimmed_any and all(
                ("verified" in ln.lower() or " — " in ln or " - " in ln) for ln in non_blank_lines
            ):
                del lines[:para_end]
                trimmed_any = True
                continue
            break
        # Case 2: plain prose paragraph. Reject if any line has markdown
        # structure (heading, list, table, blockquote, code fence).
        if any(
            ln.lstrip().startswith(("#", "- ", "* ", "+ ", "> ", "| ", "```"))
            for ln in paragraph_lines
        ):
            break
        lowered = "\n".join(paragraph_lines).lower()
        if not any(sig in lowered for sig in _LEADING_META_COMMENTARY_SIGNALS):
            break
        del lines[:para_end]
        trimmed_any = True
    result = "\n".join(lines)
    if trimmed_any and content.endswith("\n") and not result.endswith("\n"):
        result += "\n"
    return result


def _strip_trailing_meta_commentary(content: str) -> str:
    """Trim a trailing self-reflective paragraph about tool availability.

    Observed leak (2026-05-27): the synthesis agent emitted the full
    init.md, then appended a paragraph like "No write/edit tool
    available in the registered toolset — the file content above is
    ready to be written..." which got persisted verbatim into the saved
    file because nothing in the cleanup chain looks at the tail.

    Strategy: walk from the end of the document backward over blank
    lines and a single trailing paragraph. If that paragraph is plain
    prose (no markdown structure markers — no leading ``#``, ``-``,
    ``*``, ``|``, ``>``, ``    `` code indent, no fence) and contains
    any of the meta-commentary signal phrases, trim it plus its
    leading separator blank lines. Repeat for stacked paragraphs.

    Conservative on purpose: only trims trailing PROSE, never lines
    that look like document content. A user-written ``## Tool
    Availability`` section with prose underneath stays intact because
    the trim stops at the preceding heading.
    """
    if not content:
        return content
    lines = content.splitlines()
    trimmed_any = False
    while True:
        # Drop trailing blanks
        while lines and not lines[-1].strip():
            lines.pop()
        if not lines:
            break
        # Find the start of the trailing paragraph (back to a blank
        # line or to a markdown-structure line — whichever comes first).
        para_start = len(lines)
        for idx in range(len(lines) - 1, -1, -1):
            line = lines[idx]
            stripped = line.lstrip()
            if not stripped:
                para_start = idx + 1
                break
            # Hit a markdown structure line — paragraph starts after it.
            if (
                stripped.startswith("#")
                or stripped.startswith("- ")
                or stripped.startswith("* ")
                or stripped.startswith("+ ")
                or stripped.startswith("> ")
                or stripped.startswith("| ")
                or stripped.startswith("|-")
                or stripped.startswith("```")
                or (len(line) >= 4 and line[:4] == "    ")
                or (stripped[:2].isdigit() and ". " in stripped[:5])
            ):
                para_start = idx + 1
                break
            para_start = idx
        if para_start >= len(lines):
            # No trailing prose paragraph found.
            break
        paragraph = "\n".join(lines[para_start:])
        # Don't trim if the paragraph itself contains markdown structure.
        if any(
            ln.lstrip().startswith(("#", "- ", "* ", "+ ", "> ", "| ", "```"))
            for ln in lines[para_start:]
        ):
            break
        lowered = paragraph.lower()
        if not any(sig in lowered for sig in _META_COMMENTARY_SIGNALS):
            break
        del lines[para_start:]
        trimmed_any = True
    result = "\n".join(lines)
    if trimmed_any and content.endswith("\n"):
        result += "\n"
    return result


def _format_graph_context_for_prompt(graph_context: dict) -> str:
    """Format graph context for LLM synthesis prompt.

    Uses generic pattern terminology that applies to any project:
    - "Decorator pattern" not "@tool decorator"
    - "Registry pattern" not "tool registry"
    - "Protocol/Interface" not "Victor protocol"
    """
    if not graph_context:
        return ""

    stats = graph_context.get("stats", {})
    patterns = graph_context.get("patterns", {})
    complexity = graph_context.get("complexity", {})
    scale_facts = graph_context.get("scale_facts") or {}

    sections = []

    # Measured project scale must come FIRST so the LLM anchors on real
    # numbers before reading the more abstract graph/CCG sections. Without
    # this block the model previously invented file/LOC/crate counts that
    # didn't match the repo (see init.py:_gather_project_scale_facts for
    # the source of these numbers).
    if scale_facts and scale_facts.get("total_files"):
        lang_lines = []
        by_lang = scale_facts.get("files_by_language", {})
        loc = scale_facts.get("loc_by_language", {})
        for lang, count in sorted(by_lang.items(), key=lambda kv: -kv[1]):
            lang_lines.append(f"  - {lang}: {count:,} files, {loc.get(lang, 0):,} LOC")
        scale_section_lines = [
            "## Measured Project Scale (use these numbers verbatim; do not invent)",
            "",
            f"- Total source files: {scale_facts['total_files']:,}",
            f"- Total LOC (non-blank, indexable languages only): {scale_facts['total_loc']:,}",
            "- Files by language:",
            *lang_lines,
        ]
        if scale_facts.get("cargo_crate_count"):
            scale_section_lines.append(
                f"- Cargo workspace crates: {scale_facts['cargo_crate_count']}"
            )
        if scale_facts.get("python_packages"):
            scale_section_lines.append(
                f"- Python packages (__init__.py count): {scale_facts['python_packages']}"
            )
        if scale_facts.get("js_packages"):
            scale_section_lines.append(
                f"- JS/TS sub-packages (package.json): {scale_facts['js_packages']}"
            )
        scale_section_lines.append("")
        scale_section_lines.append(
            "When the generated document mentions codebase size, crate count, "
            "file count, or language breakdown, use these exact numbers. Do "
            "not estimate, round, or substitute. If a number you would "
            "otherwise write doesn't appear above, omit the claim."
        )
        sections.append("\n".join(scale_section_lines))

    # CCG section (universal - applies to any codebase)
    if graph_context.get("has_ccg", False):
        ccg_ratio = complexity.get("ccg_ratio", 0) * 100
        sections.append(f"""
## Code Structure Analysis

This project has been analyzed with Code Context Graph (CCG) providing:
- **Control Flow Graph (CFG)**: {stats.get('cfg_edges', 0)} edges showing execution paths
- **Control Dependence Graph (CDG)**: {stats.get('cdg_edges', 0)} edges showing decision dependencies
- **Data Dependence Graph (DDG)**: {stats.get('ddg_edges', 0)} edges showing data flow

CCG Coverage: {ccg_ratio:.1f}% of graph edges have statement-level granularity.

When analyzing, leverage these to:
- Identify control flow complexity hotspots (deep nesting, complex branching)
- Trace data dependencies and side effects
- Understand decision points and their impact
""")

    # Generic design patterns (not Victor-specific)
    pattern_sections = []

    if patterns.get("decorator", 0) > 0:
        pattern_sections.append(f"""
- **Decorator Pattern**: {patterns['decorator']} decorated symbols detected.
  Analyze how decorators modify behavior and cross-cutting concerns.
""")

    if patterns.get("protocol", 0) > 0:
        pattern_sections.append(f"""
- **Protocol/Interface Pattern**: {patterns['protocol']} implementation relationships.
  Map abstraction hierarchies and contract implementations.
""")

    if patterns.get("registry", 0) > 0:
        pattern_sections.append(f"""
- **Registry/Plugin Pattern**: {patterns['registry']} registration relationships.
  Identify extensibility points and plugin architecture.
""")

    if patterns.get("inheritance", 0) > 0:
        pattern_sections.append(f"""
- **Inheritance Pattern**: {patterns['inheritance']} inheritance relationships.
  Analyze class hierarchies and method overrides.
""")

    if pattern_sections:
        sections.append(f"""
## Design Pattern Analysis

The following design patterns have been detected in the codebase:
{''.join(pattern_sections)}

Use these patterns to:
- Understand the architectural approach
- Identify extensibility mechanisms
- Map abstraction layers and contracts
""")

    return "\n".join(sections)


# Public constant for backward compatibility (assembled with static rules)
SYNTHESIS_PROMPT = _build_synthesis_prompt("{base_content}")
# Note: SYNTHESIS_PROMPT.format(base_content=...) still works because
# _build_synthesis_prompt leaves {base_content} as a literal when called
# with "{base_content}" as the data argument... except it would double-format.
# So we define it directly for backward compat:
SYNTHESIS_PROMPT = _SYNTHESIS_FRAME_BEFORE.format(
    rules=SYNTHESIS_RULES, base_content="{base_content}"
)


def _parse_makefile_targets(path: Any) -> str:
    """Extract phony/documented make targets (up to 1000 chars)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    targets = [
        line
        for line in lines
        if line
        and not line.startswith("\t")
        and ":" in line
        and not line.startswith("#")
        and len(line) < 100
    ][:25]
    return "\n".join(targets)[:1000]


def _read_task_file(path: Any) -> str:
    """Read justfile/Taskfile, trimmed to 1000 chars."""
    return path.read_text(encoding="utf-8")[:1000]


TOOLS_FALLBACK_PROMPT = """Analyze this codebase and generate a comprehensive init.md file.

Use the available tools (overview, ls, read) to understand the project structure,
key components, and architecture. Then write a concise init.md with sections:
Project Overview, Package Layout, Key Entry Points, Development Commands,
Dependencies, Configuration, Architecture Notes.

Target: 80-120 lines, project-specific content only, no generic advice.
Return ONLY the init.md markdown content."""


_TOOLS_DEEP_FRAME = """You are generating an ``init.md`` "user manual" for an
AI coding assistant. You have tool access — use it to verify claims before
writing them. One-shot summarization without tool use is forbidden for this
task.

Tools you can call (live list from the registered tool set — use exactly
these names via the function-calling format, not as markdown code blocks):

{tool_listing}

When to use what:
- BEFORE writing the Project Overview, FIRST list the project root to see
  what top-level docs actually exist (README may be README.md, README.adoc,
  README.rst, or absent; the project may use CLAUDE.md / AGENTS.md /
  GEMINI.md instead). Read whichever of those are present plus the
  top-level manifest (Cargo.toml / pyproject.toml / package.json /
  go.mod) so the description is sourced, not guessed. If a specific
  filename is missing, fall back to whatever directory-listing tool the
  registry exposes to discover alternatives — do not abort.
- BEFORE listing Key Components, if a graph/symbol-graph tool is in the
  list above, query it for highest-fan-in modules and inheritance hubs
  so you name *actually* central components, not superficially-named
  ones. If no graph tool is registered, fall back to reading the
  top-level source-tree entries surfaced by directory listing.
- BEFORE describing Architecture, read 2-3 representative source files —
  prefer the entry-point modules surfaced by the listing/graph step.
- BEFORE quoting Development Commands, read Makefile / package.json
  scripts / Cargo.toml workspace config — do not invent commands.

Important: each assistant turn must either (a) issue at least one tool
call via the function-calling format, or (b) emit the final init.md
content starting with ``# init.md``. Do NOT produce narrative text like
"Now let me read X" or write tool invocations inside markdown code
blocks — narration without an actual function call is treated as a
finished answer by the runtime and the document never gets written.

How persistence works: your final assistant message IS the saved
``.victor/init.md`` file. The runtime captures that message verbatim
and writes it to disk for you. Write/edit/shell tools are
intentionally disabled for this task. Do NOT attempt to call them, and
do NOT add a postamble explaining what tools were or weren't available
— any prose appended after the last markdown section gets written into
the file as visible junk.

Rules:
1. The "Measured Project Scale" section below is ground truth. Quote
   those numbers verbatim. If your draft references a count not listed
   there, replace it with a tool-verified number or remove the claim.
2. No generic AI advice ("use clean code"). Project-specific content
   only.
3. Target 100-180 lines. Use Markdown tables for structured data.
4. Return ONLY the init.md markdown content (start with ``# init.md``),
   no preamble, no postamble, no commentary. Do NOT wrap the document
   in a ``` ```markdown ... ``` ``` fence. Do NOT add a closing note
   about tool availability, file-saving, or what you would have done
   differently — the last line of your response must be the last line
   of the document.

---

{ground_truth}

---

Generate init.md now. Begin by calling tools (using the function-calling
format) to verify your understanding, then write the document."""


def _format_registered_tools_for_prompt(agent: Any) -> str:
    """Format the agent's actually-registered tools for inclusion in the prompt.

    Builds the list from the canonical ``ToolRegistry`` attached to the
    agent (the same registry the orchestrator passes to the LLM as
    function-call schemas). The prompt is therefore guaranteed to only
    advertise tools the model can actually invoke — no more "prompt
    promises ``graph`` but the agent has no ``graph`` bound, so the model
    writes ```` ```graph ...``` ```` in markdown" drift.

    Falls back to a minimal generic list if the registry can't be reached,
    so the prompt still has *something* sensible if Agent internals change.
    """
    try:
        registry = None
        orchestrator = getattr(agent, "_orchestrator", None) or agent
        registry = getattr(orchestrator, "tools", None)
        if registry is None or not hasattr(registry, "list_tools"):
            raise RuntimeError("no tool registry on agent")

        tools = registry.list_tools(only_enabled=True)
        if not tools:
            raise RuntimeError("tool registry returned no enabled tools")

        # One short line per tool: `name` — first-sentence of description.
        # Keep it tight so the prompt stays under the model's context window.
        lines = []
        seen: set[str] = set()
        for tool in sorted(tools, key=lambda t: getattr(t, "name", "")):
            name = getattr(tool, "name", None)
            if not name or name in seen:
                continue
            seen.add(name)
            desc = (getattr(tool, "description", "") or "").strip()
            # Trim description to first sentence-ish to keep prompt compact
            for sep in (". ", "\n", "  "):
                if sep in desc:
                    desc = desc.split(sep, 1)[0].rstrip()
                    break
            desc = desc[:160]
            lines.append(f"- ``{name}`` — {desc}" if desc else f"- ``{name}``")
        if not lines:
            raise RuntimeError("registry yielded no usable tool names")
        return "\n".join(lines)
    except Exception as exc:
        logger.debug("[init] live tool listing unavailable, using fallback: %s", exc)
        return (
            "- ``read`` — read a file by path\n"
            "- ``ls`` — list a directory\n"
            "- ``overview`` — high-level project layout"
        )


def _build_tools_deep_prompt(
    graph_context: Optional[dict] = None,
    *,
    tool_listing: Optional[str] = None,
) -> str:
    """Compose the tool-driven synthesis prompt.

    Includes the same "Measured Project Scale" / graph-context block the
    one-shot path uses as an anchor, but frames the rest as an instruction
    set for tool-driven exploration rather than a write-it-now command.
    Without the scale facts upfront the agent often opens with redundant
    ``ls`` / ``overview`` calls just to derive numbers we already have.

    Args:
        graph_context: Optional pre-computed graph stats / scale facts.
        tool_listing: Optional pre-rendered tool list (one ``- `name` — desc``
            line per tool). When provided, this lets the prompt advertise
            only the tools the agent actually has bound. Pass the output of
            ``_format_registered_tools_for_prompt(agent)`` from the caller.
    """
    ground_truth = _format_graph_context_for_prompt(graph_context or {}).strip()
    if not ground_truth:
        ground_truth = (
            "(No pre-computed project scale or graph statistics available — "
            "derive counts yourself with the tools before quoting them.)"
        )
    if not tool_listing:
        # Caller didn't pass a live tool list — keep the prompt valid but
        # generic. Real call sites should pass a tool_listing from the
        # agent's registry so the prompt only advertises bound tools.
        tool_listing = (
            "- ``read`` — read a file by path\n"
            "- ``ls`` — list a directory\n"
            "- ``overview`` — high-level project layout"
        )
    return _TOOLS_DEEP_FRAME.format(
        ground_truth=ground_truth,
        tool_listing=tool_listing,
    )


class InitSynthesizer:
    """Generates init.md via Agent framework for full observability.

    Provides two synthesis modes:
    - synthesize(): Takes pre-built base_content (from CodebaseAnalyzer + graph)
      and synthesizes via LLM. Used when victor-coding is available.
    - synthesize_with_tools(): Uses Agent tools (overview, ls, read) to gather
      context and synthesize in one pass. Fallback when victor-coding unavailable.

    Both modes support agent reuse (slash command) or fresh Agent creation (CLI).
    """

    async def synthesize(
        self,
        base_content: str,
        *,
        agent: Optional["AgentOrchestrator"] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        graph_context: Optional[dict] = None,
    ) -> str:
        """Synthesize init.md from pre-built base content via Agent framework.

        Args:
            base_content: Raw analysis data (from CodebaseAnalyzer + graph insights).
            agent: Existing orchestrator to reuse (from slash command context).
            provider: Provider name for fresh Agent (CLI path). Ignored if agent given.
            model: Model name for fresh Agent. Ignored if agent given.
            graph_context: Optional graph statistics (CCG, patterns) for synthesis.

        Returns:
            Synthesized init.md markdown content.
        """
        # Enrich base_content with CLAUDE.md / existing project docs before synthesis.
        # This gives the LLM the architectural ground truth (patterns, service layer,
        # feature flags, dev commands) that static graph analysis alone can't surface.
        base_content = self._enrich_with_project_docs(base_content)

        # Pre-synthesis discovery: use graph tool for structural and semantic insights
        # This fills the "0 patterns" gap by finding loosely coupled relationships
        # and hotspots (coupling, centrality) before the LLM sees the data.
        discovery_data = await self._pre_synthesis_discovery(agent)
        if discovery_data:
            base_content = f"## Architectural Discovery (Heuristic + Semantic)\n\n{discovery_data}\n\n---\n\n{base_content}"

        resolved_provider, resolved_model = self._resolve_prompt_optimization_identity(
            agent=agent,
            provider=provider,
            model=model,
        )

        # Check for GEPA-evolved RULES section (not the full prompt)
        evolved_rules = self._get_evolved_rules(resolved_provider, resolved_model)
        if evolved_rules:
            prompt = _build_synthesis_prompt(
                base_content, rules=evolved_rules, graph_context=graph_context
            )
            logger.info(
                "[init] Using GEPA-evolved rules (%d chars) in fixed frame",
                len(evolved_rules),
            )
        else:
            prompt = _build_synthesis_prompt(base_content, graph_context=graph_context)

        if agent:
            return await self._run_with_orchestrator(agent, prompt)
        else:
            return await self._run_with_fresh_agent(prompt, resolved_provider, resolved_model)

    async def synthesize_with_tools(
        self,
        *,
        agent: Optional["AgentOrchestrator"] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        graph_context: Optional[dict] = None,
    ) -> str:
        """Tool-driven init.md synthesis.

        Two call sites today:
        - Default fallback when victor-coding is not installed (no
          CodebaseAnalyzer or graph data exists yet — pass
          ``graph_context=None`` to fall back to the generic prompt).
        - Opt-in agentic init (``victor init --agentic``) when the user
          wants the LLM to verify facts via tools before writing.
          ``graph_context`` should carry the scale_facts + graph stats
          already gathered by init.py so the prompt anchors numeric
          claims on measured ground truth instead of asking the agent
          to derive them all from scratch.

        Unlike ``synthesize``, this DOES use the agentic loop — the agent
        needs to call tools (overview, ls, read, graph, search_files) to
        explore the codebase. Slower (1-3 min) and more expensive than
        the one-shot path but produces a richer document for under-
        documented repos.
        """
        # Pick the right prompt: agentic call with anchors gets the deep
        # frame; bare fallback (no scale/graph data) keeps the minimal
        # legacy prompt so we don't crash older callers that never passed
        # graph_context.
        if graph_context is not None:
            # Build the prompt's tool listing from the agent's live
            # ToolRegistry so we only advertise tools the model can
            # actually invoke via function calls. Reusing the registry
            # avoids the prompt-vs-reality drift that caused the model to
            # emit ```` ```graph ...``` ```` as markdown when the agent
            # didn't have the ``graph`` tool bound.
            tool_listing = _format_registered_tools_for_prompt(agent) if agent else None
            prompt = _build_tools_deep_prompt(graph_context, tool_listing=tool_listing)
        else:
            prompt = TOOLS_FALLBACK_PROMPT

        if agent:
            # The agentic loop is required here — the agent needs to call
            # tools mid-conversation. Use agent.chat() / agent.run() directly,
            # NOT _run_with_orchestrator (which bypasses the loop for
            # performance on the one-shot path).
            #
            # Two agent shapes flow through here:
            #   - AgentOrchestrator.chat()  → async, CompletionResponse (.content)
            #   - victor.framework.agent.Agent.chat() → SYNC, returns ChatSession
            #     (a session factory; awaiting it raises
            #      "object ChatSession can't be used in 'await' expression").
            #     The async one-shot entry point on Agent is .run() → TaskResult,
            #     which also drives the agentic loop and exposes .content.
            import inspect

            # The deterministic read-only fast-path in TurnExecutor bypasses
            # the LLM when a prompt mentions manifest-review words, which
            # silently breaks `--agentic` init.md synthesis (the model
            # never gets to write the document). Opt out explicitly so the
            # full agentic loop drives this call.
            agentic_overrides = {"disable_deterministic_fast_path": True}

            try:
                chat_fn = getattr(agent, "chat", None)
                if inspect.iscoroutinefunction(chat_fn):
                    sig = inspect.signature(chat_fn)
                    if "runtime_context_overrides" in sig.parameters:
                        response = await agent.chat(
                            prompt, runtime_context_overrides=agentic_overrides
                        )
                    else:
                        response = await agent.chat(prompt)
                elif inspect.iscoroutinefunction(getattr(agent, "run", None)):
                    sig = inspect.signature(agent.run)
                    if "runtime_context_overrides" in sig.parameters:
                        response = await agent.run(
                            prompt, runtime_context_overrides=agentic_overrides
                        )
                    else:
                        response = await agent.run(prompt)
                else:
                    logger.warning(
                        "Init synthesis with tools: agent %s exposes neither "
                        "async chat() nor async run(); skipping",
                        type(agent).__name__,
                    )
                    return ""
                content = response.content if response else ""
                # Strip Victor's runtime completion markers (e.g.
                # ``VICTOR_FILE_DONE::``). The coding-vertical system prompt
                # encourages the model to emit these as a signal "I've
                # finished writing files", but for init.md synthesis where
                # we save the response as the deliverable they leak into
                # the top of the file as visible prose. The framework
                # already exposes the canonical strip helper — reuse it
                # rather than rolling a private regex.
                try:
                    from victor.core.completion_markers import (
                        strip_active_completion_markers,
                    )

                    content = strip_active_completion_markers(content)
                except Exception:
                    # Marker stripping is best-effort; never fail the run
                    # if the helper is unavailable.
                    pass
                # Trim any preamble before the first markdown heading.
                # The prompt explicitly asks for content "starting with
                # ``# init.md`` … no preamble", but models occasionally
                # leak a leading sentence like "Writing the init.md now."
                # before the actual document. Drop everything before the
                # first ``# `` / ``## `` line so the saved file starts at
                # the real document boundary.
                _trimmed = _trim_to_first_markdown_heading(content)
                if _trimmed is not None:
                    content = _trimmed
                # Strip a chat-style leading recap that has no markdown
                # heading at all (so the previous step couldn't help).
                # Observed when the agentic loop runs extra iterations
                # after the deliverable was already produced — the model
                # responds "the init.md was already generated... " in
                # the final turn and that recap gets persisted.
                _pre_trim = _strip_leading_meta_commentary(content)
                if _pre_trim != content:
                    logger.info(
                        "[init] trimmed leading meta-commentary from "
                        "agentic synthesis output (%d -> %d chars)",
                        len(content),
                        len(_pre_trim),
                    )
                    content = _pre_trim
                # Strip any trailing self-reflective paragraph the model
                # may have appended about tool availability / inability
                # to save the file. Observed leak: agent emitted full
                # init.md then a "No write/edit tool available …" coda
                # that got persisted verbatim into the saved doc.
                _post_trim = _strip_trailing_meta_commentary(content)
                if _post_trim != content:
                    logger.info(
                        "[init] trimmed trailing meta-commentary from "
                        "agentic synthesis output (%d -> %d chars)",
                        len(content),
                        len(_post_trim),
                    )
                    content = _post_trim
                cleaned = self._clean(content)
                # Sanity check: if the cleaned output has no markdown
                # headings, the model didn't produce a real document
                # (typically a recap-only final turn). Returning the
                # recap would persist chat junk into ``.victor/init.md``
                # (and worse, init.py then appends synthetic sections
                # like "## Architecture Patterns" AFTER the junk, making
                # it look almost-real). Treat as failed synthesis so the
                # caller falls back to base content.
                if cleaned and not self._has_document_structure(cleaned):
                    logger.warning(
                        "[init] agentic synthesis produced no markdown "
                        "structure (chars=%d, lines=%d); treating as "
                        "failed synthesis. First 200 chars: %r",
                        len(cleaned),
                        cleaned.count("\n") + 1,
                        cleaned[:200],
                    )
                    return ""
                return cleaned
            except Exception as e:
                logger.warning("Init synthesis with tools failed: %s", e)
                return ""
        else:
            return await self._run_agent_with_tools(prompt, provider, model, vertical="coding")

    async def _run_with_orchestrator(self, agent: "AgentOrchestrator", prompt: str) -> str:
        """Run synthesis using an existing orchestrator or framework Agent.

        Handles two agent types:
        - AgentOrchestrator: async chat() → CompletionResponse (slash command path)
        - victor.framework.agent.Agent: async run() → TaskResult (CLI path via _create_init_agent)
        """
        import inspect

        try:
            # Always prefer a direct provider call for init synthesis — bypass AgenticLoop.
            # AgentOrchestrator.chat() routes through AgenticLoop → edge model (Ollama)
            # calls for task classification + tool selection even when we just need one LLM
            # call: prompt → markdown. Use the already-initialized provider directly instead.

            # 1. AgentOrchestrator (from _create_init_agent profile path or slash command)
            provider_instance = getattr(agent, "provider", None)
            model = getattr(agent, "model", None)
            temperature = getattr(agent, "temperature", None)
            max_tokens = getattr(agent, "max_tokens", None)

            # 2. Framework Agent (wraps orchestrator via _orchestrator)
            if provider_instance is None:
                inner = getattr(agent, "_orchestrator", None)
                if inner is not None:
                    provider_instance = getattr(inner, "provider", None)
                    model = getattr(inner, "model", None)
                    if temperature is None:
                        temperature = getattr(inner, "temperature", None)
                    if max_tokens is None:
                        max_tokens = getattr(inner, "max_tokens", None)

            if provider_instance is not None:
                logger.debug(
                    "[init] Using initialized provider %s (bypassing AgenticLoop)",
                    type(provider_instance).__name__,
                )
                return await self._call_initialized_provider(
                    prompt,
                    provider_instance,
                    model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

            # 3. Last resort: fall back to agent.chat() for slash command contexts where
            #    the orchestrator's provider attribute may not be directly accessible.
            if inspect.iscoroutinefunction(getattr(agent, "chat", None)):
                logger.debug("[init] No direct provider found, falling back to agent.chat()")
                response = await agent.chat(prompt)
                content = response.content if response else ""
                return self._clean(content)

            # 4. Bare framework Agent with no accessible orchestrator
            provider_name = getattr(agent, "_provider", None)
            model = getattr(agent, "_model", None)
            logger.debug("[init] Falling back to fresh provider: %s", provider_name)
            return await self._run_with_fresh_agent(prompt, provider_name, model)
        except Exception as e:
            logger.warning("Init synthesis via orchestrator failed: %s", e)
            raise

    async def _preflight_provider(
        self,
        provider_name: Optional[str],
        provider_instance: Any,
        model: Optional[str],
    ) -> Optional[str]:
        """Fail fast for providers that support a cheap availability probe."""
        if provider_name != "ollama":
            return model

        list_models = getattr(provider_instance, "list_models", None)
        if not callable(list_models):
            return model

        from victor.core.errors import ProviderError

        try:
            models = await list_models()
        except Exception as exc:
            base_url = getattr(provider_instance, "base_url", "unknown")
            raise ProviderError(
                message=f"Ollama server unavailable at {base_url}: {exc}",
                provider="ollama",
                raw_error=exc,
            ) from exc

        available_names = [
            str(item.get("name") or item.get("model") or "").strip()
            for item in models
            if isinstance(item, dict)
        ]
        available_names = [name for name in available_names if name]

        if model is None:
            if not available_names:
                raise ProviderError(
                    message="Ollama server is reachable but has no models available",
                    provider="ollama",
                )
            model = available_names[0]
            logger.info(
                "[init] No model configured for Ollama init synthesis; using %s",
                model,
            )

        if model:
            available_name_set = set(available_names)
            if available_name_set and model not in available_name_set:
                logger.warning(
                    "[init] Ollama model %s not reported by %s (available: %s)",
                    model,
                    getattr(provider_instance, "base_url", "unknown"),
                    ", ".join(sorted(available_name_set)[:8]),
                )
        return model

    @staticmethod
    def _resolve_provider_request(
        provider: Optional[str],
        model: Optional[str],
    ) -> tuple[str, Optional[str], Optional[str]]:
        """Resolve init provider/model and any provider-construction routing hints.

        Returns:
            Tuple of:
            - canonical provider name to instantiate
            - user-facing/request model to send on chat calls
            - provider-construction model (may include routing suffixes like ``:coding``)
        """
        from victor.config.settings import load_settings

        settings = load_settings()
        provider_settings = getattr(settings, "provider", None)
        profiles = settings.load_profiles() if hasattr(settings, "load_profiles") else {}

        resolved_provider: Optional[str] = None
        resolved_model = model
        requested_provider = provider

        requested_profile = None
        if provider and isinstance(profiles, dict):
            requested_profile = profiles.get(provider)
        if requested_profile is not None:
            resolved_provider = getattr(requested_profile, "provider", None) or provider
            if resolved_model is None:
                resolved_model = getattr(requested_profile, "model", None)

        if resolved_provider is None:
            default_profile = profiles.get("default") if isinstance(profiles, dict) else None
            profile_provider = getattr(default_profile, "provider", None)
            profile_model = getattr(default_profile, "model", None)
            resolved_provider = (
                provider
                or profile_provider
                or getattr(settings, "default_provider", None)
                or getattr(provider_settings, "default_provider", None)
                or "ollama"
            )
            if (
                resolved_model is None
                and profile_model
                and (provider is None or profile_provider == resolved_provider)
            ):
                resolved_model = profile_model
            if resolved_model is None:
                resolved_model = getattr(settings, "default_model", None) or getattr(
                    provider_settings, "default_model", None
                )

        provider_key = str(resolved_provider or "ollama")
        provider_key_lower = provider_key.lower()
        requested_key = str(requested_provider or provider_key).lower()

        canonical_provider = "zai" if provider_key_lower in _ZAI_PROVIDER_ALIASES else provider_key
        provider_init_model = resolved_model
        if canonical_provider == "zai":
            if (
                provider_init_model
                and ":" not in provider_init_model
                and (
                    requested_key in _ZAI_CODING_PROVIDER_ALIASES
                    or provider_key_lower in _ZAI_CODING_PROVIDER_ALIASES
                )
            ):
                provider_init_model = f"{provider_init_model}:coding"

        return canonical_provider, resolved_model, provider_init_model

    @staticmethod
    def _build_profile_bootstrap(
        settings: Any,
        *,
        requested_profile_name: str,
        profile: Any,
        model_override: Optional[str],
    ) -> InitProviderBootstrap:
        """Build init bootstrap settings from a named profile.

        This mirrors the chat/orchestrator profile path: preserve the profile's
        provider extras and resolve provider kwargs through
        ``Settings.get_provider_settings(...)`` instead of reconstructing just
        provider/model manually.
        """
        effective_provider = getattr(profile, "provider", None) or requested_profile_name
        request_model = model_override or getattr(profile, "model", None)
        temperature = float(getattr(profile, "temperature", 0.7) or 0.7)
        max_tokens = int(getattr(profile, "max_tokens", 4096) or 4096)
        profile_extras = dict(getattr(profile, "__pydantic_extra__", {}) or {})

        # Thread the profile's credential-identity account so init resolves the
        # profile's OWN key/endpoint (account-scoped), matching the chat/
        # orchestrator path (orchestrator_creation). Mirrors #703's account
        # threading for the profile-backed bootstrap.
        provider_kwargs = dict(
            settings.get_provider_settings(
                effective_provider,
                profile_extras,
                account_name=getattr(profile, "account", None),
            )
        )
        provider_kwargs["timeout"] = _init_provider_timeout(
            str(effective_provider), provider_kwargs.get("timeout")
        )
        provider_kwargs["max_retries"] = INIT_PROVIDER_MAX_RETRIES

        return InitProviderBootstrap(
            provider_name=str(effective_provider),
            request_model=request_model,
            provider_init_kwargs=provider_kwargs,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    @staticmethod
    def _resolve_provider_bootstrap(
        provider: Optional[str],
        model: Optional[str],
    ) -> InitProviderBootstrap:
        """Resolve provider bootstrap settings using the same profile contract as chat.

        Profile-backed init requests should inherit provider extras, auth/base
        URL resolution, and endpoint switching from ``Settings.get_provider_settings``
        just like chat/orchestrator setup does. Bare provider requests still use a
        compatibility path with minimal overrides.
        """
        from victor.config.settings import load_settings

        settings = load_settings()
        profiles = settings.load_profiles() if hasattr(settings, "load_profiles") else {}

        requested_profile = None
        requested_profile_name: Optional[str] = None
        if provider and isinstance(profiles, dict):
            requested_profile = profiles.get(provider)
            requested_profile_name = provider if requested_profile is not None else None
        elif provider is None and isinstance(profiles, dict):
            requested_profile = profiles.get("default")
            requested_profile_name = "default" if requested_profile is not None else None

        if requested_profile is not None and requested_profile_name is not None:
            return InitSynthesizer._build_profile_bootstrap(
                settings,
                requested_profile_name=requested_profile_name,
                profile=requested_profile,
                model_override=model,
            )

        canonical_provider, resolved_model, provider_init_model = (
            InitSynthesizer._resolve_provider_request(
                provider,
                model,
            )
        )
        requested_key = str(provider or canonical_provider).lower()
        profile_overrides: dict[str, Any] = {}
        if canonical_provider == "zai" and (
            requested_key in _ZAI_CODING_PROVIDER_ALIASES
            or (provider_init_model is not None and provider_init_model.endswith(":coding"))
        ):
            profile_overrides["coding_plan"] = True

        provider_kwargs = dict(
            settings.get_provider_settings(canonical_provider, profile_overrides)
        )
        provider_kwargs["timeout"] = _init_provider_timeout(
            str(canonical_provider), provider_kwargs.get("timeout")
        )
        provider_kwargs["max_retries"] = INIT_PROVIDER_MAX_RETRIES
        if provider_init_model is not None and (
            provider_init_model != resolved_model or ":" in provider_init_model
        ):
            provider_kwargs["model"] = provider_init_model

        return InitProviderBootstrap(
            provider_name=canonical_provider,
            request_model=resolved_model,
            provider_init_kwargs=provider_kwargs,
            temperature=0.7,
            max_tokens=4096,
        )

    @staticmethod
    def _resolve_provider_selection(
        provider: Optional[str],
        model: Optional[str],
    ) -> tuple[str, Optional[str]]:
        """Resolve canonical provider/model using profile-aware init routing rules."""
        resolved_provider, resolved_model, _ = InitSynthesizer._resolve_provider_request(
            provider,
            model,
        )
        return resolved_provider, resolved_model

    @staticmethod
    def _resolve_local_fallback_selection(
        *,
        exclude_provider: Optional[str] = None,
    ) -> Optional[tuple[str, Optional[str]]]:
        """Pick a local Ollama fallback for init synthesis when remote LLMs are exhausted."""
        from victor.config.settings import load_settings

        excluded = str(exclude_provider or "").lower()
        if excluded == "ollama":
            return None

        settings = load_settings()
        provider_settings = getattr(settings, "provider", None)
        profiles = settings.load_profiles() if hasattr(settings, "load_profiles") else {}

        default_profile = profiles.get("default") if isinstance(profiles, dict) else None
        if getattr(default_profile, "provider", None) == "ollama":
            return "ollama", getattr(default_profile, "model", None)

        if isinstance(profiles, dict):
            for profile in profiles.values():
                if getattr(profile, "provider", None) == "ollama":
                    return "ollama", getattr(profile, "model", None)

        default_provider = getattr(settings, "default_provider", None) or getattr(
            provider_settings,
            "default_provider",
            None,
        )
        default_model = getattr(settings, "default_model", None) or getattr(
            provider_settings,
            "default_model",
            None,
        )
        if default_provider == "ollama":
            return "ollama", default_model

        return "ollama", None

    async def _maybe_retry_with_local_fallback(
        self,
        prompt: str,
        *,
        failed_provider: Optional[str],
        failed_model: Optional[str],
        error: Exception,
    ) -> Optional[str]:
        """Retry init synthesis with a local model once after a recoverable remote failure.

        Triggers on rate limiting, timeouts, and connection errors — a slow or
        throttled *cloud* provider can fall back to local Ollama. A local provider
        that itself times out resolves to no fallback (the excluded provider is
        the only local one), so the caller degrades to the template instead.
        """
        from victor.core.errors import ProviderError, ProviderRateLimitError

        try:
            from victor.core.errors import ProviderTimeoutError
        except Exception:  # pragma: no cover - defensive import
            ProviderTimeoutError = ()  # type: ignore[assignment, misc]

        recoverable = isinstance(error, (ProviderRateLimitError, ProviderTimeoutError))
        # Bare connection/timeout errors surface as ProviderError or OSError.
        if not recoverable and isinstance(error, (ProviderError, OSError, TimeoutError)):
            message = str(error).lower()
            recoverable = any(
                token in message for token in ("timed out", "timeout", "connection", "unavailable")
            )
        if not recoverable:
            return None

        fallback = self._resolve_local_fallback_selection(exclude_provider=failed_provider)
        if fallback is None:
            return None

        fallback_provider, fallback_model = fallback
        logger.warning(
            "[init] Provider %s/%s failed (%s); retrying synthesis with local fallback %s/%s",
            failed_provider,
            failed_model,
            type(error).__name__,
            fallback_provider,
            fallback_model,
        )
        return await self._run_with_fresh_agent(
            prompt,
            fallback_provider,
            fallback_model,
            allow_local_fallback=False,
        )

    async def _call_initialized_provider(
        self,
        prompt: str,
        provider_instance: Any,
        model: Optional[str],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Single direct call using an already-initialized provider instance.

        Used when a framework Agent is passed: reuses the provider that was
        initialized by AgentFactory (with keyring access) rather than creating a
        new instance via ProviderRegistry.create() which skips keyring in
        non-interactive mode.  This is the same mechanism `victor chat` uses.
        """
        import time as _time

        from victor.providers.base import Message

        provider_name = getattr(provider_instance, "name", type(provider_instance).__name__)
        resolved_provider = provider_name
        resolved_model = model
        if provider_name == "ollama":
            resolved_provider, resolved_model = self._resolve_provider_selection(
                provider_name,
                model,
            )
            resolved_model = await self._preflight_provider(
                resolved_provider,
                provider_instance,
                resolved_model,
            )
        messages = [Message(role="user", content=prompt)]

        logger.info(
            "[init→LLM] provider=%s model=%s prompt_chars=%d prompt_lines=%d (reused)",
            resolved_provider,
            resolved_model,
            len(prompt),
            prompt.count("\n"),
        )
        _start = _time.monotonic()

        chat_kwargs: dict[str, Any] = {
            "temperature": float(temperature if temperature is not None else 0.7),
            "max_tokens": int(max_tokens if max_tokens is not None else 4096),
        }
        if resolved_model is not None:
            chat_kwargs["model"] = resolved_model
        try:
            response = await provider_instance.chat(messages=messages, **chat_kwargs)
        except Exception as exc:
            fallback_result = await self._maybe_retry_with_local_fallback(
                prompt,
                failed_provider=resolved_provider,
                failed_model=resolved_model,
                error=exc,
            )
            if fallback_result is not None:
                return fallback_result
            raise

        _elapsed_ms = (_time.monotonic() - _start) * 1000
        content = response.content if response else ""
        result = self._clean(content)

        logger.info(
            "[init←LLM] provider=%s model=%s duration=%.1fs "
            "response_chars=%d response_lines=%d usage=%s",
            resolved_provider,
            resolved_model,
            _elapsed_ms / 1000,
            len(result),
            result.count("\n"),
            getattr(response, "usage", None),
        )
        return result

    async def _run_with_fresh_agent(
        self,
        prompt: str,
        provider: Optional[str],
        model: Optional[str],
        vertical: Optional[str] = None,
        allow_local_fallback: bool = True,
    ) -> str:
        """Run synthesis using a direct provider call with framework logging.

        Uses ProviderRegistry (not Agent.run()) to avoid the agentic loop:
        - No tool calling, no continuation nudges, no multi-turn
        - Single LLM call: prompt in → markdown out
        - Still gets provider-level logging (API_CALL_START/SUCCESS)
        """
        provider_instance: Any = None
        try:
            from victor.providers.base import Message
            from victor.providers.registry import ProviderRegistry

            bootstrap = self._resolve_provider_bootstrap(provider, model)

            provider_instance = ProviderRegistry.create(
                bootstrap.provider_name,
                **bootstrap.provider_init_kwargs,
            )
            if not provider_instance:
                raise RuntimeError(f"Could not create provider {bootstrap.provider_name}")

            model = await self._preflight_provider(
                bootstrap.provider_name,
                provider_instance,
                bootstrap.request_model,
            )

            import time as _time

            messages = [Message(role="user", content=prompt)]

            logger.info(
                "[init→LLM] provider=%s model=%s prompt_chars=%d prompt_lines=%d",
                bootstrap.provider_name,
                model,
                len(prompt),
                prompt.count("\n"),
            )
            _start = _time.monotonic()

            # Only pass model when explicitly set — lets provider use its own default
            # when model=None (passing None overrides the method's default parameter)
            chat_kwargs: dict[str, Any] = {
                "temperature": bootstrap.temperature,
                "max_tokens": bootstrap.max_tokens,
            }
            if model is not None:
                chat_kwargs["model"] = model
            try:
                response = await provider_instance.chat(messages=messages, **chat_kwargs)
            except Exception as exc:
                if allow_local_fallback:
                    fallback_result = await self._maybe_retry_with_local_fallback(
                        prompt,
                        failed_provider=bootstrap.provider_name,
                        failed_model=model,
                        error=exc,
                    )
                    if fallback_result is not None:
                        return fallback_result
                raise
            _elapsed_ms = (_time.monotonic() - _start) * 1000
            content = response.content if response else ""
            result = self._clean(content)

            if not result:
                from victor.core.errors import ProviderError

                raise ProviderError(
                    message="Init synthesis returned empty content",
                    provider=bootstrap.provider_name,
                )

            logger.info(
                "[init←LLM] provider=%s model=%s duration=%.1fs "
                "response_chars=%d response_lines=%d usage=%s",
                bootstrap.provider_name,
                model,
                _elapsed_ms / 1000,
                len(result),
                result.count("\n"),
                getattr(response, "usage", None),
            )

            # Log to usage.jsonl for GEPA/MIPROv2/CoT learning
            try:
                from victor.observability.analytics.logger import UsageLogger
                from pathlib import Path

                logs_dir = Path.home() / ".victor" / "logs"
                logs_dir.mkdir(parents=True, exist_ok=True)
                usage = UsageLogger(log_file=logs_dir / "usage.jsonl", enabled=True)

                # Emit task_classification so GEPA trace collection
                # categorizes this session as init_synthesis (not 'default')
                usage.log_event(
                    "task_classification",
                    {
                        "task_type": "init_synthesis",
                        "provider": bootstrap.provider_name,
                        "model": model,
                    },
                )

                usage.log_event(
                    "tool_call",
                    {
                        "tool_name": "init_synthesis",
                        "tool_args": {
                            "provider": bootstrap.provider_name,
                            "model": model,
                            "prompt_chars": len(prompt),
                        },
                    },
                )
                usage.log_event(
                    "tool_result",
                    {
                        "tool_name": "init_synthesis",
                        "success": bool(result),
                        "duration_ms": round(_elapsed_ms, 1),
                        "result_lines": result.count("\n"),
                        "result_chars": len(result),
                    },
                )

                # Log init output quality signal for GEPA evolution
                self._log_init_quality(usage, result)
            except Exception:
                pass  # Usage logging is best-effort

            return result
        except Exception as e:
            logger.warning(
                "Init synthesis via provider failed [%s]: %s — falling back to template",
                _classify_init_failure(e),
                e,
            )
            raise
        finally:
            if provider_instance is not None:
                close = getattr(provider_instance, "close", None)
                if callable(close):
                    await close()

    async def _run_agent_with_tools(
        self,
        prompt: str,
        provider: Optional[str],
        model: Optional[str],
        vertical: Optional[str] = None,
    ) -> str:
        """Run synthesis with Agent tools (fallback path — needs agentic loop)."""
        try:
            from victor.framework.agent import Agent

            kwargs: dict[str, Any] = {"enable_observability": True}
            if provider:
                kwargs["provider"] = provider
            if model:
                kwargs["model"] = model
            if vertical:
                kwargs["vertical"] = vertical

            agent = await Agent.create(**kwargs)
            result = await agent.run(prompt)
            return self._clean(result.content) if result.success else ""
        except Exception as e:
            logger.warning("Init synthesis with tools failed: %s", e)
            return ""

    @staticmethod
    def _has_document_structure(content: str) -> bool:
        """True if content contains at least one markdown heading.

        Used as a sanity check after cleanup — synthesis output that
        survived the trim helpers but has no headings at all is almost
        certainly a chat-style recap, not a real init.md document.
        """
        for line in content.splitlines():
            stripped = line.lstrip()
            if stripped.startswith("#"):
                # Require ``# `` / ``## `` etc. — at least one space
                # after the leading hashes to count as a heading.
                hashes = 0
                while hashes < len(stripped) and stripped[hashes] == "#":
                    hashes += 1
                if hashes < len(stripped) and stripped[hashes] in (" ", "\t"):
                    return True
        return False

    @staticmethod
    def _clean(content: str) -> str:
        """Clean LLM output — strip code fences and meta-commentary.

        Four things get trimmed here, in order:

        1. A wrapping ``` ```markdown … ``` ``` fence (model occasionally
           wraps the whole doc in a single fenced block).
        2. A leading chat-style recap before the first heading (see
           ``_strip_leading_meta_commentary``).
        3. A trailing self-reflective paragraph about tool availability
           or inability to save the file (see
           ``_strip_trailing_meta_commentary``).
        4. An orphan trailing ``` ``` ``` line on its own — happens when
           the preamble trim earlier in the pipeline ate the opening
           fence but the closing fence survived because the model
           appended postamble text after it.
        """
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            lines = lines[1:] if lines[0].startswith("```") else lines
            lines = lines[:-1] if lines and lines[-1].strip() == "```" else lines
            content = "\n".join(lines).strip()
        # Leading meta-commentary trim catches recap-only outputs that
        # the earlier first-heading trim couldn't (because the recap
        # contains no headings to anchor on). Runs before the trailing
        # trim so signal-phrase detection doesn't accidentally fire on
        # body text below.
        content = _strip_leading_meta_commentary(content)
        # Order matters: trim meta-commentary BEFORE stripping orphan
        # trailing fences, because the meta paragraph often sits AFTER
        # the orphan fence. Stripping the fence first wouldn't help
        # since the paragraph below it survives, and stripping meta
        # first exposes the orphan fence so the next step can drop it.
        content = _strip_trailing_meta_commentary(content)
        # Even if the opening fence was already trimmed (e.g. by the
        # preamble-trim path that jumps to the first ``# `` heading),
        # the closing fence may still be sitting on a line by itself
        # in the tail. Drop trailing blanks + fences, alternating, so
        # we handle ``...\n```\n\n``` patterns too.
        lines = content.splitlines()
        while lines and (
            not lines[-1].strip() or lines[-1].strip() in ("```", "```markdown", "```md")
        ):
            lines.pop()
        content = "\n".join(lines)
        return content.strip()

    async def _pre_synthesis_discovery(
        self, agent: Optional["AgentOrchestrator"] = None
    ) -> Optional[str]:
        """Perform automated discovery of architectural patterns and hotspots.

        Uses the graph tool (victor.tools.graph_tool) to:
        1. Find coupling hotspots via SQL-powered aggregate analysis.
        2. Find loosely coupled patterns (registries, decorators) via semantic edges.
        """
        try:
            try:
                from victor.core.utils.capability_loader import load_graph_tool_module

                module = load_graph_tool_module()
                graph = module.graph
                GraphMode = module.GraphMode
            except ImportError:
                return {}

            discovery_lines = []

            # 1. Structural Hotspots (SQL-Powered)
            # Find the most-imported modules — reliable indicator of "Hub" classes.
            coupling_sql = (
                "SELECT count(*) as count, dst "
                "FROM graph_edge "
                "WHERE type='IMPORTS' "
                "GROUP BY dst "
                "ORDER BY count DESC "
                "LIMIT 5"
            )
            try:
                coupling_result = await graph(mode=GraphMode.QUERY, query=coupling_sql)
                if coupling_result["success"] and coupling_result["result"]["results"]:
                    discovery_lines.append("### Highly Coupled Modules (Fan-in)")
                    for row in coupling_result["result"]["results"]:
                        discovery_lines.append(f"- {row['dst']} ({row['count']} importers)")
                    discovery_lines.append("")
            except Exception:
                pass

            # 2. Key Components Identification
            # Find modules with the most symbols — indicates core implementation depth.
            size_sql = (
                "SELECT file, count(*) as count "
                "FROM graph_node "
                "WHERE type IN ('class', 'function') "
                "GROUP BY file "
                "ORDER BY count DESC "
                "LIMIT 5"
            )
            try:
                size_result = await graph(mode=GraphMode.QUERY, query=size_sql)
                if size_result["success"] and size_result["result"]["results"]:
                    discovery_lines.append("### Module Implementation Depth")
                    for row in size_result["result"]["results"]:
                        discovery_lines.append(f"- {row['file']} ({row['count']} symbols)")
                    discovery_lines.append("")
            except Exception:
                pass

            # 3. Semantic Pattern Discovery
            # Attempt to find relationships for the top hub (from step 1).
            # This identifies registries, providers, etc.
            try:
                if coupling_result["success"] and coupling_result["result"]["results"]:
                    top_hub = coupling_result["result"]["results"][0]["dst"]
                    semantic_result = await graph(
                        mode=GraphMode.SEMANTIC, node=top_hub, threshold=0.6, top_k=5
                    )
                    if (
                        semantic_result["success"]
                        and semantic_result["result"]["potential_relationships"]
                    ):
                        discovery_lines.append(f"### Semantic Relationships for '{top_hub}'")
                        for rel in semantic_result["result"]["potential_relationships"]:
                            discovery_lines.append(
                                f"- {rel['name']} ({rel['file']}) - score: {rel['similarity']}"
                            )
                        discovery_lines.append("")
            except Exception:
                pass

            return "\n".join(discovery_lines) if discovery_lines else None

        except Exception as e:
            logger.debug(f"Pre-synthesis discovery failed: {e}")
            return None

    @staticmethod
    def _log_init_quality(usage: Any, result: str) -> None:
        """Score init output quality and log for GEPA learning.

        Quality signals:
        - Section completeness: how many expected sections are present
        - Line count: within 80-120 target range
        - Conciseness: no bloated or trivially short output
        """
        expected_sections = [
            "Project Overview",
            "System Flow",
            "Package Layout",
            "Key Entry Points",
            "Architecture Patterns",
            "Development Commands",
            "Dependencies",
            "Configuration",
            "Codebase Scale",
        ]
        sections_found = sum(1 for s in expected_sections if s in result)
        section_score = sections_found / len(expected_sections)

        line_count = result.count("\n") + 1
        # Penalize being outside 60-140 range (relaxed from 80-120)
        if 60 <= line_count <= 140:
            length_score = 1.0
        elif 40 <= line_count <= 200:
            length_score = 0.7
        else:
            length_score = 0.3

        # Combined quality score
        quality_score = section_score * 0.6 + length_score * 0.2 + (0.2 if result else 0.0)

        usage.log_event(
            "init_quality",
            {
                "sections_found": sections_found,
                "sections_total": len(expected_sections),
                "section_score": round(section_score, 2),
                "line_count": line_count,
                "length_score": round(length_score, 2),
                "quality_score": round(quality_score, 2),
                "char_count": len(result),
            },
        )

    @staticmethod
    def _enrich_with_project_docs(base_content: str) -> str:
        """Prepend high-signal project docs to base_content before synthesis.

        Static graph analysis surfaces connectivity metrics but misses: project
        narrative, declared dependencies, dev commands, and architectural decisions.
        This enrichment step reads 4 targeted file categories to fill those gaps —
        deterministic and fast (~milliseconds), works for any language/framework.

        Reading budget (to avoid crowding out graph data):
          CLAUDE.md / AI rules: 5000 chars
          README:               1500 chars
          Build manifest:       1500 chars
          Task runner:          1000 chars
        """
        from pathlib import Path

        cwd = Path.cwd()
        enrichments: list[str] = []

        # 1. AI assistant rules files — highest signal: architecture, patterns, dev commands
        from victor.context.instruction_discovery import discover_instruction_files

        for instruction in discover_instruction_files(cwd):
            if instruction.source_type not in {
                "victor_init",
                "victor_instructions",
                "victor_legacy",
                "agents",
                "claude",
                "copilot",
                "cursor",
            }:
                continue
            try:
                text = instruction.content
                trimmed = text[:5000]
                if len(text) > 5000:
                    trimmed += "\n... (truncated)"
                label = instruction.path.name
                enrichments.append(f"## Project AI Rules ({label})\n\n{trimmed}")
                logger.info(
                    "[init] Enriched with %s (%d chars)",
                    instruction.path,
                    len(trimmed),
                )
                break
            except Exception:
                pass

        # 2. README — project narrative, purpose, quick-start (language-agnostic)
        for name in ("README.md", "README.rst", "README.txt", "README"):
            path = cwd / name
            if path.exists():
                try:
                    text = path.read_text(encoding="utf-8")
                    trimmed = text[:1500]
                    if len(text) > 1500:
                        trimmed += "\n... (truncated)"
                    enrichments.append(f"## README (first 1500 chars)\n\n{trimmed}")
                    logger.info("[init] Enriched with %s (%d chars)", name, len(trimmed))
                    break
                except Exception:
                    pass

        # 3. Build manifest — declared dependencies, project metadata, scripts
        #    Priority: language-specific manifest wins over generic
        build_manifests = [
            # Python
            ("pyproject.toml", "pyproject.toml"),
            ("setup.cfg", "setup.cfg"),
            # Node/TS/Deno
            ("package.json", "package.json"),
            ("deno.json", "deno.json"),
            # Go
            ("go.mod", "go.mod"),
            # Rust
            ("Cargo.toml", "Cargo.toml"),
            # JVM
            ("pom.xml", "pom.xml"),
            ("build.gradle.kts", "build.gradle.kts"),
            ("build.gradle", "build.gradle"),
            # .NET
            ("*.csproj", None),  # glob — handled below
            # Ruby
            ("Gemfile", "Gemfile"),
            # PHP
            ("composer.json", "composer.json"),
            # Swift
            ("Package.swift", "Package.swift"),
        ]
        for name, _ in build_manifests:
            if name == "*.csproj":
                import glob as _glob

                matches = _glob.glob(str(cwd / "**" / "*.csproj"), recursive=True)
                if matches:
                    path = Path(matches[0])
                    name = path.name
                else:
                    continue
            else:
                path = cwd / name
            if path.exists():
                try:
                    text = path.read_text(encoding="utf-8")
                    trimmed = text[:1500]
                    if len(text) > 1500:
                        trimmed += "\n... (truncated)"
                    enrichments.append(f"## Build Manifest ({name})\n\n```\n{trimmed}\n```")
                    logger.info("[init] Enriched with %s (%d chars)", name, len(trimmed))
                    break
                except Exception:
                    pass

        # 4. Task runner — dev commands, build targets (always read, complements manifest)
        task_runners = [
            ("Makefile", _parse_makefile_targets),
            ("justfile", _read_task_file),
            ("Taskfile.yml", _read_task_file),
            ("Taskfile.yaml", _read_task_file),
        ]
        for name, reader in task_runners:
            path = cwd / name
            if path.exists():
                try:
                    snippet = reader(path)
                    if snippet:
                        enrichments.append(f"## Task Runner ({name})\n\n```\n{snippet}\n```")
                        logger.info("[init] Enriched with %s", name)
                    break
                except Exception:
                    pass

        if not enrichments:
            return base_content

        prefix = "\n\n".join(enrichments)
        return f"{prefix}\n\n---\n\n{base_content}"

    @staticmethod
    def _resolve_prompt_optimization_identity(
        *,
        agent: Optional["AgentOrchestrator"] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> tuple[str, str]:
        """Resolve the provider/model identity used for prompt optimization lookup."""
        if agent is not None:
            provider_instance = getattr(agent, "provider", None) or getattr(
                agent, "_provider", None
            )
            provider_name = getattr(provider_instance, "name", None)
            if not provider_name and isinstance(provider_instance, str):
                provider_name = provider_instance
            model_name = getattr(agent, "model", None) or model
            return str(provider_name or provider or "default"), str(model_name or "")

        resolved_provider, resolved_model = InitSynthesizer._resolve_provider_selection(
            provider,
            model,
        )
        return str(resolved_provider or "default"), str(resolved_model or "")

    @staticmethod
    def _get_evolved_rules(
        provider: Optional[str] = None, model: Optional[str] = None
    ) -> Optional[str]:
        """Check if GEPA has evolved RULES for the init synthesis prompt.

        Returns the evolved RULES text if a candidate exists with sufficient
        confidence, otherwise None (caller uses static SYNTHESIS_RULES).
        The rules are injected into the fixed frame — {base_content} is
        always safe because the frame handles it.

        Gated by prompt_optimization.enabled setting.
        """
        try:
            from victor.config.settings import get_settings

            po = getattr(get_settings(), "prompt_optimization", None)
            if po is None or not po.enabled:
                return None
            strategies = po.get_strategies_for_section("INIT_SYNTHESIS_RULES")
            if not strategies:
                return None
        except Exception:
            return None

        try:
            from victor.framework.rl.coordinator import get_rl_coordinator

            coordinator = get_rl_coordinator()
            learner = coordinator.get_learner("prompt_optimizer")
            if learner is None:
                return None
            rec = learner.get_recommendation(
                provider or "default",
                model or "",
                "init_synthesis",
                section_name="INIT_SYNTHESIS_RULES",
            )
            from victor.framework.rl.learners.prompt_optimizer import (
                should_serve_candidate,
            )

            if should_serve_candidate(
                rec,
                exploration_enabled=getattr(po, "exploration_enabled", True),
                exploration_epsilon=getattr(po, "exploration_epsilon", 0.1),
            ):
                rec_metadata = dict(getattr(rec, "metadata", {}) or {})
                learner.record_served(
                    "INIT_SYNTHESIS_RULES",
                    provider or "default",
                    str(rec_metadata.get("prompt_candidate_hash") or ""),
                )
                logger.info(
                    "Using GEPA-evolved init rules " "(gen=%s, confidence=%.2f, %d chars)",
                    rec.reason,
                    rec.confidence,
                    len(rec.value),
                )
                return rec.value
        except Exception:
            pass
        return None

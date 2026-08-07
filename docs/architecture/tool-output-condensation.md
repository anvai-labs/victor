# Tool Output Condensation (rtk co-design)

Status: Phases 1-2 implemented
Date: 2026-08-07

## Problem

Victor's LLM path deliberately receives **full** tool output (the accuracy-first
decision in `victor/agent/services/tool_service.py` — 1 MB formatter cap, the
Squeez `ToolOutputPruner` demoted to user-preview-only). That decision is sound
for file reads and search results, but it leaves the single largest source of
context waste untouched: **dev-command output**. A `pytest` run emits thousands
of per-test progress lines to report one failure; `git status` lists every
untracked file; `pip install` narrates every wheel download. All of it reaches
the model verbatim, and the only backstops are head-keeping line/byte caps —
which drop the *tail*, exactly where failure summaries and error counts live.
Compression exists (`ContextCompactor._smart_truncate`) but fires reactively at
compaction time, after the tokens were already paid for at least once.

## Prior art: rtk (github.com/rtk-ai/rtk)

rtk is a Rust CLI proxy that cuts 60–90% of bash output before an agent reads
it. Reviewed from source, its load-bearing ideas are:

| rtk mechanism | What it does |
|---|---|
| Structural parsers (`src/cmds/`) | State-machine parse of git/cargo/pytest/npm output; failures + summary kept, passes collapsed to counts |
| Declarative TOML filters (`src/filters/`) | Regex strip/keep line filters for the ~65-command long tail, with inline tests validated at build time |
| Frugal flag injection | Rewrites `pytest` → `pytest -q --tb=short -rxX` so less output exists in the first place |
| Tee escape hatch (`src/core/tee.rs`) | Raw output saved to disk on failure with a hint pointing the agent at the full log (lossless) |
| Fail-open passthrough | Unparseable output or unexpected exits fall back to raw (capped) |
| Hook interception (`src/hooks/`) | Claude Code PreToolUse hook rewrites commands — needed **only because rtk sits outside the agent** |
| Savings analytics (`rtk gain`) | Per-command bytes-saved tracking, tokens ≈ bytes/4 |

## Why native, not hooks/skills/prompts

rtk's entire hook subsystem (command rewriting, permission mirroring, trust
verification — ~8 kLOC) exists to compensate for not owning the harness. Victor
owns the shell tool, the cache, the settings layer, and the telemetry pipeline,
so the same effect is achieved at the emission seam in `victor/tools/bash.py`
with none of that machinery — and it composes with Victor's caches (raw output
cached, condensation applied uniformly on hits and misses). Prompt- or
skill-based approaches would spend tokens to ask the model to request less
output; condensation removes the tokens before the model ever sees them.

## Design contract

Condensation must not re-litigate the accuracy-first decision. It is held to a
stricter standard than the deprecated pruner:

1. **Diagnostic content is never dropped** — failures, errors, tracebacks, and
   summary lines are kept verbatim; only pass/progress/noise lines collapse.
2. **Fail-open** — a condenser that cannot confidently parse returns `None`;
   raw output passes through. Any exception inside condensation is swallowed.
3. **Lossless escape hatch** — condensed runs tee raw output to
   `.victor/tool_output/` (20-file rotation, 1 MB cap) and the condensed text
   ends with `[condensed by victor; full output: <path>]` so the model can
   `read` the full log on demand.
4. **Output still looks like real command output** — no novel formats.
5. **Piped commands are never condensed** — the pipe already transformed the
   output.

## Implementation (Phase 1)

- `victor/tools/output_condenser.py` — registry + two condenser tiers
  mirroring rtk's:
  - **Structural:** `pytest` (FAILURES + short summary + summary line; pass
    progress collapsed), `git status` long format (branch + grouped counts,
    20 entries per group).
  - **Declarative `LineFilterSpec`** (rtk TOML analogue, in Python): git
    push/pull/fetch/clone progress, pip/uv install noise, npm install
    spinners, cargo build `Compiling` lines.
- Wired in `victor/tools/bash.py` on both the cached and live paths, **before**
  generic truncation. The shell cache stores raw streams; condensation applies
  after retrieval so hits and misses behave identically. Results carry a
  `condensed` metadata dict (condenser name, chars before/after, raw log path).
- `_truncate_output_by_lines` (`victor/tools/subprocess_executor.py`) now keeps
  **head + tail** (70/30) instead of head-only, so even unrecognized commands
  keep their diagnostic tail.
- Settings (`victor/config/tool_settings.py`):
  `shell_output_condensation_enabled` (default True),
  `shell_output_raw_tee_enabled` (default True).

## Interaction with existing mechanisms

- **`ContextCompactor`** still applies reactive smart truncation at compaction
  time; condensation reduces what reaches it, it does not replace it.
- **`ToolOutputPruner`** remains preview-only and untouched.
- **Shell command cache / tool result caches** see raw output (cache) and
  condensed output (consumers) consistently.
- **KV prefix caching** is unaffected — condensation touches per-turn tool
  results, not the stable system prefix.

## Phase 2 (implemented)

- **Structural:** `go test` (--- FAIL blocks kept, ok/pass lines collapsed to
  package counts, verbose `=== RUN`/`--- PASS` noise stripped).
- **Line filters:** npm/pnpm/yarn/jest test (PASS/✓ lines stripped), docker
  build/pull progress, make Entering/Leaving-directory noise, apt-get and brew
  install logs.
- **User-extensible overlay** (rtk `.rtk/filters.toml` analogue):
  `{project}/.victor/output_filters.yaml` then `~/.victor/output_filters.yaml`,
  same schema as `LineFilterSpec` (`name`, `match_command`, `strip_lines`,
  `max_lines`, `on_empty`). User filters win over builtins; invalid entries are
  skipped with a warning; files are cached by mtime.

## Future phases
- **Phase 3 — frugal flag injection:** rewrite recognized commands to quieter
  forms (`pytest -q --tb=short -rxX`) before execution, with the original
  command preserved in metadata.
- **Phase 4 — savings telemetry:** aggregate `condensed` metadata into the
  existing usage-analytics pipeline (`rtk gain` equivalent), and use measured
  savings to tune per-condenser aggressiveness.
- **Phase 5 — Rust hot path:** if profiling justifies it, move line filtering
  into the existing `rust/` crates behind the `_NATIVE_AVAILABLE` pattern.
- **Optional rtk backend:** for users with rtk installed, a
  `shell_output_condenser_backend: "rtk"` mode could delegate to the binary
  (100+ commands) with the native Python implementation as fallback. Deferred
  until the native tier proves the seam.

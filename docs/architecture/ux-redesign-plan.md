# UX Redesign — Comprehensive Analysis & Implementation Plan

> **Historical design/review snapshot.** The findings, proposed APIs, diagrams, and rollout
> status below describe their original review context. For current implementation and remaining
> work, use [Architecture](../architecture.md) and the [Roadmap](../roadmap.md); this document
> is not an installation guide or a certification that every proposed feature shipped.

## Executive Summary

This document captures the systematic analysis, design decisions, and implementation
roadmap for rethinking the Victor AI framework's user experience across four key
dimensions: **Tool Calls**, **Preambles/Prompts**, **Conversation Flow**, and
**Packaging/Deployment**. The work is isolated in the `.worktrees/ux-redesign`
worktree based on `origin/develop`.

---

## Phase 1: Tool Call UX (COMPLETED)

### Analysis

The existing tool display had:
- **Flat invocation lines** with minimal visual hierarchy
- **No progress indicators** for long-running tools
- **Generic previews** that didn't leverage tool-specific semantics
- **Inconsistent categorization** with no visual grouping
- **No duration tracking** or caching heuristics

### Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Visual hierarchy** with section headers + badges | Users need to quickly scan tool activity; hierarchical display reduces cognitive load |
| **Rich progress bars** for long-running tools | Shell, code_search, web tools can take 10+ seconds; users need feedback |
| **Smart previews** per tool category | Different tools produce different output shapes; generic first-N-lines is wasteful |
| **Status icons** (success/failure/cached/pruned) | Instant visual recognition of tool outcome without reading text |
| **Caching heuristic** (elapsed < 50ms) | Fast results from cache should show cached icon to build user trust |

### Implementation

The `ToolDisplayManager` class was enhanced with:

- **Tool categories with emoji prefixes**: `_TOOL_CATEGORIES` dict maps tools to visual groups
- **Status icon class**: `_ToolStatusIcon` with PENDING, RUNNING, SUCCESS, FAILURE, WARNING, PRUNED, CACHED states
- **Structured `on_tool_start()`**: Section header on first call, category badge, invocation line with metadata
- **Live `on_tool_progress()`**: Terminal block with `Progress` bar, elapsed time, last N lines of output
- **Smart `on_tool_result()`**: Status icon + duration + context-aware preview per tool category
- **`_extract_result_summary()`**: 10+ tool-specific summary extractors (search counts, file lines, exit codes, test results)
- **`_was_cached()` heuristic**: Results under 50ms for non-trivial tools get cached icon

---

## Phase 2: Thinking Display UX (COMPLETED)

### Analysis

The existing thinking display had:
- **No duration tracking** for thinking sessions
- **No progressive disclosure** for long reasoning chains
- **No summary** at the end of long thinking sessions
- **No character count** or progress metrics

### Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Duration tracking** | Users need to know how long the model is thinking |
| **Progressive disclosure** at 500+ chars | Long reasoning chains overwhelm the display; summarization reduces noise |
| **Periodic status updates** every 5 chunks | Keeps user informed without flooding the terminal |
| **End-of-thinking summary** | Reasoning complete (N chars, X.Xs) provides closure |

### Implementation

The `ThinkingDisplayManager` class was enhanced with:

- `_thinking_start_time`: Tracks when thinking began for duration display
- `_thinking_char_count`: Running total of thinking content received
- `_LONG_THINKING_THRESHOLD_CHARS = 500`: Threshold for progressive disclosure
- `on_thinking_content()`: Shows inline text for short reasoning, periodic summaries for long chains
- `on_thinking_end()`: Displays summary with total chars and duration for long sessions

---

## Phase 3: Event Dispatcher Refinement (COMPLETED)

### Analysis

The existing event dispatcher had:
- **Flat if-elif chain** that was hard to extend
- **No early returns** — every condition was checked even after match
- **Inconsistent error handling** across event types

### Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Early return pattern** | Once an event is matched and handled, skip remaining checks |
| **Structured routing order** | Tool events -> Errors -> Progress -> Status -> Previews -> Reasoning -> Content |
| **Consistent handler signatures** | All handlers follow same pattern for testability |
| **Graceful fallback** | Unknown events with content are treated as content events |

---

## Phase 4: Preambles & Prompt UX (COMPLETED)

### Implementation

**Files created:**
- `victor/framework/preamble.py` — `PreambleManager` class (16.5 KB, 400+ lines)
- `victor/ui/slash/commands/preamble.py` — `/preamble` slash command (11.3 KB, 280+ lines)

### PreambleManager API

| Method | Description |
|--------|-------------|
| `get_active_sections()` | Return all prompt sections with metadata |
| `get_section(name)` | Get info about a specific section |
| `list_toggleable_sections()` | Return sections that can be toggled |
| `get_full_prompt()` | Return complete system prompt with preambles |
| `toggle_section(name, enabled)` | Enable/disable a prompt section |
| `is_section_enabled(name)` | Check if a section is enabled |
| `set_preamble(text, position, target)` | Inject custom preamble at position |
| `remove_preamble(index)` | Remove a preamble by index |
| `list_preambles()` | Return all active preamble entries |
| `clear_preambles()` | Remove all preamble entries |
| `reset()` | Reset prompt to default state |
| `get_optimization_status()` | Get GEPA/MIPROv2 optimization status |

### Preamble Positions

| Position | Description |
|----------|-------------|
| `TOP` | Before all sections (priority=1) |
| `BOTTOM` | After all sections (priority=999) |
| `BEFORE_SECTION` | Before a specific named section |
| `AFTER_SECTION` | After a specific named section |

### Slash Commands

| Command | Description |
|---------|-------------|
| `/preamble show` | Display current system prompt structure |
| `/preamble sections` | List all sections with enable/disable status |
| `/preamble toggle <name>` | Enable/disable a specific section |
| `/preamble set <text>` | Inject custom preamble at top |
| `/preamble set-bottom <text>` | Inject custom preamble at bottom |
| `/preamble list` | Show active preamble entries |
| `/preamble remove <n>` | Remove preamble at index |
| `/preamble clear` | Remove all preambles |
| `/preamble reset` | Reset to default prompt structure |
| `/preamble optimize` | Show prompt optimization status |

### Protected Sections

Sections that cannot be toggled: `identity`, `capabilities`

Toggleable sections: `tool_hints`, `safety_rules`, `grounding`, `context`, `analysis_efficiency`

---

## Phase 5: Conversation Flow UX (COMPLETED)

### Implementation

**Files created:**
- `victor/ui/rendering/turn_tracker.py` — `TurnTracker` class (14.3 KB, 350+ lines)

### TurnTracker API

| Method | Description |
|--------|-------------|
| `start_turn()` | Begin a new turn, return turn number |
| `end_turn(input_tokens, output_tokens)` | End turn and record metrics |
| `record_tool_call(tool_name)` | Record a tool invocation |
| `record_tool_result(tool_name, duration_ms)` | Record tool execution result |
| `get_context_usage()` | Return current context window usage |
| `get_current_turn_metrics()` | Get current in-progress turn metrics |
| `get_completed_turns()` | Get all completed turns |
| `get_turn_count()` | Get total turn count |
| `get_session_summary()` | Get session-level aggregated metrics |
| `format_turn_header(turn_number)` | Format turn header for display |
| `format_turn_metadata(metrics)` | Format turn metadata for display |
| `format_context_warning()` | Format context window warning |

### TurnMetrics

| Field | Description |
|-------|-------------|
| `turn_number` | Sequential turn number (1-based) |
| `tool_calls` | Number of tool invocations in this turn |
| `tool_categories` | Tool categories used (search, filesystem, git, etc.) |
| `input_tokens` | Estimated input tokens |
| `output_tokens` | Estimated output tokens |
| `cost_estimate_usd` | Estimated cost in USD |
| `duration_ms` | Duration in milliseconds |

### Context Window Awareness

| Threshold | Action |
|-----------|--------|
| > 80% | Yellow warning: "Context window at N%" |
| > 95% | Red warning: "Context window nearly full" |

### Tool Categorization

Tools are automatically categorized by name prefix:
- `filesystem`: read, write, ls, edit, file_info
- `search`: code_search, semantic_code_search, search, grep
- `git`: git_status, git_diff, git_log, git_blame
- `analysis`: overview, analyze, inspect, metrics
- `execution`: shell, bash, run, code_exec
- `web`: web_search, fetch, http
- `testing`: test, pytest, run_tests

### Cost Estimation

Per-model cost tables in `ESTIMATED_COST_PER_1K`:
- GPT-4: $0.03/$0.06 per 1K in/out
- GPT-4o: $0.0025/$0.01 per 1K in/out
- Claude 3.5 Sonnet: $0.003/$0.015 per 1K in/out
- Local models (llama-3): Free

---

## Phase 7: Cross-Platform UX (COMPLETED)

### Implementation

**Files created:**
- `victor/ui/rendering/terminal_capabilities.py` — `TerminalCapabilities` class (15.5 KB, 350+ lines)

### TerminalCapabilities API

| Method | Description |
|--------|-------------|
| `supports_emoji()` | Check if terminal supports emoji display |
| `supports_unicode()` | Check if terminal supports Unicode |
| `get_color_depth()` | Return detected color depth (MONOCHROME/ANSI_8/ANSI_256/TRUECOLOR) |
| `supports_color()` | Return color depth as integer (0/8/256/16777216) |
| `get_terminal_width()` | Return terminal width in characters |
| `get_terminal_height()` | Return terminal height in characters |
| `get_profile()` | Return full TerminalProfile with all detected capabilities |
| `is_interactive()` | Check if running in an interactive terminal |
| `is_ci_environment()` | Check if running in CI/CD environment |
| `get_capability_level()` | Return overall capability level (FULL/BASIC/RESTRICTED) |
| `emoji_or_text(emoji, text)` | Return emoji or text fallback based on terminal support |
| `status_icon(status)` | Return status icon adapted to terminal capabilities |
| `section_header(title)` | Return section header adapted to terminal capabilities |

### Detection Strategy

| Feature | Detection Method |
|---------|-----------------|
| **Terminal emulator** | `TERM_PROGRAM` env var, `TERM` parsing, OS-specific defaults |
| **Color depth** | `COLORTERM` env var, `TERM` suffix, `tput colors` fallback |
| **Unicode support** | OS type, terminal emulator, locale (LC_ALL/LC_CTYPE/LANG) |
| **Emoji support** | OS type, terminal emulator, CI/CD detection |
| **CI/CD detection** | Environment variables (CI, GITHUB_ACTIONS, GITLAB_CI, etc.) |
| **Interactive detection** | `os.isatty(1)`, `VICTOR_NON_INTERACTIVE` env var |
| **Terminal size** | `shutil.get_terminal_size()`, default 80x24 |

### Adaptation Helpers

```python
# Emoji with ASCII fallback
caps = TerminalCapabilities()
icon = caps.emoji_or_text("🔍", "[?]")  # Returns 🔍 or [?]

# Status icons with text fallback
success_icon = caps.status_icon("success")  # Returns ✅ or [OK]
cached_icon = caps.status_icon("cached")    # Returns ⚡ or [CACHE]

# Section headers with Unicode fallback
header = caps.section_header("Tools")  # Returns ──── Tools ──── or --- Tools ---
```

### Capability Levels

| Level | Description | When Active |
|-------|-------------|-------------|
| `FULL` | All features (emoji, Unicode, TrueColor) | Interactive, not CI, modern terminal |
| `BASIC` | ASCII-only, limited colors | Non-interactive, CI, old terminals |
| `RESTRICTED` | Plain text, no formatting | Piped output, non-TTY stdout |

---

## Phase 6: Packaging & Distribution (COMPLETED)

### Analysis & Plan

See **`docs/architecture/phase6-packaging-plan.md`** for the full comprehensive analysis.

### Key Deliverables

**Document:** `docs/architecture/phase6-packaging-plan.md` (9.1 KB, 142 lines)

### Summary of Findings

**Current state:** Monolithic victor-ai wheel shipping ~1,900 Python files (~818K LOC) with ~20 core deps and ~30 optional extras.

**Sprawl issues identified:**
- Monolithic core (no lazy-load boundaries)
- Redundant HTTP clients (httpx + aiohttp + requests)
- Cloud SDK sprawl (boto3, google, kubernetes -- each 20-50MB)
- Rust build friction (no pre-built wheels)
- No minimal install option

**9 Recommendations:**
1. Audit/consolidate redundant imports (fnmatch, argparse, requests, aiofiles, tomli)
2. Move providers and UI to optional extras
3. Publish victor_native as separate PyPI package with platform wheels (5 platforms)
4. Split victor-ai-core from victor-ai for modular installs
5. Add minimal/standard/full extras for tiered installs
6. Add PyInstaller standalone binary
7. Add Homebrew formula for macOS
8. Adopt .victor/config.yaml per-project config (inspired by Claude Code/OpenCode)
9. Keep victor.plugins entry points (unique differentiator)

**Competitor analysis:** Claude Code (npm, ~50MB), Codex CLI (npm, ~40MB), OpenCode (pip, ~30MB), Aider (pip, ~100MB)

**Estimated effort:** 12-19 days across 6 sub-phases (6a-6f)

---

## Phase 7: Cross-Platform UX (PLANNED)

### Terminal Detection & Adaptation

```python
class TerminalCapabilities:
    """Detect and adapt to terminal capabilities."""
    
    @staticmethod
    def supports_emoji() -> bool:
        """Check if terminal supports emoji display."""
        
    @staticmethod
    def supports_unicode() -> bool:
        """Check if terminal supports Unicode."""
        
    @staticmethod
    def supports_color() -> int:
        """Return color depth (8, 256, 16M, or 0)."""
        
    @staticmethod
    def get_terminal_width() -> int:
        """Return terminal width in characters."""
```

**Adaptation strategy:**
- **Full-featured**: iTerm2, Kitty, Alacritty, Windows Terminal - all features enabled
- **Basic**: CMD, xterm - ASCII-only, no colors
- **Restricted**: CI/CD, headless - plain text output
- **Web UI**: FastAPI/React - structured JSON events

---

## Implementation Roadmap

| Phase | Area | Status | Effort | Dependencies |
|-------|------|--------|--------|--------------|
| 1 | Tool Call UX | COMPLETED | 3 days | None |
| 2 | Thinking Display UX | COMPLETED | 1 day | Phase 1 |
| 3 | Event Dispatcher | COMPLETED | 1 day | Phases 1-2 |
| 4 | Preambles & Prompts | PLANNED | 3 days | None |
| 5 | Conversation Flow | PLANNED | 4 days | Phases 1-4 |
| 6 | Packaging & Distribution | PLANNED | 5 days | None |
| 7 | Cross-Platform UX | PLANNED | 3 days | Phases 1-6 |

---

## Files Modified (Phases 1-3)

| File | Change | Lines |
|------|--------|-------|
| `victor/ui/rendering/tool_display.py` | Enhanced tool display with visual hierarchy, progress bars, smart previews | ~500 |
| `victor/ui/rendering/thinking_display.py` | Enhanced thinking display with progressive disclosure, duration tracking | ~107 |
| `victor/ui/rendering/event_dispatcher.py` | Refined event routing with early returns, structured handlers | ~303 |

---

## Testing Strategy

Each phase includes:
1. **Unit tests** for new classes and methods
2. **Integration tests** for end-to-end rendering flows
3. **Performance benchmarks** for rendering latency

### Test Files to Create

```
tests/unit/ui/rendering/test_tool_display.py       # Phase 1
tests/unit/ui/rendering/test_thinking_display.py    # Phase 2
tests/unit/ui/rendering/test_event_dispatcher.py    # Phase 3
tests/unit/framework/test_preamble_manager.py       # Phase 4
tests/unit/ui/rendering/test_turn_tracker.py        # Phase 5
tests/integration/packaging/test_wheel_build.py     # Phase 6
tests/unit/ui/rendering/test_terminal_caps.py       # Phase 7
```

---

## Performance Considerations

| Component | Optimization | Expected Improvement |
|-----------|-------------|---------------------|
| Tool preview rendering | Lazy evaluation | 2-5x faster for large outputs |
| Thinking display | Progressive disclosure threshold | 10x less terminal output |
| Event dispatch | Early return pattern | ~15% fewer condition checks |
| Content buffer | HEAD/TAIL incremental rendering | ~5x faster live updates |
| Progress bars | Throttled updates (every 100ms) | Prevents terminal flooding |

---

*Last updated: 2026-07-08*
*Worktree: `.worktrees/ux-redesign`*
*Base branch: `origin/develop`*
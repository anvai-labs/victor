# Skills System

Skills are composable units of agent expertise that sit between verticals (domain apps) and tools (atomic operations).

```
Vertical (domain app) → Skills (expertise) → Tools (operations)
```

## Quick Start

```python
from victor_contracts.skills import SkillDefinition

# Define a skill
my_skill = SkillDefinition(
    name="debug_test_failure",
    description="Diagnose and fix a failing test",
    category="coding",
    prompt_fragment=(
        "1. Read the failing test\n"
        "2. Run it to see the error\n"
        "3. Trace the code path\n"
        "4. Apply a fix\n"
        "5. Re-run to verify"
    ),
    required_tools=["read", "shell", "edit"],
    optional_tools=["grep", "code_search"],
    tags=frozenset({"debug", "test", "fix"}),
    max_tool_calls=25,
)
```

## Architecture

### SkillDefinition (SDK)

A frozen dataclass in `victor_contracts.skills.definition`:

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Unique identifier |
| `description` | `str` | Human-readable description |
| `category` | `str` | Skill category (e.g., "coding") |
| `prompt_fragment` | `str` | Injected into agent prompt when active |
| `required_tools` | `List[str]` | Tools that MUST be available |
| `optional_tools` | `List[str]` | Tools that MAY be used |
| `tags` | `FrozenSet[str]` | Searchable tags |
| `max_tool_calls` | `int` | Max tool calls when active (default: 20) |
| `version` | `str` | Skill version (default: "1.0.0") |

### SkillProvider (SDK)

Protocol for classes that provide skills:

```python
from victor_contracts.skills import SkillProvider

class MyProvider:
    def get_skills(self) -> List[SkillDefinition]:
        return [my_skill]
```

### SkillRegistry (Framework)

Discovers and manages skills from verticals, plugins, and entry points:

```python
from victor.framework.skills import SkillRegistry

registry = SkillRegistry()

# Load from a vertical class
registry.from_vertical(CodingAssistant)

# Load from entry points
registry.from_entry_points()

# Search skills
results = registry.search("debug", category="coding")
```

## Adding Skills to a Vertical

Override `get_skills()` in your vertical:

```python
from victor_contracts.verticals.protocols.base import VerticalBase
from victor_contracts.skills import SkillDefinition

class MyVertical(VerticalBase):
    @classmethod
    def get_skills(cls) -> list:
        return [
            SkillDefinition(
                name="my_skill",
                description="Does something useful",
                category="my_vertical",
                prompt_fragment="Step-by-step instructions...",
                required_tools=["read", "write"],
            ),
        ]
```

## Entry Point Registration

External packages can register skills via the `victor.skills` entry point:

```toml
# pyproject.toml
[project.entry-points."victor.skills"]
my_skill = "my_package.skills:my_skill_definition"
```

## CLI

```bash
victor skill list                    # List all discovered skills
victor skill list --category coding  # Filter by category
victor skill info debug_test_failure # Show skill details
victor skill search "test debug"     # Search by keyword
```

## Built-in Skills (victor-coding)

| Skill | Description | Required Tools |
|-------|-------------|---------------|
| `debug_test_failure` | Diagnose and fix failing tests | read, shell, edit |
| `code_review` | Review code for correctness and security | read, grep |
| `implement_feature` | Implement a feature with tests | read, edit, write, shell |
| `refactor_code` | Refactor without changing behavior | read, edit, shell |
| `explore_codebase` | Understand codebase architecture | read, ls, grep |
| `text_to_sql` | Answer data questions over SQL/DuckDB/Parquet/Iceberg with derived schema grounding | database, read, shell |
| `agentbrowser` | Drive a live browser session via the AgentBrowser CLI (observe narrowly, act by ref) | shell, read |

The `agentbrowser` skill (category `browser`) conforms the upstream
[AgentBrowser agent skill](https://github.com/anvai-labs/agentbrowser/blob/develop/docs/skills/agentbrowser-cli/SKILL.md)
to victor's skill format: it teaches the observe-narrowly / act-by-ref loop, stale-ref
re-observation, `--operation-id` reconciliation for mutations, and the untrusted-page-content
invariant, driving the CLI at `http://127.0.0.1:5709` through the shell tool.

## Data & Text-to-SQL

The `text_to_sql` skill (category `data`) teaches the agent a validated methodology for
natural-language data questions — grounded in the database itself rather than memorized
knowledge:

1. **Connect & explore first** — relational databases via the `database` tool
   (SQLite/PostgreSQL/MySQL/SQL Server); DuckDB for Parquet/Iceberg files
   (`pip install 'victor-ai[data]'`, then query `read_parquet('file.parquet')` directly).
2. **Derive grounding, never hand-write it** — exact column names from introspection; join
   hints from shared column names; coded values verbatim from column documentation; value
   domains from `SELECT DISTINCT` so filters carry exact literals and casing.
3. **Generate → execute → verify** — SQL only from derived context; errors feed back into
   a retry; results are sanity-checked before reporting.

The methodology is anti-overfitting by construction: nothing about a specific dataset is
hard-coded, so the same skill transfers to any database with documentation.

### User-YAML form (override or standalone)

Skills are source-agnostic: a YAML file in `~/.victor/skills/` with the **same name
overrides the built-in** (the registry is keyed by name); a different name adds a new
skill. Copy this to `~/.victor/skills/text_to_sql.yaml` to customize the built-in without
touching the installed package:

```yaml
name: text_to_sql
description: >-
  Answer natural-language questions over any database - SQLite, PostgreSQL, MySQL,
  SQL Server, DuckDB, Parquet, or Iceberg - by deriving schema grounding from the
  database itself (introspection + column documentation), generating SQL, and
  validating the answer by execution
category: data
prompt_fragment: |
  SKILL: text_to_sql
  1. Connect & explore before writing SQL (database tool; duckdb for parquet/iceberg)
  2. Derive grounding: exact column names, join hints from shared names, coded values
     from docs, value domains via SELECT DISTINCT (exact literals and casing)
  3. Generate SQL only from derived context
  4. Validate by execution; retry with the error fed back (max 2)
  5. Verify magnitudes; report SQL with the answer; state interpretations of ambiguity
required_tools: [database, read, shell]
optional_tools: [grep, ls, write]
tags: [sql, text-to-sql, nl2sql, database, query, duckdb, parquet, iceberg, schema, analytics]
max_tool_calls: 30
```

Auto-selection picks skills by embedding similarity between the user's message and the
skill's `name`/`description`/`tags` — keep those fields vocabulary-rich so the right
skill fires on the right questions.

# Multi-Agent Formations with InferFlux

**Date:** 2026-09-17 · **Status:** All six formations verified live on the R9700 (Qwen3-Coder-30B, ROCm/WSL2).

## Overview

Victor's team framework supports nine formation patterns for multi-agent
coordination. The original six have live-verified examples
using InferFlux (self-hosted, Qwen3-Coder-30B on AMD Radeon AI PRO R9700).

## Prerequisites

- InferFlux server running on the R9700 (see model store and launch recipe)
- SSH tunnel: `ssh -N -L 8080:127.0.0.1:8080 vsingh@aiserver1`
- Victor's InferFlux provider configured (`openai_compat_model_policy.yaml`)
- Per-member session ids (the provider's session-KV cache keys on
  `x-inferflux-session-id`; members sharing a handle cross-contaminate)

## Quick Start

```python
import asyncio
from victor.framework import Agent
from victor.framework.teams import AgentTeam, TeamMemberSpec, TeamFormation

async def main():
    team = await Agent.create_team(
        name="my-team",
        goal="Describe the objective",
        members=[
            TeamMemberSpec(role="executor", goal="Do X", tool_budget=10),
            TeamMemberSpec(role="reviewer", goal="Verify X", tool_budget=8),
        ],
        formation=TeamFormation.PIPELINE,
        provider="inferflux",
        model="qwen3-coder-30b",
        timeout_seconds=300,
    )
    result = await team.run()
    print(result.final_output)

asyncio.run(main())
```

## Formations

### SEQUENTIAL

Members execute one after another; context chains from member to member.

```python
team = await Agent.create_team(
    name="research-then-write",
    goal="Research X, then write a summary",
    members=[
        TeamMemberSpec(role="researcher", goal="Research X"),
        TeamMemberSpec(role="executor", goal="Write the summary"),
    ],
    formation=TeamFormation.SEQUENTIAL,
    provider="inferflux", model="qwen3-coder-30b",
)
```

**Verified output** (R9700, 30s): `seq_output.txt` created with correct content.

### PIPELINE

Each member's output feeds into the next, like an assembly line.

```python
team = await Agent.create_team(
    name="write-then-review",
    goal="Create a file, then verify it",
    members=[
        TeamMemberSpec(role="executor", name="writer",
                      goal="Create the file"),
        TeamMemberSpec(role="reviewer", name="checker",
                      goal="Read the file and confirm correctness"),
    ],
    formation=TeamFormation.PIPELINE,
    provider="inferflux", model="qwen3-coder-30b",
)
```

**Verified output** (R9700, 18s): both pipeline stages completed, files created.

### PARALLEL

All members run simultaneously with independent contexts.

```python
team = await Agent.create_team(
    name="independent-tasks",
    goal="Three independent file creations",
    members=[
        TeamMemberSpec(role="executor", name="worker_a", goal="Create file A"),
        TeamMemberSpec(role="executor", name="worker_b", goal="Create file B"),
        TeamMemberSpec(role="executor", name="worker_c", goal="Create file C"),
    ],
    formation=TeamFormation.PARALLEL,
    provider="inferflux", model="qwen3-coder-30b",
)
```

**Verified output** (R9700, 45s): all three members delivered files concurrently.
**Important**: give each member disjoint file paths — shared paths cause
write-guard races between concurrent members.

### HIERARCHICAL

Supervisor delegates to specialists, then synthesises.

```python
team = await Agent.create_team(
    name="supervised-build",
    goal="Implement the feature",
    members=[
        TeamMemberSpec(role="executor", name="builder",
                      goal="Implement the code"),
        TeamMemberSpec(role="reviewer", name="inspector",
                      goal="Verify the implementation"),
    ],
    formation=TeamFormation.HIERARCHICAL,
    provider="inferflux", model="qwen3-coder-30b",
)
```

**Verified output** (R9700, 19s): implementer created the file, inspector verified it.

### CONSENSUS

All members must agree before the result is accepted.

```python
team = await Agent.create_team(
    name="agreement",
    goal="All members must agree on the approach",
    members=[
        TeamMemberSpec(role="executor", name="proposer", goal="Propose X"),
        TeamMemberSpec(role="executor", name="validator", goal="Validate X"),
    ],
    formation=TeamFormation.CONSENSUS,
    provider="inferflux", model="qwen3-coder-30b",
)
```

### REFLECTION

Generator produces, critic reviews, loop refines until satisfied.

```python
team = await Agent.create_team(
    name="generate-and-reflect",
    goal="Write and verify",
    members=[
        TeamMemberSpec(role="executor", name="generator", goal="Write X"),
        TeamMemberSpec(role="reviewer", name="critic", goal="Review X"),
    ],
    formation=TeamFormation.REFLECTION,
    provider="inferflux", model="qwen3-coder-30b",
)
```

## Additional formations (unit validated)

These opt-in strategies are wired through the same coordinator registry. They have
coordinator-dispatch coverage; **no live InferFlux result is claimed for them yet**.
The six original formations retain their existing defaults.

### DYNAMIC_ROUTER

Selects **one member**, preserving its structured outcome, tools used, and member ID.
The default programmatic category selector matches domain roles (`executor`,
`researcher`, `reviewer`); it does not parse model prose. An unmatched task selects the
first member and emits a warning. Explicit routes map keywords to unique member names:

```python
team = await AgentTeam.create_router_team(
    orchestrator=agent.get_orchestrator(), name="route", goal="Review the patch",
    members=[
        TeamMemberSpec(role="executor", name="builder", goal="Implement changes"),
        TeamMemberSpec(role="reviewer", name="reviewer", goal="Review once"),
    ],
    routes={"review": "reviewer", "implement": "builder"},
)
result = await team.run()
```

The preset resolves names to the already-created member IDs once, storing
`shared_context["router_routes"]`. Unknown names/IDs are configuration errors.

### MULTI_LEVEL_HIERARCHY

`AgentTeam.create_multi_level_hierarchy_team(..., members=members, max_depth=3,
split_strategy="line")` builds a binary tree in member order. The first member is
the root **supervisor**; internal members synthesize their children's outcomes;
leaves execute disjoint task portions. Every member returns its own `MemberResult`.
A failed descendant remains a failure even if a supervisor produces a synthesis.
Final output includes the member findings and root synthesis.

Use explicit line-separated assignments for meaningful boundaries. `line`, `count`,
and `auto` splitting preserve all input, including remainder text; auto partitions
lines when present, otherwise characters. Empty portions are explicit assignments.
For direct coordinator use, `shared_state["hierarchy"]` accepts a structured tree:
`{"member_id": "root-id", "children": [{"member_id": "child-id"}]}`. Every
configured member must appear exactly once. Unknown IDs, duplicate nodes, missing
members, and trees exceeding `hierarchy_max_depth` fail before execution.
`hierarchy_split_strategy` controls splitting per run.

### ADAPTIVE

`AgentTeam.create_adaptive_team(..., members=members, max_switches=3,
adaptation_strategy="error_rate")` executes the sequential → hierarchical → consensus
cycle, switching within the run when the current attempt fails. Use **idempotent**
member tasks: adaptation may repeat work. Strategy defaults also support elapsed-time
performance scoring; structured feedback requires every member to provide a numeric
`metadata.quality_score` in `[0, 1]`, otherwise execution fails explicitly.

Direct coordinator callers may set `shared_state["adaptive_options"]` with
`formation_cycle`, `max_switches`, `adaptation_strategy`, `performance_threshold`, and
`max_duration_seconds`. `initial_formation_hint` must name a formation in that cycle.
Recursive adaptive and context-role reflection cycles are rejected. All state is local
to the invocation; the same registry resolves both direct and adaptive dispatch.
Switches emit warnings and results contain `current_formation`, `formation_history`,
`formation_switches`, and `performance_score` metadata.

**Durability contract (all three):** `supports_durable_pause()` returns `False`.
Member approvals remain inline; no routing choice, recursive tree cursor, or adaptive
iteration cursor is persisted for partial resume. Do not treat these formations as
member-granular resumable runs. This limitation does not change the existing six
formations' durability support.

## Infrastructure Notes

### Session-Id Isolation

Concurrent members share the parent's ContextVar session id by default. The
provider's session-KV/prefix cache keys on this id (`x-inferflux-session-id`),
so members without distinct ids cross-contaminate. `SubAgent.execute()` and
`stream_execute()` bind `resolve_member_session_id()` per member, ensuring
isolation. If you spawn members directly, set the session id yourself.

### Per-Member Session Ids

The id derivation is: `child_session_id` (explicit) > `{parent}-{member_id}`
(dash format). The dash format keys the InferFlux session-KV cache. Planning
members with worktree isolation use colon format (`{parent}:{team}:{member}`).

### Tool Pruning

Tool pruning (per-turn semantic narrowing) is **opt-in**:
`VICTOR_TOOL_SELECTION=1` or `tools.tool_selection_enabled: true`. Default off:
the full enabled registry is supplied to every LLM call. Pruning reduces token
usage but can starve loops that need tools the ranker did not surface.

### Guardrails

InferFlux's guardrail keyword scan covers user-authored turns only (raw
prompt + user-role messages). System prompts and tool results are excluded —
keyword-blocking those poisons agent sessions permanently.

### Concurrency Limit

InferFlux's KV pool supports `max_parallel_sequences` concurrent sequences
(configured per model in the serving config). The scheduler clamps slot ids
to this limit. For multi-agent workloads with N concurrent members, ensure
`max_parallel_sequences >= N` (the scheduler clamps automatically).

### Capacity admission and tool supply (WS-C)

Enable `shared_context={"capacity_aware_parallelism": True}` for PARALLEL teams.
The coordinator asks the provider's `get_parallel_capacity(model)` for its declared
limit, then queues **every** configured member through that many admission slots.
An explicit `max_workers` further lowers concurrency in this opt-in mode. With the
flag absent, legacy behavior and output bytes remain unchanged, including the older
`max_workers` member-selection limit.

InferFlux accepts `max_parallel_sequences=2` as a provider option, verified against
the running server configuration. Without an operator declaration it queries
`/v1/admin/models` and requires a positive `max_parallel_sequences` field on the
selected model. The current R9700 ROCm server omits that field (G18), so operator
configuration is required; missing or invalid capacity fails explicitly.
Admission limits are per team run, not a global multi-process scheduler.

Each queued member emits `member_throttled` with warning level and
`concurrency_limit`; the normal member-start event occurs after admission. Tool
supply remains opt-in: set each `TeamMemberSpec.allowed_tools` to the tools required
for that member's assignment before supply. Give coding members disjoint paths (or
worktree isolation), and give reviewers read-only tools. Do not enable semantic
pruning merely to reduce N-member supply: it can remove tools a loop needs.

| New event consumer | Decision before landing |
|---|---|
| `MemberEventSink` | Bounded, nonblocking warning event with member ID and capacity |
| Framework stream bridge | Preserve structured event metadata |
| v1 wire serializer | Additive `member_throttled` with `concurrency_limit` and `level=warning` |
| Chat/TUI event mapper | Deliberately ignore this additive event; do not label capacity waiting as human approval |
| Existing clients | Unknown-event ignore behavior retained; no new core event enum |
| Live validation/evidence harness | Assert throttle event plus all deliverables and distinct wire session IDs |

Live harness: `.venv-codesign/bin/python scripts/validation/multiagent_live.py
--output-dir /private/tmp/victor-formation-evidence --capacity 2` with
`INFERFLUX_API_KEY` in the environment and the authorized tunnel active. It records
actual outbound `x-inferflux-session-id` headers through a local proxy, never auth
headers, and runs generated tests using the worktree venv.

## Verified Results

| Formation | Elapsed | Members | Deliverable | Correct |
|---|---|---|---|---|
| SEQUENTIAL | 30.0s | 1 executor | `seq_output.txt` = `sequential-ok` | ✅ |
| PIPELINE | 18.0s | 2 stages | `pipe_a.txt` = `A`, `pipe_b.txt` = `B` | ✅ |
| PARALLEL | 44.7s | 3 concurrent | 3 independent files, correct content | ✅ |
| HIERARCHICAL | 19.0s | 2 (implementer + verifier) | `hier_result.txt` = `hierarchical-ok` | ✅ |
| CONSENSUS | 18.5s | 2 (propose + validate) | `consensus.txt` = `agreed` | ✅ |
| REFLECTION | 48.2s | 2 (generate + critic) | `reflect_output.txt` = `reflected` | ✅ |

Backend: InferFlux serving Qwen3-Coder-30B-A3B (UD-Q4_K_XL, 17GB) on AMD
Radeon AI PRO R9700 (gfx1201, ROCm 7.2, WSL2). KV pool: 65536 ctx / 2
sequences. Per-member session ids bound via
`SubAgentConfig.resolve_member_session_id()`.

#### WS-C recorded live evidence (2026-09-17)

Run `capacity-b74af535a8`: three executor members, operator-verified capacity two,
35.03 seconds, all six module/test deliverables present, independent pytest **3
passed**, three distinct observed InferFlux session headers, and one
`member_throttled` event. Evidence: [capacity admission](evidence/ws-c-capacity.json).
The evidence records a dirty working tree because validation preceded the commit.
Three rejected attempts are retained as observations in G19/G20; completion was
accepted only after independent artifact and pytest checks.

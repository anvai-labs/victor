# Multi-Agent Formations with InferFlux

**Date:** 2026-09-17 · **Status:** All six formations verified live on the R9700 (Qwen3-Coder-30B, ROCm/WSL2).

## Overview

Victor's team framework supports six formation patterns for multi-agent
coordination. This document describes each formation with verified examples
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

# Multi-Agent Formations with InferFlux

**Date:** 2026-09-17 · **Status:** All six formations verified live on the R9700 (Qwen3-Coder-30B, ROCm/WSL2).

## Overview

Victor's team framework supports twelve formation patterns for multi-agent
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

## Role vocabulary

Use **supervisor** for the control plane, **member** for a team execution unit, and
**subagent** for a spawned child. A **reviewer** reviews once; a **critic** evaluates
an iterative loop; a **judge** gives a one-shot verdict over candidates. A
**synthesizer** composes outputs and a **router** classifies and dispatches.
`FormationRole` in `victor.teams.types` defines these coordination identifiers
(and reflection's `generator` producer role). Domain roles such as `executor` and
`researcher` remain `SubAgentRole` values: a domain `reviewer` may be assigned the
formation role `critic` explicitly. New formation APIs use canonical identifiers.

`explicit_supervisor_id` is the single emitted shared-state key. The deprecated
`explicit_manager_id` input alias is consumed with a warning; canonical input wins
if both are present. `set_manager`/`manager` and `max_workers` remain compatibility
API names. External FEP-0006 members use **client/remote member** terminology.

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
        TeamMemberSpec(role="executor", name="member_a", goal="Create file A"),
        TeamMemberSpec(role="executor", name="member_b", goal="Create file B"),
        TeamMemberSpec(role="executor", name="member_c", goal="Create file C"),
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
        TeamMemberSpec(role="executor", name="generator", goal="Write X", formation_role="generator"),
        TeamMemberSpec(role="reviewer", name="critic", goal="Critique X until satisfied", formation_role="critic"),
    ],
    formation=TeamFormation.REFLECTION,
    provider="inferflux", model="qwen3-coder-30b",
)
```

## Additional formations (unit validated)

These opt-in strategies are wired through the same coordinator registry. They have
coordinator-dispatch coverage; **no live InferFlux result is claimed for them yet**.
Their registration preserves the original formation behavior; WS-B hardening is described below.

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

## Formation outcome contracts (WS-B)

- **CONSENSUS:** defaults to three rounds. Successful execution is not agreement:
  members agree when their nonempty `metadata.consensus_key` strings match, or their
  complete output strings match exactly when no key is supplied. The largest matching
  group must meet the threshold; failed members count in the denominator.
  `AgentTeam.create_consensus_team(..., rounds=3, agreement_threshold=0.7,
  supervisor=spec)` lets an optional supervisor's successful final proposal break a
  tie. This remains `consensus_achieved=False`, with `consensus_tie_breaker_id` and
  `consensus_decision` in member metadata. Without resolution, team success is false.
  Direct runs use `consensus_max_rounds`, `consensus_agreement_threshold`, and
  `consensus_tie_breaker_id` in shared state. Existing round checkpointing remains.
- **REFLECTION:** opt into `verdict_format="json"` on `create_reflection_team` (or
  `reflection_verdict_format="json"` in shared state). The critic must return exactly
  `{"verdict":"satisfied","feedback":"All requested checks pass."}` or a
  `needs_work` verdict with actionable feedback. Extra keys, prose, markdown fences,
  and malformed JSON cause an immediate failed result and warning, with no extra
  generation attempt. The format is part of the checkpoint contract and cannot
  change on resume. Default legacy verdict/keyword behavior is retained for
  compatibility, with a warning whenever keyword fallback is used.
- **PARALLEL:** per-member outcomes stay in `member_results`; `final_output` includes
  a deterministic failed-member summary when any member fails. Existing any-member
  success semantics remain, and all-success output is byte-identical. Opt into
  `shared_context={"parallel_member_retries": 1}` (0–3, default 0) only for idempotent
  member tasks. Successful members and approval pauses are never retried. Attempt
  tool/duration totals are accumulated before the member completion checkpoint.

These hardening contracts are unit-tested; the earlier live matrix predates them.

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

### Ensemble aggregation (WS-H)

`AgentTeam.create_ensemble_team(orchestrator, name, goal, candidates, mode="vote")`
creates an opt-in PARALLEL-shaped run. Every candidate independently receives the
same original task plus the JSON response contract. Return exactly
`{"vote_key": "canonical-answer", "answer": "answer or artifact reference"}`.
Voting requires a strict majority; ties and invalid proposals fail explicitly.
All candidate deliverables and costs remain in member results.

Use `mode="judge", aggregator=judge_spec` for one verdict selecting a configured
candidate via `{"selected_member_id": "..."}`. Use `mode="synthesizer"` for a
single MoA-style combination pass returning `{"answer": "..."}`. These roles are
canonical formation roles; caller specs are copied rather than mutated. Candidate
answers exceeding 8192 characters are rejected with guidance to return references.
`create_consensus_team(..., mode="vote")` exposes the same aggregation policy through
CONSENSUS; the default `mode="agreement"` keeps the existing iterative behavior.

Durability: ensemble aggregation rejects checkpoint/resume before member execution.
The original PARALLEL/CONSENSUS durability contracts are unchanged without this mode.
No extra formation enum or registry is introduced for this aggregation policy.

### Opt-in PARALLEL worktree isolation (WS-D)

Use `shared_context={"parallel_worktree_isolation": True, "repo_root": "/repo",
"worktree_parent": "/tmp/member-worktrees", "branch_prefix": "feat/members"}`.
Each member receives a real git worktree; allocation failure stops the run.
The default preserves worktrees for inspection and does not auto-merge. Explicit
`cleanup_worktrees` / `auto_merge_worktrees` retain their existing meanings.
Defaults without the flag remain unchanged.

Member `allowed_tools` must use supported path adapters: `read`, `write`, `edit`,
`shell`, `ls`, `find`, or `overview`. Unsupported tools fail with a warning instead
of running in the parent's directory. `edit` requires structured operation lists.
Relative filesystem paths and shell `cwd` are rooted per member; absolute paths
are preserved. This is working-directory isolation, not a shell security sandbox.
The normal tool safety policy still applies. Nested members inherit bound tools.

Run the worktree's `.venv-codesign/bin/python scripts/validation/multiagent_live.py
--output-dir /tmp/formation-evidence --isolate` with `INFERFLUX_API_KEY` set.
Live run `isolation-18b2e5c728` on 2026-09-17 finished in 37.11 seconds: three
members wrote the same two relative filenames in separate worktrees; each
worktree's independent pytest run passed; all three actual wire session IDs
matched configured members and result metadata. See [recorded evidence](evidence/ws-d-isolation.json).

#### WS-C recorded live evidence (2026-09-17)

Run `capacity-b74af535a8`: three executor members, operator-verified capacity two,
35.03 seconds, all six module/test deliverables present, independent pytest **3
passed**, three distinct observed InferFlux session headers, and one
`member_throttled` event. Evidence: [capacity admission](evidence/ws-c-capacity.json).
The evidence records a dirty working tree because validation preceded the commit.
Three rejected attempts are retained as observations in G19/G20; completion was
accepted only after independent artifact and pytest checks.

WS-H R9700 live evidence (2026-09-17): three independent candidates sampled the
same doubling task in isolated worktrees and returned validated JSON proposals.
Strict majority selected `member.py`; all six deliverables existed, each worktree's
pytest run passed, and the three observed session headers matched configured
member IDs and result metadata. Elapsed: 35.13 seconds.
Run with `scripts/validation/multiagent_live.py --ensemble-vote --output-dir /tmp/evidence`
using the worktree `.venv-codesign/bin/python` and `INFERFLUX_API_KEY`.
See [recorded evidence](evidence/ws-h-ensemble.json). Judge and synthesizer modes
are unit-validated; this live run validates voting.

### Conversation-native formations (WS-G / FEP-0035)

| Formation | Preset | Selection and termination | Durable partial resume |
|---|---|---|---|
| GROUP_CHAT | `create_group_chat_team` | Round-robin; optional `candidate_func` + `selector_func`, or structured router; done/predicate/max turns | Unsupported; rejected before execution |
| DEBATE | `create_debate_team(..., judge=spec)` | Bounded contributions then one judge verdict | Unsupported; rejected before execution |
| HANDOFF | `create_handoff_team(..., start_member="name")` | Structured peer destination carries transcript until done/max turns | Unsupported; rejected before execution |

All three are registered in the canonical formation registry and exercised through
coordinator dispatch. These rows claim unit validation; live validation is not
claimed for the conversation formations. Existing formations retain their defaults.

Each speaking turn returns exactly
`{"content":"message or artifact reference","done":false,"handoff_to":null}`.
Only HANDOFF may supply a destination, and unfinished handoff turns require one.
Self/unknown destinations, contradictory done+handoff, and malformed JSON fail.
A router returns `{"speaker_id":"configured-id"}`; a judge returns
`{"selected_member_id":"speaker-id","verdict":"decision"}`. Contract examples
are included in every structured prompt. No prose parsing or fallback speaker exists.

`conversation_max_turns` defaults to six. `transcript_max_chars` defaults to 32768;
exhaustion fails explicitly rather than silently trimming shared context. Large
artifacts should be referenced by path. Callbacks receive immutable transcript
snapshots; candidate filters return unique eligible IDs, selectors choose one ID,
and `termination_func` returns a boolean. Callback exceptions fail with warnings.
Member results accumulate repeated-turn costs while `shared_context` exposes
`conversation_transcript` and `conversation_termination`.

Consumer decisions are in [FEP-0035](https://github.com/anvai-labs/victor/blob/develop/feps/fep-0035-conversation-native-team-formations.md).
`member_spoke` carries a transcript sequence; `member_handoff` carries source,
target, and sequence. The stream/wire bridge preserves these fields. Existing UI
lanes intentionally ignore these additive events and retain lifecycle rendering.

### WS-E ZAI gateway validation (2026-09-18)

After LAN loss, the user restricted current live testing to ZAI behind loopback
Sandhi. This run makes no InferFlux or cross-vendor claim. The
[setup walkthrough](sandhi-zai-loopback.md) explains provisioning and dashboard access.

The [recorded run](evidence/ws-e-zai-gateway.json) passed in 167.93 seconds: seven
members, fourteen Python deliverables, seven independently passing pytest tests,
and seven distinct observed session/run IDs. A three-member review pipeline paused
on an injected `MemberApprovalPause` before the reviewer's first LLM call; resume
skipped the completed writer. Dynamic selection dispatched PARALLEL; an injected
selector failure dispatched the configured default and emitted a warning event.
All seven member input/output/total usage counters exactly reconciled with
`GET /admin/usage/run/{member_session_id}`. Monetary pricing is not asserted.

New defects discovered by the live sweep are G22–G25 in the handoff. The initial
run was rejected because Victor reported zero usage; a second rejected run exposed
member overrides bypassing the gateway. The accepted evidence uses the corrected
shared accumulator and gateway resolver. Classifiers were not changed.

### All-formation Sandhi matrix

`scripts/validation/formation_gateway_matrix.py` is an opt-in artifact experiment
covering all twelve canonical formations plus ensemble vote, judge and synthesizer.
The older six-case battery remains unchanged. Run the ZAI reference and then the
same scenarios against InferFlux; preserve failed attempts as new evidence rather
than repeating calls to obtain a pass:

```bash
.venv-codesign/bin/python scripts/validation/formation_gateway_matrix.py \
  --provider zai --model glm-5.3 \
  --gateway-state /path/to/private/sandhi-state --output-dir /tmp/zai-matrix
.venv-codesign/bin/python scripts/validation/formation_gateway_matrix.py \
  --provider inferflux --model qwen3-coder-30b \
  --gateway-state /path/to/private/sandhi-state --output-dir /tmp/inferflux-matrix
```

The existing private state supplies `client.json` (ZAI) or `inferflux.json`, each
with gateway `url` and `virtual_key`, plus `admin-token` and `usage.db`. Credentials
remain local. The observer listens on loopback port 18084 and forwards through the
configured gateway; it records only correlation and usage metadata and body hashes.
Reports and member outputs live in a new private experiment directory. Run with
isolated Victor configuration when personal settings could affect the experiment.

Each case requires its declared files, independent bounded pytest, successful
member outcomes, distinct member session IDs, matching model/provider routing,
canonical member usage, and wire/SQLite/C4/dashboard conservation. Reflection's
critic persists a structured review; debate and ensemble aggregation persist a
validated decision. Router acceptance expects only the selected member. Ensemble
candidates and their aggregator use isolated worktrees. Structured task assignments
are carried in the team objective to address the matrix's exposure to G39; this
does not fix the general adapter behavior. Multi-level hierarchy uses an explicit
three-level chain to keep each task assignment intact, exercising nested execution
and synthesis without the default splitter cutting a JSON assignment in half.
Reflection uses `capture_member_usage=True` to retain the actual generator/critic
results and per-member accounting (G40), rather than the legacy synthetic aggregate.
This is a prerequisite for its strict live session/usage acceptance.

With capture enabled, `TeamResult.member_results` contains the configured member
IDs. Each member retains its stable `session_id`, summed `usage`, tool-call and
duration totals, and ordered `reflection_attempts`; `reflection_summary` carries
iteration/verdict metadata. Consumers use those counters for gateway reconciliation
and the attempts for failure diagnosis. The team's final output remains the generated
solution. Missing/inconsistent attribution and invalid verdicts fail explicitly.
Without capture, the existing `reflection_formation` aggregate and legacy defaults
are unchanged. Iteration-boundary checkpoint/resume preserves captured results;
changing capture mode during resume is rejected. This does not add mid-iteration
pause or tool/token-level replay.

Record the actual Victor, Sandhi source/binary and InferFlux serving identities,
origin readiness and gateway configuration with each run. The accepted Mac route
is loopback 18081 to aiserver1:8081, upstream `http://127.0.0.1:18081/v1`; preserve
the old route for rollback. The confirmed buffered gateway deadline is 120 seconds.
The 240-second case budget does not extend it: every timed-out physical attempt
fails acceptance. Do not clear shared cache or redeploy shared services for this
experiment. Buffered reporting does not establish executed cache reuse, tokenizer
equivalence, session leases, streaming, origin cancellation, durable partial resume,
or R9700/full-GPU acceptance. No new live pass is claimed merely by adding this runner.

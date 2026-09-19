# Streaming Runtime

!!! abstract "Current path"

    `ChatService` frames the turn. `ServiceStreamingRuntime` calls
    `StreamingChatExecutor.run_unified()`, which drives the single
    `AgenticLoop.run_streaming()` path through `StreamingActAdapter`.

    The deprecated `run()` alias, `AgenticLoop.stream_chat()` wrapper and
    `_stream_chat_impl` path are removed.

## Turn flow

```mermaid
---
title: Current streaming turn
---
%%{init: {"theme":"base","themeVariables":{"primaryColor":"#E8EFF7","primaryTextColor":"#17324D","primaryBorderColor":"#456987","lineColor":"#456987","fontFamily":"Arial"}}}%%
sequenceDiagram
  actor Client
  participant Chat as ChatService
  participant Frame as ChatTurnRuntime
  participant Runtime as ServiceStreamingRuntime
  participant Exec as StreamingChatExecutor
  participant AL as AgenticLoop
  participant Act as StreamingActAdapter
  Client->>Chat: stream_chat(message)
  Chat->>Frame: enter + start task report
  Chat->>Runtime: stream(message)
  Runtime->>Exec: run_unified(message)
  Exec->>AL: run_streaming(message)
  loop PERCEIVE · PLAN · ACT · EVALUATE · DECIDE
    AL->>Act: stream_turn_act(...)
    Act->>Exec: execute_turn_streaming(...)
    Exec-->>Act: chunks + turn result
    Act-->>AL: chunks + outcome
    AL-->>Exec: stream chunks
    Exec-->>Runtime: stream chunks
    Runtime-->>Chat: stream chunks
    Chat-->>Client: stream chunks
  end
  Runtime->>Exec: close nested generator
  Runtime-->>Chat: stream ends
  Chat->>Frame: finish task report
  Chat->>Frame: exit turn scope
  Chat->>Chat: release turn lock
```

## Capability map

The stream cluster receives an enumerated `ChatRuntimeServices` view. Related operations share
one capability so callers do not accumulate facade fields.

| Capability | Owns | Failure behavior |
| --- | --- | --- |
| Session requirements | Required files, outputs and read state | Live session owner |
| Delivery | Request and response delivery | Existing delivery contract |
| Planning | Goals, intent, guidance and tool selection | Required wiring fails early |
| Governance | Request/response policy | Invalid configured result fails closed |
| Completion | High-confidence completion and clean summary | Optional when disabled |
| Conversation | History, actual usage and terminal summary persistence | Best effort; weak owner |
| Tool calls | Reset, parse and validate | Missing runtime fails closed |
| Feedback | Outcome recording and cancellation | Best effort; weak owner |
| Recovery | Retry and fallback coordination | Existing recovery contract |

## Lifecycle invariants

!!! warning "Required invariants"

    - Close nested async generators before releasing the shared turn lock.
    - Finish task reports on success, error and cancellation.
    - Resolve mutable session and controller state from its current owner.
    - Keep configured governance gates through bootstrap; malformed results are errors.
    - Preserve one accounting path from provider usage to conversation/session totals.

| Guard | What it prevents |
| --- | --- |
| Stream-turn AST caps | New access to migrated orchestrator internals |
| Facade and hotspot caps | Growth in the composition facade |
| Run/stream parity | Divergence between buffered and streaming semantics |
| Lifetime regressions | Lock, generator or report leaks after cancellation |
| Usage regressions | Double counting or loss during reset/restore |

## Ownership status

```mermaid
---
title: FEP-0031 ownership progress
---
%%{init: {"theme":"base","themeVariables":{"primaryColor":"#E8EFF7","primaryTextColor":"#17324D","primaryBorderColor":"#456987","lineColor":"#456987","fontFamily":"Arial"}}}%%
flowchart LR
  subgraph DONE["Implemented"]
    T["Turn frame"]
    P["Planning and guidance"]
    E["Stream execution controls"]
  end
  subgraph NEXT["Remaining"]
    S["Broader runtime state"]
    F["Handler factories"]
    M["Facade shims and mixins"]
  end
  T --> S
  P --> S
  E --> S
  S --> F --> M
```

FEP-0031 remains **Draft / in progress**. See the
[proposal and measured progress](https://github.com/anvai-labs/victor/blob/develop/feps/fep-0031-chat-runtime-inversion.md)
and the [canonical architecture](../architecture.md#service-layer).

## Compatibility

Public `Agent.run()`, `Agent.stream()` and facade chat methods retain their contracts.
Compatibility coordinators forward to the service/runtime path; they do not own a second
streaming engine.

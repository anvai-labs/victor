# Streaming Chat Executor Architecture

> Project-graph analysis (Feb 2025) showed that `_stream_chat_impl` inside the
> old coordinator path still orchestrated every phase of streaming chat. The
> orchestration first moved into a dedicated pipeline module. The live runtime
> has since been consolidated again: the canonical entry point is now the
> service-owned streaming runtime backed by
> `victor.agent.services.chat_stream_executor.StreamingChatExecutor`, not
> `ChatCoordinator.stream_chat()`.

## 1. Responsibilities

| Phase | Responsibilities formerly inside `_stream_chat_impl` | Extracted component |
| --- | --- | --- |
| Session bootstrap | Create `StreamingChatContext`, emit requirement events, reset dedup state | `victor.agent.streaming.context` |
| Iteration loop | Enforce limits, request provider responses, feed tool calls | `StreamingChatHandler`, `ToolExecutionHandler` |
| Intent handling | Run classifier, apply continuation overrides | `IntentClassificationHandler`, `ContinuationHandler` |
| Recovery & fallbacks | Delegate to `RecoveryCoordinator`, `response_completer` | `victor.agent.recovery_coordinator`, `response_completer` |
| Metrics & observability | Emit streaming metrics, cumulative token usage | `StreamingController`, `StreamingCoordinator` |

The original pipeline extraction removed streaming fan-out from the old
coordinator facade. The current runtime keeps that same fan-out inside the
canonical executor rather than inside a deprecated coordinator or a parallel
pipeline surface.

## 2. Canonical Architecture

The [unified streaming sequence](../architecture.md#agenticloop) is the canonical diagram.
`ServiceStreamingRuntime` calls `StreamingChatExecutor.run_unified()`, which drives
`AgenticLoop.run_streaming()` with `StreamingActAdapter`. The adapter invokes
`execute_turn_streaming()` for ACT; the framework loop owns evaluation and continuation.
The deprecated `run()` alias and `AgenticLoop.stream_chat()` wrapper were removed in
ADR-030 step 3.

The remaining sections preserve the earlier extraction history and its test plan;
references to retired coordinators describe that migration, not current entry points.

## 3. Implemented Changes

1. **Executor landing** –
   `victor/agent/services/chat_stream_executor.py` now contains the live
   streaming implementation and reuses dedicated helper modules for intent,
   continuation, tool execution, and recovery.
2. **Service-owned runtime** –
   `victor.agent.services.chat_stream_runtime.ServiceStreamingRuntime` is the
   canonical owner of executor construction and invocation.
3. **Factory exposure** – orchestrator/runtime builders expose
   `create_streaming_chat_executor(...)` and
   `create_service_streaming_runtime(...)` so canonical wiring stays out of the
   deprecated coordinator shims.
4. **Compatibility shim retention** – `ChatCoordinator.stream_chat()` no longer
   owns streaming orchestration. It forwards to the service/runtime surfaces
   and only falls back to the legacy hook for older integrations.

## 4. Testing Strategy

1. **Executor unit tests**
   - Mock/spy runtime-owner helpers to prove the executor exercises the
     expected phases (pre-checks, continuation, tool execution, recovery).
   - Simulate error and cancellation paths to guard the retry logic.
2. **Compatibility tests**
   - Keep explicit coverage that deprecated shims prefer `ChatService`, then
     `ServiceStreamingRuntime`, and only then the legacy hook.
3. **Integration coverage**
   - Maintain `tests/unit/agent/test_orchestrator_core.py` and service
     delegation suites so the service-owned runtime remains the canonical path.
   - Maintain the streaming CLI integration tests to ensure real providers
     and tool executions still behave identically.

## 5. Follow-up Checklist

- [x] Extract streaming orchestration out of `_stream_chat_impl`.
- [x] Move canonical streaming ownership to `ServiceStreamingRuntime`.
- [x] Consolidate the live path onto `StreamingChatExecutor`.
- [x] Reduce `ChatCoordinator.stream_chat` to a compatibility forwarding layer.
- [x] Remove `_stream_chat_impl`.
- [ ] Add dedicated unit tests for executor-level behaviours beyond current service/runtime suites.
- [ ] Add CI checks that fail when streaming fan-out exceeds agreed thresholds.

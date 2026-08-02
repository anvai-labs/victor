# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Deterministic streaming-loop characterization harness — the FEP-0007 regression gate.

FEP-0007 ("Unified Agentic Loop") decomposed ``StreamingChatExecutor.run()`` into a single
``_stream_turn()`` per-turn primitive driven by a thin loop. There is one streaming path — no
feature flag, no legacy fallback — so the way we guard the refactor is to pin the loop's observable
behavior across a fixed QA battery (S1/S2/M1/M2/C1/W1–W4/U1): same answers, same tool sequence, no
streaming tracebacks. Every Phase 2 helper extraction and the ``_stream_turn`` assembly was checked
against this battery.

* :class:`ScriptedProvider` / :class:`ScriptedTool` drive the **real** streaming loop
  deterministically (no network, no real model) via ``orchestrator.stream_chat()``.
* :func:`capture_transcript` runs a scenario and normalizes the streamed output into a
  :class:`StreamTranscript` (content with the non-deterministic cost/token footer stripped,
  ordered tool-execution names, error flag).
* :data:`SCENARIOS` is the battery.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, AsyncIterator, Dict, List, Optional
from unittest.mock import MagicMock, patch

from victor.agent.orchestrator import AgentOrchestrator
from victor.config.settings import ProfileConfig, Settings
from victor.providers.base import (
    BaseProvider,
    CompletionResponse,
    Message,
    StreamChunk,
    ToolDefinition,
)
from victor.tools.base import BaseTool

# ---------------------------------------------------------------------------
# Scripted provider + tools (deterministic, offline)
# ---------------------------------------------------------------------------


@dataclass
class TurnScript:
    """One scripted provider turn.

    A turn with ``tool_calls`` makes the loop execute those tools and continue to the next
    turn; a turn without makes the loop finish. Mirrors how a real model alternates
    tool-calling turns with a final answer.
    """

    content: str = ""
    tool_calls: Optional[List[Dict[str, Any]]] = None
    finish_reason: str = "stop"


class ScriptedProvider(BaseProvider):
    """A provider whose ``stream()`` replays a fixed list of :class:`TurnScript`.

    Each call to ``stream()`` (one per loop iteration) emits the next turn; the last turn
    repeats if the loop asks for more, so spin/repetition scenarios terminate via the loop's
    own guards rather than an ``IndexError``.
    """

    def __init__(self, turns: List[TurnScript]):
        super().__init__(api_key=None)
        self._turns = turns
        self.turn_index = 0

    @property
    def name(self) -> str:
        return "scripted"

    def supports_tools(self) -> bool:
        return True

    def supports_streaming(self) -> bool:
        return True

    async def chat(
        self,
        messages: List[Message],
        *,
        model: str = "scripted",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Replay the next scripted turn as a buffered ``CompletionResponse``.

        The buffered loop (``AgenticLoop`` via ``execute_turn`` ->
        ``orchestrator_protocol_adapter.execute_turn`` -> ``provider.chat()``) consumes turns
        the same way ``stream()`` does for the streaming loop, so one ``ScriptedProvider`` drives
        BOTH loops from the same script — the basis for run/stream parity comparison.
        """
        turn = self._turns[min(self.turn_index, len(self._turns) - 1)]
        self.turn_index += 1
        has_tools = bool(turn.tool_calls)
        return CompletionResponse(
            content=turn.content or "",
            tool_calls=turn.tool_calls,
            stop_reason="tool_calls" if has_tools else turn.finish_reason,
            model="scripted",
        )

    async def stream(
        self,
        messages: List[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        turn = self._turns[min(self.turn_index, len(self._turns) - 1)]
        self.turn_index += 1
        has_tools = bool(turn.tool_calls)
        yield StreamChunk(
            content=turn.content,
            tool_calls=turn.tool_calls,
            stop_reason="tool_calls" if has_tools else turn.finish_reason,
            is_final=True,
        )

    async def close(self) -> None:
        return None


class ScriptedTool(BaseTool):
    """A tool that records its invocation (into a shared log) and returns fixed output.

    ``name``/``description``/``parameters`` are concrete class attributes (satisfying
    ``BaseTool``'s abstract properties); ``__init__`` shadows them per instance for the
    dynamic tool name.
    """

    name = "scripted_tool"
    description = "scripted tool"
    parameters: Dict[str, Any] = {"type": "object", "properties": {}}

    def __init__(self, name: str, recorder: List[str], *, output: str = "", success: bool = True):
        self.name = name
        self.description = f"scripted {name} tool"
        self.parameters = {"type": "object", "properties": {}}
        super().__init__()
        self._recorder = recorder
        self._output = output or f"[{name} output]"
        self._success = success

    async def execute(self, _exec_ctx: Optional[Dict[str, Any]] = None, **kwargs: Any):
        self._recorder.append(self.name)
        return SimpleNamespace(success=self._success, output=self._output, error=None)


# ---------------------------------------------------------------------------
# Orchestrator wiring (drives the REAL StreamingChatExecutor.run())
# ---------------------------------------------------------------------------


def _build_settings(tool_budget: int) -> Settings:
    s = Settings()
    s.analytics_enabled = False
    s.analytics_log_file = None
    s.load_profiles = lambda: {
        "default": ProfileConfig(
            provider="scripted", model="scripted", temperature=0.0, max_tokens=512
        )
    }
    s.tools.use_semantic_tool_selection = False
    s.mcp.mcp_enabled = False
    s.tools.tool_call_budget = tool_budget
    s.security.airgapped_mode = False
    s.load_tool_config = lambda: {}
    return s


def build_streaming_orchestrator(
    provider: ScriptedProvider,
    tool_names: Optional[List[str]] = None,
    *,
    max_iterations: int = 8,
    tool_budget: int = 50,
    failing_tool_names: Optional[List[str]] = None,
) -> AgentOrchestrator:
    """Build a real orchestrator wired to the scripted provider + scripted tools.

    The returned orchestrator drives the genuine ``StreamingChatExecutor.run()`` loop. The
    ordered list of executed tool names is recorded on ``orch._parity_tool_log`` for the
    transcript. Semantic selection is bypassed (``_select_tools`` returns the scripted tools)
    so behavior is deterministic and offline. Tools named in ``failing_tool_names`` return an
    unsuccessful result, exercising the loop's tool-failure recovery.
    """
    tool_names = tool_names or []
    failing = set(failing_tool_names or [])
    recorder: List[str] = []
    tools = [ScriptedTool(n, recorder, success=n not in failing) for n in dict.fromkeys(tool_names)]

    settings = _build_settings(tool_budget)
    with patch("victor.core.bootstrap_services.bootstrap_new_services"):
        orch = AgentOrchestrator(
            settings=settings, provider=provider, model="scripted", temperature=0.0
        )

    for tool in tools:
        orch.tools.register(tool)
    # Keep every component that caches the registry in sync (see test_orchestrator_stream_tool_calls).
    orch.tool_executor.tools = orch.tools
    orch._tool_pipeline.tools = orch.tools
    orch._tool_pipeline.executor.tools = orch.tools

    # Keep the real ToolService (it owns parse_and_validate_tool_calls, used in the loop) and
    # only relax tool-enablement for the scripted tools. Replacing it wholesale with a MagicMock
    # breaks tuple-returning methods (an auto-mock unpacks as empty).
    enabled = {t.name for t in tools} | {t.name for t in orch.tools.list_tools()}
    tool_svc = getattr(orch, "_tool_service", None)
    if tool_svc is None:
        tool_svc = MagicMock()
        tool_svc.parse_and_validate_tool_calls = lambda tc, fc, _adapter: (tc, fc)
        orch._tool_service = tool_svc
    tool_svc.is_tool_enabled = lambda name: name in enabled

    tool_defs = [
        ToolDefinition(name=t.name, description=t.description, parameters=t.parameters)
        for t in tools
    ]
    orch._select_tools = lambda *args, **kwargs: list(tool_defs)

    orch.unified_tracker.set_max_iterations(max_iterations, user_override=True)
    orch._parity_tool_log = recorder
    return orch


# ---------------------------------------------------------------------------
# Transcript capture + parity
# ---------------------------------------------------------------------------

# The live "📊 ~N tokens (est.) | 1.2s | 3.4 tok/s" cost footer is time-dependent; strip it so
# transcripts are deterministic across runs and across the two loop implementations.
_FOOTER_RE = re.compile(r"📊|tokens \(est\.\)|tok/s")


@dataclass
class StreamTranscript:
    """Normalized, deterministic view of one streamed turn for parity comparison."""

    content: str = ""
    tool_calls: List[str] = field(default_factory=list)
    errored: bool = False
    error: Optional[str] = None


async def capture_transcript(orch: AgentOrchestrator, message: str) -> StreamTranscript:
    """Drive the real streaming loop and normalize its output into a transcript."""
    parts: List[str] = []
    errored = False
    error: Optional[str] = None
    try:
        async for chunk in orch.stream_chat(message):
            text = getattr(chunk, "content", "") or ""
            if text and not _FOOTER_RE.search(text):
                parts.append(text)
    except Exception as exc:  # capture rather than raise — "no streaming tracebacks" is asserted
        errored = True
        error = f"{type(exc).__name__}: {exc}"
    content = re.sub(r"\s+", " ", "".join(parts)).strip()
    return StreamTranscript(
        content=content,
        tool_calls=list(getattr(orch, "_parity_tool_log", [])),
        errored=errored,
        error=error,
    )


async def capture_buffered_transcript(orch: AgentOrchestrator, message: str) -> StreamTranscript:
    """Drive the real buffered loop (``orch.chat`` -> ``AgenticLoop.run``) into a transcript.

    The buffered counterpart of :func:`capture_transcript`, normalized into the SAME
    ``StreamTranscript`` shape so run/stream parity is a direct field comparison (final content +
    executed tool sequence). A single ``CompletionResponse`` is returned rather than streamed, so
    there is no per-turn duplication to strip — the cost footer never appears.
    """
    errored = False
    error: Optional[str] = None
    content = ""
    try:
        response = await orch.chat(message)
        content = re.sub(r"\s+", " ", getattr(response, "content", "") or "").strip()
    except Exception as exc:  # capture rather than raise — "no buffered traceback" is asserted
        errored = True
        error = f"{type(exc).__name__}: {exc}"
    return StreamTranscript(
        content=content,
        tool_calls=list(getattr(orch, "_parity_tool_log", [])),
        errored=errored,
        error=error,
    )


# ---------------------------------------------------------------------------
# The QA battery (FEP-0007 S1/S2/M1/M2/C1/W1–W4/U1)
# ---------------------------------------------------------------------------


@dataclass
class Scenario:
    """One parity-battery scenario: a scripted conversation + expectations."""

    id: str
    label: str
    message: str
    turns: List[TurnScript]
    tools: List[str] = field(default_factory=list)
    failing_tools: List[str] = field(default_factory=list)
    expect_content_contains: Optional[str] = None
    expect_tools: Optional[List[str]] = None
    max_iterations: int = 8


def _read_turn(tool: str, **args: Any) -> TurnScript:
    return TurnScript(content="", tool_calls=[{"name": tool, "arguments": args}])


SCENARIOS: List[Scenario] = [
    Scenario(
        id="S1",
        label="simple Q&A, no tools",
        message="What is 6 times 7?",
        turns=[TurnScript(content="The answer is 42.")],
        expect_content_contains="42",
        expect_tools=[],
    ),
    Scenario(
        id="S2",
        label="single tool then answer",
        message="How many lines in app.py?",
        turns=[
            _read_turn("read", path="app.py"),
            TurnScript(content="app.py has 10 lines."),
        ],
        tools=["read"],
        expect_content_contains="10 lines",
        expect_tools=["read"],
    ),
    Scenario(
        id="M1",
        label="multi-tool iterative code_search→read→answer",
        message="Where is the parser defined?",
        turns=[
            _read_turn("code_search", query="parser"),
            _read_turn("read", path="parser.py"),
            TurnScript(content="The parser is defined in parser.py."),
        ],
        tools=["code_search", "read"],
        expect_content_contains="parser.py",
        expect_tools=["code_search", "read"],
    ),
    Scenario(
        id="M2",
        label="repeated-but-distinct tool calls (no spin)",
        message="List both directories.",
        turns=[
            _read_turn("ls", path="a"),
            _read_turn("ls", path="b"),
            TurnScript(content="Listed a and b."),
        ],
        tools=["ls"],
        expect_content_contains="listed",
        expect_tools=["ls", "ls"],
    ),
    Scenario(
        id="C1",
        label="complex multi-step code_search→read→read→synthesize",
        message="Explain how requests flow through the app.",
        turns=[
            _read_turn("code_search", query="request flow"),
            _read_turn("read", path="router.py"),
            _read_turn("read", path="handler.py"),
            TurnScript(content="Requests flow router.py → handler.py."),
        ],
        tools=["code_search", "read"],
        expect_content_contains="flow",
        expect_tools=["code_search", "read", "read"],
    ),
    Scenario(
        id="W1",
        label="write: read→edit→confirm",
        message="Fix the typo in config.py.",
        turns=[
            _read_turn("read", path="config.py"),
            _read_turn("edit", path="config.py"),
            TurnScript(content="Applied the edit to config.py."),
        ],
        tools=["read", "edit"],
        expect_content_contains="applied",
        expect_tools=["read", "edit"],
    ),
    Scenario(
        id="W2",
        label="write: create new file",
        message="Create a README.",
        turns=[
            _read_turn("write", path="README.md"),
            TurnScript(content="Created README.md."),
        ],
        tools=["write"],
        expect_content_contains="created",
        expect_tools=["write"],
    ),
    Scenario(
        id="W3",
        label="write: read→multi_edit→confirm",
        message="Rename the symbol everywhere.",
        turns=[
            _read_turn("read", path="mod.py"),
            _read_turn("multi_edit", path="mod.py"),
            TurnScript(content="Applied 3 edits."),
        ],
        tools=["read", "multi_edit"],
        expect_content_contains="edits",
        expect_tools=["read", "multi_edit"],
    ),
    Scenario(
        id="W4",
        label="write: read→patch→confirm",
        message="Apply the patch.",
        turns=[
            _read_turn("read", path="src.py"),
            _read_turn("patch", path="src.py"),
            TurnScript(content="Patched src.py."),
        ],
        tools=["read", "patch"],
        expect_content_contains="patched",
        expect_tools=["read", "patch"],
    ),
    Scenario(
        id="U1",
        label="edge: identical repeated tool call must terminate (spin guard)",
        message="Keep searching.",
        # Same tool + same args every turn — the loop's spin/repetition guard must stop it
        # well before max_iterations rather than hang.
        turns=[_read_turn("code_search", query="x")] * 12,
        tools=["code_search"],
        expect_tools=None,  # exact count is guard-dependent; the assertion is "it terminated"
        max_iterations=20,
    ),
    Scenario(
        id="E1",
        label="edge: garbage content handled without a streaming traceback",
        message="Say something.",
        turns=[TurnScript(content="<<<<garbage>>>>")],
        expect_tools=[],
        # The only invariant is graceful handling (no traceback); the loop returns the turn.
    ),
    Scenario(
        id="E2",
        label="edge: tool failure → loop recovers and answers",
        message="Read the file.",
        turns=[
            _read_turn("read", path="missing.py"),
            TurnScript(content="Recovered after the tool failure."),
        ],
        tools=["read"],
        failing_tools=["read"],
        expect_content_contains="recovered",
        expect_tools=["read"],
    ),
    Scenario(
        id="E4",
        label="edge: empty model turn → loop continues to a real answer",
        message="Answer me.",
        turns=[TurnScript(content=""), TurnScript(content="Finally, a real answer.")],
        expect_content_contains="real answer",
        expect_tools=[],
    ),
]


def build_for_scenario(scenario: Scenario) -> AgentOrchestrator:
    """Build a fresh orchestrator for a scenario (the single canonical streaming loop)."""
    provider = ScriptedProvider(list(scenario.turns))
    return build_streaming_orchestrator(
        provider,
        scenario.tools,
        max_iterations=scenario.max_iterations,
        failing_tool_names=scenario.failing_tools,
    )

"""FEP-0041 production wiring: the seam the Phase 3 tests missed.

The PLAN step returns a CompletionResponse (or plain string), not a dict;
the detector must extract JSON content from those shapes. The guidance
must also reach the built system prompt on tool-capable turns. These tests
pin the production chain, not the mechanism in isolation.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from victor.core.provider_messages import CompletionResponse

BATCH_PLAN = {
    "steps": [
        {"tool_name": "read", "inputs": {"file_path": "src/a.py"}},
        {
            "tool_name": "search",
            "inputs": {"query": "TODO"},
            "input_refs": {"file_path": "$step0.file_path"},
        },
    ],
    "rationale": "read then search",
    "stop_on_error": True,
}


def _make_loop() -> Any:
    from victor.framework.agentic_loop import AgenticLoop

    return AgenticLoop.__new__(AgenticLoop)


def _completion(content: str) -> CompletionResponse:
    return CompletionResponse(content=content)


class TestDetectorHandlesProductionShapes:
    def test_completion_response_bare_json_detected(self):
        loop = _make_loop()
        state: dict = {}
        loop._maybe_parse_batch_plan(state, _completion(json.dumps(BATCH_PLAN)))
        batch = state.get("_batch_plan")
        assert batch is not None
        assert len(batch.steps) == 2
        assert batch.steps[1].input_refs == {"file_path": "$step0.file_path"}

    def test_completion_response_fenced_json_detected(self):
        loop = _make_loop()
        state: dict = {}
        fenced = "```json\n" + json.dumps(BATCH_PLAN) + "\n```"
        loop._maybe_parse_batch_plan(state, _completion(fenced))
        assert state.get("_batch_plan") is not None

    def test_completion_response_prose_ignored(self):
        loop = _make_loop()
        state: dict = {}
        loop._maybe_parse_batch_plan(state, _completion("Read the file first, then search."))
        assert "_batch_plan" not in state

    def test_string_json_plan_detected(self):
        loop = _make_loop()
        state: dict = {}
        loop._maybe_parse_batch_plan(state, json.dumps(BATCH_PLAN))
        assert state.get("_batch_plan") is not None

    def test_string_prose_ignored(self):
        loop = _make_loop()
        state: dict = {}
        loop._maybe_parse_batch_plan(state, "just a plain plan narrative")
        assert "_batch_plan" not in state

    def test_object_without_content_ignored(self):
        loop = _make_loop()
        state: dict = {}
        loop._maybe_parse_batch_plan(state, SimpleNamespace(model="m1"))
        assert "_batch_plan" not in state

    def test_json_without_steps_ignored(self):
        loop = _make_loop()
        state: dict = {}
        loop._maybe_parse_batch_plan(state, _completion(json.dumps({"action": "read"})))
        assert "_batch_plan" not in state

    def test_stale_batch_cleared_by_prose_plan(self):
        loop = _make_loop()
        state: dict = {"_batch_plan": object()}
        loop._maybe_parse_batch_plan(state, _completion("replanning as single actions"))
        assert "_batch_plan" not in state


class TestGuidanceReachesPrompt:
    def _build(self, available_tools: list[str] | None) -> str:
        from victor.agent.prompt_builder import SystemPromptBuilder

        builder = SystemPromptBuilder(
            provider_name="test",
            model="test-model",
            available_tools=available_tools,
        )
        return builder.build()

    def test_prompt_includes_guidance_when_tools_available(self):
        prompt = self._build(["read", "search", "edit"])
        assert "BATCH PLANNING (FEP-0041)" in prompt
        assert "$step0.file_path" in prompt

    def test_prompt_omits_guidance_without_tools(self):
        prompt = self._build([])
        assert "BATCH PLANNING (FEP-0041)" not in prompt

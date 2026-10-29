"""Batch execution types (FEP-0041 Phase 1)."""

from __future__ import annotations


from victor.core.batch_plan import (
    BatchPlan,
    BatchResult,
    BatchStep,
    BatchStepResult,
    parse_batch_plan,
    resolve_input_refs,
)


class TestBatchStep:
    def test_minimal_step(self):
        step = BatchStep(tool_name="read")
        assert step.tool_name == "read"
        assert step.inputs == {}
        assert step.input_refs == {}
        assert step.description == ""


class TestBatchResult:
    def test_all_succeeded_empty(self):
        result = BatchResult(plan=BatchPlan())
        assert result.all_succeeded is False  # empty = not succeeded

    def test_all_succeeded(self):
        result = BatchResult(
            plan=BatchPlan(),
            step_results=[
                BatchStepResult(0, "read", True, output="content"),
                BatchStepResult(1, "edit", True, output="edited"),
            ],
        )
        assert result.all_succeeded is True
        assert result.error is None

    def test_error_on_first_failure(self):
        result = BatchResult(
            plan=BatchPlan(),
            step_results=[
                BatchStepResult(0, "read", True, output="content"),
                BatchStepResult(1, "edit", False, error="file not found"),
                BatchStepResult(2, "test", True, output="pass"),
            ],
        )
        assert result.all_succeeded is False
        assert result.error == "file not found"


class TestResolveInputRefs:
    def test_dict_output_key_lookup(self):
        step = BatchStep(
            tool_name="edit",
            input_refs={"file_path": "$step0.file_path"},
        )
        prior = [{"file_path": "src/main.py", "content": "..."}]
        resolved = resolve_input_refs(step, prior)
        assert resolved["file_path"] == "src/main.py"

    def test_scalar_prior_whole_value(self):
        step = BatchStep(
            tool_name="write",
            input_refs={"content": "$step0"},
        )
        prior = ["the scalar output"]
        resolved = resolve_input_refs(step, prior)
        assert resolved["content"] == "the scalar output"

    def test_out_of_range_ref_skipped(self):
        step = BatchStep(
            tool_name="edit",
            input_refs={"file_path": "$step5.file_path"},
        )
        resolved = resolve_input_refs(step, [{"file_path": "x.py"}])
        assert "file_path" not in resolved  # out-of-range: left out

    def test_malformed_ref_ignored(self):
        step = BatchStep(
            tool_name="edit",
            input_refs={"bad": "not-a-ref"},
        )
        resolved = resolve_input_refs(step, [])
        assert "bad" not in resolved

    def test_static_inputs_preserved(self):
        step = BatchStep(
            tool_name="edit",
            inputs={"mode": "replace"},
            input_refs={"file_path": "$step0.file_path"},
        )
        prior = [{"file_path": "a.py"}]
        resolved = resolve_input_refs(step, prior)
        assert resolved["mode"] == "replace"
        assert resolved["file_path"] == "a.py"


class TestParseBatchPlan:
    def test_valid_batch_plan(self):
        data = {
            "steps": [
                {"tool_name": "read", "inputs": {"file_path": "a.py"}},
                {"tool_name": "edit", "input_refs": {"file_path": "$step0.file_path"}},
            ],
            "rationale": "read then edit",
        }
        plan = parse_batch_plan(data)
        assert plan is not None
        assert len(plan.steps) == 2
        assert plan.steps[0].tool_name == "read"
        assert plan.steps[1].input_refs == {"file_path": "$step0.file_path"}
        assert plan.rationale == "read then edit"
        assert plan.stop_on_error is True

    def test_no_steps_returns_none(self):
        assert parse_batch_plan({"steps": []}) is None
        assert parse_batch_plan({}) is None
        assert parse_batch_plan({"steps": "not a list"}) is None

    def test_step_without_tool_name_skipped(self):
        data = {"steps": [{"inputs": {}}, {"tool_name": "read"}]}
        plan = parse_batch_plan(data)
        assert plan is not None
        assert len(plan.steps) == 1

    def test_tool_alias(self):
        """Some LLMs may emit 'tool' instead of 'tool_name'."""
        data = {"steps": [{"tool": "read"}]}
        plan = parse_batch_plan(data)
        assert plan is not None
        assert plan.steps[0].tool_name == "read"

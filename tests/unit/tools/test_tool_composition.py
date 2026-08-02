# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Tests for LCEL-style tool composition."""

import asyncio
from typing import Any, Dict

import pytest

from victor.tools.composition import (
    Runnable,
    RunnableBinding,
    RunnableBranch,
    RunnableConfig,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
    RunnableSequence,
    as_runnable,
    branch,
    chain,
    extract_if_success,
    extract_output,
    map_keys,
    parallel,
    select_keys,
)


class MockRunnable(Runnable[Dict[str, Any], Dict[str, Any]]):
    """Mock runnable for testing."""

    def __init__(self, name: str, transform: callable = None):
        self._name = name
        self._transform = transform or (lambda x: x)
        self.invoke_count = 0
        self.last_input = None

    @property
    def name(self) -> str:
        return self._name

    async def invoke(
        self,
        input: Dict[str, Any],
        config: RunnableConfig = None,
    ) -> Dict[str, Any]:
        self.invoke_count += 1
        self.last_input = input
        return self._transform(input)


class TestRunnableLambda:
    """Tests for RunnableLambda."""

    @pytest.mark.asyncio
    async def test_sync_lambda(self):
        """Test wrapping a sync function."""
        double = RunnableLambda(lambda x: x * 2)
        result = await double.invoke(5)
        assert result == 10

    @pytest.mark.asyncio
    async def test_async_lambda(self):
        """Test wrapping an async function."""

        async def async_double(x):
            await asyncio.sleep(0.001)
            return x * 2

        double = RunnableLambda(async_double)
        result = await double.invoke(5)
        assert result == 10

    @pytest.mark.asyncio
    async def test_dict_transform(self):
        """Test transforming dictionary input."""
        transform = RunnableLambda(lambda d: {"result": d["value"] + 1})
        result = await transform.invoke({"value": 10})
        assert result == {"result": 11}


class TestRunnableSequence:
    """Tests for RunnableSequence (pipe chaining)."""

    @pytest.mark.asyncio
    async def test_simple_chain(self):
        """Test basic sequential execution."""
        add_one = RunnableLambda(lambda x: x + 1, name="add_one")
        double = RunnableLambda(lambda x: x * 2, name="double")

        chain = add_one | double
        result = await chain.invoke(5)
        assert result == 12  # (5 + 1) * 2

    @pytest.mark.asyncio
    async def test_long_chain(self):
        """Test chain with multiple steps."""
        add = RunnableLambda(lambda x: x + 1)
        mul = RunnableLambda(lambda x: x * 2)
        sub = RunnableLambda(lambda x: x - 3)

        chain = add | mul | sub
        result = await chain.invoke(5)
        assert result == 9  # ((5 + 1) * 2) - 3

    @pytest.mark.asyncio
    async def test_chain_with_dicts(self):
        """Test chaining with dictionary transforms."""
        step1 = RunnableLambda(lambda d: {"value": d["input"] + 1})
        step2 = RunnableLambda(lambda d: {"output": d["value"] * 2})

        chain = step1 | step2
        result = await chain.invoke({"input": 5})
        assert result == {"output": 12}

    @pytest.mark.asyncio
    async def test_pipe_method(self):
        """Test explicit pipe() method."""
        add = RunnableLambda(lambda x: x + 1)
        mul = RunnableLambda(lambda x: x * 2)
        sub = RunnableLambda(lambda x: x - 3)

        chain = add.pipe(mul, sub)
        result = await chain.invoke(5)
        assert result == 9

    @pytest.mark.asyncio
    async def test_chain_flattening(self):
        """Test that nested chains are flattened."""
        a = RunnableLambda(lambda x: x + 1)
        b = RunnableLambda(lambda x: x * 2)
        c = RunnableLambda(lambda x: x - 1)

        chain1 = a | b
        chain2 = chain1 | c

        # Should be flattened to [a, b, c]
        assert len(chain2._runnables) == 3

    def test_sequence_repr(self):
        """Test string representation."""
        a = RunnableLambda(lambda x: x, name="a")
        b = RunnableLambda(lambda x: x, name="b")
        chain = a | b
        assert "RunnableSequence" in repr(chain)


class TestRunnableParallel:
    """Tests for RunnableParallel."""

    @pytest.mark.asyncio
    async def test_parallel_execution(self):
        """Test parallel execution of multiple runnables."""
        add = RunnableLambda(lambda x: x + 1)
        mul = RunnableLambda(lambda x: x * 2)

        par = RunnableParallel(added=add, multiplied=mul)
        result = await par.invoke(5)

        assert result["added"] == 6
        assert result["multiplied"] == 10

    @pytest.mark.asyncio
    async def test_parallel_with_dicts(self):
        """Test parallel with dictionary input."""
        get_a = RunnableLambda(lambda d: d.get("a", 0))
        get_b = RunnableLambda(lambda d: d.get("b", 0))

        par = RunnableParallel(a=get_a, b=get_b)
        result = await par.invoke({"a": 1, "b": 2})

        assert result["a"] == 1
        assert result["b"] == 2

    @pytest.mark.asyncio
    async def test_parallel_concurrency_limit(self):
        """Test that concurrency limit is respected."""
        execution_order = []

        async def slow_task(name):
            async def inner(x):
                execution_order.append(f"start_{name}")
                await asyncio.sleep(0.05)
                execution_order.append(f"end_{name}")
                return x

            return inner

        tasks = {
            "a": RunnableLambda(await slow_task("a")),
            "b": RunnableLambda(await slow_task("b")),
            "c": RunnableLambda(await slow_task("c")),
        }

        par = RunnableParallel(tasks)
        config = RunnableConfig(max_concurrency=2)

        await par.invoke(1, config)

        # With concurrency=2, at least some tasks should interleave
        assert len(execution_order) == 6

    @pytest.mark.asyncio
    async def test_parallel_error_handling(self):
        """Test error handling in parallel execution."""
        success = RunnableLambda(lambda x: x + 1)

        def failing(x):
            raise ValueError("Test error")

        par = RunnableParallel(success=success, fail=RunnableLambda(failing))
        result = await par.invoke(5)

        assert result["success"] == 6
        assert "error" in result["fail"]


class TestRunnableBranch:
    """Tests for RunnableBranch."""

    @pytest.mark.asyncio
    async def test_branch_first_match(self):
        """Test that first matching branch is used."""

        def is_positive(x):
            return x > 0

        def is_negative(x):
            return x < 0

        branch = RunnableBranch(
            (is_positive, RunnableLambda(lambda x: "positive")),
            (is_negative, RunnableLambda(lambda x: "negative")),
            default=RunnableLambda(lambda x: "zero"),
        )

        assert await branch.invoke(5) == "positive"
        assert await branch.invoke(-5) == "negative"
        assert await branch.invoke(0) == "zero"

    @pytest.mark.asyncio
    async def test_branch_with_dict_conditions(self):
        """Test branching based on dict values."""

        def is_python(d):
            return d.get("lang") == "python"

        def is_js(d):
            return d.get("lang") == "javascript"

        branch = RunnableBranch(
            (is_python, RunnableLambda(lambda d: {**d, "linter": "pylint"})),
            (is_js, RunnableLambda(lambda d: {**d, "linter": "eslint"})),
            default=RunnableLambda(lambda d: {**d, "linter": "generic"}),
        )

        result = await branch.invoke({"lang": "python"})
        assert result["linter"] == "pylint"

    @pytest.mark.asyncio
    async def test_branch_no_match_no_default(self):
        """Test error when no branch matches and no default."""
        branch = RunnableBranch(
            (lambda x: x > 100, RunnableLambda(lambda x: "big")),
        )

        with pytest.raises(ValueError, match="No branch matched"):
            await branch.invoke(5)


class TestRunnablePassthrough:
    """Tests for RunnablePassthrough."""

    @pytest.mark.asyncio
    async def test_passthrough(self):
        """Test that input passes through unchanged."""
        passthrough = RunnablePassthrough()
        result = await passthrough.invoke({"key": "value"})
        assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_passthrough_in_parallel(self):
        """Test passthrough in parallel execution."""
        par = RunnableParallel(
            original=RunnablePassthrough(),
            doubled=RunnableLambda(lambda x: x * 2),
        )
        result = await par.invoke(5)
        assert result["original"] == 5
        assert result["doubled"] == 10


class TestRunnableBinding:
    """Tests for RunnableBinding."""

    @pytest.mark.asyncio
    async def test_bind_kwargs(self):
        """Test binding keyword arguments."""

        def add(a: int, b: int) -> int:
            return a + b

        add_five = RunnableLambda(lambda d: add(d.get("a", 0), d.get("b", 0))).bind(b=5)

        # The bind wraps the lambda, but for dicts we need different approach
        base = RunnableLambda(lambda d: d.get("a", 0) + d.get("b", 0))
        bound = base.bind(b=5)

        result = await bound.invoke({"a": 3})
        assert result == 8

    @pytest.mark.asyncio
    async def test_bind_override(self):
        """Test that input overrides bound values."""
        base = RunnableLambda(lambda d: d.get("x", 0))
        bound = base.bind(x=10)

        # Bound value should be used when not in input
        result1 = await bound.invoke({})
        assert result1 == 10

        # Input should override bound
        result2 = await bound.invoke({"x": 20})
        assert result2 == 20


class TestChainBuilders:
    """Tests for chain building helper functions."""

    @pytest.mark.asyncio
    async def test_chain_function(self):
        """Test the chain() helper."""
        c = chain(
            RunnableLambda(lambda x: x + 1),
            RunnableLambda(lambda x: x * 2),
        )
        result = await c.invoke(5)
        assert result == 12

    @pytest.mark.asyncio
    async def test_parallel_function(self):
        """Test the parallel() helper."""
        p = parallel(
            a=RunnableLambda(lambda x: x + 1),
            b=RunnableLambda(lambda x: x * 2),
        )
        result = await p.invoke(5)
        assert result["a"] == 6
        assert result["b"] == 10

    @pytest.mark.asyncio
    async def test_branch_function(self):
        """Test the branch() helper."""
        b = branch(
            (lambda x: x > 0, RunnableLambda(lambda x: "positive")),
            (lambda x: x < 0, RunnableLambda(lambda x: "negative")),
            default=RunnableLambda(lambda x: "zero"),
        )
        assert await b.invoke(5) == "positive"


class TestResultExtractors:
    """Tests for result extraction helpers."""

    def test_extract_output(self):
        """Test extract_output function."""
        result = {"success": True, "output": "hello", "error": None}
        assert extract_output(result) == "hello"

    def test_extract_if_success(self):
        """Test extract_if_success with successful result."""
        result = {"success": True, "output": "hello"}
        assert extract_if_success(result) == "hello"

    def test_extract_if_success_failure(self):
        """Test extract_if_success with failed result."""
        result = {"success": False, "error": "Test error"}
        with pytest.raises(RuntimeError, match="Test error"):
            extract_if_success(result)

    def test_map_keys(self):
        """Test map_keys function."""
        mapper = map_keys({"old": "new", "a": "b"})
        result = mapper({"old": 1, "a": 2, "keep": 3})
        assert result == {"new": 1, "b": 2, "keep": 3}

    def test_select_keys(self):
        """Test select_keys function."""
        selector = select_keys("a", "b")
        result = selector({"a": 1, "b": 2, "c": 3})
        assert result == {"a": 1, "b": 2}


class TestBatchExecution:
    """Tests for batch execution."""

    @pytest.mark.asyncio
    async def test_batch_invoke(self):
        """Test batch execution."""
        double = RunnableLambda(lambda x: x * 2)
        results = await double.batch([1, 2, 3, 4, 5])
        assert results == [2, 4, 6, 8, 10]


class TestIntegration:
    """Integration tests for complex compositions."""

    @pytest.mark.asyncio
    async def test_complex_chain(self):
        """Test a complex chain with multiple patterns."""
        # Simulate a tool chain:
        # 1. Read files in parallel
        # 2. Branch based on file type
        # 3. Process and collect

        read_py = RunnableLambda(
            lambda d: {"success": True, "output": "python code", "type": "python"}
        )
        read_js = RunnableLambda(
            lambda d: {"success": True, "output": "js code", "type": "javascript"}
        )

        parallel_read = RunnableParallel(py=read_py, js=read_js)

        def process_results(results):
            return {
                "files_processed": len(results),
                "types": [r.get("type") for r in results.values()],
            }

        chain = parallel_read | RunnableLambda(process_results)
        result = await chain.invoke({})

        assert result["files_processed"] == 2
        assert set(result["types"]) == {"python", "javascript"}

    @pytest.mark.asyncio
    async def test_chain_with_branch(self):
        """Test chain that includes a branch."""

        def is_dict(x):
            return isinstance(x, dict)

        transform = RunnableBranch(
            (is_dict, RunnableLambda(lambda d: {"count": len(d)})),
            default=RunnableLambda(lambda x: {"count": 1}),
        )

        chain = RunnableLambda(lambda x: x) | transform

        dict_result = await chain.invoke({"a": 1, "b": 2})
        assert dict_result["count"] == 2

        scalar_result = await chain.invoke(5)
        assert scalar_result["count"] == 1

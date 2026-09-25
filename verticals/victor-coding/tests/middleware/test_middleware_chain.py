"""Unit tests for victor.agent.middleware_chain module.

Tests the middleware chain pattern for tool execution processing.
"""

from typing import Any, Dict, Optional

import pytest

from victor.framework.extensions import (
    MiddlewareChain,
    MiddlewareAbortError,
)
from victor.core.verticals.protocols import (
    MiddlewarePriority,
    MiddlewareResult,
)


class TestMiddlewareChain:
    """Tests for MiddlewareChain class."""

    def test_empty_chain(self):
        """Empty chain should pass through without modification."""
        chain = MiddlewareChain()
        assert len(chain._middleware) == 0

    @pytest.mark.asyncio
    async def test_empty_chain_process_before(self):
        """Empty chain should return proceed=True."""
        chain = MiddlewareChain()
        result = await chain.process_before("test_tool", {"arg": "value"})
        assert result.proceed is True
        assert result.modified_arguments is None

    @pytest.mark.asyncio
    async def test_empty_chain_process_after(self):
        """Empty chain should return original result."""
        chain = MiddlewareChain()
        original_result = {"output": "test"}
        result = await chain.process_after(
            "test_tool", {"arg": "value"}, original_result, success=True
        )
        assert result == original_result

    def test_add_middleware(self):
        """Adding middleware should increase chain size."""

        class TestMiddleware:
            async def before_tool_call(
                self, tool_name: str, arguments: Dict[str, Any]
            ) -> MiddlewareResult:
                return MiddlewareResult()

            def get_priority(self) -> MiddlewarePriority:
                return MiddlewarePriority.NORMAL

        chain = MiddlewareChain()
        chain.add(TestMiddleware())
        assert len(chain._middleware) == 1

    def test_priority_sorting(self):
        """Middleware should be sorted by priority (lowest first)."""

        class HighPriority:
            async def before_tool_call(
                self, tool_name: str, arguments: Dict[str, Any]
            ) -> MiddlewareResult:
                return MiddlewareResult()

            def get_priority(self) -> MiddlewarePriority:
                return MiddlewarePriority.HIGH

            def get_applicable_tools(self):
                return None

        class LowPriority:
            async def before_tool_call(
                self, tool_name: str, arguments: Dict[str, Any]
            ) -> MiddlewareResult:
                return MiddlewareResult()

            def get_priority(self) -> MiddlewarePriority:
                return MiddlewarePriority.LOW

            def get_applicable_tools(self):
                return None

        class CriticalPriority:
            async def before_tool_call(
                self, tool_name: str, arguments: Dict[str, Any]
            ) -> MiddlewareResult:
                return MiddlewareResult()

            def get_priority(self) -> MiddlewarePriority:
                return MiddlewarePriority.CRITICAL

            def get_applicable_tools(self):
                return None

        chain = MiddlewareChain()
        # Add in random order
        chain.add(LowPriority())
        chain.add(CriticalPriority())
        chain.add(HighPriority())

        # Trigger sorting by calling _ensure_sorted
        chain._ensure_sorted()

        # Should be sorted by priority value (lowest first)
        priorities = [m.get_priority().value for m in chain._middleware]
        assert priorities == sorted(priorities)

    @pytest.mark.asyncio
    async def test_blocking_middleware(self):
        """Middleware can block tool execution."""

        class BlockingMiddleware:
            async def before_tool_call(
                self, tool_name: str, arguments: Dict[str, Any]
            ) -> MiddlewareResult:
                return MiddlewareResult(
                    proceed=False,
                    error_message="Blocked by test",
                )

            def get_priority(self) -> MiddlewarePriority:
                return MiddlewarePriority.NORMAL

            def get_applicable_tools(self):
                return None

        chain = MiddlewareChain()
        chain.add(BlockingMiddleware())

        result = await chain.process_before("test_tool", {})
        assert result.proceed is False
        assert result.error_message == "Blocked by test"

    @pytest.mark.asyncio
    async def test_argument_modification(self):
        """Middleware can modify arguments."""

        class ModifyingMiddleware:
            async def before_tool_call(
                self, tool_name: str, arguments: Dict[str, Any]
            ) -> MiddlewareResult:
                return MiddlewareResult(
                    proceed=True,
                    modified_arguments={"path": "/modified/path"},
                )

            def get_priority(self) -> MiddlewarePriority:
                return MiddlewarePriority.NORMAL

            def get_applicable_tools(self):
                return None

        chain = MiddlewareChain()
        chain.add(ModifyingMiddleware())

        result = await chain.process_before("test_tool", {"path": "/original"})
        assert result.proceed is True
        assert result.modified_arguments["path"] == "/modified/path"

    @pytest.mark.asyncio
    async def test_middleware_chain_order(self):
        """Middleware should execute in priority order for before_tool_call."""
        execution_order = []

        class FirstMiddleware:
            async def before_tool_call(
                self, tool_name: str, arguments: Dict[str, Any]
            ) -> MiddlewareResult:
                execution_order.append("first")
                return MiddlewareResult()

            def get_priority(self) -> MiddlewarePriority:
                return MiddlewarePriority.HIGH  # value=25

            def get_applicable_tools(self):
                return None

        class SecondMiddleware:
            async def before_tool_call(
                self, tool_name: str, arguments: Dict[str, Any]
            ) -> MiddlewareResult:
                execution_order.append("second")
                return MiddlewareResult()

            def get_priority(self) -> MiddlewarePriority:
                return MiddlewarePriority.LOW  # value=75

            def get_applicable_tools(self):
                return None

        chain = MiddlewareChain()
        chain.add(SecondMiddleware())  # Add in reverse order
        chain.add(FirstMiddleware())

        await chain.process_before("test", {})

        # HIGH priority (25) should run before LOW priority (75)
        assert execution_order == ["first", "second"]

    @pytest.mark.asyncio
    async def test_after_chain_reverse_order(self):
        """after_tool_call should execute in reverse priority order."""
        execution_order = []

        class FirstMiddleware:
            async def before_tool_call(
                self, tool_name: str, arguments: Dict[str, Any]
            ) -> MiddlewareResult:
                return MiddlewareResult()

            async def after_tool_call(
                self,
                tool_name: str,
                arguments: Dict[str, Any],
                result: Any,
                success: bool,
            ) -> Optional[Any]:
                execution_order.append("first")
                return None

            def get_priority(self) -> MiddlewarePriority:
                return MiddlewarePriority.HIGH

            def get_applicable_tools(self):
                return None

        class SecondMiddleware:
            async def before_tool_call(
                self, tool_name: str, arguments: Dict[str, Any]
            ) -> MiddlewareResult:
                return MiddlewareResult()

            async def after_tool_call(
                self,
                tool_name: str,
                arguments: Dict[str, Any],
                result: Any,
                success: bool,
            ) -> Optional[Any]:
                execution_order.append("second")
                return None

            def get_priority(self) -> MiddlewarePriority:
                return MiddlewarePriority.LOW

            def get_applicable_tools(self):
                return None

        chain = MiddlewareChain()
        chain.add(FirstMiddleware())
        chain.add(SecondMiddleware())

        await chain.process_after("test", {}, "result", True)

        # Reverse order: LOW runs before HIGH in after
        assert execution_order == ["second", "first"]

    @pytest.mark.asyncio
    async def test_applicable_tools_filter(self):
        """Middleware with applicable_tools should only run for those tools."""
        executed_for = []

        class FilteredMiddleware:
            async def before_tool_call(
                self, tool_name: str, arguments: Dict[str, Any]
            ) -> MiddlewareResult:
                executed_for.append(tool_name)
                return MiddlewareResult()

            def get_priority(self) -> MiddlewarePriority:
                return MiddlewarePriority.NORMAL

            def get_applicable_tools(self):
                return {"write_file", "edit_files"}

        chain = MiddlewareChain()
        chain.add(FilteredMiddleware())

        await chain.process_before("write_file", {})  # Should run
        await chain.process_before("read_file", {})  # Should NOT run
        await chain.process_before("edit_files", {})  # Should run

        assert executed_for == ["write_file", "edit_files"]

    @pytest.mark.asyncio
    async def test_result_modification_in_after(self):
        """after_tool_call can modify the result."""

        class ResultModifier:
            async def before_tool_call(
                self, tool_name: str, arguments: Dict[str, Any]
            ) -> MiddlewareResult:
                return MiddlewareResult()

            async def after_tool_call(
                self,
                tool_name: str,
                arguments: Dict[str, Any],
                result: Any,
                success: bool,
            ) -> Optional[Any]:
                return {"modified": True, "original": result}

            def get_priority(self) -> MiddlewarePriority:
                return MiddlewarePriority.NORMAL

            def get_applicable_tools(self):
                return None

        chain = MiddlewareChain()
        chain.add(ResultModifier())

        result = await chain.process_after("test", {}, "original", True)
        assert result["modified"] is True
        assert result["original"] == "original"

    @pytest.mark.asyncio
    async def test_early_termination_on_block(self):
        """Chain should stop processing when middleware blocks."""
        second_executed = False

        class BlockingMiddleware:
            async def before_tool_call(
                self, tool_name: str, arguments: Dict[str, Any]
            ) -> MiddlewareResult:
                return MiddlewareResult(proceed=False)

            def get_priority(self) -> MiddlewarePriority:
                return MiddlewarePriority.HIGH  # Runs first

            def get_applicable_tools(self):
                return None

        class SecondMiddleware:
            async def before_tool_call(
                self, tool_name: str, arguments: Dict[str, Any]
            ) -> MiddlewareResult:
                nonlocal second_executed
                second_executed = True
                return MiddlewareResult()

            def get_priority(self) -> MiddlewarePriority:
                return MiddlewarePriority.LOW  # Would run second

            def get_applicable_tools(self):
                return None

        chain = MiddlewareChain()
        chain.add(BlockingMiddleware())
        chain.add(SecondMiddleware())

        result = await chain.process_before("test", {})
        assert result.proceed is False
        assert second_executed is False  # Should not have run

    @pytest.mark.asyncio
    async def test_exception_handling(self):
        """A broken configured middleware blocks execution through the public API."""

        class FailingMiddleware:
            async def before_tool_call(
                self, tool_name: str, arguments: Dict[str, Any]
            ) -> MiddlewareResult:
                raise ValueError("Test error")

            def get_priority(self) -> MiddlewarePriority:
                return MiddlewarePriority.NORMAL

            def get_applicable_tools(self):
                return None

        chain = MiddlewareChain()
        chain.add(FailingMiddleware())

        result = await chain.process_before("test", {})
        assert result.proceed is False
        assert result.metadata == {"enforcement_error": True}
        assert result.error_message == "Middleware enforcement failed."

    def test_clear(self):
        """clear() should remove all middleware."""

        class TestMiddleware:
            async def before_tool_call(
                self, tool_name: str, arguments: Dict[str, Any]
            ) -> MiddlewareResult:
                return MiddlewareResult()

            def get_priority(self) -> MiddlewarePriority:
                return MiddlewarePriority.NORMAL

        chain = MiddlewareChain()
        chain.add(TestMiddleware())
        chain.add(TestMiddleware())
        assert len(chain._middleware) == 2

        chain.clear()
        assert len(chain._middleware) == 0


class TestMiddlewareAbortError:
    """Tests for MiddlewareAbortError exception."""

    def test_error_creation(self):
        """MiddlewareAbortError can be created with tool_name and message."""
        error = MiddlewareAbortError(tool_name="test_tool", message="Operation aborted")
        assert error.tool_name == "test_tool"
        assert error.message == "Operation aborted"
        assert "test_tool" in str(error)
        assert "Operation aborted" in str(error)

    def test_error_attributes(self):
        """MiddlewareAbortError stores tool_name and message."""
        error = MiddlewareAbortError(tool_name="write_file", message="Blocked by safety")
        assert error.tool_name == "write_file"
        assert error.message == "Blocked by safety"


class TestMiddlewareChainAdvanced:
    """Advanced tests for MiddlewareChain functionality."""

    def test_len_and_bool(self):
        """Test __len__ and __bool__ methods."""

        class SimpleMiddleware:
            async def before_tool_call(
                self, tool_name: str, arguments: Dict[str, Any]
            ) -> MiddlewareResult:
                return MiddlewareResult()

            def get_priority(self) -> MiddlewarePriority:
                return MiddlewarePriority.NORMAL

            def get_applicable_tools(self):
                return None

        chain = MiddlewareChain()
        assert len(chain) == 0
        assert bool(chain) is False

        chain.add(SimpleMiddleware())
        assert len(chain) == 1
        assert bool(chain) is True

        chain.add(SimpleMiddleware())
        assert len(chain) == 2

    def test_get_middleware_info(self):
        """Test get_middleware_info returns correct structure."""

        class TestMiddleware:
            async def before_tool_call(
                self, tool_name: str, arguments: Dict[str, Any]
            ) -> MiddlewareResult:
                return MiddlewareResult()

            def get_priority(self) -> MiddlewarePriority:
                return MiddlewarePriority.HIGH

            def get_applicable_tools(self):
                return {"write_file", "edit_files"}

        chain = MiddlewareChain()
        chain.add(TestMiddleware())

        info = chain.get_middleware_info()
        assert len(info) == 1
        assert info[0]["name"] == "TestMiddleware"
        assert info[0]["priority"] == "HIGH"
        assert info[0]["applicable_tools"] == {"write_file", "edit_files"}

    @pytest.mark.asyncio
    async def test_process_tool_call_success(self):
        """Test process_tool_call with successful execution."""

        class PassthroughMiddleware:
            async def before_tool_call(
                self, tool_name: str, arguments: Dict[str, Any]
            ) -> MiddlewareResult:
                return MiddlewareResult()

            async def after_tool_call(
                self,
                tool_name: str,
                arguments: Dict[str, Any],
                result: Any,
                success: bool,
            ) -> Optional[Any]:
                return None

            def get_priority(self) -> MiddlewarePriority:
                return MiddlewarePriority.NORMAL

            def get_applicable_tools(self):
                return None

        chain = MiddlewareChain()
        chain.add(PassthroughMiddleware())

        async def mock_executor(**kwargs):
            return {"success": True, "data": kwargs}

        result = await chain.process_tool_call(
            tool_name="test_tool",
            arguments={"arg1": "value1"},
            executor=mock_executor,
        )

        assert result["success"] is True
        assert result["data"]["arg1"] == "value1"

    @pytest.mark.asyncio
    async def test_process_tool_call_blocking(self):
        """Test process_tool_call raises MiddlewareAbortError when blocked."""

        class BlockingMiddleware:
            async def before_tool_call(
                self, tool_name: str, arguments: Dict[str, Any]
            ) -> MiddlewareResult:
                return MiddlewareResult(
                    proceed=False,
                    error_message="Blocked for testing",
                )

            def get_priority(self) -> MiddlewarePriority:
                return MiddlewarePriority.NORMAL

            def get_applicable_tools(self):
                return None

        chain = MiddlewareChain()
        chain.add(BlockingMiddleware())

        async def mock_executor(**kwargs):
            return {"success": True}

        with pytest.raises(MiddlewareAbortError) as exc_info:
            await chain.process_tool_call(
                tool_name="blocked_tool",
                arguments={},
                executor=mock_executor,
            )

        assert exc_info.value.tool_name == "blocked_tool"
        assert "Blocked for testing" in exc_info.value.message

    @pytest.mark.asyncio
    async def test_process_tool_call_with_modification(self):
        """Test process_tool_call uses modified arguments."""

        class ModifyingMiddleware:
            async def before_tool_call(
                self, tool_name: str, arguments: Dict[str, Any]
            ) -> MiddlewareResult:
                return MiddlewareResult(
                    proceed=True,
                    modified_arguments={"path": "/modified/path"},
                )

            async def after_tool_call(
                self,
                tool_name: str,
                arguments: Dict[str, Any],
                result: Any,
                success: bool,
            ) -> Optional[Any]:
                return None

            def get_priority(self) -> MiddlewarePriority:
                return MiddlewarePriority.NORMAL

            def get_applicable_tools(self):
                return None

        chain = MiddlewareChain()
        chain.add(ModifyingMiddleware())

        async def mock_executor(**kwargs):
            return {"path": kwargs.get("path")}

        result = await chain.process_tool_call(
            tool_name="test_tool",
            arguments={"path": "/original"},
            executor=mock_executor,
        )

        assert result["path"] == "/modified/path"

    @pytest.mark.asyncio
    async def test_process_tool_call_executor_error(self):
        """Test process_tool_call handles executor errors."""

        class PassthroughMiddleware:
            async def before_tool_call(
                self, tool_name: str, arguments: Dict[str, Any]
            ) -> MiddlewareResult:
                return MiddlewareResult()

            async def after_tool_call(
                self,
                tool_name: str,
                arguments: Dict[str, Any],
                result: Any,
                success: bool,
            ) -> Optional[Any]:
                # Verify after_tool_call is called even on error
                assert success is False
                return None

            def get_priority(self) -> MiddlewarePriority:
                return MiddlewarePriority.NORMAL

            def get_applicable_tools(self):
                return None

        chain = MiddlewareChain()
        chain.add(PassthroughMiddleware())

        async def failing_executor(**kwargs):
            raise ValueError("Executor failed")

        with pytest.raises(ValueError, match="Executor failed"):
            await chain.process_tool_call(
                tool_name="test_tool",
                arguments={},
                executor=failing_executor,
            )

    def test_remove_middleware(self):
        """Test removing middleware from chain."""

        class TestMiddleware:
            async def before_tool_call(
                self, tool_name: str, arguments: Dict[str, Any]
            ) -> MiddlewareResult:
                return MiddlewareResult()

            def get_priority(self) -> MiddlewarePriority:
                return MiddlewarePriority.NORMAL

            def get_applicable_tools(self):
                return None

        chain = MiddlewareChain()
        mw = TestMiddleware()
        chain.add(mw)
        assert len(chain) == 1

        result = chain.remove(mw)
        assert result is True
        assert len(chain) == 0

        # Removing again should return False
        result = chain.remove(mw)
        assert result is False

    def test_enabled_property(self):
        """Test enabled property getter and setter."""
        chain = MiddlewareChain()

        # Default is enabled
        assert chain.enabled is True

        chain.enabled = False
        assert chain.enabled is False

        chain.enabled = True
        assert chain.enabled is True


class TestCreateMiddlewareChainFactory:
    """Tests for create_middleware_chain factory function."""

    def test_create_empty_chain(self):
        """Create chain without middleware."""
        from victor.framework.extensions import create_middleware_chain

        chain = create_middleware_chain()
        assert len(chain) == 0

    def test_create_chain_with_middleware(self):
        """Create chain with initial middleware list."""
        from victor.framework.extensions import create_middleware_chain

        class TestMiddleware:
            async def before_tool_call(
                self, tool_name: str, arguments: Dict[str, Any]
            ) -> MiddlewareResult:
                return MiddlewareResult()

            def get_priority(self) -> MiddlewarePriority:
                return MiddlewarePriority.NORMAL

            def get_applicable_tools(self):
                return None

        middleware_list = [TestMiddleware(), TestMiddleware()]
        chain = create_middleware_chain(middleware_list)

        assert len(chain) == 2


@pytest.mark.skip(reason="Vertical middleware classes moved to external packages")
class TestCodingMiddlewareIntegration:
    """Integration tests with actual CodingMiddleware."""

    @pytest.mark.asyncio
    async def test_coding_middleware_in_chain(self):
        """CodingMiddleware should work in MiddlewareChain."""
        try:
            from victor_coding.middleware import (
                CodeCorrectionMiddleware,
                GitSafetyMiddleware,
            )
        except ImportError:
            pytest.skip("victor-coding package not installed")

        chain = MiddlewareChain()
        chain.add(CodeCorrectionMiddleware())
        chain.add(GitSafetyMiddleware())

        # Should have 2 middleware
        assert len(chain._middleware) == 2

        # Trigger sorting
        chain._ensure_sorted()

        # GitSafetyMiddleware has CRITICAL priority (0)
        # CodeCorrectionMiddleware has HIGH priority (25)
        # After sorting, CRITICAL should be first (lower value first)
        priorities = [m.get_priority() for m in chain._middleware]
        assert priorities[0] == MiddlewarePriority.CRITICAL
        assert priorities[1] == MiddlewarePriority.HIGH

    @pytest.mark.asyncio
    async def test_git_safety_blocks_dangerous_commands(self):
        """GitSafetyMiddleware should block dangerous git operations when configured."""
        try:
            from victor_coding.middleware import GitSafetyMiddleware
        except ImportError:
            pytest.skip("victor-coding package not installed")

        chain = MiddlewareChain()
        # Need to enable block_dangerous to actually block
        chain.add(GitSafetyMiddleware(block_dangerous=True))

        # Test blocked command
        result = await chain.process_before(
            "execute_bash",
            {"command": "git push --force origin main"},
        )
        assert result.proceed is False
        assert (
            "dangerous" in result.error_message.lower() or "block" in result.error_message.lower()
        )

        # Test allowed command
        result = await chain.process_before(
            "execute_bash",
            {"command": "git status"},
        )
        assert result.proceed is True

"""End-to-end test for tool output preview."""

import pytest
from unittest.mock import patch

from victor.config.tool_settings import ToolSettings
from victor.ui.output_formatter import OutputFormatter
from victor.ui.rendering.live_renderer import LiveDisplayRenderer
from rich.console import Console


@pytest.mark.asyncio
async def test_preview_in_interactive_mode():
    """Test that preview is shown in interactive mode."""
    console = Console()
    renderer = LiveDisplayRenderer(console)
    renderer.start()

    # Simulate tool execution result
    original_result = "\n".join([f"Match {i}: found TODO" for i in range(20)])

    renderer.on_tool_result(
        name="grep",
        success=True,
        elapsed=0.5,
        arguments={"pattern": "TODO", "path": "src/"},
        original_result=original_result,
        preview_lines=3,
        was_pruned=False,
    )

    # Verify state is stored for expansion
    assert renderer._last_tool_result is not None
    assert renderer._last_tool_result["name"] == "grep"
    assert renderer._last_tool_result["success"] is True
    assert renderer._last_tool_result["result"] == original_result

    renderer.cleanup()


@pytest.mark.asyncio
async def test_pruning_transparency_message():
    """Test that pruning status is shown to user."""
    console = Console()
    renderer = LiveDisplayRenderer(console)
    renderer.start()

    # Enable transparency
    with patch("victor.config.tool_settings.get_tool_settings") as mock_settings:
        mock_settings.return_value = ToolSettings(
            tool_output_preview_enabled=True,
            tool_output_show_transparency=True,
        )

        original_result = "Large output\n" * 100

        renderer.on_tool_result(
            name="code_search",
            success=True,
            elapsed=1.2,
            arguments={"query": "TODO"},
            original_result=original_result,
            preview_lines=3,
            was_pruned=True,  # Pruning was applied
        )

        # Verify state reflects pruning
        assert renderer._last_tool_result is not None
        # The transparency message would be printed in on_tool_result

    renderer.cleanup()


@pytest.mark.asyncio
async def test_expand_hotkey_integration():
    """Test that Ctrl+O expands output via hotkey."""
    console = Console()
    renderer = LiveDisplayRenderer(console)
    renderer.start()

    # Simulate tool result
    full_output = "Line 1\nLine 2\nLine 3\n" * 20
    renderer.on_tool_result(
        name="read",
        success=True,
        elapsed=0.3,
        arguments={"path": "src/main.py"},
        original_result=full_output,
        preview_lines=3,
        was_pruned=False,
    )

    # Verify expand method works
    assert renderer._last_tool_result is not None
    # Calling expand_last_output should not raise exception
    renderer.expand_last_output()

    renderer.cleanup()


@pytest.mark.asyncio
async def test_preview_disabled_via_settings():
    """Test that preview can be disabled via settings."""
    console = Console()
    renderer = LiveDisplayRenderer(console)
    renderer.start()

    with patch("victor.config.tool_settings.get_tool_settings") as mock_settings:
        # Disable preview
        mock_settings.return_value = ToolSettings(
            tool_output_preview_enabled=False,
        )

        original_result = "Output\n" * 50

        # Should not raise exception even with preview disabled
        renderer.on_tool_result(
            name="grep",
            success=True,
            elapsed=0.5,
            arguments={"pattern": "test"},
            original_result=original_result,
            preview_lines=3,
            was_pruned=False,
        )

        # State should still be stored for potential expansion
        assert renderer._last_tool_result is not None

    renderer.cleanup()


@pytest.mark.asyncio
async def test_accuracy_first_default_behavior():
    """Test that safe-default pruning defaults are correct."""
    settings = ToolSettings()  # Use defaults

    # Verify safe-default pruning
    assert settings.tool_output_pruning_enabled is True
    assert settings.tool_output_pruning_safe_only is True
    assert settings.tool_output_preview_enabled is True  # Show preview to user


@pytest.mark.asyncio
async def test_backward_compatibility_mode():
    """Test that broader pruning mode can still be enabled explicitly."""
    console = Console()
    renderer = LiveDisplayRenderer(console)
    renderer.start()

    with patch("victor.config.tool_settings.get_tool_settings") as mock_settings:
        # Restore old behavior: cost-optimized
        mock_settings.return_value = ToolSettings(
            tool_output_pruning_enabled=True,  # Enable pruning (sends pruned to LLM)
            tool_output_pruning_safe_only=False,  # Broaden beyond safe-default scope
            tool_output_preview_enabled=False,  # No preview
            tool_output_show_transparency=False,  # No transparency
        )

        original_result = "Full output\n" * 100

        renderer.on_tool_result(
            name="grep",
            success=True,
            elapsed=0.5,
            arguments={"pattern": "test"},
            original_result=original_result,
            preview_lines=3,
            was_pruned=True,  # Pruning was applied
        )

        # Should still work without errors
        assert renderer._last_tool_result is not None

    renderer.cleanup()


@pytest.mark.asyncio
async def test_output_formatter_preview_integration():
    """Test preview integration in OutputFormatter."""
    formatter = OutputFormatter()

    original_result = "Line 1\nLine 2\nLine 3\n" * 10

    formatter.tool_result(
        tool_name="read",
        success=True,
        original_result=original_result,
        preview_lines=3,
        was_pruned=False,
        show_preview=True,
    )

    # Verify state is stored
    assert formatter._last_tool_result is not None
    assert formatter._last_tool_result["tool_name"] == "read"
    assert formatter._last_tool_result["result"] == original_result

    # Verify expand works
    formatter.expand_last_tool_output()


@pytest.mark.asyncio
async def test_preview_with_failed_tool():
    """Test that failed tools don't show preview."""
    console = Console()
    renderer = LiveDisplayRenderer(console)
    renderer.start()

    renderer.on_tool_result(
        name="read",
        success=False,  # Failed
        elapsed=0.1,
        arguments={"path": "nonexistent.py"},
        error="File not found",
        original_result=None,
        preview_lines=3,
        was_pruned=False,
    )

    # Should still store state for error cases
    assert renderer._last_tool_result is not None
    assert renderer._last_tool_result["success"] is False

    # Expand should not show anything for failed tools
    renderer.expand_last_output()  # Should handle gracefully

    renderer.cleanup()


@pytest.mark.asyncio
async def test_multiple_tool_results_state_management():
    """Test that multiple tool results are handled correctly."""
    console = Console()
    renderer = LiveDisplayRenderer(console)
    renderer.start()

    # First tool result
    renderer.on_tool_result(
        name="grep",
        success=True,
        elapsed=0.3,
        arguments={"pattern": "TODO"},
        original_result="First result\n" * 10,
        preview_lines=3,
        was_pruned=False,
    )

    first_result = renderer._last_tool_result
    assert first_result["name"] == "grep"

    # Second tool result (should replace first)
    renderer.on_tool_result(
        name="read",
        success=True,
        elapsed=0.2,
        arguments={"path": "main.py"},
        original_result="Second result\n" * 10,
        preview_lines=3,
        was_pruned=False,
    )

    # Should have second result, not first
    assert renderer._last_tool_result["name"] == "read"
    assert renderer._last_tool_result["result"] == "Second result\n" * 10

    renderer.cleanup()


@pytest.mark.asyncio
async def test_preview_with_custom_line_count():
    """Test preview with custom line count."""
    console = Console()
    renderer = LiveDisplayRenderer(console)
    renderer.start()

    original_result = "\n".join([f"Line {i}" for i in range(100)])

    # Test with 5 lines instead of default 3
    renderer.on_tool_result(
        name="list_directory",
        success=True,
        elapsed=0.4,
        arguments={"path": "src/"},
        original_result=original_result,
        preview_lines=5,  # Custom line count
        was_pruned=False,
    )

    # Verify state stored correctly
    assert renderer._last_tool_result is not None
    assert renderer._last_tool_result["result"] == original_result

    # Generate preview and verify it has 5 lines
    preview = renderer._generate_preview(original_result, num_lines=5)
    assert "Line 0" in preview
    assert "Line 4" in preview
    assert "Line 5" not in preview

    renderer.cleanup()

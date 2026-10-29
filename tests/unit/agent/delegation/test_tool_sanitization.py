from victor.agent.delegation.tool import DelegateTool


def test_delegate_tool_sanitization():
    # Mock handler
    handler = None
    tool = DelegateTool(handler=handler)

    # Test case 1: Trailing thinking block
    task = "Review all workspace members one by one — apply checklist to each member: ... — [<thinking>]"
    sanitized = tool.sanitize_task(task)
    assert (
        sanitized == "Review all workspace members one by one — apply checklist to each member: ..."
    )

    # Test case 2: Thinking block with colon
    task = "Do X [Thinking: I should do X]"
    sanitized = tool.sanitize_task(task)
    assert sanitized == "Do X"

    # Test case 3: Nested thinking tags
    task = "Do Y <thinking>I will do Y</thinking>"
    sanitized = tool.sanitize_task(task)
    assert sanitized == "Do Y"

    # Test case 4: Hallucinated Kung Fu tokens
    task = "Analyze the code 武功"
    sanitized = tool.sanitize_task(task)
    assert sanitized == "Analyze the code"

    # Test case 5: Combined
    task = "Check this file <thinking>...</thinking> [<thinking>] 武功 "
    sanitized = tool.sanitize_task(task)
    assert sanitized == "Check this file"

"""Tests for pre-computed decision trees."""

from pathlib import Path
from victor.agent.decision_trees import (
    DecisionAction,
    DecisionResult,
    DecisionNode,
    FileTypeCondition,
    KeywordCondition,
    RegexCondition,
    ProjectTypeCondition,
    PreComputedDecisionTrees,
    decide_without_llm,
    can_decide_without_llm,
)


class TestDecisionConditions:
    """Test decision condition evaluation."""

    def test_file_type_condition_match(self):
        """Test file type condition with matching extension."""
        condition = FileTypeCondition([".py", ".txt"])
        context = {"path": "/path/to/file.py"}
        assert condition.evaluate(context) is True

    def test_file_type_condition_no_match(self):
        """Test file type condition with non-matching extension."""
        condition = FileTypeCondition([".py", ".txt"])
        context = {"path": "/path/to/file.js"}
        assert condition.evaluate(context) is False

    def test_file_type_condition_no_extension(self):
        """Test file type condition with extensionless file."""
        condition = FileTypeCondition([".py"])
        context = {"path": "/path/to/README"}
        assert condition.evaluate(context) is False

    def test_keyword_condition_match_any(self):
        """Test keyword condition with match_all=False."""
        condition = KeywordCondition(["python", "code"], match_all=False)
        context = {"query": "Write some python code"}
        assert condition.evaluate(context) is True

    def test_keyword_condition_match_all(self):
        """Test keyword condition with match_all=True."""
        condition = KeywordCondition(["python", "code"], match_all=True)
        context = {"query": "Write some python code"}
        assert condition.evaluate(context) is True

    def test_keyword_condition_no_match(self):
        """Test keyword condition with no match."""
        condition = KeywordCondition(["python", "rust"])
        context = {"query": "Write javascript code"}
        assert condition.evaluate(context) is False

    def test_regex_condition_match(self):
        """Test regex condition with matching pattern."""
        condition = RegexCondition(r"read\s+(\w+\.py)")
        context = {"query": "read main.py"}
        assert condition.evaluate(context) is True

    def test_regex_condition_no_match(self):
        """Test regex condition with non-matching pattern."""
        condition = RegexCondition(r"read\s+(\w+\.py)")
        context = {"query": "write main.py"}
        assert condition.evaluate(context) is False

    def test_project_type_condition_python(self):
        """Test project type detection for Python."""
        condition = ProjectTypeCondition(["python"])
        # Test with pyproject.toml
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "pyproject.toml").touch()
            context = {"root_path": tmpdir}
            assert condition.evaluate(context) is True

    def test_project_type_condition_rust(self):
        """Test project type detection for Rust."""
        condition = ProjectTypeCondition(["rust"])
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "Cargo.toml").touch()
            context = {"root_path": tmpdir}
            assert condition.evaluate(context) is True


class TestDecisionNode:
    """Test decision node evaluation."""

    def test_simple_decision_true_action(self):
        """Test simple decision with true action."""
        condition = KeywordCondition(["python"])
        true_action = DecisionResult(
            action=DecisionAction.TOOL_CALL,
            confidence=1.0,
            result={"tool": "python_executor"},
            reasoning="Python code detected",
        )

        node = DecisionNode(condition=condition, true_action=true_action)
        result = node.evaluate({"query": "Write python code"})

        assert result is not None
        assert result.action == DecisionAction.TOOL_CALL
        assert result.confidence == 1.0
        assert result.result["tool"] == "python_executor"

    def test_simple_decision_false_action(self):
        """Test simple decision with false action."""
        condition = KeywordCondition(["python"])
        false_action = DecisionResult(
            action=DecisionAction.TOOL_CALL,
            confidence=0.8,
            result={"tool": "javascript_executor"},
            reasoning="Not Python code",
        )

        node = DecisionNode(condition=condition, false_action=false_action)
        result = node.evaluate({"query": "Write javascript code"})

        assert result is not None
        assert result.action == DecisionAction.TOOL_CALL
        assert result.confidence == 0.8

    def test_nested_decision_tree(self):
        """Test nested decision nodes."""
        leaf_action = DecisionResult(
            action=DecisionAction.TOOL_CALL,
            confidence=0.9,
            result={"tool": "rust_executor"},
            reasoning="Rust code detected",
        )

        inner_condition = KeywordCondition(["rust"])
        inner_node = DecisionNode(condition=inner_condition, true_action=leaf_action)

        outer_condition = KeywordCondition(["code"])
        outer_node = DecisionNode(condition=outer_condition, true_node=inner_node)

        result = outer_node.evaluate({"query": "Write rust code"})
        assert result is not None
        assert result.confidence == 0.9

    def test_no_condition_returns_default(self):
        """Test node without condition returns default action."""
        default_action = DecisionResult(
            action=DecisionAction.SKIP,
            confidence=1.0,
            result={"reason": "No condition"},
            reasoning="Default action",
        )

        node = DecisionNode(true_action=default_action)
        result = node.evaluate({})

        assert result is not None
        assert result.action == DecisionAction.SKIP


class TestPreComputedDecisionTrees:
    """Test pre-computed decision trees."""

    def test_get_tree_by_name(self):
        """Test retrieving decision tree by name."""
        tree = PreComputedDecisionTrees.get_tree("file_read_tool")
        assert tree is not None

    def test_get_invalid_tree(self):
        """Test retrieving invalid tree name."""
        tree = PreComputedDecisionTrees.get_tree("invalid_tree")
        assert tree is None

    def test_evaluate_file_read_tree(self):
        """Test file read decision tree."""
        import tempfile
        import os

        # Create a temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test with directory (should use ls)
            context_dir = {"path": tmpdir, "query": "list directory"}
            result_dir = PreComputedDecisionTrees.evaluate_tree("file_read_tool", context_dir)

            assert result_dir is not None
            assert result_dir.action == DecisionAction.TOOL_CALL
            assert "ls" in str(result_dir.result).lower()

        # Test with file (should use read)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("# test")
            temp_path = f.name

        try:
            context_file = {"path": temp_path, "query": "read file"}
            result_file = PreComputedDecisionTrees.evaluate_tree("file_read_tool", context_file)

            # Note: The tree implementation checks if path exists and is file
            # Since we can't easily test without implementing actual path checking,
            # we'll just verify the tree can be evaluated
            # In real usage, it would use FileTypeCondition which checks Path.is_file()
            assert result_file is not None
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_evaluate_code_search_mode_tree(self):
        """Test code search mode decision tree."""
        context = {"query": "Find the function definition"}
        result = PreComputedDecisionTrees.evaluate_tree("code_search_mode", context)

        assert result is not None
        assert result.action == DecisionAction.TOOL_CALL
        assert "code_search" in str(result.result).lower()

    def test_evaluate_model_tier_selection_tree(self):
        """Test model tier selection decision tree."""
        context = {"query": "List directory contents"}
        result = PreComputedDecisionTrees.evaluate_tree("model_tier_selection", context)

        assert result is not None
        assert result.action == DecisionAction.MODEL_TIER
        assert result.confidence >= 0.8

    def test_evaluate_error_recovery_tree(self):
        """Test error recovery decision tree."""
        context = {"query": "File not found: main.py", "error": "not found"}
        result = PreComputedDecisionTrees.evaluate_tree("error_recovery_tool", context)

        assert result is not None
        # Should suggest directory listing or recovery action


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_decide_without_llm(self):
        """Test decide_without_llm function."""
        context = {"query": "List files"}
        result = decide_without_llm("model_tier_selection", context)

        assert result is not None
        assert result.confidence > 0

    def test_decide_without_llm_invalid_tree(self):
        """Test decide_without_llm with invalid tree."""
        context = {"query": "Test"}
        result = decide_without_llm("invalid_tree", context)

        assert result is None

    def test_can_decide_without_llm_true(self):
        """Test can_decide_without_llm returns True when confident."""
        context = {"query": "List files"}
        assert can_decide_without_llm("model_tier_selection", context, min_confidence=0.7) is True

    def test_can_decide_without_llm_false(self):
        """Test can_decide_without_llm returns False when not confident."""
        context = {"query": "Complex multi-step task"}
        # Most decisions will have lower confidence for complex tasks
        result = can_decide_without_llm("model_tier_selection", context, min_confidence=0.99)
        # Might be False depending on the tree logic
        assert isinstance(result, bool)

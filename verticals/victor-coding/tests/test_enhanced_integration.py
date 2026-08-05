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

"""Unit tests for enhanced victor-coding integration."""

from victor_coding.safety_enhanced import (
    CodingSafetyRules,
    EnhancedCodingSafetyExtension,
)
from victor_coding.conversation_enhanced import (
    CodingContext,
    EnhancedCodingConversationManager,
)
from victor.framework.extensions import SafetyAction, SafetyCategory


class TestCodingSafetyRules:
    """Test suite for CodingSafetyRules."""

    def test_get_git_rules(self):
        """Test getting git safety rules."""
        rules = CodingSafetyRules.get_git_rules()

        assert len(rules) > 0

        # Check for force push to main rule
        force_push_main = [r for r in rules if "force_push_main" in r.rule_id]
        assert len(force_push_main) == 1
        assert force_push_main[0].action == SafetyAction.BLOCK
        assert force_push_main[0].severity == 9

    def test_get_file_rules(self):
        """Test getting file safety rules."""
        rules = CodingSafetyRules.get_file_rules()

        assert len(rules) > 0

        # Check for system write rule
        system_write = [r for r in rules if "system_write" in r.rule_id]
        assert len(system_write) == 1
        assert system_write[0].action == SafetyAction.BLOCK

    def test_get_package_rules(self):
        """Test getting package management rules."""
        rules = CodingSafetyRules.get_package_rules()

        assert len(rules) > 0

        # Should have uninstall and install rules
        assert any("uninstall" in r.rule_id for r in rules)
        assert any("force_install" in r.rule_id for r in rules)

    def test_get_database_rules(self):
        """Test getting database safety rules."""
        rules = CodingSafetyRules.get_database_rules()

        assert len(rules) > 0

        # Check for DROP TABLE rule
        drop_table = [r for r in rules if "drop_table" in r.rule_id]
        assert len(drop_table) == 1
        assert drop_table[0].action == SafetyAction.BLOCK
        assert drop_table[0].severity == 10

    def test_get_all_rules(self):
        """Test getting all coding safety rules."""
        rules = CodingSafetyRules.get_all_rules()

        # Should include rules from all categories
        assert len(rules) > 0

        # Check for rules from each category
        assert any(r.rule_id.startswith("coding_git_") for r in rules)
        assert any(r.rule_id.startswith("coding_file_") for r in rules)
        assert any(r.rule_id.startswith("coding_pip_") for r in rules)
        assert any(r.rule_id.startswith("coding_db_") for r in rules)


class TestEnhancedCodingSafetyExtension:
    """Test suite for EnhancedCodingSafetyExtension."""

    def test_initialization(self):
        """Test extension initialization."""
        extension = EnhancedCodingSafetyExtension()

        assert extension._coordinator is not None
        assert len(extension._coordinator.list_rules()) > 0

    def test_check_operation_safe(self):
        """Test checking a safe operation."""
        extension = EnhancedCodingSafetyExtension()

        result = extension.check_operation("git", ["status"])

        assert result.is_safe is True
        assert result.action == SafetyAction.ALLOW

    def test_check_operation_force_push_main(self):
        """Test checking force push to main (blocked)."""
        extension = EnhancedCodingSafetyExtension()

        result = extension.check_operation("git", ["push", "--force", "origin", "main"])

        assert result.is_safe is False
        assert result.action == SafetyAction.BLOCK
        assert "main" in result.block_reason.lower()

    def test_check_operation_recursive_delete(self):
        """Test checking recursive delete (requires confirmation)."""
        extension = EnhancedCodingSafetyExtension()

        result = extension.check_operation("shell", ["rm", "-rf", "/path/to/dir"])

        assert result.is_safe is False
        assert result.action == SafetyAction.REQUIRE_CONFIRMATION

    def test_is_operation_safe(self):
        """Test is_operation_safe convenience method."""
        extension = EnhancedCodingSafetyExtension()

        assert extension.is_operation_safe("git", ["status"])
        assert not extension.is_operation_safe("git", ["push", "--force", "origin", "main"])

    def test_get_bash_patterns(self):
        """Test getting bash patterns."""
        extension = EnhancedCodingSafetyExtension()

        patterns = extension.get_bash_patterns()

        assert isinstance(patterns, list)

    def test_get_file_patterns(self):
        """Test getting file patterns."""
        extension = EnhancedCodingSafetyExtension()

        patterns = extension.get_file_patterns()

        assert isinstance(patterns, list)

    def test_get_tool_restrictions(self):
        """Test getting tool restrictions."""
        extension = EnhancedCodingSafetyExtension()

        restrictions = extension.get_tool_restrictions()

        assert "git" in restrictions
        assert "shell" in restrictions

    def test_add_custom_rule(self):
        """Test adding a custom safety rule."""
        from victor.framework.extensions import SafetyRule

        extension = EnhancedCodingSafetyExtension()

        initial_count = len(extension._coordinator.list_rules())

        custom_rule = SafetyRule(
            rule_id="custom_test_rule",
            category=SafetyCategory.SHELL,
            pattern=r"test_pattern",
            description="Test rule",
            action=SafetyAction.WARN,
            severity=3,
        )

        extension.add_custom_rule(custom_rule)

        assert len(extension._coordinator.list_rules()) == initial_count + 1

    def test_remove_rule(self):
        """Test removing a safety rule."""
        extension = EnhancedCodingSafetyExtension()

        # Add a temporary rule
        from victor.framework.extensions import SafetyRule

        temp_rule = SafetyRule(
            rule_id="temp_rule",
            category=SafetyCategory.SHELL,
            pattern=r"temp",
            description="Temporary rule",
            action=SafetyAction.WARN,
            severity=1,
        )

        extension.add_custom_rule(temp_rule)
        assert extension._coordinator.get_rule("temp_rule") is not None

        # Remove it
        result = extension.remove_rule("temp_rule")
        assert result is True
        assert extension._coordinator.get_rule("temp_rule") is None

    def test_get_safety_stats(self):
        """Test getting safety statistics."""
        extension = EnhancedCodingSafetyExtension()

        # Make some checks to generate stats
        extension.check_operation("git", ["status"])
        extension.check_operation("git", ["push", "--force", "origin", "main"])

        stats = extension.get_safety_stats()

        assert isinstance(stats, dict)
        assert "total_checks" in stats
        assert stats["total_checks"] >= 2

    def test_get_coordinator(self):
        """Test getting the underlying coordinator."""
        extension = EnhancedCodingSafetyExtension()

        coordinator = extension.get_coordinator()

        assert coordinator is not None
        assert hasattr(coordinator, "check_safety")


class TestCodingContext:
    """Test suite for CodingContext."""

    def test_initialization(self):
        """Test context initialization."""
        context = CodingContext()

        assert context.files_edited == []
        assert context.files_read == []
        assert context.tests_run == []
        assert context.git_operations == []

    def test_add_file_edit(self):
        """Test recording a file edit."""
        context = CodingContext()

        context.add_file_edit("src/auth.py", "edit")

        assert "src/auth.py" in context.files_edited
        assert len(context.files_edited) == 1

    def test_add_file_read(self):
        """Test recording a file read."""
        context = CodingContext()

        context.add_file_read("src/auth.py")
        context.add_file_read("src/auth.py")  # Duplicate

        assert "src/auth.py" in context.files_read
        assert len(context.files_read) == 1  # No duplicates

    def test_add_test_run(self):
        """Test recording a test run."""
        context = CodingContext()

        context.add_test_run("test_auth.py", passed=True, duration=1.5)

        assert len(context.tests_run) == 1
        assert context.tests_run[0]["name"] == "test_auth.py"
        assert context.tests_run[0]["passed"] is True
        assert context.tests_run[0]["duration"] == 1.5

    def test_add_git_operation(self):
        """Test recording a git operation."""
        context = CodingContext()

        context.add_git_operation("commit", "main")

        assert len(context.git_operations) == 1
        assert context.git_operations[0]["operation"] == "commit"
        assert context.git_operations[0]["target"] == "main"

    def test_to_dict(self):
        """Test converting context to dictionary."""
        context = CodingContext()

        context.add_file_edit("test.py")
        context.add_test_run("test.py", passed=True)

        context_dict = context.to_dict()

        assert "files_edited" in context_dict
        assert "tests_run" in context_dict
        assert context_dict["files_edited"] == ["test.py"]


class TestEnhancedCodingConversationManager:
    """Test suite for EnhancedCodingConversationManager."""

    def test_initialization(self):
        """Test manager initialization."""
        manager = EnhancedCodingConversationManager()

        assert manager._conversation_coordinator is not None
        assert isinstance(manager._coding_context, CodingContext)

    def test_add_message(self):
        """Test adding a message."""
        from victor.framework.extensions import TurnType

        manager = EnhancedCodingConversationManager()

        turn_id = manager.add_message("user", "Hello", TurnType.USER)

        assert turn_id == "turn_1"
        assert manager.get_stats().total_turns == 1

    def test_track_file_edit(self):
        """Test tracking file edits."""
        manager = EnhancedCodingConversationManager()

        manager.track_file_edit("src/auth.py")
        manager.track_file_edit("test.py", "create")

        context = manager.get_coding_context()

        assert len(context.files_edited) == 2
        assert "src/auth.py" in context.files_edited
        assert "test.py" in context.files_edited

    def test_track_test_run(self):
        """Test tracking test runs."""
        manager = EnhancedCodingConversationManager()

        manager.track_test_run("test_auth.py", passed=True)
        manager.track_test_run("test_user.py", passed=False)

        context = manager.get_coding_context()

        assert len(context.tests_run) == 2
        assert context.tests_run[0]["passed"] is True
        assert context.tests_run[1]["passed"] is False

    def test_track_git_operation(self):
        """Test tracking git operations."""
        manager = EnhancedCodingConversationManager()

        manager.track_git_operation("commit", None)
        manager.track_git_operation("push", "main")

        context = manager.get_coding_context()

        assert len(context.git_operations) == 2
        assert context.git_operations[0]["operation"] == "commit"
        assert context.git_operations[1]["target"] == "main"

    def test_get_coding_summary(self):
        """Test getting coding summary."""
        from victor.framework.extensions import TurnType

        manager = EnhancedCodingConversationManager()

        # Add some activity
        manager.track_file_edit("src/auth.py")
        manager.track_test_run("test_auth.py", passed=True)
        manager.add_message("user", "Fix the auth bug", TurnType.USER)

        summary = manager.get_coding_summary()

        assert "Files Modified" in summary
        assert "src/auth.py" in summary
        assert "Tests Run" in summary

    def test_needs_summarization(self):
        """Test summarization threshold detection."""
        from victor.framework.extensions import TurnType

        manager = EnhancedCodingConversationManager(
            summarization_threshold=5,
        )

        assert not manager.needs_summarization()

        # Add messages (not just file tracks) to trigger summarization
        for i in range(5):
            manager.add_message("user", f"Message {i}", TurnType.USER)

        assert manager.needs_summarization()

    def test_get_observability_data(self):
        """Test getting observability data."""
        manager = EnhancedCodingConversationManager()

        manager.track_file_edit("test.py")

        obs_data = manager.get_observability_data()

        assert obs_data["vertical"] == "coding"
        assert "coding_context" in obs_data
        assert obs_data["coding_context"]["files_edited"] == ["test.py"]

    def test_clear_history(self):
        """Test clearing conversation history."""
        from victor.framework.extensions import TurnType

        manager = EnhancedCodingConversationManager()

        manager.add_message("user", "Test", TurnType.USER)
        manager.track_file_edit("test.py")

        # Check that we have some turns
        assert manager.get_stats().total_turns >= 1
        assert len(manager.get_coding_context().files_edited) == 1

        manager.clear_history(keep_summaries=False)

        # Conversation should be cleared
        assert manager.get_stats().total_turns == 0
        assert len(manager.get_coding_context().files_edited) == 0

    def test_get_conversation_coordinator(self):
        """Test getting the underlying coordinator."""
        manager = EnhancedCodingConversationManager()

        coordinator = manager.get_conversation_coordinator()

        assert coordinator is not None
        assert hasattr(coordinator, "get_history")

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

"""LLM prompt templates for workflow refinement.

This module provides structured prompt templates for LLM-based workflow
refinement, with few-shot examples and clear instructions.

Design Principles:
    - Clear structure with sections
    - Concise error descriptions
    - Actionable fix suggestions
    - Few-shot examples for common errors
    - Output format specification

Example:
    from victor.workflows.generation import RefinementPromptBuilder

    builder = RefinementPromptBuilder()
    prompt = builder.build(
        user_request="Create a workflow for data analysis",
        workflow_dict=workflow,
        validation_result=result
    )
"""

import json
from typing import Any, Dict, List, Optional

from victor.workflows.generation.types import (
    ErrorCategory,
    ErrorSeverity,
    WorkflowValidationError,
    WorkflowGenerationValidationResult,
)
from victor.workflows.generation.error_reporter import ErrorReporter


class RefinementPromptBuilder:
    """Builds prompts for LLM workflow refinement.

    Provides structured prompts with:
    - Task description
    - Validation errors
    - Current workflow
    - Few-shot examples
    - Instructions
    - Output format

    Example:
        builder = RefinementPromptBuilder()

        prompt = builder.build(
            user_request="Analyze customer data",
            workflow_dict=workflow,
            validation_result=result,
            examples=get_examples_for_errors(result)
        )
    """

    def __init__(self):
        """Initialize prompt builder."""
        self.error_reporter = ErrorReporter()

    def build(
        self,
        user_request: str,
        workflow_dict: Dict[str, Any],
        validation_result: WorkflowGenerationValidationResult,
        examples: Optional[List[Dict[str, Any]]] = None,
        iteration: int = 0,
    ) -> str:
        """Build refinement prompt for LLM.

        Args:
            user_request: Original user request
            workflow_dict: Current workflow schema
            validation_result: Validation result with errors
            examples: Optional few-shot examples
            iteration: Current iteration number

        Returns:
            Formatted prompt string
        """
        sections = []

        # Section 1: Task description
        sections.append(self._task_description(user_request, iteration))

        # Section 2: Validation errors
        sections.append(self._validation_errors(validation_result))

        # Section 3: Current workflow
        sections.append(self._current_workflow(workflow_dict))

        # Section 4: Few-shot examples (if provided)
        if examples:
            sections.append(self._examples_section(examples))

        # Section 5: Instructions
        sections.append(self._instructions())

        # Section 6: Output format
        sections.append(self._output_format())

        return "\n\n".join(sections)

    def _task_description(self, user_request: str, iteration: int) -> str:
        """Task description section."""
        return f"""# Task: Fix Workflow Validation Errors

You are helping to fix a workflow that failed validation. The workflow was generated to fulfill this user request:

**User Request:** {user_request}

This is refinement iteration #{iteration + 1}. Previous attempts to fix this workflow have failed. Please carefully address all validation errors."""

    def _validation_errors(self, result: WorkflowGenerationValidationResult) -> str:
        """Format validation errors for LLM."""
        llm_report = self.error_reporter.llm_report(
            result, include_fixes=True, include_context=False
        )

        return f"""# Validation Errors

{llm_report}

**Critical:** You must fix all critical and error-level issues. Warnings should be addressed if possible."""

    def _current_workflow(self, workflow_dict: Dict[str, Any]) -> str:
        """Show current workflow YAML."""
        import yaml

        yaml_str = yaml.dump(workflow_dict, default_flow_style=False, sort_keys=False)

        return f"""# Current Workflow (Invalid)

```yaml
{yaml_str}
```"""

    def _examples_section(self, examples: List[Dict[str, Any]]) -> str:
        """Few-shot examples section."""
        sections = []
        for i, example in enumerate(examples, 1):
            sections.append(f"""## Example {i}

**Error:** {example['error']}

**Original:**
```yaml
{example.get('original', '(not provided)')}
```

**Fixed:**
```yaml
{example.get('fixed', '(not provided)')}
```

**Explanation:** {example.get('explanation', '(not provided)')}
""")

        return "\n".join(sections)

    def _instructions(self) -> str:
        """Clear instructions for LLM."""
        return """# Instructions

1. Review the validation errors carefully
2. Understand what each error means
3. Fix the workflow by addressing each error
4. Maintain the workflow's original purpose (fulfill user request)
5. Preserve correct parts of the workflow
6. Output ONLY the fixed workflow YAML (no explanations outside YAML)

**Key Guidelines:**
- All node IDs must be unique
- All edge sources/targets must reference valid nodes
- Entry point must be a valid node ID
- Agent nodes must have 'role' and 'goal'
- Compute nodes must have 'tools' or 'handler'
- Condition nodes must have 'branches' mapping
- Tool budgets must be integers between 1-500
- No unconditional cycles (add conditions if needed)
- All tools must exist in the tool registry

**Valid Node Types:**
- agent: LLM-powered agent with role and goal
- compute: Direct tool execution
- condition: Conditional routing with branches
- parallel: Execute multiple nodes in parallel
- transform: Transform workflow state
- team: Multi-agent collaboration
- hitl: Human-in-the-loop approval

**Valid Agent Roles:**
- researcher: Research and information gathering
- planner: Plan and strategize
- executor: Execute tasks
- reviewer: Review and validate
- writer: Write and document
- analyst: Analyze data

**Do NOT:**
- Change the workflow's purpose
- Remove nodes unless they cause errors
- Make assumptions about tools that don't exist
- Add unnecessary complexity"""

    def _output_format(self) -> str:
        """Specify expected output format."""
        return """# Output Format

Output ONLY the fixed workflow in YAML format. Start your response with the workflow 'name' field.

Example:
```yaml
name: my_workflow
description: "Fixed workflow"
entry_point: start
nodes:
  - id: start
    type: agent
    role: executor
    goal: "Execute the task"
    tool_budget: 15
edges:
  - source: start
    target: __end__
```

Do NOT include any explanations, comments, or text outside the YAML."""


class ExampleLibrary:
    """Library of few-shot examples for common errors.

    Provides examples for common validation errors and their fixes.
    """

    EXAMPLES = {
        "missing_entry_point": {
            "error": "ValidationError: Entry point 'start' not found in nodes",
            "original": """name: data_pipeline
nodes:
  - id: fetch
    type: agent
    role: researcher
    goal: "Fetch data"
edges:
  - source: fetch
    target: __end__
entry_point: start  # INVALID: 'start' node doesn't exist
""",
            "fixed": """name: data_pipeline
nodes:
  - id: fetch
    type: agent
    role: researcher
    goal: "Fetch data"
edges:
  - source: fetch
    target: __end__
entry_point: fetch  # FIXED: Use existing node as entry point
""",
            "explanation": "The entry_point must reference an existing node ID. Changed 'start' to 'fetch'.",
        },
        "unconditional_cycle": {
            "error": "GraphValidationError: Unconditional cycle detected: step1 -> step2 -> step1",
            "original": """name: iterative_process
nodes:
  - id: step1
    type: compute
    tools: [process_a]
  - id: step2
    type: compute
    tools: [process_b]
edges:
  - source: step1
    target: step2
  - source: step2
    target: step1  # INVALID: Unconditional cycle
entry_point: step1
""",
            "fixed": """name: iterative_process
nodes:
  - id: step1
    type: compute
    tools: [process_a]
  - id: step2
    type: compute
    tools: [process_b]
  - id: should_continue
    type: condition
    branches:
      continue: step1
      done: __end__
edges:
  - source: step1
    target: step2
  - source: step2
    target: should_continue
entry_point: step1
""",
            "explanation": "Added a condition node to break the cycle. The condition checks if processing should continue or finish.",
        },
        "missing_agent_fields": {
            "error": "SemanticValidationError: Agent node 'worker' must specify 'role' and 'goal'",
            "original": """name: task_runner
nodes:
  - id: worker
    type: agent
    # Missing 'role' and 'goal'
    tool_budget: 20
edges:
  - source: worker
    target: __end__
entry_point: worker
""",
            "fixed": """name: task_runner
nodes:
  - id: worker
    type: agent
    role: executor  # FIXED: Added role
    goal: "Execute the assigned task"  # FIXED: Added goal
    tool_budget: 20
edges:
  - source: worker
    target: __end__
entry_point: worker
""",
            "explanation": "Agent nodes require 'role' (what type of agent) and 'goal' (what task to perform).",
        },
        "invalid_tool_reference": {
            "error": "SemanticValidationError: Tool 'magic_wand' not found in registry",
            "original": """name: automation
nodes:
  - id: do_magic
    type: compute
    tools: [magic_wand]  # INVALID: Tool doesn't exist
edges:
  - source: do_magic
    target: __end__
entry_point: do_magic
""",
            "fixed": """name: automation
nodes:
  - id: do_magic
    type: compute
    tools: [bash_execute]  # FIXED: Use existing tool
edges:
  - source: do_magic
    target: __end__
entry_point: do_magic
""",
            "explanation": "Replaced non-existent tool 'magic_wand' with 'bash_execute' which exists in the tool registry.",
        },
        "orphan_node": {
            "error": "GraphValidationError: Node 'cleanup' is not reachable from entry point 'start'",
            "original": """name: data_flow
nodes:
  - id: start
    type: agent
    role: executor
    goal: "Start process"
  - id: cleanup
    type: compute
    tools: [cleanup_files]
edges:
  - source: start
    target: __end__
entry_point: start
# 'cleanup' node is unreachable
""",
            "fixed": """name: data_flow
nodes:
  - id: start
    type: agent
    role: executor
    goal: "Start process"
  - id: cleanup
    type: compute
    tools: [cleanup_files]
edges:
  - source: start
    target: cleanup
  - source: cleanup
    target: __end__
entry_point: start
""",
            "explanation": "Connected orphan node 'cleanup' to the workflow by adding edge from 'start' to 'cleanup'.",
        },
        "missing_condition_branches": {
            "error": "SchemaValidationError: Condition node missing 'branches' field",
            "original": """name: decision_flow
nodes:
  - id: check_quality
    type: condition
    # Missing 'branches'
edges:
  - source: check_quality
    target: __end__
entry_point: check_quality
""",
            "fixed": """name: decision_flow
nodes:
  - id: check_quality
    type: condition
    branches:
      high_quality: proceed
      needs_work: refine
edges:
  - source: check_quality
    target: __end__
entry_point: check_quality
""",
            "explanation": "Added 'branches' mapping to condition node, specifying which nodes to route to based on condition result.",
        },
        "invalid_node_type": {
            "error": "SchemaValidationError: Invalid node type: 'worker'",
            "original": """name: simple_task
nodes:
  - id: task1
    type: worker  # INVALID: 'worker' is not a valid node type
    role: executor
    goal: "Do work"
edges:
  - source: task1
    target: __end__
entry_point: task1
""",
            "fixed": """name: simple_task
nodes:
  - id: task1
    type: agent  # FIXED: Use 'agent' instead of 'worker'
    role: executor
    goal: "Do work"
edges:
  - source: task1
    target: __end__
entry_point: task1
""",
            "explanation": "Changed node type from 'worker' (invalid) to 'agent' (valid). Node types must be one of: agent, compute, condition, parallel, transform, team, hitl.",
        },
    }

    @classmethod
    def get_examples_for_errors(
        cls, validation_result: WorkflowGenerationValidationResult
    ) -> List[Dict[str, Any]]:
        """Get relevant examples based on validation errors.

        Args:
            validation_result: Validation result with errors

        Returns:
            List of relevant examples
        """
        # Group error types
        error_types = set()
        for error in validation_result.all_errors:
            # Extract error type from message
            error_msg = error.message.lower()

            if "entry point" in error_msg:
                error_types.add("missing_entry_point")
            elif "cycle" in error_msg:
                error_types.add("unconditional_cycle")
            elif "agent node" in error_msg and ("role" in error_msg or "goal" in error_msg):
                error_types.add("missing_agent_fields")
            elif "tool" in error_msg and "not found" in error_msg:
                error_types.add("invalid_tool_reference")
            elif "not reachable" in error_msg:
                error_types.add("orphan_node")
            elif "condition" in error_msg and "branches" in error_msg:
                error_types.add("missing_condition_branches")
            elif "invalid node type" in error_msg:
                error_types.add("invalid_node_type")

        # Retrieve examples
        examples = []
        for error_type in error_types:
            if error_type in cls.EXAMPLES:
                examples.append(cls.EXAMPLES[error_type])

        # Limit to 5 examples
        return examples[:5]

    @classmethod
    def get_all_examples(cls) -> Dict[str, Dict[str, Any]]:
        """Get all examples.

        Returns:
            Dictionary of all examples by error type
        """
        return cls.EXAMPLES.copy()

    @classmethod
    def add_example(
        cls, error_type: str, error: str, original: str, fixed: str, explanation: str
    ) -> None:
        """Add a new example to the library.

        Args:
            error_type: Type of error
            error: Error description
            original: Original workflow YAML
            fixed: Fixed workflow YAML
            explanation: Explanation of the fix
        """
        cls.EXAMPLES[error_type] = {
            "error": error,
            "original": original,
            "fixed": fixed,
            "explanation": explanation,
        }


def build_refinement_prompt(
    user_request: str,
    workflow_dict: Dict[str, Any],
    validation_result: WorkflowGenerationValidationResult,
    include_examples: bool = True,
    iteration: int = 0,
) -> str:
    """Convenience function to build refinement prompt.

    Args:
        user_request: Original user request
        workflow_dict: Current workflow schema
        validation_result: Validation result with errors
        include_examples: Whether to include few-shot examples
        iteration: Current iteration number

    Returns:
        Formatted prompt string

    Example:
        prompt = build_refinement_prompt(
            user_request="Create data pipeline",
            workflow_dict=workflow,
            validation_result=result
        )
    """
    builder = RefinementPromptBuilder()

    examples = None
    if include_examples:
        examples = ExampleLibrary.get_examples_for_errors(validation_result)

    return builder.build(
        user_request=user_request,
        workflow_dict=workflow_dict,
        validation_result=validation_result,
        examples=examples,
        iteration=iteration,
    )


__all__ = [
    "RefinementPromptBuilder",
    "ExampleLibrary",
    "build_refinement_prompt",
]

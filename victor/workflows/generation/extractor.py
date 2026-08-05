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

"""LLM-based requirement extraction from natural language.

This module provides the primary requirement extraction system using LLMs
with structured output. It extracts functional, structural, quality, and
context requirements from natural language workflow descriptions.

Design Pattern: Structured Extraction
- Uses JSON schema for validated LLM output
- Provides few-shot examples in prompt
- Calculates confidence scores per section
- Handles provider variations (Anthropic, OpenAI, etc.)

Example:
    from victor.workflows.generation.extractor import RequirementExtractor
    from victor.framework.protocols import FrameworkOrchestratorProtocol as OrchestratorProtocol

    extractor = RequirementExtractor(orchestrator)
    requirements = await extractor.extract(
        "Analyze code, find bugs, fix them, run tests"
    )
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from victor.framework.protocols import (
    FrameworkOrchestratorProtocol as OrchestratorProtocol,
)
from victor.workflows.generation.requirements import (
    BranchRequirement,
    ContextRequirements,
    ExtractionMetadata,
    FunctionalRequirements,
    LoopRequirement,
    ProjectContext,
    QualityRequirements,
    StructuralRequirements,
    TaskRequirement,
    WorkflowRequirements,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Prompt Templates
# =============================================================================


EXTRACTION_PROMPT = """
You are a workflow analysis expert. Extract structured workflow requirements
from the user's natural language description.

User Description:
{description}

Context:
{context}

Extract the following information:

1. **Functional Requirements**:
   - Tasks: What steps should be performed? (in order)
   - Tools: What tools are needed for each task?
   - Inputs: What data/files are needed?
   - Outputs: What should be produced?
   - Success Criteria: How do we know the workflow succeeded?

2. **Structural Requirements**:
   - Execution Order: Sequential, parallel, or conditional?
   - Dependencies: Which tasks depend on others?
   - Branches: Are there any conditional paths?
   - Loops: Are there any repetitive patterns?

3. **Quality Requirements**:
   - Performance Constraints: Max duration, max cost?
   - Quality Targets: Accuracy thresholds?
   - Resource Limits: Max tool calls, max tokens?

4. **Context Requirements**:
   - Domain: Which vertical (coding, devops, research, rag, dataanalysis)?
   - Environment: Local, cloud, or sandbox?
   - Preferences: Any user preferences mentioned?

Respond with a JSON object following this schema:
{schema}

Guidelines:
- Be specific but make reasonable assumptions if unclear
- If uncertain, set confidence score lower
- Extract conditional phrases ("if X then Y") as branches
- Detect parallel patterns ("do X and Y simultaneously")
- Identify success criteria explicitly stated or implied

Examples:

Input: "Analyze this Python codebase, find bugs, fix them, and run pytest to verify"
Output:
{{
  "functional": {{
    "tasks": [
      {{"id": "analyze", "description": "Analyze Python codebase", "task_type": "agent", "role": "researcher", "goal": "Find bugs using static analysis"}},
      {{"id": "find_bugs", "description": "Find bugs in code", "task_type": "agent", "role": "researcher", "dependencies": ["analyze"]}},
      {{"id": "fix", "description": "Fix identified bugs", "task_type": "agent", "role": "executor", "dependencies": ["find_bugs"]}},
      {{"id": "test", "description": "Run pytest to verify fixes", "task_type": "compute", "dependencies": ["fix"]}}
    ],
    "tools": {{
      "analyze": ["code_search", "ast_analyzer"],
      "find_bugs": ["code_search", "linter"],
      "test": ["bash"]
    }},
    "success_criteria": ["All tests pass", "No critical bugs remain"]
  }},
  "structural": {{
    "execution_order": "sequential",
    "dependencies": {{"find_bugs": ["analyze"], "fix": ["find_bugs"], "test": ["fix"]}}
  }},
  "context": {{
    "vertical": "coding",
    "subdomain": "bug_fix",
    "project_context": {{"primary_language": "Python"}}
  }}
}}

Input: "Research AI trends from 5 sources, summarize, and if quality score > 0.8, create report"
Output:
{{
  "functional": {{
    "tasks": [
      {{"id": "research", "description": "Research AI trends from 5 sources", "task_type": "agent", "role": "researcher"}},
      {{"id": "summarize", "description": "Summarize findings", "task_type": "agent", "role": "writer", "dependencies": ["research"]}},
      {{"id": "create_report", "description": "Create report", "task_type": "agent", "role": "writer", "dependencies": ["summarize"]}}
    ],
    "success_criteria": ["Quality score > 0.8", "At least 5 sources cited"]
  }},
  "structural": {{
    "execution_order": "conditional",
    "branches": [
      {{
        "condition_id": "quality_check",
        "condition": "quality_score > 0.8",
        "true_branch": "create_report",
        "false_branch": "end",
        "condition_type": "quality_threshold"
      }}
    ]
  }}
}}
"""


# =============================================================================
# JSON Schema
# =============================================================================


def build_requirements_schema() -> Dict[str, Any]:
    """Build JSON schema for LLM structured output.

    This schema is passed to the LLM to ensure it returns valid
    workflow requirements in the expected format.

    Returns:
        JSON schema dictionary for structured output.
    """
    return {
        "type": "object",
        "properties": {
            "functional": {
                "type": "object",
                "properties": {
                    "tasks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "description": {"type": "string"},
                                "task_type": {
                                    "type": "string",
                                    "enum": [
                                        "agent",
                                        "compute",
                                        "condition",
                                        "transform",
                                        "parallel",
                                    ],
                                },
                                "role": {
                                    "type": "string",
                                    "enum": [
                                        "researcher",
                                        "executor",
                                        "planner",
                                        "reviewer",
                                        "writer",
                                    ],
                                },
                                "goal": {"type": "string"},
                                "tools": {"type": "array", "items": {"type": "string"}},
                                "dependencies": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": ["id", "description", "task_type"],
                        },
                    },
                    "tools": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "inputs": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "type": {"type": "string"},
                                "source": {"type": "string"},
                                "required": {"type": "boolean"},
                            },
                        },
                    },
                    "outputs": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "type": {"type": "string"},
                                "destination": {"type": "string"},
                                "format": {"type": "string"},
                            },
                        },
                    },
                    "success_criteria": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["tasks", "success_criteria"],
            },
            "structural": {
                "type": "object",
                "properties": {
                    "execution_order": {
                        "type": "string",
                        "enum": ["sequential", "parallel", "mixed", "conditional"],
                    },
                    "dependencies": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "branches": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "condition_id": {"type": "string"},
                                "condition": {"type": "string"},
                                "true_branch": {"type": "string"},
                                "false_branch": {"type": "string"},
                                "condition_type": {"type": "string"},
                            },
                        },
                    },
                    "loops": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "loop_id": {"type": "string"},
                                "task_to_repeat": {"type": "string"},
                                "exit_condition": {"type": "string"},
                                "max_iterations": {"type": "integer"},
                            },
                        },
                    },
                },
            },
            "quality": {
                "type": "object",
                "properties": {
                    "max_duration_seconds": {"type": "integer"},
                    "max_cost_tier": {
                        "type": "string",
                        "enum": ["FREE", "LOW", "MEDIUM", "HIGH"],
                    },
                    "accuracy_threshold": {"type": "number"},
                    "max_tool_calls": {"type": "integer"},
                    "max_tokens": {"type": "integer"},
                    "retry_policy": {
                        "type": "string",
                        "enum": ["retry", "fail_fast", "continue", "fallback"],
                    },
                },
            },
            "context": {
                "type": "object",
                "properties": {
                    "vertical": {
                        "type": "string",
                        "enum": [
                            "coding",
                            "devops",
                            "research",
                            "rag",
                            "dataanalysis",
                            "benchmark",
                        ],
                    },
                    "subdomain": {"type": "string"},
                    "environment": {
                        "type": "string",
                        "enum": ["local", "cloud", "sandbox"],
                    },
                    "user_preferences": {"type": "object"},
                    "project_context": {
                        "type": "object",
                        "properties": {
                            "repo_path": {"type": "string"},
                            "primary_language": {"type": "string"},
                            "framework": {"type": "string"},
                            "testing_framework": {"type": "string"},
                            "build_system": {"type": "string"},
                        },
                    },
                },
                "required": ["vertical"],
            },
        },
        "required": ["functional", "structural", "quality", "context"],
    }


# =============================================================================
# Main Extractor Class
# =============================================================================


class RequirementExtractor:
    """Extract workflow requirements from natural language using LLM.

    Uses structured output (JSON schema) to extract all requirement
    categories in a single LLM call.

    Attributes:
        _orchestrator: Orchestrator for LLM access
        _schema: JSON schema for structured output

    Example:
        extractor = RequirementExtractor(orchestrator)
        requirements = await extractor.extract(
            "Analyze this codebase, find bugs, fix them, and run tests"
        )
        # requirements.functional.tasks -> [
        #     TaskRequirement(id="analyze", description="Analyze codebase"),
        #     TaskRequirement(id="fix", description="Fix bugs"),
        #     TaskRequirement(id="test", description="Run tests")
        # ]
    """

    def __init__(self, orchestrator: OrchestratorProtocol) -> None:
        """Initialize the extractor.

        Args:
            orchestrator: Orchestrator instance for LLM calls
        """
        self._orchestrator = orchestrator
        self._schema = build_requirements_schema()

    async def extract(
        self,
        description: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> WorkflowRequirements:
        """Extract structured requirements from natural language.

        Args:
            description: Natural language workflow description
            context: Optional context (project info, user preferences)

        Returns:
            WorkflowRequirements with all extracted information

        Raises:
            ValueError: If extraction fails or returns invalid data
            RuntimeError: If LLM call fails
        """
        start_time = time.time()

        logger.info(f"Extracting requirements from: {description[:100]}...")

        # Build prompt
        prompt = self._build_extraction_prompt(description, context)

        # Call LLM with structured output
        try:
            result = await self._call_llm_with_schema(prompt)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise RuntimeError(f"Failed to extract requirements: {e}") from e

        # Parse and validate
        try:
            requirements = self._parse_requirements(result)
        except Exception as e:
            logger.error(f"Failed to parse requirements: {e}")
            raise ValueError(f"Invalid LLM output format: {e}") from e

        # Calculate confidence scores
        confidence_scores = self._calculate_confidence(requirements, result)

        # Create metadata
        extraction_time = time.time() - start_time
        metadata = ExtractionMetadata(
            extraction_method="llm",
            model=self._orchestrator.current_model,
            extraction_time=extraction_time,
            confidence=(
                sum(confidence_scores.values()) / len(confidence_scores)
                if confidence_scores
                else 0.0
            ),
        )

        # Add metadata and confidence scores
        requirements.metadata = metadata
        requirements.confidence_scores = confidence_scores

        logger.info(
            f"Extraction complete in {extraction_time:.2f}s, "
            f"confidence: {metadata.confidence:.2f}"
        )

        return requirements

    def _build_extraction_prompt(self, description: str, context: Optional[Dict[str, Any]]) -> str:
        """Build the extraction prompt.

        Args:
            description: User's workflow description
            context: Optional additional context

        Returns:
            Formatted prompt string
        """
        context_str = json.dumps(context, indent=2) if context else "{}"
        schema_str = json.dumps(self._schema, indent=2)

        return EXTRACTION_PROMPT.format(
            description=description,
            context=context_str,
            schema=schema_str,
        )

    async def _call_llm_with_schema(self, prompt: str) -> Dict[str, Any]:
        """Call LLM with JSON schema for structured output.

        Args:
            prompt: Extraction prompt

        Returns:
            Parsed JSON response

        Raises:
            RuntimeError: If LLM call fails
        """
        # For now, we'll use a simple chat call and request JSON output
        # In production, you'd use provider-specific structured output APIs
        system_prompt = """You are a workflow analysis expert. Always respond with valid JSON only.
Do not include any explanatory text outside the JSON structure."""

        try:
            # Request JSON-only response
            response = await self._orchestrator.chat(
                f"{system_prompt}\n\n{prompt}",
                context={"response_format": "json"},
            )

            # Parse JSON response
            # Remove markdown code blocks if present
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()

            return json.loads(response)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response content: {response[:500]}")
            raise RuntimeError(f"LLM returned invalid JSON: {e}") from e

    def _parse_requirements(self, data: Dict[str, Any]) -> WorkflowRequirements:
        """Parse LLM output into WorkflowRequirements.

        Args:
            data: Raw JSON data from LLM

        Returns:
            Parsed WorkflowRequirements

        Raises:
            ValueError: If data is missing required fields
        """
        # Parse functional requirements
        functional_data = data.get("functional", {})
        tasks = [
            TaskRequirement(
                id=t["id"],
                description=t["description"],
                task_type=t["task_type"],
                role=t.get("role"),
                goal=t.get("goal"),
                tools=t.get("tools", []),
                dependencies=t.get("dependencies", []),
            )
            for t in functional_data.get("tasks", [])
        ]

        functional = FunctionalRequirements(
            tasks=tasks,
            tools=functional_data.get("tools", {}),
            success_criteria=functional_data.get("success_criteria", []),
        )

        # Parse structural requirements
        structural_data = data.get("structural", {})
        branches = [
            BranchRequirement(
                condition_id=b["condition_id"],
                condition=b["condition"],
                true_branch=b["true_branch"],
                false_branch=b["false_branch"],
                condition_type=b["condition_type"],
            )
            for b in structural_data.get("branches", [])
        ]

        loops = [
            LoopRequirement(
                loop_id=loop_data["loop_id"],
                task_to_repeat=loop_data["task_to_repeat"],
                exit_condition=loop_data["exit_condition"],
                max_iterations=loop_data.get("max_iterations", 3),
            )
            for loop_data in structural_data.get("loops", [])
        ]

        structural = StructuralRequirements(
            execution_order=structural_data.get("execution_order", "sequential"),
            dependencies=structural_data.get("dependencies", {}),
            branches=branches,
            loops=loops,
        )

        # Parse quality requirements
        quality_data = data.get("quality", {})
        quality = QualityRequirements(
            max_duration_seconds=quality_data.get("max_duration_seconds"),
            max_cost_tier=quality_data.get("max_cost_tier", "MEDIUM"),
            accuracy_threshold=quality_data.get("accuracy_threshold"),
            max_tool_calls=quality_data.get("max_tool_calls"),
            max_tokens=quality_data.get("max_tokens"),
            retry_policy=quality_data.get("retry_policy", "retry"),
        )

        # Parse context requirements
        context_data = data.get("context", {})
        project_context_data = context_data.get("project_context")
        project_context = (
            ProjectContext(
                repo_path=project_context_data.get("repo_path"),
                primary_language=project_context_data.get("primary_language"),
                framework=project_context_data.get("framework"),
                testing_framework=project_context_data.get("testing_framework"),
                build_system=project_context_data.get("build_system"),
            )
            if project_context_data
            else None
        )

        context = ContextRequirements(
            vertical=context_data.get("vertical", "coding"),
            subdomain=context_data.get("subdomain"),
            environment=context_data.get("environment", "local"),
            user_preferences=context_data.get("user_preferences", {}),
            project_context=project_context,
        )

        # Get original description
        description = data.get("description", "")

        return WorkflowRequirements(
            description=description,
            functional=functional,
            structural=structural,
            quality=quality,
            context=context,
        )

    def _calculate_confidence(
        self, requirements: WorkflowRequirements, raw_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate confidence scores for each requirement section.

        Args:
            requirements: Parsed requirements
            raw_data: Raw LLM output for analysis

        Returns:
            Dict mapping section names to confidence scores (0.0-1.0)
        """
        scores = {}

        # Functional confidence: based on task completeness
        if requirements.functional.tasks:
            complete_tasks = sum(
                1 for t in requirements.functional.tasks if t.description and t.task_type
            )
            scores["functional"] = complete_tasks / len(requirements.functional.tasks)
        else:
            scores["functional"] = 0.0

        # Structural confidence: based on execution order clarity
        if requirements.structural.execution_order in [
            "sequential",
            "parallel",
            "conditional",
        ]:
            scores["structural"] = 0.9
        else:
            scores["structural"] = 0.5

        # Quality confidence: based on constraint specificity
        quality_count = sum(
            1
            for v in [
                requirements.quality.max_duration_seconds,
                requirements.quality.max_cost_tier,
                requirements.quality.accuracy_threshold,
            ]
            if v is not None
        )
        scores["quality"] = min(1.0, quality_count / 2.0)

        # Context confidence: based on vertical detection
        if requirements.context.vertical:
            scores["context"] = 0.9
        else:
            scores["context"] = 0.0

        return scores

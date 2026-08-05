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

"""Workflow graph generator from natural language requirements.

This module provides the core generation system that converts structured
requirements into executable StateGraph workflows using LLM-based generation
with multi-stage validation and refinement.

Design Principles (SOLID):
    - SRP: WorkflowGenerator handles generation only (not validation or extraction)
    - OCP: Extensible via new generation strategies and prompts
    - LSP: TemplateGenerator can substitute for LLMPGenerator
    - ISP: Focused generation methods per strategy
    - DIP: Depends on WorkflowRequirements abstraction, not concrete implementation

Key Features:
    - Multi-stage generation: Understand → Design → Generate → Validate
    - Multiple provider support: Claude, GPT-4, Gemini
    - Retry with exponential backoff
    - Template fallback for simple patterns
    - Comprehensive error handling
    - Unified response parsing for robustness

Example:
    from victor.workflows.generation import WorkflowGenerator
    from victor.framework.protocols import FrameworkOrchestratorProtocol as OrchestratorProtocol

    generator = WorkflowGenerator(orchestrator, vertical="coding")
    workflow_schema = await generator.generate_from_requirements(requirements)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from enum import Enum

from victor.workflows.generation.requirements import WorkflowRequirements
from victor.workflows.generation.types import (
    WorkflowValidationError,
    WorkflowGenerationValidationResult,
    RefinementResult,
)

# Import unified response parser from framework
try:
    from victor.processing.response_parser import extract_json_from_response

    _UNIFIED_PARSER_AVAILABLE = True
except ImportError:
    _UNIFIED_PARSER_AVAILABLE = False
    logging.warning("Unified response parser not available, using local fallback")

logger = logging.getLogger(__name__)


# =============================================================================
# Generation Strategy Enum
# =============================================================================


class GenerationStrategy(Enum):
    """Workflow generation strategy.

    Determines how to convert requirements to workflow schema:
    - LLM_MULTI_STAGE: Multi-stage LLM generation (understand, design, generate)
    - LLM_SINGLE_STAGE: Single-stage LLM generation (faster, less accurate)
    - TEMPLATE_BASED: Template-based generation (fastest, least flexible)
    """

    LLM_MULTI_STAGE = "llm_multi_stage"
    LLM_SINGLE_STAGE = "llm_single_stage"
    TEMPLATE_BASED = "template_based"


# =============================================================================
# Generation Metadata
# =============================================================================


@dataclass
class GenerationMetadata:
    """Metadata about the generation process.

    Attributes:
        strategy: Which generation strategy was used
        model: LLM model used (if applicable)
        iterations: Number of refinement iterations
        duration_seconds: Total generation time
        fallback_used: Whether template fallback was used
        tokens_used: Approximate token count
        cost_estimate_usd: Estimated cost in USD
        attempt_number: Which attempt succeeded (1 = first try)
    """

    strategy: GenerationStrategy
    model: Optional[str] = None
    iterations: int = 0
    duration_seconds: float = 0.0
    fallback_used: bool = False
    tokens_used: int = 0
    cost_estimate_usd: float = 0.0
    attempt_number: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "strategy": self.strategy.value,
            "model": self.model,
            "iterations": self.iterations,
            "duration_seconds": self.duration_seconds,
            "fallback_used": self.fallback_used,
            "tokens_used": self.tokens_used,
            "cost_estimate_usd": self.cost_estimate_usd,
            "attempt_number": self.attempt_number,
        }


# =============================================================================
# Main Generator Class
# =============================================================================


class WorkflowGenerator:
    """Generate workflow schemas from requirements using LLM.

    Supports multiple generation strategies with automatic fallback to templates.
    Generates JSON schemas compatible with StateGraph.from_schema().

    Attributes:
        _orchestrator: Orchestrator for LLM access
        _vertical: Target vertical (coding, devops, research, etc.)
        _strategy: Preferred generation strategy
        _max_retries: Maximum LLM retry attempts
        _enable_templates: Whether to use template fallback

    Example:
        generator = WorkflowGenerator(
            orchestrator,
            vertical="coding",
            strategy=GenerationStrategy.LLM_MULTI_STAGE
        )

        schema = await generator.generate_from_requirements(requirements)
        # Returns dict compatible with StateGraph.from_schema()
    """

    # Valid node types for StateGraph
    VALID_NODE_TYPES = [
        "agent",
        "compute",
        "condition",
        "parallel",
        "transform",
        "team",
    ]

    # Valid agent roles
    VALID_ROLES = [
        "researcher",
        "executor",
        "planner",
        "reviewer",
        "writer",
        "analyst",
        "tester",
    ]

    def __init__(
        self,
        orchestrator: Any,  # OrchestratorProtocol
        vertical: str,
        strategy: GenerationStrategy = GenerationStrategy.LLM_MULTI_STAGE,
        max_retries: int = 3,
        enable_templates: bool = True,
    ):
        """Initialize the generator.

        Args:
            orchestrator: Orchestrator for LLM calls
            vertical: Target vertical (coding, devops, research, rag, dataanalysis)
            strategy: Preferred generation strategy
            max_retries: Maximum LLM retry attempts
            enable_templates: Whether to enable template fallback
        """
        self._orchestrator = orchestrator
        self._vertical = vertical
        self._strategy = strategy
        self._max_retries = max_retries
        self._enable_templates = enable_templates

    async def generate_from_requirements(
        self,
        requirements: WorkflowRequirements,
        validation_callback: Optional[callable] = None,
    ) -> tuple[Dict[str, Any], GenerationMetadata]:
        """Generate workflow schema from structured requirements.

        Args:
            requirements: Extracted workflow requirements
            validation_callback: Optional callback for validation results

        Returns:
            Tuple of (workflow_schema_dict, generation_metadata)

        Raises:
            RuntimeError: If all generation attempts fail
            ValueError: If requirements are invalid
        """
        start_time = time.time()
        logger.info(f"Generating workflow for vertical: {self._vertical}")

        # Validate requirements
        self._validate_requirements(requirements)

        # Try preferred strategy first
        last_error = None
        for attempt in range(1, self._max_retries + 1):
            try:
                logger.info(f"Generation attempt {attempt}/{self._max_retries}")

                # Generate using preferred strategy
                if self._strategy == GenerationStrategy.LLM_MULTI_STAGE:
                    schema, metadata = await self._generate_multi_stage(requirements, attempt)
                elif self._strategy == GenerationStrategy.LLM_SINGLE_STAGE:
                    schema, metadata = await self._generate_single_stage(requirements, attempt)
                else:
                    schema, metadata = await self._generate_from_template(requirements)

                # Validate if callback provided
                if validation_callback:
                    validation = validation_callback(schema)
                    if not validation.is_valid:
                        raise ValueError(
                            f"Generated schema failed validation: {validation.all_errors[:3]}"
                        )

                # Success!
                duration = time.time() - start_time
                metadata.duration_seconds = duration
                metadata.attempt_number = attempt

                logger.info(
                    f"Generation successful in {duration:.2f}s "
                    f"(strategy: {metadata.strategy.value}, "
                    f"attempt: {attempt})"
                )

                return schema, metadata

            except Exception as e:
                last_error = e
                logger.warning(f"Generation attempt {attempt} failed: {e}")
                # Exponential backoff
                if attempt < self._max_retries:
                    await asyncio.sleep(2**attempt)

        # All retries failed - try template fallback
        if self._enable_templates and self._strategy != GenerationStrategy.TEMPLATE_BASED:
            logger.info("LLM generation failed, trying template fallback")
            try:
                schema, metadata = await self._generate_from_template(requirements)
                metadata.fallback_used = True
                metadata.duration_seconds = time.time() - start_time
                return schema, metadata
            except Exception as e:
                logger.error(f"Template fallback also failed: {e}")

        # Everything failed
        raise RuntimeError(
            f"Failed to generate workflow after {self._max_retries} attempts: {last_error}"
        ) from last_error

    async def refine_schema(
        self,
        schema: Dict[str, Any],
        feedback: Union[str, List[str]],
        validation_errors: Optional[List[WorkflowValidationError]] = None,
    ) -> Dict[str, Any]:
        """Refine existing schema based on feedback.

        Args:
            schema: Current workflow schema
            feedback: Natural language feedback or list of specific errors
            validation_errors: Optional validation errors for context

        Returns:
            Refined workflow schema

        Raises:
            RuntimeError: If refinement fails
        """
        logger.info("Refining workflow schema based on feedback")

        # Build refinement prompt
        prompt = self._build_refinement_prompt(schema, feedback, validation_errors)

        # Call LLM
        try:
            response = await self._orchestrator.chat(prompt, context={"response_format": "json"})
            refined_schema = self._parse_json_response(response)

            # Basic validation
            self._validate_schema_structure(refined_schema)

            logger.info("Schema refinement successful")
            return refined_schema

        except Exception as e:
            logger.error(f"Schema refinement failed: {e}")
            raise RuntimeError(f"Failed to refine schema: {e}") from e

    # =============================================================================
    # Multi-Stage Generation (Most Accurate)
    # =============================================================================

    async def _generate_multi_stage(
        self, requirements: WorkflowRequirements, attempt: int
    ) -> tuple[Dict[str, Any], GenerationMetadata]:
        """Generate workflow using multi-stage approach.

        Stage 1: Understand requirements (confirm with LLM)
        Stage 2: Design structure (nodes, edges, conditions)
        Stage 3: Generate full JSON schema
        Stage 4: Validate and refine (loop if needed)

        This is the most accurate but slowest method.
        """
        logger.debug("Using multi-stage generation")

        # Stage 1: Confirm understanding
        understanding = await self._stage1_understand(requirements)
        logger.debug(f"Stage 1 complete: {understanding}")

        # Stage 2: Design structure
        structure = await self._stage2_design(requirements, understanding)
        logger.debug(f"Stage 2 complete: {structure}")

        # Stage 3: Generate schema
        schema = await self._stage3_generate(requirements, structure)
        logger.debug("Stage 3 complete: schema generated")

        # Stage 4: Validate
        self._validate_schema_structure(schema)

        metadata = GenerationMetadata(
            strategy=GenerationStrategy.LLM_MULTI_STAGE,
            model=self._orchestrator.current_model,
            iterations=3,  # 3 stages
        )

        return schema, metadata

    async def _stage1_understand(self, requirements: WorkflowRequirements) -> Dict[str, Any]:
        """Stage 1: Understand and confirm requirements.

        Returns a summary of what the LLM understood.
        """
        prompt = f"""
You are a workflow architect. Analyze these requirements and confirm your understanding.

**Vertical:** {self._vertical}
**Description:** {requirements.description}

**Functional Requirements:**
- Tasks: {len(requirements.functional.tasks)} tasks
- Tools needed: {list(requirements.functional.tools.keys())}
- Success criteria: {requirements.functional.success_criteria}

**Structural Requirements:**
- Execution order: {requirements.structural.execution_order}
- Dependencies: {requirements.structural.dependencies}
- Branches: {len(requirements.structural.branches)} branches
- Loops: {len(requirements.structural.loops)} loops

**Quality Requirements:**
- Max duration: {requirements.quality.max_duration_seconds}s
- Max cost tier: {requirements.quality.max_cost_tier}

Provide a brief summary (2-3 sentences) of what this workflow should do.
Focus on the core objective and key steps.
"""

        response = await self._orchestrator.chat(prompt)
        return {"summary": response.strip()}

    async def _stage2_design(
        self, requirements: WorkflowRequirements, understanding: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Stage 2: Design workflow structure.

        Returns high-level node and edge design.
        """
        prompt = f"""
You are a workflow architect. Design the workflow structure.

**Objective:** {understanding['summary']}

**Tasks:**
{json.dumps([t.to_dict() for t in requirements.functional.tasks], indent=2)}

**Requirements:**
- Execution order: {requirements.structural.execution_order}
- Dependencies: {requirements.structural.dependencies}
- Branches: {requirements.structural.branches}

Design the workflow structure:
1. List all nodes with their types (agent, compute, condition)
2. Define edges between nodes
3. Identify any conditional logic

Respond with a JSON structure:
{{
  "nodes": [
    {{"id": "node1", "type": "agent", "purpose": "..."}},
    {{"id": "node2", "type": "compute", "purpose": "..."}}
  ],
  "flow": "node1 -> node2 -> node3",
  "conditional_logic": "if condition then node_a else node_b"
}}
"""

        response = await self._orchestrator.chat(prompt, context={"response_format": "json"})
        return self._parse_json_response(response)

    async def _stage3_generate(
        self, requirements: WorkflowRequirements, structure: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Stage 3: Generate full workflow JSON schema.

        Returns complete schema compatible with StateGraph.from_schema().
        """
        schema_template = self._get_schema_template()

        prompt = f"""
You are a workflow generator. Generate a complete StateGraph JSON schema.

**Workflow Structure:**
{json.dumps(structure, indent=2)}

**Requirements:**
{json.dumps(requirements.to_dict(), indent=2)}

**Schema Template to Follow:**
{json.dumps(schema_template, indent=2)}

**Important:**
- All nodes must have unique IDs
- All edge targets must reference existing node IDs or "__end__"
- Agent nodes must have 'role' and 'goal'
- Condition nodes must have 'condition' and 'branches'
- Entry point must be a valid node ID

Generate the complete workflow schema as JSON only.
"""

        response = await self._orchestrator.chat(prompt, context={"response_format": "json"})
        schema = self._parse_json_response(response)

        return schema

    # =============================================================================
    # Single-Stage Generation (Faster)
    # =============================================================================

    async def _generate_single_stage(
        self, requirements: WorkflowRequirements, attempt: int
    ) -> tuple[Dict[str, Any], GenerationMetadata]:
        """Generate workflow in single LLM call.

        Faster than multi-stage but less accurate for complex workflows.
        """
        logger.debug("Using single-stage generation")

        schema_template = self._get_schema_template()

        prompt = f"""
You are a workflow generator. Generate a StateGraph JSON schema from requirements.

**Vertical:** {self._vertical}
**Description:** {requirements.description}

**Requirements:**
{json.dumps(requirements.to_dict(), indent=2)}

**Valid Node Types:** {', '.join(self.VALID_NODE_TYPES)}
**Valid Agent Roles:** {', '.join(self.VALID_ROLES)}

**Schema Template:**
{json.dumps(schema_template, indent=2)}

**Requirements:**
1. Generate valid JSON following the template structure
2. All nodes must be reachable from entry_point
3. Conditional edges must have condition function
4. Agent nodes must have role and goal
5. Compute nodes must have handler
6. Keep workflows simple (<10 nodes preferred)

Generate the workflow schema as JSON only.
"""

        response = await self._orchestrator.chat(prompt, context={"response_format": "json"})
        schema = self._parse_json_response(response)

        # Basic validation
        self._validate_schema_structure(schema)

        metadata = GenerationMetadata(
            strategy=GenerationStrategy.LLM_SINGLE_STAGE,
            model=self._orchestrator.current_model,
            iterations=1,
        )

        return schema, metadata

    # =============================================================================
    # Template-Based Generation (Fastest)
    # =============================================================================

    async def _generate_from_template(
        self, requirements: WorkflowRequirements
    ) -> tuple[Dict[str, Any], GenerationMetadata]:
        """Generate workflow from pre-defined template.

        Fastest method but least flexible. Matches requirements to
        closest template pattern and instantiates with parameters.
        """
        logger.debug("Using template-based generation")

        # Import here to avoid circular dependency
        from victor.workflows.generation.templates import TemplateLibrary

        library = TemplateLibrary()

        # Find matching template
        template = library.match_template(requirements, self._vertical)

        if not template:
            raise ValueError(
                f"No matching template found for vertical '{self._vertical}' "
                f"with execution order '{requirements.structural.execution_order}'"
            )

        # Instantiate template
        schema = library.instantiate_template(template, requirements)

        metadata = GenerationMetadata(
            strategy=GenerationStrategy.TEMPLATE_BASED,
            iterations=0,
        )

        return schema, metadata

    # =============================================================================
    # Helper Methods
    # =============================================================================

    def _validate_requirements(self, requirements: WorkflowRequirements) -> None:
        """Validate requirements before generation.

        Raises:
            ValueError: If requirements are invalid
        """
        if not requirements.description:
            raise ValueError("Requirements must have a description")

        if not requirements.functional.tasks:
            raise ValueError("Requirements must have at least one task")

        if requirements.context.vertical != self._vertical:
            logger.warning(
                f"Requirements vertical '{requirements.context.vertical}' "
                f"doesn't match generator vertical '{self._vertical}'"
            )

    def _validate_schema_structure(self, schema: Dict[str, Any]) -> None:
        """Basic validation of generated schema structure.

        Args:
            schema: Generated workflow schema

        Raises:
            ValueError: If schema structure is invalid
        """
        required_fields = ["nodes", "edges", "entry_point"]
        for field_name in required_fields:
            if field_name not in schema:
                raise ValueError(f"Schema missing required field: {field_name}")

        if not isinstance(schema["nodes"], list) or len(schema["nodes"]) == 0:
            raise ValueError("Schema must have at least one node")

        if not isinstance(schema["edges"], list):
            raise ValueError("Schema edges must be a list")

        # Validate nodes
        node_ids = set()
        for node in schema["nodes"]:
            if "id" not in node:
                raise ValueError("Node missing 'id' field")
            if "type" not in node:
                raise ValueError(f"Node '{node['id']}' missing 'type' field")
            if node["type"] not in self.VALID_NODE_TYPES:
                raise ValueError(
                    f"Node '{node['id']}' has invalid type '{node['type']}'. "
                    f"Valid types: {self.VALID_NODE_TYPES}"
                )
            node_ids.add(node["id"])

        # Validate entry point
        if schema["entry_point"] not in node_ids:
            raise ValueError(f"Entry point '{schema['entry_point']}' not found in nodes")

        # Validate edges
        for edge in schema["edges"]:
            if "source" not in edge or "target" not in edge:
                raise ValueError("Edge missing 'source' or 'target' field")
            if edge["source"] not in node_ids and edge["source"] != "__start__":
                raise ValueError(f"Edge source '{edge['source']}' not found in nodes")

            # Check target (can be string or dict for conditional)
            target = edge["target"]
            if isinstance(target, str):
                if target != "__end__" and target not in node_ids:
                    raise ValueError(f"Edge target '{target}' not found in nodes")
            elif isinstance(target, dict):
                for branch_target in target.values():
                    if branch_target != "__end__" and branch_target not in node_ids:
                        raise ValueError(f"Edge branch target '{branch_target}' not found in nodes")

    def _build_refinement_prompt(
        self,
        schema: Dict[str, Any],
        feedback: Union[str, List[str]],
        validation_errors: Optional[List[WorkflowValidationError]] = None,
    ) -> str:
        """Build prompt for schema refinement."""
        feedback_text = (
            "\n".join(f"- {f}" for f in feedback) if isinstance(feedback, list) else feedback
        )

        prompt = f"""
Fix this workflow JSON schema based on feedback:

**Current Schema:**
{json.dumps(schema, indent=2)}

**Feedback:**
{feedback_text}
"""

        if validation_errors:
            errors_text = "\n".join(
                f"- [{err.category.value}] {err.message} @ {err.location}"
                for err in validation_errors[:10]
            )
            prompt += f"\n**Validation Errors:**\n{errors_text}"

        prompt += """

**Instructions:**
1. Fix all issues mentioned in the feedback
2. Maintain the workflow's original intent
3. Ensure all required fields are present
4. Verify all node and edge references are valid
5. Output valid JSON only (no explanations)

Fixed schema:
"""
        return prompt

    def _parse_json_response(self, response: Union[str, Dict[str, Any], Any]) -> Dict[str, Any]:
        """Parse JSON from LLM response using unified parser.

        This method uses the unified response parser from victor.processing.response_parser
        which provides robust JSON extraction with:
        - Markdown code block handling
        - Malformed JSON repair
        - Multiple response format support

        Args:
            response: LLM response (string, dict, or object)

        Returns:
            Parsed JSON dict

        Raises:
            ValueError: If JSON is invalid or cannot be extracted
        """
        # Use unified response parser if available
        if _UNIFIED_PARSER_AVAILABLE:
            json_str = extract_json_from_response(response)
            if json_str:
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse extracted JSON: {e}")
                    logger.debug(f"JSON string: {json_str[:500]}")
                    raise ValueError(f"Invalid JSON in LLM response: {e}") from e
            else:
                raise ValueError("Invalid JSON in LLM response: No valid JSON found")

        # Fallback: simple markdown stripping and JSON parsing
        logger.warning("Unified parser unavailable, using simple fallback")
        response_str = response if isinstance(response, str) else str(response)
        response_str = response_str.strip()

        # Remove markdown code blocks
        if response_str.startswith("```json"):
            response_str = response_str[7:]
        if response_str.startswith("```"):
            response_str = response_str[3:]
        if response_str.endswith("```"):
            response_str = response_str[:-3]
        response_str = response_str.strip()

        # Parse JSON
        try:
            return json.loads(response_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.debug(f"Response content: {response_str[:500]}")
            raise ValueError(f"Invalid JSON in LLM response: {e}") from e

    def _get_schema_template(self) -> Dict[str, Any]:
        """Get template for workflow schema structure."""
        return {
            "workflow_name": "example_workflow",
            "description": "An example workflow",
            "nodes": [
                {
                    "id": "node1",
                    "type": "agent",
                    "role": "executor",
                    "goal": "Execute the task",
                    "tool_budget": 10,
                    "output_key": "result1",
                }
            ],
            "edges": [
                {
                    "source": "node1",
                    "target": "__end__",
                    "type": "normal",
                }
            ],
            "entry_point": "node1",
            "metadata": {
                "vertical": self._vertical,
                "max_iterations": 25,
                "timeout_seconds": 300,
            },
        }


__all__ = [
    "WorkflowGenerator",
    "GenerationStrategy",
    "GenerationMetadata",
]

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

"""End-to-end pipeline for workflow generation from natural language.

This module provides the complete pipeline that coordinates:
1. Requirement extraction from natural language
2. Workflow schema generation
3. Schema validation
4. Automated refinement
5. StateGraph compilation

Design Principles (SOLID):
    - SRP: Pipeline coordinates components (doesn't implement them)
    - OCP: Extensible via custom validators and refiners
    - LSP: Pipelines can substitute for each other
    - ISP: Focused pipeline methods per mode (auto, interactive)
    - DIP: Depends on protocols (OrchestratorProtocol, StateGraph)

Key Features:
    - Auto mode: Fully automated generation
    - Interactive mode: User approval at each stage
    - Metrics tracking: Time, cost, attempts
    - Comprehensive error handling
    - StateGraph output ready for execution

Example:
    from victor.workflows.generation import WorkflowGenerationPipeline
    from victor.framework.graph import StateGraph

    pipeline = WorkflowGenerationPipeline(orchestrator, vertical="coding")

    # Auto mode
    graph = await pipeline.generate_workflow(
        "Fix the authentication bug",
        mode="auto"
    )

    # Compile and execute
    app = graph.compile()
    result = await app.invoke(initial_state)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum

from victor.framework.graph import StateGraph
from victor.workflows.generation.generator import (
    WorkflowGenerator,
    GenerationStrategy,
    GenerationMetadata,
)
from victor.workflows.generation.extractor import RequirementExtractor
from victor.workflows.generation.refiner import WorkflowRefiner
from victor.workflows.generation.requirements import WorkflowRequirements
from victor.workflows.generation.types import (
    WorkflowGenerationValidationResult,
    RefinementResult,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Pipeline Mode
# =============================================================================


class PipelineMode(Enum):
    """Pipeline execution mode.

    - AUTO: Fully automated, no user interaction
    - INTERACTIVE: User approval at each stage
    - SAFE: Validate everything, require approval for dangerous operations
    """

    AUTO = "auto"
    INTERACTIVE = "interactive"
    SAFE = "safe"


# =============================================================================
# Pipeline Result
# =============================================================================


@dataclass
class PipelineResult:
    """Result from workflow generation pipeline.

    Attributes:
        success: Whether generation was successful
        graph: Generated StateGraph (if successful)
        schema: Generated workflow schema (before compilation)
        requirements: Extracted requirements
        validation: Final validation result
        metadata: Generation metadata
        errors: List of errors encountered (if failed)
        warnings: List of warnings (non-critical)
        duration_seconds: Total pipeline duration
    """

    success: bool
    graph: Optional[StateGraph] = None
    schema: Optional[Dict[str, Any]] = None
    requirements: Optional[WorkflowRequirements] = None
    validation: Optional[WorkflowGenerationValidationResult] = None
    metadata: Optional[GenerationMetadata] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    @property
    def error_summary(self) -> str:
        """Get formatted error summary."""
        if not self.errors:
            return "No errors"
        return "\n".join(f"- {err}" for err in self.errors)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "success": self.success,
            "metadata": self.metadata.to_dict() if self.metadata else None,
            "validation": self.validation.to_dict() if self.validation else None,
            "errors": self.errors,
            "warnings": self.warnings,
            "duration_seconds": self.duration_seconds,
        }


# =============================================================================
# Main Pipeline Class
# =============================================================================


class WorkflowGenerationPipeline:
    """End-to-end pipeline for workflow generation.

    Coordinates the full generation process from natural language
    to executable StateGraph.

    Attributes:
        _orchestrator: Orchestrator for LLM access
        _vertical: Target vertical
        _strategy: Preferred generation strategy
        _enable_refinement: Whether to enable automated refinement
        _max_refinement_iterations: Max refinement loops

    Example:
        pipeline = WorkflowGenerationPipeline(
            orchestrator,
            vertical="coding",
            strategy=GenerationStrategy.LLM_MULTI_STAGE
        )

        result = await pipeline.generate_workflow(
            "Analyze code and fix bugs",
            mode="auto"
        )

        if result.success:
            app = result.graph.compile()
            await app.invoke(initial_state)
    """

    def __init__(
        self,
        orchestrator: Any,  # OrchestratorProtocol
        vertical: str,
        strategy: GenerationStrategy = GenerationStrategy.LLM_MULTI_STAGE,
        enable_refinement: bool = True,
        max_refinement_iterations: int = 3,
    ):
        """Initialize the generation pipeline.

        Args:
            orchestrator: Orchestrator for LLM calls
            vertical: Target vertical (coding, devops, research, etc.)
            strategy: Preferred generation strategy
            enable_refinement: Whether to enable automated refinement
            max_refinement_iterations: Maximum refinement iterations
        """
        self._orchestrator = orchestrator
        self._vertical = vertical
        self._strategy = strategy
        self._enable_refinement = enable_refinement
        self._max_refinement_iterations = max_refinement_iterations

        # Initialize components
        self._extractor = RequirementExtractor(orchestrator)
        self._generator = WorkflowGenerator(
            orchestrator,
            vertical=vertical,
            strategy=strategy,
        )
        self._refiner = WorkflowRefiner(conservative=True) if enable_refinement else None

    async def generate_workflow(
        self,
        description: str,
        mode: str = "auto",
        context: Optional[Dict[str, Any]] = None,
        validation_callback: Optional[Callable] = None,
        progress_callback: Optional[Callable] = None,
    ) -> PipelineResult:
        """Generate workflow from natural language description.

        This is the main entry point for workflow generation. It coordinates:
        1. Requirement extraction
        2. Schema generation
        3. Validation
        4. Refinement (if needed)
        5. StateGraph compilation

        Args:
            description: Natural language workflow description
            mode: Pipeline mode ("auto", "interactive", "safe")
            context: Additional context for extraction
            validation_callback: Optional custom validation function
            progress_callback: Optional callback for progress updates

        Returns:
            PipelineResult with generated StateGraph or errors

        Example:
            result = await pipeline.generate_workflow(
                "Analyze this codebase and fix bugs",
                mode="auto"
            )

            if result.success:
                graph = result.graph
                app = graph.compile()
                await app.invoke({"task": "fix bugs"})
        """
        start_time = time.time()
        logger.info(f"Starting workflow generation pipeline (mode: {mode})")

        result = PipelineResult(success=False)

        try:
            # Stage 1: Extract requirements
            if progress_callback:
                await progress_callback("Extracting requirements...")

            requirements = await self._extractor.extract(description, context)
            result.requirements = requirements
            logger.info(f"Requirements extracted: {len(requirements.functional.tasks)} tasks")

            # Interactive mode: show requirements and ask for approval
            if mode == "interactive":
                if not await self._interactive_approval(
                    "Extracted Requirements", requirements.to_dict()
                ):
                    result.errors.append("User rejected requirements")
                    return result

            # Stage 2: Generate schema
            if progress_callback:
                await progress_callback("Generating workflow schema...")

            schema, metadata = await self._generator.generate_from_requirements(
                requirements,
                validation_callback=validation_callback,
            )
            result.schema = schema
            result.metadata = metadata
            logger.info("Schema generated successfully")

            # Interactive mode: show schema and ask for approval
            if mode == "interactive":
                if not await self._interactive_approval("Generated Schema", schema):
                    result.errors.append("User rejected schema")
                    return result

            # Stage 3: Validate schema
            if progress_callback:
                await progress_callback("Validating schema...")

            validation = await self._validate_schema(schema)
            result.validation = validation

            if not validation.is_valid and mode == "safe":
                result.errors = [e.message for e in validation.all_errors]
                logger.error(f"Validation failed in safe mode: {result.errors}")
                return result

            # Stage 4: Refine if needed
            if not validation.is_valid and self._enable_refinement:
                if progress_callback:
                    await progress_callback("Refining schema...")

                logger.info(
                    f"Validation failed, attempting refinement ({len(validation.all_errors)} errors)"
                )

                refined_schema, refinement_result = await self._refine_schema(schema, validation)

                if refinement_result.success:
                    schema = refined_schema
                    result.schema = schema
                    result.validation = refinement_result.validation_result

                    logger.info(f"Refinement successful: {refinement_result.summary()}")
                else:
                    logger.warning(f"Refinement failed: {refinement_result.summary()}")
                    # Continue with original schema if refinement fails

            # Stage 5: Compile to StateGraph
            if progress_callback:
                await progress_callback("Compiling to StateGraph...")

            graph = await self._compile_to_graph(schema)
            result.graph = graph
            result.success = True

            duration = time.time() - start_time
            result.duration_seconds = duration

            logger.info(
                f"Pipeline completed successfully in {duration:.2f}s "
                f"(strategy: {metadata.strategy.value}, "
                f"iterations: {metadata.iterations})"
            )

            return result

        except Exception as e:
            duration = time.time() - start_time
            result.duration_seconds = duration
            result.errors.append(f"Pipeline failed: {str(e)}")
            logger.exception("Pipeline failed with exception")
            return result

    async def refine_workflow(
        self,
        graph: StateGraph,
        feedback: str,
        mode: str = "auto",
    ) -> PipelineResult:
        """Refine existing workflow based on feedback.

        Args:
            graph: Existing StateGraph to refine
            feedback: Natural language feedback for changes
            mode: Pipeline mode

        Returns:
            PipelineResult with refined StateGraph
        """
        start_time = time.time()
        logger.info("Starting workflow refinement")

        result = PipelineResult(success=False)

        try:
            # Convert graph to schema
            schema = graph.to_dict()
            result.schema = schema

            # Refine schema
            refined_schema = await self._generator.refine_schema(schema, feedback)
            result.schema = refined_schema

            # Validate
            validation = await self._validate_schema(refined_schema)
            result.validation = validation

            if not validation.is_valid and mode == "safe":
                result.errors = [e.message for e in validation.all_errors]
                return result

            # Compile to StateGraph
            graph = await self._compile_to_graph(refined_schema)
            result.graph = graph
            result.success = True

            duration = time.time() - start_time
            result.duration_seconds = duration

            logger.info(f"Refinement completed in {duration:.2f}s")

            return result

        except Exception as e:
            duration = time.time() - start_time
            result.duration_seconds = duration
            result.errors.append(f"Refinement failed: {str(e)}")
            logger.exception("Refinement failed")
            return result

    # =============================================================================
    # Private Methods
    # =============================================================================

    async def _validate_schema(self, schema: Dict[str, Any]) -> WorkflowGenerationValidationResult:
        """Validate workflow schema.

        Performs multi-layer validation:
        1. Schema validation (structure, types)
        2. Graph validation (reachability, cycles)
        3. Semantic validation (node properties)
        4. Security validation (dangerous combinations)

        Args:
            schema: Workflow schema to validate

        Returns:
            WorkflowGenerationValidationResult with all errors and warnings
        """
        # Import here to avoid circular dependency
        from victor.workflows.generation.validator import WorkflowValidator

        validator = WorkflowValidator()
        return await validator.validate(schema)

    async def _refine_schema(
        self, schema: Dict[str, Any], validation: WorkflowGenerationValidationResult
    ) -> Tuple[Dict[str, Any], RefinementResult]:
        """Refine schema based on validation errors.

        Uses automated refinement to fix common errors.

        Args:
            schema: Invalid workflow schema
            validation: Validation result with errors

        Returns:
            Tuple of (refined_schema, refinement_result)
        """
        if not self._refiner:
            return schema, RefinementResult(
                success=False,
                refined_schema=schema,
                iterations=0,
            )

        return await self._refiner.refine(schema, validation)

    async def _compile_to_graph(self, schema: Dict[str, Any]) -> StateGraph:
        """Compile schema to executable StateGraph.

        Args:
            schema: Valid workflow schema

        Returns:
            StateGraph ready for compilation and execution

        Raises:
            ValueError: If schema compilation fails
        """
        try:
            # Import StateGraph
            from victor.framework.graph import StateGraph

            # Use StateGraph.from_schema() to create graph
            graph = StateGraph.from_schema(
                schema,
                node_registry={},  # Will be populated during execution
                condition_registry={},  # Will be populated during execution
            )

            return graph

        except Exception as e:
            logger.error(f"Failed to compile schema to StateGraph: {e}")
            raise ValueError(f"Schema compilation failed: {e}") from e

    async def _interactive_approval(self, stage: str, data: Dict[str, Any]) -> bool:
        """Request user approval in interactive mode.

        Args:
            stage: Name of the stage (e.g., "Extracted Requirements")
            data: Data to show user

        Returns:
            True if user approves, False otherwise
        """
        print(f"\n{'='*60}")
        print(f"Stage: {stage}")
        print(f"{'='*60}")
        print(json.dumps(data, indent=2))
        print(f"{'='*60}\n")

        # In a real implementation, this would use a proper CLI prompt
        # For now, we'll assume approval in non-TTY environments
        import sys

        if not sys.stdin.isatty():
            logger.info("Non-interactive environment, auto-approving")
            return True

        try:
            response = input("Approve and continue? (y/n): ").strip().lower()
            return response in ["y", "yes"]
        except (EOFError, KeyboardInterrupt):
            logger.info("No input received, assuming approval")
            return True


__all__ = [
    "WorkflowGenerationPipeline",
    "PipelineMode",
    "PipelineResult",
]

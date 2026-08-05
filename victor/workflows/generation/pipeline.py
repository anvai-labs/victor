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

"""End-to-end pipeline for requirement extraction and validation.

This module provides the complete pipeline for extracting workflow requirements
from natural language, validating them, and resolving ambiguities.

Design Principles (SOLID):
    - SRP: Pipeline orchestrates but doesn't implement extraction/validation
    - OCP: Extensible with new extraction/validation strategies
    - LSP: All components implement standard interfaces
    - ISP: Focused pipeline interface
    - DIP: Depends on abstractions (Extractor, Validator, Clarifier)

Key Features:
    - Hybrid Extraction: LLM + rule-based with fallback
    - Validation: Completeness, consistency, feasibility checks
    - Ambiguity Resolution: Interactive or assumption-based
    - Error Handling: Graceful degradation with informative errors

Example:
    from victor.workflows.generation.pipeline import RequirementPipeline

    pipeline = RequirementPipeline(orchestrator)
    requirements = await pipeline.extract_and_validate(
        "Analyze code, find bugs, fix them, run tests"
    )

    if requirements:
        print(f"Extracted {len(requirements.functional.tasks)} tasks")
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from victor.workflows.generation.clarifier import (
    AmbiguityDetector,
    AmbiguityResolver,
    InteractiveClarifier,
)
from victor.workflows.generation.extractor import RequirementExtractor
from victor.workflows.generation.requirements import (
    ExtractionMetadata,
    WorkflowRequirements,
)
from victor.workflows.generation.rule_extractor import RuleBasedExtractor

if TYPE_CHECKING:
    from victor.framework.protocols import (
        FrameworkOrchestratorProtocol as OrchestratorProtocol,
    )

logger = logging.getLogger(__name__)


# =============================================================================
# Main Pipeline
# =============================================================================


class RequirementPipeline:
    """End-to-end pipeline for requirement extraction and validation.

    Coordinates:
    1. Extraction (LLM or rule-based)
    2. Validation (completeness, consistency, feasibility)
    3. Ambiguity resolution (interactive or assumptions)

    Attributes:
        _orchestrator: Orchestrator for LLM access
        _extractor: LLM-based extractor
        _rule_extractor: Rule-based fallback extractor
        _clarifier: Interactive clarification system
        _detector: Ambiguity detector
        _resolver: Ambiguity resolver

    Example:
        pipeline = RequirementPipeline(orchestrator)

        # Auto-resolve ambiguities
        requirements = await pipeline.extract_and_validate(
            "Analyze code and fix bugs",
            resolve_ambiguities="assumptions"
        )

        # Interactive clarification
        requirements = await pipeline.extract_and_validate_interactive(
            "Research AI trends and create report"
        )
    """

    def __init__(
        self,
        orchestrator: OrchestratorProtocol,
        enable_llm_extraction: bool = True,
        enable_rule_fallback: bool = True,
    ):
        """Initialize requirement pipeline.

        Args:
            orchestrator: Orchestrator for LLM access
            enable_llm_extraction: Use LLM-based extraction
            enable_rule_fallback: Use rule-based fallback
        """
        self._orchestrator = orchestrator
        self._enable_llm = enable_llm_extraction
        self._enable_rules = enable_rule_fallback

        # Initialize components
        if self._enable_llm:
            self._extractor = RequirementExtractor(orchestrator)

        if self._enable_rules:
            self._rule_extractor = RuleBasedExtractor()

        self._detector = AmbiguityDetector()

        if self._enable_llm:
            self._resolver = AmbiguityResolver(orchestrator)
            self._clarifier = InteractiveClarifier(orchestrator)

    async def extract_and_validate(
        self,
        description: str,
        context: Optional[Dict[str, Any]] = None,
        resolve_ambiguities: str = "assumptions",
        validate: bool = True,
    ) -> Optional[WorkflowRequirements]:
        """Extract and validate requirements from natural language.

        Args:
            description: Natural language workflow description
            context: Optional context (project info, user preferences)
            resolve_ambiguities: How to resolve ambiguities
                - "assumptions": Make reasonable assumptions
                - "defaults": Use template defaults
                - "fail": Fail if ambiguities found
            validate: Whether to validate requirements

        Returns:
            WorkflowRequirements if successful, None if failed

        Raises:
            ValueError: If validation fails and resolve_ambiguities="fail"
        """
        start_time = time.time()

        logger.info(f"Extracting requirements from: {description[:100]}...")

        try:
            # Stage 1: Extract requirements
            requirements = await self._extract_requirements(description, context)

            # Stage 2: Validate (if requested)
            if validate:
                # Import here to avoid circular dependency
                from victor.workflows.generation.validator import RequirementValidator

                validator = RequirementValidator()
                validation_result = validator.validate(requirements)

                if not validation_result.is_valid:
                    logger.error(f"Validation failed with {len(validation_result.errors)} errors")
                    for error in validation_result.errors:
                        logger.error(f"  - {error.severity}: {error.message}")

                    if resolve_ambiguities == "fail":
                        raise ValueError(
                            f"Requirements validation failed: {validation_result.errors}"
                        )

            # Stage 3: Detect and resolve ambiguities
            ambiguities = self._detector.detect(requirements)

            if ambiguities:
                logger.info(f"Found {len(ambiguities)} ambiguities")

                if resolve_ambiguities == "fail":
                    raise ValueError(f"Ambiguities detected: {ambiguities}")

                # Resolve ambiguities
                requirements = await self._resolver.resolve(
                    requirements, ambiguities, strategy=resolve_ambiguities
                )

                # Re-validate after resolution
                if validate:
                    validation_result = validator.validate(requirements)
                    if not validation_result.is_valid:
                        logger.error("Re-validation failed after ambiguity resolution")
                        raise ValueError(f"Requirements still invalid: {validation_result.errors}")

            # Update metadata
            extraction_time = time.time() - start_time
            if requirements.metadata:
                requirements.metadata.extraction_time = extraction_time
                requirements.metadata.ambiguity_count = len(ambiguities)
                requirements.metadata.resolution_strategy = resolve_ambiguities

            logger.info(
                f"Extraction complete in {extraction_time:.2f}s: "
                f"{len(requirements.functional.tasks)} tasks, "
                f"{len(requirements.structural.branches)} branches"
            )

            return requirements

        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            raise

    async def extract_and_validate_interactive(
        self,
        description: str,
        context: Optional[Dict[str, Any]] = None,
        validate: bool = True,
    ) -> Optional[WorkflowRequirements]:
        """Extract and validate with interactive clarification.

        Like extract_and_validate(), but asks user questions to resolve
        ambiguities instead of making assumptions.

        Args:
            description: Natural language workflow description
            context: Optional context
            validate: Whether to validate requirements

        Returns:
            WorkflowRequirements if successful, None if failed
        """
        logger.info("Using interactive clarification mode")

        # Extract requirements
        requirements = await self._extract_requirements(description, context)

        # Validate if requested
        if validate:
            from victor.workflows.generation.validator import RequirementValidator

            validator = RequirementValidator()
            validation_result = validator.validate(requirements)

            if not validation_result.is_valid:
                logger.error(f"Validation failed: {validation_result.errors}")
                # Show errors but don't fail - let user decide
                for error in validation_result.errors:
                    print(f"[{error.severity.upper()}] {error.message}")
                    if error.suggestion:
                        print(f"  Suggestion: {error.suggestion}")

        # Interactive clarification
        requirements = await self._clarifier.clarify(requirements)

        return requirements

    async def _extract_requirements(
        self,
        description: str,
        context: Optional[Dict[str, Any]],
    ) -> WorkflowRequirements:
        """Extract requirements using available methods.

        Tries LLM extraction first, falls back to rule-based if LLM fails.

        Args:
            description: Natural language description
            context: Optional context

        Returns:
            Extracted requirements

        Raises:
            RuntimeError: If all extraction methods fail
        """
        # Try LLM extraction first
        if self._enable_llm:
            try:
                logger.debug("Attempting LLM-based extraction")
                return await self._extractor.extract(description, context)
            except Exception as e:
                logger.warning(f"LLM extraction failed: {e}")

                if not self._enable_rules:
                    raise RuntimeError(f"LLM extraction failed and rule fallback disabled: {e}")

        # Fall back to rule-based extraction
        if self._enable_rules:
            logger.debug("Using rule-based extraction fallback")
            return self._rule_extractor.extract(description)

        raise RuntimeError("All extraction methods failed")

    def extract_sync(
        self,
        description: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> WorkflowRequirements:
        """Synchronous extraction (no validation).

        Convenience method for simple use cases.

        Args:
            description: Natural language description
            context: Optional context

        Returns:
            Extracted requirements
        """
        # Use rule-based extractor (synchronous)
        if self._enable_rules:
            return self._rule_extractor.extract(description)

        raise RuntimeError("Synchronous extraction requires rule-based extractor")


# =============================================================================
# Hybrid Extractor (Advanced)
# =============================================================================


class HybridExtractor:
    """Combine LLM and rule-based extraction with validation.

    This is a more advanced extractor that:
    1. Uses LLM for primary extraction
    2. Validates output with rules
    3. Uses rules to correct errors
    4. Falls back to pure rule-based if LLM fails

    Attributes:
        _orchestrator: Orchestrator for LLM access
        _llm_extractor: LLM-based extractor
        _rule_extractor: Rule-based extractor

    Example:
        extractor = HybridExtractor(orchestrator)
        requirements = await extractor.extract(
            "Analyze code and fix bugs"
        )
    """

    def __init__(self, orchestrator: OrchestratorProtocol):
        """Initialize hybrid extractor.

        Args:
            orchestrator: Orchestrator for LLM access
        """
        self._orchestrator = orchestrator
        self._llm_extractor = RequirementExtractor(orchestrator)
        self._rule_extractor = RuleBasedExtractor()

    async def extract(
        self,
        description: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> WorkflowRequirements:
        """Extract requirements using hybrid approach.

        Strategy:
        1. Use LLM for primary extraction (handles ambiguity well)
        2. Validate LLM output with rules (catch hallucinations)
        3. Use rules to fill missing fields (graceful degradation)
        4. Fallback to pure rule-based if LLM fails

        Args:
            description: Natural language workflow description
            context: Optional context

        Returns:
            Extracted requirements
        """
        try:
            # Try LLM extraction first
            requirements = await self._llm_extractor.extract(description, context)

            # Validate with rules
            validation_errors = self._validate_with_rules(requirements)

            if validation_errors:
                logger.warning(f"LLM output has {len(validation_errors)} validation errors")
                # Use rules to correct errors
                requirements = self._correct_with_rules(requirements, validation_errors)

            return requirements

        except Exception as e:
            logger.warning(f"LLM extraction failed: {e}, falling back to rules")
            # Fallback to rule-based
            return self._rule_extractor.extract(description)

    def _validate_with_rules(self, requirements: WorkflowRequirements) -> List[str]:
        """Validate LLM output with rule-based checks.

        Args:
            requirements: Requirements to validate

        Returns:
            List of error descriptions
        """
        errors = []

        # Check: Tasks must have descriptions
        for task in requirements.functional.tasks:
            if not task.description:
                errors.append(f"Task {task.id} missing description")

        # Check: Branches must have condition
        for branch in requirements.structural.branches:
            if not branch.condition:
                errors.append(f"Branch {branch.condition_id} missing condition")

        # Check: Tool names are valid (basic check)
        known_tools = {
            "bash",
            "code_search",
            "file_read",
            "web_search",
            "git",
            "pytest",
            "npm",
            "docker",
            "kubectl",
        }

        all_tools = set()
        for tools in requirements.functional.tools.values():
            all_tools.update(tools)

        invalid_tools = all_tools - known_tools
        if invalid_tools:
            errors.append(f"Potentially invalid tools: {invalid_tools}")

        return errors

    def _correct_with_rules(
        self,
        requirements: WorkflowRequirements,
        errors: List[str],
    ) -> WorkflowRequirements:
        """Correct LLM output using rules.

        Args:
            requirements: Requirements with errors
            errors: List of error descriptions

        Returns:
            Corrected requirements
        """
        # Apply rule-based corrections
        for error in errors:
            if "missing description" in error:
                # Extract task ID and add description
                # This is simplified - real implementation would be smarter
                pass

            elif "missing condition" in error:
                # Add default condition
                for branch in requirements.structural.branches:
                    if not branch.condition:
                        branch.condition = "success"

        return requirements


__all__ = [
    "RequirementPipeline",
    "HybridExtractor",
]

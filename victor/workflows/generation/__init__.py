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

"""Workflow generation, validation, and refinement system.

This module provides:
    1. Requirement extraction from natural language
    2. Workflow validation (4-layer validation system)
    3. Automated refinement strategies

Public API - Requirement Extraction:
    - RequirementExtractor: LLM-based extraction
    - RuleBasedExtractor: Rule-based fallback extraction
    - RequirementPipeline: End-to-end extraction with validation
    - AmbiguityDetector: Detect missing/unclear requirements
    - AmbiguityResolver: Resolve ambiguities
    - InteractiveClarifier: Interactive clarification system

Public API - Workflow Validation:
    - WorkflowValidator: Main validator with 4-layer validation
    - WorkflowRefiner: Automated refinement strategies
    - ErrorReporter: Error formatting and aggregation

Example - Requirement Extraction:
    from victor.workflows.generation import RequirementPipeline

    pipeline = RequirementPipeline(orchestrator)
    requirements = await pipeline.extract_and_validate(
        "Analyze code, find bugs, fix them, run tests"
    )

    print(f"Extracted {len(requirements.functional.tasks)} tasks")

Example - Workflow Validation:
    from victor.workflows.generation import WorkflowValidator

    validator = WorkflowValidator(strict_mode=True)
    result = validator.validate(workflow_dict)

    if not result.is_valid:
        refiner = WorkflowRefiner(conservative=True)
        refined = refiner.refine(workflow_dict, result)
"""

# Requirements extraction
from victor.workflows.generation.requirements import (
    TaskRequirement,
    InputRequirement,
    OutputRequirement,
    FunctionalRequirements,
    BranchRequirement,
    LoopRequirement,
    StructuralRequirements,
    TaskQualityRequirements,
    QualityRequirements,
    ProjectContext,
    ContextRequirements,
    ExtractionMetadata,
    WorkflowRequirements,
    Ambiguity,
    RequirementValidationResult,
    RequirementValidationError,
)

from victor.workflows.generation.extractor import RequirementExtractor
from victor.workflows.generation.rule_extractor import RuleBasedExtractor
from victor.workflows.generation.pipeline import RequirementPipeline, HybridExtractor
from victor.workflows.generation.clarifier import (
    AmbiguityDetector,
    AmbiguityResolver,
    InteractiveClarifier,
    Question,
    QuestionGenerator,
)

# Workflow validation types
from victor.workflows.generation.types import (
    ErrorSeverity,
    ErrorCategory,
    WorkflowValidationError,
    WorkflowGenerationValidationResult,
    RefinementResult,
    RefinementIteration,
    RefinementHistory,
    WorkflowFix,
)

# Workflow validators
from victor.workflows.generation.validator import (
    WorkflowValidator,
    SchemaValidator,
    GraphStructureValidator,
    SemanticValidator,
    SecurityValidator,
)

# Workflow refiner
from victor.workflows.generation.refiner import (
    WorkflowRefiner,
    SchemaRefiner,
    StructureRefiner,
    SemanticRefiner,
    SecurityRefiner,
)

# Error reporting
from victor.workflows.generation.error_reporter import (
    ErrorReporter,
    ErrorReport,
)

# Workflow generation
from victor.workflows.generation.generator import (
    WorkflowGenerator,
    GenerationStrategy,
    GenerationMetadata,
)
from victor.workflows.generation.templates import (
    TemplateLibrary,
    WorkflowTemplate,
    TemplateType,
)
from victor.workflows.generation.workflow_pipeline import (
    WorkflowGenerationPipeline,
    PipelineMode,
    PipelineResult,
)

# Prompts
from victor.workflows.generation.prompts import (
    RefinementPromptBuilder,
    ExampleLibrary,
    build_refinement_prompt,
)

__all__ = [
    # Requirements extraction types
    "TaskRequirement",
    "InputRequirement",
    "OutputRequirement",
    "FunctionalRequirements",
    "BranchRequirement",
    "LoopRequirement",
    "StructuralRequirements",
    "TaskQualityRequirements",
    "QualityRequirements",
    "ProjectContext",
    "ContextRequirements",
    "ExtractionMetadata",
    "WorkflowRequirements",
    "Ambiguity",
    "RequirementValidationResult",
    "RequirementValidationError",
    # Requirements extraction
    "RequirementExtractor",
    "RuleBasedExtractor",
    "RequirementPipeline",
    "HybridExtractor",
    "AmbiguityDetector",
    "AmbiguityResolver",
    "InteractiveClarifier",
    "Question",
    "QuestionGenerator",
    # Workflow validation types
    "ErrorSeverity",
    "ErrorCategory",
    "WorkflowValidationError",
    "WorkflowGenerationValidationResult",
    "RefinementResult",
    "RefinementIteration",
    "RefinementHistory",
    "WorkflowFix",
    # Workflow validators
    "WorkflowValidator",
    "SchemaValidator",
    "GraphStructureValidator",
    "SemanticValidator",
    "SecurityValidator",
    # Workflow refiner
    "WorkflowRefiner",
    "SchemaRefiner",
    "StructureRefiner",
    "SemanticRefiner",
    "SecurityRefiner",
    # Error reporting
    "ErrorReporter",
    "ErrorReport",
    # Prompts
    "RefinementPromptBuilder",
    "ExampleLibrary",
    "build_refinement_prompt",
    # Workflow generation
    "WorkflowGenerator",
    "GenerationStrategy",
    "GenerationMetadata",
    "TemplateLibrary",
    "WorkflowTemplate",
    "TemplateType",
    "WorkflowGenerationPipeline",
    "PipelineMode",
    "PipelineResult",
]

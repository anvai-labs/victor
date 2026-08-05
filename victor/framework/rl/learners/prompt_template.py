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

"""RL learner for prompt template optimization.

This learner uses multi-armed bandit (Thompson Sampling) to learn
optimal prompt templates and prompt engineering patterns based on
task type, provider, and outcome feedback.

Problem:
- Different tasks may benefit from different prompt structures
- Different providers may respond better to different prompt styles
- Optimal prompt patterns should be learned from outcomes

Strategy:
- State: (task_type, provider, model)
- Arms: Different prompt template variants
- Reward: Task success, quality score, grounding verification

Algorithm: Thompson Sampling with Beta priors

Enrichment Learning:
- Tracks which enrichment types improve task success
- Uses Thompson Sampling to learn optimal enrichment strategies per vertical
- Integrates with PromptEnrichmentService via callback

Sprint 5: Advanced RL Patterns
"""

from victor.core.json_utils import json_dumps
import logging
import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from victor.framework.rl.base import BaseLearner, RLOutcome, RLRecommendation
from victor.core.schema import Tables
from victor.framework.rl.migration import RLTableMigrator

logger = logging.getLogger(__name__)


class PromptStyle(str, Enum):
    """Available prompt styles/templates."""

    CONCISE = "concise"  # Minimal, direct prompts
    DETAILED = "detailed"  # Comprehensive with context
    STRUCTURED = "structured"  # Step-by-step with headers
    COT = "cot"  # Chain-of-thought prompting
    FEW_SHOT = "few_shot"  # With examples
    ROLE_BASED = "role_based"  # "You are an expert..." prefix


class PromptElement(str, Enum):
    """Optional prompt elements that can be included."""

    TASK_CONTEXT = "task_context"  # Explain the broader context
    CONSTRAINTS = "constraints"  # List constraints/requirements
    EXAMPLES = "examples"  # Include examples
    OUTPUT_FORMAT = "output_format"  # Specify expected output format
    THINKING_PROMPT = "thinking_prompt"  # Encourage step-by-step thinking
    VERIFICATION = "verification"  # Ask to verify/validate response
    CONFIDENCE = "confidence"  # Request confidence level


@dataclass
class PromptTemplate:
    """Represents a prompt template configuration.

    Attributes:
        style: The overall prompt style
        elements: Which optional elements to include
        confidence: Confidence in this template (from learning)
        sample_count: Number of times this template was used
    """

    style: PromptStyle = PromptStyle.STRUCTURED
    elements: List[PromptElement] = field(default_factory=list)
    confidence: float = 0.5
    sample_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "style": self.style.value,
            "elements": [e.value for e in self.elements],
            "confidence": self.confidence,
            "sample_count": self.sample_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptTemplate":
        """Create from dictionary."""
        return cls(
            style=PromptStyle(data.get("style", "structured")),
            elements=[PromptElement(e) for e in data.get("elements", [])],
            confidence=data.get("confidence", 0.5),
            sample_count=data.get("sample_count", 0),
        )


@dataclass
class BetaDistribution:
    """Beta distribution for Thompson Sampling.

    Attributes:
        alpha: Number of successes + 1 (prior)
        beta: Number of failures + 1 (prior)
    """

    alpha: float = 1.0  # Prior
    beta: float = 1.0  # Prior
    # Independent per-instance RNG (decoupled from the global random module so
    # concurrent candidates sample independently and tests don't share state).
    _rng: random.Random = field(default_factory=random.Random, repr=False, compare=False)

    def sample(self) -> float:
        """Sample from the beta distribution."""
        return self._rng.betavariate(self.alpha, self.beta)

    def mean(self) -> float:
        """Get the mean of the distribution."""
        return self.alpha / (self.alpha + self.beta)

    def variance(self) -> float:
        """Get the variance of the distribution."""
        total = self.alpha + self.beta
        return (self.alpha * self.beta) / (total * total * (total + 1))

    def update(self, success: bool) -> None:
        """Update distribution with outcome."""
        if success:
            self.alpha += 1.0
        else:
            self.beta += 1.0


class PromptTemplateLearner(BaseLearner):
    """Learn optimal prompt templates per context.

    Uses Thompson Sampling to balance exploration and exploitation
    when selecting prompt templates for different task/provider contexts.

    Attributes:
        name: Always "prompt_template"
        db: SQLite database connection
        exploration_rate: Additional exploration probability
    """

    # Default templates for different task types
    DEFAULT_TEMPLATES: Dict[str, PromptTemplate] = {
        "analysis": PromptTemplate(
            style=PromptStyle.STRUCTURED,
            elements=[PromptElement.TASK_CONTEXT, PromptElement.OUTPUT_FORMAT],
        ),
        "code_generation": PromptTemplate(
            style=PromptStyle.DETAILED,
            elements=[
                PromptElement.CONSTRAINTS,
                PromptElement.OUTPUT_FORMAT,
                PromptElement.VERIFICATION,
            ],
        ),
        "debugging": PromptTemplate(
            style=PromptStyle.COT,
            elements=[PromptElement.THINKING_PROMPT, PromptElement.VERIFICATION],
        ),
        "explanation": PromptTemplate(
            style=PromptStyle.STRUCTURED,
            elements=[PromptElement.TASK_CONTEXT, PromptElement.EXAMPLES],
        ),
        "refactoring": PromptTemplate(
            style=PromptStyle.DETAILED,
            elements=[
                PromptElement.CONSTRAINTS,
                PromptElement.THINKING_PROMPT,
                PromptElement.VERIFICATION,
            ],
        ),
        "default": PromptTemplate(
            style=PromptStyle.STRUCTURED,
            elements=[PromptElement.TASK_CONTEXT, PromptElement.OUTPUT_FORMAT],
        ),
    }

    # All available arms (template configurations)
    ALL_STYLES = list(PromptStyle)
    ALL_ELEMENTS = list(PromptElement)

    # Minimum samples before confident recommendation
    MIN_SAMPLES_FOR_CONFIDENCE = 10

    def __init__(
        self,
        name: str,
        db_connection: Any,
        learning_rate: float = 0.1,
        provider_adapter: Optional[Any] = None,
        exploration_rate: float = 0.1,
    ):
        """Initialize prompt template learner.

        Args:
            name: Learner name (should be "prompt_template")
            db_connection: SQLite database connection
            learning_rate: Not used (Thompson Sampling is Bayesian)
            provider_adapter: Optional provider adapter
            exploration_rate: Additional pure exploration probability
        """
        super().__init__(name, db_connection, learning_rate, provider_adapter)

        self.exploration_rate = exploration_rate

        # Beta distributions for each (context, style) pair
        # Key: (task_type, provider, style) -> BetaDistribution
        self._style_posteriors: Dict[Tuple[str, str, str], BetaDistribution] = {}

        # Beta distributions for each (context, element) pair
        # Key: (task_type, provider, element) -> BetaDistribution
        self._element_posteriors: Dict[Tuple[str, str, str], BetaDistribution] = {}

        # Track recent selections for feedback attribution
        self._recent_selections: Dict[str, PromptTemplate] = {}
        self._max_tracked_selections = 500

        # Sample counts per context
        self._sample_counts: Dict[Tuple[str, str], int] = {}

        # Beta distributions for enrichment types per vertical
        # Key: (vertical, enrichment_type, task_type) -> BetaDistribution
        self._enrichment_posteriors: Dict[Tuple[str, str, str], BetaDistribution] = {}

        # Sample counts for enrichments
        self._enrichment_sample_counts: Dict[Tuple[str, str], int] = {}

        # Load state from database
        self._load_state()

    def _ensure_tables(self) -> None:
        """Migrate legacy per-learner tables to unified RL tables."""
        RLTableMigrator(self.db).run_if_needed(self.name, RLTableMigrator.migrate_prompt_template)

    def _load_state(self) -> None:
        """Load state from database."""
        cursor = self.db.cursor()

        try:
            cursor.execute(
                f"SELECT param_key, param_value, context, sample_count FROM {Tables.RL_PARAM}"
                f" WHERE learner_id = ?",
                (self.name,),
            )
            for row in cursor.fetchall():
                row_dict = dict(row)
                key = row_dict["param_key"]
                value = row_dict["param_value"]
                ctx = row_dict["context"] or ""
                sample_count = row_dict.get("sample_count") or 0
                if value is None:
                    continue

                # Style posteriors: "style_alpha:{style}", context="task_type:provider"
                if key.startswith("style_alpha:") or key.startswith("style_beta:"):
                    is_alpha = key.startswith("style_alpha:")
                    style_val = key[len("style_alpha:") if is_alpha else len("style_beta:") :]
                    # context = "task_type:provider"
                    sep = ctx.rfind(":")
                    if sep < 0:
                        continue
                    task_type, provider = ctx[:sep], ctx[sep + 1 :]
                    posterior_key = (task_type, provider, style_val)
                    existing = self._style_posteriors.get(posterior_key, BetaDistribution())
                    if is_alpha:
                        self._style_posteriors[posterior_key] = BetaDistribution(
                            alpha=value, beta=existing.beta
                        )
                    else:
                        self._style_posteriors[posterior_key] = BetaDistribution(
                            alpha=existing.alpha, beta=value
                        )
                    context_key = (task_type, provider)
                    self._sample_counts[context_key] = max(
                        self._sample_counts.get(context_key, 0), sample_count
                    )

                # Element posteriors: "elem_alpha:{element}", context="task_type:provider"
                elif key.startswith("elem_alpha:") or key.startswith("elem_beta:"):
                    is_alpha = key.startswith("elem_alpha:")
                    elem_val = key[len("elem_alpha:") if is_alpha else len("elem_beta:") :]
                    sep = ctx.rfind(":")
                    if sep < 0:
                        continue
                    task_type, provider = ctx[:sep], ctx[sep + 1 :]
                    posterior_key = (task_type, provider, elem_val)
                    existing = self._element_posteriors.get(posterior_key, BetaDistribution())
                    if is_alpha:
                        self._element_posteriors[posterior_key] = BetaDistribution(
                            alpha=value, beta=existing.beta
                        )
                    else:
                        self._element_posteriors[posterior_key] = BetaDistribution(
                            alpha=existing.alpha, beta=value
                        )

                # Enrichment posteriors: "enrichment_alpha:{type}", context="vertical:task_type"
                elif key.startswith("enrichment_alpha:") or key.startswith("enrichment_beta:"):
                    is_alpha = key.startswith("enrichment_alpha:")
                    enr_type = key[
                        (len("enrichment_alpha:") if is_alpha else len("enrichment_beta:")) :
                    ]
                    sep = ctx.find(":")
                    vertical = ctx[:sep] if sep >= 0 else ctx
                    task_type_part = ctx[sep + 1 :] if sep >= 0 else ""
                    posterior_key = (vertical, enr_type, task_type_part)
                    existing = self._enrichment_posteriors.get(posterior_key, BetaDistribution())
                    if is_alpha:
                        self._enrichment_posteriors[posterior_key] = BetaDistribution(
                            alpha=value, beta=existing.beta
                        )
                    else:
                        self._enrichment_posteriors[posterior_key] = BetaDistribution(
                            alpha=existing.alpha, beta=value
                        )
                    context_key2 = (vertical, enr_type)
                    self._enrichment_sample_counts[context_key2] = max(
                        self._enrichment_sample_counts.get(context_key2, 0),
                        sample_count,
                    )

        except Exception as e:
            logger.debug(f"RL: Could not load prompt template state: {e}")

        if self._style_posteriors:
            logger.info(
                f"RL: Loaded prompt template posteriors for "
                f"{len(self._style_posteriors)} style contexts"
            )

        if self._enrichment_posteriors:
            logger.info(
                f"RL: Loaded enrichment posteriors for "
                f"{len(self._enrichment_posteriors)} enrichment contexts"
            )

    def _get_context_key(self, task_type: str, provider: str) -> Tuple[str, str]:
        """Get context key for lookup."""
        return (task_type.lower(), provider.lower())

    def _get_style_posterior(
        self, task_type: str, provider: str, style: PromptStyle
    ) -> BetaDistribution:
        """Get or create posterior for a style."""
        key = (task_type.lower(), provider.lower(), style.value)
        if key not in self._style_posteriors:
            self._style_posteriors[key] = BetaDistribution()
        return self._style_posteriors[key]

    def _get_element_posterior(
        self, task_type: str, provider: str, element: PromptElement
    ) -> BetaDistribution:
        """Get or create posterior for an element."""
        key = (task_type.lower(), provider.lower(), element.value)
        if key not in self._element_posteriors:
            self._element_posteriors[key] = BetaDistribution()
        return self._element_posteriors[key]

    def record_outcome(self, outcome: RLOutcome) -> None:
        """Record prompt template outcome and update posteriors.

        Expected metadata:
        - session_id: For matching to selection
        - template_used: Optional explicit template that was used
        - grounding_score: Optional grounding verification score

        Args:
            outcome: Outcome with prompt usage data
        """
        task_type = outcome.task_type
        provider = outcome.provider
        model = outcome.model

        # Get template that was used
        session_id = outcome.metadata.get("session_id", "")
        template_dict = outcome.metadata.get("template_used")

        if template_dict:
            template = PromptTemplate.from_dict(template_dict)
        elif session_id in self._recent_selections:
            template = self._recent_selections[session_id]
        else:
            # No template tracking, use default
            default_key = task_type if task_type in self.DEFAULT_TEMPLATES else "default"
            template = self.DEFAULT_TEMPLATES.get(default_key, self.DEFAULT_TEMPLATES["default"])

        # Compute success signal
        success_score = self._compute_success_score(outcome)
        success = success_score >= 0.5

        # Update style posterior
        style_posterior = self._get_style_posterior(task_type, provider, template.style)
        style_posterior.update(success)

        # Update element posteriors
        for element in template.elements:
            element_posterior = self._get_element_posterior(task_type, provider, element)
            element_posterior.update(success)

        # Update sample count
        context_key = self._get_context_key(task_type, provider)
        self._sample_counts[context_key] = self._sample_counts.get(context_key, 0) + 1

        # Save to database
        self._save_to_db(task_type, provider, model, template, success_score)

        logger.debug(
            f"RL: Prompt template updated for {task_type}/{provider}, "
            f"style={template.style.value}, success={success}"
        )

    def _compute_success_score(self, outcome: RLOutcome) -> float:
        """Compute success score from outcome.

        Args:
            outcome: RLOutcome to evaluate

        Returns:
            Success score between 0 and 1
        """
        # Combine multiple signals
        task_success = 1.0 if outcome.success else 0.0
        quality_score = outcome.quality_score

        grounding_score = outcome.metadata.get("grounding_score", 0.5)
        if isinstance(grounding_score, dict):
            grounding_score = grounding_score.get("overall", 0.5)

        # Weighted combination
        return 0.4 * task_success + 0.4 * quality_score + 0.2 * grounding_score

    def _save_to_db(
        self,
        task_type: str,
        provider: str,
        model: Optional[str],
        template: PromptTemplate,
        success: float,
    ) -> None:
        """Save posteriors and history to database."""
        cursor = self.db.cursor()
        timestamp = datetime.now().isoformat()
        context_key = self._get_context_key(task_type, provider)
        sample_count = self._sample_counts.get(context_key, 0)
        ctx_str = f"{task_type.lower()}:{provider.lower()}"

        # Save style posterior to rl_param
        style_posterior = self._get_style_posterior(task_type, provider, template.style)
        for prefix, value in (
            ("style_alpha", style_posterior.alpha),
            ("style_beta", style_posterior.beta),
        ):
            cursor.execute(
                f"""
                INSERT OR REPLACE INTO {Tables.RL_PARAM}
                (learner_id, param_key, param_value, context, sample_count, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    self.name,
                    f"{prefix}:{template.style.value}",
                    value,
                    ctx_str,
                    sample_count,
                    timestamp,
                ),
            )

        # Save element posteriors to rl_param
        for element in template.elements:
            element_posterior = self._get_element_posterior(task_type, provider, element)
            for prefix, value in (
                ("elem_alpha", element_posterior.alpha),
                ("elem_beta", element_posterior.beta),
            ):
                cursor.execute(
                    f"""
                    INSERT OR REPLACE INTO {Tables.RL_PARAM}
                    (learner_id, param_key, param_value, context, sample_count, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        self.name,
                        f"{prefix}:{element.value}",
                        value,
                        ctx_str,
                        sample_count,
                        timestamp,
                    ),
                )

        # Save history to rl_transition
        cursor.execute(
            f"""
            INSERT INTO {Tables.RL_TRANSITION}
            (learner_id, from_state, to_state, action, reward, metadata, created_at)
            VALUES (?, ?, '', ?, ?, ?, ?)
            """,
            (
                self.name,
                task_type.lower(),
                json_dumps(template.to_dict()),
                success,
                json_dumps({"provider": provider.lower(), "model": model}),
                timestamp,
            ),
        )

        self.db.commit()

    def get_recommendation(
        self,
        provider: str,
        model: str,
        task_type: str,
        session_id: Optional[str] = None,
    ) -> Optional[RLRecommendation]:
        """Get recommended prompt template for context.

        Uses Thompson Sampling to select style and elements.

        Args:
            provider: Provider name
            model: Model name
            task_type: Task type
            session_id: Optional session ID for tracking

        Returns:
            Recommendation with PromptTemplate dictionary
        """
        context_key = self._get_context_key(task_type, provider)
        sample_count = self._sample_counts.get(context_key, 0)

        # Check for pure exploration
        if self._rng.random() < self.exploration_rate:
            template = self._random_template()
            confidence = 0.3
            reason = "Exploration: random template"
        else:
            # Thompson Sampling for style
            style_samples = {}
            for style in self.ALL_STYLES:
                posterior = self._get_style_posterior(task_type, provider, style)
                style_samples[style] = posterior.sample()

            best_style = max(style_samples, key=style_samples.get)

            # Thompson Sampling for elements
            selected_elements = []
            for element in self.ALL_ELEMENTS:
                posterior = self._get_element_posterior(task_type, provider, element)
                if posterior.sample() > 0.5:  # Include if sampled value > threshold
                    selected_elements.append(element)

            # Limit elements to avoid overwhelming prompts
            if len(selected_elements) > 4:
                # Keep highest-sampled elements
                element_samples = {
                    e: self._get_element_posterior(task_type, provider, e).sample()
                    for e in selected_elements
                }
                selected_elements = sorted(element_samples, key=element_samples.get, reverse=True)[
                    :4
                ]

            template = PromptTemplate(
                style=best_style,
                elements=selected_elements,
                sample_count=sample_count,
            )

            # Compute confidence from posterior variance
            best_posterior = self._get_style_posterior(task_type, provider, best_style)
            confidence = self._compute_confidence(best_posterior, sample_count)
            reason = f"Thompson Sampling: {best_style.value} with {len(selected_elements)} elements"

        # Track selection for feedback
        if session_id:
            self._track_selection(session_id, template)

        # Determine if baseline
        is_baseline = sample_count < self.MIN_SAMPLES_FOR_CONFIDENCE

        return RLRecommendation(
            value=template.to_dict(),
            confidence=confidence,
            reason=reason,
            sample_size=sample_count,
            is_baseline=is_baseline,
        )

    def _random_template(self) -> PromptTemplate:
        """Generate a random template for exploration."""
        style = self._rng.choice(self.ALL_STYLES)
        # Random subset of elements (0-4)
        num_elements = self._rng.randint(0, 4)
        elements = self._rng.sample(self.ALL_ELEMENTS, k=num_elements)
        return PromptTemplate(style=style, elements=elements)

    def _compute_confidence(self, posterior: BetaDistribution, sample_count: int) -> float:
        """Compute confidence from posterior and samples.

        Args:
            posterior: Beta posterior for best arm
            sample_count: Total samples for this context

        Returns:
            Confidence score between 0.3 and 0.95
        """
        # Base confidence from sample count
        sample_confidence = 1 - math.exp(-sample_count / 20)

        # Posterior concentration (inverse of variance)
        variance = posterior.variance()
        posterior_confidence = 1 - min(1.0, variance * 4)

        # Combine
        confidence = 0.3 + 0.65 * (0.6 * sample_confidence + 0.4 * posterior_confidence)
        return min(0.95, max(0.3, confidence))

    def _track_selection(self, session_id: str, template: PromptTemplate) -> None:
        """Track template selection for later feedback.

        Args:
            session_id: Session identifier
            template: Selected template
        """
        self._recent_selections[session_id] = template

        # Limit tracked selections
        if len(self._recent_selections) > self._max_tracked_selections:
            # Remove oldest entries
            oldest = list(self._recent_selections.keys())[: self._max_tracked_selections // 2]
            for key in oldest:
                del self._recent_selections[key]

    def get_template(
        self, task_type: str, provider: str, session_id: Optional[str] = None
    ) -> PromptTemplate:
        """Get template for context (convenience method).

        Args:
            task_type: Task type
            provider: Provider name
            session_id: Optional session ID for tracking

        Returns:
            PromptTemplate instance
        """
        rec = self.get_recommendation(provider, "", task_type, session_id)
        if rec and rec.value:
            return PromptTemplate.from_dict(rec.value)
        return self.DEFAULT_TEMPLATES.get(task_type, self.DEFAULT_TEMPLATES["default"])

    def get_style_probabilities(self, task_type: str, provider: str) -> Dict[str, float]:
        """Get posterior mean probabilities for each style.

        Args:
            task_type: Task type
            provider: Provider name

        Returns:
            Dictionary of style -> probability
        """
        return {
            style.value: self._get_style_posterior(task_type, provider, style).mean()
            for style in self.ALL_STYLES
        }

    def get_element_probabilities(self, task_type: str, provider: str) -> Dict[str, float]:
        """Get posterior mean probabilities for each element.

        Args:
            task_type: Task type
            provider: Provider name

        Returns:
            Dictionary of element -> probability
        """
        return {
            element.value: self._get_element_posterior(task_type, provider, element).mean()
            for element in self.ALL_ELEMENTS
        }

    def _compute_reward(self, outcome: RLOutcome) -> float:
        """Compute reward signal from outcome.

        Args:
            outcome: Outcome to compute reward for

        Returns:
            Reward value (0 to 1)
        """
        return self._compute_success_score(outcome)

    # =========================================================================
    # Enrichment Tracking Methods
    # =========================================================================

    def _get_enrichment_posterior(
        self, vertical: str, enrichment_type: str, task_type: str = ""
    ) -> BetaDistribution:
        """Get or create posterior for an enrichment type.

        Args:
            vertical: Vertical name (coding, research, etc.)
            enrichment_type: Type of enrichment (knowledge_graph, web_search, etc.)
            task_type: Optional task type for more specific learning

        Returns:
            BetaDistribution for this enrichment context
        """
        key = (vertical.lower(), enrichment_type.lower(), task_type.lower())
        if key not in self._enrichment_posteriors:
            self._enrichment_posteriors[key] = BetaDistribution()
        return self._enrichment_posteriors[key]

    def record_enrichment_outcome(
        self,
        vertical: str,
        enrichment_type: str,
        enrichment_count: int,
        task_success: bool,
        quality_improvement: float,
        task_type: Optional[str] = None,
    ) -> None:
        """Record enrichment outcome and update posteriors.

        This method is designed to be used as a callback from
        PromptEnrichmentService.on_outcome().

        Args:
            vertical: Vertical name (coding, research, devops, data_analysis)
            enrichment_type: Type of enrichment applied
            enrichment_count: Number of enrichments applied
            task_success: Whether the task succeeded
            quality_improvement: Quality improvement score (-1.0 to 1.0)
            task_type: Optional specific task type
        """
        task_type = task_type or ""

        # Only update if enrichments were actually used
        if enrichment_count == 0:
            return

        # Compute success signal
        # Combine task success and quality improvement
        success_score = 0.6 * (1.0 if task_success else 0.0) + 0.4 * (
            (quality_improvement + 1.0) / 2.0
        )
        success = success_score >= 0.5

        # Update enrichment posterior
        posterior = self._get_enrichment_posterior(vertical, enrichment_type, task_type)
        posterior.update(success)

        # Update sample count
        context_key = (vertical.lower(), enrichment_type.lower())
        self._enrichment_sample_counts[context_key] = (
            self._enrichment_sample_counts.get(context_key, 0) + 1
        )

        # Save to database
        self._save_enrichment_to_db(
            vertical=vertical,
            enrichment_type=enrichment_type,
            task_type=task_type,
            enrichment_count=enrichment_count,
            task_success=task_success,
            quality_improvement=quality_improvement,
        )

        logger.debug(
            f"RL: Enrichment outcome recorded: vertical={vertical}, "
            f"type={enrichment_type}, success={success}"
        )

    def _save_enrichment_to_db(
        self,
        vertical: str,
        enrichment_type: str,
        task_type: str,
        enrichment_count: int,
        task_success: bool,
        quality_improvement: float,
    ) -> None:
        """Save enrichment stats and history to database.

        Args:
            vertical: Vertical name
            enrichment_type: Type of enrichment
            task_type: Task type
            enrichment_count: Number of enrichments
            task_success: Whether task succeeded
            quality_improvement: Quality improvement score
        """
        cursor = self.db.cursor()
        timestamp = datetime.now().isoformat()
        context_key = (vertical.lower(), enrichment_type.lower())
        sample_count = self._enrichment_sample_counts.get(context_key, 0)
        ctx_str = f"{vertical.lower()}:{task_type.lower()}"

        # Get posterior values
        posterior = self._get_enrichment_posterior(vertical, enrichment_type, task_type)

        # Save enrichment posteriors to rl_param
        for prefix, value in (
            ("enrichment_alpha", posterior.alpha),
            ("enrichment_beta", posterior.beta),
        ):
            cursor.execute(
                f"""
                INSERT OR REPLACE INTO {Tables.RL_PARAM}
                (learner_id, param_key, param_value, context, sample_count, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    self.name,
                    f"{prefix}:{enrichment_type.lower()}",
                    value,
                    ctx_str,
                    sample_count,
                    timestamp,
                ),
            )

        # Save enrichment history to rl_transition
        cursor.execute(
            f"""
            INSERT INTO {Tables.RL_TRANSITION}
            (learner_id, from_state, to_state, action, reward, metadata, created_at)
            VALUES (?, ?, '', ?, ?, ?, ?)
            """,
            (
                self.name,
                f"{vertical.lower()}:{task_type.lower()}",
                enrichment_type.lower(),
                quality_improvement,
                json_dumps({"enrichment_count": enrichment_count, "task_success": task_success}),
                timestamp,
            ),
        )

        self.db.commit()

    def get_enrichment_probabilities(self, vertical: str, task_type: str = "") -> Dict[str, float]:
        """Get posterior mean probabilities for each enrichment type.

        Args:
            vertical: Vertical name
            task_type: Optional task type for filtering

        Returns:
            Dictionary of enrichment_type -> probability
        """
        probabilities = {}
        for key, posterior in self._enrichment_posteriors.items():
            v, etype, ttype = key
            if v == vertical.lower():
                if not task_type or ttype == task_type.lower():
                    probabilities[etype] = posterior.mean()
        return probabilities

    def get_enrichment_recommendation(self, vertical: str, task_type: str = "") -> Dict[str, bool]:
        """Get recommendation on which enrichment types to use.

        Uses Thompson Sampling to decide which enrichments are worth applying.

        Args:
            vertical: Vertical name
            task_type: Optional task type

        Returns:
            Dictionary of enrichment_type -> should_use (True/False)
        """
        recommendations = {}
        enrichment_types = [
            "knowledge_graph",
            "code_snippet",
            "conversation",
            "web_search",
            "schema",
            "tool_history",
            "project_context",
        ]

        for etype in enrichment_types:
            posterior = self._get_enrichment_posterior(vertical, etype, task_type)
            # Use Thompson Sampling: sample from posterior
            sample = posterior.sample()
            recommendations[etype] = sample > 0.5

        return recommendations

    def create_enrichment_callback(self):
        """Create a callback function for PromptEnrichmentService.

        Returns:
            Callback function that can be passed to service.on_outcome()
        """
        from victor.framework.enrichment import EnrichmentOutcome

        def callback(outcome: EnrichmentOutcome) -> None:
            self.record_enrichment_outcome(
                vertical=outcome.vertical or "unknown",
                enrichment_type=outcome.enrichment_type,
                enrichment_count=outcome.enrichment_count,
                task_success=outcome.task_success,
                quality_improvement=outcome.quality_improvement,
                task_type=outcome.task_type,
            )

        return callback

    def export_metrics(self) -> Dict[str, Any]:
        """Export learner metrics for monitoring.

        Returns:
            Dictionary with learner stats
        """
        total_samples = sum(self._sample_counts.values())

        # Get top styles per task type
        top_styles = {}
        seen_contexts = set()

        for key, posterior in self._style_posteriors.items():
            task_type, provider, style = key
            context = f"{task_type}:{provider}"

            if context not in seen_contexts:
                seen_contexts.add(context)
                context_styles = {
                    k[2]: v.mean()
                    for k, v in self._style_posteriors.items()
                    if k[0] == task_type and k[1] == provider
                }
                if context_styles:
                    best = max(context_styles, key=context_styles.get)
                    top_styles[context] = {
                        "best_style": best,
                        "probability": context_styles[best],
                    }

        # Get enrichment stats
        enrichment_stats = {}
        for key, posterior in self._enrichment_posteriors.items():
            vertical, etype, ttype = key
            context = f"{vertical}:{etype}"
            if context not in enrichment_stats:
                enrichment_stats[context] = {
                    "probability": posterior.mean(),
                    "sample_count": self._enrichment_sample_counts.get((vertical, etype), 0),
                }

        return {
            "learner": self.name,
            "contexts_learned": len(self._sample_counts),
            "total_samples": total_samples,
            "exploration_rate": self.exploration_rate,
            "style_posteriors_count": len(self._style_posteriors),
            "element_posteriors_count": len(self._element_posteriors),
            "tracked_selections": len(self._recent_selections),
            "top_styles_by_context": top_styles,
            "samples_per_context": {f"{k[0]}:{k[1]}": v for k, v in self._sample_counts.items()},
            # Enrichment stats
            "enrichment_posteriors_count": len(self._enrichment_posteriors),
            "enrichment_samples_total": sum(self._enrichment_sample_counts.values()),
            "enrichment_stats": enrichment_stats,
        }

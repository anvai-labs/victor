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

"""Specialized RL learners for different parameter types."""

from victor.framework.rl.learners.context_pruning import ContextPruningLearner
from victor.framework.rl.learners.continuation_patience import (
    ContinuationPatienceLearner,
)
from victor.framework.rl.learners.continuation_prompts import ContinuationPromptLearner
from victor.framework.rl.learners.cross_vertical import CrossVerticalLearner
from victor.framework.rl.learners.model_selector import ModelSelectorLearner
from victor.framework.rl.learners.semantic_threshold import SemanticThresholdLearner
from victor.framework.rl.learners.workflow_execution import WorkflowExecutionLearner

__all__ = [
    "ContextPruningLearner",
    "ContinuationPatienceLearner",
    "ContinuationPromptLearner",
    "CrossVerticalLearner",
    "ModelSelectorLearner",
    "SemanticThresholdLearner",
    "WorkflowExecutionLearner",
]

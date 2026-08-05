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

"""Agent definition and ensemble orchestration system.

Victor's agent system uses a protocol-based, composable DSL inspired by:
- Prefect/Airflow: Task DAG composition
- Kubernetes: Declarative YAML specifications
- Pydantic: Type-safe validation
- Protocol-oriented design: Behavior contracts over inheritance

Key Concepts:
- AgentSpec: Declarative agent specification (YAML/JSON/Python)
- AgentCapabilities: What an agent can do (tools, skills)
- AgentConstraints: Limits and requirements (tokens, cost, time)
- Ensemble: Coordinated group of agents working together
- Pipeline: Sequential or parallel agent execution

Example (Python DSL):
    from victor.agent.specs import AgentSpec, Ensemble, Pipeline

    researcher = AgentSpec(
        name="researcher",
        description="Gathers information and analyzes data",
        capabilities=["web_search", "code_search", "read"],
        model_preference="reasoning",  # Let system pick best model
    )

    coder = AgentSpec(
        name="coder",
        description="Writes and modifies code",
        capabilities=["edit", "write", "shell"],
        model_preference="coding",
    )

    # Create a pipeline: research -> code
    pipeline = Pipeline([researcher, coder])
    result = await pipeline.execute(task="Add user authentication")

Example (YAML):
    agents:
      - name: researcher
        description: Gathers information
        capabilities: [web_search, code_search]

      - name: coder
        description: Writes code
        capabilities: [edit, write]

    ensemble:
      type: pipeline
      agents: [researcher, coder]
"""

from victor.agent.specs.models import (
    AgentSpec,
    AgentCapabilities,
    AgentConstraints,
    ModelPreference,
)
from victor.agent.specs.ensemble import (
    Ensemble,
    EnsembleType,
    Pipeline,
    Parallel,
    Hierarchical,
)
from victor.agent.specs.presets import (
    researcher_agent,
    coder_agent,
    reviewer_agent,
    devops_agent,
    analyst_agent,
)
from victor.agent.specs.loader import (
    load_agents_from_yaml,
    load_agents_from_dict,
)
from victor.agent.specs.converter import (
    EnsembleConverter,
    ensemble_to_workflow,
    workflow_to_ensemble,
)

__all__ = [
    # Specifications
    "AgentSpec",
    "AgentCapabilities",
    "AgentConstraints",
    "ModelPreference",
    # Ensemble patterns
    "Ensemble",
    "EnsembleType",
    "Pipeline",
    "Parallel",
    "Hierarchical",
    # Presets
    "researcher_agent",
    "coder_agent",
    "reviewer_agent",
    "devops_agent",
    "analyst_agent",
    # Loaders
    "load_agents_from_yaml",
    "load_agents_from_dict",
    # Converters
    "EnsembleConverter",
    "ensemble_to_workflow",
    "workflow_to_ensemble",
]

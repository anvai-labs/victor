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

"""YAML/JSON loader for agent configurations.

Supports loading agent definitions and ensembles from configuration files.

Example YAML format:
```yaml
agents:
  - name: researcher
    description: Gathers information
    capabilities:
      tools: [web_search, read, code_search]
      can_browse_web: true
    model_preference: reasoning

  - name: coder
    description: Writes code
    capabilities: [edit, write, shell]
    model_preference: coding

ensemble:
  type: pipeline
  agents: [researcher, coder]
```

Usage:
    config = load_agents_from_yaml("agents.yaml")
    ensemble = config.ensemble
    await ensemble.execute("Add user authentication")
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from victor.agent.specs.models import AgentSpec
from victor.agent.specs.ensemble import (
    Ensemble,
    EnsembleType,
    Pipeline,
    Parallel,
    Hierarchical,
)

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration loaded from file.

    Attributes:
        agents: Dictionary of agent specs by name
        ensemble: Optional ensemble configuration
        metadata: Additional configuration metadata
    """

    agents: Dict[str, AgentSpec] = field(default_factory=dict)
    ensemble: Optional[Ensemble] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_agent(self, name: str) -> AgentSpec:
        """Get an agent by name.

        Args:
            name: Agent name

        Returns:
            AgentSpec

        Raises:
            KeyError: If agent not found
        """
        if name not in self.agents:
            raise KeyError(f"Agent '{name}' not found. Available: {list(self.agents.keys())}")
        return self.agents[name]


def load_agents_from_yaml(
    path: Union[str, Path],
    include_presets: bool = True,
) -> AgentConfig:
    """Load agent configuration from YAML file.

    Args:
        path: Path to YAML file
        include_presets: Include preset agents in lookup

    Returns:
        AgentConfig with loaded agents and ensemble
    """
    import yaml

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Agent config file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    return load_agents_from_dict(data, include_presets=include_presets)


def load_agents_from_dict(
    data: Dict[str, Any],
    include_presets: bool = True,
) -> AgentConfig:
    """Load agent configuration from dictionary.

    Args:
        data: Configuration dictionary
        include_presets: Include preset agents in lookup

    Returns:
        AgentConfig with loaded agents and ensemble
    """
    from victor.agent.specs.presets import get_preset_agent, list_preset_agents

    config = AgentConfig()

    # Load preset agents if requested
    if include_presets:
        for preset_name in list_preset_agents():
            config.agents[preset_name] = get_preset_agent(preset_name)

    # Load custom agents
    agents_data = data.get("agents", [])
    for agent_data in agents_data:
        # Check if extending a preset
        extends = agent_data.pop("extends", None)
        if extends:
            base_agent = config.agents.get(extends)
            if not base_agent:
                raise ValueError(f"Cannot extend unknown agent: {extends}")

            # Merge with base
            agent = _merge_agent_spec(base_agent, agent_data)
        else:
            agent = AgentSpec.from_dict(agent_data)

        config.agents[agent.name] = agent

    # Load ensemble configuration
    ensemble_data = data.get("ensemble")
    if ensemble_data:
        config.ensemble = _build_ensemble(ensemble_data, config.agents)

    # Store metadata
    config.metadata = {k: v for k, v in data.items() if k not in ("agents", "ensemble")}

    return config


def _merge_agent_spec(base: AgentSpec, overrides: Dict[str, Any]) -> AgentSpec:
    """Merge overrides into a base agent spec.

    Args:
        base: Base agent spec
        overrides: Override values

    Returns:
        New merged AgentSpec
    """
    # Start with base as dict
    base_dict = base.to_dict()

    # Merge capabilities
    if "capabilities" in overrides:
        caps_override = overrides["capabilities"]
        if isinstance(caps_override, list):
            # Add to existing tools
            base_dict["capabilities"]["tools"] = list(
                set(base_dict["capabilities"]["tools"]) | set(caps_override)
            )
        elif isinstance(caps_override, dict):
            # Merge capabilities dict
            for key, value in caps_override.items():
                if key == "tools":
                    base_dict["capabilities"]["tools"] = list(
                        set(base_dict["capabilities"]["tools"]) | set(value)
                    )
                else:
                    base_dict["capabilities"][key] = value
        del overrides["capabilities"]

    # Merge constraints
    if "constraints" in overrides:
        base_dict["constraints"].update(overrides["constraints"])
        del overrides["constraints"]

    # Override other fields
    for key, value in overrides.items():
        if key == "tags" and key in base_dict:
            base_dict[key] = list(set(base_dict[key]) | set(value))
        else:
            base_dict[key] = value

    return AgentSpec.from_dict(base_dict)


def _build_ensemble(
    data: Dict[str, Any],
    agents: Dict[str, AgentSpec],
) -> Ensemble:
    """Build ensemble from configuration.

    Args:
        data: Ensemble configuration
        agents: Available agents

    Returns:
        Configured Ensemble
    """
    ensemble_type = EnsembleType(data.get("type", "pipeline"))
    agent_names = data.get("agents", [])
    name = data.get("name")

    # Resolve agent names to specs
    resolved_agents: List[AgentSpec] = []
    for agent_name in agent_names:
        if agent_name not in agents:
            raise ValueError(f"Unknown agent in ensemble: {agent_name}")
        resolved_agents.append(agents[agent_name])

    if ensemble_type == EnsembleType.PIPELINE:
        return Pipeline(
            resolved_agents,
            name=name,
            continue_on_error=data.get("continue_on_error", False),
        )
    elif ensemble_type == EnsembleType.PARALLEL:
        return Parallel(
            resolved_agents,
            name=name,
            require_all=data.get("require_all", True),
        )
    elif ensemble_type == EnsembleType.HIERARCHICAL:
        if len(resolved_agents) < 2:
            raise ValueError("Hierarchical ensemble needs manager + workers")
        return Hierarchical(
            manager=resolved_agents[0],
            workers=resolved_agents[1:],
            name=name,
            max_delegations=data.get("max_delegations", 10),
        )
    else:
        raise ValueError(f"Unsupported ensemble type: {ensemble_type}")


def save_agents_to_yaml(
    config: AgentConfig,
    path: Union[str, Path],
    include_presets: bool = False,
) -> None:
    """Save agent configuration to YAML file.

    Args:
        config: Configuration to save
        path: Output path
        include_presets: Include preset agents in output
    """
    import yaml

    from victor.agent.specs.presets import list_preset_agents

    preset_names = set(list_preset_agents()) if not include_presets else set()

    data: Dict[str, Any] = {}

    # Export agents
    agents_data = []
    for name, agent in config.agents.items():
        if name not in preset_names:
            agents_data.append(agent.to_dict())
    if agents_data:
        data["agents"] = agents_data

    # Export ensemble
    if config.ensemble:
        data["ensemble"] = config.ensemble.to_dict()

    # Include metadata
    data.update(config.metadata)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

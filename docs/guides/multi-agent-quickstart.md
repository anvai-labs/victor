# Multi-Agent Teams

**Coordinate multiple specialized agents for complex tasks.**

## Quick Start

```python
from victor.teams import TeamFormation, PersonaTraits, create_coordinator

# Define team members
lead = PersonaTraits(name="Lead", role="architect", expertise_level="EXPERT")
dev1 = PersonaTraits(name="Dev1", role="implementer", expertise_level="SENIOR")
dev2 = PersonaTraits(name="Dev2", role="implementer", expertise_level="SENIOR")
reviewer = PersonaTraits(name="Reviewer", role="reviewer", expertise_level="EXPERT")

# Create team
team = TeamFormation(
    name="code_review_team",
    formation="HUB_SPOKE",
    members=[lead, dev1, dev2, reviewer],
)

# Execute
coordinator = create_coordinator(team)
result = await coordinator.run("Review the authentication module")
```

## Team Topologies

The table below compares communication patterns. The [team execution diagram](../architecture.md#multi-agent-teams)
shows how a coordinator and its formation run inside the compiled workflow engine.

| Topology | Use Case | Communication | Pros | Cons |
|----------|----------|----------------|------|------|
| **HIERARCHY** | Manager-worker | Top-down | Clear chain | Bottleneck |
| **MESH** | Peers | All-to-all | Resilient | Chaotic |
| **PIPELINE** | Sequential | Handoff | Simple | Slow |
| **HUB_SPOKE** | Coordination | Through hub | Balanced | Hub dependency |

## Persona Traits

| Trait | Range | Impact |
|-------|-------|--------|
| `verbosity` | 0.0-1.0 | Response length |
| `risk_tolerance` | 0.0-1.0 | Conservative vs bold |
| `creativity` | 0.0-1.0 | Novel vs standard |
| `formality` | 0.0-1.0 | Casual vs formal |
| `expertise` | NOVICE→EXPERT | Tool selection |

## Pre-configured Teams

| Team | Members | Use Case |
|------|---------|----------|
| `code_review` | architect + 2 developers + reviewer | PR reviews |
| `research` | lead + 2 researchers + synthesizer | Deep research |
| `debug` | investigator + reproducer + fixer | Bug hunting |
| `refactor` | analyst + 2 implementers | Code cleanup |

## Communication Styles

| Style | Description |
|-------|-------------|
| `DIRECT` | Straightforward, concise |
| `DETAILED` | Comprehensive explanations |
| `QUESTIONING` | Asks clarifying questions |
| `COLLABORATIVE` | Seeks input from others |

## Example Teams

### Code Review Team

```python
from victor.framework.multi_agent import TeamSpec, TeamMember

code_review = TeamSpec(
    name="code_review",
    topology=TeamTopology.HUB_SPOKE,
    members=[
        TeamMember(persona=architect, role="coordinator", is_leader=True),
        TeamMember(persona=developer_1, role="reviewer"),
        TeamMember(persona=developer_2, role="reviewer"),
        TeamMember(persona=reviewer, role="final_approver"),
    ],
)
```

### Research Team

```python
research_team = TeamSpec(
    name="research",
    topology=TeamTopology.PIPELINE,
    members=[
        TeamMember(persona=researcher_1, role="source_finder"),
        TeamMember(persona=researcher_2, role="analyst"),
        TeamMember(persona=synthesizer, role="writer"),
    ],
)
```

## Execution Modes

| Mode | Description | When to Use |
|------|-------------|-------------|
| `sequential` | One agent at a time | Dependent tasks |
| `parallel` | All agents at once | Independent tasks |
| `consensus` | Vote on decisions | Quality critical |
| `debate` | Agents discuss | Complex problems |

## Best Practices

| Practice | Description |
|----------|-------------|
| **Start small** | Begin with 2-3 agents |
| **Clear roles** | Define specific responsibilities |
| **Match expertise** | Align skill with task |
| **Limit overhead** | More agents ≠ better results |
| **Use topology** | Match structure to task |

## See Also

- [Multi-Agent Teams Guide](MULTI_AGENT_TEAMS.md) - Comprehensive guide
- [Multi-Agent Teams](MULTI_AGENT_TEAMS.md) - Team patterns

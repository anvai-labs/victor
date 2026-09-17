# Multi-Agent Teams

Create a team through the public `Agent.create_team` factory. `TeamFormation` is an
enum selecting how the team coordinates; it is not a team constructor.

## Quick start

Configure an Ollama server and pull the chosen model before running this example.

```python
import asyncio
from victor.framework import Agent, TeamFormation, TeamMemberSpec

async def main():
    team = await Agent.create_team(
        name="design-review",
        goal="Propose and review a small REST API design",
        provider="ollama",
        model="llama3.1:8b",
        formation=TeamFormation.SEQUENTIAL,
        members=[
            TeamMemberSpec(role="researcher", goal="Find relevant API design practices"),
            TeamMemberSpec(role="reviewer", goal="Review the proposed design for gaps"),
        ],
    )
    result = await team.run()
    print(result.final_output)

asyncio.run(main())
```

## Formations

| Formation | Coordination |
| --- | --- |
| `SEQUENTIAL` | Members run in sequence with shared context |
| `PARALLEL` | Members perform independent work concurrently |
| `HIERARCHICAL` | A supervisor coordinates specialists |
| `PIPELINE` | Each stage supplies input to the next |
| `CONSENSUS` | Members work toward agreement |
| `REFLECTION` | A generator and critic refine a result |

Hierarchical teams coordinate through a supervisor. The factory promotes the first
member when none is explicitly designated. To designate one, use the team member's
`agent_category` field with the corresponding `TeamAgentCategory`; `is_manager`
is a compatibility alias. The shared coordinator validates formation requirements.

## Team configuration

`Agent.create_team` takes `name`, `goal`, and a list of `TeamMemberSpec` objects.
Optional factory arguments include `total_tool_budget`, `max_iterations`,
`timeout_seconds`, and `shared_context`. A member specifies its role and goal;
optional fields include name, tool budget, priority, expertise, and backstory.

`await team.run()` executes the configured goal and returns a `TeamResult`, including
`final_output`. For an existing vertical's named team use the separate named-team API
on a configured agent; do not pass an `AgentTeam` object to `Agent.run_team`.

## Events and workflow integration

Application-facing agent streams use [framework events](../reference/api/python-api.md).
Runtime instrumentation uses asynchronous [topic subscriptions](observability/event-bus.md).
Team workflow nodes execute through the same compiled graph engine as other node types;
see [workflow syntax](../user-guide/yaml_workflow_syntax.md).

The [team architecture diagram](../architecture.md#multi-agent-teams) shows the runtime
boundary. Earlier low-level examples are preserved in [page history](https://github.com/anvai-labs/victor/commits/develop/docs/guides/MULTI_AGENT_TEAMS.md).

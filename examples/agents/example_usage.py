#!/usr/bin/env python3
# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0

"""Example usage of Victor's Agent Ensemble and Entity Memory systems.

This script demonstrates:
1. Creating agents programmatically
2. Building ensembles (Pipeline, Parallel, Hierarchical)
3. Loading agents from YAML
4. Using entity memory for context
5. Integrating with HITL workflow nodes

Run with:
    python examples/agents/example_usage.py
"""

import asyncio
from pathlib import Path

# Agent system imports
from victor.agent.specs import (
    AgentSpec,
    AgentCapabilities,
    AgentConstraints,
    ModelPreference,
    Pipeline,
    Parallel,
    Hierarchical,
    researcher_agent,
    coder_agent,
    reviewer_agent,
    load_agents_from_yaml,
    load_agents_from_dict,
)

# Entity memory imports
from victor.storage.memory import (
    EntityMemory,
    EntityGraph,
    CompositeExtractor,
    EntityType,
)


async def example_programmatic_agents():
    """Create agents programmatically."""
    print("\n=== Programmatic Agent Creation ===\n")

    # Create a custom agent from scratch
    api_developer = AgentSpec(
        name="api_developer",
        description="Specializes in REST API development",
        capabilities=AgentCapabilities(
            tools={"read", "edit", "write", "shell"},
            skills={"api_design", "openapi", "validation"},
            can_execute_code=True,
            can_modify_files=True,
        ),
        constraints=AgentConstraints(
            max_iterations=40,
            max_tool_calls=80,
            timeout_seconds=600.0,
        ),
        model_preference=ModelPreference.CODING,
        system_prompt="""You are an API developer specializing in REST APIs.
Follow OpenAPI standards and implement proper validation.""",
    )

    print(f"Created: {api_developer}")
    print(f"  Tools: {api_developer.capabilities.tools}")
    print(f"  Model: {api_developer.model_preference.value}")

    # Extend a preset agent
    enhanced_coder = coder_agent.with_capabilities(
        tools={"test", "code_review"},
        can_browse_web=True,  # Allow looking up documentation
    )

    print(f"\nExtended coder: {enhanced_coder}")
    print(f"  Original tools + new: {len(enhanced_coder.capabilities.tools)} tools")


async def example_pipeline_ensemble():
    """Create and execute a pipeline ensemble."""
    print("\n=== Pipeline Ensemble ===\n")

    # Create a simple pipeline using presets
    pipeline = Pipeline(
        agents=[researcher_agent, coder_agent, reviewer_agent],
        name="standard_development",
    )

    print(f"Pipeline: {pipeline.name}")
    print(f"  Agents: {[a.name for a in pipeline.agents]}")
    print(f"  Type: {pipeline.ensemble_type.value}")

    # Execute the pipeline
    result = await pipeline.execute(
        task="Add input validation to the user registration form",
        context={"project": "victor", "language": "python"},
    )

    print("\nExecution result:")
    print(f"  Status: {result.status.value}")
    print(f"  Agents completed: {len(result.agent_results)}")
    for ar in result.agent_results:
        print(f"    - {ar.agent_name}: {ar.status.value}")


async def example_parallel_ensemble():
    """Create and execute a parallel ensemble."""
    print("\n=== Parallel Ensemble ===\n")

    # Create custom analysis agents
    security = AgentSpec(
        name="security_check",
        description="Security analysis",
        capabilities=AgentCapabilities(tools={"code_search", "read"}),
    )
    performance = AgentSpec(
        name="perf_check",
        description="Performance analysis",
        capabilities=AgentCapabilities(tools={"read"}),
    )

    parallel = Parallel(
        agents=[security, performance, reviewer_agent],
        name="code_analysis",
        require_all=False,  # Continue even if one fails
    )

    result = await parallel.execute("Analyze the authentication module")

    print("Parallel execution completed:")
    print(f"  Status: {result.status.value}")
    print(f"  Results: {result.final_output}")


async def example_yaml_loading():
    """Load agents from YAML configuration."""
    print("\n=== YAML Configuration Loading ===\n")

    # Load from dictionary (simulating YAML)
    config = load_agents_from_dict(
        {
            "agents": [
                {
                    "name": "custom_researcher",
                    "extends": "researcher",  # Extend preset
                    "constraints": {
                        "max_iterations": 20,
                    },
                },
                {
                    "name": "rapid_coder",
                    "extends": "coder",
                    "model_preference": "fast",
                },
            ],
            "ensemble": {
                "type": "pipeline",
                "agents": ["custom_researcher", "rapid_coder"],
            },
        }
    )

    print(f"Loaded {len(config.agents)} agents")
    for name, agent in config.agents.items():
        if name in ["custom_researcher", "rapid_coder"]:
            print(f"  {name}: {agent.model_preference.value}")

    if config.ensemble:
        result = await config.ensemble.execute("Quick prototype")
        print(f"\nEnsemble result: {result.status.value}")


async def example_entity_memory():
    """Use entity memory for context extraction."""
    print("\n=== Entity Memory ===\n")

    # Create entity memory and extractor
    memory = EntityMemory(session_id="example_session")
    extractor = CompositeExtractor.create_default()

    # Sample code to analyze
    code_sample = """
class UserAuthentication:
    def __init__(self, scanner: SecurityScanner):
        self.scanner = scanner

    async def authenticate(self, username: str, password: str) -> User:
        # Hash password with bcrypt
        hashed = bcrypt.hash(password)
        return await self.db.verify(username, hashed)

from victor.security import SecurityScanner, SecurityPolicy
"""

    # Extract entities
    result = await extractor.extract(code_sample, source="auth.py")

    print(f"Extracted {len(result.entities)} entities:")
    for entity in result.entities:
        print(f"  - {entity.name} ({entity.entity_type.value})")

    # Store in memory
    for entity in result.entities:
        await memory.store(entity)

    # Search entities
    auth_entities = await memory.search("auth", entity_types=[EntityType.CLASS])
    print(f"\nSearch for 'auth': {len(auth_entities)} results")


async def example_entity_graph():
    """Use entity graph for relationship tracking."""
    print("\n=== Entity Graph ===\n")

    graph = EntityGraph(in_memory=True)
    await graph.initialize()

    # Create entities
    from victor.storage.memory import Entity, EntityRelation, RelationType

    module = Entity.create("auth_module", EntityType.MODULE)
    auth_class = Entity.create("UserAuth", EntityType.CLASS)
    scanner = Entity.create("SecurityScanner", EntityType.CLASS)

    await graph.add_entity(module)
    await graph.add_entity(auth_class)
    await graph.add_entity(scanner)

    # Add relationships
    await graph.add_relation(
        EntityRelation(
            source_id=module.id,
            target_id=auth_class.id,
            relation_type=RelationType.CONTAINS,
        )
    )
    await graph.add_relation(
        EntityRelation(
            source_id=auth_class.id,
            target_id=scanner.id,
            relation_type=RelationType.DEPENDS_ON,
        )
    )

    # Query graph
    neighbors = await graph.get_neighbors(module.id, depth=2)
    print(f"Neighbors of {module.name} (depth=2): {len(neighbors)}")
    for entity, relation, depth in neighbors:
        print(f"  [{depth}] {entity.name} via {relation.relation_type.value}")

    # Get statistics
    stats = await graph.get_stats()
    print("\nGraph stats:")
    print(f"  Entities: {stats.entity_count}")
    print(f"  Relations: {stats.relation_count}")


async def example_tree_sitter_extraction():
    """Use Tree-sitter for accurate code parsing."""
    print("\n=== Tree-sitter Code Extraction ===\n")

    from victor.storage.memory import (
        has_tree_sitter,
        create_extractor,
        TreeSitterFileExtractor,
    )
    from pathlib import Path
    import tempfile

    # Check Tree-sitter availability
    if has_tree_sitter():
        print("Tree-sitter is available!")

        # Create Tree-sitter-enabled extractor
        extractor = create_extractor(use_tree_sitter=True)

        # Sample Python code
        code = '''
class OrderProcessor:
    """Processes customer orders."""

    def __init__(self, db: Database):
        self.db = db

    async def process_order(self, order_id: int) -> Order:
        order = await self.db.get_order(order_id)
        validated = self.validate(order)
        return await self.fulfill(validated)

    def validate(self, order: Order) -> Order:
        if not order.items:
            raise ValueError("Empty order")
        return order
'''

        # Write to temp file for extraction
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_path = Path(f.name)

        try:
            result = await extractor.extract(code, source=str(temp_path))

            print(f"Extracted {len(result.entities)} entities:")
            for entity in result.entities:
                print(f"  - {entity.name} ({entity.entity_type.value})")

            if result.relations:
                print(f"\nExtracted {len(result.relations)} relations")
        finally:
            temp_path.unlink(missing_ok=True)

    else:
        print("Tree-sitter not available, using regex extraction")
        extractor = create_extractor(use_tree_sitter=False)


async def example_ensemble_to_workflow():
    """Convert ensemble to workflow for execution."""
    print("\n=== Ensemble to Workflow Conversion ===\n")

    from victor.agent.specs import (
        Pipeline,
        researcher_agent,
        coder_agent,
        ensemble_to_workflow,
        workflow_to_ensemble,
    )

    # Create a pipeline
    pipeline = Pipeline(
        [researcher_agent, coder_agent],
        name="dev_pipeline",
    )

    # Convert to workflow
    workflow = ensemble_to_workflow(
        pipeline,
        add_hitl=True,  # Add HITL approval nodes
    )

    print(f"Workflow: {workflow.name}")
    print(f"Nodes: {len(workflow.nodes)}")
    for node_id, node in workflow.nodes.items():
        print(f"  - {node_id}: {type(node).__name__}")

    # Roundtrip conversion
    recovered = workflow_to_ensemble(workflow)
    print(f"\nRecovered ensemble: {type(recovered).__name__}")
    print(f"  Agents: {[a.name for a in recovered.agents]}")


async def main():
    """Run all examples."""
    print("=" * 60)
    print("Victor Framework Enhancement Examples")
    print("=" * 60)

    await example_programmatic_agents()
    await example_pipeline_ensemble()
    await example_parallel_ensemble()
    await example_yaml_loading()
    await example_entity_memory()
    await example_entity_graph()
    await example_tree_sitter_extraction()
    await example_ensemble_to_workflow()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

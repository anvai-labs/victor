# API Reference (auto-generated)

!!! info "Generated from source docstrings"
    The signatures and docstrings on this page are rendered directly from the code by
    [mkdocstrings](https://mkdocstrings.github.io/) at build time. Signatures follow the
    implementation; docstring explanations and examples still require review when behavior changes. The hand-written pages
    under **API Reference** (Protocols, Providers, Tools, Workflows) provide the prose,
    guidance, and examples that complement these signatures.

## Public framework API

The core abstractions exposed from `victor.framework`.

### Agent

::: victor.framework.agent.Agent
    options:
      show_root_heading: true
      show_source: false
      members_order: source
      filters: ["!^_"]

### AgentBuilder

::: victor.framework.agent_components.AgentBuilder
    options:
      show_root_heading: true
      show_source: false
      filters: ["!^_"]

### StateGraph

::: victor.framework.graph.StateGraph
    options:
      show_root_heading: true
      show_source: false
      members_order: source
      filters: ["!^_"]

### WorkflowEngine

::: victor.framework.workflow_engine.WorkflowEngine
    options:
      show_root_heading: true
      show_source: false
      filters: ["!^_"]

## UI entry point

The only supported entry point for surface layers (CLI, TUI, web chat).

### VictorClient

::: victor.framework.client.VictorClient
    options:
      show_root_heading: true
      show_source: false
      members_order: source
      filters: ["!^_"]

### SessionConfig

::: victor.framework.session_config.SessionConfig
    options:
      show_root_heading: true
      show_source: false
      filters: ["!^_"]

## Extension base classes

Inherit from these to add providers and tools.

### BaseProvider

::: victor.providers.base.BaseProvider
    options:
      show_root_heading: true
      show_source: false
      members_order: source
      filters: ["!^_"]

### BaseTool

::: victor.tools.base.BaseTool
    options:
      show_root_heading: true
      show_source: false
      members_order: source
      filters: ["!^_"]

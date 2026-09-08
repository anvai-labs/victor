# Tool Plugin Development

A `ToolPlugin` packages related tools. Implement the tool itself using the
[custom tool tutorial](../../tutorials/build-custom-tool.md), then expose instances
through `get_tools`.

## Plugin contract

```python
from victor.tools.plugin import ToolPlugin

class Plugin(ToolPlugin):
    name = "my_plugin"
    version = "1.0.0"
    description = "Application-specific tools"

    def get_tools(self):
        # Return instances of tools implemented by your package.
        return []
```

The empty list is a scaffold; replace it with your actual tool instances before
registering the plugin. `initialize()` and `cleanup()` are optional lifecycle hooks.
A plugin that opens resources should close them in `cleanup()`.

## Explicit tool registration

An embedding application can own a registry and the plugin's lifetime:

```python
from victor.tools.base import ToolRegistry

plugin = Plugin()
registry = ToolRegistry()
plugin.initialize()
try:
    for tool in plugin.get_tools():
        registry.register(tool)
    # Use this registry in your application's runtime integration.
finally:
    plugin.cleanup()
```

This does not automatically install the registry into an already-created Agent.
Runtime discovery and entry-point configuration are separate integration concerns.
There is no current `victor.tools.plugin_manager.ToolPluginManager` API, and this guide
does not promise automatic hot reload.

## Vertical plugins

For a separately packaged vertical, use the [vertical extension guide](verticals.md).
Vertical definition packages depend on `victor_contracts`; keep runtime imports out of
the definition layer.

The previous plugin manager examples are preserved in [page history](https://github.com/anvai-labs/victor/commits/develop/docs/development/extending/plugins.md).

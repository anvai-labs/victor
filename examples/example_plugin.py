#!/usr/bin/env python3
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

"""Example plugin demonstrating Victor's plugin system.

This module shows how to create a custom tool plugin that can be
dynamically loaded at runtime. It includes:

1. A custom tool implementation
2. A plugin class that registers the tool
3. Usage with ToolPluginRegistry

To run this example:
    python examples/example_plugin.py

To load the plugin programmatically:
    from victor.tools.plugin_registry import ToolPluginRegistry
    from examples.example_plugin import WeatherPlugin

    manager = ToolPluginRegistry()
    manager.load(WeatherPlugin())
"""

import asyncio
import random
from datetime import datetime
from typing import Any, Dict, List

from victor.tools.base import BaseTool, CostTier, ToolResult
from victor.tools.plugin import ToolPlugin

# =============================================================================
# Example Tool Implementation
# =============================================================================


class WeatherTool(BaseTool):
    """A sample tool that provides mock weather information.

    This demonstrates how to create a custom tool with:
    - Proper parameter definitions
    - Async execution
    - Structured output
    - Error handling
    """

    @property
    def name(self) -> str:
        return "get_weather"

    @property
    def description(self) -> str:
        return (
            "Get current weather information for a city. Returns temperature, "
            "conditions, humidity, and wind speed. Note: This is mock data for "
            "demonstration purposes."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "Name of the city to get weather for",
                },
                "units": {
                    "type": "string",
                    "description": "Temperature units: 'celsius' or 'fahrenheit'",
                    "default": "celsius",
                },
            },
            "required": ["city"],
        }

    @property
    def cost_tier(self) -> CostTier:
        # This is a low-cost tool (mock data, no API calls)
        return CostTier.LOW

    async def execute(self, context: Dict[str, Any], **kwargs: Any) -> ToolResult:
        """Execute the weather lookup."""
        city = kwargs.get("city", "").strip()
        units = kwargs.get("units", "celsius").lower()

        if not city:
            return ToolResult(
                success=False,
                output=None,
                error="City name is required",
            )

        if units not in ("celsius", "fahrenheit"):
            return ToolResult(
                success=False,
                output=None,
                error="Units must be 'celsius' or 'fahrenheit'",
            )

        # Generate mock weather data
        weather_conditions = ["Sunny", "Cloudy", "Partly Cloudy", "Rainy", "Stormy"]
        temp_c = random.randint(-5, 35)
        temp = temp_c if units == "celsius" else int(temp_c * 9 / 5 + 32)
        unit_symbol = "°C" if units == "celsius" else "°F"

        weather_data = {
            "city": city.title(),
            "temperature": f"{temp}{unit_symbol}",
            "condition": random.choice(weather_conditions),
            "humidity": f"{random.randint(30, 90)}%",
            "wind_speed": f"{random.randint(0, 30)} km/h",
            "timestamp": datetime.now().isoformat(),
            "is_mock": True,
        }

        return ToolResult(
            success=True,
            output=weather_data,
            error=None,
        )


class TemperatureConverterTool(BaseTool):
    """A simple tool to convert temperatures between units."""

    @property
    def name(self) -> str:
        return "convert_temperature"

    @property
    def description(self) -> str:
        return "Convert temperature between Celsius, Fahrenheit, and Kelvin."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "value": {
                    "type": "number",
                    "description": "Temperature value to convert",
                },
                "from_unit": {
                    "type": "string",
                    "description": "Source unit: 'celsius', 'fahrenheit', or 'kelvin'",
                },
                "to_unit": {
                    "type": "string",
                    "description": "Target unit: 'celsius', 'fahrenheit', or 'kelvin'",
                },
            },
            "required": ["value", "from_unit", "to_unit"],
        }

    @property
    def cost_tier(self) -> CostTier:
        return CostTier.FREE  # No external calls, pure computation

    async def execute(self, context: Dict[str, Any], **kwargs: Any) -> ToolResult:
        """Execute temperature conversion."""
        try:
            value = float(kwargs.get("value", 0))
            from_unit = kwargs.get("from_unit", "").lower()
            to_unit = kwargs.get("to_unit", "").lower()

            valid_units = {"celsius", "fahrenheit", "kelvin"}
            if from_unit not in valid_units or to_unit not in valid_units:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Invalid unit. Must be one of: {valid_units}",
                )

            # Convert to Celsius first
            if from_unit == "fahrenheit":
                celsius = (value - 32) * 5 / 9
            elif from_unit == "kelvin":
                celsius = value - 273.15
            else:
                celsius = value

            # Convert from Celsius to target
            if to_unit == "fahrenheit":
                result = celsius * 9 / 5 + 32
            elif to_unit == "kelvin":
                result = celsius + 273.15
            else:
                result = celsius

            return ToolResult(
                success=True,
                output={
                    "original": f"{value} {from_unit}",
                    "converted": f"{result:.2f} {to_unit}",
                    "value": round(result, 2),
                },
                error=None,
            )

        except (ValueError, TypeError) as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Invalid temperature value: {e}",
            )


# =============================================================================
# Plugin Implementation
# =============================================================================


class WeatherPlugin(ToolPlugin):
    """Example plugin that provides weather-related tools.

    This plugin demonstrates:
    - Plugin metadata configuration
    - Multiple tool registration
    - Lifecycle hooks (initialize/cleanup)
    - Configuration validation
    - Tool execution callbacks

    Configuration:
        api_key: Optional API key (not used in mock mode)
        cache_ttl: Cache time-to-live in seconds (default: 300)
    """

    # Plugin metadata
    name = "weather_plugin"
    version = "1.0.0"
    description = "Weather information and temperature conversion tools"
    author = "Victor Team"
    homepage = "https://github.com/example/victor-weather-plugin"
    dependencies = []  # No external dependencies

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the weather plugin."""
        super().__init__(config)
        self._cache: Dict[str, Any] = {}
        self._tool_call_count: Dict[str, int] = {}

    def get_tools(self) -> List[BaseTool]:
        """Return the tools provided by this plugin."""
        return [
            WeatherTool(),
            TemperatureConverterTool(),
        ]

    def initialize(self) -> None:
        """Initialize plugin resources.

        In a real plugin, you might:
        - Connect to an API
        - Load cache from disk
        - Initialize rate limiters
        """
        print(f"[{self.name}] Initializing plugin v{self.version}")
        self._cache = {}
        self._tool_call_count = {
            "get_weather": 0,
            "convert_temperature": 0,
        }

    def cleanup(self) -> None:
        """Clean up plugin resources."""
        print(f"[{self.name}] Cleanup - Tool calls: {self._tool_call_count}")
        self._cache.clear()

    def validate_config(self) -> List[str]:
        """Validate plugin configuration."""
        errors = []

        # Check optional cache_ttl
        cache_ttl = self.config.get("cache_ttl")
        if cache_ttl is not None:
            try:
                ttl = int(cache_ttl)
                if ttl < 0:
                    errors.append("cache_ttl must be positive")
            except (ValueError, TypeError):
                errors.append("cache_ttl must be an integer")

        return errors

    def on_tool_registered(self, tool: BaseTool) -> None:
        """Called when a tool is registered."""
        print(f"[{self.name}] Registered tool: {tool.name}")

    def on_tool_executed(self, tool_name: str, success: bool, result: Any) -> None:
        """Track tool execution metrics."""
        if tool_name in self._tool_call_count:
            self._tool_call_count[tool_name] += 1

        status = "success" if success else "failed"
        print(f"[{self.name}] {tool_name} execution {status}")


# =============================================================================
# Demo and Usage
# =============================================================================


async def demo_plugin_usage():
    """Demonstrate plugin usage with ToolPluginRegistry."""
    from victor.tools.registry import ToolRegistry
    from victor.tools.plugin_registry import ToolPluginRegistry

    print("=" * 60)
    print("Victor Plugin System Demo")
    print("=" * 60)

    # Create plugin registry and tool registry
    manager = ToolPluginRegistry()
    registry = ToolRegistry()

    # Load our example plugin
    plugin = WeatherPlugin(config={"cache_ttl": 600})
    manager.load(plugin)

    # Register plugin tools with registry
    manager.register_tools(registry)

    print("\n--- Loaded Plugins ---")
    for name, meta in manager.list_plugins().items():
        print(f"  {name} v{meta['version']}: {meta['description']}")
        print(f"    Tools: {meta['tools']}")

    print("\n--- Registered Tools ---")
    for tool in registry.list_tools():
        print(f"  {tool.name}: {tool.description[:50]}...")

    print("\n--- Tool Execution ---")

    # Create context (shared resources for tools)
    context: Dict[str, Any] = {}

    # Execute weather tool
    weather_result = await registry.execute("get_weather", context, city="London", units="celsius")
    print("\nWeather for London:")
    print(f"  Success: {weather_result.success}")
    if weather_result.success:
        for key, value in weather_result.output.items():
            print(f"  {key}: {value}")

    # Execute temperature converter
    temp_result = await registry.execute(
        "convert_temperature",
        context,
        value=100,
        from_unit="celsius",
        to_unit="fahrenheit",
    )
    print("\nTemperature Conversion:")
    print(f"  Success: {temp_result.success}")
    if temp_result.success:
        print(f"  {temp_result.output['original']} = {temp_result.output['converted']}")

    print("\n--- Plugin Metrics ---")
    print(f"  Tool calls: {plugin._tool_call_count}")

    # Unload plugin
    print("\n--- Unloading Plugin ---")
    manager.unload("weather_plugin")

    print("\n" + "=" * 60)
    print("Demo Complete")
    print("=" * 60)


async def demo_mcp_registry():
    """Demonstrate MCP Registry usage."""
    from victor.integrations.mcp import MCPRegistry, MCPServerConfig

    print("\n" + "=" * 60)
    print("MCP Registry Demo")
    print("=" * 60)

    # Create registry
    registry = MCPRegistry(health_check_enabled=False)

    # Register server configuration (not connecting since we don't have a real server)
    config = MCPServerConfig(
        name="demo-server",
        command=["python", "-m", "some_mcp_server"],
        description="Demo MCP server (not actually running)",
        auto_connect=False,  # Don't try to connect
        tags=["demo", "example"],
    )
    registry.register_server(config)

    print("\n--- Registered MCP Servers ---")
    for server_name in registry.list_servers():
        status = registry.get_server_status(server_name)
        print(f"  {server_name}:")
        print(f"    Status: {status['status']}")
        print(f"    Tags: {status['tags']}")

    print("\n--- Registry Status ---")
    status = registry.get_registry_status()
    print(f"  Total servers: {status['total_servers']}")
    print(f"  Connected: {status['connected_servers']}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║                 Victor Plugin System Example                  ║
╠══════════════════════════════════════════════════════════════╣
║  This example demonstrates:                                   ║
║  1. Creating custom tools (WeatherTool, TemperatureConverter) ║
║  2. Packaging tools in a plugin (WeatherPlugin)               ║
║  3. Using ToolPluginRegistry to load/unload plugins           ║
║  4. Tool execution through the registry                       ║
║  5. MCP Registry for managing external MCP servers            ║
╚══════════════════════════════════════════════════════════════╝
""")

    asyncio.run(demo_plugin_usage())
    asyncio.run(demo_mcp_registry())

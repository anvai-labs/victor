# Building Custom Tools in Victor

Learn how to create custom tools that extend Victor's capabilities. This tutorial walks you through building a weather fetching tool from scratch.

## What You'll Build

A `weather` tool that fetches current weather conditions for any location. This example demonstrates:
- Tool structure and required attributes
- Parameter validation with JSON Schema
- Async execution
- Error handling
- Testing patterns

## Prerequisites

- Python 3.11+
- Victor installed (`pip install -e ./victor-contracts -e ".[dev]"`)
- Basic understanding of async/await in Python
- Familiarity with JSON Schema

## Time Estimate

30 minutes

---

## 1. Tool Anatomy

Victor provides two approaches for creating tools:

1. **Decorator-based** (`@tool`) - Recommended for most cases
2. **Class-based** (`BaseTool`) - For complex tools needing fine-grained control

### The @tool Decorator

The decorator automatically extracts metadata from your function's signature and docstring:

```python
from victor.tools.decorators import tool
from victor.tools.base import CostTier

@tool(cost_tier=CostTier.MEDIUM)
async def my_tool(param1: str, param2: int = 5) -> Dict[str, Any]:
    """Short description of what the tool does.

    Detailed description goes here. This becomes part of the
    tool's documentation shown to the LLM.

    Args:
        param1: Description of the first parameter.
        param2: Description of the second parameter.

    Returns:
        Dictionary with success status and output data.
    """
    # Tool implementation
    return {"success": True, "output": "result"}
```

### The BaseTool Class

For more control, inherit from `BaseTool`:

```python
from victor.tools.base import BaseTool, ToolResult, CostTier
from typing import Dict, Any

class MyTool(BaseTool):
    @property
    def name(self) -> str:
        return "my_tool"

    @property
    def description(self) -> str:
        return "Description shown to the LLM"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "First parameter"}
            },
            "required": ["param1"]
        }

    @property
    def cost_tier(self) -> CostTier:
        return CostTier.MEDIUM

    async def execute(self, _exec_ctx: Dict[str, Any], **kwargs) -> ToolResult:
        return ToolResult(success=True, output="result")
```

### Required Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Unique tool identifier (e.g., `"get_weather"`) |
| `description` | `str` | What the tool does (shown to LLM for selection) |
| `parameters` | `Dict` | JSON Schema defining accepted parameters |

### Optional Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `cost_tier` | `CostTier` | `FREE` | Resource cost classification |
| `access_mode` | `AccessMode` | `READONLY` | Type of system access required |
| `priority` | `Priority` | `MEDIUM` | Selection priority level |
| `danger_level` | `DangerLevel` | `SAFE` | Risk level for warnings |

### Cost Tiers

```python
from victor.tools.base import CostTier

CostTier.FREE    # Local operations (filesystem, git status)
CostTier.LOW     # Compute-only (code review, analysis)
CostTier.MEDIUM  # External API calls (web search, HTTP)
CostTier.HIGH    # Resource-intensive (batch processing 100+ files)
```

### Access Modes

```python
from victor.tools.base import AccessMode

AccessMode.READONLY   # Only reads data (safe)
AccessMode.WRITE      # Modifies files/state
AccessMode.EXECUTE    # Runs external commands
AccessMode.NETWORK    # Makes network calls
AccessMode.MIXED      # Multiple access types
```

---

## 2. Step-by-Step Implementation

### Step 1: Create the Tool File

Create `victor/tools/weather_tool.py`:

```python
"""Weather tool for fetching current weather conditions.

This tool demonstrates:
- External API integration
- Parameter validation
- Error handling patterns
- Async execution
"""

from typing import Any, Dict, Optional
import httpx

from victor.tools.base import AccessMode, CostTier, DangerLevel, Priority
from victor.tools.decorators import tool
```

### Step 2: Define the Tool Function

Add the decorated function with full type hints and docstring:

```python
@tool(
    cost_tier=CostTier.MEDIUM,
    category="weather",
    priority=Priority.CONTEXTUAL,
    access_mode=AccessMode.NETWORK,
    danger_level=DangerLevel.SAFE,
    keywords=["weather", "temperature", "forecast", "climate"],
)
async def get_weather(
    location: str,
    units: str = "celsius",
) -> Dict[str, Any]:
    """Get current weather conditions for a location.

    Fetches real-time weather data including temperature, conditions,
    humidity, and wind information. Useful for weather-related queries.

    Args:
        location: City name or location (e.g., "San Francisco", "London, UK").
        units: Temperature units - "celsius" or "fahrenheit" (default: celsius).

    Returns:
        Dictionary containing:
        - success: Whether the request succeeded
        - temperature: Current temperature in requested units
        - conditions: Weather conditions (e.g., "Sunny", "Cloudy")
        - humidity: Humidity percentage
        - wind_speed: Wind speed in km/h or mph
        - location: Resolved location name
        - error: Error message if failed
    """
```

### Step 3: Implement the Execute Logic

Add the implementation inside the function:

```python
    # Validate required parameter
    if not location:
        return {"success": False, "error": "Missing required parameter: location"}

    # Validate units parameter
    if units not in ("celsius", "fahrenheit"):
        return {
            "success": False,
            "error": f"Invalid units '{units}'. Must be 'celsius' or 'fahrenheit'.",
        }

    try:
        # Make API request (using wttr.in as a free, no-auth-required API)
        url = f"https://wttr.in/{location}?format=j1"

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)

            if response.status_code != 200:
                return {
                    "success": False,
                    "error": f"Weather service returned status {response.status_code}",
                }

            data = response.json()

            # Extract current conditions
            current = data.get("current_condition", [{}])[0]
            area = data.get("nearest_area", [{}])[0]

            # Get temperature in requested units
            if units == "celsius":
                temp = current.get("temp_C", "N/A")
                temp_unit = "C"
            else:
                temp = current.get("temp_F", "N/A")
                temp_unit = "F"

            # Build response
            return {
                "success": True,
                "temperature": f"{temp} {temp_unit}",
                "conditions": current.get("weatherDesc", [{}])[0].get("value", "Unknown"),
                "humidity": f"{current.get('humidity', 'N/A')}%",
                "wind_speed": f"{current.get('windspeedKmph', 'N/A')} km/h",
                "location": area.get("areaName", [{}])[0].get("value", location),
            }

    except httpx.TimeoutException:
        return {"success": False, "error": "Weather service request timed out"}
    except Exception as e:
        return {"success": False, "error": f"Failed to fetch weather: {str(e)}"}
```

### Step 4: Handle Errors Properly

Victor tools should always return a dictionary with `success` and `error` keys for failures:

```python
# Good error handling
return {"success": False, "error": "Clear, actionable error message"}

# Include context when helpful
return {
    "success": False,
    "error": f"Invalid location '{location}': not found in weather database",
}
```

### Step 5: Return ToolResult (Class-Based)

If using `BaseTool`, return a `ToolResult` object:

```python
from victor.tools.base import ToolResult

# Success case
return ToolResult(
    success=True,
    output={"temperature": "22 C", "conditions": "Sunny"},
    metadata={"source": "wttr.in", "cached": False},
)

# Failure case
return ToolResult(
    success=False,
    output=None,
    error="Location not found",
    metadata={"attempted_location": location},
)
```

---

## 3. Parameter Validation

### JSON Schema Basics

Victor uses JSON Schema Draft 7 for parameter validation:

```python
parameters = {
    "type": "object",
    "properties": {
        "location": {
            "type": "string",
            "description": "City name or location",
        },
        "units": {
            "type": "string",
            "enum": ["celsius", "fahrenheit"],
            "description": "Temperature units",
        },
        "include_forecast": {
            "type": "boolean",
            "description": "Include 3-day forecast",
            "default": False,
        },
    },
    "required": ["location"],
    "additionalProperties": False,  # Reject unknown parameters
}
```

### Type Mapping

| Python Type | JSON Schema Type |
|------------|------------------|
| `str` | `"string"` |
| `int` | `"integer"` |
| `float` | `"number"` |
| `bool` | `"boolean"` |
| `list` | `"array"` |
| `dict` | `"object"` |
| `Optional[X]` | Same as X (nullable) |

### Enum Constraints

Restrict values to a specific set:

```python
"units": {
    "type": "string",
    "enum": ["celsius", "fahrenheit", "kelvin"],
    "description": "Temperature units",
}
```

### Array Parameters

```python
"cities": {
    "type": "array",
    "items": {"type": "string"},
    "description": "List of city names",
    "minItems": 1,
    "maxItems": 10,
}
```

### Nested Objects

```python
"coordinates": {
    "type": "object",
    "properties": {
        "lat": {"type": "number", "description": "Latitude"},
        "lon": {"type": "number", "description": "Longitude"},
    },
    "required": ["lat", "lon"],
}
```

---

## 4. Error Handling

### Return Format for Errors

Always return a dictionary with `success: False` and a clear `error` message:

```python
# Input validation error
if not location:
    return {"success": False, "error": "Missing required parameter: location"}

# Invalid value error
if units not in ("celsius", "fahrenheit"):
    return {
        "success": False,
        "error": f"Invalid units '{units}'. Must be 'celsius' or 'fahrenheit'.",
    }

# External service error
except httpx.TimeoutException:
    return {"success": False, "error": "Weather service request timed out"}

# Generic exception
except Exception as e:
    return {"success": False, "error": f"Failed to fetch weather: {str(e)}"}
```

### Using Structured Errors

For more complex error handling, use Victor's error classes:

```python
from victor.core.errors import ToolExecutionError, ToolValidationError

# Validation error (parameter issues)
raise ToolValidationError(
    message="Invalid location format",
    tool_name="get_weather",
    recovery_hint="Use format 'City' or 'City, Country'",
)

# Execution error (runtime issues)
raise ToolExecutionError(
    message="Weather service unavailable",
    tool_name="get_weather",
    recovery_hint="Try again in a few seconds",
)
```

---

## 5. Registration

### Automatic Registration

Tools decorated with `@tool` are automatically registered with the metadata registry when the module is imported.

### Manual Registration with ToolRegistry

```python
from victor.tools.registry import ToolRegistry
from victor.tools.weather_tool import get_weather

registry = ToolRegistry()
registry.register(get_weather)  # Registers get_weather.Tool
```

### Entry Point Registration (Plugins)

For tools distributed as separate packages, use entry points in `pyproject.toml`:

```toml
[project.entry-points."victor.tools"]
weather = "victor_weather:get_weather"
```

The tool will be discovered and registered when Victor starts.

### Adding to Victor's Default Tools

1. Create your tool in `victor/tools/your_tool.py`
2. Import and register in `victor/tools/__init__.py`:

```python
from victor.tools.weather_tool import get_weather

# Tools are auto-registered via the @tool decorator
# Just ensure the module is imported
```

---

## 6. Testing the Tool

### Unit Test Example

Create `tests/unit/tools/test_weather_tool.py`:

```python
"""Tests for weather_tool module."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from victor.tools.weather_tool import get_weather


class TestGetWeather:
    """Tests for get_weather function."""

    @pytest.mark.asyncio
    async def test_weather_success(self):
        """Test successful weather fetch."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "current_condition": [{
                "temp_C": "22",
                "temp_F": "72",
                "humidity": "65",
                "windspeedKmph": "15",
                "weatherDesc": [{"value": "Sunny"}],
            }],
            "nearest_area": [{
                "areaName": [{"value": "San Francisco"}],
            }],
        }

        with patch("victor.tools.weather_tool.httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_instance

            result = await get_weather(location="San Francisco")

            assert result["success"] is True
            assert result["temperature"] == "22 C"
            assert result["conditions"] == "Sunny"
            assert result["humidity"] == "65%"

    @pytest.mark.asyncio
    async def test_weather_missing_location(self):
        """Test with missing location parameter."""
        result = await get_weather(location="")

        assert result["success"] is False
        assert "Missing required parameter" in result["error"]

    @pytest.mark.asyncio
    async def test_weather_invalid_units(self):
        """Test with invalid units parameter."""
        result = await get_weather(location="London", units="kelvin")

        assert result["success"] is False
        assert "Invalid units" in result["error"]

    @pytest.mark.asyncio
    async def test_weather_fahrenheit(self):
        """Test temperature in fahrenheit."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "current_condition": [{
                "temp_C": "22",
                "temp_F": "72",
                "humidity": "65",
                "windspeedKmph": "15",
                "weatherDesc": [{"value": "Cloudy"}],
            }],
            "nearest_area": [{"areaName": [{"value": "London"}]}],
        }

        with patch("victor.tools.weather_tool.httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_instance

            result = await get_weather(location="London", units="fahrenheit")

            assert result["success"] is True
            assert result["temperature"] == "72 F"

    @pytest.mark.asyncio
    async def test_weather_timeout(self):
        """Test timeout handling."""
        import httpx

        with patch("victor.tools.weather_tool.httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_instance.get = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
            mock_client.return_value = mock_instance

            result = await get_weather(location="Tokyo")

            assert result["success"] is False
            assert "timed out" in result["error"]

    @pytest.mark.asyncio
    async def test_weather_api_error(self):
        """Test API error handling."""
        mock_response = MagicMock()
        mock_response.status_code = 503

        with patch("victor.tools.weather_tool.httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_instance

            result = await get_weather(location="Paris")

            assert result["success"] is False
            assert "503" in result["error"]
```

### Running Tests

```bash
# Run the specific test file
pytest tests/unit/tools/test_weather_tool.py -v

# Run with coverage
pytest tests/unit/tools/test_weather_tool.py --cov=victor.tools.weather_tool

# Run a specific test
pytest tests/unit/tools/test_weather_tool.py::TestGetWeather::test_weather_success -v
```

### Manual Testing in Victor

Start Victor and test your tool interactively:

```bash
victor chat
```

Then ask:
```
> What's the weather in San Francisco?
```

Victor will select and execute your `get_weather` tool automatically.

---

## 7. Complete Code Listing

Here's the full `victor/tools/weather_tool.py`:

```python
# Copyright 2025 Your Name
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

"""Weather tool for fetching current weather conditions.

This tool demonstrates:
- External API integration
- Parameter validation
- Error handling patterns
- Async execution
"""

from typing import Any, Dict

import httpx

from victor.tools.base import AccessMode, CostTier, DangerLevel, Priority
from victor.tools.decorators import tool


@tool(
    cost_tier=CostTier.MEDIUM,
    category="weather",
    priority=Priority.CONTEXTUAL,
    access_mode=AccessMode.NETWORK,
    danger_level=DangerLevel.SAFE,
    keywords=["weather", "temperature", "forecast", "climate", "conditions"],
    task_types=["research", "analysis"],
    execution_category="network",
)
async def get_weather(
    location: str,
    units: str = "celsius",
) -> Dict[str, Any]:
    """Get current weather conditions for a location.

    Fetches real-time weather data including temperature, conditions,
    humidity, and wind information. Useful for weather-related queries.

    Args:
        location: City name or location (e.g., "San Francisco", "London, UK").
        units: Temperature units - "celsius" or "fahrenheit" (default: celsius).

    Returns:
        Dictionary containing:
        - success: Whether the request succeeded
        - temperature: Current temperature in requested units
        - conditions: Weather conditions (e.g., "Sunny", "Cloudy")
        - humidity: Humidity percentage
        - wind_speed: Wind speed in km/h
        - location: Resolved location name
        - error: Error message if failed
    """
    # Validate required parameter
    if not location:
        return {"success": False, "error": "Missing required parameter: location"}

    # Validate units parameter
    if units not in ("celsius", "fahrenheit"):
        return {
            "success": False,
            "error": f"Invalid units '{units}'. Must be 'celsius' or 'fahrenheit'.",
        }

    try:
        # Make API request (using wttr.in as a free, no-auth-required API)
        url = f"https://wttr.in/{location}?format=j1"

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)

            if response.status_code != 200:
                return {
                    "success": False,
                    "error": f"Weather service returned status {response.status_code}",
                }

            data = response.json()

            # Extract current conditions
            current = data.get("current_condition", [{}])[0]
            area = data.get("nearest_area", [{}])[0]

            # Get temperature in requested units
            if units == "celsius":
                temp = current.get("temp_C", "N/A")
                temp_unit = "C"
            else:
                temp = current.get("temp_F", "N/A")
                temp_unit = "F"

            # Build response
            return {
                "success": True,
                "temperature": f"{temp} {temp_unit}",
                "conditions": current.get("weatherDesc", [{}])[0].get("value", "Unknown"),
                "humidity": f"{current.get('humidity', 'N/A')}%",
                "wind_speed": f"{current.get('windspeedKmph', 'N/A')} km/h",
                "location": area.get("areaName", [{}])[0].get("value", location),
            }

    except httpx.TimeoutException:
        return {"success": False, "error": "Weather service request timed out"}
    except Exception as e:
        return {"success": False, "error": f"Failed to fetch weather: {str(e)}"}
```

---

## 8. Advanced Topics

### Adding Semantic Metadata

Enhance tool selection with rich metadata:

```python
@tool(
    cost_tier=CostTier.MEDIUM,
    category="weather",
    keywords=["weather", "temperature", "forecast"],
    use_cases=["Check current weather", "Get temperature for a city"],
    examples=["What's the weather in NYC?", "Temperature in Tokyo"],
    mandatory_keywords=["weather forecast", "current weather"],
    stages=["initial", "planning"],
    task_types=["research"],
    progress_params=["location"],  # Different locations = progress
    execution_category="network",
)
```

### Idempotent Tools

Mark read-only tools as idempotent to enable caching and safe retries:

```python
class WeatherTool(BaseTool):
    @property
    def is_idempotent(self) -> bool:
        return True  # Same input always gives same output
```

### Accessing Execution Context

Tools can access shared resources through the execution context:

```python
@tool(...)
async def my_tool(
    param: str,
    _exec_ctx: Dict[str, Any] = None,  # Framework injects this
) -> Dict[str, Any]:
    # Access shared resources
    cache_manager = _exec_ctx.get("cache_manager") if _exec_ctx else None
    code_manager = _exec_ctx.get("code_manager") if _exec_ctx else None

    # Use resources...
    return {"success": True, "output": "result"}
```

### Tool Configuration via ToolConfig

Access global tool configuration:

```python
from victor.tools.base import ToolConfig

@tool(...)
async def my_tool(
    param: str,
    context: Dict[str, Any] = None,
) -> Dict[str, Any]:
    config = ToolConfig.from_context(context)
    if config and config.provider:
        # Use LLM provider for AI-powered features
        pass

    return {"success": True, "output": "result"}
```

---

## Summary

You've learned how to:

1. Create tools using the `@tool` decorator or `BaseTool` class
2. Define parameters with JSON Schema validation
3. Handle errors gracefully with clear messages
4. Register tools for discovery
5. Write comprehensive unit tests
6. Add semantic metadata for intelligent tool selection

For more examples, explore the existing tools in `victor/tools/` - they demonstrate patterns for filesystem operations, HTTP requests, git integration, and more.

## Next Steps

- Read the [Tool Selection Guide](/docs/guides/development/TOOL_SELECTION.md)
- Explore [Tool Calling Formats](/docs/guides/development/TOOL_CALLING_FORMATS.md)
- Learn about [Creating Verticals](/docs/development/extending/verticals.md)

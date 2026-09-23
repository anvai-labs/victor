# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0

"""Server schemas survive discovery and model-facing adapter presentation."""

from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from victor.integrations.mcp.client import MCPClient
from victor.integrations.mcp.protocol import MCPTool
from victor.integrations.mcp.server import MCPServer
from victor.tools.enums import SchemaLevel
from victor.tools.mcp_adapter_tool import MCPAdapterTool


@pytest.fixture
def schema():
    return {
        "type": "object",
        "$defs": {"value": {"type": ["string", "null"], "maxLength": 100}},
        "properties": {
            "fields": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "match": {"oneOf": [{"const": "label"}, {"const": "role"}]},
                        "value": {"$ref": "#/$defs/value"},
                    },
                    "required": ["match", "value"],
                    "additionalProperties": False,
                },
            },
            "attempts": {"type": "integer", "minimum": 0, "maximum": 3, "default": 1},
            "policy": {"enum": ["stop", "report"]},
        },
        "required": ["fields"],
        "additionalProperties": False,
    }


async def discover(tool_data):
    client = MCPClient()
    client.initialized = True
    client._send_request = AsyncMock(return_value={"result": {"tools": [tool_data]}})
    tools = await client.refresh_tools()
    return tools[0]


@pytest.mark.parametrize("level", [None, *SchemaLevel])
async def test_discovery_preserves_complete_model_contract(schema, level):
    wire = {"name": "autofill", "description": "Fill a form", "inputSchema": schema}
    before = deepcopy(wire)
    tool = await discover(wire)
    adapter = MCPAdapterTool(tool, MagicMock(), "browser")
    assert adapter.to_schema(level)["function"]["parameters"] == before["inputSchema"]
    assert adapter.to_json_schema()["function"]["parameters"] == before["inputSchema"]
    assert adapter.default_schema_level == "full"
    assert wire == before


async def test_schema_snapshots_do_not_share_mutable_state(schema):
    expected = deepcopy(schema)
    tool = await discover({"name": "fill", "description": "Fill", "inputSchema": schema})
    schema["properties"]["fields"]["items"]["required"].clear()
    assert tool.input_schema == expected
    adapter = MCPAdapterTool(tool, MagicMock(), "browser")
    tool.input_schema["properties"].clear()
    adapter.parameters["$defs"].clear()
    adapter.to_schema(SchemaLevel.STUB)["function"]["parameters"]["required"].clear()
    assert adapter.parameters == expected


@pytest.mark.parametrize("bad", [None, [], "PRIVATE_SCHEMA", {}, {"type": "array"}])
async def test_invalid_present_schema_refuses_without_payload_diagnostic(bad, caplog):
    client = MCPClient()
    client.initialized = True
    previous = MCPTool(name="previous", description="Previously valid catalog")
    client.tools = [previous]
    wire = {
        "name": "fill",
        "description": "PRIVATE_SCHEMA",
        "inputSchema": bad,
        "parameters": [{"name": "fallback", "type": "string", "description": "Legacy"}],
    }
    before = deepcopy(wire)
    client._send_request = AsyncMock(return_value={"result": {"tools": [wire]}})
    with pytest.raises(ValidationError) as error:
        await client.refresh_tools()
    assert "PRIVATE_SCHEMA" not in str(error.value)
    assert "PRIVATE_SCHEMA" not in caplog.text
    assert client.tools == [previous]
    assert wire == before


async def test_absent_schema_keeps_legacy_conversion_and_serialization():
    wire = {
        "name": "search",
        "description": "Search",
        "parameters": [
            {"name": "query", "description": "Query", "type": "string", "required": True},
            {"name": "limit", "description": "Limit", "type": "number", "default": 5},
        ],
    }
    tool = await discover(wire)
    adapter = MCPAdapterTool(tool, MagicMock(), "legacy")
    assert adapter.parameters == {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Query"},
            "limit": {"type": "number", "description": "Limit", "default": 5},
        },
        "required": ["query"],
    }
    assert adapter.default_schema_level == "stub"
    assert (
        "limit" not in adapter.to_schema(SchemaLevel.STUB)["function"]["parameters"]["properties"]
    )
    assert set(tool.model_dump()) == {"name", "description", "parameters", "version"}
    assert adapter.validate_parameters(query="ok")
    assert not adapter.validate_parameters(query=42)
    assert adapter.preserve_arguments is False


def test_captured_schema_does_not_change_legacy_server_serialization(schema):
    tool = MCPTool(name="fill", description="Fill", inputSchema=schema)
    assert tool.model_dump() == {
        "name": "fill",
        "description": "Fill",
        "parameters": [],
        "version": "1.0.0",
    }
    assert "inputSchema" not in tool.model_dump(by_alias=True)


async def test_adapter_dispatches_original_arguments_without_schema_rewrite(schema):
    tool = await discover({"name": "fill", "description": "Fill", "inputSchema": schema})
    registry = MagicMock()
    registry.call_tool = AsyncMock(return_value=MagicMock(success=True, result={"ok": True}))
    adapter = MCPAdapterTool(tool, registry, "browser")
    arguments = {"fields": [{"match": "label", "value": None}], "attempts": 2}
    result = await adapter.execute({}, **arguments)
    assert result.success
    registry.call_tool.assert_awaited_once_with("fill", **arguments)


@pytest.mark.parametrize("property_schema", [{"type": ["string", "null"]}, True, False])
def test_legacy_server_export_handles_unrepresentable_property_types(property_schema):
    schema = {"type": "object", "properties": {"value": property_schema}}
    tool = MCPTool(name="fill", description="Fill", inputSchema=schema)
    adapter = MCPAdapterTool(tool, MagicMock(), "browser")
    exported = MCPServer()._tool_to_mcp(adapter)
    # Legacy export still has only five primitive types; it must not crash on discovery.
    assert exported.parameters[0].type.value == "string"
    assert adapter.parameters == schema


@pytest.mark.parametrize("property_schema", [True, False, {"type": ["string", "null"]}])
def test_executor_leaves_boolean_and_union_schema_values_to_validation(property_schema):
    from victor.agent.tool_executor import ToolExecutor

    schema = {"type": "object", "properties": {"value": property_schema}}
    tool = MCPTool(name="fill", description="Fill", inputSchema=schema)
    adapter = MCPAdapterTool(tool, MagicMock(), "browser")
    arguments = {"value": "5"}
    ToolExecutor._coerce_arg_types(adapter, arguments)
    assert arguments == {"value": "5"}
    valid = adapter.validate_parameters(**arguments)
    assert valid is (property_schema is not False)


def adapter_for_schema(schema):
    registry = MagicMock()
    registry.call_tool = AsyncMock(return_value=MagicMock(success=True, result="ok"))
    return MCPAdapterTool(
        MCPTool(name="fill", description="Fill", inputSchema=schema), registry, "browser"
    )


@pytest.mark.parametrize(
    "reference",
    ["https://example.invalid/PRIVATE_SCHEMA", "file:///PRIVATE_SCHEMA", "#/$defs/missing"],
)
async def test_unresolved_references_refuse_without_io_or_payload(reference):
    adapter = adapter_for_schema({"type": "object", "properties": {"value": {"$ref": reference}}})
    with patch("urllib.request.urlopen", side_effect=OSError("PRIVATE_SCHEMA")) as opener:
        validation = adapter.validate_parameters_detailed(value="PRIVATE_VALUE")
        result = await adapter.execute({}, value="PRIVATE_VALUE")
    opener.assert_not_called()
    assert not validation.valid
    assert validation.errors == ["MCP input schema could not be validated locally"]
    assert not result.success
    assert "PRIVATE" not in result.error
    adapter._registry.call_tool.assert_not_awaited()


@pytest.mark.parametrize("value,valid", [("ok", True), (42, False)])
async def test_local_reference_constraints_apply_before_dispatch(value, valid):
    adapter = adapter_for_schema(
        {
            "type": "object",
            "$defs": {"value": {"type": "string"}},
            "properties": {"value": {"$ref": "#/$defs/value"}},
        }
    )
    assert adapter.validate_parameters(value=value) is valid
    result = await adapter.execute({}, value=value)
    assert result.success is valid
    assert adapter._registry.call_tool.await_count == int(valid)


async def test_malformed_nested_schema_refuses_without_diagnostic_payload():
    adapter = adapter_for_schema(
        {"type": "object", "properties": {"value": {"type": "PRIVATE_SCHEMA"}}}
    )
    validation = adapter.validate_parameters_detailed(value="PRIVATE_VALUE")
    assert not validation.valid
    assert validation.errors == ["MCP input schema could not be validated locally"]
    result = await adapter.execute({}, value="PRIVATE_VALUE")
    assert not result.success
    assert "PRIVATE" not in result.error
    adapter._registry.call_tool.assert_not_awaited()


@pytest.mark.parametrize("mode", ["strict", "lenient", "off"])
async def test_executor_never_dispatches_unresolved_schema(mode, caplog):
    from victor.agent.tool_executor import ToolExecutor, ValidationMode
    from victor.tools.registry import ToolRegistry

    adapter = adapter_for_schema(
        {"type": "object", "properties": {"value": {"$ref": "#/$defs/missing"}}}
    )
    registry = ToolRegistry()
    registry.register(adapter)
    executor = ToolExecutor(tool_registry=registry, validation_mode=ValidationMode(mode))
    result = await executor.execute(adapter.name, {"value": "PRIVATE_VALUE"})
    assert not result.success
    adapter._registry.call_tool.assert_not_awaited()
    assert "PRIVATE" not in caplog.text


@pytest.mark.parametrize(
    "schema,arguments,valid",
    [
        ({"type": "object", "additionalProperties": {"type": "string"}}, {"extra": "ok"}, True),
        ({"type": "object"}, {"_exec_ctx": {"private": "PRIVATE_VALUE"}}, False),
        ({"type": "object", "additionalProperties": False}, {"extra": "PRIVATE_VALUE"}, False),
        ({"type": "object", "properties": {"count": {"type": "integer"}}}, {"count": "5"}, False),
        ({"type": "object", "required": "PRIVATE_SCHEMA"}, {}, False),
        ({"type": "object", "$schema": "https://example.invalid/PRIVATE_SCHEMA"}, {}, False),
        (
            {"type": "object", "properties": {"value": {"const": "PRIVATE_SCHEMA"}}},
            {"value": "PRIVATE_VALUE"},
            False,
        ),
        (
            {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "properties": {"value": {"type": "array", "prefixItems": [{"type": "string"}]}},
            },
            {"value": [42]},
            False,
        ),
        (
            {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "properties": {"value": {"type": "array", "prefixItems": [{"type": "string"}]}},
            },
            {"value": ["ok"]},
            True,
        ),
    ],
)
async def test_executor_honors_full_contract_without_heuristic_rewrite(
    schema, arguments, valid, caplog
):
    from victor.agent.tool_executor import ToolExecutor
    from victor.tools.registry import ToolRegistry

    adapter = adapter_for_schema(schema)
    registry = ToolRegistry()
    registry.register(adapter)
    executor = ToolExecutor(tool_registry=registry)
    before = deepcopy(arguments)
    with patch(
        "urllib.request.urlopen", side_effect=AssertionError("unexpected retrieval")
    ) as opener:
        result = await executor.execute(adapter.name, arguments)
    opener.assert_not_called()
    assert result.success is valid
    assert arguments == before
    if valid:
        adapter._registry.call_tool.assert_awaited_once_with("fill", **before)
    else:
        adapter._registry.call_tool.assert_not_awaited()
    assert "PRIVATE" not in caplog.text


async def test_executor_does_not_correct_external_paths_or_code():
    from victor.agent.tool_executor import ToolExecutor
    from victor.tools.registry import ToolRegistry

    adapter = adapter_for_schema({"type": "object", "properties": {"path": {"type": "string"}}})
    registry = ToolRegistry()
    registry.register(adapter)
    normalizer = MagicMock()
    correction = MagicMock()
    executor = ToolExecutor(tool_registry=registry, argument_normalizer=normalizer)
    executor.enable_code_correction = True
    executor.code_correction_middleware = correction
    executor._failed_path_redirects = {"remote-only-path": "different-local-path"}
    result = await executor.execute(adapter.name, {"path": "remote-only-path"})
    assert result.success
    normalizer.normalize_arguments.assert_not_called()
    correction.process.assert_not_called()
    adapter._registry.call_tool.assert_awaited_once_with("fill", path="remote-only-path")

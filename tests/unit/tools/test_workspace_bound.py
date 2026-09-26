"""Concurrent tool binding preserves paths, schemas, and caller arguments."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.tools.base import ToolResult
from victor.tools.workspace_bound import WorkspaceBoundTool


def tool(name):
    wrapped = MagicMock(name=name)
    wrapped.name = name
    wrapped.execute = AsyncMock(return_value=ToolResult(success=True, output="ok"))
    return wrapped


@pytest.mark.asyncio
async def test_concurrent_relative_writes_have_distinct_roots(tmp_path):
    original_cwd = Path.cwd()

    async def write(_ctx, **args):
        await asyncio.sleep(0)
        Path(args["path"]).write_text(args["content"])
        return ToolResult(success=True, output=None)

    delegate = tool("write")
    delegate.execute = write
    roots = [tmp_path / name for name in ("a", "b")]
    for root in roots:
        root.mkdir()
    await asyncio.gather(
        *[
            WorkspaceBoundTool(delegate, str(root)).execute({}, path="same.py", content=root.name)
            for root in roots
        ]
    )
    assert [(root / "same.py").read_text() for root in roots] == ["a", "b"]
    assert Path.cwd() == original_cwd


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name, args, expected",
    [
        ("shell", {"cmd": "pwd"}, {"cwd": "."}),
        ("read", {"path": "file.py"}, {"path": "file.py"}),
        ("ls", {}, {"path": "."}),
    ],
)
async def test_supported_paths_and_contract(name, args, expected, tmp_path):
    delegate = tool(name)
    bound = WorkspaceBoundTool(delegate, str(tmp_path))
    assert bound.parameters is delegate.parameters
    assert bound.metadata is delegate.metadata
    assert bound.description is delegate.description
    assert bound.cost_tier is delegate.cost_tier
    assert bound.is_idempotent is delegate.is_idempotent
    assert bound.custom_attribute is delegate.custom_attribute
    await bound.execute({}, **args)
    passed = delegate.execute.call_args.kwargs
    for key, value in expected.items():
        assert passed[key] == str(tmp_path / value)


@pytest.mark.asyncio
async def test_edit_transforms_copy_including_rename(tmp_path):
    delegate = tool("edit")
    bound = WorkspaceBoundTool(delegate, str(tmp_path))
    ops = [{"type": "rename", "path": "a", "new_path": "/absolute/b"}]
    await bound.execute({}, ops=ops)
    assert ops[0]["path"] == "a"
    assert delegate.execute.call_args.kwargs["ops"][0]["path"] == str(tmp_path / "a")
    assert delegate.execute.call_args.kwargs["ops"][0]["new_path"] == "/absolute/b"
    result = await bound.execute({}, ops='[{"path":"a"}]')
    assert not result.success
    delegate.execute.assert_awaited_once()


def test_missing_adapter_or_workspace_fails_explicitly(tmp_path, caplog):
    with pytest.raises(ValueError, match="adapter"):
        WorkspaceBoundTool(tool("unknown"), str(tmp_path))
    assert "no worktree path adapter" in caplog.text
    with pytest.raises(ValueError, match="does not exist"):
        WorkspaceBoundTool(tool("read"), str(tmp_path / "missing"))
    root = tmp_path / "child"
    root.mkdir()
    first = WorkspaceBoundTool(tool("read"), str(tmp_path))
    second = WorkspaceBoundTool(first, str(root))
    assert second._tool is first._tool

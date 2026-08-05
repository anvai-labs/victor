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

"""Core graph primitives shared by StateGraph and compiled execution."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    TypeVar,
    Union,
    runtime_checkable,
)

from victor.framework.graph_state import CopyOnWriteState

if TYPE_CHECKING:
    from victor.framework.graph import CompiledGraph

StateType = TypeVar("StateType")


class EdgeType(Enum):
    """Types of edges in the graph."""

    NORMAL = "normal"
    CONDITIONAL = "conditional"


class FrameworkNodeStatus(Enum):
    """Execution status of a framework graph node."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@runtime_checkable
class StateProtocol(Protocol):
    """Protocol for dict-like state objects."""

    def __getitem__(self, key: str) -> Any: ...

    def __setitem__(self, key: str, value: Any) -> None: ...

    def get(self, key: str, default: Any = None) -> Any: ...

    def copy(self) -> "StateProtocol": ...


@runtime_checkable
class NodeFunctionProtocol(Protocol[StateType]):
    """Protocol for node functions."""

    def __call__(self, state: StateType) -> Union[StateType, Awaitable[StateType]]: ...


@runtime_checkable
class ConditionFunctionProtocol(Protocol[StateType]):
    """Protocol for condition functions."""

    def __call__(self, state: StateType) -> str: ...


@dataclass
class Send:
    """Directive for dynamic fan-out / parallel execution."""

    node: str
    state: Dict[str, Any]
    join_at: Optional[str] = None


class ParallelBranchExecutionError(RuntimeError):
    """Raised when one or more fan-out branches fail."""


@dataclass
class Edge:
    """Represents an edge between nodes."""

    source: str
    target: Union[str, Dict[str, str]]
    edge_type: EdgeType = EdgeType.NORMAL
    condition: Optional[Callable[[Any], Union[str, List[Send]]]] = None

    def get_target(self, state: Any) -> Union[str, List[Send], None]:
        if self.edge_type == EdgeType.NORMAL:
            return self.target if isinstance(self.target, str) else None

        if self.condition is None:
            return None

        result = self.condition(state)
        if isinstance(result, list) and result and isinstance(result[0], Send):
            return result

        branch = result
        if isinstance(self.target, dict):
            return self.target.get(branch)  # type: ignore[arg-type]
        return None


@dataclass
class Node:
    """Represents a graph node."""

    id: str
    func: Callable[[Any], Union[Any, Awaitable[Any]]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    async def execute(self, state: Any) -> Any:
        result = self.func(state)
        if asyncio.iscoroutine(result):
            return await result
        return result


_SUBGRAPH_DEPTH_KEY = "__subgraph_depth__"
_MAX_SUBGRAPH_DEPTH = 10


@dataclass
class SubgraphNode:
    """A node that wraps a compiled subgraph for modular composition."""

    id: str
    compiled_graph: CompiledGraph
    input_mapper: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
    output_mapper: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    async def execute(self, state: Any) -> Any:
        if isinstance(state, CopyOnWriteState):
            input_state = state.to_dict()
        elif isinstance(state, dict):
            input_state = dict(state)
        else:
            input_state = state

        depth = input_state.get(_SUBGRAPH_DEPTH_KEY, 0)
        if depth >= _MAX_SUBGRAPH_DEPTH:
            raise RecursionError(
                f"Subgraph nesting depth {depth} exceeds "
                f"maximum ({_MAX_SUBGRAPH_DEPTH}). "
                f"Check for self-referencing subgraphs."
            )
        input_state[_SUBGRAPH_DEPTH_KEY] = depth + 1

        if self.input_mapper:
            input_state = self.input_mapper(input_state)

        result = await self.compiled_graph.invoke(input_state)
        if not result.success:
            raise RuntimeError(f"Subgraph '{self.id}' failed: {result.error}")

        output_state = result.state
        output_state.pop(_SUBGRAPH_DEPTH_KEY, None)
        if depth > 0:
            output_state[_SUBGRAPH_DEPTH_KEY] = depth

        if self.output_mapper:
            output_state = self.output_mapper(output_state)

        return output_state

"""Bounded control dependence for an explicit finite CFG.

Postdominance quantifies exit-reaching paths, not termination or source-program
semantics. Callers must retain CFG provenance. Regions without a path to a
declared/inferred sink abstain as a whole; no synthetic exit is invented for an
infinite loop. Integer bitsets keep the typed Python reference allocation small.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

ALGORITHM = "postdominance-exit-paths-v1"


@dataclass(frozen=True)
class DependenceLimits:
    max_nodes: int = 4096
    max_edges: int = 32768
    max_work: int = 1_000_000
    max_dependencies: int = 100_000

    def __post_init__(self) -> None:
        for value, upper in (
            (self.max_nodes, 16384),
            (self.max_edges, 262144),
            (self.max_work, 10_000_000),
            (self.max_dependencies, 1_000_000),
        ):
            if type(value) is not int or not 1 <= value <= upper:
                raise ValueError("Invalid control-dependence budget")


@dataclass(frozen=True)
class DependenceResult:
    dependencies: tuple[tuple[str, str], ...]
    complete: bool
    reason: str | None
    exit_basis: str
    work: int = 0
    algorithm: str = ALGORITHM


def control_dependence(
    successors: Mapping[str, Sequence[str]],
    *,
    exits: Collection[str] | None = None,
    limits: DependenceLimits = DependenceLimits(),
) -> DependenceResult:
    """Return deterministic graph dependencies or an explicit whole-graph abstention.

    Malformed endpoints/exits raise ValueError. Exhausted resource budgets return
    no partial dependencies. Exit nodes must be terminal; absent explicit exits,
    sinks are inferred from this graph only. A loop may depend on itself.
    """
    basis = "inferred_sinks" if exits is None else "declared_exits"

    def incomplete(reason: str, work: int = 0) -> DependenceResult:
        return DependenceResult((), False, reason, basis, work)

    if len(successors) > limits.max_nodes:
        return incomplete("budget_exceeded")
    names = list(successors)
    if any(not isinstance(name, str) or not name or len(name) > 1024 for name in names):
        raise ValueError("CFG node IDs must be nonempty bounded strings")
    names.sort()
    indices = {name: index for index, name in enumerate(names)}
    adjacency: list[tuple[int, ...]] = []
    predecessors: list[list[int]] = [[] for _ in names]
    edge_count = 0
    for index, name in enumerate(names):
        children = successors[name]
        if isinstance(children, (str, bytes)) or not isinstance(children, Sequence):
            raise ValueError("CFG successors must be sequences of node IDs")
        edges: set[int] = set()
        for child in children:
            edge_count += 1
            if edge_count > limits.max_edges:
                return incomplete("budget_exceeded")
            if not isinstance(child, str) or child not in indices:
                raise ValueError("CFG edge has an unknown endpoint")
            edges.add(indices[child])
        adjacency.append(tuple(sorted(edges)))
        for child_index in adjacency[-1]:
            predecessors[child_index].append(index)
    if exits is None:
        terminal = {index for index, children in enumerate(adjacency) if not children}
    else:
        if isinstance(exits, (str, bytes)) or len(exits) > limits.max_nodes:
            raise ValueError("CFG exits must be a bounded collection of node IDs")
        if any(not isinstance(name, str) or name not in indices for name in exits):
            raise ValueError("CFG exit has an unknown endpoint")
        terminal = {indices[name] for name in exits}
        if any(adjacency[index] for index in terminal):
            raise ValueError("CFG exit nodes must be terminal")
    reachable = set(terminal)
    reverse_queue = list(terminal)
    while reverse_queue:
        for pred in predecessors[reverse_queue.pop()]:
            if pred not in reachable:
                reachable.add(pred)
                reverse_queue.append(pred)
    if len(reachable) != len(names):
        return incomplete("non_exit_reaching_nodes")
    universe = (1 << len(names)) - 1
    postdom = [universe] * len(names)
    for index in terminal:
        postdom[index] = 1 << index
    queue = deque(reversed(range(len(names))))
    pending = set(queue)
    work = 0
    while queue:
        index = queue.popleft()
        pending.remove(index)
        if index in terminal:
            continue
        value = universe
        for child in adjacency[index]:
            work += 1
            if work > limits.max_work:
                return incomplete("budget_exceeded", work)
            value &= postdom[child]
        value |= 1 << index
        if value != postdom[index]:
            postdom[index] = value
            for pred in predecessors[index]:
                if pred not in pending:
                    pending.add(pred)
                    queue.append(pred)
    dependencies = []
    for index, children in enumerate(adjacency):
        union = 0
        for child in children:
            union |= postdom[child]
        # Y postdominates some successor of X but does not strictly
        # postdominate X. Keeping X here correctly captures loop self-dependence.
        mask = union & ~(postdom[index] & ~(1 << index))
        while mask:
            if len(dependencies) >= limits.max_dependencies:
                return incomplete("budget_exceeded", work)
            bit = mask & -mask
            dependencies.append((names[index], names[bit.bit_length() - 1]))
            mask ^= bit
    return DependenceResult(tuple(dependencies), True, None, basis, work)

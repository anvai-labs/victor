"""Subtree fingerprints with consistent local-name repetition patterns.

The join follows the standard Jaccard prefix filter. Fingerprints retain where
each local repeats: alpha-renaming matches, while x+x and x+y remain distinct.
This is syntax evidence, not a test of functional equivalence.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import math
import re
from typing import Callable, Iterable, Sequence

from victor.core.graph_rag.semantic_clones import MAX_PAIRS, TooManyPairsError


@dataclass(frozen=True)
class NormalizedNode:
    label: str
    children: tuple[NormalizedNode, ...] = ()


def subtree_fingerprints(root: NormalizedNode, namespace: str) -> tuple[str, ...]:
    """Hash each subtree, attaching repeat links at their lowest common ancestor.

    A repeat link is handled once rather than re-normalizing names inside every
    subtree. The traversal itself is iterative; LCA walks are bounded by depth.
    """
    nodes: list[NormalizedNode] = []
    parents: list[int] = []
    depths: list[int] = []
    first_local: list[int] = []
    children: list[list[int]] = []
    repeats: dict[int, list[tuple[int, int]]] = defaultdict(list)
    previous: dict[str, tuple[int, int]] = {}
    ordinal = 0
    stack = [(root, -1)]
    while stack:
        node, parent = stack.pop()
        idx = len(nodes)
        nodes.append(node)
        parents.append(parent)
        depths.append(depths[parent] + 1 if parent >= 0 else 0)
        first_local.append(ordinal)
        children.append([])
        if parent >= 0:
            children[parent].append(idx)
        if re.fullmatch(r"v\d+", node.label):
            if node.label in previous:
                earlier, position = previous[node.label]
                left, right = earlier, idx
                while depths[left] > depths[right]:
                    left = parents[left]
                while depths[right] > depths[left]:
                    right = parents[right]
                while left != right:
                    left, right = parents[left], parents[right]
                repeats[left].append((position - first_local[left], ordinal - first_local[left]))
            previous[node.label] = (idx, ordinal)
            ordinal += 1
        stack.extend((child, idx) for child in reversed(node.children))
    hashes: list[str] = [""] * len(nodes)
    for idx in range(len(nodes) - 1, -1, -1):
        label = "local" if re.fullmatch(r"v\d+", nodes[idx].label) else nodes[idx].label
        values = [namespace, label, *(hashes[c] for c in children[idx]), repr(repeats[idx])]
        hashes[idx] = hashlib.blake2b("\x1f".join(values).encode(), digest_size=16).hexdigest()
    return tuple(sorted(set(hashes)))


def jaccard_pairs(
    fingerprints: Sequence[Iterable[str]],
    threshold: float,
    *,
    max_candidates: int = MAX_PAIRS,
    eligible: Callable[[int, int], bool] | None = None,
) -> list[tuple[int, int, float]]:
    """Exact threshold join with rare-first prefixes and a bounded candidate budget."""
    if not math.isfinite(threshold) or not 0 < threshold <= 1 or max_candidates < 0:
        raise ValueError("Jaccard threshold and candidate budget are invalid")
    sets = [set(values) for values in fingerprints]
    counts = Counter(value for values in sets for value in values)
    ranked = [sorted(values, key=lambda value: (counts[value], value)) for values in sets]
    index: dict[str, list[int]] = defaultdict(list)
    compared = 0
    pairs = []
    for i in sorted(range(len(sets)), key=lambda i: (len(sets[i]), i)):
        size = len(sets[i])
        if not size:
            continue
        needed = math.ceil(threshold * size - 1e-9)
        prefix = ranked[i][: min(size, size - needed + 1)]
        seen = set()
        for token in prefix:
            for j in index[token]:
                if j in seen or len(sets[j]) < needed:
                    continue
                seen.add(j)
                if eligible is not None and not eligible(i, j):
                    continue
                compared += 1
                if compared > max_candidates:
                    raise TooManyPairsError(
                        f"over {max_candidates:,} AST candidate pairs; raise --min-nodes"
                    )
                intersection = len(sets[i] & sets[j])
                score = intersection / (size + len(sets[j]) - intersection)
                if score >= threshold:
                    pairs.append((min(i, j), max(i, j), score))
        for token in prefix:
            index[token].append(i)
    return sorted(pairs)

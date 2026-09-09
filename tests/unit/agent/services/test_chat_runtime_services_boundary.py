"""Conservative, receiver-name-independent ratchets for FEP-0031 migration.

Counts include runtime-local private accesses, not just facade references. This
prevents a receiver rename or a previously unknown private attribute from hiding
new coupling. It is an inventory/shrink guard, not a claim of zero coupling.
"""

import ast
from pathlib import Path

import pytest

CLUSTER_CAPS = {
    "chat_stream_runtime.py": {
        "private_attributes": 48,
        "private_probes": 5,
        "dynamic_probes": 0,
        "raw_state": 11,
    },
    "chat_stream_executor.py": {
        "private_attributes": 94,
        "private_probes": 17,
        "dynamic_probes": 4,
        "raw_state": 0,
    },
    "chat_stream_helpers.py": {
        "private_attributes": 99,
        "private_probes": 26,
        "dynamic_probes": 0,
        "raw_state": 8,
    },
    "streaming_act_adapter.py": {
        "private_attributes": 14,
        "private_probes": 1,
        "dynamic_probes": 0,
        "raw_state": 0,
    },
}
CLUSTER = (
    "chat_stream_runtime.py",
    "chat_stream_executor.py",
    "chat_stream_helpers.py",
    "streaming_act_adapter.py",
)
ROOT = Path(__file__).resolve().parents[4]


def inventory(source):
    tree = ast.parse(source)
    builtins = {"getattr", "setattr", "hasattr", "delattr", "vars"}
    names = {name: name for name in builtins}
    constants = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "builtins":
            names.update(
                {
                    alias.asname or alias.name: alias.name
                    for alias in node.names
                    if alias.name in builtins
                }
            )

    def literal(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            return constants.get(node.id)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left, right = literal(node.left), literal(node.right)
            return left + right if left is not None and right is not None else None
        return None

    # Resolve simple aliases/constants independently of receiver spelling and
    # declaration order. Include assignments to self fields.
    for _ in range(3):
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    key = ast.unparse(target)
                    if literal(node.value) is not None:
                        constants[key] = literal(node.value)
                    if ast.unparse(node.value) in names:
                        names[key] = names[ast.unparse(node.value)]
                    elif isinstance(node.value, ast.Attribute) and node.value.attr in builtins:
                        names[key] = node.value.attr

    counts = {"private_attributes": 0, "private_probes": 0, "dynamic_probes": 0, "raw_state": 0}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            counts["private_attributes"] += 1
            counts["raw_state"] += node.attr == "__dict__"
        if isinstance(node, ast.Subscript):
            key = literal(node.slice)
            if key and key.startswith("_"):
                counts["raw_state"] += 1
        if not isinstance(node, ast.Call):
            continue
        function = names.get(ast.unparse(node.func))
        if isinstance(node.func, ast.Attribute) and node.func.attr in builtins:
            function = node.func.attr
        if function == "vars":
            counts["raw_state"] += 1
        elif function in builtins and len(node.args) >= 2:
            key = literal(node.args[1])
            if key is None:
                counts["dynamic_probes"] += 1
            elif key.startswith("_"):
                counts["private_probes"] += 1
                counts["raw_state"] += key == "__dict__"
        elif isinstance(node.func, ast.Attribute) and node.func.attr == "get" and node.args:
            key = literal(node.args[0])
            if key and key.startswith("_"):
                counts["raw_state"] += 1
    return counts


@pytest.mark.parametrize("filename", CLUSTER)
def test_chat_cluster_boundary_counts_only_shrink(filename):
    measured = inventory((ROOT / "victor/agent/services" / filename).read_text())
    for category, count in measured.items():
        assert count <= CLUSTER_CAPS[filename][category], (
            f"{filename} {category}: {count} exceeds {CLUSTER_CAPS[filename][category]}; "
            "migrate through an explicit service capability rather than raising this cap"
        )


@pytest.mark.parametrize(
    "source, category",
    [
        ("renamed._required_files = []", "private_attributes"),
        ("alias = original\nalias._new_facade_private()", "private_attributes"),
        ("self.host = original\nself.host._required_files", "private_attributes"),
        ("getattr(renamed, '_required_files')", "private_probes"),
        (
            "from builtins import getattr as read\nread(renamed, '_required_files')",
            "private_probes",
        ),
        ("read = getattr\nkey = '_' + 'required_files'\nread(renamed, key)", "private_probes"),
        (
            "import builtins as b\nread = b.getattr\nread(renamed, '_required_files')",
            "private_probes",
        ),
        ("setattr(renamed, key_from_elsewhere, [])", "dynamic_probes"),
        ("vars(renamed)['_required_files']", "raw_state"),
        ("renamed.__dict__.get('_required_files')", "raw_state"),
    ],
)
def test_boundary_inventory_detects_aliases_and_dynamic_bypasses(source, category):
    assert inventory(source)[category] > 0


def test_migrated_requirement_consumer_has_no_private_or_dynamic_access():
    tree = ast.parse((ROOT / "victor/agent/services/chat_stream_executor.py").read_text())
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_extract_task_requirements"
    )
    assert not any(inventory(ast.unparse(method)).values())

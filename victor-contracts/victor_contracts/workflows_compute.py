"""Compute-handler envelope kernel shared by every vertical's ``handlers``.

Each compute handler repeats the same envelope around its domain logic:
start a timer, resolve ``output_key = node.output_key or node.id``, store the
output in the context, and build a ``NodeResult`` — fail-closed on exceptions.
The envelope lives here once; handlers keep only their domain logic::

    from victor_contracts.workflows_compute import ComputeStep

    @dataclass
    class MyHandler:
        async def __call__(self, node, context, tool_registry) -> NodeResult:
            step = ComputeStep(node, context)
            try:
                result = await tool_registry.execute("shell", command=build_cmd(node))
                return step.complete(
                    {"passed": result.success},
                    status=COMPLETED if result.success else FAILED,
                )
            except Exception as e:
                return step.fail(str(e))

Also provides ``make_handler_registrar`` — the ``register_handlers`` tail every
vertical repeats, with an optional negative check that the GLOBAL registry is
never mutated.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, Mapping, Optional

from victor_contracts.workflows import (
    ComputeHandlerRegistrar,
    ExecutorNodeStatus,
    NodeResult,
    register_compute_handlers,
)

if TYPE_CHECKING:
    from victor_contracts.workflows import (
        ComputeNodeProtocol as ComputeNode,
        WorkflowContextProtocol as WorkflowContext,
    )

logger = logging.getLogger(__name__)


class ComputeStep:
    """Timing, output-key, and NodeResult envelope for one compute step.

    Fail policy is the caller's choice via ``status`` on :meth:`complete`
    (some handlers always COMPLETE with failure detail in the output; others
    FAIL the node on a failed tool result). Exceptions map to
    :meth:`fail` — fail-closed, never a partial success.
    """

    def __init__(self, node: "ComputeNode", context: "WorkflowContext") -> None:
        self._node = node
        self._context = context
        self._start = time.time()
        self._output_key = node.output_key or node.id

    @property
    def elapsed(self) -> float:
        """Seconds since step start."""
        return time.time() - self._start

    def complete(
        self,
        output: Dict[str, Any],
        *,
        status: ExecutorNodeStatus = ExecutorNodeStatus.COMPLETED,
        tool_calls_used: int = 1,
        error: Optional[str] = None,
    ) -> NodeResult:
        """Store ``output`` under the step's output key and build the result.

        ``error`` is for the dict-shaped result pattern where a handler
        completes with BOTH a payload and a domain error field (e.g. MLOps:
        ``{"success": False, "error": ...}`` with FAILED status).
        """
        self._context.set(self._output_key, output)
        return NodeResult(
            node_id=self._node.id,
            status=status,
            output=output,
            duration_seconds=self.elapsed,
            tool_calls_used=tool_calls_used,
            error=error,
        )

    def fail(self, error: str) -> NodeResult:
        """Build a FAILED result without touching the context."""
        return NodeResult(
            node_id=self._node.id,
            status=ExecutorNodeStatus.FAILED,
            error=error,
            duration_seconds=self.elapsed,
        )


def make_handler_registrar(
    vertical_name: str,
    handlers: Mapping[str, Callable[..., Any]],
) -> Callable[..., Any]:
    """Build a vertical's ``register_handlers(registrar=None)`` tail.

    Always registers through the explicit host-side registrar
    (:func:`victor_contracts.workflows.register_compute_handlers`) — never the
    global registry, which pollutes across tests and sessions.
    """

    def register_handlers(
        registrar: Optional[ComputeHandlerRegistrar] = None,
    ) -> Any:
        registered = register_compute_handlers(registrar, handlers)
        if registrar is not None:
            for name in handlers:
                logger.debug("Registered %s handler: %s", vertical_name, name)
        return registered

    register_handlers.__doc__ = (
        f"Register {vertical_name} handlers through an explicit host-side registrar."
    )
    register_handlers.__name__ = "register_handlers"
    return register_handlers

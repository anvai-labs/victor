"""SDK host adapters for handler-registration runtime helpers."""

from __future__ import annotations

import importlib
import time
import warnings
from typing import TYPE_CHECKING, Any

from victor_contracts._deprecation import deprecated_bridge_message

if TYPE_CHECKING:
    from victor.framework.handler_registry import (
        HandlerRegistry,
        get_handler_registry,
        handler_decorator,
        register_global_handler,
        register_vertical_handlers,
        sync_handlers_with_executor,
    )
__all__ = [
    "BaseHandler",  # noqa: F822
    "HandlerRegistry",
    "get_handler_registry",
    "handler_decorator",  # noqa: F822
    "register_global_handler",
    "register_vertical_handlers",
    "sync_handlers_with_executor",
]

_LAZY_IMPORTS = {
    "HandlerRegistry": "victor.framework.handler_registry",
    "get_handler_registry": "victor.framework.handler_registry",
    "handler_decorator": "victor.framework.handler_registry",
    "register_global_handler": "victor.framework.handler_registry",
    "register_vertical_handlers": "victor.framework.handler_registry",
    "sync_handlers_with_executor": "victor.framework.handler_registry",
}


def __getattr__(name: str) -> Any:
    """Resolve handler helpers lazily from the Victor host runtime."""
    warnings.warn(
        deprecated_bridge_message("handler_runtime"),
        DeprecationWarning,
        stacklevel=2,
    )
    if name == "BaseHandler":
        return _load_base_handler()
    if name == "handler_decorator":
        return _load_handler_decorator()

    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'victor_contracts.handler_runtime' has no attribute {name!r}")

    module = importlib.import_module(module_name)
    return getattr(module, name)


def _load_base_handler() -> type:
    """Resolve BaseHandler from the host runtime or provide a compatibility shim."""
    try:
        module = importlib.import_module("victor.framework.workflows.base_handler")
        return module.BaseHandler  # type: ignore[no-any-return]
    except Exception:

        class BaseHandler:
            """Compatibility base class for class-based workflow handlers."""

            async def execute(self, node: Any, context: Any, tool_registry: Any) -> Any:
                raise NotImplementedError("BaseHandler.execute() must be implemented")

            async def __call__(self, node: Any, context: Any, tool_registry: Any) -> Any:
                from victor_contracts.workflow_executor_runtime import (
                    ExecutorNodeStatus,
                    NodeResult,
                )

                start_time = time.time()
                try:
                    output, tool_calls_used = await self.execute(node, context, tool_registry)

                    output_key = getattr(node, "output_key", None) or getattr(node, "id", "output")
                    if hasattr(context, "set"):
                        context.set(output_key, output)
                    elif isinstance(context, dict):
                        context[output_key] = output

                    return NodeResult(
                        node_id=getattr(node, "id", "unknown"),
                        status=ExecutorNodeStatus.COMPLETED,
                        output=output,
                        duration_seconds=time.time() - start_time,
                        tool_calls_used=int(tool_calls_used or 0),
                    )
                except Exception as exc:
                    return NodeResult(
                        node_id=getattr(node, "id", "unknown"),
                        status=ExecutorNodeStatus.FAILED,
                        error=str(exc),
                        duration_seconds=time.time() - start_time,
                        tool_calls_used=0,
                    )

        return BaseHandler


def _load_handler_decorator() -> Any:
    """Resolve handler_decorator from the host runtime or provide a compatibility shim."""
    try:
        module = importlib.import_module("victor.framework.handler_registry")
        return module.handler_decorator
    except Exception:

        def handler_decorator(
            name: str,
            *,
            vertical: str | None = None,
            description: str | None = None,
        ) -> Any:
            """Decorator compatibility shim for class-based handlers."""

            def _decorator(handler_cls: type) -> type:
                try:
                    instance = handler_cls()
                except Exception:
                    return handler_cls

                try:
                    registry_module = importlib.import_module("victor.framework.handler_registry")
                    if vertical:
                        registry_module.register_vertical_handlers(
                            vertical_name=vertical,
                            handlers={name: instance},
                            category="general",
                            description=description or "",
                        )
                    else:
                        registry_module.register_global_handler(
                            name=name,
                            handler=instance,
                            category="global",
                        )
                except Exception:
                    pass

                try:
                    from victor_contracts.workflow_executor_runtime import (
                        register_compute_handler,
                    )

                    register_compute_handler(name, instance)
                except Exception:
                    pass

                return handler_cls

            return _decorator

        return handler_decorator

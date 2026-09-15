"""Captured Popen ownership and bounded admission for non-cancellable stdio."""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import threading
from typing import Any, TYPE_CHECKING

from victor.config.timeouts import McpTimeouts

if TYPE_CHECKING:
    from victor.integrations.mcp.sandbox import SandboxedProcess

logger = logging.getLogger(__name__)


class StdioTransport:
    """One process/owner pair, never redirected to a replacement connection.

    Popen's pipe I/O cannot be portably cancelled on Python 3.11. A daemon
    worker owns it instead of the default executor, whose shutdown would wait
    forever for inherited pipes. The client quarantines an unsettled worker
    and refuses new connections until it drains, bounding stranded resources.
    """

    def __init__(
        self, process: subprocess.Popen | None, owner: SandboxedProcess | None = None
    ) -> None:
        self.process = process
        self.owner = owner
        self.retired = threading.Event()
        self.worker_done = threading.Event()
        self.worker_done.set()
        self.cleanup_done = threading.Event()
        self.cleanup_task: asyncio.Task[None] | None = None
        self._close_lock = threading.Lock()
        self._closed = False

    def exchange(self, request: str, message_id: str) -> asyncio.Future[dict[str, Any]]:
        if self.retired.is_set() or not self.worker_done.is_set():
            raise RuntimeError("MCP transport is not available for another request")
        process = self.process
        if process is None or process.stdin is None or process.stdout is None:
            raise RuntimeError("MCP transport has no stdio")
        loop = asyncio.get_running_loop()
        result: asyncio.Future[dict[str, Any]] = loop.create_future()
        self.worker_done.clear()

        def complete(response: dict[str, Any] | None, error: Exception | None) -> None:
            if result.done():
                return
            if error is not None:
                result.set_exception(error)
            elif response is not None:
                result.set_result(response)

        def work() -> None:
            response = None
            error = None
            try:
                process.stdin.write(request + "\n")
                process.stdin.flush()
                while not self.retired.is_set():
                    line = process.stdout.readline()
                    if not line:
                        raise EOFError("MCP transport closed before response")
                    candidate = json.loads(line)
                    if not isinstance(candidate, dict):
                        raise ValueError("MCP response must be an object")
                    if candidate.get("id") != message_id:
                        continue
                    if "result" not in candidate and "error" not in candidate:
                        raise ValueError("MCP response lacks result/error")
                    response = candidate
                    break
                if response is None:
                    raise RuntimeError("MCP transport was retired")
            except Exception as exc:
                error = exc
            finally:
                # This worker has released its TextIO locks; no other exchange
                # can start on a retired transport. Never close from the loop.
                self.worker_done.set()
                # Publish released I/O locks before checking retirement. Either
                # cleanup now sees completion or this worker observes retirement.
                if self.retired.is_set():
                    self._close_streams()
                try:
                    loop.call_soon_threadsafe(complete, response, error)
                except RuntimeError:
                    pass  # The owning loop has already closed.

        try:
            threading.Thread(target=work, name="victor-mcp-stdio", daemon=True).start()
        except BaseException:
            self.worker_done.set()
            result.cancel()
            raise
        return result

    def settled(self) -> bool:
        return (
            self.cleanup_done.is_set()
            and self.worker_done.is_set()
            and (self.process is None or self._closed)
            and (self.process is None or self.process.poll() is not None)
        )

    def retire(self) -> None:
        if self.retired.is_set():
            return
        self.retired.set()
        if self.process is not None:
            try:
                self.process.terminate()
            except Exception as exc:
                logger.debug("Error terminating retired MCP process: %s", exc)

    def _close_streams(self) -> None:
        with self._close_lock:
            if self._closed or self.process is None:
                return
            for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except Exception as exc:
                        logger.debug("Error closing retired MCP pipe: %s", exc)
            self._closed = True

    def wait_and_close(self) -> None:
        if self.process is not None:
            try:
                self.process.wait(timeout=McpTimeouts.TERMINATE)
            except subprocess.TimeoutExpired:
                try:
                    self.process.kill()
                    self.process.wait(timeout=McpTimeouts.KILL)
                except Exception as exc:
                    logger.debug("Error killing retired MCP process: %s", exc)
            except Exception as exc:
                logger.debug("Error waiting for retired MCP process: %s", exc)
        # A descendant may retain a pipe after the direct child exits. In that
        # case the daemon worker owns deferred closure, and admission stays shut.
        if self.worker_done.is_set():
            self._close_streams()
        self.cleanup_done.set()

    async def cleanup(self) -> None:
        try:
            if self.owner is not None:
                await self.owner.terminate_all()
        except Exception as exc:
            logger.debug("Error terminating captured MCP sandbox: %s", exc)
        finally:
            await asyncio.to_thread(self.wait_and_close)

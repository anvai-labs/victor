import asyncio
import argparse
from dataclasses import dataclass
import sys

from victor.tools.base import AccessMode, DangerLevel, ExecutionCategory, Priority
from victor.tools.decorators import tool
from victor.agent.background_tasks import BackgroundTaskDef
from victor.tools.unified.parser import split_command


async def execute_bash(cmd: str, sandbox: bool = False, readonly: bool = False):
    """Execute through the production shell tool implementation."""
    from victor.tools.bash import shell

    return await shell(
        cmd=cmd,
        readonly=readonly,
        action="read" if readonly else "write",
    )


class UnifiedShellParser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        raise ValueError(f"Argument parsing error: {message}")


def create_shell_parser() -> UnifiedShellParser:
    parser = UnifiedShellParser(
        prog="shell", description="Unified shell operations.", exit_on_error=False
    )
    # The shell wrapper is simpler: it takes a command string and optional flags
    parser.add_argument("cmd", help="The bash command to run")
    parser.add_argument("--sandbox", action="store_true", help="Run the command inside a sandbox")
    parser.add_argument("--readonly", action="store_true", help="Enforce readonly limits")
    parser.add_argument(
        "--timebound-sync",
        type=int,
        default=None,
        help="Max time (in seconds) to wait synchronously before backgrounding or timing out.",
    )

    return parser


@dataclass
class ParsedShellCommand:
    cmd: str
    sandbox: bool = False
    readonly: bool = False
    timebound_sync: int | None = None


def _strip_shell_prefix(command: str) -> str:
    stripped = command.lstrip()
    if stripped == "shell":
        return ""
    if stripped.startswith("shell "):
        return stripped[len("shell ") :]
    return command


def _consume_prefix_flags(command: str) -> tuple[ParsedShellCommand, str]:
    parsed = ParsedShellCommand(cmd="")
    remaining = command.lstrip()

    while remaining.startswith("--"):
        if remaining.startswith("--sandbox"):
            after = remaining[len("--sandbox") :]
            if after and not after[0].isspace():
                break
            parsed.sandbox = True
            remaining = after.lstrip()
            continue
        if remaining.startswith("--readonly"):
            after = remaining[len("--readonly") :]
            if after and not after[0].isspace():
                break
            parsed.readonly = True
            remaining = after.lstrip()
            continue
        if remaining.startswith("--timebound-sync"):
            after = remaining[len("--timebound-sync") :].lstrip()
            if not after:
                raise ValueError("Argument parsing error: --timebound-sync requires seconds")
            parts = after.split(maxsplit=1)
            try:
                parsed.timebound_sync = int(parts[0])
            except ValueError as exc:
                raise ValueError(
                    "Argument parsing error: --timebound-sync requires integer seconds"
                ) from exc
            remaining = parts[1].lstrip() if len(parts) > 1 else ""
            continue
        break

    return parsed, remaining


def parse_shell_command(command: str) -> ParsedShellCommand:
    """Parse Victor wrapper flags while preserving the bash command body."""
    raw = _strip_shell_prefix(command)
    parsed, remaining = _consume_prefix_flags(raw)

    # Backward compatibility for the old form:
    #   shell "python script.py" --sandbox
    # Only use shlex-style parsing when the command body is a single quoted arg,
    # so arbitrary bash syntax such as pipes, redirection, and heredocs stays raw.
    if remaining[:1] in {"'", '"'} and "\n" not in remaining and "<<" not in remaining:
        args = split_command(remaining)
        command_parts = []
        for arg in args:
            if arg == "--sandbox":
                parsed.sandbox = True
            elif arg == "--readonly":
                parsed.readonly = True
            elif arg == "--timebound-sync":
                # Leave malformed legacy suffixes to the argparse-compatible
                # error below by appending this token as command content.
                command_parts.append(arg)
            elif command_parts and command_parts[-1] == "--timebound-sync":
                try:
                    parsed.timebound_sync = int(arg)
                    command_parts.pop()
                except ValueError:
                    command_parts.append(arg)
            else:
                command_parts.append(arg)
        parsed.cmd = " ".join(command_parts).strip()
    else:
        parsed.cmd = remaining.strip()

    if not parsed.cmd:
        raise ValueError("Argument parsing error: missing command")
    return parsed


@tool(
    name="shell",
    access_mode=AccessMode.MIXED,
    danger_level=DangerLevel.HIGH,
    execution_category=ExecutionCategory.MIXED,
    priority=Priority.HIGH,
)
async def shell_tool(cmd: str) -> str:
    """Unified shell tool.
    Example commands:
      shell "ls -la"
      shell "npm start" --timebound-sync 300
    """
    try:
        parsed_args = parse_shell_command(cmd)
    except ValueError as e:
        return f"### ❌ ERROR\n{e}"
    except Exception as e:
        return f"### ❌ ERROR\nUnexpected error: {e}"

    try:
        coro = execute_bash(
            cmd=parsed_args.cmd,
            sandbox=parsed_args.sandbox,
            readonly=parsed_args.readonly,
        )

        # Determine if we should enforce a timebound sync
        if parsed_args.timebound_sync is not None:
            task = asyncio.create_task(coro)
            done, pending = await asyncio.wait([task], timeout=parsed_args.timebound_sync)

            if pending:
                # Task is still running after timeout
                # If we are in an agent environment (watcher exists), we background it.
                # For this prototype, we'll assume the watcher is active and return the Def.
                has_watcher = True
                if has_watcher:
                    return BackgroundTaskDef(task=task, context=f"shell_tool: {parsed_args.cmd}")
                else:
                    task.cancel()
                    return f"### ❌ ERROR\nCommand timed out after {parsed_args.timebound_sync}s"

            # Task completed
            results = task.result()
        else:
            # Traditional synchronous wait
            results = await coro

        if not isinstance(results, dict):
            return str(results)

        stdout = results.get("stdout", "").strip()
        stderr = results.get("stderr", "").strip()

        out = []
        if stdout:
            out.append(f"### STDOUT\n```text\n{stdout}\n```")
        if stderr:
            out.append(f"### STDERR\n```text\n{stderr}\n```")

        if not out:
            return "Command executed successfully with no output."
        return "\n\n".join(out)
    except Exception as e:
        return f"### ❌ ERROR\nShell execution failed: {e}"

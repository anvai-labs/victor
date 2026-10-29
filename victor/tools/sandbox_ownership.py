# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Owner labels for Victor sandbox containers and the liveness check reapers use.

Every long-lived sandbox container (code-executor sandboxes, evaluation
containers) records which process owns it: the kernel boot id and PID
namespace, the owner PID and that PID's kernel start time (so a reused PID is
not mistaken for the owner). A reaper removes a container only when
:func:`owner_gone` proves its owner in this boot and namespace is dead;
containers of another host or namespace, or of a live process, are never
touched.

Stdlib only, so both the core evaluation harness and the coding vertical can
share one definition.
"""

from __future__ import annotations

import os
import socket
import time
from datetime import datetime
from pathlib import Path
from typing import Mapping, Optional

OWNER_HOST_LABEL = "victor.owner.host"
OWNER_PID_LABEL = "victor.owner.pid"
OWNER_START_LABEL = "victor.owner.start"


def host_identity() -> str:
    """Identity of this kernel boot and PID namespace; other owners are never reaped.

    The PID namespace matters because a PID only means something inside its
    namespace: WSL2 distros (and containers driving the host's Docker) share the
    kernel's boot id, so a PID that looks dead here may be a live owner there.
    """
    try:
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return socket.gethostname()
    try:
        return f"{boot}:{os.readlink('/proc/self/ns/pid')}"
    except OSError:
        return boot


def process_start_marker(pid: int) -> str:
    """Kernel start time of ``pid`` (distinguishes a reused PID); "" if unknown."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return ""
    # comm (field 2) may contain spaces and parens: split after the last ')'.
    fields = stat.rsplit(")", 1)[1].split()
    return fields[19] if len(fields) > 19 else ""  # field 22: starttime


def owner_alive(pid: int, start: str) -> bool:
    """True unless ``pid`` is provably gone or now belongs to another process."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass  # exists, owned by another user
    except OSError:
        return True  # cannot tell: never reap on doubt
    current = process_start_marker(pid)
    return not (start and current and current != start)


def owner_labels() -> dict[str, str]:
    """Owner labels identifying the current process."""
    pid = os.getpid()
    return {
        OWNER_HOST_LABEL: host_identity(),
        OWNER_PID_LABEL: str(pid),
        OWNER_START_LABEL: process_start_marker(pid),
    }


def owner_gone(labels: Mapping[str, str]) -> Optional[bool]:
    """Whether the owner recorded in ``labels`` is provably dead.

    Returns ``None`` for a container without owner labels (the caller decides
    how to treat legacy containers), ``False`` for an owner on another host or
    one that is still alive, and ``True`` only for a dead owner on this host.
    """
    host = labels.get(OWNER_HOST_LABEL)
    if host is None:
        return None
    if host != host_identity():
        return False
    try:
        pid = int(labels.get(OWNER_PID_LABEL, "0"))
    except ValueError:
        pid = 0
    return not owner_alive(pid, labels.get(OWNER_START_LABEL, ""))


def created_age_seconds(created: str) -> float:
    """Age of a container from Docker's ``Created`` timestamp; 0.0 if unparsable."""
    try:
        # Docker reports nanoseconds; fromisoformat accepts at most microseconds.
        head, _, frac = (created or "").rstrip("Z").partition(".")
        stamp = datetime.fromisoformat(f"{head}.{(frac + '000000')[:6]}+00:00")
    except ValueError:
        return 0.0
    return time.time() - stamp.timestamp()

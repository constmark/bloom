#!/usr/bin/env python3
"""Run a command with a portable deadline (exit 124 on timeout).

On POSIX, the command and inherited descendants share a private process group
so a worker holding an output pipe cannot outlive the timed command.
"""
from __future__ import annotations

import argparse
import math
import os
import signal
import subprocess
import sys
import time


def _signal_group(group: int, signum: int) -> None:
    # Darwin's killpg1 filters zombies but can still find their process group,
    # returning EPERM until the group is reaped. Retry that short exit race;
    # a persistent permission error remains a cleanup failure.
    deadline = time.monotonic() + .2
    while True:
        try:
            os.killpg(group, signum)
            return
        except ProcessLookupError:
            return
        except PermissionError:
            if sys.platform != "darwin" or time.monotonic() >= deadline:
                raise
            time.sleep(.01)


def terminate(process: subprocess.Popen, grace: float = 2) -> None:
    if os.name == "posix":
        _signal_group(process.pid, signal.SIGTERM)
    elif process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    finally:
        if os.name == "posix":
            _signal_group(process.pid, signal.SIGKILL)
        elif process.poll() is None:
            process.kill()
        process.wait(timeout=5)


def run(command: list[str], seconds: float, parent_pid: int | None = None) -> int:
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("timeout must be finite and positive")
    if parent_pid is not None and (os.name != "posix" or os.getppid() != parent_pid):
        return 125
    process = subprocess.Popen(command, start_new_session=os.name == "posix")
    end = time.monotonic() + seconds
    try:
        while True:
            if parent_pid is not None and os.getppid() != parent_pid:
                return 125
            remaining = end - time.monotonic()
            if remaining <= 0:
                return 124
            try:
                return process.wait(timeout=min(remaining, .2) if parent_pid else remaining)
            except subprocess.TimeoutExpired:
                continue
    finally:
        # A command can exit while one of its workers still owns inherited
        # output pipes. Success and failure both release the entire group.
        terminate(process)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-pid", type=int,
                        help="POSIX owner to monitor; exit 125 and clean up if it disappears")
    parser.add_argument("seconds", type=float)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.command:
        parser.error("a command is required")
    def interrupted(signum, _frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    try:
        return run(args.command, args.seconds, args.parent_pid)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

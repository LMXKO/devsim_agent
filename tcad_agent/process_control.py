from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Any, Sequence


DEFAULT_CANCEL_ENV = "ACTSOFT_CANCEL_FILE"


def cancel_file_from_env() -> Path | None:
    raw = os.environ.get(DEFAULT_CANCEL_ENV)
    return Path(raw) if raw else None


def cancel_requested(cancel_file: Path | str | None) -> bool:
    if not cancel_file:
        return False
    return Path(cancel_file).exists()


def run_cancellable(
    command: Sequence[str],
    *,
    cwd: Path | str | None = None,
    capture_output: bool = True,
    text: bool = True,
    timeout: float | None = None,
    check: bool = False,
    cancel_file: Path | str | None = None,
    poll_interval_seconds: float = 0.25,
    terminate_grace_seconds: float = 2.0,
    **kwargs: Any,
) -> subprocess.CompletedProcess:
    """Run a subprocess while polling an ActSoft cancel token.

    The return shape intentionally matches `subprocess.run` for existing tool
    callers. Cancellation returns a non-zero completed process with stderr text
    explaining the cooperative termination.
    """
    actual_cancel_file = Path(cancel_file) if cancel_file else cancel_file_from_env()
    stdout = subprocess.PIPE if capture_output else None
    stderr = subprocess.PIPE if capture_output else None
    start = time.monotonic()
    process = subprocess.Popen(
        list(command),
        cwd=str(cwd) if cwd is not None else None,
        stdout=stdout,
        stderr=stderr,
        text=text,
        **kwargs,
    )
    while True:
        try:
            out, err = process.communicate(timeout=poll_interval_seconds)
            completed = subprocess.CompletedProcess(list(command), process.returncode, out, err)
            if check and completed.returncode:
                raise subprocess.CalledProcessError(completed.returncode, list(command), output=out, stderr=err)
            return completed
        except subprocess.TimeoutExpired:
            if timeout is not None and time.monotonic() - start >= timeout:
                process.terminate()
                try:
                    out, err = process.communicate(timeout=terminate_grace_seconds)
                except subprocess.TimeoutExpired:
                    process.kill()
                    out, err = process.communicate()
                raise subprocess.TimeoutExpired(list(command), timeout, output=out, stderr=err)
            if cancel_requested(actual_cancel_file):
                process.terminate()
                try:
                    out, err = process.communicate(timeout=terminate_grace_seconds)
                except subprocess.TimeoutExpired:
                    process.kill()
                    out, err = process.communicate()
                marker = f"ACTSOFT_CANCELLED: cancel token observed at {actual_cancel_file}"
                if text:
                    err = ((err or "") + ("\n" if err else "") + marker)
                return subprocess.CompletedProcess(list(command), process.returncode or -15, out, err)

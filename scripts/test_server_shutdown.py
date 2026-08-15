#!/usr/bin/env python3
"""Exercise bloom_server SIGTERM, catalog ownership, escalation, and drain."""

from __future__ import annotations

import http.client
import os
import pathlib
import re
import signal
import socket
import subprocess
import tempfile
import time
import traceback


ROOT = pathlib.Path(__file__).resolve().parents[1]
SERVER_OVERRIDE = os.environ.get("BLOOM_TEST_SERVER_BINARY")
if SERVER_OVERRIDE:
    override_path = pathlib.Path(SERVER_OVERRIDE)
    SERVER = (
        override_path if override_path.is_absolute() else ROOT / override_path
    ).resolve()
else:
    SERVER = ROOT / "target" / "debug" / "bloom_server"
STARTUP_PATTERN = re.compile(r"server running on http://127\.0\.0\.1:(\d+)")
ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-9;]*m")


def read_log(path: pathlib.Path) -> str:
    try:
        return ANSI_ESCAPE_PATTERN.sub(
            "", path.read_text(encoding="utf-8", errors="replace")
        )
    except FileNotFoundError:
        return ""


def wait_for_server(process: subprocess.Popen[bytes], log_path: pathlib.Path) -> int:
    deadline = time.monotonic() + 10
    port = None
    while time.monotonic() < deadline:
        log = read_log(log_path)
        if port is None:
            match = STARTUP_PATTERN.search(log)
            if match:
                port = int(match.group(1))
        if port is not None:
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=0.5)
            try:
                connection.request("GET", "/health")
                response = connection.getresponse()
                response.read()
                if response.status == 200:
                    return port
            except (OSError, http.client.HTTPException):
                pass
            finally:
                connection.close()
        status = process.poll()
        if status is not None:
            raise AssertionError(
                f"bloom_server exited with status {status} before startup:\n{log}"
            )
        time.sleep(0.05)
    raise AssertionError(
        f"bloom_server did not become healthy within 10 seconds:\n{read_log(log_path)}"
    )


def start_server(
    directory: pathlib.Path, drain_seconds: int
) -> tuple[subprocess.Popen[bytes], pathlib.Path, int]:
    models = directory / "models"
    log_path = directory / "server.log"
    log_handle = log_path.open("wb")
    environment = os.environ.copy()
    environment["RUST_LOG"] = "bloom_server=info,tower_http=debug"
    try:
        process = subprocess.Popen(
            [
                str(SERVER),
                "--models-dir",
                str(models),
                "--port",
                "0",
                "--shutdown-timeout-seconds",
                str(drain_seconds),
            ],
            cwd=ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=environment,
        )
    finally:
        log_handle.close()
    return process, log_path, wait_for_server(process, log_path)


def wait_for_log(
    process: subprocess.Popen[bytes], log_path: pathlib.Path, marker: str
) -> str:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        log = read_log(log_path)
        if marker in log:
            return log
        status = process.poll()
        if status is not None:
            raise AssertionError(
                f"bloom_server exited with status {status} before logging {marker!r}:\n{log}"
            )
        time.sleep(0.02)
    raise AssertionError(
        f"bloom_server did not log {marker!r} within 5 seconds:\n{read_log(log_path)}"
    )


def open_active_request(
    process: subprocess.Popen[bytes], log_path: pathlib.Path, port: int, request_id: str
) -> socket.socket:
    """Open a request whose incomplete body keeps graceful shutdown draining."""
    connection = socket.create_connection(("127.0.0.1", port), timeout=2)
    connection.sendall(
        (
            "POST /v1/chat/completions HTTP/1.1\r\n"
            "Host: localhost\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: 1024\r\n"
            f"X-Request-Id: {request_id}\r\n"
            "\r\n"
            "{"
        ).encode("ascii")
    )
    try:
        wait_for_log(process, log_path, f'request_id="{request_id}"')
    except BaseException:
        connection.close()
        raise
    return connection


def stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2)


def assert_catalog_owned_during_drain(directory: pathlib.Path) -> None:
    log_path = directory / "drain-catalog-conflict.log"
    with log_path.open("wb") as log_handle:
        contender = subprocess.Popen(
            [
                str(SERVER),
                "--models-dir",
                str(directory / "models"),
                "--port",
                "0",
            ],
            cwd=ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
    try:
        status = contender.wait(timeout=5)
    except subprocess.TimeoutExpired as error:
        stop_process(contender)
        raise AssertionError(
            "a contender acquired the catalog while the owner drained HTTP"
        ) from error
    log = read_log(log_path)
    if status == 0 or "already owned by another Bloom server" not in log:
        raise AssertionError(
            f"catalog ownership was not retained during drain ({status}):\n{log}"
        )


def test_clean_sigterm(directory: pathlib.Path) -> None:
    process, log_path, _port = start_server(directory, 2)
    try:
        process.send_signal(signal.SIGTERM)
        status = process.wait(timeout=5)
        log = read_log(log_path)
        if status != 0:
            raise AssertionError(f"clean SIGTERM exited with status {status}:\n{log}")
        if 'signal="terminate"' not in log or "Server shut down gracefully" not in log:
            raise AssertionError(f"clean SIGTERM log is incomplete:\n{log}")
    finally:
        stop_process(process)


def test_forced_deadline(directory: pathlib.Path) -> None:
    process, log_path, port = start_server(directory, 1)
    connection = open_active_request(
        process, log_path, port, "shutdown-deadline-request"
    )
    try:
        process.send_signal(signal.SIGTERM)
        status = process.wait(timeout=5)
        log = read_log(log_path)
        if status != 1:
            raise AssertionError(
                f"expired shutdown deadline exited with status {status}, expected 1:\n{log}"
            )
        if "Graceful shutdown deadline expired" not in log:
            raise AssertionError(f"shutdown deadline log is incomplete:\n{log}")
    finally:
        connection.close()
        stop_process(process)


def test_repeated_sigterm(directory: pathlib.Path) -> None:
    process, log_path, port = start_server(directory, 30)
    connection = open_active_request(
        process, log_path, port, "shutdown-repeated-signal-request"
    )
    try:
        process.send_signal(signal.SIGTERM)
        wait_for_log(process, log_path, "Received shutdown signal")
        assert_catalog_owned_during_drain(directory)
        process.send_signal(signal.SIGTERM)
        status = process.wait(timeout=5)
        log = read_log(log_path)
        if status != 1:
            raise AssertionError(
                f"repeated SIGTERM exited with status {status}, expected 1:\n{log}"
            )
        if "Received a repeated shutdown signal" not in log:
            raise AssertionError(f"repeated SIGTERM log is incomplete:\n{log}")
    finally:
        connection.close()
        stop_process(process)


def main() -> int:
    if os.name != "posix" or not hasattr(signal, "SIGTERM"):
        print("SKIP: process-level SIGTERM tests require a POSIX host")
        return 0

    if SERVER_OVERRIDE is None:
        subprocess.run(
            ["cargo", "build", "--locked", "--bin", "bloom_server"],
            cwd=ROOT,
            check=True,
        )
    if not SERVER.is_file():
        raise AssertionError(f"bloom_server test binary does not exist: {SERVER}")
    with tempfile.TemporaryDirectory(prefix="bloom-shutdown-test-") as raw_temp:
        temporary = pathlib.Path(raw_temp)
        clean = temporary / "clean"
        forced = temporary / "forced"
        repeated = temporary / "repeated"
        clean.mkdir()
        forced.mkdir()
        repeated.mkdir()
        test_clean_sigterm(clean)
        test_forced_deadline(forced)
        test_repeated_sigterm(repeated)
    print(
        "OK: bloom_server retains catalog ownership through SIGTERM drain, escalation, and deadline"
    )
    return 0


def github_command_escape(value: str) -> str:
    return value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


if __name__ == "__main__":
    try:
        exit_code = main()
    except BaseException:
        if os.environ.get("GITHUB_ACTIONS") == "true":
            failure = traceback.format_exc()[-50_000:]
            print(
                "::error title=Server shutdown lifecycle failure::"
                + github_command_escape(failure),
                flush=True,
            )
        raise
    raise SystemExit(exit_code)

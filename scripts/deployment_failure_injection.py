#!/usr/bin/env python3
"""Run bounded failure-injection checks for one Bloom deployment cell.

The scenarios are intentionally operator-facing and require a built
``bloom_server``.  They exercise failure handling at the process boundary:
an invalid model must become a terminal readiness failure, and an abruptly
interrupted server must release its catalog lease so the same cell can start
again.  A storage-quota case verifies bounded model imports fail before
staging. The report contains no model paths or server logs.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import pathlib
import re
import signal
import subprocess
import tempfile
import time
from typing import Any


STARTUP = re.compile(r"server running on http://127\.0\.0\.1:(\d+)")
MAX_BODY = 256 * 1024


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_identity(path: pathlib.Path | None) -> str | None:
    if path is None:
        return None
    if path.is_file():
        return _sha256_file(path)
    if not path.is_dir():
        raise ValueError("model path must be a regular file or directory")
    entries = sorted(path.rglob("*"))
    if len(entries) > 256 or any(entry.is_symlink() for entry in entries):
        raise ValueError("model directory has too many entries or contains a symlink")
    if not entries or any(not entry.is_file() for entry in entries):
        raise ValueError("model directory must contain only regular files")
    digest = hashlib.sha256()
    for entry in entries:
        relative = entry.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256_file(entry)))
    return digest.hexdigest()


def _read_log(path: pathlib.Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def _request(port: int, path: str, timeout: float) -> tuple[int, dict[str, Any] | None]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        body = response.read(MAX_BODY + 1)
        if len(body) > MAX_BODY:
            raise RuntimeError("readiness response exceeded bound")
        try:
            payload = json.loads(body) if body else None
        except json.JSONDecodeError:
            payload = None
        return response.status, payload if isinstance(payload, dict) else None
    finally:
        connection.close()


def _start(
    binary: pathlib.Path,
    models_dir: pathlib.Path,
    log_path: pathlib.Path,
    model: pathlib.Path | None,
    backend: str,
    device: str,
    extra_args: list[str] | None = None,
    environment: dict[str, str] | None = None,
) -> tuple[subprocess.Popen[bytes], int]:
    log = log_path.open("wb")
    try:
        command = [str(binary)]
        if model is not None:
            command.extend(["--model", str(model)])
        command.extend([
            "--models-dir", str(models_dir),
            "--host", "127.0.0.1",
            "--port", "0",
            "--backend", backend,
            "--device", device,
        ])
        command.extend(extra_args or [])
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "RUST_LOG": "bloom_server=info", **(environment or {})},
        )
    finally:
        log.close()
    return process, _wait_for_port(process, log_path, 30.0)


def _wait_for_port(process: subprocess.Popen[bytes], log_path: pathlib.Path, timeout: float) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        match = STARTUP.search(_read_log(log_path))
        if match:
            return int(match.group(1))
        if process.poll() is not None:
            return 0
        time.sleep(0.05)
    return 0


def _wait_ready(
    process: subprocess.Popen[bytes], log_path: pathlib.Path, timeout: float
) -> tuple[int, dict[str, Any] | None, float]:
    deadline = time.monotonic() + timeout
    started = time.monotonic()
    port = 0
    while time.monotonic() < deadline:
        if not port:
            match = STARTUP.search(_read_log(log_path))
            if match:
                port = int(match.group(1))
        if port:
            try:
                status, payload = _request(port, "/ready", 1.0)
                if status in {200, 503}:
                    return status, payload, (time.monotonic() - started) * 1000
            except (OSError, http.client.HTTPException):
                pass
        if process.poll() is not None:
            return 0, None, (time.monotonic() - started) * 1000
        time.sleep(0.05)
    return 0, None, (time.monotonic() - started) * 1000


def _stop(process: subprocess.Popen[bytes], *, hard: bool = False) -> int:
    if process.poll() is None:
        if hard and os.name != "nt":
            process.send_signal(signal.SIGKILL)
        else:
            process.terminate()
    try:
        return process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        return process.wait(timeout=10)


def _ready(status: int, payload: dict[str, Any] | None) -> bool:
    return status == 200 and isinstance(payload, dict) and payload.get("status") == "ready"


def _invalid_model_failed(
    status: int,
    payload: dict[str, Any] | None,
    process_exited: bool,
    returncode: int | None,
    log: str,
) -> bool:
    terminal = (
        status == 503
        and isinstance(payload, dict)
        and payload.get("status") in {"not_ready", "failed"}
        and isinstance(payload.get("load_error"), str)
        and bool(payload["load_error"].strip())
    )
    normalized_log = log.lower()
    logged_failure = (
        any(word in normalized_log for word in ("failed", "error", "not found"))
        and any(
            word in normalized_log
            for word in ("model", "safetensors", "gguf", "tokenizer", "load")
        )
    )
    return terminal or (process_exited and returncode not in (None, 0) and logged_failure)


def _invalid_model(binary: pathlib.Path, root: pathlib.Path, backend: str, device: str) -> dict[str, Any]:
    models_dir = root / "invalid-model" / "models"
    models_dir.mkdir(parents=True)
    log_path = root / "invalid-model.log"
    missing = root / "missing-model.gguf"
    process, _port = _start(binary, models_dir, log_path, missing, backend, device)
    try:
        status, payload, elapsed = _wait_ready(process, log_path, 30.0)
        exited = process.poll() is not None
        if not _invalid_model_failed(status, payload, exited, process.returncode, _read_log(log_path)):
            raise RuntimeError(
                "invalid model did not produce a terminal readiness/load failure"
            )
        return {
            "result": "pass",
            "readiness_status": status,
            "process_exited": exited,
            "elapsed_ms": round(elapsed, 3),
        }
    finally:
        if process.poll() is None:
            _stop(process)


def _post_json(
    port: int,
    path: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float = 5.0,
) -> tuple[int, dict[str, Any] | None]:
    body = json.dumps(payload).encode("utf-8")
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        connection.request(
            "POST",
            path,
            body=body,
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
                **headers,
            },
        )
        response = connection.getresponse()
        raw = response.read(MAX_BODY + 1)
        if len(raw) > MAX_BODY:
            raise RuntimeError("model-management response exceeded bound")
        try:
            value = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            value = None
        return response.status, value if isinstance(value, dict) else None
    finally:
        connection.close()


def _storage_quota(
    binary: pathlib.Path, root: pathlib.Path, backend: str, device: str
) -> dict[str, Any]:
    """Verify quota admission rejects before creating import staging files."""
    models_dir = root / "quota" / "models"
    models_dir.mkdir(parents=True)
    log_path = root / "quota.log"
    operator_key = "deployment-quota-operator-key"
    process, port = _start(
        binary,
        models_dir,
        log_path,
        None,
        backend,
        device,
        extra_args=[
            "--enable-model-imports",
            "--max-model-import-bytes", "64",
            "--max-model-import-chunk-bytes", "16",
            "--max-model-storage-bytes", "1",
        ],
        environment={"BLOOM_OPERATOR_API_KEY": operator_key},
    )
    try:
        if not port:
            raise RuntimeError("server did not publish a port for storage-quota injection")
        content = b"quota-injection-fixture"
        status, payload = _post_json(
            port,
            "/v1/model-management/imports",
            {
                "filename": "quota-fixture.gguf",
                "total_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            },
            {"X-API-Key": operator_key},
        )
        error = payload.get("error") if isinstance(payload, dict) else None
        error_type = error.get("type") if isinstance(error, dict) else None
        message = error.get("message") if isinstance(error, dict) else ""
        staging_root = models_dir / ".bloom-imports"
        residual = (
            sorted(
                path.relative_to(models_dir).as_posix()
                for path in staging_root.rglob("*")
                if path.is_file()
            )
            if staging_root.is_dir()
            else []
        )
        if (
            status != 409
            or error_type != "model_import_conflict"
            or "quota exceeded" not in str(message).lower()
        ):
            raise RuntimeError("storage quota rejection did not return the expected conflict")
        if residual:
            raise RuntimeError(f"quota rejection left staging files: {residual}")
        return {
            "result": "pass",
            "status": status,
            "error_type": error_type,
            "staging_files": residual,
        }
    finally:
        if process.poll() is None:
            _stop(process)


def _process_interruption(
    binary: pathlib.Path,
    model: pathlib.Path,
    root: pathlib.Path,
    backend: str,
    device: str,
) -> dict[str, Any]:
    if os.name != "posix":
        raise RuntimeError("process-interruption requires POSIX hard-interruption semantics")
    models_dir = root / "interruption" / "models"
    models_dir.mkdir(parents=True)
    first_log = root / "interruption-first.log"
    second_log = root / "interruption-second.log"
    first, _port = _start(binary, models_dir, first_log, model, backend, device)
    try:
        first_status, first_payload, first_ms = _wait_ready(first, first_log, 180.0)
        if not _ready(first_status, first_payload):
            raise RuntimeError("valid model did not become ready before interruption")
        first_model = first_payload.get("model")
        if not isinstance(first_model, str) or not first_model:
            raise RuntimeError("ready response omitted the active model identity")
        first_exit = _stop(first, hard=True)
        if first_exit == 0:
            raise RuntimeError("hard interruption unexpectedly exited cleanly")
        second, _port = _start(binary, models_dir, second_log, model, backend, device)
        try:
            second_status, second_payload, second_ms = _wait_ready(second, second_log, 180.0)
            if not _ready(second_status, second_payload):
                raise RuntimeError("server could not reacquire catalog after interruption")
            second_model = second_payload.get("model")
            if second_model != first_model:
                raise RuntimeError(
                    "server reacquired a different active model after interruption"
                )
            return {
                "result": "pass",
                "first_ready_ms": round(first_ms, 3),
                "first_exit_code": first_exit,
                "second_ready_ms": round(second_ms, 3),
                "model_reacquired": True,
            }
        finally:
            if second.poll() is None:
                _stop(second)
    finally:
        if first.poll() is None:
            _stop(first, hard=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    binary = args.server_bin.resolve(strict=True)
    model = args.model.resolve(strict=True) if args.model else None
    if binary.is_symlink() or not binary.is_file():
        raise ValueError("server binary must be a regular file")
    if args.scenario in {"process-interruption", "all"}:
        if model is None or model.is_symlink() or not (model.is_file() or model.is_dir()):
            raise ValueError(
                "process-interruption requires a regular model file or directory"
            )
    server_sha256 = _sha256_file(binary)
    model_sha256 = _model_identity(model)
    with tempfile.TemporaryDirectory(prefix="bloom-failure-injection-") as raw:
        root = pathlib.Path(raw)
        results: dict[str, Any] = {}
        if args.scenario in {"invalid-model", "all"}:
            results["invalid_model"] = _invalid_model(binary, root, args.backend, args.device)
        if args.scenario in {"process-interruption", "all"}:
            assert model is not None
            results["process_interruption"] = _process_interruption(
                binary, model, root, args.backend, args.device
            )
        if args.scenario in {"storage-quota", "all"}:
            results["storage_quota"] = _storage_quota(binary, root, args.backend, args.device)
    return {
        "schema_version": 1,
        "object": "bloom.deployment_failure_injection",
        "result": "pass",
        "deployment": {
            "server_sha256": server_sha256,
            "model_sha256": model_sha256,
            "backend": args.backend,
            "device": args.device,
        },
        "scenarios": results,
        "limitations": [
            "The storage scenario exercises Bloom's quota boundary rather than consuming the host filesystem; physical disk-full and upgrade/rollback remain separate gates.",
            "The interruption scenario requires a real model and target runtime.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-bin", type=pathlib.Path, required=True)
    parser.add_argument("--model", type=pathlib.Path)
    parser.add_argument(
        "--scenario",
        choices=("invalid-model", "process-interruption", "storage-quota", "all"),
        default="all",
    )
    parser.add_argument("--backend", default="candle")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        result = {
            "schema_version": 1,
            "object": "bloom.deployment_failure_injection",
            "result": "fail",
            "error": str(error),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"deployment failure injection: {result['result']} ({args.output})")
    return 0 if result["result"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

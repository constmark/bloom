#!/usr/bin/env python3
"""Exercise Bloom's catalog-ownership, HTTP, authentication, protocol, and UI boundaries."""

from __future__ import annotations

import http.client
import json
import os
import pathlib
import re
import struct
import subprocess
import tempfile
import time

from readiness_contract import (
    READINESS_SERVER_PROTOCOL_VERSION,
    ReadinessContractError,
    validate_readiness_document,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
SERVER_OVERRIDE = os.environ.get("BLOOM_TEST_SERVER_BINARY")
SERVER = (
    pathlib.Path(SERVER_OVERRIDE).resolve()
    if SERVER_OVERRIDE
    else ROOT / "target" / "debug" / "bloom_server"
)
EXPECT_EMBEDDED_UI = os.environ.get("BLOOM_EXPECT_EMBEDDED_UI", "").lower() in {
    "1",
    "true",
    "yes",
}
STARTUP_PATTERN = re.compile(r"server running on http://127\.0\.0\.1:(\d+)")
MAX_RESPONSE_BYTES = 1024 * 1024
ALLOWED_BROWSER_ORIGIN = "https://bloom-boundary.invalid"
REJECTED_BROWSER_ORIGIN = "https://malicious-boundary.invalid"


def read_log(path: pathlib.Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def request(
    port: int,
    method: str,
    path: str,
    headers: dict[str, str] | None = None,
    body: bytes | None = None,
) -> tuple[int, dict[str, str], bytes]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
    try:
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        body = response.read(MAX_RESPONSE_BYTES + 1)
        if len(body) > MAX_RESPONSE_BYTES:
            raise AssertionError(f"response exceeded {MAX_RESPONSE_BYTES} bytes: {path}")
        response_headers = {
            key.lower(): value for key, value in response.getheaders()
        }
        return response.status, response_headers, body
    finally:
        connection.close()


def valid_request_id(value: str | None) -> bool:
    return (
        isinstance(value, str)
        and 1 <= len(value) <= 128
        and all(
            character.isascii()
            and (character.isalnum() or character in "-_.:")
            for character in value
        )
    )


def response_request_id(headers: dict[str, str], path: str) -> str:
    request_id = headers.get("x-request-id")
    if not valid_request_id(request_id):
        raise AssertionError(f"response has an invalid request ID for {path}: {headers}")
    return request_id


def assert_dynamic_headers(headers: dict[str, str], path: str) -> str:
    request_id = response_request_id(headers, path)
    if headers.get("cache-control") != "no-store":
        raise AssertionError(f"dynamic response is cacheable for {path}: {headers}")
    return request_id


def assert_readiness_contract(headers: dict[str, str], body: bytes) -> None:
    if not headers.get("content-type", "").startswith("application/json"):
        raise AssertionError(f"readiness response is not JSON: {headers}")
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AssertionError("readiness response is not valid UTF-8 JSON") from error
    try:
        validate_readiness_document(
            payload,
            expected_server_protocol_version=READINESS_SERVER_PROTOCOL_VERSION,
        )
    except ReadinessContractError as error:
        raise AssertionError(f"readiness contract is invalid: {error}") from error
    if payload["status"] != "not_ready" or payload["model_tasks"] != []:
        raise AssertionError(f"readiness no-model state is invalid: {payload}")


def decode_json_object(headers: dict[str, str], body: bytes, path: str) -> dict:
    if not headers.get("content-type", "").startswith("application/json"):
        raise AssertionError(f"protocol error is not JSON for {path}: {headers}")
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AssertionError(f"protocol error is not valid UTF-8 JSON for {path}") from error
    if not isinstance(payload, dict):
        raise AssertionError(f"protocol error is not an object for {path}: {payload}")
    return payload


def assert_openai_error(
    headers: dict[str, str], body: bytes, path: str, expected_type: str
) -> None:
    payload = decode_json_object(headers, body, path)
    error = payload.get("error")
    if not isinstance(error, dict) or error.get("type") != expected_type:
        raise AssertionError(f"invalid OpenAI error for {path}: {payload}")
    message = error.get("message")
    if not isinstance(message, str) or not 1 <= len(message) <= 1024:
        raise AssertionError(f"invalid OpenAI error message for {path}: {payload}")


def assert_ollama_error(headers: dict[str, str], body: bytes, path: str) -> None:
    payload = decode_json_object(headers, body, path)
    message = payload.get("error")
    if not isinstance(message, str) or not 1 <= len(message) <= 1024:
        raise AssertionError(f"invalid Ollama error for {path}: {payload}")


def wait_for_server(
    process: subprocess.Popen[bytes], log_path: pathlib.Path
) -> int:
    deadline = time.monotonic() + 10
    port = None
    while time.monotonic() < deadline:
        log = read_log(log_path)
        if port is None:
            match = STARTUP_PATTERN.search(log)
            if match:
                port = int(match.group(1))
        if port is not None:
            try:
                status, _headers, _body = request(port, "GET", "/health")
                if status == 200:
                    return port
            except (OSError, http.client.HTTPException):
                pass
        status = process.poll()
        if status is not None:
            raise AssertionError(
                f"bloom_server exited with status {status} before startup:\n{log}"
            )
        time.sleep(0.05)
    raise AssertionError(
        f"bloom_server did not become healthy within 10 seconds:\n{read_log(log_path)}"
    )


def wait_for_log(path: pathlib.Path, marker: str) -> str:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        log = read_log(path)
        if marker in log:
            return log
        time.sleep(0.02)
    raise AssertionError(f"server log did not contain {marker!r}:\n{read_log(path)}")


def stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def validate_catalog_ownership_conflict(
    models_dir: pathlib.Path,
    directory: pathlib.Path,
    environment: dict[str, str],
) -> None:
    log_path = directory / "catalog-owner-conflict.log"
    with log_path.open("wb") as log_handle:
        process = subprocess.Popen(
            [
                str(SERVER),
                "--models-dir",
                str(models_dir),
                "--host",
                "127.0.0.1",
                "--port",
                "0",
            ],
            cwd=ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=environment,
        )
    try:
        status = process.wait(timeout=5)
    except subprocess.TimeoutExpired as error:
        stop_process(process)
        raise AssertionError(
            "a second bloom_server acquired an already-owned model catalog"
        ) from error
    log = read_log(log_path)
    if status == 0 or "already owned by another Bloom server" not in log:
        raise AssertionError(
            f"catalog ownership conflict did not fail explicitly ({status}):\n{log}"
        )


def validate_concurrency_limit(
    directory: pathlib.Path, environment: dict[str, str]
) -> None:
    pointer_bits = struct.calcsize("P") * 8
    maximum_runtime_permits = ((1 << pointer_bits) - 1) >> 3
    log_path = directory / "invalid-concurrency.log"
    models_dir = directory / "invalid-concurrency-models"
    with log_path.open("wb") as log_handle:
        process = subprocess.Popen(
            [
                str(SERVER),
                "--models-dir",
                str(models_dir),
                "--host",
                "127.0.0.1",
                "--port",
                "0",
                "--context-size",
                "1",
                "--max-concurrent",
                str(maximum_runtime_permits + 1),
            ],
            cwd=ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=environment,
        )
    try:
        status = process.wait(timeout=5)
    except subprocess.TimeoutExpired as error:
        stop_process(process)
        raise AssertionError("oversized concurrency unexpectedly started a server") from error
    log = read_log(log_path)
    if (
        status == 0
        or "Maximum concurrency must not exceed" not in log
        or "panicked" in log.lower()
        or models_dir.exists()
    ):
        raise AssertionError(
            f"oversized concurrency did not fail safely ({status}):\n{log}"
        )


def validate_chunked_prefill_limit(
    directory: pathlib.Path, environment: dict[str, str]
) -> None:
    log_path = directory / "invalid-prefill-chunk.log"
    models_dir = directory / "invalid-prefill-chunk-models"
    with log_path.open("wb") as log_handle:
        process = subprocess.Popen(
            [
                str(SERVER),
                "--models-dir",
                str(models_dir),
                "--host",
                "127.0.0.1",
                "--port",
                "0",
                "--enable-ifb",
                "--enable-chunked-prefill",
                "--prefill-chunk-size",
                "0",
            ],
            cwd=ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=environment,
        )
    try:
        status = process.wait(timeout=5)
    except subprocess.TimeoutExpired as error:
        stop_process(process)
        raise AssertionError("zero prefill chunk size unexpectedly started a server") from error
    log = read_log(log_path)
    if (
        status == 0
        or "Chunked prefill size must be at least 1 token" not in log
        or "panicked" in log.lower()
        or models_dir.exists()
    ):
        raise AssertionError(
            f"zero prefill chunk size did not fail safely ({status}):\n{log}"
        )


def validate_catalog_ownership_reuse(
    models_dir: pathlib.Path,
    directory: pathlib.Path,
    environment: dict[str, str],
) -> None:
    log_path = directory / "catalog-owner-reuse.log"
    with log_path.open("wb") as log_handle:
        process = subprocess.Popen(
            [
                str(SERVER),
                "--models-dir",
                str(models_dir),
                "--host",
                "127.0.0.1",
                "--port",
                "0",
            ],
            cwd=ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=environment,
        )
    try:
        wait_for_server(process, log_path)
    finally:
        stop_process(process)


def validate_boundary(port: int, log_path: pathlib.Path) -> None:
    status, headers, _body = request(
        port,
        "GET",
        "/health",
        {"Origin": ALLOWED_BROWSER_ORIGIN},
    )
    if status != 200:
        raise AssertionError(f"health returned {status}")
    first_generated = assert_dynamic_headers(headers, "/health")
    exposed = {
        value.strip().lower()
        for value in headers.get("access-control-expose-headers", "").split(",")
    }
    missing_exposed = {"x-request-id", "retry-after", "www-authenticate"}.difference(
        exposed
    )
    if missing_exposed:
        raise AssertionError(
            f"CORS does not expose required support headers {sorted(missing_exposed)}: {headers}"
        )
    if headers.get("access-control-allow-origin") != ALLOWED_BROWSER_ORIGIN:
        raise AssertionError(f"CORS did not publish the exact allowed origin: {headers}")

    status, headers, _body = request(port, "GET", "/health")
    if status != 200:
        raise AssertionError(f"second health request returned {status}")
    second_generated = assert_dynamic_headers(headers, "/health")
    if first_generated == second_generated:
        raise AssertionError("independent requests received the same generated request ID")

    supplied = "proxy.node_1:request-42"
    status, headers, _body = request(
        port, "GET", "/health", {"X-Request-ID": supplied}
    )
    if status != 200 or headers.get("x-request-id") != supplied:
        raise AssertionError(f"bounded proxy request ID was not preserved: {headers}")
    assert_dynamic_headers(headers, "/health")

    status, headers, body = request(port, "GET", "/ready")
    assert_dynamic_headers(headers, "/ready")
    if status != 503:
        raise AssertionError(f"/ready returned {status}, expected 503")
    assert_readiness_contract(headers, body)

    for path, expected_status in [
        ("/metrics", 200),
        ("/v1/models", 200),
        ("/api/version", 200),
    ]:
        status, headers, _body = request(port, "GET", path)
        assert_dynamic_headers(headers, path)
        if status != expected_status:
            raise AssertionError(f"{path} returned {status}, expected {expected_status}")

    unsafe = "unsafe request id"
    missing_api_path = "/v1/bloom-http-boundary-not-found"
    status, headers, body = request(
        port,
        "GET",
        missing_api_path,
        {"Accept": "text/html", "X-Request-ID": unsafe},
    )
    replacement = assert_dynamic_headers(headers, missing_api_path)
    if status != 404 or replacement == unsafe:
        raise AssertionError(
            f"unknown API route or unsafe request ID was not rejected: {status}, {headers}"
        )
    assert_openai_error(headers, body, missing_api_path, "not_found_error")
    if missing_api_path.encode() in body:
        raise AssertionError("OpenAI route fallback reflected the request path")

    for path, family in [
        ("/v1", "openai"),
        ("/api", "ollama"),
        ("/api/bloom-http-boundary-not-found", "ollama"),
    ]:
        status, headers, body = request(port, "GET", path, {"Accept": "text/html"})
        assert_dynamic_headers(headers, path)
        if status != 404:
            raise AssertionError(f"embedded UI shadowed {path}: HTTP {status}")
        if family == "openai":
            assert_openai_error(headers, body, path, "not_found_error")
        else:
            assert_ollama_error(headers, body, path)

    for method, path, expected_allow, family in [
        ("POST", "/v1/models", "GET,HEAD", "openai"),
        ("GET", "/api/show", "POST", "ollama"),
    ]:
        status, headers, body = request(port, method, path)
        assert_dynamic_headers(headers, path)
        if status != 405 or headers.get("allow") != expected_allow:
            raise AssertionError(
                f"method fallback is invalid for {method} {path}: {status}, {headers}"
            )
        if family == "openai":
            assert_openai_error(headers, body, path, "invalid_request_error")
        else:
            assert_ollama_error(headers, body, path)

    private_marker = b"bloom-private-request-value"
    rejection_cases = [
        (
            "/v1/chat/completions",
            {"Content-Type": "application/json"},
            b'{"messages":[{"content":"' + private_marker,
            400,
            "openai",
        ),
        (
            "/v1/chat/completions",
            {"Content-Type": "application/json"},
            b"[]",
            422,
            "openai",
        ),
        ("/v1/chat/completions", {}, b"{}", 415, "openai"),
        (
            "/v1/model-management/downloads/inspect",
            {"Content-Type": "application/json"},
            b"x" * 4097,
            413,
            "openai",
        ),
        (
            "/v1/multimodal/upload",
            {"Content-Type": "multipart/form-data"},
            private_marker,
            400,
            "openai",
        ),
        (
            "/api/chat",
            {"Content-Type": "application/json"},
            b'{"model":"default","messages":[' + private_marker,
            400,
            "ollama",
        ),
        ("/api/chat", {}, b"{}", 400, "ollama"),
        (
            "/api/chat",
            {"Content-Type": "application/json"},
            b"x" * 4097,
            413,
            "ollama",
        ),
    ]
    for path, request_headers, request_body, expected_status, family in rejection_cases:
        status, headers, body = request(
            port, "POST", path, request_headers, request_body
        )
        assert_dynamic_headers(headers, path)
        if status != expected_status:
            raise AssertionError(
                f"framework rejection has invalid status for {path}: {status}, {body!r}"
            )
        if family == "openai":
            assert_openai_error(headers, body, path, "invalid_request_error")
        else:
            assert_ollama_error(headers, body, path)
        if private_marker in body:
            raise AssertionError(f"framework rejection reflected request data for {path}")

    missing_asset = "/assets/missing.js"
    status, headers, _body = request(
        port, "GET", missing_asset, {"Accept": "text/html"}
    )
    response_request_id(headers, missing_asset)
    if status != 404:
        raise AssertionError(f"embedded UI shadowed {missing_asset}: HTTP {status}")

    navigation = "/conversations/bloom-boundary-check"
    status, headers, _body = request(
        port, "GET", navigation, {"Accept": "text/html"}
    )
    response_request_id(headers, navigation)
    expected_status = 200 if EXPECT_EMBEDDED_UI else 404
    if status != expected_status:
        raise AssertionError(
            f"browser navigation returned {status}, expected {expected_status}"
        )
    if EXPECT_EMBEDDED_UI and not headers.get("content-type", "").startswith(
        "text/html"
    ):
        raise AssertionError(f"SPA navigation did not return HTML: {headers}")

    for method, accept in [("GET", "application/json"), ("POST", "text/html")]:
        status, headers, _body = request(
            port, method, navigation, {"Accept": accept}
        )
        response_request_id(headers, navigation)
        if status != 404:
            raise AssertionError(
                f"non-navigation request reached the SPA fallback: {method} {status}"
            )

    trace_id = "trace-check-42"
    secret = "bloom-query-secret-must-not-be-logged"
    status, headers, _body = request(
        port,
        "GET",
        f"/health?token={secret}",
        {"X-Request-ID": trace_id},
    )
    if status != 200 or headers.get("x-request-id") != trace_id:
        raise AssertionError(f"trace request correlation failed: {status}, {headers}")
    assert_dynamic_headers(headers, "/health")
    log = wait_for_log(log_path, f'request_id="{trace_id}"')
    if 'path="/health"' not in log:
        raise AssertionError(f"trace span did not include the normalized path:\n{log}")
    if secret in log:
        raise AssertionError("trace output disclosed a URL query value")


def validate_authentication_boundary(port: int) -> None:
    challenge = 'Bearer realm="Bloom"'
    status, headers, body = request(
        port,
        "GET",
        "/v1/models",
        {"Origin": ALLOWED_BROWSER_ORIGIN},
    )
    assert_dynamic_headers(headers, "/v1/models")
    if status != 401 or headers.get("www-authenticate") != challenge:
        raise AssertionError(
            f"OpenAI authentication challenge is invalid: {status}, {headers}"
        )
    assert_openai_error(headers, body, "/v1/models", "authentication_error")
    exposed = {
        value.strip().lower()
        for value in headers.get("access-control-expose-headers", "").split(",")
    }
    if "www-authenticate" not in exposed:
        raise AssertionError(f"CORS hides the authentication challenge: {headers}")

    status, headers, body = request(
        port,
        "GET",
        "/v1/models",
        {"Authorization": "Bearer invalid-boundary-key"},
    )
    if status != 401 or headers.get("www-authenticate") != challenge:
        raise AssertionError(f"invalid Bearer credential lost its challenge: {headers}")
    assert_openai_error(headers, body, "/v1/models", "authentication_error")

    status, headers, body = request(port, "GET", "/api/version")
    assert_dynamic_headers(headers, "/api/version")
    if status != 401 or headers.get("www-authenticate") != challenge:
        raise AssertionError(
            f"Ollama authentication challenge is invalid: {status}, {headers}"
        )
    assert_ollama_error(headers, body, "/api/version")

    status, headers, _body = request(
        port,
        "GET",
        "/v1/models",
        {"Authorization": "Bearer boundary-secret"},
    )
    if status != 200:
        raise AssertionError(f"valid Bearer credential was rejected: {status}, {headers}")
    assert_dynamic_headers(headers, "/v1/models")

    status, headers, _body = request(
        port, "GET", "/api/version", {"X-API-Key": "boundary-secret"}
    )
    if status != 200:
        raise AssertionError(f"valid X-API-Key credential was rejected: {status}, {headers}")
    assert_dynamic_headers(headers, "/api/version")

    missing_path = "/v1/bloom-auth-boundary-not-found"
    status, headers, body = request(port, "GET", missing_path)
    if status != 404 or "www-authenticate" in headers:
        raise AssertionError(
            f"unknown route entered the authentication boundary: {status}, {headers}"
        )
    assert_dynamic_headers(headers, missing_path)
    assert_openai_error(headers, body, missing_path, "not_found_error")


def validate_browser_origin_boundary(port: int) -> None:
    same_origin = f"http://127.0.0.1:{port}"
    status, headers, _body = request(
        port, "GET", "/health", {"Origin": same_origin}
    )
    if status != 200:
        raise AssertionError(f"same-origin browser request was rejected: {status}, {headers}")
    assert_dynamic_headers(headers, "/health")

    for path, family in [("/v1/models", "openai"), ("/api/version", "ollama")]:
        status, headers, body = request(
            port, "GET", path, {"Origin": REJECTED_BROWSER_ORIGIN}
        )
        if status != 403 or "access-control-allow-origin" in headers:
            raise AssertionError(
                f"untrusted browser origin was not rejected for {path}: {status}, {headers}"
            )
        assert_dynamic_headers(headers, path)
        if family == "openai":
            assert_openai_error(headers, body, path, "invalid_request_error")
        else:
            assert_ollama_error(headers, body, path)

    status, headers, _body = request(
        port,
        "OPTIONS",
        "/v1/models",
        {
            "Origin": ALLOWED_BROWSER_ORIGIN,
            "Access-Control-Request-Method": "GET",
        },
    )
    allowed_methods = {
        method.strip().upper()
        for method in headers.get("access-control-allow-methods", "").split(",")
    }
    if (
        status not in {200, 204}
        or headers.get("access-control-allow-origin") != ALLOWED_BROWSER_ORIGIN
        or not ({"GET", "*"} & allowed_methods)
    ):
        raise AssertionError(f"trusted CORS preflight failed: {status}, {headers}")

    status, headers, body = request(
        port,
        "OPTIONS",
        "/v1/models",
        {
            "Origin": REJECTED_BROWSER_ORIGIN,
            "Access-Control-Request-Method": "GET",
        },
    )
    if status != 403 or "access-control-allow-origin" in headers:
        raise AssertionError(f"untrusted CORS preflight was admitted: {status}, {headers}")
    assert_dynamic_headers(headers, "/v1/models")
    assert_openai_error(headers, body, "/v1/models", "invalid_request_error")

    for origin, host in [("null", None), ("http://rebind.invalid", "rebind.invalid")]:
        request_headers = {"Origin": origin}
        if host is not None:
            request_headers["Host"] = host
        status, headers, _body = request(port, "GET", "/health", request_headers)
        if status != 403 or "access-control-allow-origin" in headers:
            raise AssertionError(
                f"opaque or rebound browser origin was admitted: {status}, {headers}"
            )
        assert_dynamic_headers(headers, "/health")


def validate_default_browser_origin_boundary(port: int) -> None:
    same_origin = f"http://127.0.0.1:{port}"
    status, headers, _body = request(
        port, "GET", "/health", {"Origin": same_origin}
    )
    if status != 200:
        raise AssertionError(
            f"default policy rejected the embedded same origin: {status}, {headers}"
        )
    assert_dynamic_headers(headers, "/health")

    status, headers, _body = request(
        port, "GET", "/health", {"Origin": REJECTED_BROWSER_ORIGIN}
    )
    if status != 403 or "access-control-allow-origin" in headers:
        raise AssertionError(
            f"default policy admitted a cross-origin browser: {status}, {headers}"
        )
    assert_dynamic_headers(headers, "/health")

    status, headers, body = request(
        port,
        "OPTIONS",
        "/api/version",
        {
            "Origin": REJECTED_BROWSER_ORIGIN,
            "Access-Control-Request-Method": "GET",
        },
    )
    if status != 403 or "access-control-allow-origin" in headers:
        raise AssertionError(
            f"default policy admitted a cross-origin preflight: {status}, {headers}"
        )
    assert_dynamic_headers(headers, "/api/version")
    assert_ollama_error(headers, body, "/api/version")

    status, headers, _body = request(port, "GET", "/api/version")
    if status != 200:
        raise AssertionError(
            f"default policy rejected an origin-free SDK request: {status}, {headers}"
        )
    assert_dynamic_headers(headers, "/api/version")


def main() -> int:
    if SERVER_OVERRIDE is None:
        subprocess.run(
            ["cargo", "build", "--locked", "--bin", "bloom_server"],
            cwd=ROOT,
            check=True,
        )
    if not SERVER.is_file():
        raise AssertionError(f"bloom_server test binary does not exist: {SERVER}")

    with tempfile.TemporaryDirectory(prefix="bloom-http-boundary-") as raw_temp:
        temporary = pathlib.Path(raw_temp)
        log_path = temporary / "server.log"
        log_handle = log_path.open("wb")
        environment = os.environ.copy()
        environment["RUST_LOG"] = "tower_http=debug,bloom_server=info"
        validate_concurrency_limit(temporary, environment)
        validate_chunked_prefill_limit(temporary, environment)
        try:
            process = subprocess.Popen(
                [
                    str(SERVER),
                    "--models-dir",
                    str(temporary / "models"),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "0",
                    "--max-body-bytes",
                    "4096",
                    "--cors-allow-origin",
                    ALLOWED_BROWSER_ORIGIN,
                ],
                cwd=ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=environment,
            )
        finally:
            log_handle.close()
        try:
            port = wait_for_server(process, log_path)
            validate_boundary(port, log_path)
            validate_browser_origin_boundary(port)
            validate_catalog_ownership_conflict(
                temporary / "models", temporary, environment
            )
        finally:
            stop_process(process)
        validate_catalog_ownership_reuse(
            temporary / "models", temporary, environment
        )

        auth_log_path = temporary / "auth-server.log"
        auth_log_handle = auth_log_path.open("wb")
        try:
            auth_process = subprocess.Popen(
                [
                    str(SERVER),
                    "--models-dir",
                    str(temporary / "auth-models"),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "0",
                    "--max-body-bytes",
                    "4096",
                    "--api-key",
                    "boundary-secret",
                    "--cors-allow-origin",
                    ALLOWED_BROWSER_ORIGIN,
                ],
                cwd=ROOT,
                stdout=auth_log_handle,
                stderr=subprocess.STDOUT,
                env=environment,
            )
        finally:
            auth_log_handle.close()
        try:
            auth_port = wait_for_server(auth_process, auth_log_path)
            validate_authentication_boundary(auth_port)
        finally:
            stop_process(auth_process)

        default_log_path = temporary / "default-origin-server.log"
        default_log_handle = default_log_path.open("wb")
        try:
            default_process = subprocess.Popen(
                [
                    str(SERVER),
                    "--models-dir",
                    str(temporary / "default-origin-models"),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "0",
                ],
                cwd=ROOT,
                stdout=default_log_handle,
                stderr=subprocess.STDOUT,
                env=environment,
            )
        finally:
            default_log_handle.close()
        try:
            default_port = wait_for_server(default_process, default_log_path)
            validate_default_browser_origin_boundary(default_port)
        finally:
            stop_process(default_process)

    mode = "embedded UI" if EXPECT_EMBEDDED_UI else "server-only"
    print(
        f"OK: bloom_server startup numeric limits, catalog ownership, browser-origin, correlation, authentication, readiness, and {mode} routing boundaries"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

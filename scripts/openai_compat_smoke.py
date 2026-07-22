#!/usr/bin/env python3
"""Smoke-test Bloom's OpenAI-compatible HTTP API.

The script is CI-friendly by default: when no model is present it exits
successfully with a SKIP message. Use --require-model or BLOOM_REQUIRE_MODEL=1
for release/production gates where a real model must be validated.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def request_json(
    url: str,
    payload: dict | None = None,
    timeout: float = 5.0,
    allow_http_error: bool = False,
    headers: dict[str, str] | None = None,
) -> dict:
    data = None
    request_headers = dict(headers or {})
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        request_headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=request_headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        if not allow_http_error:
            raise
        body = exc.read().decode("utf-8")
        parsed = json.loads(body)
        parsed["_http_status"] = exc.code
        return parsed
    return json.loads(body)


def request_sse(
    url: str,
    payload: dict,
    timeout: float = 10.0,
    headers: dict[str, str] | None = None,
) -> list[str]:
    request_headers = dict(headers or {})
    request_headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=request_headers,
    )
    events: list[str] = []
    with urllib.request.urlopen(req, timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            events.append(data)
            if data == "[DONE]":
                break
    return events


def request_openai_sdk_stream(
    base_url: str,
    model_id: str,
    max_tokens: int,
    api_key: str,
) -> dict:
    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        return {"status": "skipped", "reason": "python package 'openai' is not installed"}

    client = OpenAI(base_url=f"{base_url}/v1", api_key=api_key)
    chunks = 0
    content_chunks = 0
    stream = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": "Say hello in one short sentence."}],
        max_tokens=max_tokens,
        temperature=0,
        stream=True,
    )
    for event in stream:
        chunks += 1
        if not event.choices:
            continue
        delta = event.choices[0].delta
        if getattr(delta, "content", None):
            content_chunks += 1
    return {"status": "ok", "stream_chunks": chunks, "content_chunks": content_chunks}


def wait_ready(base_url: str, proc: subprocess.Popen[str], timeout: float) -> None:
    deadline = time.time() + timeout
    last_error = ""
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"bloom_server exited early with code {proc.returncode}")
        try:
            ready = request_json(f"{base_url}/ready", timeout=2.0, allow_http_error=True)
            if ready.get("status") == "ready":
                return
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            last_error = str(exc)
        time.sleep(0.5)
    raise TimeoutError(f"server did not become ready within {timeout}s: {last_error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=os.environ.get("BLOOM_MODEL_PATH", os.environ.get("MODEL_DIR", "/tmp/smoke_model")),
        help="Model directory or GGUF file. Missing path causes a CI-friendly skip.",
    )
    parser.add_argument("--server-bin", default="target/release/bloom_server")
    parser.add_argument("--build", action="store_true", help="Build bloom_server before running.")
    parser.add_argument("--backend", default="candle")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--startup-timeout", type=float, default=120.0)
    parser.add_argument(
        "--api-key",
        default=os.environ.get("BLOOM_API_KEY", ""),
        help="Enable and validate /v1 API-key authentication for the spawned server.",
    )
    parser.add_argument(
        "--require-model",
        action="store_true",
        default=os.environ.get("BLOOM_REQUIRE_MODEL", "").lower() in {"1", "true", "yes"},
        help="Fail instead of SKIP when the model path is missing.",
    )
    parser.add_argument(
        "--require-openai-sdk",
        action="store_true",
        help="Fail if the optional OpenAI Python SDK compatibility check cannot run.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model = Path(args.model)
    if not model.exists():
        if args.require_model:
            print(f"FAIL: model not found at {model} (required by --require-model)", file=sys.stderr)
            return 1
        print(f"SKIP: model not found at {model} (set BLOOM_MODEL_PATH or --model)")
        return 0

    if args.build or not Path(args.server_bin).exists():
        subprocess.run(["cargo", "build", "--release", "--bin", "bloom_server"], check=True)

    port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    cmd = [
        args.server_bin,
        "--model",
        str(model),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--backend",
        args.backend,
        "--device",
        args.device,
        "--timeout",
        "120",
    ]
    env = os.environ.copy()
    if args.api_key:
        env["BLOOM_API_KEY"] = args.api_key

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        wait_ready(base_url, proc, args.startup_timeout)

        auth_headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
        auth_status = "disabled"
        if args.api_key:
            unauthenticated = request_json(
                f"{base_url}/v1/models",
                allow_http_error=True,
            )
            if unauthenticated.get("_http_status") != 401:
                raise AssertionError(
                    f"expected unauthenticated /v1/models to return 401: {unauthenticated}"
                )
            err = unauthenticated.get("error", {})
            if err.get("type") != "authentication_error":
                raise AssertionError(f"unexpected auth error payload: {unauthenticated}")
            auth_status = "validated"

        models = request_json(f"{base_url}/v1/models", headers=auth_headers)
        if models.get("object") != "list" or not models.get("data"):
            raise AssertionError(f"unexpected /v1/models response: {models}")
        model_id = models["data"][0].get("id", "bloom-local")

        backends = request_json(f"{base_url}/v1/backends", headers=auth_headers)
        backend_rows = backends.get("data") or []
        if not backend_rows:
            raise AssertionError(f"unexpected /v1/backends response: {backends}")
        required_backend_fields = {
            "supports_streaming",
            "supports_quantized_models",
            "supports_embeddings",
            "supports_rerank",
            "supports_structured_output",
        }
        missing_fields = [
            field
            for field in required_backend_fields
            if field not in backend_rows[0]
        ]
        if missing_fields:
            raise AssertionError(f"/v1/backends missing capability fields: {missing_fields}")

        payload = {
            "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
            "stream": False,
        }
        completion = request_json(
            f"{base_url}/v1/chat/completions",
            payload,
            timeout=60.0,
            headers=auth_headers,
        )
        choices = completion.get("choices") or []
        if not choices or "message" not in choices[0]:
            raise AssertionError(f"unexpected chat completion response: {completion}")

        structured_payload = {
            "messages": [{"role": "user", "content": "Return {\"ok\": true}."}],
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "smoke",
                    "schema": {
                        "type": "object",
                        "required": ["ok"],
                        "properties": {"ok": {"type": "boolean"}},
                        "additionalProperties": False,
                    },
                },
            },
        }
        structured = request_json(
            f"{base_url}/v1/chat/completions",
            structured_payload,
            timeout=60.0,
            allow_http_error=True,
            headers=auth_headers,
        )
        if structured.get("_http_status") not in (None, 422):
            raise AssertionError(f"unexpected structured output response: {structured}")
        if structured.get("_http_status") == 422:
            err = structured.get("error", {})
            if err.get("type") != "invalid_response_format":
                raise AssertionError(f"unexpected structured output error: {structured}")

        embeddings = request_json(
            f"{base_url}/v1/embeddings",
            {"input": "hello", "encoding_format": "float"},
            timeout=60.0,
            allow_http_error=True,
            headers=auth_headers,
        )
        if embeddings.get("_http_status") not in (None, 501):
            raise AssertionError(f"unexpected embeddings response: {embeddings}")
        if embeddings.get("_http_status") is None and embeddings.get("object") != "list":
            raise AssertionError(f"unexpected embeddings payload: {embeddings}")
        if embeddings.get("_http_status") == 501:
            err = embeddings.get("error", {})
            if err.get("type") != "unsupported_operation":
                raise AssertionError(f"unexpected embeddings error: {embeddings}")

        rerank = request_json(
            f"{base_url}/v1/rerank",
            {"query": "hello", "documents": ["hello world"], "top_n": 1},
            timeout=60.0,
            allow_http_error=True,
            headers=auth_headers,
        )
        if rerank.get("_http_status") not in (None, 501):
            raise AssertionError(f"unexpected rerank response: {rerank}")
        if rerank.get("_http_status") is None and rerank.get("object") != "rerank":
            raise AssertionError(f"unexpected rerank payload: {rerank}")
        if rerank.get("_http_status") == 501:
            err = rerank.get("error", {})
            if err.get("type") != "unsupported_operation":
                raise AssertionError(f"unexpected rerank error: {rerank}")

        payload["stream"] = True
        events = request_sse(
            f"{base_url}/v1/chat/completions",
            payload,
            timeout=60.0,
            headers=auth_headers,
        )
        if "[DONE]" not in events:
            raise AssertionError("streaming response did not include [DONE]")

        sdk = request_openai_sdk_stream(
            base_url,
            model_id,
            args.max_tokens,
            args.api_key or "bloom-smoke",
        )
        if args.require_openai_sdk and sdk.get("status") != "ok":
            raise AssertionError(f"OpenAI SDK smoke failed or skipped: {sdk}")

        print(
            json.dumps(
                {
                    "status": "ok",
                    "base_url": base_url,
                    "model_count": len(models["data"]),
                    "backend_count": len(backend_rows),
                    "stream_events": len(events),
                    "auth": auth_status,
                    "structured_output": "ok"
                    if structured.get("_http_status") is None
                    else "validated_error",
                    "embeddings": "ok" if embeddings.get("_http_status") is None else "unsupported",
                    "rerank": "ok" if rerank.get("_http_status") is None else "unsupported",
                    "openai_sdk": sdk,
                },
                indent=2,
            )
        )
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)

#!/usr/bin/env python3
"""Smoke-test Bloom's bounded Ollama-compatible HTTP API.

Discovery, authentication, and fail-closed admission run without a model. When
the configured model path is missing, model-backed checks are reported as SKIP
unless --require-model is set. A text model exercises generation; an embedding
model exercises current and legacy embedding routes. The optional official
Python client check requires the ``ollama`` package.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from readiness_contract import ReadinessContractError, validate_readiness_document

MAX_RESPONSE_BYTES = 16 * 1024 * 1024
MISSING_DELETE_PROBE_MODEL = "__bloom_missing_delete_probe__.gguf"
MISSING_PULL_PROBE_MODEL = "bloom-missing-pull-probe"


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def request_json(
    url: str,
    payload: dict[str, Any] | None = None,
    *,
    method: str | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 10.0,
    allow_http_error: bool = False,
) -> tuple[int, dict[str, Any]]:
    request_headers = dict(headers or {})
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        request_headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        url, data=data, headers=request_headers, method=method
    )
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        if not allow_http_error:
            raise
        response = exc
    with response:
        body = response.read(MAX_RESPONSE_BYTES + 1)
        status = response.status
    if len(body) > MAX_RESPONSE_BYTES:
        raise AssertionError(f"response from {url} exceeded {MAX_RESPONSE_BYTES} bytes")
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise AssertionError(f"response from {url} was not a JSON object: {parsed!r}")
    return status, parsed


def request_protocol_error(
    url: str,
    *,
    method: str,
    headers: dict[str, str],
    body: bytes | None = None,
    timeout: float = 10.0,
) -> tuple[int, dict[str, str], dict[str, Any]]:
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        response = exc
    with response:
        body = response.read(MAX_RESPONSE_BYTES + 1)
        status = response.status
        response_headers = {
            key.lower(): value for key, value in response.headers.items()
        }
    if len(body) > MAX_RESPONSE_BYTES:
        raise AssertionError(f"response from {url} exceeded {MAX_RESPONSE_BYTES} bytes")
    try:
        parsed = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AssertionError(f"protocol error from {url} was not valid UTF-8 JSON") from error
    if not isinstance(parsed, dict):
        raise AssertionError(f"protocol error from {url} was not an object: {parsed!r}")
    message = parsed.get("error")
    if not isinstance(message, str) or not 1 <= len(message) <= 1024:
        raise AssertionError(f"protocol error from {url} was invalid: {parsed!r}")
    return status, response_headers, parsed


def request_ndjson(
    url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str],
    timeout: float,
) -> list[dict[str, Any]]:
    request_headers = dict(headers)
    request_headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=request_headers,
    )
    events: list[dict[str, Any]] = []
    total_bytes = 0
    with urllib.request.urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get_content_type()
        if content_type != "application/x-ndjson":
            raise AssertionError(f"unexpected stream content type: {content_type!r}")
        for raw_line in response:
            total_bytes += len(raw_line)
            if total_bytes > MAX_RESPONSE_BYTES:
                raise AssertionError("NDJSON stream exceeded its smoke-test byte limit")
            line = raw_line.strip()
            if not line:
                continue
            event = json.loads(line)
            if not isinstance(event, dict):
                raise AssertionError(f"NDJSON event was not an object: {event!r}")
            if isinstance(event.get("error"), str):
                raise AssertionError(f"generation stream failed: {event['error']}")
            events.append(event)
            if event.get("done") is True:
                break
    if not events or events[-1].get("done") is not True:
        raise AssertionError(f"NDJSON stream omitted its terminal event: {events!r}")
    return events


def wait_healthy(base_url: str, process: subprocess.Popen[str], timeout: float) -> None:
    deadline = time.time() + timeout
    last_error = ""
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"bloom_server exited early with code {process.returncode}")
        try:
            status, health = request_json(f"{base_url}/health", timeout=2.0)
            if status == 200 and health.get("status") == "ok":
                return
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            last_error = str(exc)
        time.sleep(0.25)
    raise TimeoutError(f"server did not become healthy within {timeout}s: {last_error}")


def wait_ready(base_url: str, process: subprocess.Popen[str], timeout: float) -> None:
    deadline = time.time() + timeout
    last_error = ""
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"bloom_server exited early with code {process.returncode}")
        try:
            status, readiness = request_json(
                f"{base_url}/ready", timeout=2.0, allow_http_error=True
            )
            try:
                validate_readiness_document(readiness)
            except ReadinessContractError as error:
                raise RuntimeError(
                    f"bloom_server returned an incompatible readiness contract: {error}"
                ) from error
            if status == 200 and readiness.get("status") == "ready":
                return
            if readiness.get("load_error") is not None and readiness.get("loading") is False:
                raise RuntimeError("bloom_server reported a terminal model load failure")
            last_error = f"HTTP {status}: {readiness!r}"
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            last_error = str(exc)
        time.sleep(0.25)
    raise TimeoutError(f"server did not become ready within {timeout}s: {last_error}")


def wait_for_no_running_models(
    base_url: str,
    headers: dict[str, str],
    timeout: float,
) -> dict[str, Any]:
    deadline = time.time() + timeout
    last_processes: dict[str, Any] = {}
    while time.time() < deadline:
        status, processes = request_json(f"{base_url}/api/ps", headers=headers)
        if status != 200 or not isinstance(processes.get("models"), list):
            raise AssertionError(f"unexpected /api/ps response: {processes}")
        last_processes = processes
        if not processes["models"]:
            return processes
        time.sleep(0.025)
    raise AssertionError(
        f"timed keep_alive did not unload the active model: {last_processes}"
    )


def check_embedding_vectors(
    payload: dict[str, Any],
    *,
    expected_count: int,
    expected_dimensions: int | None = None,
    normalized: bool,
) -> None:
    embeddings = payload.get("embeddings")
    if not isinstance(embeddings, list) or len(embeddings) != expected_count:
        raise AssertionError(f"unexpected embedding batch: {payload!r}")
    for embedding in embeddings:
        if not isinstance(embedding, list) or not embedding:
            raise AssertionError(f"embedding vector was empty or invalid: {embedding!r}")
        if expected_dimensions is not None and len(embedding) != expected_dimensions:
            raise AssertionError(
                f"expected {expected_dimensions} dimensions, got {len(embedding)}"
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in embedding
        ):
            raise AssertionError(f"embedding vector contained a non-finite value: {embedding!r}")
        if normalized:
            norm = math.sqrt(math.fsum(float(value) ** 2 for value in embedding))
            if not math.isclose(norm, 1.0, rel_tol=1e-4, abs_tol=1e-4):
                raise AssertionError(f"embedding vector was not L2-normalized: norm={norm}")


def validate_structured_outputs(
    base_url: str,
    model_id: str,
    headers: dict[str, str],
    *,
    timeout: float,
    max_tokens: int,
) -> dict[str, str]:
    expected = {"ok": True}
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }
    options = {"num_predict": max_tokens, "temperature": 0}

    for output_format, label in [("json", "json"), (schema, "schema")]:
        status, chat = request_json(
            f"{base_url}/api/chat",
            {
                "model": model_id,
                "messages": [{"role": "user", "content": "Return ok as JSON."}],
                "format": output_format,
                "options": options,
                "stream": False,
            },
            headers=headers,
            timeout=timeout,
        )
        content = chat.get("message", {}).get("content")
        if (
            status != 200
            or chat.get("model") != model_id
            or chat.get("done") is not True
            or not isinstance(content, str)
            or json.loads(content) != expected
        ):
            raise AssertionError(
                f"Ollama chat {label} output was invalid: {chat}"
            )

        status, generated = request_json(
            f"{base_url}/api/generate",
            {
                "model": model_id,
                "prompt": "Return ok as JSON.",
                "format": output_format,
                "options": options,
                "stream": False,
            },
            headers=headers,
            timeout=timeout,
        )
        generated_text = generated.get("response")
        if (
            status != 200
            or generated.get("model") != model_id
            or generated.get("done") is not True
            or not isinstance(generated_text, str)
            or json.loads(generated_text) != expected
        ):
            raise AssertionError(
                f"Ollama generate {label} output was invalid: {generated}"
            )

    chat_events = request_ndjson(
        f"{base_url}/api/chat",
        {
            "model": model_id,
            "messages": [{"role": "user", "content": "Return ok as JSON."}],
            "format": schema,
            "options": options,
            "stream": True,
        },
        headers=headers,
        timeout=timeout,
    )
    chat_text = "".join(
        event.get("message", {}).get("content", "")
        for event in chat_events
        if isinstance(event.get("message"), dict)
        and isinstance(event["message"].get("content"), str)
    )
    if chat_events[-1].get("model") != model_id or json.loads(chat_text) != expected:
        raise AssertionError(f"Ollama chat schema stream was invalid: {chat_events}")

    generate_events = request_ndjson(
        f"{base_url}/api/generate",
        {
            "model": model_id,
            "prompt": "Return ok as JSON.",
            "format": schema,
            "options": options,
            "stream": True,
        },
        headers=headers,
        timeout=timeout,
    )
    generate_text = "".join(
        event.get("response", "")
        for event in generate_events
        if isinstance(event.get("response"), str)
    )
    if (
        generate_events[-1].get("model") != model_id
        or json.loads(generate_text) != expected
    ):
        raise AssertionError(
            f"Ollama generate schema stream was invalid: {generate_events}"
        )

    return {
        "chat_json": "ok",
        "chat_schema": "ok",
        "chat_schema_stream": "ok",
        "generate_json": "ok",
        "generate_schema": "ok",
        "generate_schema_stream": "ok",
    }


def validate_tool_outputs(
    base_url: str,
    model_id: str,
    headers: dict[str, str],
    *,
    timeout: float,
    max_tokens: int,
) -> dict[str, str]:
    tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Return the current weather for one city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    }
    user_message = {"role": "user", "content": "What is the weather in Paris?"}
    request_payload = {
        "model": model_id,
        "messages": [user_message],
        "tools": [tool],
        "options": {"num_predict": max_tokens, "temperature": 0},
        "stream": False,
    }

    def require_call(call: object, label: str) -> dict[str, Any]:
        if not isinstance(call, dict):
            raise AssertionError(f"{label} was not an object: {call!r}")
        function = call.get("function")
        arguments = function.get("arguments") if isinstance(function, dict) else None
        if (
            call.get("type") != "function"
            or "id" in call
            or not isinstance(function, dict)
            or function.get("index") != 0
            or function.get("name") != "get_weather"
            or arguments != {"city": "Paris"}
        ):
            raise AssertionError(f"{label} was invalid: {call}")
        return call

    status, response = request_json(
        f"{base_url}/api/chat",
        request_payload,
        headers=headers,
        timeout=timeout,
    )
    message = response.get("message")
    calls = message.get("tool_calls") if isinstance(message, dict) else None
    if (
        status != 200
        or response.get("model") != model_id
        or response.get("done") is not True
        or response.get("done_reason") != "stop"
        or not isinstance(message, dict)
        or message.get("role") != "assistant"
        or message.get("content") != ""
        or not isinstance(calls, list)
        or len(calls) != 1
    ):
        raise AssertionError(f"Ollama tool call was invalid: {response}")
    require_call(calls[0], "Ollama tool call")
    if "ollama_call_" in json.dumps(response, separators=(",", ":")):
        raise AssertionError("Ollama tool call exposed Bloom's private correlation ID")

    streaming_payload = dict(request_payload)
    streaming_payload["stream"] = True
    events = request_ndjson(
        f"{base_url}/api/chat",
        streaming_payload,
        headers=headers,
        timeout=timeout,
    )
    streamed_calls: list[dict[str, Any]] = []
    for event in events[:-1]:
        event_message = event.get("message")
        event_calls = (
            event_message.get("tool_calls")
            if isinstance(event_message, dict)
            else None
        )
        if isinstance(event_calls, list):
            streamed_calls.extend(event_calls)
    if (
        events[-1].get("model") != model_id
        or events[-1].get("done_reason") != "stop"
        or len(streamed_calls) != 1
    ):
        raise AssertionError(f"Ollama tool stream was invalid: {events}")
    require_call(streamed_calls[0], "streamed Ollama tool call")
    if "ollama_call_" in json.dumps(events, separators=(",", ":")):
        raise AssertionError("Ollama tool stream exposed Bloom's private correlation ID")

    status, continuation = request_json(
        f"{base_url}/api/chat",
        {
            "model": model_id,
            "messages": [
                user_message,
                message,
                {
                    "role": "tool",
                    "tool_name": "get_weather",
                    "content": "15 C and clear",
                },
            ],
            "options": {"num_predict": max_tokens, "temperature": 0},
            "stream": False,
        },
        headers=headers,
        timeout=timeout,
    )
    continuation_content = continuation.get("message", {}).get("content")
    if (
        status != 200
        or continuation.get("model") != model_id
        or continuation.get("done") is not True
        or not isinstance(continuation_content, str)
        or not continuation_content
    ):
        raise AssertionError(
            f"Ollama tool-result continuation was invalid: {continuation}"
        )

    return {
        "chat": "ok",
        "chat_stream": "ok",
        "result_continuation": "ok",
        "private_ids": "not_exposed",
    }


def check_official_sdk(
    base_url: str,
    headers: dict[str, str],
    model_id: str | None,
    model_kind: str | None,
    max_tokens: int,
    *,
    structured_only: bool = False,
    tool_only: bool = False,
) -> dict[str, Any]:
    try:
        from ollama import Client, ResponseError  # type: ignore
    except ImportError:
        return {"status": "skipped", "reason": "python package 'ollama' is not installed"}

    client = Client(host=base_url, headers=headers)
    listed = client.list()
    processes = client.ps()
    result: dict[str, Any] = {
        "status": "discovery_ok",
        "client_version": importlib.metadata.version("ollama"),
        "listed_models": len(listed.models),
        "running_models": len(processes.models),
    }
    if headers:
        try:
            Client(host=base_url).list()
        except ResponseError as error:
            if error.status_code != 401 or not isinstance(error.error, str):
                raise AssertionError(
                    f"official Ollama client decoded an unexpected auth error: {error!r}"
                ) from error
        else:
            raise AssertionError("official Ollama client bypassed API-key authentication")
        result["authentication"] = "ok"
    else:
        result["authentication"] = "disabled"

    try:
        client.delete(MISSING_DELETE_PROBE_MODEL)
    except ResponseError as error:
        if (
            error.status_code != 404
            or not isinstance(error.error, str)
            or not error.error
        ):
            raise AssertionError(
                f"official Ollama client decoded an unexpected delete error: {error!r}"
            ) from error
    else:
        raise AssertionError("official Ollama client accepted deletion of a missing model")
    result["delete_error"] = "ok"

    try:
        client.pull(MISSING_PULL_PROBE_MODEL, stream=False)
    except ResponseError as error:
        if (
            error.status_code != 403
            or not isinstance(error.error, str)
            or not error.error
        ):
            raise AssertionError(
                f"official Ollama client decoded an unexpected pull error: {error!r}"
            ) from error
    else:
        raise AssertionError("official Ollama client bypassed verified pull admission")
    result["pull_admission"] = "ok"

    if model_id is None:
        return result

    shown = client.show(model_id)
    if shown.details is None:
        raise AssertionError("official Ollama client did not decode show details")
    if model_kind == "embedding":
        embedded = client.embed(model=model_id, input=["hello", "local retrieval"])
        if len(embedded.embeddings) != 2 or any(
            not embedding for embedding in embedded.embeddings
        ):
            raise AssertionError(
                f"unexpected official-client embedding response: {embedded!r}"
            )
        result["status"] = "embedding_ok"
        return result

    if tool_only:
        tool = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Return the current weather for one city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                    "additionalProperties": False,
                },
            },
        }
        user_message = {
            "role": "user",
            "content": "What is the weather in Paris?",
        }

        def require_sdk_call(call: object, label: str) -> None:
            function = getattr(call, "function", None)
            name = getattr(function, "name", None)
            arguments = getattr(function, "arguments", None)
            try:
                decoded_arguments = dict(arguments)
            except (TypeError, ValueError) as error:
                raise AssertionError(
                    f"{label} arguments were not a decoded mapping: {call!r}"
                ) from error
            if name != "get_weather" or decoded_arguments != {"city": "Paris"}:
                raise AssertionError(f"{label} was invalid: {call!r}")

        chat = client.chat(
            model=model_id,
            messages=[user_message],
            tools=[tool],
            options={"num_predict": max_tokens, "temperature": 0},
        )
        calls = chat.message.tool_calls or []
        if chat.model != model_id or chat.done is not True or len(calls) != 1:
            raise AssertionError(
                f"official Ollama client decoded an invalid tool response: {chat!r}"
            )
        require_sdk_call(calls[0], "official Ollama tool call")

        chunks = client.chat(
            model=model_id,
            messages=[user_message],
            tools=[tool],
            options={"num_predict": max_tokens, "temperature": 0},
            stream=True,
        )
        terminal = None
        streamed_calls: list[object] = []
        for chunk in chunks:
            terminal = chunk
            streamed_calls.extend(chunk.message.tool_calls or [])
        if (
            terminal is None
            or terminal.model != model_id
            or terminal.done is not True
            or len(streamed_calls) != 1
        ):
            raise AssertionError(
                "official Ollama client decoded an invalid tool stream: "
                f"terminal={terminal!r}, calls={streamed_calls!r}"
            )
        require_sdk_call(streamed_calls[0], "official streamed Ollama tool call")

        assistant_message = chat.message.model_dump(mode="json", exclude_none=True)
        continuation = client.chat(
            model=model_id,
            messages=[
                user_message,
                assistant_message,
                {
                    "role": "tool",
                    "tool_name": "get_weather",
                    "content": "15 C and clear",
                },
            ],
            options={"num_predict": max_tokens, "temperature": 0},
        )
        if (
            continuation.model != model_id
            or continuation.done is not True
            or not isinstance(continuation.message.content, str)
            or not continuation.message.content
        ):
            raise AssertionError(
                "official Ollama client decoded an invalid tool continuation: "
                f"{continuation!r}"
            )
        result["status"] = "tool_ok"
        return result

    if structured_only:
        schema = {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        }
        chat = client.chat(
            model=model_id,
            messages=[{"role": "user", "content": "Return ok as JSON."}],
            format=schema,
            options={"num_predict": max_tokens, "temperature": 0},
        )
        if (
            chat.model != model_id
            or chat.done is not True
            or chat.message.role != "assistant"
            or json.loads(chat.message.content) != {"ok": True}
        ):
            raise AssertionError(
                f"official Ollama client decoded invalid schema output: {chat!r}"
            )
        chunks = client.generate(
            model=model_id,
            prompt="Return ok as JSON.",
            format="json",
            options={"num_predict": max_tokens, "temperature": 0},
            stream=True,
        )
        terminal = None
        generated_text = ""
        for chunk in chunks:
            terminal = chunk
            if isinstance(chunk.response, str):
                generated_text += chunk.response
        if (
            terminal is None
            or terminal.done is not True
            or terminal.model != model_id
            or json.loads(generated_text) != {"ok": True}
        ):
            raise AssertionError(
                "official Ollama client decoded invalid JSON stream: "
                f"terminal={terminal!r}, output={generated_text!r}"
            )
        result["status"] = "structured_ok"
        return result

    chat = client.chat(
        model=model_id,
        messages=[{"role": "user", "content": "Say hello briefly."}],
        options={
            "num_predict": max_tokens,
            "temperature": 0,
            "stop": ["__BLOOM_STOP_NEVER__"],
        },
    )
    if (
        chat.model != model_id
        or chat.message.role != "assistant"
        or not isinstance(chat.message.content, str)
        or not chat.message.content
    ):
        raise AssertionError(f"unexpected official-client chat response: {chat!r}")
    chunks = client.generate(
        model=model_id,
        prompt="Say hello briefly.",
        options={
            "num_predict": max_tokens,
            "temperature": 0,
            "stop": ["__BLOOM_STOP_NEVER__"],
        },
        stream=True,
    )
    terminal = None
    generated_text = ""
    for chunk in chunks:
        terminal = chunk
        if isinstance(chunk.response, str):
            generated_text += chunk.response
    if terminal is None or terminal.done is not True:
        raise AssertionError("official Ollama client stream omitted its terminal response")
    if not generated_text:
        raise AssertionError("official Ollama client stream completed without text output")
    result["status"] = "generation_ok"
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=os.environ.get("BLOOM_MODEL_PATH", "/tmp/smoke_model"),
        help="Optional model directory or GGUF file. A missing path skips generation only.",
    )
    parser.add_argument("--server-bin", default="target/release/bloom_server")
    parser.add_argument("--build", action="store_true", help="Build bloom_server first.")
    parser.add_argument("--backend", default="candle")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument(
        "--semantic-system",
        help="Optional system message for an exact trained-model chat assertion.",
    )
    parser.add_argument(
        "--semantic-prompt",
        help="Optional user message for an exact trained-model chat assertion.",
    )
    parser.add_argument(
        "--expected-output",
        help="Require buffered and streamed chat text to match this value after stripping.",
    )
    parser.add_argument("--startup-timeout", type=float, default=120.0)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument(
        "--api-key",
        default=os.environ.get("BLOOM_API_KEY", ""),
        help="Enable and validate API-key authentication for the spawned server.",
    )
    parser.add_argument(
        "--require-model",
        action="store_true",
        default=os.environ.get("BLOOM_REQUIRE_MODEL", "").lower() in {"1", "true", "yes"},
        help="Fail rather than skip generation when the model path is missing.",
    )
    parser.add_argument(
        "--catalog-only",
        action="store_true",
        help=(
            "Copy the supplied model into the isolated managed catalog and "
            "activate it through an empty generate request instead of startup loading."
        ),
    )
    parser.add_argument(
        "--require-ollama-sdk",
        action="store_true",
        help="Fail when the optional official Python client check cannot run.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--structured-only",
        action="store_true",
        default=os.environ.get("BLOOM_STRUCTURED_ONLY", "").lower()
        in {"1", "true", "yes"},
        help="Require successful chat/generate JSON and JSON Schema lifecycles.",
    )
    mode.add_argument(
        "--tool-only",
        action="store_true",
        default=os.environ.get("BLOOM_TOOL_ONLY", "").lower()
        in {"1", "true", "yes"},
        help="Require successful chat function-call and result lifecycles.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (args.semantic_prompt is None) != (args.expected_output is None):
        print(
            "FAIL: --semantic-prompt and --expected-output must be provided together",
            file=sys.stderr,
        )
        return 1
    if not 1 <= args.max_tokens <= 32_768:
        print("FAIL: --max-tokens must be between 1 and 32768", file=sys.stderr)
        return 1
    model = Path(args.model)
    has_model = model.exists()
    if args.catalog_only and not has_model:
        print("FAIL: --catalog-only requires an existing --model path", file=sys.stderr)
        return 1
    if not has_model and args.require_model:
        print(f"FAIL: model not found at {model} (required by --require-model)", file=sys.stderr)
        return 1

    server_bin = Path(args.server_bin)
    if args.build or not server_bin.exists():
        subprocess.run(["cargo", "build", "--release", "--bin", "bloom_server"], check=True)
    if not server_bin.exists():
        raise RuntimeError(f"server binary not found after build: {server_bin}")

    port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    with tempfile.TemporaryDirectory(prefix="bloom-ollama-smoke-") as models_dir:
        if args.catalog_only:
            catalog_model = Path(models_dir) / model.name
            if model.is_dir():
                shutil.copytree(model, catalog_model)
            else:
                shutil.copy2(model, catalog_model)
        command = [
            str(server_bin),
            "--models-dir",
            models_dir,
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
            "--max-body-bytes",
            "65536",
        ]
        if has_model and not args.catalog_only:
            command.extend(["--model", str(model)])
        environment = os.environ.copy()
        if args.api_key:
            environment["BLOOM_API_KEY"] = args.api_key
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
        )
        try:
            if has_model and not args.catalog_only:
                wait_ready(base_url, process, args.startup_timeout)
            else:
                wait_healthy(base_url, process, args.startup_timeout)
            readiness_http_status, readiness = request_json(
                f"{base_url}/ready", timeout=2.0, allow_http_error=True
            )
            try:
                validate_readiness_document(readiness)
            except ReadinessContractError as error:
                raise RuntimeError(
                    f"bloom_server returned an incompatible readiness contract: {error}"
                ) from error
            expected_readiness_status = (
                "ready" if has_model and not args.catalog_only else "not_ready"
            )
            expected_readiness_http_status = (
                200 if expected_readiness_status == "ready" else 503
            )
            if (
                readiness_http_status != expected_readiness_http_status
                or readiness["status"] != expected_readiness_status
            ):
                raise AssertionError(
                    "unexpected initial readiness state: "
                    f"HTTP {readiness_http_status}, {readiness}"
                )
            readiness_contract = {
                "status": readiness["status"],
                "schema_version": readiness["schema_version"],
                "protocol_version": readiness["protocol_version"],
                "minimum_ui_protocol_version": readiness[
                    "minimum_ui_protocol_version"
                ],
                "maximum_ui_protocol_version": readiness[
                    "maximum_ui_protocol_version"
                ],
                "server_version": readiness["server_version"],
                "model_tasks": readiness["model_tasks"],
            }
            headers = (
                {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
            )
            auth_status = "disabled"
            if args.api_key:
                status, auth_response_headers, error = request_protocol_error(
                    f"{base_url}/api/version",
                    method="GET",
                    headers={},
                )
                if (
                    status != 401
                    or auth_response_headers.get("www-authenticate")
                    != 'Bearer realm="Bloom"'
                ):
                    raise AssertionError(
                        "unexpected Ollama auth challenge: "
                        f"{status} {auth_response_headers} {error}"
                    )
                auth_status = "validated"

            status, version = request_json(f"{base_url}/api/version", headers=headers)
            if status != 200 or not isinstance(version.get("version"), str):
                raise AssertionError(f"unexpected /api/version response: {version}")
            status, tags = request_json(f"{base_url}/api/tags", headers=headers)
            if status != 200 or not isinstance(tags.get("models"), list):
                raise AssertionError(f"unexpected /api/tags response: {tags}")
            status, processes = request_json(f"{base_url}/api/ps", headers=headers)
            if status != 200 or not isinstance(processes.get("models"), list):
                raise AssertionError(f"unexpected /api/ps response: {processes}")

            missing_path = "/api/bloom-smoke-not-found"
            status, error_headers, missing_route = request_protocol_error(
                f"{base_url}{missing_path}", method="GET", headers=headers
            )
            if (
                status != 404
                or not error_headers.get("content-type", "").startswith(
                    "application/json"
                )
                or error_headers.get("cache-control") != "no-store"
            ):
                raise AssertionError(
                    "Ollama route fallback did not preserve its HTTP boundary: "
                    f"{status}, {error_headers}, {missing_route}"
                )
            if missing_path in missing_route["error"]:
                raise AssertionError("Ollama route fallback reflected the request path")

            status, error_headers, method_error = request_protocol_error(
                f"{base_url}/api/show", method="GET", headers=headers
            )
            if (
                status != 405
                or error_headers.get("allow") != "POST"
                or not error_headers.get("content-type", "").startswith(
                    "application/json"
                )
                or error_headers.get("cache-control") != "no-store"
            ):
                raise AssertionError(
                    "Ollama method fallback did not preserve status and Allow: "
                    f"{status}, {error_headers}, {method_error}"
                )
            protocol_errors = {"not_found": "ok", "method_not_allowed": "ok"}

            private_marker = b"bloom-private-request-value"
            for case_headers, request_body, expected_status in [
                (
                    {"Content-Type": "application/json"},
                    b'{"model":"default","messages":[' + private_marker,
                    400,
                ),
                ({}, b"{}", 400),
                (
                    {"Content-Type": "application/json"},
                    b"x" * 65537,
                    413,
                ),
            ]:
                rejection_headers = dict(headers)
                rejection_headers.update(case_headers)
                status, error_headers, rejection = request_protocol_error(
                    f"{base_url}/api/chat",
                    method="POST",
                    headers=rejection_headers,
                    body=request_body,
                )
                if (
                    status != expected_status
                    or not error_headers.get("content-type", "").startswith(
                        "application/json"
                    )
                    or error_headers.get("cache-control") != "no-store"
                ):
                    raise AssertionError(
                        "Ollama framework rejection has an invalid boundary: "
                        f"{status}, {error_headers}, {rejection}"
                    )
                if private_marker.decode() in rejection["error"]:
                    raise AssertionError(
                        "Ollama framework rejection reflected request data"
                    )
            protocol_errors["framework_rejections"] = "ok"

            status, missing_delete = request_json(
                f"{base_url}/api/delete",
                {"model": MISSING_DELETE_PROBE_MODEL},
                method="DELETE",
                headers=headers,
                allow_http_error=True,
            )
            if status != 404 or not isinstance(missing_delete.get("error"), str):
                raise AssertionError(
                    f"unexpected /api/delete admission response: {missing_delete}"
                )
            delete_admission = "validated"

            status, missing_pull = request_json(
                f"{base_url}/api/pull",
                {"model": MISSING_PULL_PROBE_MODEL, "stream": False},
                headers=headers,
                allow_http_error=True,
            )
            if status != 403 or not isinstance(missing_pull.get("error"), str):
                raise AssertionError(
                    f"unexpected /api/pull admission response: {missing_pull}"
                )
            status, insecure_pull = request_json(
                f"{base_url}/api/pull",
                {
                    "model": MISSING_PULL_PROBE_MODEL,
                    "insecure": True,
                    "stream": False,
                },
                headers=headers,
                allow_http_error=True,
            )
            if status != 400 or not isinstance(insecure_pull.get("error"), str):
                raise AssertionError(
                    f"insecure pull semantics did not fail closed: {insecure_pull}"
                )
            pull_admission = "validated"

            status, rejected = request_json(
                f"{base_url}/api/chat",
                {
                    "model": "default",
                    "messages": [{"role": "user", "content": "Do not run."}],
                    "options": {"top_k": 40},
                },
                headers=headers,
                allow_http_error=True,
            )
            if status != 400 or not isinstance(rejected.get("error"), str):
                raise AssertionError(f"unsupported semantics did not fail closed: {rejected}")

            status, rejected_embed = request_json(
                f"{base_url}/api/embed",
                {
                    "model": "default",
                    "input": "Do not run.",
                    "options": {"num_ctx": 1024},
                },
                headers=headers,
                allow_http_error=True,
            )
            if status != 400 or not isinstance(rejected_embed.get("error"), str):
                raise AssertionError(
                    f"unsupported embedding semantics did not fail closed: {rejected_embed}"
                )

            model_id = None
            model_kind = None
            activation_status = "not_applicable"
            generation_status = "skipped_no_model"
            embedding_status = "skipped_no_model"
            structured_lifecycle: dict[str, str] | None = None
            tool_lifecycle: dict[str, str] | None = None
            residency_status = "skipped_no_model"
            if not has_model:
                status, unavailable_embed = request_json(
                    f"{base_url}/api/embed",
                    {"model": "default", "input": "Hello"},
                    headers=headers,
                    allow_http_error=True,
                )
                if status != 404 or not isinstance(unavailable_embed.get("error"), str):
                    raise AssertionError(
                        f"unexpected empty-runtime embedding admission: {unavailable_embed}"
                    )
            if has_model:
                if args.catalog_only:
                    if processes["models"]:
                        raise AssertionError(
                            "catalog-only smoke unexpectedly started with a loaded model"
                        )
                    if len(tags["models"]) != 1:
                        raise AssertionError(
                            "catalog-only smoke expected exactly one discovered model: "
                            f"{tags}"
                        )
                    model_id = tags["models"][0].get("model")
                    if not isinstance(model_id, str) or not model_id:
                        raise AssertionError(f"/api/tags omitted a model ID: {tags}")
                    status, preload = request_json(
                        f"{base_url}/api/generate",
                        {"model": model_id, "prompt": "", "stream": False},
                        headers=headers,
                        timeout=args.request_timeout,
                    )
                    if (
                        status != 200
                        or preload.get("done") is not True
                        or preload.get("done_reason") != "load"
                        or preload.get("model") != model_id
                    ):
                        raise AssertionError(
                            f"catalog model preload returned an invalid response: {preload}"
                        )
                    status, processes = request_json(
                        f"{base_url}/api/ps", headers=headers
                    )
                    if status != 200 or len(processes.get("models", [])) != 1:
                        raise AssertionError(
                            f"catalog model activation was not published by /api/ps: {processes}"
                        )
                    activation_status = "on_demand"
                else:
                    activation_status = "startup"
                if not processes["models"]:
                    raise AssertionError("/api/ps omitted the active model")
                process_model_id = processes["models"][0].get("model")
                if model_id is not None and process_model_id != model_id:
                    raise AssertionError(
                        "activated model identity changed between tags and process discovery: "
                        f"{model_id!r} != {process_model_id!r}"
                    )
                model_id = process_model_id
                if not isinstance(model_id, str) or not model_id:
                    raise AssertionError(f"/api/ps omitted a model ID: {processes}")
                status, shown = request_json(
                    f"{base_url}/api/show", {"model": model_id}, headers=headers
                )
                if status != 200 or not isinstance(shown.get("details"), dict):
                    raise AssertionError(f"unexpected /api/show response: {shown}")
                capabilities = shown.get("capabilities")
                if not isinstance(capabilities, list):
                    raise AssertionError(f"/api/show omitted capabilities: {shown}")
                model_kind = "embedding" if "embedding" in capabilities else "generation"
                if (args.structured_only or args.tool_only) and model_kind != "generation":
                    raise AssertionError(
                        "structured and tool smoke modes require a text-generation model"
                    )
                if model_kind == "embedding":
                    status, embedded = request_json(
                        f"{base_url}/api/embed",
                        {
                            "model": model_id,
                            "input": ["hello", "local retrieval"],
                            "truncate": False,
                            "dimensions": 2,
                        },
                        headers=headers,
                        timeout=args.request_timeout,
                    )
                    if status != 200 or embedded.get("model") != model_id:
                        raise AssertionError(f"unexpected /api/embed response: {embedded}")
                    check_embedding_vectors(
                        embedded,
                        expected_count=2,
                        expected_dimensions=2,
                        normalized=True,
                    )
                    status, legacy = request_json(
                        f"{base_url}/api/embeddings",
                        {"model": model_id, "prompt": "hello"},
                        headers=headers,
                        timeout=args.request_timeout,
                    )
                    legacy_vector = legacy.get("embedding")
                    if status != 200 or not isinstance(legacy_vector, list) or not legacy_vector:
                        raise AssertionError(
                            f"unexpected legacy /api/embeddings response: {legacy}"
                        )
                    generation_status = "not_applicable_embedding_model"
                    embedding_status = "ok"
                else:
                    if args.tool_only:
                        tool_lifecycle = validate_tool_outputs(
                            base_url,
                            model_id,
                            headers,
                            timeout=args.request_timeout,
                            max_tokens=max(args.max_tokens, 64),
                        )
                        generation_status = "tool_ok"
                    elif args.structured_only:
                        structured_lifecycle = validate_structured_outputs(
                            base_url,
                            model_id,
                            headers,
                            timeout=args.request_timeout,
                            max_tokens=max(args.max_tokens, 8),
                        )
                        generation_status = "structured_ok"
                    else:
                        chat_messages = [
                            {
                                "role": "user",
                                "content": args.semantic_prompt or "Say hello briefly.",
                            }
                        ]
                        if args.semantic_system:
                            chat_messages.insert(
                                0,
                                {"role": "system", "content": args.semantic_system},
                            )
                        status, chat = request_json(
                            f"{base_url}/api/chat",
                            {
                                "model": model_id,
                                "messages": chat_messages,
                                "options": {
                                    "num_predict": args.max_tokens,
                                    "temperature": 0,
                                    "stop": ["__BLOOM_STOP_NEVER__"],
                                },
                                "stream": False,
                            },
                            headers=headers,
                            timeout=args.request_timeout,
                        )
                        if (
                            status != 200
                            or chat.get("model") != model_id
                            or chat.get("done") is not True
                            or not isinstance(
                                chat.get("message", {}).get("content"), str
                            )
                            or not chat["message"]["content"]
                        ):
                            raise AssertionError(
                                f"unexpected non-streaming chat response: {chat}"
                            )
                        if (
                            args.expected_output is not None
                            and chat["message"]["content"].strip()
                            != args.expected_output
                        ):
                            raise AssertionError(
                                "buffered Ollama chat returned the wrong semantic output: "
                                f"expected {args.expected_output!r}, "
                                f"got {chat['message']['content']!r}"
                            )
                        status, generated = request_json(
                            f"{base_url}/api/generate",
                            {
                                "model": model_id,
                                "prompt": "Say hello briefly.",
                                "options": {
                                    "num_predict": args.max_tokens,
                                    "temperature": 0,
                                    "stop": ["__BLOOM_STOP_NEVER__"],
                                },
                                "stream": False,
                            },
                            headers=headers,
                            timeout=args.request_timeout,
                        )
                        if (
                            status != 200
                            or generated.get("done") is not True
                            or not isinstance(generated.get("response"), str)
                            or not generated["response"]
                        ):
                            raise AssertionError(
                                f"unexpected generate response: {generated}"
                            )
                        events = request_ndjson(
                            f"{base_url}/api/chat",
                            {
                                "model": model_id,
                                "messages": chat_messages,
                                "options": {
                                    "num_predict": args.max_tokens,
                                    "temperature": 0,
                                    "stop": ["__BLOOM_STOP_NEVER__"],
                                },
                                "stream": True,
                            },
                            headers=headers,
                            timeout=args.request_timeout,
                        )
                        if events[-1].get("model") != model_id:
                            raise AssertionError(
                                f"stream reported the wrong model: {events[-1]}"
                            )
                        streamed_text = "".join(
                            event.get("message", {}).get("content", "")
                            for event in events
                            if isinstance(event.get("message"), dict)
                            and isinstance(event["message"].get("content"), str)
                        )
                        if not streamed_text:
                            raise AssertionError(
                                "Ollama chat stream completed without text output"
                            )
                        if (
                            args.expected_output is not None
                            and streamed_text.strip() != args.expected_output
                        ):
                            raise AssertionError(
                                "streamed Ollama chat returned the wrong semantic output: "
                                f"expected {args.expected_output!r}, got {streamed_text!r}"
                            )
                        generation_status = "ok"
                    embedding_status = "not_applicable_generation_model"

            sdk = check_official_sdk(
                base_url,
                headers,
                model_id,
                model_kind,
                max(args.max_tokens, 64)
                if args.tool_only
                else max(args.max_tokens, 8)
                if args.structured_only
                else args.max_tokens,
                structured_only=args.structured_only,
                tool_only=args.tool_only,
            )
            if args.require_ollama_sdk and sdk["status"] == "skipped":
                raise AssertionError(sdk["reason"])
            if (
                args.structured_only
                and sdk["status"] != "skipped"
                and sdk["status"] != "structured_ok"
            ):
                raise AssertionError(
                    f"official Ollama SDK did not complete structured outputs: {sdk}"
                )
            if (
                args.tool_only
                and sdk["status"] != "skipped"
                and sdk["status"] != "tool_ok"
            ):
                raise AssertionError(
                    f"official Ollama SDK did not complete function tools: {sdk}"
                )
            if has_model:
                if model_kind == "embedding":
                    status, residency_probe = request_json(
                        f"{base_url}/api/embed",
                        {
                            "model": model_id,
                            "input": "Timed residency probe",
                            "keep_alive": "500ms",
                        },
                        headers=headers,
                        timeout=args.request_timeout,
                    )
                    if status != 200 or residency_probe.get("model") != model_id:
                        raise AssertionError(
                            f"timed embedding residency probe failed: {residency_probe}"
                        )
                else:
                    residency_events = request_ndjson(
                        f"{base_url}/api/generate",
                        {
                            "model": model_id,
                            "prompt": "Timed residency probe.",
                            "keep_alive": "500ms",
                            "options": {"num_predict": 1, "temperature": 0},
                            "stream": True,
                        },
                        headers=headers,
                        timeout=args.request_timeout,
                    )
                    if (
                        residency_events[-1].get("model") != model_id
                        or residency_events[-1].get("done") is not True
                    ):
                        raise AssertionError(
                            "timed generation residency stream failed: "
                            f"{residency_events}"
                        )
                status, expiring_processes = request_json(
                    f"{base_url}/api/ps", headers=headers
                )
                expiring_models = expiring_processes.get("models")
                expires_at = (
                    expiring_models[0].get("expires_at")
                    if isinstance(expiring_models, list) and len(expiring_models) == 1
                    else None
                )
                if (
                    status != 200
                    or not isinstance(expires_at, str)
                    or not expires_at.endswith("Z")
                    or expires_at == "9999-12-31T23:59:59Z"
                ):
                    raise AssertionError(
                        "timed residency was not published by /api/ps: "
                        f"{expiring_processes}"
                    )
                processes = wait_for_no_running_models(base_url, headers, 5.0)
                residency_status = "timed_expiry_ok"
            print(
                json.dumps(
                    {
                        "status": "ok",
                        "server_version": version["version"],
                        "readiness_contract": readiness_contract,
                        "authentication": auth_status,
                        "discovered_models": len(tags["models"]),
                        "running_models": len(processes["models"]),
                        "activation": activation_status,
                        "delete_admission": delete_admission,
                        "pull_admission": pull_admission,
                        "protocol_errors": protocol_errors,
                        "generation": generation_status,
                        "semantic_output": "validated"
                        if args.expected_output is not None
                        else "not_requested",
                        "embedding": embedding_status,
                        "residency": residency_status,
                        "structured_lifecycle": structured_lifecycle,
                        "tool_lifecycle": tool_lifecycle,
                        "official_sdk": sdk,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            if not has_model:
                print(
                    f"SKIP: model not found at {model}; discovery and admission passed"
                )
            return 0
        finally:
            failed = sys.exc_info()[0] is not None
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            if failed or process.returncode not in (0, -15):
                stderr = process.stderr.read() if process.stderr else ""
                if stderr:
                    print("bloom_server stderr tail:", file=sys.stderr)
                    print(stderr[-4000:], file=sys.stderr)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, RuntimeError, TimeoutError, urllib.error.URLError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

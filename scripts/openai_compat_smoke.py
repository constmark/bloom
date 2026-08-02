#!/usr/bin/env python3
"""Smoke-test Bloom's OpenAI-compatible HTTP API.

Discovery, authentication, official-client decoding, and fail-closed admission
run without a model. Model-backed checks are reported as skipped unless
--require-model or BLOOM_REQUIRE_MODEL=1 is set. Use --embedding-only with an
embedding model to validate vector projection and reranking, or --tool-only
with a deterministic tool fixture to require successful function lifecycles.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from readiness_contract import (
    READINESS_SCHEMA_VERSION,
    ReadinessContractError,
    validate_readiness_document,
)

MAX_BOUNDARY_RESPONSE_BYTES = 64 * 1024


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


def request_status_headers(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    body: bytes | None = None,
    timeout: float = 5.0,
) -> tuple[int, dict[str, str], bytes]:
    req = urllib.request.Request(
        url, data=body, headers=dict(headers or {}), method=method
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = response.read(MAX_BOUNDARY_RESPONSE_BYTES + 1)
            status = response.status
            response_headers = response.headers
    except urllib.error.HTTPError as error:
        body = error.read(MAX_BOUNDARY_RESPONSE_BYTES + 1)
        status = error.code
        response_headers = error.headers
    if len(body) > MAX_BOUNDARY_RESPONSE_BYTES:
        raise AssertionError(
            f"response from {url} exceeded {MAX_BOUNDARY_RESPONSE_BYTES} bytes"
        )
    return (
        status,
        {key.lower(): value for key, value in response_headers.items()},
        body,
    )


def assert_openai_protocol_error(
    body: bytes, *, expected_type: str, request_path: str
) -> None:
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AssertionError(
            f"OpenAI protocol error was not valid UTF-8 JSON for {request_path}"
        ) from error
    error = payload.get("error") if isinstance(payload, dict) else None
    message = error.get("message") if isinstance(error, dict) else None
    if (
        not isinstance(error, dict)
        or error.get("type") != expected_type
        or not isinstance(message, str)
        or not 1 <= len(message) <= 1024
    ):
        raise AssertionError(
            f"unexpected OpenAI protocol error for {request_path}: {payload!r}"
        )


def valid_http_request_id(value: str | None) -> bool:
    return (
        isinstance(value, str)
        and 1 <= len(value) <= 128
        and all(
            character.isascii()
            and (character.isalnum() or character in "-_.:")
            for character in value
        )
    )


def validate_http_request_correlation(base_url: str) -> dict[str, str]:
    allowed_origin = "https://bloom-smoke.invalid"
    status, generated_headers, _body = request_status_headers(
        f"{base_url}/health",
        headers={"Origin": allowed_origin},
    )
    generated = generated_headers.get("x-request-id")
    if status != 200 or not valid_http_request_id(generated):
        raise AssertionError(
            "health response did not publish a bounded request ID: "
            f"{status}, {generated_headers}"
        )
    if generated_headers.get("cache-control") != "no-store":
        raise AssertionError(f"health response is cacheable: {generated_headers}")
    exposed = {
        value.strip().lower()
        for value in generated_headers.get("access-control-expose-headers", "").split(",")
    }
    missing_exposed = {"x-request-id", "retry-after", "www-authenticate"}.difference(
        exposed
    )
    if missing_exposed:
        raise AssertionError(
            "browser clients cannot read required support response headers "
            f"{sorted(missing_exposed)}: {generated_headers}"
        )
    if generated_headers.get("access-control-allow-origin") != allowed_origin:
        raise AssertionError(
            f"server did not publish the exact allowed browser origin: {generated_headers}"
        )

    status, rejected_headers, _body = request_status_headers(
        f"{base_url}/health",
        headers={"Origin": "https://malicious-smoke.invalid"},
    )
    if status != 403 or "access-control-allow-origin" in rejected_headers:
        raise AssertionError(
            f"server admitted an untrusted browser origin: {status}, {rejected_headers}"
        )

    supplied = "proxy.node_1:request-42"
    status, supplied_headers, _body = request_status_headers(
        f"{base_url}/health",
        headers={"X-Request-ID": supplied},
    )
    if status != 200 or supplied_headers.get("x-request-id") != supplied:
        raise AssertionError(
            "server did not preserve a bounded client request ID: "
            f"{status}, {supplied_headers}"
        )
    if supplied_headers.get("cache-control") != "no-store":
        raise AssertionError(f"correlated health response is cacheable: {supplied_headers}")

    invalid = "invalid request id"
    missing_path = "/v1/bloom-smoke-not-found"
    status, rejected_headers, body = request_status_headers(
        f"{base_url}{missing_path}",
        headers={"X-Request-ID": invalid},
    )
    replacement = rejected_headers.get("x-request-id")
    if status != 404 or replacement == invalid or not valid_http_request_id(replacement):
        raise AssertionError(
            "error response did not replace an unsafe request ID: "
            f"{status}, {rejected_headers}"
        )
    if rejected_headers.get("cache-control") != "no-store":
        raise AssertionError(f"unknown API response is cacheable: {rejected_headers}")
    if not rejected_headers.get("content-type", "").startswith("application/json"):
        raise AssertionError(f"unknown API response is not JSON: {rejected_headers}")
    assert_openai_protocol_error(
        body, expected_type="not_found_error", request_path=missing_path
    )
    if missing_path.encode() in body:
        raise AssertionError("OpenAI route fallback reflected the request path")

    method_path = "/v1/models"
    status, method_headers, body = request_status_headers(
        f"{base_url}{method_path}", method="POST"
    )
    if status != 405 or method_headers.get("allow") != "GET,HEAD":
        raise AssertionError(
            "OpenAI method fallback did not preserve status and Allow: "
            f"{status}, {method_headers}"
        )
    if (
        method_headers.get("cache-control") != "no-store"
        or not method_headers.get("content-type", "").startswith("application/json")
    ):
        raise AssertionError(f"invalid method error headers: {method_headers}")
    assert_openai_protocol_error(
        body, expected_type="invalid_request_error", request_path=method_path
    )

    return {
        "generated": "ok",
        "preserved": "ok",
        "unsafe_replaced": "ok",
        "retry_after_exposed": "ok",
        "authentication_challenge_exposed": "ok",
        "browser_origin_rejected": "ok",
        "no_store": "ok",
        "protocol_404": "ok",
        "protocol_405": "ok",
    }


def validate_framework_rejections(
    base_url: str, auth_headers: dict[str, str]
) -> dict[str, str]:
    private_marker = b"bloom-private-request-value"
    cases = [
        (
            "/v1/chat/completions",
            {"Content-Type": "application/json"},
            b'{"messages":[{"content":"' + private_marker,
            400,
        ),
        (
            "/v1/chat/completions",
            {"Content-Type": "application/json"},
            b"[]",
            422,
        ),
        ("/v1/chat/completions", {}, b"{}", 415),
        (
            "/v1/model-management/downloads/inspect",
            {"Content-Type": "application/json"},
            b"x" * 4097,
            413,
        ),
    ]
    for path, case_headers, request_body, expected_status in cases:
        headers = dict(auth_headers)
        headers.update(case_headers)
        status, response_headers, body = request_status_headers(
            f"{base_url}{path}",
            method="POST",
            headers=headers,
            body=request_body,
        )
        if (
            status != expected_status
            or not response_headers.get("content-type", "").startswith(
                "application/json"
            )
            or response_headers.get("cache-control") != "no-store"
            or not valid_http_request_id(response_headers.get("x-request-id"))
        ):
            raise AssertionError(
                f"framework rejection has an invalid boundary for {path}: "
                f"{status}, {response_headers}, {body!r}"
            )
        assert_openai_protocol_error(
            body, expected_type="invalid_request_error", request_path=path
        )
        if private_marker in body:
            raise AssertionError(f"framework rejection reflected request data for {path}")
    return {
        "malformed_json": "ok",
        "schema_mismatch": "ok",
        "unsupported_media_type": "ok",
        "body_limit": "ok",
    }


def validate_readiness_contract(
    base_url: str,
    *,
    expected_ready: bool,
    expected_tasks: list[str],
) -> dict[str, object]:
    payload = request_json(f"{base_url}/ready", timeout=2.0, allow_http_error=True)
    try:
        validate_readiness_document(payload)
    except ReadinessContractError as error:
        raise AssertionError(f"invalid readiness contract: {error}") from error
    expected_status = "ready" if expected_ready else "not_ready"
    expected_http_status = 200 if expected_ready else 503
    actual_http_status = payload.get("_http_status", 200)
    if (
        payload.get("status") != expected_status
        or actual_http_status != expected_http_status
        or payload.get("model_tasks") != expected_tasks
    ):
        raise AssertionError(f"unexpected readiness contract identity or state: {payload}")
    protocol_version = payload.get("protocol_version")
    minimum_ui_protocol_version = payload.get("minimum_ui_protocol_version")
    maximum_ui_protocol_version = payload.get("maximum_ui_protocol_version")
    server_version = payload.get("server_version")
    return {
        "status": expected_status,
        "schema_version": READINESS_SCHEMA_VERSION,
        "protocol_version": protocol_version,
        "minimum_ui_protocol_version": minimum_ui_protocol_version,
        "maximum_ui_protocol_version": maximum_ui_protocol_version,
        "server_version": server_version,
        "model_tasks": expected_tasks,
    }


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


def extract_responses_output_text(response: dict) -> str:
    parts: list[str] = []
    for item in response.get("output", []):
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if isinstance(content, dict) and content.get("type") == "output_text":
                text = content.get("text")
                if isinstance(text, str):
                    parts.append(text)
    return "".join(parts)


def validate_structured_output_round_trips(
    base_url: str,
    model_id: str,
    auth_headers: dict[str, str],
    *,
    max_tokens: int,
) -> dict[str, str]:
    expected = {"ok": True}
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }
    chat_formats = [
        {"type": "json_object"},
        {
            "type": "json_schema",
            "json_schema": {
                "name": "bloom_structured_smoke",
                "strict": True,
                "schema": schema,
            },
        },
    ]
    for response_format in chat_formats:
        response = request_json(
            f"{base_url}/v1/chat/completions",
            {
                "model": model_id,
                "messages": [{"role": "user", "content": "Return ok as JSON."}],
                "response_format": response_format,
                "max_completion_tokens": max_tokens,
                "temperature": 0,
            },
            timeout=60.0,
            headers=auth_headers,
        )
        choices = response.get("choices")
        choice = (
            choices[0]
            if isinstance(choices, list)
            and len(choices) == 1
            and isinstance(choices[0], dict)
            else {}
        )
        content = choice.get("message", {}).get("content")
        if (
            response.get("model") != model_id
            or choice.get("finish_reason") != "stop"
            or not isinstance(content, str)
            or json.loads(content) != expected
        ):
            raise AssertionError(
                f"Chat structured output violated {response_format}: {response}"
            )

    chat_stream_events = request_sse(
        f"{base_url}/v1/chat/completions",
        {
            "model": model_id,
            "messages": [{"role": "user", "content": "Return ok as JSON."}],
            "response_format": chat_formats[1],
            "max_completion_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
        },
        timeout=60.0,
        headers=auth_headers,
    )
    if not chat_stream_events or chat_stream_events[-1] != "[DONE]":
        raise AssertionError("Chat structured stream omitted [DONE]")
    chat_stream_text = ""
    chat_stream_finished = False
    for event in chat_stream_events[:-1]:
        decoded = json.loads(event)
        choices = decoded.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        content = choices[0].get("delta", {}).get("content")
        if isinstance(content, str):
            chat_stream_text += content
        if choices[0].get("finish_reason") == "stop":
            chat_stream_finished = True
    if not chat_stream_finished or json.loads(chat_stream_text) != expected:
        raise AssertionError(
            f"Chat structured stream was invalid: {chat_stream_events}"
        )

    responses_formats = [
        {"type": "json_object"},
        {
            "type": "json_schema",
            "name": "bloom_structured_smoke",
            "strict": True,
            "schema": schema,
        },
    ]
    for response_format in responses_formats:
        response = request_json(
            f"{base_url}/v1/responses",
            {
                "model": model_id,
                "input": "Return ok as JSON.",
                "text": {"format": response_format},
                "max_output_tokens": max_tokens,
                "temperature": 0,
            },
            timeout=60.0,
            headers=auth_headers,
        )
        if (
            response.get("status") != "completed"
            or response.get("model") != model_id
            or response.get("text", {}).get("format") != response_format
            or json.loads(extract_responses_output_text(response)) != expected
        ):
            raise AssertionError(
                f"Responses structured output violated {response_format}: {response}"
            )

    response_stream_events = request_sse(
        f"{base_url}/v1/responses",
        {
            "model": model_id,
            "input": "Return ok as JSON.",
            "text": {"format": responses_formats[1]},
            "max_output_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
        },
        timeout=60.0,
        headers=auth_headers,
    )
    decoded_events = [
        json.loads(event) for event in response_stream_events if event != "[DONE]"
    ]
    event_types = {event.get("type") for event in decoded_events}
    required_events = {
        "response.created",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.completed",
    }
    if not required_events.issubset(event_types):
        raise AssertionError(
            "Responses structured stream omitted events: "
            f"{sorted(required_events.difference(event_types))}"
        )
    response_stream_text = "".join(
        event.get("delta", "")
        for event in decoded_events
        if event.get("type") == "response.output_text.delta"
        and isinstance(event.get("delta"), str)
    )
    terminal = next(
        (
            event.get("response")
            for event in decoded_events
            if event.get("type") == "response.completed"
        ),
        None,
    )
    if (
        json.loads(response_stream_text) != expected
        or not isinstance(terminal, dict)
        or terminal.get("status") != "completed"
        or terminal.get("text", {}).get("format") != responses_formats[1]
    ):
        raise AssertionError(
            f"Responses structured stream was invalid: {response_stream_events}"
        )

    return {
        "chat_json_object": "ok",
        "chat_json_schema": "ok",
        "chat_stream": "ok",
        "responses_json_object": "ok",
        "responses_json_schema": "ok",
        "responses_stream": "ok",
    }


def validate_function_tool_round_trips(
    base_url: str,
    model_id: str,
    auth_headers: dict[str, str],
    *,
    max_tokens: int,
) -> dict[str, str]:
    chat_tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Return the current weather for one city.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    }
    chat_payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
        "tools": [chat_tool],
        "tool_choice": {
            "type": "function",
            "function": {"name": "get_weather"},
        },
        "parallel_tool_calls": False,
        "max_completion_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }

    def require_chat_call(response: dict, *, streamed: bool) -> dict:
        choices = response.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise AssertionError(f"function call omitted its only choice: {response}")
        choice = choices[0]
        message_key = "delta" if streamed else "message"
        message = choice.get(message_key)
        calls = message.get("tool_calls") if isinstance(message, dict) else None
        if (
            choice.get("finish_reason") != "tool_calls"
            or not isinstance(calls, list)
            or len(calls) != 1
        ):
            raise AssertionError(f"function call had an invalid terminal shape: {response}")
        call = calls[0]
        function = call.get("function") if isinstance(call, dict) else None
        arguments = function.get("arguments") if isinstance(function, dict) else None
        try:
            decoded_arguments = json.loads(arguments)
        except (TypeError, json.JSONDecodeError) as error:
            raise AssertionError(f"function arguments were not JSON: {call}") from error
        if (
            call.get("type") != "function"
            or function.get("name") != "get_weather"
            or decoded_arguments != {"city": "Paris"}
        ):
            raise AssertionError(f"function call selected invalid output: {call}")
        call_id = call.get("id")
        if not valid_http_request_id(call_id):
            raise AssertionError(f"function call ID was not bounded: {call}")
        return call

    chat = request_json(
        f"{base_url}/v1/chat/completions",
        chat_payload,
        timeout=60.0,
        headers=auth_headers,
    )
    if chat.get("model") != model_id:
        raise AssertionError(f"function call reported the wrong model: {chat}")
    chat_call = require_chat_call(chat, streamed=False)

    streaming_payload = dict(chat_payload)
    streaming_payload["stream"] = True
    stream_events = request_sse(
        f"{base_url}/v1/chat/completions",
        streaming_payload,
        timeout=60.0,
        headers=auth_headers,
    )
    if not stream_events or stream_events[-1] != "[DONE]":
        raise AssertionError(f"function-call stream omitted [DONE]: {stream_events}")
    stream_terminal = None
    for event in stream_events[:-1]:
        decoded = json.loads(event)
        choices = decoded.get("choices")
        if (
            isinstance(choices, list)
            and len(choices) == 1
            and choices[0].get("finish_reason") == "tool_calls"
        ):
            stream_terminal = decoded
    if stream_terminal is None:
        raise AssertionError(f"function-call stream omitted its terminal call: {stream_events}")
    require_chat_call(stream_terminal, streamed=True)
    if '"type":"function_calls"' in "".join(stream_events):
        raise AssertionError("function-call stream leaked Bloom's private control protocol")

    follow_up = request_json(
        f"{base_url}/v1/chat/completions",
        {
            "model": model_id,
            "messages": [
                chat_payload["messages"][0],
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [chat_call],
                },
                {
                    "role": "tool",
                    "tool_call_id": chat_call["id"],
                    "name": "get_weather",
                    "content": "15 C and clear",
                },
            ],
            "tools": [chat_tool],
            "tool_choice": "none",
            "max_completion_tokens": max_tokens,
            "temperature": 0,
        },
        timeout=60.0,
        headers=auth_headers,
    )
    follow_up_choices = follow_up.get("choices")
    if (
        follow_up.get("model") != model_id
        or not isinstance(follow_up_choices, list)
        or len(follow_up_choices) != 1
        or not isinstance(follow_up_choices[0].get("message", {}).get("content"), str)
    ):
        raise AssertionError(f"Chat tool-result continuation failed: {follow_up}")

    responses_tool = {
        "type": "function",
        "name": "get_weather",
        "description": "Return the current weather for one city.",
        "strict": True,
        "parameters": chat_tool["function"]["parameters"],
    }
    responses_payload = {
        "model": model_id,
        "input": "What is the weather in Paris?",
        "tools": [responses_tool],
        "tool_choice": {"type": "function", "name": "get_weather"},
        "parallel_tool_calls": False,
        "max_output_tokens": max_tokens,
        "temperature": 0,
        "store": True,
        "metadata": {"bloom_smoke": "raw_tool"},
    }
    response = request_json(
        f"{base_url}/v1/responses",
        responses_payload,
        timeout=60.0,
        headers=auth_headers,
    )
    calls = [item for item in response.get("output", []) if item.get("type") == "function_call"]
    if (
        response.get("status") != "completed"
        or response.get("model") != model_id
        or response.get("store") is not True
        or response.get("metadata") != {"bloom_smoke": "raw_tool"}
        or len(calls) != 1
        or calls[0].get("name") != "get_weather"
        or json.loads(calls[0].get("arguments", "null")) != {"city": "Paris"}
    ):
        raise AssertionError(f"Responses function call was invalid: {response}")

    streamed_responses_payload = dict(responses_payload)
    streamed_responses_payload["stream"] = True
    streamed_responses_payload["store"] = False
    response_events = request_sse(
        f"{base_url}/v1/responses",
        streamed_responses_payload,
        timeout=60.0,
        headers=auth_headers,
    )
    decoded_response_events = [
        json.loads(event) for event in response_events if event != "[DONE]"
    ]
    response_event_types = {event.get("type") for event in decoded_response_events}
    required_response_events = {
        "response.created",
        "response.output_item.added",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.output_item.done",
        "response.completed",
    }
    if not required_response_events.issubset(response_event_types):
        raise AssertionError(
            "Responses function-call stream omitted required events: "
            f"{sorted(required_response_events.difference(response_event_types))}"
        )
    if '"type":"function_calls"' in "".join(response_events):
        raise AssertionError("Responses stream leaked Bloom's private control protocol")

    response_id = response.get("id")
    call_id = calls[0].get("call_id")
    continuation = request_json(
        f"{base_url}/v1/responses",
        {
            "model": model_id,
            "previous_response_id": response_id,
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": "15 C and clear",
                }
            ],
            "tools": [responses_tool],
            "tool_choice": "none",
            "max_output_tokens": max_tokens,
            "temperature": 0,
            "store": True,
        },
        timeout=60.0,
        headers=auth_headers,
    )
    continuation_id = continuation.get("id")
    if (
        continuation.get("previous_response_id") != response_id
        or continuation.get("store") is not True
        or not extract_responses_output_text(continuation)
    ):
        raise AssertionError(f"Responses function-result continuation failed: {continuation}")
    input_items = request_json(
        f"{base_url}/v1/responses/{continuation_id}/input_items?order=asc&limit=100",
        headers=auth_headers,
    )
    input_types = [item.get("type") for item in input_items.get("data", [])]
    if "function_call" not in input_types or "function_call_output" not in input_types:
        raise AssertionError(
            f"Responses retained state omitted the function lifecycle: {input_items}"
        )

    for stored_id in [continuation_id, response_id]:
        status, _headers, body = request_status_headers(
            f"{base_url}/v1/responses/{stored_id}",
            method="DELETE",
            headers=auth_headers,
        )
        if status != 200 or json.loads(body).get("deleted") is not True:
            raise AssertionError(
                f"failed to delete retained response {stored_id}: {status}, {body!r}"
            )
        deleted = request_json(
            f"{base_url}/v1/responses/{stored_id}",
            allow_http_error=True,
            headers=auth_headers,
        )
        if (
            deleted.get("_http_status") != 404
            or deleted.get("error", {}).get("type") != "invalid_request_error"
        ):
            raise AssertionError(
                f"deleted response {stored_id} remained retrievable: {deleted}"
            )

    return {
        "chat": "ok",
        "chat_stream": "ok",
        "chat_continuation": "ok",
        "responses": "ok",
        "responses_stream": "ok",
        "responses_continuation": "ok",
        "retained_state": "ok",
    }


def validate_openai_sdk_model_resource(model: object, expected_id: str) -> None:
    model_id = getattr(model, "id", None)
    created = getattr(model, "created", None)
    if (
        model_id != expected_id
        or getattr(model, "object", None) != "model"
        or getattr(model, "owned_by", None) != "bloom"
        or isinstance(created, bool)
        or not isinstance(created, int)
        or created <= 0
    ):
        raise AssertionError(f"official OpenAI client decoded an invalid model: {model!r}")


def request_openai_sdk_model_free(
    base_url: str,
    api_key: str,
    *,
    expect_authentication: bool,
) -> dict:
    try:
        from openai import APIStatusError, AuthenticationError, OpenAI  # type: ignore
    except ImportError:
        return {"status": "skipped", "reason": "python package 'openai' is not installed"}

    client = OpenAI(base_url=f"{base_url}/v1", api_key=api_key)
    models = client.models.list()
    if models.data:
        raise AssertionError(
            f"official OpenAI client found models in an isolated empty catalog: {models!r}"
        )
    result = {
        "status": "ok",
        "client_version": importlib.metadata.version("openai"),
        "models": 0,
    }

    if expect_authentication:
        unauthorized = OpenAI(
            base_url=f"{base_url}/v1",
            api_key="invalid-bloom-smoke-key",
        )
        try:
            unauthorized.models.list()
        except AuthenticationError as error:
            body = error.body if isinstance(error.body, dict) else {}
            challenge = error.response.headers.get("www-authenticate")
            if (
                error.status_code != 401
                or body.get("type") != "authentication_error"
                or challenge != 'Bearer realm="Bloom"'
            ):
                raise AssertionError(
                    f"official OpenAI client decoded an unexpected auth error: {error!r}"
                ) from error
        else:
            raise AssertionError("official OpenAI client bypassed API-key authentication")
        result["authentication"] = "ok"
    else:
        result["authentication"] = "disabled"

    try:
        client.chat.completions.create(
            model="default",
            messages=[{"role": "user", "content": "This must not run."}],
            max_tokens=1,
        )
    except APIStatusError as error:
        body = error.body if isinstance(error.body, dict) else {}
        if error.status_code != 503 or body.get("type") != "model_not_loaded":
            raise AssertionError(
                f"official OpenAI client decoded an unexpected readiness error: {error!r}"
            ) from error
    else:
        raise AssertionError("official OpenAI client admitted chat without a loaded model")
    result["model_unavailable"] = "ok"

    try:
        client.models.retrieve("bloom-smoke-missing")
    except APIStatusError as error:
        body = error.body if isinstance(error.body, dict) else {}
        if error.status_code != 404 or body.get("type") != "model_not_found":
            raise AssertionError(
                f"official OpenAI client decoded an unexpected model lookup error: {error!r}"
            ) from error
    else:
        raise AssertionError("official OpenAI client retrieved a missing model")
    result["missing_model"] = "ok"

    try:
        client.responses.retrieve("resp-bloom-smoke-missing")
    except APIStatusError as error:
        body = error.body if isinstance(error.body, dict) else {}
        if error.status_code != 404 or body.get("type") != "invalid_request_error":
            raise AssertionError(
                f"official OpenAI client decoded an unexpected response lookup error: {error!r}"
            ) from error
    else:
        raise AssertionError("official OpenAI client retrieved missing response state")
    result["missing_response"] = "ok"
    return result


def request_openai_sdk_response_stream(
    base_url: str,
    model_id: str,
    max_tokens: int,
    api_key: str,
) -> dict:
    try:
        from openai import APIStatusError, OpenAI  # type: ignore
    except ImportError:
        return {"status": "skipped", "reason": "python package 'openai' is not installed"}

    client = OpenAI(base_url=f"{base_url}/v1", api_key=api_key)
    validate_openai_sdk_model_resource(client.models.retrieve(model_id), model_id)
    stream = client.responses.create(
        model=model_id,
        instructions="Answer briefly.",
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Say hello in one short sentence.",
                    }
                ],
            }
        ],
        max_output_tokens=max_tokens,
        temperature=0,
        stream=True,
        store=True,
        metadata={"bloom_smoke": "stateful"},
    )
    event_types: list[str] = []
    delta_chars = 0
    response = None
    for event in stream:
        event_types.append(event.type)
        if event.type == "response.output_text.delta":
            delta_chars += len(event.delta)
        elif event.type in ("response.completed", "response.incomplete"):
            response = event.response
    if response is None:
        raise AssertionError(f"Responses SDK stream had no terminal event: {event_types}")
    if response.model != model_id:
        raise AssertionError(f"Responses SDK reported the wrong model: {response.model}")
    if response.status not in ("completed", "incomplete"):
        raise AssertionError(f"unexpected Responses SDK status: {response.status}")
    if delta_chars <= 0 or not isinstance(response.output_text, str) or not response.output_text:
        raise AssertionError(
            "Responses SDK stream completed without decoded text output: "
            f"delta_chars={delta_chars}, response={response.model_dump(mode='json')}"
        )
    required_events = {
        "response.created",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
    }
    missing_events = sorted(required_events.difference(event_types))
    if missing_events:
        raise AssertionError(f"Responses SDK stream missed events: {missing_events}")

    retrieved = client.responses.retrieve(response.id)
    if (
        retrieved.id != response.id
        or retrieved.store is not True
        or retrieved.metadata != {"bloom_smoke": "stateful"}
    ):
        raise AssertionError(
            f"Responses SDK retrieval lost retained state: {retrieved.model_dump(mode='json')}"
        )
    first_input_page = client.responses.input_items.list(
        response.id,
        order="asc",
        limit=100,
    )
    if not first_input_page.data or first_input_page.data[0].type != "message":
        raise AssertionError(
            f"Responses SDK input-items listing was incomplete: {first_input_page}"
        )

    chained = client.responses.create(
        model=model_id,
        input="Reply with one short acknowledgement.",
        previous_response_id=response.id,
        max_output_tokens=max_tokens,
        temperature=0,
        store=True,
        metadata={"bloom_smoke": "follow_up"},
    )
    if (
        chained.previous_response_id != response.id
        or chained.store is not True
        or chained.metadata != {"bloom_smoke": "follow_up"}
        or not isinstance(chained.output_text, str)
        or not chained.output_text
    ):
        raise AssertionError(
            f"Responses SDK chaining lost state metadata: {chained.model_dump(mode='json')}"
        )
    chained_input_page = client.responses.input_items.list(
        chained.id,
        order="asc",
        limit=100,
    )
    if len(chained_input_page.data) < 3:
        raise AssertionError(
            "Responses SDK chained input-items omitted inherited history: "
            f"{chained_input_page}"
        )

    structured_schema = {
        "type": "object",
        "required": ["ok"],
        "properties": {"ok": {"type": "boolean"}},
        "additionalProperties": False,
    }
    chat_structured_status = "ok"
    try:
        chat_structured = client.chat.completions.create(
            model=model_id,
            messages=[
                {
                    "role": "user",
                    "content": "Return a JSON object with ok set to true.",
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "smoke",
                    "schema": structured_schema,
                    "strict": True,
                },
            },
            max_completion_tokens=max(max_tokens, 8),
            temperature=0,
        )
    except APIStatusError as exc:
        body = exc.body if isinstance(exc.body, dict) else {}
        error = body.get("error", body)
        if exc.status_code != 422 or error.get("type") != "invalid_response_format":
            raise
        chat_structured_status = "validated_error"
    else:
        choice = chat_structured.choices[0]
        content = choice.message.content
        if (
            chat_structured.model != model_id
            or choice.finish_reason != "stop"
            or not isinstance(content, str)
            or json.loads(content) != {"ok": True}
        ):
            raise AssertionError(
                f"structured Chat SDK output was invalid: {chat_structured}"
            )

    responses_structured_status = "ok"
    try:
        structured = client.responses.create(
            model=model_id,
            input="Return a JSON object with the boolean field ok set to true.",
            max_output_tokens=max(max_tokens, 8),
            temperature=0,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "smoke",
                    "schema": structured_schema,
                    "strict": True,
                }
            },
        )
    except APIStatusError as exc:
        body = exc.body if isinstance(exc.body, dict) else {}
        error = body.get("error", body)
        if exc.status_code != 422 or error.get("type") != "invalid_response_format":
            raise
        responses_structured_status = "validated_error"
    else:
        structured_payload = structured.model_dump(mode="json")
        if structured_payload.get("model") != model_id:
            raise AssertionError(
                f"structured Responses SDK call reported the wrong model: {structured_payload}"
            )
        if structured_payload.get("text", {}).get("format", {}).get("type") != "json_schema":
            raise AssertionError(
                f"structured Responses SDK call lost text.format: {structured_payload}"
            )
        parsed_output = json.loads(structured.output_text)
        if parsed_output != {"ok": True}:
            raise AssertionError(
                f"structured Responses SDK output violated its schema: {parsed_output}"
            )

    tool_status = "ok"
    weather_tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Return the current weather for one city.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    }
    try:
        tool_completion = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "What is the weather in Paris?"}],
            tools=[weather_tool],
            tool_choice={
                "type": "function",
                "function": {"name": "get_weather"},
            },
            parallel_tool_calls=False,
            max_completion_tokens=max(max_tokens, 32),
            temperature=0,
        )
    except APIStatusError as exc:
        body = exc.body if isinstance(exc.body, dict) else {}
        error = body.get("error", body)
        if exc.status_code != 422 or error.get("type") != "invalid_tool_call":
            raise
        tool_status = "validated_error"
    else:
        choice = tool_completion.choices[0]
        calls = choice.message.tool_calls or []
        if choice.finish_reason != "tool_calls" or len(calls) != 1:
            raise AssertionError(
                f"Chat SDK function call had an invalid terminal shape: {tool_completion}"
            )
        call = calls[0]
        if call.type != "function" or call.function.name != "get_weather":
            raise AssertionError(f"Chat SDK selected the wrong function: {call}")
        arguments = json.loads(call.function.arguments)
        if arguments != {"city": "Paris"}:
            raise AssertionError(f"Chat SDK returned invalid function arguments: {arguments}")

        follow_up = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "user", "content": "What is the weather in Paris?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [call.model_dump(mode="json")],
                },
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": "get_weather",
                    "content": "15 C and clear",
                },
            ],
            tools=[weather_tool],
            tool_choice="none",
            max_completion_tokens=max_tokens,
            temperature=0,
        )
        if follow_up.model != model_id or not isinstance(
            follow_up.choices[0].message.content, str
        ):
            raise AssertionError(
                f"Chat SDK tool-result continuation was invalid: {follow_up}"
            )

    responses_tool_status = "ok"
    responses_tool = {
        "type": "function",
        "name": "get_weather",
        "description": "Return the current weather for one city.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
    }
    responses_tool_response = None
    responses_tool_follow_up = None
    try:
        responses_tool_response = client.responses.create(
            model=model_id,
            input="What is the weather in Paris?",
            tools=[responses_tool],
            tool_choice={"type": "function", "name": "get_weather"},
            parallel_tool_calls=False,
            max_output_tokens=max(max_tokens, 32),
            temperature=0,
            store=True,
            metadata={"bloom_smoke": "responses_tool"},
        )
    except APIStatusError as exc:
        body = exc.body if isinstance(exc.body, dict) else {}
        error = body.get("error", body)
        if exc.status_code != 422 or error.get("type") != "invalid_tool_call":
            raise
        responses_tool_status = "validated_error"
    else:
        calls = [
            item
            for item in responses_tool_response.output
            if item.type == "function_call"
        ]
        if len(calls) != 1 or calls[0].name != "get_weather":
            raise AssertionError(
                "Responses SDK function call had an invalid output shape: "
                f"{responses_tool_response.model_dump(mode='json')}"
            )
        arguments = json.loads(calls[0].arguments)
        if arguments != {"city": "Paris"}:
            raise AssertionError(
                f"Responses SDK returned invalid function arguments: {arguments}"
            )
        responses_tool_follow_up = client.responses.create(
            model=model_id,
            previous_response_id=responses_tool_response.id,
            input=[
                {
                    "type": "function_call_output",
                    "call_id": calls[0].call_id,
                    "output": "15 C and clear",
                }
            ],
            tools=[responses_tool],
            tool_choice="none",
            max_output_tokens=max_tokens,
            temperature=0,
            store=True,
        )
        if (
            responses_tool_follow_up.previous_response_id
            != responses_tool_response.id
            or not isinstance(responses_tool_follow_up.output_text, str)
        ):
            raise AssertionError(
                "Responses SDK function-result continuation was invalid: "
                f"{responses_tool_follow_up.model_dump(mode='json')}"
            )
        tool_input_items = client.responses.input_items.list(
            responses_tool_follow_up.id,
            order="asc",
            limit=100,
        )
        item_types = [item.type for item in tool_input_items.data]
        if "function_call" not in item_types or "function_call_output" not in item_types:
            raise AssertionError(
                "Responses SDK retained state omitted function lifecycle items: "
                f"{item_types}"
            )

    if responses_tool_follow_up is not None:
        client.responses.delete(responses_tool_follow_up.id)
    if responses_tool_response is not None:
        client.responses.delete(responses_tool_response.id)
    client.responses.delete(chained.id)
    client.responses.delete(response.id)
    try:
        client.responses.retrieve(response.id)
    except APIStatusError as exc:
        if exc.status_code != 404:
            raise
    else:
        raise AssertionError("deleted Responses state remained retrievable")

    return {
        "status": "ok",
        "response_id": response.id,
        "response_status": response.status,
        "output_chars": len(response.output_text),
        "delta_chars": delta_chars,
        "stream_events": len(event_types),
        "chat_structured_output": chat_structured_status,
        "structured_output": responses_structured_status,
        "function_calling": tool_status,
        "responses_function_calling": responses_tool_status,
        "stateful_round_trip": "ok",
        "model_retrieve": "ok",
    }


def request_openai_sdk_embeddings(
    base_url: str,
    model_id: str,
    api_key: str,
) -> dict:
    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        return {"status": "skipped", "reason": "python package 'openai' is not installed"}

    client = OpenAI(base_url=f"{base_url}/v1", api_key=api_key)
    validate_openai_sdk_model_resource(client.models.retrieve(model_id), model_id)
    response = client.embeddings.create(
        model=model_id,
        input=["local AI runtime", "bounded retrieval"],
        dimensions=2,
        encoding_format="float",
    )
    if response.model != model_id or len(response.data) != 2:
        raise AssertionError(f"OpenAI SDK decoded an invalid embedding response: {response}")
    for index, embedding in enumerate(response.data):
        if embedding.index != index or len(embedding.embedding) != 2:
            raise AssertionError(f"OpenAI SDK decoded an invalid embedding item: {embedding}")
        norm = math.sqrt(math.fsum(float(value) ** 2 for value in embedding.embedding))
        if not math.isclose(norm, 1.0, rel_tol=1e-4, abs_tol=1e-4):
            raise AssertionError(f"OpenAI SDK embedding was not L2-normalized: {norm}")
    return {
        "status": "ok",
        "vectors": len(response.data),
        "dimensions": 2,
        "model_retrieve": "ok",
    }


def validate_embedding_and_rerank(
    base_url: str,
    model_id: str,
    headers: dict[str, str],
    *,
    require_supported: bool,
) -> tuple[dict, dict]:
    embeddings = request_json(
        f"{base_url}/v1/embeddings",
        {
            "model": model_id,
            "input": ["local AI runtime", "bounded retrieval"],
            "encoding_format": "float",
            "dimensions": 2,
        },
        timeout=120.0,
        allow_http_error=True,
        headers=headers,
    )
    if embeddings.get("_http_status") == 501 and not require_supported:
        return embeddings, {}
    if embeddings.get("_http_status") is not None:
        raise AssertionError(f"embedding request failed: {embeddings}")
    if embeddings.get("object") != "list" or embeddings.get("model") != model_id:
        raise AssertionError(f"unexpected embeddings payload: {embeddings}")
    data = embeddings.get("data")
    if not isinstance(data, list) or len(data) != 2:
        raise AssertionError(f"embedding response omitted its batch: {embeddings}")
    for index, item in enumerate(data):
        vector = item.get("embedding") if isinstance(item, dict) else None
        if (
            not isinstance(item, dict)
            or item.get("index") != index
            or not isinstance(vector, list)
            or len(vector) != 2
        ):
            raise AssertionError(f"embedding item had an invalid shape: {item}")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in vector
        ):
            raise AssertionError(f"embedding item contained invalid values: {item}")
        norm = math.sqrt(math.fsum(float(value) ** 2 for value in vector))
        if not math.isclose(norm, 1.0, rel_tol=1e-4, abs_tol=1e-4):
            raise AssertionError(f"embedding item was not L2-normalized: norm={norm}")
    embedding_usage = embeddings.get("usage")
    if (
        not isinstance(embedding_usage, dict)
        or isinstance(embedding_usage.get("prompt_tokens"), bool)
        or not isinstance(embedding_usage.get("prompt_tokens"), int)
        or embedding_usage["prompt_tokens"] <= 0
        or embedding_usage.get("total_tokens") != embedding_usage["prompt_tokens"]
    ):
        raise AssertionError(f"embedding response had invalid usage: {embeddings}")

    documents = ["local AI runtime", "banana orchard", "local AI runtime"]
    rerank = request_json(
        f"{base_url}/v1/rerank",
        {
            "model": model_id,
            "query": "local AI runtime",
            "documents": documents,
            "top_n": 2,
            "return_documents": True,
        },
        timeout=120.0,
        allow_http_error=True,
        headers=headers,
    )
    if rerank.get("_http_status") is not None:
        raise AssertionError(f"rerank request failed: {rerank}")
    if (
        rerank.get("object") != "rerank"
        or rerank.get("model") != model_id
        or not isinstance(rerank.get("id"), str)
        or not rerank["id"].startswith("rerank-")
    ):
        raise AssertionError(f"unexpected rerank payload: {rerank}")
    results = rerank.get("results")
    if not isinstance(results, list) or len(results) != 2:
        raise AssertionError(f"rerank response violated top_n: {rerank}")
    if any(not isinstance(item, dict) for item in results):
        raise AssertionError(f"rerank results were not objects: {rerank}")
    if [item.get("index") for item in results] != [0, 2]:
        raise AssertionError(f"rerank ties were not stable by input index: {rerank}")
    previous_score = math.inf
    for item in results:
        if not isinstance(item, dict):
            raise AssertionError(f"rerank result was not an object: {item}")
        score = item.get("relevance_score")
        index = item.get("index")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < len(documents)
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or not -1.0 <= float(score) <= 1.0
            or float(score) > previous_score
            or item.get("document") != {"text": documents[index]}
        ):
            raise AssertionError(f"rerank item violated its response contract: {item}")
        previous_score = float(score)
    rerank_usage = rerank.get("usage")
    if (
        not isinstance(rerank_usage, dict)
        or isinstance(rerank_usage.get("prompt_tokens"), bool)
        or not isinstance(rerank_usage.get("prompt_tokens"), int)
        or rerank_usage["prompt_tokens"] <= 0
        or rerank_usage.get("total_tokens") != rerank_usage["prompt_tokens"]
    ):
        raise AssertionError(f"rerank response had invalid usage: {rerank}")
    return embeddings, rerank


def cosine_similarity(left: list[float], right: list[float]) -> float:
    return math.fsum(float(a) * float(b) for a, b in zip(left, right))


def validate_trained_embedding_quality(
    base_url: str,
    model_id: str,
    headers: dict[str, str],
    *,
    expected_dimensions: int,
    expected_context_size: int | None,
) -> dict:
    texts = [
        "A man is eating food.",
        "A person eats a meal.",
        "A spaceship lands on Mars.",
    ]
    started = time.perf_counter()
    response = request_json(
        f"{base_url}/v1/embeddings",
        {"model": model_id, "input": texts, "encoding_format": "float"},
        timeout=120.0,
        allow_http_error=True,
        headers=headers,
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    if response.get("_http_status") is not None:
        raise AssertionError(f"trained embedding request failed: {response}")
    data = response.get("data")
    if not isinstance(data, list) or len(data) != len(texts):
        raise AssertionError(f"trained embedding response omitted vectors: {response}")
    vectors: list[list[float]] = []
    for index, item in enumerate(data):
        vector = item.get("embedding") if isinstance(item, dict) else None
        if (
            not isinstance(vector, list)
            or len(vector) != expected_dimensions
            or item.get("index") != index
        ):
            raise AssertionError(
                f"expected native {expected_dimensions}-dimensional embedding at index {index}: {item}"
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in vector
        ):
            raise AssertionError(f"trained embedding contained invalid values: {item}")
        normalized = [float(value) for value in vector]
        norm = math.sqrt(math.fsum(value * value for value in normalized))
        if not math.isclose(norm, 1.0, rel_tol=1e-4, abs_tol=1e-4):
            raise AssertionError(f"trained embedding was not L2-normalized: norm={norm}")
        vectors.append(normalized)

    paraphrase_score = cosine_similarity(vectors[0], vectors[1])
    unrelated_score = cosine_similarity(vectors[0], vectors[2])
    similarity_margin = paraphrase_score - unrelated_score
    if similarity_margin < 0.50:
        raise AssertionError(
            "trained semantic quality gate failed: "
            f"paraphrase={paraphrase_score:.6f}, unrelated={unrelated_score:.6f}, "
            f"margin={similarity_margin:.6f}"
        )

    rerank = request_json(
        f"{base_url}/v1/rerank",
        {
            "model": model_id,
            "query": texts[0],
            "documents": [texts[1], texts[2]],
            "top_n": 2,
            "return_documents": True,
        },
        timeout=120.0,
        allow_http_error=True,
        headers=headers,
    )
    if rerank.get("_http_status") is not None:
        raise AssertionError(f"trained rerank request failed: {rerank}")
    results = rerank.get("results")
    if (
        not isinstance(results, list)
        or len(results) != 2
        or results[0].get("index") != 0
        or results[1].get("index") != 1
    ):
        raise AssertionError(f"trained rerank ordering was semantically wrong: {rerank}")
    rerank_margin = float(results[0]["relevance_score"]) - float(
        results[1]["relevance_score"]
    )
    if rerank_margin < 0.50:
        raise AssertionError(f"trained rerank margin was too small: {rerank}")

    generation_requests = [
        (
            "/v1/chat/completions",
            {"model": model_id, "messages": [{"role": "user", "content": "hello"}]},
        ),
        ("/v1/completions", {"model": model_id, "prompt": "hello"}),
        ("/v1/responses", {"model": model_id, "input": "hello"}),
        (
            "/api/chat",
            {
                "model": model_id,
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
            },
        ),
        ("/api/generate", {"model": model_id, "prompt": "hello", "stream": False}),
    ]
    for route, payload in generation_requests:
        rejected = request_json(
            f"{base_url}{route}",
            payload,
            timeout=30.0,
            allow_http_error=True,
            headers=headers,
        )
        if rejected.get("_http_status") != 422:
            raise AssertionError(
                f"embedding-only model did not reject generation on {route}: {rejected}"
            )
        if route.startswith("/v1/"):
            error = rejected.get("error")
            if not isinstance(error, dict) or error.get("type") != "unsupported_operation":
                raise AssertionError(
                    f"OpenAI generation rejection had the wrong error contract on {route}: {rejected}"
                )
        elif not isinstance(rejected.get("error"), str):
            raise AssertionError(
                f"Ollama generation rejection had the wrong error contract on {route}: {rejected}"
            )

    ready = request_json(f"{base_url}/ready", timeout=10.0)
    if expected_context_size is not None and ready.get("context_window") != expected_context_size:
        raise AssertionError(
            "runtime did not enforce the trained encoder context limit: "
            f"expected {expected_context_size}, got {ready.get('context_window')}"
        )

    context_checks = 0
    if expected_context_size is not None:
        long_input = ("hello " * (expected_context_size + 16)).strip()
        rejected_openai = request_json(
            f"{base_url}/v1/embeddings",
            {"model": model_id, "input": long_input},
            timeout=30.0,
            allow_http_error=True,
            headers=headers,
        )
        if rejected_openai.get("_http_status") != 400:
            raise AssertionError(
                f"OpenAI embeddings did not enforce the model context limit: {rejected_openai}"
            )
        context_checks += 1
        rejected_ollama = request_json(
            f"{base_url}/api/embed",
            {"model": model_id, "input": long_input, "truncate": False},
            timeout=30.0,
            allow_http_error=True,
            headers=headers,
        )
        if rejected_ollama.get("_http_status") != 400:
            raise AssertionError(
                f"Ollama embed did not enforce truncate=false: {rejected_ollama}"
            )
        context_checks += 1
        truncated_ollama = request_json(
            f"{base_url}/api/embed",
            {"model": model_id, "input": long_input, "truncate": True},
            timeout=120.0,
            allow_http_error=True,
            headers=headers,
        )
        truncated_vectors = truncated_ollama.get("embeddings")
        if (
            truncated_ollama.get("_http_status") is not None
            or not isinstance(truncated_vectors, list)
            or len(truncated_vectors) != 1
            or not isinstance(truncated_vectors[0], list)
            or len(truncated_vectors[0]) != expected_dimensions
        ):
            raise AssertionError(
                f"Ollama embed could not safely truncate to the model context: {truncated_ollama}"
            )
        context_checks += 1

    return {
        "dimensions": expected_dimensions,
        "context_window": ready.get("context_window"),
        "paraphrase_cosine": round(paraphrase_score, 6),
        "unrelated_cosine": round(unrelated_score, 6),
        "similarity_margin": round(similarity_margin, 6),
        "rerank_margin": round(rerank_margin, 6),
        "batch_latency_ms": round(latency_ms, 3),
        "generation_routes_rejected": len(generation_requests),
        "context_limit_checks": context_checks,
    }


def wait_healthy(base_url: str, proc: subprocess.Popen[str], timeout: float) -> None:
    deadline = time.time() + timeout
    last_error = ""
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"bloom_server exited early with code {proc.returncode}")
        try:
            health = request_json(f"{base_url}/health", timeout=2.0)
            if health.get("status") == "ok":
                return
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            last_error = str(exc)
        time.sleep(0.25)
    raise TimeoutError(f"server did not become healthy within {timeout}s: {last_error}")


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
            if ready.get("load_error") is not None and ready.get("loading") is False:
                raise RuntimeError("bloom_server reported a terminal model load failure")
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
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--embedding-only",
        action="store_true",
        default=os.environ.get("BLOOM_EMBEDDING_ONLY", "").lower()
        in {"1", "true", "yes"},
        help="Validate embeddings and reranking without invoking text generation.",
    )
    parser.add_argument(
        "--embedding-quality-dimensions",
        type=int,
        help="In embedding mode, require this native vector width and semantic quality evidence.",
    )
    parser.add_argument(
        "--expected-context-size",
        type=int,
        help="Require /ready to report this model-bounded context window.",
    )
    mode.add_argument(
        "--tool-only",
        action="store_true",
        default=os.environ.get("BLOOM_TOOL_ONLY", "").lower()
        in {"1", "true", "yes"},
        help="Require successful Chat and Responses function-call lifecycles.",
    )
    mode.add_argument(
        "--structured-only",
        action="store_true",
        default=os.environ.get("BLOOM_STRUCTURED_ONLY", "").lower()
        in {"1", "true", "yes"},
        help="Require successful Chat and Responses JSON object/schema lifecycles.",
    )
    args = parser.parse_args()
    return args


def main() -> int:
    args = parse_args()
    if args.embedding_quality_dimensions is not None and not args.embedding_only:
        print(
            "FAIL: --embedding-quality-dimensions requires --embedding-only",
            file=sys.stderr,
        )
        return 1
    if args.embedding_quality_dimensions is not None and args.embedding_quality_dimensions <= 0:
        print("FAIL: --embedding-quality-dimensions must be positive", file=sys.stderr)
        return 1
    if (args.semantic_prompt is None) != (args.expected_output is None):
        print(
            "FAIL: --semantic-prompt and --expected-output must be provided together",
            file=sys.stderr,
        )
        return 1
    model = Path(args.model)
    has_model = model.exists()
    if not has_model and args.require_model:
        print(f"FAIL: model not found at {model} (required by --require-model)", file=sys.stderr)
        return 1

    if args.build or not Path(args.server_bin).exists():
        subprocess.run(["cargo", "build", "--release", "--bin", "bloom_server"], check=True)

    port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    models_dir = tempfile.TemporaryDirectory(prefix="bloom-openai-smoke-")
    cmd = [
        args.server_bin,
        "--models-dir",
        models_dir.name,
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
        "--cors-allow-origin",
        "https://bloom-smoke.invalid",
    ]
    if has_model:
        cmd.extend(["--model", str(model)])
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
        if has_model:
            wait_ready(base_url, proc, args.startup_timeout)
        else:
            wait_healthy(base_url, proc, args.startup_timeout)

        request_correlation = validate_http_request_correlation(base_url)
        readiness_contract = validate_readiness_contract(
            base_url,
            expected_ready=has_model,
            expected_tasks=(
                ["embedding", "rerank"]
                if args.embedding_only
                else (["generation"] if has_model else [])
            ),
        )
        auth_headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
        auth_status = "disabled"
        if args.api_key:
            auth_http_status, auth_response_headers, _auth_body = request_status_headers(
                f"{base_url}/v1/models"
            )
            if (
                auth_http_status != 401
                or auth_response_headers.get("www-authenticate")
                != 'Bearer realm="Bloom"'
            ):
                raise AssertionError(
                    "unauthenticated OpenAI route did not publish the Bearer challenge: "
                    f"{auth_http_status}, {auth_response_headers}"
                )
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

        framework_rejections = validate_framework_rejections(base_url, auth_headers)

        models = request_json(f"{base_url}/v1/models", headers=auth_headers)
        model_rows = models.get("data")
        if models.get("object") != "list" or not isinstance(model_rows, list):
            raise AssertionError(f"unexpected /v1/models response: {models}")

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

        if not has_model:
            if model_rows:
                raise AssertionError(
                    f"isolated empty catalog unexpectedly exposed a model: {models}"
                )
            sdk = request_openai_sdk_model_free(
                base_url,
                args.api_key or "bloom-smoke",
                expect_authentication=bool(args.api_key),
            )
            if args.require_openai_sdk and sdk.get("status") != "ok":
                raise AssertionError(f"OpenAI SDK smoke failed or skipped: {sdk}")
            print(
                json.dumps(
                    {
                        "status": "ok",
                        "base_url": base_url,
                        "model_count": 0,
                        "backend_count": len(backend_rows),
                        "auth": auth_status,
                        "request_correlation": request_correlation,
                        "framework_rejections": framework_rejections,
                        "readiness_contract": readiness_contract,
                        "mode": "model_free",
                        "openai_sdk": sdk,
                    },
                    indent=2,
                )
            )
            print(
                f"SKIP: model not found at {model}; discovery and admission passed"
            )
            return 0

        if not model_rows:
            raise AssertionError(f"/v1/models omitted the active startup model: {models}")
        model_id = model_rows[0].get("id", "bloom-local")

        missing_model_id = "__bloom_smoke_model_not_loaded__"
        if missing_model_id == model_id:
            missing_model_id = "__bloom_smoke_other_model_not_loaded__"
        mismatched_model = request_json(
            f"{base_url}/v1/chat/completions",
            {
                "model": missing_model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "This must not run."}
                        ],
                    }
                ],
                "max_completion_tokens": 1,
            },
            allow_http_error=True,
            headers=auth_headers,
        )
        if mismatched_model.get("_http_status") != 404:
            raise AssertionError(
                f"expected an unloaded model selector to return 404: {mismatched_model}"
            )
        if mismatched_model.get("error", {}).get("type") != "model_not_found":
            raise AssertionError(f"unexpected model mismatch error: {mismatched_model}")

        mismatched_response_model = request_json(
            f"{base_url}/v1/responses",
            {
                "model": missing_model_id,
                "instructions": "Be concise.",
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "This must not run."}
                        ],
                    }
                ],
                "max_output_tokens": 1,
                "stream": True,
            },
            allow_http_error=True,
            headers=auth_headers,
        )
        if mismatched_response_model.get("_http_status") != 404:
            raise AssertionError(
                "expected an unloaded Responses model selector to return 404: "
                f"{mismatched_response_model}"
            )
        if mismatched_response_model.get("error", {}).get("type") != "model_not_found":
            raise AssertionError(
                f"unexpected Responses model mismatch error: {mismatched_response_model}"
            )

        stateful_response = request_json(
            f"{base_url}/v1/responses",
            {
                "model": model_id,
                "input": "This must not run.",
                "previous_response_id": "resp-unsupported",
            },
            allow_http_error=True,
            headers=auth_headers,
        )
        if stateful_response.get("_http_status") != 404:
            raise AssertionError(
                "expected a missing previous Responses ID to return 404: "
                f"{stateful_response}"
            )
        if stateful_response.get("error", {}).get("type") != "invalid_request_error":
            raise AssertionError(
                f"unexpected stateful Responses error: {stateful_response}"
            )

        unsupported_tools = request_json(
            f"{base_url}/v1/chat/completions",
            {
                "model": model_id,
                "messages": [{"role": "user", "content": "This must not run."}],
                "tools": [
                    {
                        "type": "custom",
                        "custom": {"name": "lookup"},
                    }
                ],
            },
            allow_http_error=True,
            headers=auth_headers,
        )
        if unsupported_tools.get("_http_status") != 400:
            raise AssertionError(
                "expected unsupported custom-tool semantics to return 400: "
                f"{unsupported_tools}"
            )
        unsupported_error = unsupported_tools.get("error", {})
        if unsupported_error.get("type") != "invalid_request_error" or "custom" not in str(
            unsupported_error.get("message", "")
        ):
            raise AssertionError(f"unexpected unsupported-tools error: {unsupported_tools}")

        if args.tool_only:
            tool_lifecycle = validate_function_tool_round_trips(
                base_url,
                model_id,
                auth_headers,
                max_tokens=max(args.max_tokens, 64),
            )
            sdk = request_openai_sdk_response_stream(
                base_url,
                model_id,
                max(args.max_tokens, 64),
                args.api_key or "bloom-smoke",
            )
            if args.require_openai_sdk and sdk.get("status") != "ok":
                raise AssertionError(f"OpenAI SDK smoke failed or skipped: {sdk}")
            if sdk.get("status") == "ok" and (
                sdk.get("function_calling") != "ok"
                or sdk.get("responses_function_calling") != "ok"
            ):
                raise AssertionError(
                    f"OpenAI SDK did not complete native function calls: {sdk}"
                )
            print(
                json.dumps(
                    {
                        "status": "ok",
                        "base_url": base_url,
                        "model_count": len(models["data"]),
                        "backend_count": len(backend_rows),
                        "auth": auth_status,
                        "request_correlation": request_correlation,
                        "framework_rejections": framework_rejections,
                        "readiness_contract": readiness_contract,
                        "mode": "tool",
                        "function_lifecycle": tool_lifecycle,
                        "openai_sdk": sdk,
                    },
                    indent=2,
                )
            )
            return 0

        if args.structured_only:
            structured_lifecycle = validate_structured_output_round_trips(
                base_url,
                model_id,
                auth_headers,
                max_tokens=max(args.max_tokens, 8),
            )
            sdk = request_openai_sdk_response_stream(
                base_url,
                model_id,
                max(args.max_tokens, 8),
                args.api_key or "bloom-smoke",
            )
            if args.require_openai_sdk and sdk.get("status") != "ok":
                raise AssertionError(f"OpenAI SDK smoke failed or skipped: {sdk}")
            if sdk.get("status") == "ok" and (
                sdk.get("chat_structured_output") != "ok"
                or sdk.get("structured_output") != "ok"
            ):
                raise AssertionError(
                    f"OpenAI SDK did not complete structured outputs: {sdk}"
                )
            print(
                json.dumps(
                    {
                        "status": "ok",
                        "base_url": base_url,
                        "model_count": len(models["data"]),
                        "backend_count": len(backend_rows),
                        "auth": auth_status,
                        "request_correlation": request_correlation,
                        "framework_rejections": framework_rejections,
                        "readiness_contract": readiness_contract,
                        "mode": "structured",
                        "structured_lifecycle": structured_lifecycle,
                        "openai_sdk": sdk,
                    },
                    indent=2,
                )
            )
            return 0

        if args.embedding_only:
            embeddings, rerank = validate_embedding_and_rerank(
                base_url,
                model_id,
                auth_headers,
                require_supported=True,
            )
            quality = (
                validate_trained_embedding_quality(
                    base_url,
                    model_id,
                    auth_headers,
                    expected_dimensions=args.embedding_quality_dimensions,
                    expected_context_size=args.expected_context_size,
                )
                if args.embedding_quality_dimensions is not None
                else None
            )
            sdk = request_openai_sdk_embeddings(
                base_url,
                model_id,
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
                        "auth": auth_status,
                        "request_correlation": request_correlation,
                        "framework_rejections": framework_rejections,
                        "readiness_contract": readiness_contract,
                        "mode": "embedding",
                        "embedding_vectors": len(embeddings["data"]),
                        "rerank_results": len(rerank["results"]),
                        "trained_quality": quality,
                        "openai_sdk": sdk,
                    },
                    indent=2,
                )
            )
            return 0

        chat_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": args.semantic_prompt
                        or "Say hello in one short sentence.",
                    }
                ],
            }
        ]
        if args.semantic_system:
            chat_messages.insert(
                0,
                {"role": "system", "content": args.semantic_system},
            )
        payload = {
            "model": "default",
            "messages": chat_messages,
            "max_completion_tokens": args.max_tokens,
            "temperature": 0.0,
            "stop": ["__BLOOM_STOP_NEVER__"],
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
        message_content = choices[0]["message"].get("content")
        if not isinstance(message_content, str) or not message_content:
            raise AssertionError(f"chat completion returned no text: {completion}")
        if (
            args.expected_output is not None
            and message_content.strip() != args.expected_output
        ):
            raise AssertionError(
                "buffered chat returned the wrong semantic output: "
                f"expected {args.expected_output!r}, got {message_content!r}"
            )
        if completion.get("model") != model_id:
            raise AssertionError(f"chat completion reported the wrong model: {completion}")

        structured_payload = {
            "model": model_id,
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
        elif structured.get("model") != model_id:
            raise AssertionError(f"structured response reported the wrong model: {structured}")

        embeddings = request_json(
            f"{base_url}/v1/embeddings",
            {"model": model_id, "input": "hello", "encoding_format": "float"},
            timeout=60.0,
            allow_http_error=True,
            headers=auth_headers,
        )
        if embeddings.get("_http_status") not in (None, 501):
            raise AssertionError(f"unexpected embeddings response: {embeddings}")
        if embeddings.get("_http_status") is None and embeddings.get("object") != "list":
            raise AssertionError(f"unexpected embeddings payload: {embeddings}")
        if embeddings.get("_http_status") is None and embeddings.get("model") != model_id:
            raise AssertionError(f"embeddings response reported the wrong model: {embeddings}")
        if embeddings.get("_http_status") == 501:
            err = embeddings.get("error", {})
            if err.get("type") != "unsupported_operation":
                raise AssertionError(f"unexpected embeddings error: {embeddings}")

        rerank = request_json(
            f"{base_url}/v1/rerank",
            {
                "model": model_id,
                "query": "hello",
                "documents": ["hello world"],
                "top_n": 1,
            },
            timeout=60.0,
            allow_http_error=True,
            headers=auth_headers,
        )
        if rerank.get("_http_status") not in (None, 501):
            raise AssertionError(f"unexpected rerank response: {rerank}")
        if rerank.get("_http_status") is None and rerank.get("object") != "rerank":
            raise AssertionError(f"unexpected rerank payload: {rerank}")
        if rerank.get("_http_status") is None and rerank.get("model") != model_id:
            raise AssertionError(f"rerank response reported the wrong model: {rerank}")
        if rerank.get("_http_status") == 501:
            err = rerank.get("error", {})
            if err.get("type") != "unsupported_operation":
                raise AssertionError(f"unexpected rerank error: {rerank}")

        payload["model"] = model_id
        payload["stream"] = True
        events = request_sse(
            f"{base_url}/v1/chat/completions",
            payload,
            timeout=60.0,
            headers=auth_headers,
        )
        if "[DONE]" not in events:
            raise AssertionError("streaming response did not include [DONE]")
        streamed_text = ""
        for event in events:
            if event == "[DONE]":
                continue
            decoded_event = json.loads(event)
            choices = decoded_event.get("choices") or []
            if choices:
                content = choices[0].get("delta", {}).get("content")
                if isinstance(content, str):
                    streamed_text += content
        if not streamed_text:
            raise AssertionError("streaming response completed without text deltas")
        if (
            args.expected_output is not None
            and streamed_text.strip() != args.expected_output
        ):
            raise AssertionError(
                "streamed chat returned the wrong semantic output: "
                f"expected {args.expected_output!r}, got {streamed_text!r}"
            )

        sdk = request_openai_sdk_response_stream(
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
                    "semantic_output": "validated"
                    if args.expected_output is not None
                    else "not_requested",
                    "auth": auth_status,
                    "request_correlation": request_correlation,
                    "framework_rejections": framework_rejections,
                    "readiness_contract": readiness_contract,
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
        failed = sys.exc_info()[0] is not None
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
        if failed and proc.stderr:
            stderr = proc.stderr.read()
            if stderr:
                print("bloom_server stderr tail:", file=sys.stderr)
                print(stderr[-4000:], file=sys.stderr)
        models_dir.cleanup()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)

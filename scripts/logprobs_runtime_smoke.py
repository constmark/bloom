#!/usr/bin/env python3
"""Validate real CPU Candle logprobs offline with a supplied local model.

No build, package installation or download is performed. Each request/response,
server log and binary/model digest is retained in a new output directory.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import signal
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

from metal_runtime_smoke import sha256
from openai_compat_smoke import find_free_port, wait_ready
from run_with_timeout import terminate

LIMIT = 16 * 1024 * 1024


def check_score(value):
    assert isinstance(value, (int, float)) and not isinstance(value, bool)
    assert math.isfinite(value) and value <= 0, value


def validate_choice(choice, *, chat, top):
    text = choice["message"]["content"] if chat else choice["text"]
    assert isinstance(text, str), choice
    assert choice["finish_reason"] in ("stop", "length"), choice
    probabilities = choice["logprobs"]
    if chat:
        records = probabilities["content"]
        raw = bytearray()
        for record in records:
            check_score(record["logprob"])
            value = bytes(record["bytes"])
            assert value.decode("utf-8", errors="replace") == record["token"]
            raw.extend(value)
            alternatives = record["top_logprobs"]
            assert len(alternatives) <= top
            scores = []
            for alternative in alternatives:
                check_score(alternative["logprob"])
                assert bytes(alternative["bytes"]).decode("utf-8", errors="replace") == alternative["token"]
                scores.append(alternative["logprob"])
            assert scores == sorted(scores, reverse=True)
            if top:
                assert alternatives and math.isclose(scores[0], record["logprob"], abs_tol=1e-6)
        assert raw.decode("utf-8") == text, (raw, text)
        count = len(records)
    else:
        count = len(probabilities["tokens"])
        assert all(len(probabilities[field]) == count for field in ("token_logprobs", "top_logprobs", "text_offset"))
        offsets = probabilities["text_offset"]
        assert not offsets or offsets[0] == 0
        assert offsets == sorted(offsets)
        assert all(isinstance(offset, int) and 0 <= offset <= len(text) for offset in offsets)
        for token, score, alternatives in zip(probabilities["tokens"], probabilities["token_logprobs"], probabilities["top_logprobs"]):
            check_score(score)
            # Legacy maps include the chosen token even when top=0.
            assert 1 <= len(alternatives) <= top + 1 and alternatives[token] == score
            for value in alternatives.values():
                check_score(value)
        # Legacy has no raw bytes field. Exact offsets are checkable when no
        # token has a lossy split-byte display; fragments share an offset.
        tokens = probabilities["tokens"]
        if all("\ufffd" not in token for token in tokens):
            assert "".join(tokens) == text
            assert offsets == [len("".join(tokens[:index])) for index in range(count)]
    assert count > 0 or (text == "" and choice["finish_reason"] == "stop")
    return count


def collect_stream(events, *, chat, n):
    choices = [{"index": index, "finish_reason": None,
                **({"message": {"content": ""}, "logprobs": {"content": []}}
                   if chat else {"text": "", "logprobs": {key: [] for key in ("tokens", "token_logprobs", "top_logprobs", "text_offset")}})}
               for index in range(n)]
    usage = None
    finished = set()
    for event in events:
        assert "error" not in event, event
        if event.get("usage") is not None:
            assert usage is None, "duplicate usage chunk"
            usage = event["usage"]
        for value in event.get("choices", []):
            index = value["index"]
            assert 0 <= index < n and index not in finished, value
            target = choices[index]
            if chat:
                target["message"]["content"] += value.get("delta", {}).get("content") or ""
            else:
                target["text"] += value.get("text") or ""
            if value.get("logprobs"):
                for field in target["logprobs"]:
                    target["logprobs"][field].extend(value["logprobs"].get(field) or [])
            if value.get("finish_reason"):
                target["finish_reason"] = value["finish_reason"]
                finished.add(index)
    assert finished == set(range(n)) and (not chat or usage is not None)
    return {"choices": choices, "usage": usage}


def validate_response(result, *, chat, top, n, streamed=False):
    assert [choice["index"] for choice in result["choices"]] == list(range(n))
    count = sum(validate_choice(choice, chat=chat, top=top) for choice in result["choices"])
    if result.get("usage") is None:
        # Legacy streaming does not admit stream_options or publish usage.
        assert streamed and not chat
        return
    # Engine usage counts sampled EOS while metadata contains visible tokens.
    # Each stopped candidate can therefore contribute one additional token.
    eos_allowance = sum(choice["finish_reason"] == "stop" for choice in result["choices"])
    completion_tokens = result["usage"]["completion_tokens"]
    assert count <= completion_tokens <= count + eos_allowance
    if count == 0:
        assert completion_tokens == n, "first EOS must count once per candidate"
    assert result["usage"]["prompt_tokens"] > 0
    assert result["usage"]["total_tokens"] == completion_tokens + result["usage"]["prompt_tokens"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-bin", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--choices", type=int, nargs="+", choices=range(1, 9), default=[1, 2])
    parser.add_argument("--top-counts", type=int, nargs="+", choices=range(6), default=[0, 2],
                        help="Counts shared by Chat and Legacy; Legacy admits at most five.")
    parser.add_argument("--logit-bias", type=json.loads, default=None,
                        help='Optional JSON object, e.g. {"108386":100} for cached Qwen2 Chinese token.')
    parser.add_argument("--expected-text", help="Optional exact text per candidate.")
    parser.add_argument("--require-openai-sdk", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.max_tokens <= 4096:
        parser.error("--max-tokens must be between 1 and 4096")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    server, model = args.server_bin.resolve(), args.model.resolve()
    report = {"schema_version": 1, "status": "running", "backend": "candle", "device": "cpu", "cases": []}
    process = None

    def save():
        temporary = output / "report.json.tmp"
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(output / "report.json")

    def record(name, payload, result):
        (output / f"{name}.json").write_text(json.dumps({"request": payload, "response": result}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    try:
        assert server.is_file() and os.access(server, os.X_OK), server
        assert model.exists(), model
        report["server"] = {"path": str(server), "sha256": sha256(server)}
        artifacts = [model] if model.is_file() else sorted(path for path in model.rglob("*") if path.is_file())
        report["model"] = {"path": str(model), "artifacts": {str(path.relative_to(model) if model.is_dir() else path.name): sha256(path) for path in artifacts}}
        env = {key: value for key, value in os.environ.items() if not key.startswith("BLOOM_")}
        key = "bloom-local-logprobs-acceptance"
        env.update(BLOOM_API_KEY=key, BLOOM_CONFIG=str(output / "config.json"),
                   BLOOM_CONFIG_HOME=str(output / "config"), TOKENIZERS_PARALLELISM="false")
        base = f"http://127.0.0.1:{find_free_port()}"
        command = [str(server), "--host", "127.0.0.1", "--port", base.rsplit(":", 1)[1],
                   "--backend", "candle", "--device", "cpu", "--model", str(model),
                   "--models-dir", str(output / "catalog"), "--timeout", "120"]
        report["command"] = command
        save()
        with (output / "server.stdout.log").open("wb") as stdout, (output / "server.stderr.log").open("wb") as stderr:
            process = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
        wait_ready(base, process, 180)

        def raw(name, path, payload):
            request = urllib.request.Request(base + path, data=json.dumps(payload).encode(),
                        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    body = response.read(LIMIT + 1)
                    assert len(body) <= LIMIT
            except urllib.error.HTTPError as error:
                body = error.read(LIMIT + 1).decode(errors="replace")
                record(name, payload, {"status": error.code, "body": body})
                raise AssertionError(f"{name}: HTTP {error.code}: {body}") from error
            if payload.get("stream"):
                lines = [line[6:] for line in body.decode().splitlines() if line.startswith("data: ")]
                assert lines and lines[-1] == "[DONE]" and lines.count("[DONE]") == 1
                result = [json.loads(line) for line in lines[:-1]]
            else:
                result = json.loads(body)
            record(name, payload, result)
            return result

        for chat in (True, False):
            route = "chat" if chat else "completion"
            path = "/v1/chat/completions" if chat else "/v1/completions"
            for n in args.choices:
                for top in args.top_counts:
                    payload = {"model": "default", "max_tokens": args.max_tokens, "temperature": 0,
                               "seed": 42, "n": n, **({"messages": [{"role": "user", "content": args.prompt}],
                               "logprobs": True, "top_logprobs": top} if chat else {"prompt": args.prompt, "logprobs": top})}
                    if args.logit_bias is not None:
                        payload["logit_bias"] = args.logit_bias
                    name = f"{route}-n{n}-top{top}"
                    buffered = raw(name + "-buffered", path, payload)
                    stream_payload = {**payload, "stream": True}
                    if chat:
                        stream_payload["stream_options"] = {"include_usage": True}
                    streamed = collect_stream(raw(name + "-stream", path, stream_payload), chat=chat, n=n)
                    for index, result in enumerate((buffered, streamed)):
                        validate_response(result, chat=chat, top=top, n=n, streamed=index == 1)
                        if args.expected_text is not None:
                            assert all((choice["message"]["content"] if chat else choice["text"]) == args.expected_text for choice in result["choices"])
                    for first, second in zip(buffered["choices"], streamed["choices"]):
                        for field in ("logprobs", "finish_reason", "message" if chat else "text"):
                            if field == "message":
                                assert first[field]["content"] == second[field]["content"]
                            elif field == "logprobs" and chat:
                                assert first[field]["content"] == second[field]["content"]
                            else:
                                assert first[field] == second[field], (name, field)
                    if chat:
                        assert buffered["usage"] == streamed["usage"]
                    report["cases"].append({"name": name, "status": "passed", "buffered_stream_equal": True})
                    save()
            if args.require_openai_sdk:
                from openai import OpenAI
                report["openai_sdk"] = importlib.metadata.version("openai")
                with OpenAI(base_url=base + "/v1", api_key=key, timeout=120, max_retries=0) as client:
                    create = client.chat.completions.create if chat else client.completions.create
                    result = create(**payload).model_dump()
                    record(route + "-sdk-buffered", payload, result)
                    validate_response(result, chat=chat, top=top, n=n)
                    sdk_stream_args = {"stream": True}
                    if chat:
                        sdk_stream_args["stream_options"] = {"include_usage": True}
                    with create(**payload, **sdk_stream_args) as stream:
                        events = [event.model_dump() for event in stream]
                    record(route + "-sdk-stream", payload, events)
                    validate_response(collect_stream(events, chat=chat, n=n), chat=chat, top=top, n=n, streamed=True)
                report["cases"].append({"name": route + "-sdk", "status": "passed"})
                save()
        assert sha256(server) == report["server"]["sha256"], "server binary changed during validation"
        report["status"] = "passed"
        return 0
    except (Exception, KeyboardInterrupt) as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        print(report["error"], file=sys.stderr)
        return 1
    finally:
        if process is not None:
            terminate(process)
        save()
        print(output / "report.json")


if __name__ == "__main__":
    def interrupted(_signum, _frame):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, interrupted)
    raise SystemExit(main())

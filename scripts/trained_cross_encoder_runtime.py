#!/usr/bin/env python3
"""Pinned trained BERT cross-encoder quality gate; Python standard library only.

Scores are raw classifier logits, not calibrated probabilities. This small,
authored acceptance set tests executable ranking behavior, not MS MARCO/MTEB
benchmark parity. All model bytes and the Apache-2.0 model card are hash pinned.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import socket
import stat
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

from readiness_contract import validate_readiness_document


WORKSPACE = Path(__file__).resolve().parents[1]
REPOSITORY = "cross-encoder/ms-marco-MiniLM-L6-v2"
REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
FILES = {
    "model.safetensors": (90870598, "821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae"),
    "config.json": (794, "380e02c93f431831be65d99a4e7e5f67c133985bf2e77d9d4eba46847190bacc"),
    "tokenizer.json": (711396, "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66"),
    "tokenizer_config.json": (1330, "a5c2e5a7b1a29a0702cd28c08a399b5ecc110c263009d17f7e3b415f25905fd8"),
    "special_tokens_map.json": (132, "3c3507f36dff57bce437223db3b3081d1e2b52ec3e56ee55438193ecb2c94dd6"),
    "vocab.txt": (231508, "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3"),
    "README.md": (3673, "7c0ec39941d0d1d766ccde671f592971d82fe45b7eea550a80d1ea864ebc3baa"),
}
MINIMUM_MARGIN = 3.0
SCORE_TOLERANCE = 2e-4
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
CASES = (
    {
        "id": "geography",
        "query": "What is the capital of France?",
        "documents": [
            "Bananas are a tropical fruit often used in smoothies.",
            "Paris is the capital and largest city of France.",
            "Berlin is the capital and largest city of Germany.",
        ],
        "relevant_index": 1,
    },
    {
        "id": "programming",
        "query": "How can a Python list be sorted in descending order?",
        "documents": [
            "Call sorted(values, reverse=True) to create a Python list ordered from largest to smallest.",
            "Use len(values) to count the elements in a Python list.",
            "A python is a snake that can swallow its prey whole.",
        ],
        "relevant_index": 0,
    },
    {
        "id": "biology",
        "query": "Why do leaves appear green?",
        "documents": [
            "Plant roots take up water from the soil.",
            "A bicycle has two wheels and handlebars.",
            "Leaves look green because chlorophyll absorbs red and blue light while reflecting green wavelengths.",
        ],
        "relevant_index": 2,
    },
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file(path: Path, size: int, digest: str) -> None:
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError(f"pinned artifact is not a regular, non-symlink file: {path.name}")
    if path.stat().st_size != size or sha256(path) != digest:
        raise ValueError(f"pinned artifact size/SHA-256 mismatch: {path.name}")


def acquire_model(root: Path) -> dict:
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        raise ValueError("model cache must be a non-symlink directory")
    root.mkdir(parents=True, exist_ok=True)
    unknown = {path.name for path in root.iterdir()} - FILES.keys()
    if unknown:
        raise ValueError("model cache contains unpinned files; use a clean profile directory")
    for name, (size, digest) in FILES.items():
        path = root / name
        if not path.exists() and not path.is_symlink():
            print(f"Downloading pinned {name} ({size} bytes)...", file=sys.stderr)
            fd, temporary = tempfile.mkstemp(prefix=".cross-encoder-download-", dir=root)
            stage = Path(temporary)
            try:
                with os.fdopen(fd, "wb") as output, urllib.request.urlopen(
                    f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}", timeout=180
                ) as response:
                    if not response.geturl().startswith("https://"):
                        raise ValueError("model download redirected away from HTTPS")
                    transferred = 0
                    while block := response.read(1024 * 1024):
                        transferred += len(block)
                        if transferred > size:
                            raise ValueError(f"download exceeded pinned size: {name}")
                        output.write(block)
                verify_file(stage, size, digest)
                # Link publishes atomically without overwriting an existing file.
                os.link(stage, path)
            finally:
                stage.unlink(missing_ok=True)
        verify_file(path, size, digest)
    card = (root / "README.md").read_text(encoding="utf-8")
    if "license: apache-2.0" not in card.split("---", 2)[1]:
        raise ValueError("pinned model card lacks the Apache-2.0 declaration")
    return {name: {"bytes": size, "sha256": digest} for name, (size, digest) in FILES.items()}


def request(base: str, route: str, payload=None, *, key: str = "", timeout: float = 120) -> tuple[int, dict]:
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    data = None if payload is None else json.dumps(payload, allow_nan=False).encode()
    req = urllib.request.Request(base + route, data=data, headers=headers)
    try:
        response = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        status_code = response.status
        raw = response.read(MAX_RESPONSE_BYTES + 1)
    if len(raw) > MAX_RESPONSE_BYTES:
        raise AssertionError("HTTP response exceeded the gate byte limit")
    result = json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    if not isinstance(result, dict):
        raise AssertionError("HTTP response must be a JSON object")
    return status_code, result


def validate_ranking(payload: dict, model: str, documents: list[str], top_n: int) -> list[float | None]:
    if type(top_n) is not int or not 1 <= top_n <= len(documents):
        raise AssertionError("top_n must select at least one existing document")
    if payload.get("object") != "rerank" or payload.get("model") != model:
        raise AssertionError("rerank response changed object or model identity")
    if not isinstance(payload.get("id"), str) or not payload["id"].startswith("rerank-"):
        raise AssertionError("rerank response omitted its request identity")
    rows = payload.get("results")
    if not isinstance(rows, list) or len(rows) != top_n:
        raise AssertionError("rerank response did not preserve top_n")
    scores: list[float | None] = [None] * len(documents)
    order = []
    for row in rows:
        if not isinstance(row, dict):
            raise AssertionError("rerank result is not an object")
        index, score = row.get("index"), row.get("relevance_score")
        if type(index) is not int or not 0 <= index < len(documents) or scores[index] is not None:
            raise AssertionError("rerank response contains invalid or repeated document indices")
        if type(score) not in (int, float) or not math.isfinite(score):
            raise AssertionError("rerank response contains nonfinite or invalid scores")
        if row.get("document") != {"text": documents[index]}:
            raise AssertionError("rerank response changed the indexed document text")
        scores[index] = float(score)
        order.append((-score, index))
    if order != sorted(order):
        raise AssertionError("rerank results are not stably sorted by descending score")
    usage = payload.get("usage", {})
    if (not isinstance(usage, dict) or type(usage.get("prompt_tokens")) is not int
            or type(usage.get("total_tokens")) is not int or usage["prompt_tokens"] <= 0
            or usage.get("total_tokens") != usage["prompt_tokens"]):
        raise AssertionError("rerank response omitted valid token usage")
    return scores


def quality_margin(scores: list[float], relevant: int) -> float:
    if type(relevant) is not int or not 0 <= relevant < len(scores):
        raise AssertionError("relevant passage index is invalid")
    if len(scores) < 2 or any(type(value) not in (int, float) or not math.isfinite(value) for value in scores):
        raise AssertionError("quality scores must be finite and contain distractors")
    margin = scores[relevant] - max(value for index, value in enumerate(scores) if index != relevant)
    if margin < MINIMUM_MARGIN:
        raise AssertionError(f"relevant passage margin {margin:.6f} is below {MINIMUM_MARGIN}")
    return margin


def run_quality(base: str, model: str, key: str) -> dict:
    request_ids = set()
    def rank(query, documents, top_n=None):
        selected = len(documents) if top_n is None else top_n
        status_code, payload = request(base, "/v1/rerank", {
            "model": model, "query": query, "documents": documents,
            "top_n": selected, "return_documents": True,
        }, key=key)
        if status_code != 200:
            raise AssertionError(f"trained rerank failed with HTTP {status_code}: {payload}")
        scores = validate_ranking(payload, model, documents, selected)
        if payload["id"] in request_ids:
            raise AssertionError("rerank request ID was reused")
        request_ids.add(payload["id"])
        return scores, payload

    results = []
    max_difference = 0.0
    for case in CASES:
        documents, query = case["documents"], case["query"]
        started = time.monotonic()
        scores, payload = rank(query, documents)
        elapsed = (time.monotonic() - started) * 1000
        margin = quality_margin(scores, case["relevant_index"])
        for index, document in enumerate(documents):
            single, _ = rank(query, [document])
            difference = abs(scores[index] - single[0])
            max_difference = max(max_difference, difference)
            if difference > SCORE_TOLERANCE:
                raise AssertionError("padded batch and independent pair scores differ")
        reversed_scores, _ = rank(query, list(reversed(documents)))
        for first, second in zip(scores, reversed(reversed_scores)):
            difference = abs(first - second)
            max_difference = max(max_difference, difference)
            if difference > SCORE_TOLERANCE:
                raise AssertionError("document permutation changed pair scores")
        _, top = rank(query, documents, 1)
        if top["results"][0]["index"] != case["relevant_index"]:
            raise AssertionError("top_n=1 did not return the relevant passage")
        results.append({"id": case["id"], "scores_by_input_index": scores,
                        "ranked_indices": [row["index"] for row in payload["results"]],
                        "relevant_index": case["relevant_index"], "margin": margin,
                        "batch_latency_ms": elapsed})
    if not any(abs(score) > 1.0 for case in results for score in case["scores_by_input_index"]):
        raise AssertionError("cross-encoder raw logits were unexpectedly clamped or normalized")

    # Eighteen inputs cross the native 16-item microbatch boundary. Duplicate
    # passages must keep their score and stable input-index tie ordering.
    large_documents = CASES[0]["documents"] * 6
    large_scores, _ = rank(CASES[0]["query"], large_documents)
    for index, score in enumerate(large_scores):
        if abs(score - results[0]["scores_by_input_index"][index % 3]) > SCORE_TOLERANCE:
            raise AssertionError("native microbatch boundary changed pair scores")
    repeat, _ = rank(CASES[0]["query"], CASES[0]["documents"])
    if any(abs(a - b) > SCORE_TOLERANCE for a, b in zip(repeat, results[0]["scores_by_input_index"])):
        raise AssertionError("intervening queries changed a repeated request's scores")
    return {"cases": results, "minimum_required_margin": MINIMUM_MARGIN,
            "score_semantics": "raw_classifier_logit", "score_tolerance": SCORE_TOLERANCE,
            "maximum_batch_single_permutation_difference": max_difference,
            "microbatch_documents": len(large_documents), "rerank_requests": len(request_ids)}


def verify_isolation(base: str, model: str, key: str) -> dict:
    generation = [
        ("/v1/chat/completions", {"messages": [{"role": "user", "content": "hello"}]}),
        ("/v1/completions", {"prompt": "hello"}),
        ("/v1/responses", {"input": "hello"}),
        ("/api/chat", {"messages": [{"role": "user", "content": "hello"}], "stream": False}),
        ("/api/generate", {"prompt": "hello", "stream": False}),
    ]
    embeddings = [
        ("/v1/embeddings", {"input": "hello"}),
        ("/api/embed", {"input": "hello"}),
        ("/api/embeddings", {"prompt": "hello"}),
    ]
    for route, body in generation + embeddings:
        status_code, payload = request(base, route, {"model": model, **body}, key=key)
        expected = 422 if (route, body) in generation else 501
        if status_code != expected:
            raise AssertionError(f"task isolation failed on {route}: HTTP {status_code}, {payload}")
        error = payload.get("error")
        if route.startswith("/v1/"):
            if not isinstance(error, dict) or error.get("type") != "unsupported_operation":
                raise AssertionError(f"task isolation error contract changed on {route}")
        elif not isinstance(error, str) or not error:
            raise AssertionError(f"task isolation error contract changed on {route}")
    status_code, _ = request(base, "/v1/rerank", {
        "model": model, "query": "hello " * 600, "documents": ["short"],
    }, key=key)
    if status_code != 400:
        raise AssertionError("rerank did not reject input beyond the model context")
    status_code, _ = request(base, "/v1/rerank", {"model": model, "query": "q", "documents": ["d"]})
    if status_code != 401:
        raise AssertionError("rerank did not require authentication")
    return {"generation_routes_rejected": len(generation), "embedding_routes_rejected": len(embeddings),
            "context_limit": "validated", "authentication": "validated"}


def inspect_model(infer_bin: Path, model: Path, env: dict) -> dict:
    result = subprocess.run([str(infer_bin), "--model", str(model), "--device", "cpu",
                             "--context-size", "4096", "--inspect"],
                            capture_output=True, text=True, env=env, timeout=60, check=True)
    report = json.loads(result.stdout)
    manifest, memory = report["manifest"], report["memory_estimate"]
    parameters = manifest["parameters"]
    if (manifest["family"] != "Bert" or manifest["primary_dtype"] != "F32"
            or parameters.get("bloom_task") != "rerank"
            or parameters.get("bloom_encoder_kind") != "cross_encoder"
            or parameters.get("max_position_embeddings") != 512
            or memory.get("weight_bytes") != FILES["model.safetensors"][0]
            or memory.get("weight_dtype") != "F32"
            or memory.get("kv_cache_bytes") != 0 or memory.get("kv_cache_bytes_per_token") != 0):
        raise AssertionError("cross-encoder routing or stateless CPU memory accounting changed")
    rejected = subprocess.run([str(infer_bin), "--model", str(model), "--device", "cpu",
                               "--prompt", "This generation request must fail.", "--max-tokens", "1"],
                              capture_output=True, text=True, env=env, timeout=60)
    if rejected.returncode == 0 or "cross-encoder" not in rejected.stderr or "/v1/rerank" not in rejected.stderr:
        raise AssertionError("CLI did not explicitly reject generation for the rerank-only encoder")
    return {"family": "Bert", "task": "rerank", "context_window": 512,
            "weight_bytes": memory["weight_bytes"], "kv_cache_bytes": 0, "cli_task_isolation": "validated"}


def run(args) -> dict:
    model = args.model_path.absolute()
    files = acquire_model(model)
    if not args.skip_build:
        subprocess.run(["cargo", "build", "--manifest-path", str(WORKSPACE / "Cargo.toml"),
                        "--locked", "--release", "--bin", "bloom_server", "--bin", "bloom_infer"],
                       check=True, stdout=sys.stderr)
    server_bin, infer_bin = args.server_bin.resolve(strict=True), args.infer_bin.resolve(strict=True)
    identities = {"server_sha256": sha256(server_bin), "infer_sha256": sha256(infer_bin)}
    env = {name: value for name, value in os.environ.items() if not name.startswith("BLOOM_")}
    key = secrets.token_urlsafe(24)
    env["BLOOM_API_KEY"] = key
    with tempfile.TemporaryDirectory(prefix="bloom-cross-encoder-gate-") as temporary:
        root = Path(temporary)
        env["BLOOM_CONFIG"] = str(root / "config.json")
        inspection = inspect_model(infer_bin, model, env)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        command = [str(server_bin), "--model", str(model), "--models-dir", str(root / "models"),
                   "--host", "127.0.0.1", "--port", str(port), "--backend", "candle", "--device", "cpu",
                   "--context-size", "4096", "--max-concurrent", "1", "--timeout", "120"]
        base = f"http://127.0.0.1:{port}"
        with (root / "server.log").open("w+") as log:
            process = subprocess.Popen(command, stdout=log, stderr=log, env=env)
            try:
                deadline = time.monotonic() + args.startup_timeout
                while True:
                    if process.poll() is not None:
                        raise AssertionError("cross-encoder server exited before readiness")
                    try:
                        status_code, ready = request(base, "/ready", timeout=2)
                        if status_code == 200:
                            validate_readiness_document(ready)
                            break
                        if ready.get("load_error"):
                            raise AssertionError(f"cross-encoder load failed: {ready['load_error']}")
                    except (urllib.error.URLError, TimeoutError):
                        pass
                    if time.monotonic() >= deadline:
                        raise AssertionError("cross-encoder server exceeded startup deadline")
                    time.sleep(0.1)
                if ready["model_tasks"] != ["rerank"] or ready["context_window"] != 512:
                    raise AssertionError("readiness did not advertise only bounded reranking")
                model_id = ready["model"]
                quality = run_quality(base, model_id, key)
                isolation = verify_isolation(base, model_id, key)
            except Exception:
                log.flush()
                log.seek(0)
                print(log.read()[-12000:], file=sys.stderr)
                raise
            finally:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
    if identities != {"server_sha256": sha256(server_bin), "infer_sha256": sha256(infer_bin)}:
        raise AssertionError("a tested binary changed during validation")
    for name, (size, digest) in FILES.items():
        verify_file(model / name, size, digest)
    return {"status": "ok", "schema_version": 1, "profile": "bloom.trained_cross_encoder",
            "model": REPOSITORY, "revision": REVISION, "license": "Apache-2.0",
            "model_files": files, "binaries": identities, "backend": "candle", "device": "cpu",
            "inspection": inspection, "quality": quality, "isolation": isolation,
            "stable_claim": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path(os.environ.get(
        "BLOOM_TRAINED_CROSS_ENCODER_CACHE", "/tmp/bloom-trained-cross-encoder-runtime")))
    parser.add_argument("--server-bin", type=Path, default=WORKSPACE / "target/release/bloom_server")
    parser.add_argument("--infer-bin", type=Path, default=WORKSPACE / "target/release/bloom_infer")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--startup-timeout", type=float, default=180)
    args = parser.parse_args()
    if not math.isfinite(args.startup_timeout) or args.startup_timeout <= 0:
        parser.error("--startup-timeout must be finite and positive")
    try:
        result = run(args)
    except Exception as error:
        result = {"status": "failed", "profile": "bloom.trained_cross_encoder", "error": str(error)}
        print(f"FAIL: {error}", file=sys.stderr)
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Validate Bloom JSON schemas and bundled example artifacts.

The script has a standard-library structural fallback. CI and release checks
install the pinned `jsonschema` dependency and perform full Draft-07 validation.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any

from readiness_contract import (
    READINESS_SERVER_PROTOCOL_VERSION,
    validate_readiness_document,
)

ROOT = Path(__file__).resolve().parents[1]


def model_package_digest(files: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    digest.update(b"bloom.model_package.v1\0")
    ordered = sorted(files, key=lambda file: file["filename"])
    digest.update(struct.pack(">I", len(ordered)))
    for file in ordered:
        name = file["filename"].encode("utf-8")
        digest.update(struct.pack(">I", len(name)))
        digest.update(name)
        digest.update(struct.pack(">Q", file["size_bytes"]))
        digest.update(bytes.fromhex(file["sha256"]))
    return digest.hexdigest()


def validate_model_package_weight_layout(files: list[dict[str, Any]]) -> bool:
    names = [file["filename"] for file in files]
    has_single = "model.safetensors" in names
    has_index = "model.safetensors.index.json" in names
    shard_names = [
        name
        for name in names
        if name.startswith("model-") and name.endswith(".safetensors")
    ]
    if has_single:
        return not has_index and not shard_names
    if not has_index or not shard_names:
        return False
    shards: list[tuple[int, int]] = []
    for name in shard_names:
        body = name.removeprefix("model-").removesuffix(".safetensors")
        parts = body.split("-of-")
        if (
            len(parts) != 2
            or any(len(part) != 5 or not part.isascii() or not part.isdigit() for part in parts)
        ):
            return False
        index, total = (int(part) for part in parts)
        if index == 0 or total == 0 or index > total or total > 256:
            return False
        shards.append((index, total))
    shards.sort()
    expected_total = shards[0][1]
    return len(shards) == expected_total and all(
        index == position and total == expected_total
        for position, (index, total) in enumerate(shards, start=1)
    )


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def require_keys(path: Path, data: dict[str, Any], keys: list[str]) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        raise AssertionError(f"{path}: missing required keys: {', '.join(missing)}")


def synthetic_benchmark_result() -> dict[str, Any]:
    return {
        "backend": "cpu",
        "model_id": "example-model",
        "dtype": "Q4",
        "quantization": "Q4_K_M",
        "tokens_per_second": 12.5,
        "ttft_ms": 120.0,
        "avg_latency_ms": 40.0,
        "peak_memory_bytes": 123456789,
        "memory_breakdown": {
            "weight_bytes": 100000000,
            "host_weight_bytes": 100000000,
            "device_weight_bytes": 0,
            "kv_cache_bytes": 20000000,
            "kv_cache_bytes_per_token": 131072,
            "temp_tensor_bytes": 3456789,
            "total_bytes": 123456789,
            "weight_dtype": "Q4",
            "quantization": {
                "scheme": {"GGUF": "Q4_K_M"},
                "bits": 4,
                "group_size": None,
                "act_order": False,
                "kv_cache_dtype": "F16",
                "imatrix": False,
            },
            "kv_cache_dtype": "F16",
            "num_layers": 32,
            "offloaded_layers": None,
            "mmap_residency_applied": False,
            "memory_scope": "host-resident estimate",
        },
        "tokens_generated": 16,
        "duration_secs": 1.28,
        "timestamp": "2026-06-18T00:00:00Z",
        "notes": "schema smoke fixture",
        "tbt_ms": 35.0,
        "prompt_processing_speed": 100.0,
        "hardware": {
            "device": "cpu",
            "backend": "candle",
            "os": "Linux",
        },
        "timing_breakdown": {
            "model_load_secs": 0.2,
            "avg_ttft_ms": 120.0,
            "avg_tbt_ms": 35.0,
            "avg_latency_ms": 40.0,
            "throughput_tokens_per_sec": 12.5,
            "prompt_processing_tokens_per_sec": 100.0,
        },
        "cache_metrics": {
            "enabled": False,
            "total_blocks": 0,
            "free_blocks": 0,
            "active_blocks": 0,
            "cached_blocks": 0,
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "reuses": 0,
            "utilization": 0.0,
        },
        "prompts_tested": 1,
        "avg_ttft_ms": 120.0,
        "avg_tbt_ms": 35.0,
    }


def synthetic_benchmark_result_speculative() -> dict[str, Any]:
    """Variant of `synthetic_benchmark_result` with speculative decoding enabled.

    Exercises the `speculative_mode`, `speculative_draft_tokens`,
    `speculative_accepted_tokens` and `speculative_acceptance_rate` schema
    properties so the Draft-07 validator covers the populated path, not just
    the omitted/null path.
    """
    result = synthetic_benchmark_result()
    result.update(
        {
            "speculative_mode": "ngram",
            "speculative_draft_tokens": 200,
            "speculative_accepted_tokens": 120,
            "speculative_acceptance_rate": 0.6,
        }
    )
    return result


def basic_validate() -> list[tuple[Path, str]]:
    checked: list[tuple[Path, str]] = []

    schema_paths = [
        ROOT / "examples/manifest/model-manifest.schema.json",
        ROOT / "examples/plugins/plugin-schema.json",
        ROOT / "examples/benchmark-schema.json",
        ROOT / "examples/world-model-schema.json",
        ROOT / "examples/model-catalog.schema.json",
        ROOT / "examples/model-inventory.schema.json",
        ROOT / "examples/model-inventory-reconciliation.schema.json",
        ROOT / "examples/conversation-archive.schema.json",
        ROOT / "examples/model-preflight.schema.json",
        ROOT / "examples/readiness.schema.json",
        ROOT / "examples/encoder-result.schema.json",
        ROOT / "examples/observability-snapshot.schema.json",
        ROOT / "examples/server-doctor.schema.json",
        ROOT / "examples/release-manifest.schema.json",
        ROOT / "examples/model-index-payload.schema.json",
        ROOT / "examples/model-index-envelope.schema.json",
        ROOT / "examples/model-index-response.schema.json",
    ]
    for path in schema_paths:
        data = load_json(path)
        require_keys(path, data, ["$schema", "title", "type"])
        checked.append((path, "schema-json"))

    manifest_path = ROOT / "examples/manifest/bloom.json"
    manifest = load_json(manifest_path)
    require_keys(
        manifest_path,
        manifest,
        ["id", "family", "version", "license", "io_schema", "memory_profile", "files", "primary_dtype"],
    )
    require_keys(manifest_path, manifest["io_schema"], ["inputs", "outputs"])
    checked.append((manifest_path, "model-manifest-basic"))

    for path in sorted((ROOT / "examples/plugins").glob("*.json")):
        data = load_json(path)
        if path.name == "plugin-schema.json":
            continue
        require_keys(path, data, ["metadata"])
        if "entry_point" in data:
            require_keys(path, data["entry_point"], ["type", "path"])
            checked.append((path, "runtime-plugin-basic"))
        else:
            require_keys(
                path,
                data,
                ["family", "model_version", "primary_dtype", "quantizations", "files", "total_size_bytes"],
            )
            checked.append((path, "model-package-basic"))

    benchmark = synthetic_benchmark_result()
    require_keys(
        Path("<synthetic benchmark>"),
        benchmark,
        [
            "backend",
            "model_id",
            "dtype",
            "tokens_per_second",
            "avg_latency_ms",
            "peak_memory_bytes",
            "tokens_generated",
            "duration_secs",
            "timestamp",
            "cache_metrics",
        ],
    )
    checked.append((Path("<synthetic benchmark>"), "benchmark-basic"))

    world_schema_path = ROOT / "examples/world-model-example.json"
    world_schema = load_json(world_schema_path)
    require_keys(world_schema_path, world_schema, ["world_state_schema", "action_schema"])
    require_keys(
        world_schema_path,
        world_schema["world_state_schema"],
        ["scalar_ranges", "allowed_image_mimes", "tensor_shapes", "allow_text", "allow_audio"],
    )
    require_keys(
        world_schema_path,
        world_schema["action_schema"],
        ["allowed_action_spaces", "action_dimensions", "value_range"],
    )
    checked.append((world_schema_path, "world-model-basic"))

    model_catalog_path = ROOT / "examples/model-catalog.json"
    model_catalog = load_json(model_catalog_path)
    require_keys(
        model_catalog_path,
        model_catalog,
        [
            "schema_version",
            "object",
            "root",
            "root_exists",
            "data",
            "active_model",
            "download",
            "import",
            "index",
            "storage",
            "integrity",
            "load",
        ],
    )
    if (
        model_catalog["schema_version"] != 1
        or model_catalog["object"] != "bloom.model_catalog"
    ):
        raise AssertionError(f"{model_catalog_path}: unsupported model catalog identity")
    if not model_catalog["root_exists"] and model_catalog["data"]:
        raise AssertionError(f"{model_catalog_path}: missing root contains models")
    catalog_ids = [model["id"] for model in model_catalog["data"]]
    if len(catalog_ids) != len(set(catalog_ids)) or len(catalog_ids) > 4096:
        raise AssertionError(f"{model_catalog_path}: invalid model identities")
    active_entries = [model["id"] for model in model_catalog["data"] if model["active"]]
    active_model = model_catalog["active_model"]
    if active_model is not None and active_model["source"] == "catalog":
        if active_entries != [active_model["catalog_id"]]:
            raise AssertionError(f"{model_catalog_path}: inconsistent active catalog model")
    elif active_entries:
        raise AssertionError(f"{model_catalog_path}: active entry has no catalog runtime")
    load = model_catalog["load"]
    if load["phase"] == "ready" and active_model is None:
        raise AssertionError(f"{model_catalog_path}: ready catalog has no active model")
    storage = model_catalog["storage"]
    used_bytes = (
        storage["installed_bytes"]
        + storage["staged_download_bytes"]
        + storage["staged_import_bytes"]
    )
    if storage["used_bytes"] != used_bytes or storage["committed_bytes"] != (
        used_bytes + storage["reserved_bytes"]
    ):
        raise AssertionError(f"{model_catalog_path}: inconsistent storage accounting")
    checked.append((model_catalog_path, "model-catalog-basic"))

    empty_catalog_path = ROOT / "examples/model-catalog-empty.json"
    empty_catalog = load_json(empty_catalog_path)
    require_keys(
        empty_catalog_path,
        empty_catalog,
        [
            "schema_version",
            "object",
            "root_exists",
            "data",
            "active_model",
            "download",
            "import",
            "index",
            "storage",
            "integrity",
            "load",
        ],
    )
    if (
        empty_catalog["schema_version"] != 1
        or empty_catalog["object"] != "bloom.model_catalog"
        or empty_catalog["data"]
        or empty_catalog["active_model"] is not None
        or empty_catalog["load"]["phase"] != "idle"
        or empty_catalog["download"]["enabled"]
        or empty_catalog["import"]["enabled"]
        or empty_catalog["index"]["enabled"]
    ):
        raise AssertionError(f"{empty_catalog_path}: inconsistent empty catalog")
    checked.append((empty_catalog_path, "model-catalog-basic"))

    inventory_path = ROOT / "examples/model-inventory.json"
    inventory = load_json(inventory_path)
    require_keys(inventory_path, inventory, ["schema_version", "object", "summary", "models"])
    if inventory["schema_version"] != 2 or inventory["object"] != "bloom.model_inventory":
        raise AssertionError(f"{inventory_path}: unsupported example inventory identity")
    require_keys(
        inventory_path,
        inventory["summary"],
        [
            "model_count",
            "provenance_count",
            "source_locked_count",
            "quarantined_count",
            "invalid_provenance_count",
        ],
    )
    if inventory["summary"]["model_count"] != len(inventory["models"]):
        raise AssertionError(f"{inventory_path}: model_count does not match models")
    for model in inventory["models"]:
        require_keys(
            inventory_path,
            model,
            [
                "id",
                "provenance_status",
                "acquisition",
                "model_index_id",
            ],
        )
        model_index_id = model["model_index_id"]
        if model_index_id is None:
            continue
        if (
            not isinstance(model_index_id, str)
            or not 1 <= len(model_index_id) <= 64
            or not model_index_id[0].isalnum()
            or not model_index_id[0].isascii()
            or not all(
                character.isascii()
                and (character.islower() or character.isdigit() or character in ".-_")
                for character in model_index_id
            )
        ):
            raise AssertionError(
                f"{inventory_path}: invalid signed-index ID for {model['id']}"
            )
        if (
            model["provenance_status"] != "recorded"
            or model["acquisition"] != "download"
        ):
            raise AssertionError(
                f"{inventory_path}: signed-index ID requires recorded download provenance"
            )
    checked.append((inventory_path, "model-inventory-basic"))

    reconciliation_path = ROOT / "examples/model-inventory-reconciliation.json"
    reconciliation = load_json(reconciliation_path)
    require_keys(
        reconciliation_path,
        reconciliation,
        ["schema_version", "object", "in_sync", "truncated", "summary", "drift"],
    )
    if (
        reconciliation["schema_version"] != 1
        or reconciliation["object"] != "bloom.model_inventory_reconciliation"
    ):
        raise AssertionError(f"{reconciliation_path}: unsupported reconciliation identity")
    require_keys(
        reconciliation_path,
        reconciliation["summary"],
        [
            "expected_model_count",
            "current_model_count",
            "matching_count",
            "missing_count",
            "unexpected_count",
            "changed_count",
            "blocking_count",
            "restorable_count",
            "drift_count",
        ],
    )
    summary = reconciliation["summary"]
    if summary["drift_count"] < len(reconciliation["drift"]):
        raise AssertionError(f"{reconciliation_path}: drift_count is smaller than drift details")
    if summary["expected_model_count"] != (
        summary["matching_count"] + summary["missing_count"] + summary["changed_count"]
    ):
        raise AssertionError(f"{reconciliation_path}: expected_model_count is inconsistent")
    if summary["current_model_count"] != (
        summary["matching_count"] + summary["unexpected_count"] + summary["changed_count"]
    ):
        raise AssertionError(f"{reconciliation_path}: current_model_count is inconsistent")
    if summary["drift_count"] != (
        summary["missing_count"] + summary["unexpected_count"] + summary["changed_count"]
    ):
        raise AssertionError(f"{reconciliation_path}: drift_count is inconsistent")
    if summary["restorable_count"] > summary["missing_count"]:
        raise AssertionError(f"{reconciliation_path}: restorable_count is inconsistent")
    if reconciliation["in_sync"] != (summary["drift_count"] == 0):
        raise AssertionError(f"{reconciliation_path}: in_sync is inconsistent")
    if reconciliation["truncated"] != (len(reconciliation["drift"]) < summary["drift_count"]):
        raise AssertionError(f"{reconciliation_path}: truncated is inconsistent")
    drift_ids = [drift["id"] for drift in reconciliation["drift"]]
    if drift_ids != sorted(set(drift_ids)):
        raise AssertionError(f"{reconciliation_path}: drift IDs are not sorted and unique")
    detailed_restorable = sum(
        1 for drift in reconciliation["drift"] if drift["restore_available"]
    )
    if detailed_restorable > summary["restorable_count"]:
        raise AssertionError(f"{reconciliation_path}: restore capabilities are inconsistent")
    checked.append((reconciliation_path, "model-inventory-reconciliation-basic"))

    conversation_archive_path = ROOT / "examples/conversation-archive.json"
    conversation_archive = load_json(conversation_archive_path)
    require_keys(
        conversation_archive_path,
        conversation_archive,
        ["version", "object", "active_conversation", "conversations"],
    )
    if (
        conversation_archive["version"] != 2
        or conversation_archive["object"] != "bloom.conversation_archive"
    ):
        raise AssertionError(
            f"{conversation_archive_path}: unsupported conversation archive identity"
        )
    conversations = conversation_archive["conversations"]
    if not conversations:
        raise AssertionError(f"{conversation_archive_path}: no conversations")
    active_conversation = conversation_archive["active_conversation"]
    if not isinstance(active_conversation, int) or not 0 <= active_conversation < len(
        conversations
    ):
        raise AssertionError(
            f"{conversation_archive_path}: invalid active conversation index"
        )
    for conversation in conversations:
        require_keys(conversation_archive_path, conversation, ["title", "messages"])
        for message in conversation["messages"]:
            require_keys(conversation_archive_path, message, ["role", "content"])
            if message["role"] not in {"user", "assistant"}:
                raise AssertionError(
                    f"{conversation_archive_path}: unsupported conversation role"
                )
            if message.get("attachment_unavailable", False) and message["role"] != "user":
                raise AssertionError(
                    f"{conversation_archive_path}: invalid attachment replay metadata"
                )
            model = message.get("model")
            if model is not None and (
                message["role"] != "assistant"
                or not isinstance(model, str)
                or not 1 <= len(model) <= 256
                or model != model.strip()
                or any(ord(character) < 32 or 127 <= ord(character) <= 159 for character in model)
            ):
                raise AssertionError(
                    f"{conversation_archive_path}: invalid execution model provenance"
                )
    checked.append((conversation_archive_path, "conversation-archive-basic"))

    model_preflight_path = ROOT / "examples/model-preflight.json"
    model_preflight = load_json(model_preflight_path)
    require_keys(
        model_preflight_path,
        model_preflight,
        ["schema_version", "object", "data"],
    )
    if (
        model_preflight["schema_version"] != 1
        or model_preflight["object"] != "bloom.model_preflight"
    ):
        raise AssertionError(f"{model_preflight_path}: unsupported preflight identity")
    preflight_data = model_preflight["data"]
    require_keys(
        model_preflight_path,
        preflight_data,
        [
            "model_id",
            "inspected_at",
            "loadable",
            "load_blocker",
            "manifest",
            "runtime",
            "memory",
            "warnings",
        ],
    )
    require_keys(
        model_preflight_path,
        preflight_data["manifest"],
        ["id", "family", "version", "model_tasks"],
    )
    require_keys(
        model_preflight_path,
        preflight_data["runtime"],
        ["backend_available", "support"],
    )
    require_keys(
        model_preflight_path,
        preflight_data["memory"],
        [
            "per_request_context_tokens",
            "max_concurrent",
            "planned_context_tokens",
            "memory_utilization",
            "fits_budget",
        ],
    )
    preflight_tasks = preflight_data["manifest"]["model_tasks"]
    if preflight_tasks not in (["generation"], ["embedding", "rerank"]):
        raise AssertionError(f"{model_preflight_path}: invalid model tasks")
    preflight_memory = preflight_data["memory"]
    if preflight_memory["planned_context_tokens"] != (
        preflight_memory["per_request_context_tokens"]
        * preflight_memory["max_concurrent"]
    ):
        raise AssertionError(f"{model_preflight_path}: inconsistent context plan")
    utilization = preflight_memory["memory_utilization"]
    if not isinstance(utilization, (int, float)) or not math.isfinite(utilization) or not 0 <= utilization <= 1:
        raise AssertionError(f"{model_preflight_path}: invalid memory utilization")
    if preflight_data["loadable"] != (preflight_data["load_blocker"] is None):
        raise AssertionError(f"{model_preflight_path}: inconsistent load decision")
    if preflight_data["loadable"] and (
        not preflight_data["runtime"]["backend_available"]
        or preflight_data["runtime"]["support"] == "unsupported"
        or not preflight_memory["fits_budget"]
    ):
        raise AssertionError(f"{model_preflight_path}: unsafe load decision")
    checked.append((model_preflight_path, "model-preflight-basic"))

    readiness_path = ROOT / "examples/readiness.json"
    readiness = load_json(readiness_path)
    validate_readiness_document(
        readiness,
        expected_server_protocol_version=READINESS_SERVER_PROTOCOL_VERSION,
    )
    checked.append((readiness_path, "readiness-basic"))

    embedding_result_path = ROOT / "examples/embedding-result.json"
    embedding_result = load_json(embedding_result_path)
    require_keys(
        embedding_result_path,
        embedding_result,
        ["schema_version", "object", "model", "prompt_tokens", "vectors"],
    )
    if (
        embedding_result["schema_version"] != 1
        or embedding_result["object"] != "bloom.embedding_result"
        or not embedding_result["vectors"]
        or len(embedding_result["vectors"]) > 256
    ):
        raise AssertionError(f"{embedding_result_path}: invalid embedding result identity")
    dimensions = None
    total_values = 0
    total_input_bytes = 0
    for expected_index, vector in enumerate(embedding_result["vectors"]):
        require_keys(embedding_result_path, vector, ["index", "input", "embedding"])
        values = vector["embedding"]
        if vector["index"] != expected_index or not values or len(values) > 16384:
            raise AssertionError(f"{embedding_result_path}: invalid vector shape or index")
        if dimensions is not None and len(values) != dimensions:
            raise AssertionError(f"{embedding_result_path}: inconsistent vector dimensions")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in values
        ):
            raise AssertionError(f"{embedding_result_path}: non-finite vector value")
        norm = math.sqrt(sum(float(value) * float(value) for value in values))
        if not math.isfinite(norm) or abs(norm - 1.0) > 0.001:
            raise AssertionError(f"{embedding_result_path}: vector is not L2-normalized")
        dimensions = len(values)
        total_values += len(values)
        total_input_bytes += len(vector["input"].encode("utf-8"))
    if total_values > 1048576 or total_input_bytes > 768 * 1024:
        raise AssertionError(f"{embedding_result_path}: aggregate embedding limit exceeded")
    checked.append((embedding_result_path, "embedding-result-basic"))

    rerank_result_path = ROOT / "examples/rerank-result.json"
    rerank_result = load_json(rerank_result_path)
    require_keys(
        rerank_result_path,
        rerank_result,
        [
            "schema_version",
            "object",
            "id",
            "model",
            "query",
            "prompt_tokens",
            "results",
        ],
    )
    if (
        rerank_result["schema_version"] != 1
        or rerank_result["object"] != "bloom.rerank_result"
        or not rerank_result["results"]
        or len(rerank_result["results"]) > 256
    ):
        raise AssertionError(f"{rerank_result_path}: invalid rerank result identity")
    seen_indices: set[int] = set()
    previous: tuple[float, int] | None = None
    total_rerank_bytes = len(rerank_result["query"].encode("utf-8"))
    for result in rerank_result["results"]:
        require_keys(
            rerank_result_path, result, ["index", "relevance_score", "document"]
        )
        index = result["index"]
        score = result["relevance_score"]
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < 256
            or index in seen_indices
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
            or not -1 <= score <= 1
            or (
                previous is not None
                and (score > previous[0] or (score == previous[0] and index <= previous[1]))
            )
        ):
            raise AssertionError(f"{rerank_result_path}: invalid rerank order or score")
        seen_indices.add(index)
        previous = (float(score), index)
        total_rerank_bytes += len(result["document"].encode("utf-8"))
    if total_rerank_bytes > 768 * 1024:
        raise AssertionError(f"{rerank_result_path}: aggregate rerank limit exceeded")
    checked.append((rerank_result_path, "rerank-result-basic"))

    observability_path = ROOT / "examples/observability-snapshot.json"
    observability = load_json(observability_path)
    require_keys(
        observability_path,
        observability,
        [
            "schema_version",
            "object",
            "created",
            "server",
            "model",
            "ready",
            "load",
            "requests",
            "tokens",
            "scheduler",
            "kv_cache",
            "memory",
        ],
    )
    if (
        observability["schema_version"] != 1
        or observability["object"] != "bloom.observability_snapshot"
    ):
        raise AssertionError(f"{observability_path}: unsupported snapshot identity")
    requests = observability["requests"]
    require_keys(
        observability_path,
        requests,
        ["total", "completed", "failed", "in_flight"],
    )
    if requests["completed"] + requests["failed"] + requests["in_flight"] != requests["total"]:
        raise AssertionError(f"{observability_path}: request counters are inconsistent")
    load = observability["load"]
    require_keys(
        observability_path,
        load,
        ["phase", "progress", "requested_model", "failure_present"],
    )
    if load["failure_present"] != (load["phase"] == "failed"):
        raise AssertionError(f"{observability_path}: load failure state is inconsistent")
    kv_cache = observability["kv_cache"]
    if kv_cache["free_blocks"] > kv_cache["total_blocks"]:
        raise AssertionError(f"{observability_path}: KV cache counters are inconsistent")
    checked.append((observability_path, "observability-snapshot-basic"))

    doctor_path = ROOT / "examples/server-doctor.json"
    doctor = load_json(doctor_path)
    require_keys(
        doctor_path,
        doctor,
        [
            "schema_version",
            "object",
            "created",
            "bloom_version",
            "status",
            "summary",
            "checks",
        ],
    )
    if doctor["schema_version"] != 1 or doctor["object"] != "bloom.server_doctor":
        raise AssertionError(f"{doctor_path}: unsupported server doctor identity")
    checks = doctor["checks"]
    expected_check_ids = [
        "configuration",
        "arguments",
        "network_security",
        "runtime_engine",
        "device_backend",
        "model_catalog",
        "startup_model",
        "storage_policy",
        "model_license_policy",
        "model_index",
        "model_index_state",
        "embedded_ui",
    ]
    check_ids = [check["id"] for check in checks]
    if check_ids != expected_check_ids:
        raise AssertionError(f"{doctor_path}: check IDs are incomplete or out of order")
    counts = {
        "passed": sum(check["status"] == "pass" for check in checks),
        "warnings": sum(check["status"] == "warn" for check in checks),
        "failures": sum(check["status"] == "fail" for check in checks),
    }
    if doctor["summary"] != counts:
        raise AssertionError(f"{doctor_path}: summary does not match check statuses")
    expected_status = (
        "fail" if counts["failures"] else "warn" if counts["warnings"] else "pass"
    )
    if doctor["status"] != expected_status:
        raise AssertionError(f"{doctor_path}: top-level status is inconsistent")
    checked.append((doctor_path, "server-doctor-basic"))

    release_path = ROOT / "examples/release-manifest.json"
    release = load_json(release_path)
    require_keys(
        release_path,
        release,
        [
            "schema_version",
            "object",
            "bloom_version",
            "target",
            "embedded_ui",
            "self_check",
            "binaries",
        ],
    )
    if release["schema_version"] != 1 or release["object"] != "bloom.release":
        raise AssertionError(f"{release_path}: unsupported release manifest identity")
    binary_names = [binary["name"] for binary in release["binaries"]]
    expected_binaries = ["bloom_bench", "bloom_infer", "bloom_server", "inspect_gguf"]
    if binary_names != expected_binaries:
        raise AssertionError(f"{release_path}: binaries are incomplete or not sorted")
    binary_hashes = [binary["sha256"] for binary in release["binaries"]]
    if len(binary_hashes) != len(set(binary_hashes)):
        raise AssertionError(f"{release_path}: binary checksums are not unique")
    self_check = release["self_check"]
    if self_check["status"] == "passed":
        if self_check["failures"] != 0 or self_check["doctor_status"] not in {
            "pass",
            "warn",
        }:
            raise AssertionError(f"{release_path}: passed self-check is inconsistent")
    elif self_check["doctor_status"] is not None or self_check["failures"] is not None:
        raise AssertionError(f"{release_path}: skipped self-check must not claim results")
    checked.append((release_path, "release-manifest-basic"))

    model_index_payload_path = ROOT / "examples/model-index-payload.json"
    model_index_payload = load_json(model_index_payload_path)
    require_keys(
        model_index_payload_path,
        model_index_payload,
        ["schema_version", "object", "name", "generated_at", "expires_at", "models"],
    )
    if (
        model_index_payload["schema_version"] != 1
        or model_index_payload["object"] != "bloom.model_index"
        or model_index_payload["expires_at"] <= model_index_payload["generated_at"]
    ):
        raise AssertionError(f"{model_index_payload_path}: invalid model index identity or validity")
    model_ids = [model["id"] for model in model_index_payload["models"]]
    model_filenames = [model["filename"].lower() for model in model_index_payload["models"]]
    if len(model_ids) != len(set(model_ids)) or len(model_filenames) != len(set(model_filenames)):
        raise AssertionError(f"{model_index_payload_path}: duplicate model entries")
    checked.append((model_index_payload_path, "model-index-payload-basic"))

    model_index_payload_v2_path = ROOT / "examples/model-index-payload-v2.json"
    model_index_payload_v2 = load_json(model_index_payload_v2_path)
    if (
        model_index_payload_v2["schema_version"] != 2
        or model_index_payload_v2["object"] != "bloom.model_index"
    ):
        raise AssertionError(f"{model_index_payload_v2_path}: invalid v2 index identity")
    for model in model_index_payload_v2["models"]:
        files = model["files"]
        if (
            len(files) < 2
            or sum(file["size_bytes"] for file in files) != model["size_bytes"]
            or not any(file["filename"] == "config.json" for file in files)
            or not validate_model_package_weight_layout(files)
        ):
            raise AssertionError(f"{model_index_payload_v2_path}: invalid package manifest")
    checked.append((model_index_payload_v2_path, "model-index-payload-v2-basic"))

    signed_index_path = ROOT / "examples/model-index.signed.json"
    signed_index = load_json(signed_index_path)
    require_keys(
        signed_index_path,
        signed_index,
        ["schema_version", "object", "algorithm", "key_id", "payload", "signature"],
    )
    if (
        signed_index["schema_version"] != 1
        or signed_index["object"] != "bloom.signed_model_index"
        or signed_index["algorithm"] != "ed25519"
    ):
        raise AssertionError(f"{signed_index_path}: invalid signed model index identity")
    payload_text = signed_index["payload"]
    signature_text = signed_index["signature"]
    decoded_payload = base64.urlsafe_b64decode(payload_text + "=" * (-len(payload_text) % 4))
    decoded_signature = base64.urlsafe_b64decode(
        signature_text + "=" * (-len(signature_text) % 4)
    )
    if decoded_payload != model_index_payload_path.read_bytes():
        raise AssertionError(f"{signed_index_path}: embedded payload differs from the payload example")
    if len(decoded_signature) != 64:
        raise AssertionError(f"{signed_index_path}: Ed25519 signature is not 64 bytes")
    packaged_fixture_dir = ROOT / "crates/engine/examples/fixtures"
    if (
        packaged_fixture_dir / "model-index-payload.json"
    ).read_bytes() != model_index_payload_path.read_bytes():
        raise AssertionError("packaged model index payload fixture differs from public example")
    if (
        packaged_fixture_dir / "model-index.signed.json"
    ).read_bytes() != signed_index_path.read_bytes():
        raise AssertionError("packaged signed model index fixture differs from public example")
    checked.append((signed_index_path, "model-index-envelope-basic"))

    model_index_response_path = ROOT / "examples/model-index-response.json"
    model_index_response = load_json(model_index_response_path)
    require_keys(
        model_index_response_path,
        model_index_response,
        [
            "schema_version",
            "object",
            "key_id",
            "source_kind",
            "cache_status",
            "warning",
            "data",
        ],
    )
    if (
        model_index_response["schema_version"] != 1
        or model_index_response["object"] != "bloom.model_index"
    ):
        raise AssertionError(f"{model_index_response_path}: invalid response identity")
    for entry in model_index_response["data"]:
        if entry["downloadable"] != (len(entry["blocking_reasons"]) == 0):
            raise AssertionError(
                f"{model_index_response_path}: inconsistent download policy state"
            )
    checked.append((model_index_response_path, "model-index-response-basic"))
    model_index_response_v2_path = ROOT / "examples/model-index-response-v2.json"
    model_index_response_v2 = load_json(model_index_response_v2_path)
    if (
        model_index_response_v2["schema_version"] != 2
        or model_index_response_v2["object"] != "bloom.model_index"
    ):
        raise AssertionError(f"{model_index_response_v2_path}: invalid v2 response identity")
    for entry in model_index_response_v2["data"]:
        if (
            entry["sha256"] != model_package_digest(entry["files"])
            or entry["size_bytes"]
            != sum(file["size_bytes"] for file in entry["files"])
        ):
            raise AssertionError(f"{model_index_response_v2_path}: invalid package identity")
    checked.append((model_index_response_v2_path, "model-index-response-v2-basic"))
    return checked


def jsonschema_validate() -> list[tuple[Path, str]]:
    try:
        import jsonschema  # type: ignore
    except ImportError:
        return []

    checked: list[tuple[Path, str]] = []
    manifest_schema = load_json(ROOT / "examples/manifest/model-manifest.schema.json")
    plugin_schema = load_json(ROOT / "examples/plugins/plugin-schema.json")
    benchmark_schema = load_json(ROOT / "examples/benchmark-schema.json")
    world_model_schema = load_json(ROOT / "examples/world-model-schema.json")
    model_catalog_schema = load_json(ROOT / "examples/model-catalog.schema.json")
    inventory_schema = load_json(ROOT / "examples/model-inventory.schema.json")
    reconciliation_schema = load_json(
        ROOT / "examples/model-inventory-reconciliation.schema.json"
    )
    conversation_archive_schema = load_json(
        ROOT / "examples/conversation-archive.schema.json"
    )
    model_preflight_schema = load_json(ROOT / "examples/model-preflight.schema.json")
    readiness_schema = load_json(ROOT / "examples/readiness.schema.json")
    encoder_result_schema = load_json(ROOT / "examples/encoder-result.schema.json")
    observability_schema = load_json(
        ROOT / "examples/observability-snapshot.schema.json"
    )
    server_doctor_schema = load_json(ROOT / "examples/server-doctor.schema.json")
    release_manifest_schema = load_json(
        ROOT / "examples/release-manifest.schema.json"
    )
    model_index_payload_schema = load_json(
        ROOT / "examples/model-index-payload.schema.json"
    )
    model_index_envelope_schema = load_json(
        ROOT / "examples/model-index-envelope.schema.json"
    )
    model_index_response_schema = load_json(
        ROOT / "examples/model-index-response.schema.json"
    )

    manifest_path = ROOT / "examples/manifest/bloom.json"
    jsonschema.Draft7Validator.check_schema(manifest_schema)
    jsonschema.validate(load_json(manifest_path), manifest_schema)
    checked.append((manifest_path, "model-manifest-draft7"))

    jsonschema.Draft7Validator.check_schema(plugin_schema)
    for path in sorted((ROOT / "examples/plugins").glob("*.json")):
        if path.name == "plugin-schema.json":
            continue
        jsonschema.validate(load_json(path), plugin_schema)
        checked.append((path, "plugin-draft7"))

    jsonschema.Draft7Validator.check_schema(benchmark_schema)
    jsonschema.validate(synthetic_benchmark_result(), benchmark_schema)
    checked.append((Path("<synthetic benchmark>"), "benchmark-draft7"))
    jsonschema.validate(synthetic_benchmark_result_speculative(), benchmark_schema)
    checked.append((Path("<synthetic benchmark speculative>"), "benchmark-draft7"))

    world_schema_path = ROOT / "examples/world-model-example.json"
    jsonschema.Draft7Validator.check_schema(world_model_schema)
    jsonschema.validate(load_json(world_schema_path), world_model_schema)
    checked.append((world_schema_path, "world-model-draft7"))

    jsonschema.Draft7Validator.check_schema(model_catalog_schema)
    for model_catalog_path in (
        ROOT / "examples/model-catalog.json",
        ROOT / "examples/model-catalog-empty.json",
    ):
        jsonschema.validate(load_json(model_catalog_path), model_catalog_schema)
        checked.append((model_catalog_path, "model-catalog-draft7"))

    inventory_path = ROOT / "examples/model-inventory.json"
    jsonschema.Draft7Validator.check_schema(inventory_schema)
    jsonschema.validate(load_json(inventory_path), inventory_schema)
    checked.append((inventory_path, "model-inventory-draft7"))

    reconciliation_path = ROOT / "examples/model-inventory-reconciliation.json"
    jsonschema.Draft7Validator.check_schema(reconciliation_schema)
    jsonschema.validate(load_json(reconciliation_path), reconciliation_schema)
    checked.append((reconciliation_path, "model-inventory-reconciliation-draft7"))

    conversation_archive_path = ROOT / "examples/conversation-archive.json"
    jsonschema.Draft7Validator.check_schema(conversation_archive_schema)
    jsonschema.validate(
        load_json(conversation_archive_path), conversation_archive_schema
    )
    checked.append((conversation_archive_path, "conversation-archive-draft7"))

    model_preflight_path = ROOT / "examples/model-preflight.json"
    jsonschema.Draft7Validator.check_schema(model_preflight_schema)
    jsonschema.validate(load_json(model_preflight_path), model_preflight_schema)
    checked.append((model_preflight_path, "model-preflight-draft7"))

    readiness_path = ROOT / "examples/readiness.json"
    jsonschema.Draft7Validator.check_schema(readiness_schema)
    jsonschema.validate(load_json(readiness_path), readiness_schema)
    checked.append((readiness_path, "readiness-draft7"))

    jsonschema.Draft7Validator.check_schema(encoder_result_schema)
    for encoder_result_path in (
        ROOT / "examples/embedding-result.json",
        ROOT / "examples/rerank-result.json",
    ):
        jsonschema.validate(load_json(encoder_result_path), encoder_result_schema)
        checked.append((encoder_result_path, "encoder-result-draft7"))

    observability_path = ROOT / "examples/observability-snapshot.json"
    jsonschema.Draft7Validator.check_schema(observability_schema)
    jsonschema.validate(load_json(observability_path), observability_schema)
    checked.append((observability_path, "observability-snapshot-draft7"))

    server_doctor_path = ROOT / "examples/server-doctor.json"
    jsonschema.Draft7Validator.check_schema(server_doctor_schema)
    jsonschema.validate(load_json(server_doctor_path), server_doctor_schema)
    checked.append((server_doctor_path, "server-doctor-draft7"))

    release_manifest_path = ROOT / "examples/release-manifest.json"
    jsonschema.Draft7Validator.check_schema(release_manifest_schema)
    jsonschema.validate(load_json(release_manifest_path), release_manifest_schema)
    checked.append((release_manifest_path, "release-manifest-draft7"))

    model_index_payload_path = ROOT / "examples/model-index-payload.json"
    jsonschema.Draft7Validator.check_schema(model_index_payload_schema)
    jsonschema.validate(load_json(model_index_payload_path), model_index_payload_schema)
    checked.append((model_index_payload_path, "model-index-payload-draft7"))
    model_index_payload_v2_path = ROOT / "examples/model-index-payload-v2.json"
    jsonschema.validate(load_json(model_index_payload_v2_path), model_index_payload_schema)
    checked.append((model_index_payload_v2_path, "model-index-payload-v2-draft7"))

    signed_index_path = ROOT / "examples/model-index.signed.json"
    jsonschema.Draft7Validator.check_schema(model_index_envelope_schema)
    jsonschema.validate(load_json(signed_index_path), model_index_envelope_schema)
    checked.append((signed_index_path, "model-index-envelope-draft7"))

    model_index_response_path = ROOT / "examples/model-index-response.json"
    jsonschema.Draft7Validator.check_schema(model_index_response_schema)
    jsonschema.validate(load_json(model_index_response_path), model_index_response_schema)
    checked.append((model_index_response_path, "model-index-response-draft7"))
    model_index_response_v2_path = ROOT / "examples/model-index-response-v2.json"
    jsonschema.validate(load_json(model_index_response_v2_path), model_index_response_schema)
    checked.append((model_index_response_v2_path, "model-index-response-v2-draft7"))
    return checked


def main() -> int:
    try:
        checked = basic_validate()
        checked.extend(jsonschema_validate())
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    for path, mode in checked:
        try:
            label = path.relative_to(ROOT)
        except ValueError:
            label = path
        print(f"valid {mode}: {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

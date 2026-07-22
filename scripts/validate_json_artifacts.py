#!/usr/bin/env python3
"""Validate Bloom JSON schemas and bundled example artifacts.

The script uses only the Python standard library by default. If the optional
`jsonschema` package is installed, it also performs full Draft-07 validation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


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

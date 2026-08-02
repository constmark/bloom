"""Shared standard-library validation for Bloom readiness artifacts."""

from __future__ import annotations

import math
import unicodedata
from typing import Any


READINESS_SCHEMA_VERSION = 3
READINESS_SERVER_PROTOCOL_VERSION = 3
UI_PROTOCOL_VERSION = 3
READINESS_OBJECT = "bloom.readiness"
MAX_PROTOCOL_VERSION = 2**32 - 1
MAX_U64 = 2**64 - 1
READINESS_REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "object",
        "protocol_version",
        "minimum_ui_protocol_version",
        "maximum_ui_protocol_version",
        "server_version",
        "status",
        "progress",
        "model",
        "loading",
        "load_error",
        "input_modalities",
        "model_tasks",
        "context_window",
        "in_flight_requests",
        "available_permits",
        "memory_pressure_high",
        "ram_utilization",
    }
)


class ReadinessContractError(ValueError):
    """A readiness document violates the supported public contract."""


def _integer(value: object, *, minimum: int, maximum: int | None = None) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= minimum
        and (maximum is None or value <= maximum)
    )


def _bounded_text(
    value: object,
    *,
    maximum: int,
    trim: bool = False,
    no_whitespace: bool = False,
) -> bool:
    if not isinstance(value, str) or not value or len(value) > maximum:
        return False
    if trim and value.strip() != value:
        return False
    if no_whitespace and any(character.isspace() for character in value):
        return False
    return not any(unicodedata.category(character) == "Cc" for character in value)


def validate_readiness_document(
    value: object,
    *,
    ui_protocol_version: int = UI_PROTOCOL_VERSION,
    expected_server_protocol_version: int | None = None,
) -> dict[str, Any]:
    """Validate one schema-v3 readiness response and return its object."""

    if not isinstance(value, dict):
        raise ReadinessContractError("readiness document must be a JSON object")
    missing = sorted(READINESS_REQUIRED_FIELDS.difference(value))
    if missing:
        raise ReadinessContractError(f"readiness document is missing fields: {missing}")
    if (
        value.get("schema_version") != READINESS_SCHEMA_VERSION
        or value.get("object") != READINESS_OBJECT
    ):
        raise ReadinessContractError("unsupported readiness document identity")

    protocol = value.get("protocol_version")
    minimum_ui = value.get("minimum_ui_protocol_version")
    maximum_ui = value.get("maximum_ui_protocol_version")
    if not all(
        _integer(item, minimum=1, maximum=MAX_PROTOCOL_VERSION)
        for item in (protocol, minimum_ui, maximum_ui, ui_protocol_version)
    ):
        raise ReadinessContractError("readiness protocol versions must be positive u32 values")
    assert isinstance(protocol, int)
    assert isinstance(minimum_ui, int)
    assert isinstance(maximum_ui, int)
    if minimum_ui > maximum_ui or not minimum_ui <= protocol <= maximum_ui:
        raise ReadinessContractError("readiness protocol compatibility range is invalid")
    if not minimum_ui <= ui_protocol_version <= maximum_ui:
        raise ReadinessContractError(
            f"UI protocol {ui_protocol_version} is outside the server's supported range"
        )
    if (
        expected_server_protocol_version is not None
        and protocol != expected_server_protocol_version
    ):
        raise ReadinessContractError(
            f"readiness server protocol is {protocol}; expected {expected_server_protocol_version}"
        )

    if not _bounded_text(
        value.get("server_version"), maximum=64, trim=True, no_whitespace=True
    ):
        raise ReadinessContractError("readiness server version is missing or invalid")
    status = value.get("status")
    if status not in {"ready", "not_ready"}:
        raise ReadinessContractError("readiness status is invalid")
    progress = value.get("progress")
    if not _integer(progress, minimum=0, maximum=100):
        raise ReadinessContractError("readiness progress is invalid")
    model = value.get("model")
    if not _bounded_text(model, maximum=256, trim=True):
        raise ReadinessContractError("readiness model identity is missing or invalid")
    if not isinstance(value.get("loading"), bool):
        raise ReadinessContractError("readiness loading flag is invalid")
    load_error = value.get("load_error")
    if load_error is not None and not _bounded_text(load_error, maximum=512):
        raise ReadinessContractError("readiness load error is invalid")

    modalities = value.get("input_modalities")
    if (
        not isinstance(modalities, list)
        or len(modalities) > 16
        or len(set(item for item in modalities if isinstance(item, str)))
        != len(modalities)
        or any(not _bounded_text(item, maximum=64) for item in modalities)
    ):
        raise ReadinessContractError("readiness input modalities are invalid")
    tasks = value.get("model_tasks")
    if (
        not isinstance(tasks, list)
        or len(tasks) > 3
        or len(set(item for item in tasks if isinstance(item, str))) != len(tasks)
        or any(
            not isinstance(item, str)
            or item not in {"generation", "embedding", "rerank"}
            for item in tasks
        )
        or ("rerank" in tasks and "embedding" not in tasks)
    ):
        raise ReadinessContractError("readiness model tasks are invalid")
    context_window = value.get("context_window")
    if context_window is not None and not _integer(
        context_window, minimum=1, maximum=MAX_U64
    ):
        raise ReadinessContractError("readiness context window is invalid")
    if not _integer(value.get("in_flight_requests"), minimum=0, maximum=MAX_U64):
        raise ReadinessContractError("readiness in-flight request count is invalid")
    if not _integer(value.get("available_permits"), minimum=0, maximum=MAX_U64):
        raise ReadinessContractError("readiness available permit count is invalid")
    if not isinstance(value.get("memory_pressure_high"), bool):
        raise ReadinessContractError("readiness memory-pressure flag is invalid")
    ram_utilization = value.get("ram_utilization")
    if (
        isinstance(ram_utilization, bool)
        or not isinstance(ram_utilization, (int, float))
        or not math.isfinite(ram_utilization)
        or not 0 <= ram_utilization <= 1
    ):
        raise ReadinessContractError("readiness RAM utilization is invalid")

    if status == "ready" and (
        model == "not loaded"
        or progress != 100
        or value["loading"]
        or load_error is not None
        or not tasks
        or context_window is None
        or value["available_permits"] == 0
        or value["memory_pressure_high"]
    ):
        raise ReadinessContractError("readiness ready state is internally inconsistent")
    return value


def validate_readiness_schema_document(value: object) -> dict[str, Any]:
    """Validate the compatibility-critical structure of the Draft-07 schema."""

    if not isinstance(value, dict):
        raise ReadinessContractError("readiness schema must be a JSON object")
    if value.get("$schema") != "http://json-schema.org/draft-07/schema#":
        raise ReadinessContractError("readiness schema must declare Draft-07")
    if value.get("type") != "object" or value.get("additionalProperties") is not True:
        raise ReadinessContractError("readiness schema object policy is invalid")
    required = value.get("required")
    if (
        not isinstance(required, list)
        or len(required) != len(READINESS_REQUIRED_FIELDS)
        or any(not isinstance(field, str) for field in required)
        or set(required) != READINESS_REQUIRED_FIELDS
    ):
        raise ReadinessContractError("readiness schema required fields are invalid")
    properties = value.get("properties")
    if not isinstance(properties, dict):
        raise ReadinessContractError("readiness schema properties are missing")
    if properties.get("schema_version") != {"const": READINESS_SCHEMA_VERSION}:
        raise ReadinessContractError("readiness schema version identity is invalid")
    if properties.get("object") != {"const": READINESS_OBJECT}:
        raise ReadinessContractError("readiness schema object identity is invalid")
    for field in (
        "protocol_version",
        "minimum_ui_protocol_version",
        "maximum_ui_protocol_version",
    ):
        schema = properties.get(field)
        if (
            not isinstance(schema, dict)
            or schema.get("type") != "integer"
            or schema.get("minimum") != 1
            or schema.get("maximum") != MAX_PROTOCOL_VERSION
        ):
            raise ReadinessContractError(
                f"readiness schema {field} definition is invalid"
            )
    for field in ("in_flight_requests", "available_permits"):
        schema = properties.get(field)
        if (
            not isinstance(schema, dict)
            or schema.get("type") != "integer"
            or schema.get("minimum") != 0
            or schema.get("maximum") != MAX_U64
        ):
            raise ReadinessContractError(
                f"readiness schema {field} definition is invalid"
            )
    context_window = properties.get("context_window")
    if (
        not isinstance(context_window, dict)
        or context_window.get("type") != ["integer", "null"]
        or context_window.get("minimum") != 1
        or context_window.get("maximum") != MAX_U64
    ):
        raise ReadinessContractError(
            "readiness schema context_window definition is invalid"
        )
    return value

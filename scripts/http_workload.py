#!/usr/bin/env python3
"""Measure one local Bloom deployment cell through streaming HTTP requests.

The report is evidence for a specific binary, model and host. It is not a
performance budget or a production support claim. Only the Python standard
library is required.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import ctypes
import hashlib
import http.client
import json
import math
import os
import pathlib
import platform
import re
import secrets
import signal
import subprocess
import tempfile
import threading
import time
from typing import Any


STARTUP_PORT = re.compile(r"server running on http://127\.0\.0\.1:(\d+)")
MAX_EVENT_BYTES = 1024 * 1024
MAX_STREAM_BYTES = 16 * 1024 * 1024
RUNTIME_STATS_MAX_BYTES = 256 * 1024


class RuntimeStatsError(ValueError):
    """A bounded, credential-free error from an optional stats endpoint."""

    def __init__(self, endpoint: str, reason: str) -> None:
        super().__init__(f"{endpoint}: {reason}")
        self.endpoint = endpoint
        self.reason = reason


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_identity(path: pathlib.Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError("model path must not be a symlink")
    if path.is_file():
        return {"kind": "file", "sha256": sha256_file(path), "files": 1}
    if not path.is_dir():
        raise ValueError("model path must be a regular file or directory")
    files = sorted(path.rglob("*"))
    if len(files) > 256 or any(item.is_symlink() for item in files):
        raise ValueError("model tree has too many entries or contains a symlink")
    regular = [item for item in files if item.is_file()]
    if not regular or len(regular) != len(files):
        raise ValueError("model directory must contain only regular files")
    digest = hashlib.sha256()
    for item in regular:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        file_hash = bytes.fromhex(sha256_file(item))
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(file_hash)
    return {"kind": "tree-v1", "sha256": digest.hexdigest(), "files": len(regular)}


def percentile(samples: list[float], fraction: float) -> float | None:
    if not samples:
        return None
    ordered = sorted(samples)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    return round(
        ordered[lower] + (ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower])
        * (position - lower),
        3,
    )


def distribution(samples: list[float]) -> dict[str, Any]:
    return {
        "count": len(samples),
        "p50": percentile(samples, 0.50),
        "p95": percentile(samples, 0.95),
        "p99": percentile(samples, 0.99),
        "max": round(max(samples), 3) if samples else None,
    }


def read_rss_bytes(pid: int) -> int:
    if os.name == "nt":
        return _read_windows_rss_bytes(pid)
    status_path = pathlib.Path(f"/proc/{pid}/status")
    if status_path.is_file():
        match = re.search(r"^VmRSS:\s*(\d+)\s+kB$", status_path.read_text(), re.MULTILINE)
        if not match:
            raise ValueError("/proc did not report VmRSS")
        return int(match.group(1)) * 1024
    output = subprocess.check_output(
        ["ps", "-o", "rss=", "-p", str(pid)], text=True, timeout=2
    ).strip()
    return int(output) * 1024


def _read_windows_rss_bytes(pid: int) -> int:
    """Read a process working set without relying on localized shell output.

    ``ps`` is a PowerShell alias on Windows and does not accept the POSIX
    ``-o rss`` flags used by the Unix fallback.  Calling the PSAPI directly
    keeps workload evidence usable on Windows runners and installations with
    a non-English system locale.
    """
    from ctypes import wintypes

    process_query_limited_information = 0x1000
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCounters),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
    if not handle:
        error = ctypes.get_last_error()
        raise OSError(error, f"OpenProcess({pid}) failed")
    try:
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(counters), counters.cb
        ):
            error = ctypes.get_last_error()
            raise OSError(error, f"GetProcessMemoryInfo({pid}) failed")
        return int(counters.WorkingSetSize)
    finally:
        kernel32.CloseHandle(handle)


def rss_source() -> str:
    """Describe the platform API used by :func:`read_rss_bytes`."""
    if os.name == "nt":
        return "Windows PSAPI WorkingSetSize"
    if pathlib.Path("/proc").is_dir():
        return "/proc/<pid>/status VmRSS"
    return "ps RSS"


def host_hardware() -> dict[str, Any]:
    cpu_model = platform.processor() or None
    total_memory_bytes = None
    if pathlib.Path("/proc/cpuinfo").is_file():
        match = re.search(
            r"^(?:model name|Hardware)\s*:\s*(.+)$",
            pathlib.Path("/proc/cpuinfo").read_text(), re.MULTILINE,
        )
        if match:
            cpu_model = match.group(1).strip()
    if pathlib.Path("/proc/meminfo").is_file():
        match = re.search(
            r"^MemTotal:\s*(\d+)\s+kB$",
            pathlib.Path("/proc/meminfo").read_text(), re.MULTILINE,
        )
        if match:
            total_memory_bytes = int(match.group(1)) * 1024
    if platform.system() == "Darwin":
        try:
            cpu_model = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True, timeout=2
            ).strip() or cpu_model
            total_memory_bytes = int(subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], text=True, timeout=2
            ).strip())
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    elif os.name == "nt":
        # GlobalMemoryStatusEx is locale-independent and available on all
        # supported Windows versions.  The ctypes call is kept here rather
        # than shelling out to PowerShell so reports work in restricted CI
        # environments as well.
        try:
            from ctypes import wintypes

            class MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", wintypes.DWORD),
                    ("dwMemoryLoad", wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatusEx()
            status.dwLength = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                total_memory_bytes = int(status.ullTotalPhys)
        except (AttributeError, OSError, TypeError):
            pass
    return {"cpu_model": cpu_model, "host_memory_total_bytes": total_memory_bytes}


class MemorySampler:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.samples: list[int] = []
        self.stop = threading.Event()
        self.worker = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self.stop.is_set():
            try:
                self.samples.append(read_rss_bytes(self.pid))
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            self.stop.wait(0.1)

    def __enter__(self) -> MemorySampler:
        self.worker.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.stop.set()
        self.worker.join(timeout=3)


def get_ready(port: int, timeout: float = 2) -> tuple[int, dict[str, Any]]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        connection.request("GET", "/ready")
        response = connection.getresponse()
        data = response.read(64 * 1024 + 1)
        if len(data) > 64 * 1024:
            raise ValueError("readiness response is too large")
        decoded = json.loads(data)
        if not isinstance(decoded, dict):
            raise ValueError("readiness response is not a JSON object")
        return response.status, decoded
    finally:
        connection.close()


def _require_stats_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeStatsError("runtime stats", f"{field} is not a non-negative integer")
    return value


def _require_stats_number(value: Any, field: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeStatsError("runtime stats", f"{field} is not numeric")
    if not math.isfinite(value) or value < 0:
        raise RuntimeStatsError("runtime stats", f"{field} is not a finite non-negative number")
    return value


def _require_stats_fraction(value: Any, field: str) -> int | float:
    number = _require_stats_number(value, field)
    if number > 1:
        raise RuntimeStatsError("runtime stats", f"{field} is greater than one")
    return number


def _stats_int_group(payload: dict[str, Any], name: str, fields: tuple[str, ...]) -> dict[str, int]:
    value = payload.get(name)
    if not isinstance(value, dict):
        raise RuntimeStatsError("runtime stats", f"{name} is not an object")
    return {
        field: _require_stats_int(value.get(field), f"{name}.{field}")
        for field in fields
    }


def _sanitize_cachemesh(value: Any) -> dict[str, Any] | None:
    """Keep cache counters and gauges, excluding unbounded/string metadata."""
    if value is None:
        return None
    if not isinstance(value, dict) or not isinstance(value.get("enabled"), bool):
        raise RuntimeStatsError("/v1/observability", "cachemesh is malformed")
    tiers: dict[str, dict[str, int | float]] = {}
    tier_fields = (
        "hits", "misses", "evictions", "offloads", "restores",
        "failed_offloads", "dropped", "bytes", "items", "hit_rate",
    )
    for tier in ("l1", "l2", "l3"):
        tiers[tier] = _stats_int_group(value, tier, tier_fields[:-1])
        tiers[tier]["hit_rate"] = _require_stats_fraction(
            value[tier].get("hit_rate"), f"{tier}.hit_rate"
        )
    return {"enabled": value["enabled"], **tiers}


def _sanitize_observability(payload: Any) -> dict[str, Any]:
    """Validate and retain only bounded numeric runtime evidence.

    Model names, paths, device labels, load errors, prompts and responses are
    intentionally omitted. This keeps an optional benchmark report safe to
    share while retaining the counters and gauges useful for interpretation.
    """
    if not isinstance(payload, dict):
        raise RuntimeStatsError("/v1/observability", "response is not a JSON object")
    if payload.get("schema_version") != 1 or payload.get("object") != "bloom.observability_snapshot":
        raise RuntimeStatsError("/v1/observability", "unsupported snapshot identity")
    server = payload.get("server")
    if not isinstance(server, dict) or not isinstance(server.get("version"), str):
        raise RuntimeStatsError("/v1/observability", "server metadata is malformed")
    if not isinstance(payload.get("ready"), bool):
        raise RuntimeStatsError("/v1/observability", "ready is malformed")
    requests = _stats_int_group(payload, "requests", ("total", "completed", "failed", "in_flight"))
    tokens = _stats_int_group(payload, "tokens", ("prompt_total", "generated_total"))
    scheduler = payload.get("scheduler")
    if not isinstance(scheduler, dict) or not isinstance(scheduler.get("ifb_enabled"), bool):
        raise RuntimeStatsError("/v1/observability", "scheduler is malformed")
    scheduler_values = _stats_int_group(
        payload, "scheduler", ("prefill_queue", "decoding_queue", "active_requests")
    )
    kv_cache = _stats_int_group(
        payload, "kv_cache",
        ("total_blocks", "free_blocks", "active_blocks", "cached_blocks",
         "hits", "misses", "evictions", "reuses"),
    )
    kv_cache["utilization"] = _require_stats_fraction(
        payload["kv_cache"].get("utilization"), "kv_cache.utilization"
    )
    memory = payload.get("memory")
    if not isinstance(memory, dict):
        raise RuntimeStatsError("/v1/observability", "memory is malformed")
    memory_values = {
        field: _require_stats_int(memory.get(field), f"memory.{field}")
        for field in ("total_vram", "used_vram", "total_ram", "used_ram", "peak_vram", "peak_ram")
    }
    load = payload.get("load")
    if (
        not isinstance(load, dict)
        or load.get("phase") not in {"idle", "loading", "ready", "failed"}
        or not isinstance(load.get("failure_present"), bool)
    ):
        raise RuntimeStatsError("/v1/observability", "load is malformed")
    progress = _require_stats_int(load.get("progress"), "load.progress")
    if progress > 100:
        raise RuntimeStatsError("/v1/observability", "load.progress is greater than 100")
    return {
        "schema_version": 1,
        "object": "bloom.observability_snapshot",
        "server": {
            "version": server["version"],
            "uptime_seconds": _require_stats_int(server.get("uptime_seconds"), "server.uptime_seconds"),
        },
        "ready": payload["ready"],
        "load": {
            "phase": load["phase"],
            "progress": progress,
            "failure_present": load["failure_present"],
        },
        "requests": requests,
        "tokens": tokens,
        "scheduler": {"ifb_enabled": scheduler["ifb_enabled"], **scheduler_values},
        "kv_cache": kv_cache,
        "cachemesh": _sanitize_cachemesh(payload.get("cachemesh")),
        "memory": memory_values,
    }


def _sanitize_kv_cache(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise RuntimeStatsError("/v1/kv-cache-stats", "response is not a JSON object")
    return {
        "total_blocks": _require_stats_int(payload.get("total_blocks"), "total_blocks"),
        "free_blocks": _require_stats_int(payload.get("free_blocks"), "free_blocks"),
        "active_blocks": _require_stats_int(payload.get("active_blocks"), "active_blocks"),
        "cached_blocks": _require_stats_int(payload.get("cached_blocks"), "cached_blocks"),
        "hits": _require_stats_int(payload.get("hits"), "hits"),
        "misses": _require_stats_int(payload.get("misses"), "misses"),
        "evictions": _require_stats_int(payload.get("evictions"), "evictions"),
        "reuses": _require_stats_int(payload.get("reuses"), "reuses"),
        "utilization": _require_stats_fraction(payload.get("utilization"), "utilization"),
        "cachemesh": _sanitize_cachemesh(payload.get("cachemesh")),
    }


def _fetch_json_endpoint(port: int, path: str, api_key: str, timeout: float) -> Any:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        connection.request(
            "GET", path,
            headers={"Accept": "application/json", "Authorization": f"Bearer {api_key}"},
        )
        response = connection.getresponse()
        body = response.read(RUNTIME_STATS_MAX_BYTES + 1)
        if response.status != 200:
            raise RuntimeStatsError(path, f"HTTP status {response.status}")
        if len(body) > RUNTIME_STATS_MAX_BYTES:
            raise RuntimeStatsError(path, "response exceeded byte limit")
        try:
            return json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise RuntimeStatsError(path, "response was not valid JSON") from error
    except (OSError, http.client.HTTPException) as error:
        raise RuntimeStatsError(path, "request failed") from error
    finally:
        connection.close()


def fetch_runtime_stats(port: int, api_key: str, timeout: float) -> dict[str, Any]:
    """Fetch both authenticated snapshots, retaining partial availability."""
    snapshots: dict[str, Any] = {}
    errors: dict[str, dict[str, str]] = {}
    for name, path, sanitizer in (
        ("observability", "/v1/observability", _sanitize_observability),
        ("kv_cache", "/v1/kv-cache-stats", _sanitize_kv_cache),
    ):
        try:
            snapshots[name] = sanitizer(_fetch_json_endpoint(port, path, api_key, timeout))
        except RuntimeStatsError as error:
            errors[name] = {"status": "unavailable", "error": error.reason}
    return {
        "status": "available" if not errors else "unavailable",
        "snapshots": snapshots,
        "errors": errors,
    }


def _flatten_numeric(value: Any, prefix: str = "") -> dict[str, int | float]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, int | float] = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(item, bool):
            continue
        if isinstance(item, (int, float)):
            result[path] = item
        elif isinstance(item, dict):
            result.update(_flatten_numeric(item, path))
    return result


COUNTER_SUFFIXES = (
    ".requests.total", ".requests.completed", ".requests.failed",
    ".tokens.prompt_total", ".tokens.generated_total",
    ".kv_cache.hits", ".kv_cache.misses", ".kv_cache.evictions", ".kv_cache.reuses",
    ".hits", ".misses", ".evictions", ".offloads", ".restores",
    ".failed_offloads", ".dropped",
)


def compare_runtime_stats(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Compare snapshots without turning gauges into misleading deltas."""
    deltas: dict[str, dict[str, int | float | None]] = {}
    gauges: dict[str, dict[str, int | float]] = {}
    reset = False
    for endpoint in ("observability", "kv_cache"):
        old_values = _flatten_numeric(before.get(endpoint, {}), endpoint)
        new_values = _flatten_numeric(after.get(endpoint, {}), endpoint)
        endpoint_deltas: dict[str, int | float | None] = {}
        endpoint_gauges: dict[str, int | float] = {}
        for path, value in new_values.items():
            relative = path.removeprefix(f"{endpoint}.")
            if any(relative.endswith(suffix.lstrip(".")) for suffix in COUNTER_SUFFIXES):
                old = old_values.get(path)
                if old is None or value < old:
                    endpoint_deltas[relative] = None
                    reset = True
                else:
                    endpoint_deltas[relative] = value - old
            else:
                endpoint_gauges[relative] = value
        deltas[endpoint] = endpoint_deltas
        gauges[endpoint] = endpoint_gauges
    old_uptime = before.get("observability", {}).get("server", {}).get("uptime_seconds")
    new_uptime = after.get("observability", {}).get("server", {}).get("uptime_seconds")
    if isinstance(old_uptime, int) and isinstance(new_uptime, int) and new_uptime < old_uptime:
        reset = True
    return {
        "status": "reset" if reset else "available",
        "counter_deltas": deltas,
        "gauges": gauges,
    }


class RuntimeStatsCollector:
    """Collect optional endpoint snapshots outside request timing windows."""

    def __init__(self, enabled: bool, port: int, api_key: str, timeout: float) -> None:
        self.enabled = enabled
        self.port = port
        self.api_key = api_key
        self.timeout = timeout
        self.baseline: dict[str, Any] | None = None
        self.after_waves: list[dict[str, Any]] = []
        self.reset_seen = False
        self.unavailable_seen = False

    def capture_baseline(self) -> None:
        if self.enabled:
            self.baseline = fetch_runtime_stats(self.port, self.api_key, self.timeout)
            self.unavailable_seen |= self.baseline["status"] == "unavailable"

    def capture_after_wave(self, wave: int) -> None:
        if not self.enabled:
            return
        current = fetch_runtime_stats(self.port, self.api_key, self.timeout)
        entry: dict[str, Any] = {"wave": wave, **current}
        if current["status"] == "unavailable":
            self.unavailable_seen = True
        elif self.baseline and self.baseline["status"] == "available":
            comparison = compare_runtime_stats(self.baseline["snapshots"], current["snapshots"])
            entry.update(comparison)
            self.reset_seen |= comparison["status"] == "reset"
        elif self.baseline and self.baseline["status"] == "unavailable":
            self.unavailable_seen = True
            entry["status"] = "unavailable"
            entry["errors"] = {
                "baseline": {"status": "unavailable", "error": "baseline snapshot unavailable"}
            }
        self.after_waves.append(entry)

    def report(self) -> dict[str, Any]:
        if not self.enabled:
            return {"status": "disabled", "baseline": None, "after_waves": []}
        status = "unavailable" if self.unavailable_seen else "reset" if self.reset_seen else "available"
        return {
            "status": status,
            "baseline": self.baseline,
            "after_waves": self.after_waves,
            "limitations": [
                "Snapshots are fetched outside timed request waves and are process-local observations.",
                "Counter deltas are null when a server reset is detected; gauges are reported as observed values.",
            ],
        }


def wait_ready(
    process: subprocess.Popen[bytes], log_path: pathlib.Path, timeout: float
) -> tuple[int, dict[str, Any]]:
    deadline = time.monotonic() + timeout
    port = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited during startup: {process.returncode}")
        if port is None:
            match = STARTUP_PORT.search(
                log_path.read_text(encoding="utf-8", errors="replace")[-64 * 1024 :]
            )
            if match:
                port = int(match.group(1))
        if port is not None:
            try:
                status, ready = get_ready(port)
                if ready.get("load_error") and not ready.get("loading"):
                    raise RuntimeError("server reported a terminal model load failure")
                if status == 200 and ready.get("status") == "ready":
                    if (
                        ready.get("object") != "bloom.readiness"
                        or "generation" not in ready.get("model_tasks", [])
                        or not isinstance(ready.get("model"), str)
                    ):
                        raise RuntimeError("readiness does not describe a generation model")
                    return port, ready
            except (OSError, http.client.HTTPException, json.JSONDecodeError):
                pass
        time.sleep(0.1)
    raise TimeoutError(f"server did not become ready within {timeout}s")


def stream_request(
    port: int, model: str, api_key: str, max_tokens: int, timeout: float,
    *, disconnect: bool = False,
) -> dict[str, Any]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with a short greeting."}],
            "max_completion_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    )
    started = time.monotonic()
    try:
        connection.request(
            "POST", "/v1/chat/completions", body,
            {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        )
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError(f"chat HTTP status {response.status}: {response.read(2048)!r}")
        if "text/event-stream" not in response.getheader("Content-Type", ""):
            raise RuntimeError("chat response is not an SSE stream")
        if disconnect:
            return {"accepted": True}
        first_content = None
        last_content = None
        gaps: list[float] = []
        deltas = 0
        completion_tokens = None
        finished = False
        total = 0
        while True:
            # Read decoded body bytes: HTTP chunks need not align with SSE
            # lines, and a persistent connection can outlive its response.
            line = response.readline(MAX_EVENT_BYTES + 1)
            total += len(line)
            if len(line) > MAX_EVENT_BYTES or total > MAX_STREAM_BYTES:
                raise RuntimeError("SSE stream exceeded byte limit")
            if not line:
                raise RuntimeError("SSE stream ended before [DONE]")
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            event = json.loads(payload)
            if not isinstance(event, dict):
                raise RuntimeError("SSE event is not a JSON object")
            if event.get("model") != model:
                raise RuntimeError("SSE event reported a different model")
            if event.get("usage") is not None:
                if not isinstance(event["usage"], dict):
                    raise RuntimeError("SSE usage is not an object")
                completion_tokens = event["usage"].get("completion_tokens")
            choices = event.get("choices", [])
            if not isinstance(choices, list):
                raise RuntimeError("SSE choices is not a list")
            if choices:
                choice = choices[0]
                if not isinstance(choice, dict) or not isinstance(choice.get("delta"), dict):
                    raise RuntimeError("SSE choice or delta is invalid")
                if choice.get("finish_reason") is not None:
                    finished = True
                content = choice.get("delta", {}).get("content")
                if isinstance(content, str) and content:
                    now = time.monotonic()
                    if first_content is None:
                        first_content = now
                    if last_content is not None:
                        gaps.append((now - last_content) * 1000)
                    last_content = now
                    deltas += 1
        if first_content is None or not finished:
            raise RuntimeError("SSE stream omitted content or finish event")
        if (
            not isinstance(completion_tokens, int)
            or isinstance(completion_tokens, bool)
            or completion_tokens < 1
        ):
            raise RuntimeError("SSE stream omitted completion-token usage")
        return {
            "latency_ms": (time.monotonic() - started) * 1000,
            "first_content_ms": (first_content - started) * 1000,
            "inter_delta_ms": gaps,
            "content_deltas": deltas,
            "completion_tokens": completion_tokens,
        }
    finally:
        connection.close()


def wait_recovered(process: subprocess.Popen[bytes], port: int, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("server exited before disconnect recovery")
        try:
            status, ready = get_ready(port)
            if (
                status == 200
                and ready.get("status") == "ready"
                and ready.get("in_flight_requests") == 0
            ):
                return ready
        except (OSError, http.client.HTTPException, json.JSONDecodeError):
            pass
        time.sleep(0.1)
    raise TimeoutError("server did not recover to zero in-flight requests")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.server_bin.is_symlink() or args.model.is_symlink():
        raise ValueError("server and model paths must not be symlinks")
    binary = args.server_bin.resolve(strict=True)
    model = args.model.resolve(strict=True)
    if not binary.is_file():
        raise ValueError("server binary must be a regular file")
    server_sha256 = sha256_file(binary)
    if args.expected_server_sha256 and server_sha256 != args.expected_server_sha256:
        raise ValueError("server SHA-256 does not match the requested deployment cell")
    identity = model_identity(model)
    if args.expected_model_sha256 and identity["sha256"] != args.expected_model_sha256:
        raise ValueError("model SHA-256 does not match the requested deployment cell")
    levels = [int(value) for value in args.concurrency.split(",")]
    if not levels or any(value < 1 or value > 32 for value in levels):
        raise ValueError("concurrency levels must be between 1 and 32")
    if sum(levels) > args.max_requests:
        raise ValueError("max-requests must cover one wave at every concurrency level")
    api_key = secrets.token_urlsafe(24)
    environment = os.environ.copy()
    environment["BLOOM_API_KEY"] = api_key
    environment["RUST_LOG"] = "bloom_server=info"
    with tempfile.TemporaryDirectory(prefix="bloom-workload-") as temporary:
        directory = pathlib.Path(temporary)
        log_path = directory / "server.log"
        with log_path.open("wb") as log:
            started = time.monotonic()
            creation_flags = (
                getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                if os.name == "nt"
                else 0
            )
            process = subprocess.Popen(
                [str(binary), "--model", str(model), "--models-dir", str(directory / "models"),
                 "--host", "127.0.0.1", "--port", "0", "--backend", args.backend,
                 "--device", args.device, "--max-concurrent", str(max(levels)),
                 "--timeout", str(max(1, int(args.request_timeout) + 1))],
                stdout=log, stderr=subprocess.STDOUT, env=environment,
                creationflags=creation_flags,
            )
        try:
            port, ready = wait_ready(process, log_path, args.startup_timeout)
            cold_start_ms = (time.monotonic() - started) * 1000
            active_model = ready["model"]
            with MemorySampler(process.pid) as sampler:
                for _ in range(args.warmup):
                    stream_request(port, active_model, api_key, args.max_tokens, args.request_timeout)
                baseline_rss = read_rss_bytes(process.pid)
                runtime_stats = RuntimeStatsCollector(
                    args.collect_runtime_stats, port, api_key, args.request_timeout
                )
                # Endpoint probes are deliberately outside the timed workload
                # window. They are authenticated but do not carry prompts or
                # model responses.
                runtime_stats.capture_baseline()
                waves: list[dict[str, Any]] = []
                workload_started = time.monotonic()
                deadline = time.monotonic() + args.duration_seconds
                stats_elapsed = 0.0
                request_count = 0
                while request_count + sum(levels) <= args.max_requests:
                    for level in levels:
                        wave_started = time.monotonic()
                        with concurrent.futures.ThreadPoolExecutor(max_workers=level) as pool:
                            futures = [
                                pool.submit(stream_request, port, active_model, api_key,
                                            args.max_tokens, args.request_timeout)
                                for _ in range(level)
                            ]
                            results = [future.result() for future in futures]
                        waves.append({"concurrency": level, "elapsed_ms":
                                      (time.monotonic() - wave_started) * 1000,
                                      "results": results})
                        request_count += level
                        # Keep observability probes out of wave elapsed time so
                        # their latency cannot improve or penalize throughput.
                        stats_started = time.monotonic()
                        runtime_stats.capture_after_wave(len(waves))
                        stats_elapsed += time.monotonic() - stats_started
                    if time.monotonic() - stats_elapsed >= deadline:
                        break
                workload_elapsed_ms = (time.monotonic() - workload_started - stats_elapsed) * 1000
                if time.monotonic() - stats_elapsed < deadline:
                    raise RuntimeError("request cap reached before target duration")
                accepted = 0
                for _ in range(args.disconnects):
                    accepted += int(stream_request(
                        port, active_model, api_key, args.max_tokens,
                        args.request_timeout, disconnect=True)["accepted"])
                recovery_started = time.monotonic()
                wait_recovered(process, port, args.request_timeout)
                recovery_ms = (time.monotonic() - recovery_started) * 1000
                stream_request(port, active_model, api_key, args.max_tokens, args.request_timeout)
                after_rss = read_rss_bytes(process.pid)
            if not sampler.samples:
                raise RuntimeError("host RSS sampler did not record any values")
            if os.name == "nt":
                # ``Popen.terminate`` calls TerminateProcess on Windows and
                # returns a non-zero status even after a healthy shutdown.
                # A new process group lets the server observe Ctrl-Break and
                # run its bounded graceful-drain path just like SIGTERM on
                # POSIX.  Fall back to hard termination if a console is not
                # available (for example, a detached CI job).
                try:
                    process.send_signal(signal.CTRL_BREAK_EVENT)
                except (OSError, ValueError):
                    process.terminate()
            else:
                process.terminate()
            process.wait(timeout=args.shutdown_timeout)
            if process.returncode != 0:
                raise RuntimeError(f"server shutdown status {process.returncode}")
            if sha256_file(binary) != server_sha256 or model_identity(model) != identity:
                raise RuntimeError("server binary or model changed during the workload")
            all_results = [item for wave in waves for item in wave["results"]]
            return {
                "schema_version": 1,
                "object": "bloom.http_workload",
                "result": "pass",
                "identity": {
                    "model": identity,
                    "server_sha256": server_sha256,
                    "server_version": ready["server_version"],
                    "backend": args.backend,
                    "device": args.device,
                    "selection_source": "server CLI arguments",
                    "os": platform.platform(),
                    "arch": platform.machine(),
                    "python": platform.python_version(),
                    **host_hardware(),
                },
                "settings": {
                    "duration_seconds": args.duration_seconds,
                    "max_requests": args.max_requests,
                    "concurrency": levels,
                    "max_completion_tokens": args.max_tokens,
                    "warmup": args.warmup,
                    "collect_runtime_stats": args.collect_runtime_stats,
                },
                "cold_start_ms": round(cold_start_ms, 3),
                "workload_elapsed_ms": round(workload_elapsed_ms, 3),
                "requests_completed": len(all_results),
                "completion_tokens": sum(item["completion_tokens"] for item in all_results),
                "latency_ms": distribution([item["latency_ms"] for item in all_results]),
                "first_content_ms": distribution([item["first_content_ms"] for item in all_results]),
                "inter_content_delta_ms": distribution([
                    gap for item in all_results for gap in item["inter_delta_ms"]
                ]),
                "waves": [
                    {
                        "concurrency": wave["concurrency"],
                        "requests": len(wave["results"]),
                        "elapsed_ms": round(wave["elapsed_ms"], 3),
                        "completions_per_second": round(
                            len(wave["results"]) * 1000 / wave["elapsed_ms"], 3
                        ),
                        "first_content_ms": distribution([
                            item["first_content_ms"] for item in wave["results"]
                        ]),
                    }
                    for wave in waves
                ],
                "disconnect": {"accepted": accepted, "recovery_ms": round(recovery_ms, 3),
                               "post_recovery_request": "pass"},
                "runtime_stats": runtime_stats.report(),
                "host_memory": {
                    "source": rss_source()
                    + ", sampled every 100 ms plus warm/recovery endpoints",
                    "warm_baseline_bytes": baseline_rss,
                    "observed_peak_bytes": max(*sampler.samples, baseline_rss, after_rss),
                    "after_recovery_bytes": after_rss,
                    "drift_bytes": after_rss - baseline_rss,
                },
                "device_peak_memory_bytes": None,
                "limitations": [
                    "SSE content deltas can contain multiple tokens; inter-delta timing is not token-by-token TBT.",
                    "RSS sampling can miss shorter memory peaks and does not measure accelerator memory.",
                    "A disconnect after HTTP acceptance does not prove an active model step was interrupted.",
                ],
            }
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-bin", type=pathlib.Path, required=True)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--expected-model-sha256")
    parser.add_argument("--expected-server-sha256")
    parser.add_argument("--backend", default="candle")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--concurrency", default="1,2,4")
    parser.add_argument("--duration-seconds", type=float, default=10)
    parser.add_argument("--max-requests", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--disconnects", type=int, default=2)
    parser.add_argument(
        "--collect-runtime-stats", action="store_true",
        help="collect authenticated /v1/observability and /v1/kv-cache-stats snapshots",
    )
    parser.add_argument("--startup-timeout", type=float, default=120)
    parser.add_argument("--request-timeout", type=float, default=60)
    parser.add_argument("--shutdown-timeout", type=float, default=10)
    args = parser.parse_args()
    if (
        args.duration_seconds < 0 or args.max_requests < 1 or args.max_tokens < 1
        or args.warmup < 0 or args.disconnects < 0
        or args.startup_timeout <= 0 or args.request_timeout <= 0
        or args.shutdown_timeout <= 0
    ):
        parser.error("counts and timeouts must be positive (duration/warmup/disconnects may be zero)")
    try:
        report = run(args)
    except (OSError, ValueError, RuntimeError, TimeoutError, subprocess.SubprocessError) as error:
        report = {"schema_version": 1, "object": "bloom.http_workload",
                  "result": "fail", "error": str(error)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"HTTP workload: {report['result']} ({args.output})")
    return 0 if report["result"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

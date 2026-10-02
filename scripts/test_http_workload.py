#!/usr/bin/env python3
"""Regression checks for HTTP workload evidence and SSE validation."""

from __future__ import annotations

import http.server
import json
import os
import pathlib
import sys
import tempfile
import threading
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import http_workload as workload  # noqa: E402


class FakeChat(http.server.BaseHTTPRequestHandler):
    event_model = "fixture"
    include_done = True
    include_usage = True
    chunk_size = None
    content_length = False

    def do_POST(self) -> None:
        size = int(self.headers["Content-Length"])
        request = json.loads(self.rfile.read(size))
        assert request["stream"] is True
        events = []
        for event in (
            {"model": self.event_model, "choices": [{"delta": {"content": "A"}}]},
            {"model": self.event_model, "choices": [{"delta": {"content": "B"}}]},
            {"model": self.event_model, "choices": [{"delta": {}, "finish_reason": "stop"}]},
        ):
            events.append(b"data: " + json.dumps(event).encode() + b"\n\n")
        if self.include_usage:
            events.append(
                b'data: {"model":"fixture","choices":[],"usage":{"completion_tokens":2}}\n\n'
            )
        if self.include_done:
            events.append(b"data: [DONE]\n\n")
        body = b"".join(events)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        if self.chunk_size is not None:
            self.send_header("Transfer-Encoding", "chunked")
            chunks = [body[index:index + self.chunk_size]
                      for index in range(0, len(body), self.chunk_size)]
            body = b"".join(
                f"{len(chunk):X}\r\n".encode() + chunk + b"\r\n" for chunk in chunks
            ) + b"0\r\n\r\n"
        elif self.content_length:
            self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args: object) -> None:
        pass


class ChunkedChat(FakeChat):
    protocol_version = "HTTP/1.1"
    chunk_size = 7


class ContentLengthChat(FakeChat):
    protocol_version = "HTTP/1.1"
    content_length = True


def observability_snapshot(*, requests_total: int, generated_total: int,
                           uptime_seconds: int = 10) -> dict:
    return {
        "schema_version": 1,
        "object": "bloom.observability_snapshot",
        "created": 1,
        "server": {"version": "test", "uptime_seconds": uptime_seconds},
        "model": "/secret/model.gguf",
        "ready": True,
        "load": {"phase": "ready", "progress": 100,
                  "requested_model": "/secret/model.gguf", "failure_present": False},
        "speculative_mode": "none",
        "requests": {"total": requests_total, "completed": requests_total,
                      "failed": 0, "in_flight": 0},
        "tokens": {"prompt_total": requests_total * 2, "generated_total": generated_total},
        "scheduler": {"ifb_enabled": False, "prefill_queue": 0,
                       "decoding_queue": 0, "active_requests": 0},
        "startup_memory_estimate": None,
        "kv_cache": {"total_blocks": 8, "free_blocks": 8, "active_blocks": 0,
                      "cached_blocks": 0, "hits": requests_total, "misses": 0,
                      "evictions": 0, "reuses": 0, "utilization": 0.0},
        "cachemesh": None,
        "memory": {"total_vram": 0, "used_vram": 0, "total_ram": 100,
                    "used_ram": 50, "peak_vram": 0, "peak_ram": 50,
                    "device_name": "secret-device"},
    }


def kv_cache_snapshot(*, hits: int) -> dict:
    return {"total_blocks": 8, "free_blocks": 8, "active_blocks": 0,
            "cached_blocks": 0, "hits": hits, "misses": 1, "evictions": 0,
            "reuses": 0, "utilization": 0.0, "cachemesh": None}


class RuntimeStats(http.server.BaseHTTPRequestHandler):
    observability = observability_snapshot(requests_total=1, generated_total=2)
    kv_cache = kv_cache_snapshot(hits=1)
    expected_authorization = "Bearer test-key"

    def do_GET(self) -> None:
        if self.headers.get("Authorization") != self.expected_authorization:
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":"authorization rejected"}')
            return
        if self.path == "/v1/observability":
            payload = self.observability
        elif self.path == "/v1/kv-cache-stats":
            payload = self.kv_cache
        else:
            self.send_response(404)
            self.end_headers()
            return
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args: object) -> None:
        pass


class WorkloadTests(unittest.TestCase):
    def test_model_tree_digest_is_ordered_and_rejects_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = pathlib.Path(root)
            (path / "b").write_bytes(b"b")
            (path / "a").write_bytes(b"a")
            first = workload.model_identity(path)
            self.assertEqual(first["files"], 2)
            self.assertEqual(first, workload.model_identity(path))
            (path / "a").write_bytes(b"changed")
            self.assertNotEqual(first["sha256"], workload.model_identity(path)["sha256"])
            (path / "link").symlink_to(path / "a")
            with self.assertRaisesRegex(ValueError, "symlink"):
                workload.model_identity(path)

    def test_percentile_is_deterministic(self) -> None:
        self.assertEqual(workload.distribution([4, 1, 3, 2]), {
            "count": 4, "p50": 2.5, "p95": 3.85, "p99": 3.97, "max": 4,
        })
        self.assertIsNone(workload.distribution([])["p95"])

    def test_rss_sampler_uses_a_platform_source(self) -> None:
        # Keep this probe process-local so the regression remains independent
        # of a running Bloom server or a model fixture.  Windows uses the
        # locale-independent PSAPI path; POSIX hosts use /proc or ps.
        self.assertGreater(workload.read_rss_bytes(os.getpid()), 0)
        source = workload.rss_source()
        if os.name == "nt":
            self.assertEqual(source, "Windows PSAPI WorkingSetSize")
        else:
            self.assertIn(source, {"/proc/<pid>/status VmRSS", "ps RSS"})

    def request(self, handler: type[FakeChat]) -> dict:
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            return workload.stream_request(
                server.server_port, "fixture", "test", 2, 3
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)

    def test_stream_records_content_and_usage(self) -> None:
        result = self.request(FakeChat)
        self.assertEqual(result["completion_tokens"], 2)
        self.assertEqual(result["content_deltas"], 2)
        self.assertEqual(len(result["inter_delta_ms"]), 1)

    def test_chunked_stream_reassembles_events_before_parsing(self) -> None:
        for chunk_size in (1, 7, 64):
            with self.subTest(chunk_size=chunk_size):
                class FragmentedChat(ChunkedChat):
                    pass

                FragmentedChat.chunk_size = chunk_size
                # A one-byte HTTP chunk has more framing than payload. The
                # stream limit applies to decoded SSE bytes, not wire bytes.
                with mock.patch.object(workload, "MAX_STREAM_BYTES", 1024):
                    result = self.request(FragmentedChat)
                self.assertEqual(result["completion_tokens"], 2)
                self.assertEqual(result["content_deltas"], 2)
                self.assertEqual(len(result["inter_delta_ms"]), 1)

    def test_chunked_stream_preserves_decoded_byte_limits(self) -> None:
        for limit, value in (("MAX_EVENT_BYTES", 32), ("MAX_STREAM_BYTES", 128)):
            with self.subTest(limit=limit):
                with mock.patch.object(workload, limit, value):
                    with self.assertRaisesRegex(RuntimeError, "exceeded byte limit"):
                        self.request(ChunkedChat)

    def test_stream_rejects_wrong_model(self) -> None:
        class WrongModel(FakeChat):
            event_model = "other"

        with self.assertRaisesRegex(RuntimeError, "different model"):
            self.request(WrongModel)

    def test_stream_requires_terminal_and_usage(self) -> None:
        class NoUsage(FakeChat):
            include_usage = False

        for handler in (FakeChat, ChunkedChat, ContentLengthChat):
            with self.subTest(handler=handler.__name__):
                class NoTerminal(handler):
                    include_done = False

                # HTTP/1.1 keeps the socket open after the body ends. Detect
                # its framing boundary instead of waiting for a socket EOF.
                with self.assertRaisesRegex(RuntimeError, "before \\[DONE\\]"):
                    self.request(NoTerminal)
        with self.assertRaisesRegex(RuntimeError, "omitted completion-token usage"):
            self.request(NoUsage)

    def runtime_stats_server(self) -> tuple[http.server.ThreadingHTTPServer, threading.Thread]:
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), RuntimeStats)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, thread

    def test_runtime_stats_collector_authenticates_and_separates_deltas(self) -> None:
        server, thread = self.runtime_stats_server()
        try:
            collector = workload.RuntimeStatsCollector(
                True, server.server_port, "test-key", 3
            )
            collector.capture_baseline()
            RuntimeStats.observability = observability_snapshot(
                requests_total=3, generated_total=8, uptime_seconds=11
            )
            RuntimeStats.kv_cache = kv_cache_snapshot(hits=3)
            collector.capture_after_wave(1)
            report = collector.report()
            self.assertEqual(report["status"], "available")
            self.assertEqual(report["after_waves"][0]["status"], "available")
            self.assertEqual(
                report["after_waves"][0]["counter_deltas"]["observability"]["requests.total"],
                2,
            )
            self.assertEqual(
                report["after_waves"][0]["counter_deltas"]["kv_cache"]["hits"],
                2,
            )
            # The report only retains bounded numeric stats, not model paths,
            # device labels, API keys or response bodies.
            serialized = json.dumps(report)
            self.assertNotIn("/secret/model.gguf", serialized)
            self.assertNotIn("secret-device", serialized)
            self.assertNotIn("test-key", serialized)
            self.assertNotIn("authorization rejected", serialized)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)

    def test_runtime_stats_marks_counter_reset_and_unavailable(self) -> None:
        server, thread = self.runtime_stats_server()
        try:
            collector = workload.RuntimeStatsCollector(
                True, server.server_port, "test-key", 3
            )
            collector.capture_baseline()
            RuntimeStats.observability = observability_snapshot(
                requests_total=0, generated_total=0, uptime_seconds=1
            )
            RuntimeStats.kv_cache = kv_cache_snapshot(hits=0)
            collector.capture_after_wave(1)
            self.assertEqual(collector.report()["status"], "reset")
            self.assertEqual(collector.after_waves[0]["status"], "reset")
            self.assertIsNone(
                collector.after_waves[0]["counter_deltas"]["observability"]["requests.total"]
            )
            RuntimeStats.observability = {"schema_version": 0}
            collector.capture_after_wave(2)
            self.assertEqual(collector.report()["status"], "reset")
            self.assertEqual(collector.after_waves[1]["status"], "unavailable")
            self.assertEqual(collector.after_waves[1]["errors"]["observability"]["status"],
                             "unavailable")
            RuntimeStats.observability = {"schema_version": 0}
            baseline_missing = workload.RuntimeStatsCollector(
                True, server.server_port, "test-key", 3
            )
            baseline_missing.capture_baseline()
            RuntimeStats.observability = observability_snapshot(requests_total=4, generated_total=9)
            baseline_missing.capture_after_wave(1)
            self.assertEqual(baseline_missing.after_waves[0]["status"], "unavailable")
            self.assertEqual(
                baseline_missing.after_waves[0]["errors"]["baseline"]["status"],
                "unavailable",
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)

    def test_runtime_stats_disabled_makes_no_endpoint_calls(self) -> None:
        with mock.patch.object(workload, "fetch_runtime_stats") as fetch:
            collector = workload.RuntimeStatsCollector(False, 1, "secret", 1)
            collector.capture_baseline()
            collector.capture_after_wave(1)
            self.assertEqual(collector.report(), {
                "status": "disabled", "baseline": None, "after_waves": []
            })
            fetch.assert_not_called()


if __name__ == "__main__":
    unittest.main()

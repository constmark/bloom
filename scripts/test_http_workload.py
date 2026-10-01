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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import http_workload as workload  # noqa: E402


class FakeChat(http.server.BaseHTTPRequestHandler):
    event_model = "fixture"
    include_done = True
    include_usage = True

    def do_POST(self) -> None:
        size = int(self.headers["Content-Length"])
        request = json.loads(self.rfile.read(size))
        assert request["stream"] is True
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for event in (
            {"model": self.event_model, "choices": [{"delta": {"content": "A"}}]},
            {"model": self.event_model, "choices": [{"delta": {"content": "B"}}]},
            {"model": self.event_model, "choices": [{"delta": {}, "finish_reason": "stop"}]},
        ):
            self.wfile.write(b"data: " + json.dumps(event).encode() + b"\n\n")
        if self.include_usage:
            self.wfile.write(
                b'data: {"model":"fixture","choices":[],"usage":{"completion_tokens":2}}\n\n'
            )
        if self.include_done:
            self.wfile.write(b"data: [DONE]\n\n")

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

    def test_stream_rejects_wrong_model(self) -> None:
        class WrongModel(FakeChat):
            event_model = "other"

        with self.assertRaisesRegex(RuntimeError, "different model"):
            self.request(WrongModel)

    def test_stream_requires_terminal_and_usage(self) -> None:
        class NoTerminal(FakeChat):
            include_done = False

        class NoUsage(FakeChat):
            include_usage = False

        with self.assertRaisesRegex(RuntimeError, "before \\[DONE\\]"):
            self.request(NoTerminal)
        with self.assertRaisesRegex(RuntimeError, "omitted completion-token usage"):
            self.request(NoUsage)


if __name__ == "__main__":
    unittest.main()
